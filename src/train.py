import sys
from typing import NoReturn
import torch
import hydra
import os
import datetime
import logging
from timm.loss import LabelSmoothingCrossEntropy
from torch.utils.tensorboard import SummaryWriter

from torchvision.ops import focal_loss
from src.models.engine import train_epoch, valid_epoch
from src.models.classifications_models import get_model
from src.utils.utils import (
    update_lr,
    get_dict_classes
)
from src.enities.train_pipeline import TrainingPipelineParams
from src.data.loaders import get_loaders, count_classes
import torch.nn as nn
import torch.nn.functional as F
from src.models.focal_loss import sigmoid_focal_loss
from torch.autograd import Variable
writer = SummaryWriter()

_log_format = "%(asctime)s\t%(levelname)s\t%(name)s\t" \
              "%(filename)s.%(funcName)s " \
              "line: %(lineno)d | \t%(message)s"
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(_log_format))
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

file_handler = logging.FileHandler(os.path.join(writer.log_dir, "log.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(_log_format))
logger.addHandler(file_handler)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = torch.Tensor([gamma])
#         self.size_average = size_average
#         if isinstance(alpha, (float, int)):
#             if self.alpha > 1:
#                 raise ValueError('Not supported value, alpha should be small than 1.0')
#             else:
#                 self.alpha = torch.Tensor([alpha, 1.0 - alpha])
#         if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
#         self.alpha /= torch.sum(self.alpha)
#
#     def forward(self, input, target):
#         if input.dim() > 2:
#             input = input.view(input.size(0), input.size(1), -1)  # [N,C,H,W]->[N,C,H*W] ([N,C,D,H,W]->[N,C,D*H*W])
#         # target
#         # [N,1,D,H,W] ->[N*D*H*W,1]
#         if self.alpha.device != input.device:
#             self.alpha = torch.tensor(self.alpha, device=input.device)
#         target = target.view(-1, 1)
#         logpt = torch.log(input + 1e-10)
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1, 1)
#         pt = torch.exp(logpt)
#         alpha = self.alpha.gather(0, target.view(-1))
#
#         gamma = self.gamma
#
#         if not self.gamma.device == input.device:
#             gamma = torch.tensor(self.gamma, device=input.device)
#
#         loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#         return loss

# class FocalLoss(nn.Module):
#     def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         if alpha is None:  # alpha is the balance factor
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = Variable(alpha)
#         self.gamma = gamma  # Index
#         self.class_num = class_num  # Number of categories
#         self.size_average = size_average  # Does the returned loss need to mean?
#
#     def forward(self, inputs, targets):
#         N = inputs.size(0)  # inputs is the output of the top layer of the network
#         C = inputs.size(1)
#         P = F.softmax(inputs)  # Seek p_t first
#
#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)  # Get the one_hot encoding of label
#
#         if inputs.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1)]
#         # y*p_t If * is not used here, you can also use gather to extract the probability of the correct category.
#         # The reason why sum can be used is because class_mask has cleared the probability of prediction error to zero.
#         probs = (P * class_mask).sum(1).view(-1, 1)
#         # y*log(p_t)
#         log_p = probs.log()
#         # -a * (1-p_t)^2 * log(p_t)
#         batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
#
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss


@hydra.main(version_base=None, config_path='../configs', config_name='train_config')
def train_pipeline(params: TrainingPipelineParams) -> NoReturn:
    epochs = params.train_params.epochs
    lr = params.train_params.lr
    best_f1_score = 0
    pretrained_epochs = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    params.train_params.path_save_checkpoint = os.path.join(writer.log_dir, 'models')

    os.makedirs(params.train_params.path_save_checkpoint, exist_ok=True)
    encode_classes2index, decode_index2classes = get_dict_classes(params.train_params.path_to_classes)

    model_weights = None
    if os.path.exists(params.train_params.path_to_weights):
        logger.info(f"Loading the model: {params.train_params.path_to_weights}")
        state_dict = torch.load(params.train_params.path_to_weights)
        if "epochs" in state_dict:
            pretrained_epochs = state_dict["epochs"]

        if "best_f1_score" in state_dict:
            best_f1_score = state_dict["best_f1_score"]

        if "model_state" not in state_dict:
            model_weights = state_dict
        else:
            model_weights = state_dict["model_state"]

        if "lr" in state_dict:
            lr = state_dict["lr"]

    logger.info(f"Model: {params.train_params.name_model}")
    logger.info(f"The currently used device: {device}")
    logger.info(f"Num epochs: {epochs}. Pretrained epochs: {pretrained_epochs}")
    logger.info(f"Pretrained F1 Score: {best_f1_score}")
    logger.info(f"Lr: {lr}")
    logger.info(f"Batch size: {params.train_params.batch_size}")
    logger.info(f"Image size: {params.train_params.img_size}")

    model = get_model(params.train_params.name_model,
                      len(encode_classes2index),
                      requires_grad=False,
                      model_weights=model_weights,
                      pretrained=params.train_params.pretrained).to(device)

    if model_weights is not None:
        logger.info("The model has been loaded successfully")

    class_counts = count_classes(params.train_params)
    total_samples = sum(class_counts)
    class_weights = []
    for count in class_counts:
        weight = 1 / (count / total_samples)
        class_weights.append(weight)
    class_weights = torch.FloatTensor(class_weights).to(device)

    # criterion = LabelSmoothingCrossEntropy()
    criterion = FocalLoss(class_weights)
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_data, train_loader, valid_data, valid_loader = get_loaders(params.train_params)

    logger.info(f"Training dataset size: {len(train_data)}")
    logger.info(f"Validating dataset size: {len(valid_data)}")
    logger.info(f"--------==== Start of training ====--------")

    for epoch in range(epochs):
        logger.info(f"Pretrained epoch: {pretrained_epochs}. Epoch in the current session: {epoch} ")

        train_epoch_loss, train_epoch_acc, train_epoch_f1_score = train_epoch(
            model, train_loader, optimizer, criterion, train_data, decode_index2classes, device
        )

        valid_epoch_loss, valid_epoch_acc, valid_epoch_f1_score = valid_epoch(
            model, valid_loader, criterion, valid_data, decode_index2classes, device
        )

        logger.info(f"Train Loss: {train_epoch_loss:.4f} ")
        logger.info(f"Train Accuracy: {train_epoch_acc:.4f}")
        logger.info(f"Train F1 Score: {train_epoch_f1_score:.4f}")

        writer.add_scalar("Train/Loss", train_epoch_loss, pretrained_epochs + epoch)
        writer.add_scalar("Train/F1 Score", train_epoch_f1_score, pretrained_epochs + epoch)

        logger.info(f'Val Loss: {valid_epoch_loss:.4f}')
        logger.info(f'Val Accuracy: {valid_epoch_acc:.4f}')
        logger.info(f'Val F1 Score MEAD {valid_epoch_f1_score:.4f}')

        writer.add_scalar("Validate/Loss", valid_epoch_loss, pretrained_epochs + epoch)
        writer.add_scalar("Validate/F1 Score", valid_epoch_f1_score, pretrained_epochs + epoch)

        if valid_epoch_f1_score > max(best_f1_score, 0.5):
            best_f1_score = valid_epoch_f1_score

            logger.info(f"New best score: {best_f1_score:.4f}")
            logger.info(f"Save checkpoint to {params.train_params.path_save_checkpoint}")

            with open(os.path.join(params.train_params.path_save_checkpoint,
                                   f'results_{params.train_params.name_model}.txt'), 'w') as f:
                f.write(f"Train Loss: {train_epoch_loss:.4f} "
                        f"Train Accuracy: {train_epoch_acc:.4f} "
                        f"Train F1 Score: {train_epoch_f1_score:.4f}\n")
                f.write(f'Val Loss: {valid_epoch_loss:.4f} '
                        f'Val Accuracy: {valid_epoch_acc:.4f} '
                        f'Val F1 Score: {valid_epoch_f1_score:.4f}')

            torch.save({"best_f1_score": best_f1_score,
                        "epochs": pretrained_epochs + epoch,
                        "model_state": model.state_dict(),
                        "lr": lr
                        },
                       os.path.join(params.train_params.path_save_checkpoint,
                                    f"checkpoint_{params.train_params.name_model}.pth"))


if __name__ == "__main__":
    train_pipeline()

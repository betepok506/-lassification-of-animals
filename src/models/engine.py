from tqdm import tqdm
import torch
from src.utils.metric import Metric


def train_epoch(model, train_loader, optimizer, criterion, train_data, classes, device):
    model.train()
    train_running_loss, train_running_acc, train_running_f1_score = .0, .0, .0
    cnt = 0
    for i, data in tqdm(enumerate(train_loader), total=int(len(train_data) / train_loader.batch_size),
                        ncols=120,
                        desc="Training"):
        cnt += 1
        images, labels = data['image'].to(device), data['label'].to(device)

        outputs = model(images)
        metric = Metric(outputs.detach().cpu(), labels.detach().cpu())
        loss = criterion(outputs, labels)

        train_running_loss += loss.item()
        train_running_acc += metric.accuracy()
        train_running_f1_score += metric.f1_score()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / cnt
    train_acc = train_running_acc / cnt
    train_f1_score = train_running_f1_score / cnt
    return train_loss, train_acc, train_f1_score


def valid_epoch(model, valid_loader, criterion, valid_data, classes, device):
    model.eval()
    cnt = 0
    val_running_loss, val_running_acc, val_running_f1_score = .0, .0, .0
    with torch.no_grad():
        for i, data in tqdm(enumerate(valid_loader), total=int(len(valid_data) / valid_loader.batch_size),
                            ncols=120,
                            desc="Validation"):
            cnt += 1
            images, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(images)

            metric = Metric(outputs.detach().cpu(), labels.detach().cpu())
            val_running_acc += metric.accuracy()
            val_running_f1_score += metric.f1_score()
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

        val_loss = val_running_loss / cnt
        val_acc = val_running_acc / cnt
        val_f1_score = val_running_f1_score / cnt
        return val_loss, val_acc, val_f1_score

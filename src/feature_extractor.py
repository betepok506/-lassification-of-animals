import torch
from torch import nn
import re
import os
import sys
import logging
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.data.dataset import ImageDataset
from src.models.classifications_models import get_model
from src.utils.utils import validate_transform
from src.utils.utils import (
    update_lr,
    get_dict_classes
)

_log_format = "%(asctime)s\t%(levelname)s\t%(name)s\t" \
              "%(filename)s.%(funcName)s " \
              "line: %(lineno)d | \t%(message)s"
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(_log_format))
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


class Params:
    path_to_classes = "D:\\projects_andrey\\cv_made_dataset\\vk-made-sports-image-classification\\encode_classes.json"
    path_to_weights = "./runs/Apr08_21-14-20_LAI-02/models/checkpoint_swin_base_patch4_window7_224.pth"
    name_model = "swin_base_patch4_window7_224"
    path_to_save_result = "./results/features_extractions"
    pretrained = True
    path_to_dataset = "D:\\projects_andrey\\cv_made_dataset\\vk-made-sports-image-classification"
    img_size = (224, 224)
    batch_size = 512


params = Params()


def feature_extractor():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    encode_classes2index, decode_index2classes = get_dict_classes(params.path_to_classes)
    model_weights = None
    if os.path.exists(params.path_to_weights):
        logger.info(f"Loading the model: {params.path_to_weights}")
        state_dict = torch.load(params.path_to_weights)
        if "model_state" not in state_dict:
            model_weights = state_dict

    model = get_model(params.name_model,
                      len(encode_classes2index),
                      requires_grad=False,
                      model_weights=model_weights,
                      pretrained=params.pretrained).to(device)

    if model_weights is not None:
        logger.info("The model has been loaded successfully")

    if re.compile("swin*").match(params.name_model) is not None:
        model.head = nn.Identity()
    elif re.compile("resnet*").match(params.name_model) is not None:
        model.fc = nn.Identity()

    images_id = []
    features_images = []

    transform = validate_transform(params.img_size)
    train_csv = pd.read_csv(os.path.join(params.path_to_dataset, 'train.csv'))
    train_df = pd.DataFrame(
        data={
            "image_id": [],
            'label': []
        }
    )
    for cls in encode_classes2index.keys():
        ind_cls = encode_classes2index[cls]

        new_df = train_csv[train_csv['label'] == cls]
        new_df = new_df.drop(['label'], axis=1)
        new_df['label'] = np.ones(len(new_df), dtype=np.int8) * ind_cls
        train_df = pd.concat([new_df, train_df])

    train_data = ImageDataset(
        params.path_to_dataset, train_df, transform=transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=params.batch_size,
        shuffle=False
    )
    for i, data in tqdm(enumerate(train_loader), total=int(len(train_data) / train_loader.batch_size),
                        ncols=120,
                        desc="Feature extracting"):
        images, labels = data['image'].to(device), data['label'].to(device)

        features = list(model(images).detach().cpu().numpy())
        features_images.extend([list(f) for f in features])

    os.makedirs(params.path_to_save_result, exist_ok=True)
    pd.DataFrame({"image_id": train_csv["image_id"],
                  'features_images': features_images,
                  "label": train_csv["label"]}).to_csv(os.path.join(params.path_to_save_result,
                                                                    f'result_{params.name_model}.csv'), index=False,sep=";")
    logger.info("Done!")


if __name__ == "__main__":
    feature_extractor()

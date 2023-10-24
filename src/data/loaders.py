import torchvision.transforms as transforms
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from src.data.dataset import ImageDataset
from src.enities.training_params import TrainingParams
from src.data.augmentations import get_augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_loaders(params: TrainingParams):
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(params.img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
    ])
    # augmentation = 'hard'
    # augmentations = get_augmentations(augmentation)
    # transform_train = [
    #     # transforms.ToPILImage(),
    #     A.Resize(params.img_size[0], params.img_size[1]),
    #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ToTensorV2(),
    # ]
    # transform_train = A.Compose(augmentations + transform_train)

    train_csv = pd.read_csv(os.path.join(params.path_to_dataset, 'train_.csv'))
    train_data = ImageDataset(
        params.path_to_dataset, train_csv, transform=transform_train
    )
    transform_valid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(params.img_size),
        transforms.ToTensor(),
    ])

    # transform_valid = A.Compose(transform_train)

    valid_csv = pd.read_csv(os.path.join(params.path_to_dataset, 'valid_.csv'))
    valid_data = ImageDataset(
        params.path_to_dataset, valid_csv, transform=transform_valid
    )
    train_loader = DataLoader(
        train_data,
        batch_size=params.batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=params.batch_size,
        shuffle=False
    )
    return train_data, train_loader, valid_data, valid_loader


def count_classes(params: TrainingParams):
    train_csv = pd.read_csv(os.path.join(params.path_to_dataset, 'train_.csv'))
    unique_classes = train_csv['label'].unique()
    unique_classes.sort()
    return [len(train_csv['label'][train_csv['label'] == cls]) for cls in unique_classes]

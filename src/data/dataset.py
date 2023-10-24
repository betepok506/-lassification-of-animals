import os.path
import cv2
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, csv, transform=None, mode="train"):
        self.csv = csv
        self.mode = mode
        self.root = root
        self.transform = transform

        self.image_names = list(self.csv[:]['image_id'])
        self.labels = list(self.csv[:]['label'])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.root, self.mode, self.image_names[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        targets = self.labels[index]

        return {
            'image': image,
            'label': torch.tensor(targets, dtype=torch.long)
        }

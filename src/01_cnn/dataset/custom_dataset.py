import os
import random
import cv2
import joblib
import numpy as np
from torch.utils.data import Dataset
import albumentations as album

import transform as custom_album


def get_transforms(cfg):
    def get_object(transform):
        if hasattr(album, transform.name):
            return getattr(album, transform.name)
        elif hasattr(custom_album, transform.name):
            return getattr(custom_album, transform.name)
        else:
            return eval(transform.name)
    if cfg.transforms:
        transforms = [get_object(transform)(**transform.params) for name, transform in cfg.transforms.items()]
        return album.Compose(transforms)
    else:
        return None


class CustomDataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.image_ids = df['image_name'].values
        self.transforms = get_transforms(self.cfg)
        self.is_train = cfg.is_train
        if cfg.is_train:
            self.labels = df['target'].values
            self.year = df.loc[:, 'year'].values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        if self.is_train and self.year[idx] == 2020:
            image = cv2.imread(f'../data/input/jpeg_resized_{self.cfg.img_size.height}/train/{image_id}.jpg')
        elif self.is_train and self.year[idx] == 2019:
            image = cv2.imread(f'../data/input/2019_{self.cfg.img_size.height}/train/{image_id}.jpg')
        elif not self.is_train:
            image = cv2.imread(f'../data/input/jpeg_resized_{self.cfg.img_size.height}/test/{image_id}.jpg')
        image = 255 - (image * (255.0/image.max())).astype(np.uint8)
        if self.transforms:
            image = self.transforms(image=image)['image']
        image = image.transpose(2, 0, 1).astype(np.float32)

        if self.is_train:
            label = self.labels[idx]
            return image, label
        else:
            return image
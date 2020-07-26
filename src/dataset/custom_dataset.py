import os
import random
import cv2
import joblib
import numpy as np
from torch.utils.data import Dataset
import albumentations as album

import transform


def get_transforms(cfg):
    def get_object(transform):
        if hasattr(album, transform.name):
            return getattr(album, transform.name)
        elif hasattr(transform, transform.name):
            return getattr(transforms, transform.name)
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
        self.feats = df.loc[:, df.columns.str.startswith('target_encoding')].values
        self.transforms = get_transforms(self.cfg)
        self.is_train = cfg.is_train
        if cfg.is_train:
            self.labels = df['target'].values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        if self.is_train:
            image = cv2.imread(f'../data/input/jpeg_resized_{self.cfg.img_size.height}/train/{image_id}.png')
        else:
            image = cv2.imread(f'../data/input/jpeg_resized_{self.cfg.img_size.height}/test/{image_id}.png')
        image = 255 - (image * (255.0/image.max())).astype(np.uint8)
        image = cv2.resize(image, dsize=(self.cfg.img_size.height, self.cfg.img_size.width))
        if self.transforms:
            image = self.transforms(image=image)['image']
        image = image.transpose(2, 0, 1).astype(np.float32)

        feats = self.feats[idx, :]

        if self.is_train:
            label = self.labels[idx]
            return image, feats, label
        else:
            return image, feats
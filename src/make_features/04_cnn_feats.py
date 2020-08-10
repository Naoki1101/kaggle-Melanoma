import sys
import cv2
import time
import numpy as np
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict

from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader

sys.path.append('../src')
from utils import DataHandler
from feature_utils import save_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64


class CustomDataset(Dataset):
    def __init__(self, df, is_train):
        self.image_ids = df['image_name'].values
        self.is_train = is_train
        if is_train:
            self.year = df.loc[:, 'year'].values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        if self.is_train and self.year[idx] == 2020:
            image = cv2.imread(f'../data/input/jpeg_resized_384/train/{image_id}.jpg')
        elif self.is_train and self.year[idx] == 2019:
            image = cv2.imread(f'../data/input/2019_384/train/{image_id}.jpg')
        elif not self.is_train:
            image = cv2.imread(f'../data/input/jpeg_resized_384/test/{image_id}.jpg')
        image = 255 - (image * (255.0/image.max())).astype(np.uint8)
        image = image.transpose(2, 0, 1).astype(np.float32)

        return image


def _efficientnet(model_name, pretrained):
    if pretrained:
        model = EfficientNet.from_pretrained(model_name, advprop=True)
    else:
        model = EfficientNet.from_name(model_name)
    
    return model


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.model = _efficientnet(model_name='efficientnet-b3', pretrained=True)
        
        self.linear = nn.Linear(1000, 256)

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)

        return x


def extract_logits(df, is_train):
    dataset = CustomDataset(df, is_train)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=4)

    model  =CustomModel().to(device)

    preds = np.zeros((len(df), 256))

    model.eval()
    with torch.no_grad():
        for i, (images) in enumerate(data_loader):
            images = images.to(device)

            logits = model(images.float())
            preds[i * batch_size: (i + 1) * batch_size, :] = logits.cpu().detach().numpy()

    return preds


def get_features(train, test):
    train_features_df = pd.DataFrame()
    test_features_df = pd.DataFrame()

    train_preds = extract_logits(train, is_train=True)
    test_preds = extract_logits(test, is_train=False)
    whole_preds = np.concatenate([train_preds, test_preds], axis=0)

    tsne = TSNE(n_components=2, random_state=2020)
    whole_tsne_array = tsne.fit_transform(whole_preds)

    for i in range(2):
        train_features_df[f'tsne_{i}'] = whole_tsne_array[:len(train), i]
        test_features_df[f'tsne_{i}'] = whole_tsne_array[len(train):, i]

    return train_features_df, test_features_df 


def main():
    train_df = pd.read_csv('../data/input/train_concated.csv')
    test_df = pd.read_csv('../data/input/test.csv')

    train_features_df, test_features_df = get_features(train_df, test_df)

    save_features(train_features_df, data_type='train')
    save_features(test_features_df, data_type='test')


if __name__ == '__main__':
    main()

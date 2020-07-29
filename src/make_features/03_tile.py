import sys
import cv2
import pandas as pd
from easydict import EasyDict as edict

sys.path.append('../src')
from utils import DataHandler
from factory import get_fold
from feature_utils import save_features

dh = DataHandler()


def load_img(id_, data_type):
    img = cv2.imread(f'../data/input/jpeg_resized_384/{data_type}/{id_}.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def split(img):
    tiles = []
    tiles_mean = []

    for i in range(12):
        for j in range(12):
            tile = img[32 * i: 32 * (i + 1), 32 * j: 32 * (j + 1), 0]
            tiles.append(tile)
            tiles_mean.append(np.mean(tile))

    return tiles


def get_features(train, test):
    features_df = pd.DataFrame()

    features_array = np.zeros((len(df), 6))

    for t, df in zip(['train', 'test'], [train, test]):
        for i, id_ in enumerate(df['image_name']):
            img = load_img(id_, t)
            tiles, tiles_mean = split(img)

            features_array[i, 0] = np.mean(img) / 255.
            features_array[i, 1] = np.sort(mean_list)[0] / 255.
            features_array[i, 2] = np.sort(mean_list)[0] / np.sort(mean_list)[-1]
            features_array[i, 3] = np.sort(mean_list)[3] / np.sort(mean_list)[-3]
            features_array[i, 4] = np.sort(mean_list)[5] / np.sort(mean_list)[-5]
            features_array[i, 5] = np.sort(mean_list)[10] / np.sort(mean_list)[-10]

    for c in range(6):
        features_df[f'image_features_{i}'] = features_array[:, c]


    return features_df


def main():
    train_df = dh.load('../data/input/train_concated.csv')
    test_df = dh.load('../data/input/test.csv')

    train_features_df, test_features_df = get_features(train_df, test_df)

    save_features(train_features_df, data_type='train')
    save_features(test_features_df, data_type='test')


if __name__ == '__main__':
    main()
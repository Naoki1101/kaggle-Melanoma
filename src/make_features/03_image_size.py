import sys
import pandas as pd
from tqdm import tqdm
import pydicom as dicom
from pandarallel import pandarallel

sys.path.append('../src')
from utils import DataHandler
from feature_utils import save_features

dh = DataHandler()
pandarallel.initialize(progress_bar=True)


def extract_train_height(image_name):
    ds = dicom.dcmread(f'../data/input/train/{image_name}.dcm')
    height = ds.pixel_array.shape[0]
    return height


def extract_train_width(image_name):
    ds = dicom.dcmread(f'../data/input/train/{image_name}.dcm')
    width = ds.pixel_array.shape[1]
    return width


def extract_test_height(image_name):
    ds = dicom.dcmread(f'../data/input/test/{image_name}.dcm')
    height = ds.pixel_array.shape[0]
    return height


def extract_test_width(image_name):
    ds = dicom.dcmread(f'../data/input/test/{image_name}.dcm')
    width = ds.pixel_array.shape[1]
    return width


def get_features(train, test):
    train_features_df = pd.DataFrame()
    test_features_df = pd.DataFrame()

    train_features_df['height'] = train['image_name'].parallel_apply(extract_train_height)
    train_features_df['width'] = train['image_name'].parallel_apply(extract_train_width)

    test_features_df['height'] = test['image_name'].parallel_apply(extract_test_height)
    test_features_df['width'] = test['image_name'].parallel_apply(extract_test_width)

    return train_features_df, test_features_df


def main():
    train_df = dh.load('../data/input/train.csv')
    test_df = dh.load('../data/input/test.csv')

    train_features_df, test_features_df = get_features(train_df, test_df)

    save_features(train_features_df, data_type='train')
    save_features(test_features_df, data_type='test')


if __name__ == '__main__':
    main()
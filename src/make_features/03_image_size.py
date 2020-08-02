import sys
import pandas as pd
from tqdm import tqdm
import multiprocessing

sys.path.append('../src')
from utils import DataHandler
from feature_utils import save_features

dh = DataHandler()


def extract_train_height(image_name):
    ds = dicom.dcmread(f'../data/input/train/{image_name}.dcm')
    height = ds.pixel_array.shape[0]
    return height


def extract_train_height(image_name):
    ds = dicom.dcmread(f'../data/input/train/{image_name}.dcm')
    width = ds.pixel_array.shape[1]
    return width


def extract_test_height(image_name):
    ds = dicom.dcmread(f'../data/input/test/{image_name}.dcm')
    height = ds.pixel_array.shape[0]
    return height


def extract_test_height(image_name):
    ds = dicom.dcmread(f'../data/input/test/{image_name}.dcm')
    width = ds.pixel_array.shape[1]
    return width


def get_features(train, test):
    train_features_df = pd.DataFrame()
    test_features_df = pd.DataFrame()

    train_hsize, train_wsize = extract_image_size(train, 'train')
    train_features_df['height'] = train_hsize
    train_features_df['width'] = train_wsize

    test_hsize, test_wsize = extract_image_size(test, 'test')
    test_features_df['height'] = test_hsize
    test_features_df['width'] = test_wsize

    return train_features_df, test_features_df


def main():
    train_df = dh.load('../data/input/train.csv')
    test_df = dh.load('../data/input/test.csv')

    train_features_df, test_features_df = get_features(train_df, test_df)

    save_features(train_features_df, data_type='train')
    save_features(test_features_df, data_type='test')


if __name__ == '__main__':
    main()
import sys
import pandas as pd
from easydict import EasyDict as edict

sys.path.append('../src')
from utils import DataHandler
from factory import get_fold
from feature_utils import save_features, TargetEncoding, CountEncoding

dh = DataHandler()


def get_features(df):
    features_df = pd.DataFrame()

    features_df['one_hot_sex'] = df['sex'].map({'male': 0, 'female': 1})

    age_df = pd.get_dummies(df['age_approx'])
    for col in age_df.columns:
        features_df[f'one_hot_age_{col}'] = age_df[col]

    site_df = pd.get_dummies(df['anatom_site_general_challenge'])
    for col in site_df.columns:
        features_df[f'one_hot_site_{col.replace("/", "-")}'] = site_df[col]

    return features_df


def main():
    train_df = dh.load('../data/input/train.csv')
    test_df = dh.load('../data/input/test.csv')

    whole_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)

    whole_features_df = get_features(whole_df)

    train_features_df = whole_features_df.iloc[:len(train_df)]
    test_features_df = whole_features_df.iloc[len(train_df):]

    save_features(train_features_df, data_type='train')
    save_features(test_features_df, data_type='test')


if __name__ == '__main__':
    main()
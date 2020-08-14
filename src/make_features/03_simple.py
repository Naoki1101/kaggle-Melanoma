import sys
import pandas as pd
from easydict import EasyDict as edict

sys.path.append('../src')
from utils import DataHandler
from feature_utils import save_features

dh = DataHandler()


def get_features(df):
    features_df = pd.DataFrame()

    features_df['age_approx'] = df['age_approx']

    le = {k: i for i, k in enumerate(df['anatom_site_general_challenge'])}
    features_df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].map(le)

    return features_df


def main():
    train_df = dh.load('../data/input/train_concated.csv')
    test_df = dh.load('../data/input/test.csv')

    whole_df = pd.concat([train_df, test_df], axis=0, sort=False, ignore_index=True)

    whole_features_df = get_features(whole_df)

    train_features_df = whole_features_df.iloc[:len(train_df)]
    test_features_df = whole_features_df.iloc[len(train_df):]

    save_features(train_features_df, data_type='train')
    save_features(test_features_df, data_type='test')


if __name__ == '__main__':
    main()
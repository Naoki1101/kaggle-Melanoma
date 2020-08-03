import sys
import pandas as pd
from easydict import EasyDict as edict

sys.path.append('../src')
from utils import DataHandler
from factory import get_fold
from feature_utils import save_features, TargetEncoding, CountEncoding

dh = DataHandler()


def get_features(train, test):
    train_features_df = pd.DataFrame()
    test_features_df = pd.DataFrame()

    train['age_site'] = train['age_approx'].astype(str) + '_' + train['anatom_site_general_challenge']
    test['age_site'] = test['age_approx'].astype(str) + '_' + test['anatom_site_general_challenge']

    train['size'] = train['height'].astype(str) + '-' + train['width'].astype(str)
    train_le = {k: k for k, v in dict(train['size'].value_counts()).items() if v >= 50 }
    train['size'] = train['size'].map(train_le).fillna('Other')

    test['size'] = test['age_approx'].astype(str) + '-' + test['width'].astype(str)
    test_le = {k: k for k, v in dict(test['size'].value_counts()).items() if v >= 50 }
    test['size'] = test['size'].map(test_le).fillna('Other')

    # TargetEncoding
    cfg = edict({
        'name': 'StratifiedKFold',
        'params': {
            'n_splits': 5,
            'shuffle': True,
            'random_state': 0,
        },
        'split': {
            'y': 'target',
            'groups': None
        },
        'weight': [1.0]
        })
    fold_df = get_fold(cfg, train, train[['target']])

    te = TargetEncoding(fold_df)
    train_features_df['target_encoding_sex'] = te.fit_transform(train['sex'], train['target'])
    test_features_df['target_encoding_sex'] = te.transform(test['sex'])

    te = TargetEncoding(fold_df)
    train_features_df['target_encoding_age'] = te.fit_transform(train['age_approx'], train['target'])
    test_features_df['target_encoding_age'] = te.transform(test['age_approx'])

    te = TargetEncoding(fold_df)
    train_features_df['target_encoding_age_approx'] = te.fit_transform(train['age_approx'], train['target'])
    test_features_df['target_encoding_age_approx'] = te.transform(test['age_approx'])

    te = TargetEncoding(fold_df)
    train_features_df['target_encoding_site'] = te.fit_transform(train['anatom_site_general_challenge'], train['target'])
    test_features_df['target_encoding_site'] = te.transform(test['anatom_site_general_challenge'])

    te = TargetEncoding(fold_df)
    train_features_df['target_encoding_age_site'] = te.fit_transform(train['age_site'], train['target'])
    test_features_df['target_encoding_age_site'] = te.transform(test['age_site'])

    te = TargetEncoding(fold_df)
    train_features_df['target_encoding_size'] = te.fit_transform(train['size'], train['target'])
    test_features_df['target_encoding_size'] = te.transform(test['size'])

    return train_features_df, test_features_df


def main():
    train_df = dh.load('../data/input/train_concated.csv')
    test_df = dh.load('../data/input/test.csv')

    train2020_size_df = pd.read_csv('../data/input/train_image_size.csv')
    train2019_size_df = pd.read_csv('../data/input/train_2019.csv', usecols=['image_name', 'height', 'width'])

    train_size_df = pd.concat([
        train2020_size_df,
        train2019_size_df
    ], axis=0, sort=False, ignore_index=True)

    test_size_df = pd.read_csv('../data/input/test_image_size.csv')

    train_df = train_df.merge(train_size_df, on='image_name', how='left')
    test_df = test_df.merge(test_size_df, on='image_name', how='left')

    train_features_df, test_features_df = get_features(train_df, test_df)

    save_features(train_features_df, data_type='train')
    save_features(test_features_df, data_type='test')


if __name__ == '__main__':
    main()
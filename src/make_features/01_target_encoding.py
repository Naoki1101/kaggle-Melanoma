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

    # TargetEncoding
    cfg = edict({
        'name': 'SingleFold',
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
    train_features_df['target_encoding_age_approx'] = te.fit_transform(train['age_approx'], train['target'])
    test_features_df['target_encoding_age_approx'] = te.transform(test['age_approx'])

    te = TargetEncoding(fold_df)
    train_features_df['target_encoding_site'] = te.fit_transform(train['anatom_site_general_challenge'], train['target'])
    test_features_df['target_encoding_site'] = te.transform(test['anatom_site_general_challenge'])

    te = TargetEncoding(fold_df)
    train_features_df['target_encoding_age_site'] = te.fit_transform(train['age_site'], train['target'])
    test_features_df['target_encoding_age_site'] = te.transform(test['age_site'])

    return train_features_df, test_features_df


def main():
    train_df = dh.load('../data/input/train.csv')
    test_df = dh.load('../data/input/test.csv')

    train_features_df, test_features_df = get_features(train_df, test_df)

    save_features(train_features_df, data_type='train')
    save_features(test_features_df, data_type='test')


if __name__ == '__main__':
    main()
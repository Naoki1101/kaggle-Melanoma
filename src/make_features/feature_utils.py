import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from abc import ABCMeta, abstractmethod

sys.path.append('../src')
from utils import DataHandler


def save_features(df, data_type='train'):
    save_path = Path('../configs/feature/all.yml')
    dh = DataHandler()

    if not save_path.exists():
        save_path.touch()
        feature_dict = {'features': []}
    else:
        feature_dict = dh.load(save_path)

    new_feature = sorted(set(feature_dict['features'] + df.columns.tolist()))
    feature_dict['features'] = new_feature
    dh.save(save_path, feature_dict)

    for col in df.columns:
        df[[col]].reset_index(drop=True).to_feather(f'../features/{col}_{data_type}.feather')


class _BaseEncoding(metaclass=ABCMeta):
    def __init__(self):
        self.encoder = {}

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def fit_transform(self):
        pass



class CountEncoding(_BaseEncoding):
    def __init__(self):
        super(CountEncoding, self).__init__()
        self.levels = None

    def fit(self, values: pd.Series):
        self.levels = values.dropna().unique()
        self.encoder = values.value_counts().to_dict()

    def add(self, values: pd.Series):
        self.levels = np.unique(np.concatenate([self.levels, values.dropna().unique()]))
        add_encoder = values.value_counts().to_dict()
        for level in self.levels:
            if level in self.encoder and level in add_encoder:
                add_encoder[level] += self.encoder[level]
        self.encoder.update(add_encoder)

    def transform(self, values: pd.Series):
        return values.map(self.encoder)

    def fit_transform(self, values: pd.Series):
        self.levels = values.unique()
        self.encoder = values.value_counts().to_dict()
        return values.map(self.encoder)

    def get_levels(self):
        return self.levels

    def get_encoder(self):
        return self.encoder


class TargetEncoding(_BaseEncoding):
    def __init__(self, folds: pd.DataFrame):
        super(TargetEncoding, self).__init__()
        self.folds = folds
        self.all_fold = folds.columns
        self.feature_name = None
        self.target_name = None

    def fit(self, values: pd.Series, target: pd.Series):
        self.feature_name = values.name
        self.target_name = target.name
        df = pd.concat([values, target], axis=1)
        for col in self.all_fold:
            df_fold = self._get_df(df, col)
            self.encoder[col] = self._get_encoder(df_fold)

    def transform(self, values: pd.Series):
        values_encoded = np.zeros(len(values))
        for fold_, encoder in self.encoder.items():
            values_encoded += values.map(encoder) / len(self.all_fold)
        return values_encoded

    def fit_transform(self, values: pd.Series, target: pd.Series):
        self.feature_name = values.name
        self.target_name = target.name
        values_encoded = np.zeros(len(values))
        df = pd.concat([values, target], axis=1)
        for col in self.all_fold:
            val_idx = self._get_index(col)
            train_df = self._get_df(df, col)
            self.encoder[col] = self._get_encoder(train_df)
            values_encoded[val_idx] += values[val_idx].map(self.encoder[col])
        return values_encoded

    def _get_index(self, col):
        return self.folds[self.folds[col] > 0].index

    def _get_df(self, df, col):
        return df[self.folds[col] == 0]

    def _get_encoder(self, df):
        return df.groupby(self.feature_name)[self.target_name].mean().to_dict()


class OneHotEncoding:
    def __init__(self):
        self.cat_features = None

    def fit(self, cat_features):
        self.cat_features = cat_features

    def transform(self, df):
        return pd.get_dummies(data=df, columns=self.cat_features)

    def fit_transform(self, df, cat_features):
        self.cat_features = cat_features
        return pd.get_dummies(data=df, columns=self.cat_features)


class Aggregation:
    def __init__(self, by, columns, aggs={'min', 'max', 'mean', 'std'}):
        self.aggs = aggs
        self.by = by
        self.columns = columns
        self.agg_df = None
        self.output_df = pd.DataFrame()

    def fit(self, df):
        self.agg_df = df.groupby(by=self.by)[self.columns].agg(self.aggs).add_prefix(f'{self.columns}_')

    def transform(self, df):
        for col in self.agg_df.columns:
            encoder = dict(self.agg_df[col])
            self.output_df[col] = df[self.by].map(encoder)
        return self.output_df

    def fit_transform(self, df):
        self.agg_df = df.groupby(by=self.by)[self.columns].agg(self.aggs).add_prefix(f'{self.columns}_')
        for col in self.agg_df.columns:
            encoder = dict(self.agg_df[col])
            self.output_df[col] = df[self.by].map(encoder)
        return self.output_df

    def get_columns(self):
        return [f'{self.columns}_{agg}' for agg in self.aggs]
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append('../src')
import models
import factory
from utils import DataHandler


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cat_features = self.cfg.data.features.cat_features
        self.oof = None
        self.raw_preds = None
        self.weights = []
        self.models = []
        self.scores = []
        self.feature_importance_df = pd.DataFrame(columns=['feature', 'importance'])
        self.dh = DataHandler()

    def train(self, train_df: pd.DataFrame, target_df: pd.DataFrame, fold_df: pd.DataFrame):
        self.oof = np.zeros(len(train_df))

        for fold_, col in enumerate(fold_df.columns):
            print(f'\n========================== FOLD {fold_} ... ==========================\n')
            logging.debug(f'\n========================== FOLD {fold_} ... ==========================\n')

            self._train_fold(train_df, target_df, fold_df[col])

        print('\n\n===================================\n')
        print(f'CV: {np.mean(self.scores):.4f}')
        print('\n===================================\n\n')
        logging.debug('\n\n===================================\n')
        logging.debug(f'CV: {np.mean(self.scores):.4f}')
        logging.debug('\n===================================\n\n')

        return np.mean(self.scores)

    def _train_fold(self, train_df, target_df, fold):
        tr_x, va_x = train_df[fold == 0], train_df[fold > 0]
        tr_y, va_y = target_df[fold == 0], target_df[fold > 0]
        weight = fold.max()
        self.weights.append(weight)

        model = factory.get_model(self.cfg.model)
        model.fit(tr_x, tr_y, va_x, va_y, self.cat_features)
        va_pred = model.predict(va_x, self.cat_features)

        if self.cfg.data.target.reconvert_type:
            va_y = getattr(np, self.cfg.data.target.reconvert_type)(va_y)
            va_pred = getattr(np, self.cfg.data.target.reconvert_type)(va_pred)
            va_pred = np.where(va_pred >= 0, va_pred, 0)

        self.models.append(model)
        self.oof[va_x.index] = va_pred.copy()

        score = factory.get_metrics(self.cfg.common.metrics.name)(va_y, va_pred)
        self.scores.append(score)

        if self.cfg.model.name in ['lightgbm', 'catboost', 'xgboost']:
            importance_fold_df = pd.DataFrame()
            fold_importance = model.extract_importances()
            importance_fold_df['feature'] = train_df.columns
            importance_fold_df['importance'] = fold_importance
            self.feature_importance_df = pd.concat([self.feature_importance_df, importance_fold_df], axis=0)

    def predict(self, test_df):
        preds = np.zeros(len(test_df))
        for fold_, model in enumerate(self.models):
            pred = model.predict(test_df, self.cat_features)
            if self.cfg.data.target.reconvert_type:
                pred = getattr(np, self.cfg.data.target.reconvert_type)(pred)
                pred = np.where(pred >= 0, pred, 0)
            preds += pred.copy() * self.weights[fold_]
        self.raw_preds = preds.copy()
        return preds

    def save(self, run_name):
        log_dir = Path(f'../logs/{run_name}')
        self.dh.save(log_dir / 'oof.npy', self.oof)
        self.dh.save(log_dir / 'raw_preds.npy', self.raw_preds)
        self.dh.save(log_dir / 'importance.csv', self.feature_importance_df)
        self.dh.save(log_dir / 'model_weight.pkl', self.models)
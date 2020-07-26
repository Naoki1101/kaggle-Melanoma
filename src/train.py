import gc
import os
import argparse
import datetime
from datetime import date
from collections import Counter, defaultdict
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG
from sklearn.model_selection import train_test_split
import joblib
import torch

from utils import Timer, seed_everything, DataHandler, Kaggle, make_submission
from utils import send_line, Notion
from trainer import train_model
from predict import predict_test
import factory

import warnings
warnings.filterwarnings('ignore')


# ===============
# Settings
# ===============
parser = argparse.ArgumentParser()
parser.add_argument('--common', default='../configs/common/compe.yml')
parser.add_argument('--notify', default='../configs/common/notify.yml')
parser.add_argument('-m', '--model')
parser.add_argument('-c', '--comment')
options = parser.parse_args()

dh = DataHandler()
cfg = dh.load(options.common)
cfg.update(dh.load(f'../configs/exp/{options.model}.yml'))


# ===============
# Constants
# ===============
comment = options.comment
now = datetime.datetime.now()
model_name = options.model
run_name = f'{model_name}_{now:%Y%m%d%H%M%S}'
notify_params = dh.load(options.notify)

logger_path = Path(f'../logs/{run_name}')


# ===============
# Main
# ===============
def main():
    t = Timer()
    seed_everything(cfg.common.seed)

    logger_path.mkdir(exist_ok=True)
    logging.basicConfig(filename=logger_path / 'train.log', level=logging.DEBUG)

    dh.save(logger_path / 'config.yml', cfg)

    with t.timer('load data'):
        train_x = dh.load('../data/input/train.csv')
        test_x = dh.load('../data/input/test.csv')

    with t.timer('load features'):
        features = dh.load('../configs/feature/all.yml')['features']
        for f in features:
            train_x[f] = dh.load(f'../features/{f}_train.feather')[f].fillna(-1)
            test_x[f] = dh.load(f'../features/{f}_test.feather')[f].fillna(-1)

    with t.timer('drop several rows'):
        if cfg.common.drop is not None:
            drop_idx_list = []
            for drop_name in cfg.common.drop:
                drop_idx = dh.load(f'../pickle/{drop_name}.npy')
                drop_idx_list.append(drop_idx)
            all_drop_idx = np.unique(np.concatenate(drop_idx_list))
            train_x = train_x.drop(all_drop_idx, axis=0).reset_index(drop=True)

    with t.timer('make folds'):
        fold_df = factory.get_fold(cfg.validation, train_x, train_x[[cfg.common.target]])
        if cfg.validation.single:
            fold_df = fold_df[['fold_0']]

    with t.timer('train model'):
        result = train_model(run_name, train_x, fold_df, cfg)
    
    logging.disable(logging.FATAL)
    run_name_cv = f'{run_name}_{result["cv"]:.3f}'
    logger_path.rename(f'../logs/{run_name_cv}')

    with t.timer('predict'):
        preds = predict_test(run_name_cv, test_x, fold_df, cfg)

    with t.timer('make submission'):
        sample_path = f'../data/input/sample_submission.csv'
        output_path = f'../data/output/{run_name_cv}.csv'
        make_submission(y_pred=preds,
                        target_name=cfg.common.target,
                        sample_path=sample_path,
                        output_path=output_path,
                        comp=False)

    with t.timer('kaggle api'):
        kaggle = Kaggle(cfg.compe.compe_name, run_name_cv)
        if cfg.common.kaggle.submit:
            kaggle.submit(comment)

    with t.timer('notify'):
        process_minutes = t.get_processing_time()
        message = f'''{model_name}\ncv: {result["cv"]:.3f}\ntime: {process_minutes:.2f}[h]'''
        send_line(notify_params.line.token, message)

        notion = Notion(token=notify_params.notion.token_v2)
        notion.set_url(url=notify_params.notion.url)
        notion.insert_rows({
            'name': run_name_cv,
            'created': now,
            'model': cfg.model.name,
            'local_cv': round(result['cv'], 4),
            'time': process_minutes,
            'comment': comment
        })



if __name__ == '__main__':
    main()
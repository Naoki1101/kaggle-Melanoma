import gc
import os
import sys
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

sys.path.append('../src')
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
        train_x = dh.load('../data/input/train_concated.csv')
        train_org_x = dh.load('../data/input/train.csv')
        train_2019_x = dh.load('../data/input/train_2019.csv')
        test_x = dh.load('../data/input/test.csv')

    with t.timer('make folds'):
        fold_df = factory.get_fold(cfg.validation, train_org_x, train_org_x[[cfg.common.target]])
        fold_df = pd.concat([fold_df,
                             pd.DataFrame(np.zeros((len(train_2019_x), len(fold_df.columns))), columns=fold_df.columns)]
                             , axis=0, sort=False, ignore_index=True)
        if cfg.validation.single:
            fold_df = fold_df[['fold_0']]
            fold_df /= fold_df['fold_0'].max()

    with t.timer('load features'):
        features = dh.load('../configs/feature/all.yml')['features']
        for f in features:
            train_x[f] = dh.load(f'../features/{f}_train.feather')[f].fillna(-1)
            test_x[f] = dh.load(f'../features/{f}_test.feather')[f].fillna(-1)

    with t.timer('drop several rows'):
        if cfg.common.drop is not None:
            drop_idx = factory.get_drop_idx(cfg.common.drop)
            train_x = train_x.drop(drop_idx, axis=0).reset_index(drop=True)
            fold_df = fold_df.drop(drop_idx, axis=0).reset_index(drop=True)

    with t.timer('train model'):
        result = train_model(run_name, train_x, fold_df, cfg)
    
    logging.disable(logging.FATAL)
    run_name_cv = f'{run_name}_{result["cv"]:.3f}'
    logger_path.rename(f'../logs/{run_name_cv}')

    with t.timer('predict'):
        preds = predict_test(run_name_cv, test_x, fold_df, cfg)

    with t.timer('post process'):
        duplicates = {
            'ISIC_5224960': 1,
            'ISIC_9207777': 1,
            'ISIC_6457527': 1,
            'ISIC_8347588': 0,
            'ISIC_8372206': 1,
            'ISIC_9353360': 1,
            'ISIC_3689290': 0,
            'ISIC_3584949': 0,  
        }
        for image_name, target in duplicates.items():
            idx = test_x[test_x['image_name'] == image_name].index[0]
            preds[idx] = target

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
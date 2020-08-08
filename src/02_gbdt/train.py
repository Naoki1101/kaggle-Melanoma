import sys
import argparse
import datetime
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import factory
from trainer import Trainer
from utils import (DataHandler, Kaggle, Notion, Timer, make_submission,
                   seed_everything, send_line)

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

notify_params = dh.load(options.notify)

features_params = dh.load(f'../configs/feature/{cfg.data.features.name}.yml')
features = features_params.features

comment = options.comment
model_name = options.model
now = datetime.datetime.now()
if cfg.model.task_type != 'optuna':
    run_name = f'{model_name}_{now:%Y%m%d%H%M%S}'
else:
    run_name = f'{model_name}_optuna_{now:%Y%m%d%H%M%S}'

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
    dh.save(logger_path / 'features.yml', features_params)

    with t.timer('load data'):
        train_df = dh.load('../data/input/train.csv')
        train2019_df = dh.load('../data/input/train_concated.csv')
        train_x = factory.get_features(features, cfg.data.loader.train)
        test_x = factory.get_features(features, cfg.data.loader.test)
        train_y = factory.get_target(cfg.data.target)

    with t.timer('add oof'):
        if cfg.data.features.oof.name is not None:
            oof, preds = factory.get_oof(cfg.data)
            train_x['oof'] = oof
            test_x['oof'] = preds
            features.append('oof')

    with t.timer('make folds'):
        fold_df = factory.get_fold(cfg.validation, train_df, train_df[['target']])
        fold_df = pd.concat([fold_df,
                             pd.DataFrame(np.zeros((len(train2019_df), len(fold_df.columns))), columns=fold_df.columns)]
                             , axis=0, sort=False, ignore_index=True)
        if cfg.validation.single:
            fold_df = fold_df[['fold_0']]
            fold_df /= fold_df['fold_0'].max()

    with t.timer('drop index'):
        if cfg.common.drop is not None:
            drop_idx = factory.get_drop_idx(cfg.common.drop)
            train_x = train_x.drop(drop_idx, axis=0).reset_index(drop=True)
            train_y = train_y.drop(drop_idx, axis=0).reset_index(drop=True)
            fold_df = fold_df.drop(drop_idx, axis=0).reset_index(drop=True)

    with t.timer('prepare for ad'):
        if cfg.data.adversarial_validation:
            train_x, train_y = factory.get_ad(cfg, train_x, test_x)

    with t.timer('train and predict'):
        trainer = Trainer(cfg)
        cv = trainer.train(train_df=train_x,
                           target_df=train_y,
                           fold_df=fold_df)
        preds = trainer.predict(test_x)
        trainer.save(run_name)

        run_name_cv = f'{run_name}_{cv:.3f}'
        logger_path.rename(f'../logs/{run_name_cv}')
        logging.disable(logging.FATAL)

    with t.timer('make submission'):
        sample_path = f'../data/input/sample_submission.csv'
        output_path = f'../data/output/{run_name_cv}.csv'
        make_submission(y_pred=preds,
                        target_name=cfg.data.target.name,
                        sample_path=sample_path,
                        output_path=output_path,
                        comp=False)
        if cfg.common.kaggle.submit:
            kaggle = Kaggle(cfg.compe.name, run_name_cv)
            kaggle.submit(comment)

    with t.timer('notify'):
        process_minutes = t.get_processing_time()
        message = f'''{cfg.model.name}\ncv: {cv:.3f}\ntime: {process_minutes}[min]'''
        send_line(notify_params.line.token, message)

        notion = Notion(token=notify_params.notion.token_v2)
        notion.set_url(url=notify_params.notion.url)
        notion.insert_rows({
            'name': run_name_cv,
            'created': now,
            'model': options.model,
            'local_cv': round(cv, 4),
            'time': process_minutes,
            'comment': comment
        })


if __name__ == '__main__':
    main()
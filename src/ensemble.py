import argparse
import datetime
import logging
import warnings
import numpy as np
from pathlib import Path
import optuna

import factory
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

comment = options.comment
model_name = options.model
now = datetime.datetime.now()
run_name = f'{model_name}_{now:%Y%m%d%H%M%S}'

logger_path = Path(f'../logs/{run_name}')


# ===============
# Main
# ===============
def main():
    t = Timer()
    seed_everything(cfg.common.seed)

    logger_path.mkdir(exist_ok=True)

    dh.save(logger_path / 'config.yml', cfg)

    with t.timer('load data'):
        train_df = dh.load('../data/input/train.csv')
        test_df = dh.load('../data/input/test.csv')

        oof = np.zeros((len(train_df), len(cfg.data.models)))
        preds = np.zeros((len(test_df), len(cfg.data.models)))

        for i, m in enumerate(cfg.data.models):
            name = getattr(cfg.data.models, m).name

            log_dir = Path(f'../logs/{name}')
            model_oof = dh.load(log_dir / 'oof.npy')
            model_cfg = dh.load(log_dir / 'config.yml')
            if model_cfg.common.drop:
                drop_idxs = np.array([])
                for drop_name in model_cfg.common.drop:
                    drop_idx = dh.load(f'../pickle/{drop_name}.npy')
                    drop_idxs = np.append(drop_idxs, drop_idx)
                model_oof = factory.extend_data(model_oof, drop_idx)

            model_preds = dh.load(f'../logs/{name}/raw_preds.npy')

            if cfg.data.option.rank_avg:
                model_oof = np.argsort(np.argsort(model_oof)) / (len(model_oof) - 1)
                model_preds = np.argsort(np.argsort(model_preds)) / (len(model_preds) - 1)

            oof[:, i] = model_oof[:len(train_df)]
            preds[:, i] = model_preds

    with t.timer('drop index'):
        if cfg.common.drop is not None:
            drop_idxs = np.array([])
            for drop_name in model_cfg.common.drop:
                drop_idx = dh.load(f'../pickle/{drop_name}.npy')
                drop_idxs = np.append(drop_idxs, drop_idx)
            train_df = train_df.drop(drop_idxs, axis=0).reset_index(drop=True)

    with t.timer('optimize model weight'):
        metric = factory.get_metrics(cfg.common.metrics.name)
        y_true = train_df[cfg.common.target]

        def objective(trial):
            p_list = [0 for i in range(len(cfg.data.models))]
            for i in range(len(cfg.data.models) - 1):
                p_list[i] = trial.suggest_discrete_uniform(f'p{i}', 0.0, 1.0 - sum(p_list), 0.01)
            p_list[-1] = round(1 - sum(p_list[:-1]), 2)

            y_pred = np.zeros(len(train_df))
            for i in range(oof.shape[1]):
                y_pred += oof[:, i] * p_list[i]

            return metric(y_true, y_pred)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, timeout=10)
        best_params = list(study.best_params.values())
        best_weight = best_params + [round(1 - sum(best_params), 2)]

    with t.timer('ensemble'):
        ensemble_oof = np.zeros(len(train_df))
        ensemble_preds = np.zeros(len(test_df))
        for i in range(len(best_weight)):
            ensemble_oof += oof[:, i] * best_weight[i]
            ensemble_preds += preds[:, i] * best_weight[i]

        dh.save(f'../logs/{run_name}/oof.npy', ensemble_oof)
        dh.save(f'../logs/{run_name}/raw_preds.npy', ensemble_preds)

        cv = metric(y_true, ensemble_oof)
        run_name_cv = f'{run_name}_{cv:.3f}'
        logger_path.rename(f'../logs/{run_name_cv}')

        print('\n\n===================================\n')
        print(f'CV: {cv:.4f}')
        print(f'BEST WEIGHT: {best_weight}')
        print('\n===================================\n\n')

    with t.timer('make submission'):
        sample_path = f'../data/input/{cfg.data.sample.name}.csv'
        output_path = f'../data/output/{run_name_cv}.csv'
        make_submission(y_pred=ensemble_preds,
                        target_name=cfg.common.target,
                        sample_path=sample_path,
                        output_path=output_path,
                        comp=False)
        if cfg.common.kaggle.submit:
            kaggle = Kaggle(cfg.compe.compe_name, run_name_cv)
            kaggle.submit(comment)

    with t.timer('notify'):
        process_minutes = t.get_processing_time()
        message = f'''{options.model}\ncv: {cv:.3f}\ntime: {process_minutes}[min]'''
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
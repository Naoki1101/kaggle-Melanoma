import sys
import numpy as np 
import pandas as pd

sys.path.append('../src')
import models
import metrics
import validation


def get_fold(cfg, df, target):
    df_ = df.copy()
    target_columns = target.columns[0]
    df_[target_columns] = target[target_columns].values

    fold_df = pd.DataFrame(index=range(len(df_)))

    if len(cfg.weight) == 1:
        weight_list = [cfg.weight[0] for i in range(cfg.params.n_splits)]
    else:
        weight_list = cfg.weight

    fold = getattr(validation, cfg.name)(cfg)
    for fold_, (trn_idx, val_idx) in enumerate(fold.split(df_)):
        fold_df[f'fold_{fold_}'] = 0
        fold_df.loc[val_idx, f'fold_{fold_}'] = weight_list[fold_]
    
    return fold_df


def get_model(cfg):
    model = getattr(models, cfg.name)(cfg=cfg)
    return model


def get_metrics(cfg):
    evaluator = getattr(metrics, cfg)
    return evaluator


def fill_dropped(dropped_array, drop_idx):
    filled_array = np.zeros(len(dropped_array) + len(drop_idx))
    idx_array = np.arange(len(filled_array))
    use_idx = np.delete(idx_array, drop_idx)
    filled_array[use_idx] = dropped_array
    return filled_array


def get_features(features, cfg):
    dfs = [pd.read_feather(f'../features/{f}_{cfg.data_type}.feather') for f in features if f is not None]
    df = pd.concat(dfs, axis=1)
    if cfg.reduce:
        df = reduce_mem_usage(df)
    return df


def get_target(cfg):
    target = pd.read_feather(f'../features/{cfg.name}.feather')
    if cfg.convert_type is not None:
        target = getattr(np, cfg.convert_type)(target)
    return target


def get_drop_idx(cfg):
    drop_idx_list = []
    for drop_name in cfg:
        drop_idx = np.load(f'../pickle/{drop_name}.npy')
        drop_idx_list.append(drop_idx)
    all_drop_idx = np.unique(np.concatenate(drop_idx_list))
    return all_drop_idx

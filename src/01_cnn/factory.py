import sys
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import metrics
import validation
import loss
import layer
from models import efficientnet, resnet, resnest, senet, ghostnet
from models.custom_model import CustomModel
from dataset.custom_dataset import CustomDataset


def get_model(cfg, is_train=True):
    model = CustomModel(cfg)

    if cfg.model.multi_gpu and is_train:
        model = nn.DataParallel(model)

    return model


def get_loss(cfg):
    loss_ = getattr(loss, cfg.loss.name)(**cfg.loss.params)
    return loss_


def get_dataloader(df, cfg):
    dataset = CustomDataset(df, cfg)
    loader = DataLoader(dataset, **cfg.loader)
    return loader


def get_optim(cfg, parameters):
    optim = getattr(torch.optim, cfg.optimizer.name)(params=parameters, **cfg.optimizer.params)
    return optim


def get_scheduler(cfg, optimizer):
    if cfg.scheduler.name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **cfg.scheduler.params,
        )
    else:
        scheduler = getattr(torch.optim.lr_scheduler, cfg.scheduler.name)(
            optimizer,
            **cfg.scheduler.params,
        )
    return scheduler


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


def get_metrics(cfg):
    evaluator = getattr(metrics, cfg)
    return evaluator


def fill_dropped(dropped_array, drop_idx):
    filled_array = np.zeros(len(dropped_array) + len(drop_idx))
    idx_array = np.arange(len(filled_array))
    use_idx = np.delete(idx_array, drop_idx)
    filled_array[use_idx] = dropped_array
    return filled_array


def get_drop_idx(cfg):
    drop_idx_list = []
    for drop_name in cfg:
        drop_idx = np.load(f'../pickle/{drop_name}.npy')
        drop_idx_list.append(drop_idx)
    all_drop_idx = np.unique(np.concatenate(drop_idx_list))
    return all_drop_idx

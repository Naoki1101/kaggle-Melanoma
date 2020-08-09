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


# def set_channels(child, cfg):
#     if cfg.model.n_channels < 3:
#         child_weight = child.weight.data[:, :cfg.model.n_channels, :, :]
#     else:
#         child_weight = torch.cat([child.weight.data[:, :, :, :], child.weight.data[:, :int(cfg.model.n_channels - 3), :, :]], dim=1)
#     setattr(child, 'in_channels', cfg.model.n_channels)

#     if cfg.model.pretrained:
#         setattr(child.weight, 'data', child_weight)


# def replace_channels(model, cfg):
#     if cfg.model.name.startswith('densenet'):
#         set_channels(model.features[0], cfg)
#     elif cfg.model.name.startswith('efficientnet'):
#         set_channels(model._conv_stem, cfg)
#     elif cfg.model.name.startswith('mobilenet'):
#         set_channels(model.features[0][0], cfg)
#     elif cfg.model.name.startswith('se_resnext'):
#         set_channels(model.layer0.conv1, cfg)
#     elif cfg.model.name.startswith('resnet') or cfg.model.name.startswith('resnex') or cfg.model.name.startswith('wide_resnet'):
#         set_channels(model.conv1, cfg)
#     elif cfg.model.name.startswith('resnest'):
#         set_channels(model.conv1[0], cfg)
#     elif cfg.model.name.startswith('ghostnet'):
#         set_channels(model.features[0][0], cfg)


# def get_head(cfg):
#     head_modules = []
    
#     for m in cfg.values():
#         module = getattr(nn, m['name'])(**m['params'])
#         head_modules.append(module)

#     head_modules = nn.Sequential(*head_modules)
    
#     return head_modules


# def replace_fc(model, cfg):
#     if cfg.model.metric:
#         classes = 1000
#     else:
#         classes = cfg.model.n_classes

#     if cfg.model.name.startswith('densenet'):
#         model.classifier = get_head(cfg.model.head)
#     elif cfg.model.name.startswith('efficientnet'):
#         model._fc = get_head(cfg.model.head)
#     elif cfg.model.name.startswith('mobilenet'):
#         model.classifier[1] = get_head(cfg.model.head)
#     elif cfg.model.name.startswith('se_resnext'):
#         model.last_linear = get_head(cfg.model.head)
#     elif (cfg.model.name.startswith('resnet') or
#           cfg.model.name.startswith('resnex') or
#           cfg.model.name.startswith('wide_resnet') or
#           cfg.model.name.startswith('resnest')):
#         model.fc = get_head(cfg.model.head)
#     elif cfg.model.name.startswith('ghostnet'):
#         model.classifier = get_head(cfg.model.head)

#     return model


# def replace_pool(model, cfg):
#     avgpool = getattr(layer, cfg.model.avgpool.name)(**cfg.model.avgpool.params)
#     if cfg.model.name.startswith('efficientnet'):
#         model._avg_pooling = avgpool
#     elif cfg.model.name.startswith('se_resnext'):
#         model.avg_pool = avgpool
#     elif (cfg.model.name.startswith('resnet') or
#           cfg.model.name.startswith('resnex') or
#           cfg.model.name.startswith('wide_resnet') or
#           cfg.model.name.startswith('resnest')):
#         model.avgpool = avgpool
#     elif cfg.model.name.startswith('ghostnet'):
#         model.squeeze[-1] = avgpool
#     return model


def get_model(cfg, is_train=True):
    model = CustomModel(cfg)
#     if cfg.model.avgpool:
#         model = replace_pool(model, cfg)

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

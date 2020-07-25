import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import loss
import layer
import validation
from models import efficientnet, resnet, resnest, senet, ghostnet
from dataset.custom_dataset import CustomDataset

model_encoder = {
    # efficientnet
    'efficientnet_b0': efficientnet.efficientnet_b0,
    'efficientnet_b1': efficientnet.efficientnet_b1,
    'efficientnet_b2': efficientnet.efficientnet_b2,
    'efficientnet_b3': efficientnet.efficientnet_b3,
    'efficientnet_b4': efficientnet.efficientnet_b4,
    'efficientnet_b5': efficientnet.efficientnet_b5,
    'efficientnet_b6': efficientnet.efficientnet_b6,
    'efficientnet_b7': efficientnet.efficientnet_b7,

    # resnet
    'resnet18': resnet.resnet18,
    'resnet34': resnet.resnet34,
    'resnet50': resnet.resnet50,
    'resnet101': resnet.resnet101,
    'resnet152': resnet.resnet152,
    'resnext50_32x4d': resnet.resnext50_32x4d,
    'resnext101_32x8d': resnet.resnext101_32x8d,
    'wide_resnet50_2': resnet.wide_resnet50_2,
    'wide_resnet101_2': resnet.wide_resnet101_2,

    # resnest
    'resnest50': resnest.resnest50,
    'resnest101': resnest.resnest101,
    'resnest200': resnest.resnest200,
    'resnest269': resnest.resnest269,

    # senet
    'se_resnext50_32x4d': senet.se_resnext50_32x4d,
    'se_resnext101_32x4d': senet.se_resnext101_32x4d,

    # ghostnet
    'ghostnet': ghostnet.ghost_net
}


def set_channels(child, cfg):
    if cfg.model.n_channels < 3:
        child_weight = child.weight.data[:, :cfg.model.n_channels, :, :]
    else:
        child_weight = torch.cat([child.weight.data[:, :, :, :], child.weight.data[:, :int(cfg.model.n_channels - 3), :, :]], dim=1)
    setattr(child, 'in_channels', cfg.model.n_channels)

    if cfg.model.pretrained:
        setattr(child.weight, 'data', child_weight)


def replace_channels(model, cfg):
    if cfg.model.name.startswith('densenet'):
        set_channels(model.features[0], cfg)
    elif cfg.model.name.startswith('efficientnet'):
        set_channels(model._conv_stem, cfg)
    elif cfg.model.name.startswith('mobilenet'):
        set_channels(model.features[0][0], cfg)
    elif cfg.model.name.startswith('se_resnext'):
        set_channels(model.layer0.conv1, cfg)
    elif cfg.model.name.startswith('resnet') or cfg.model.name.startswith('resnex') or cfg.model.name.startswith('wide_resnet'):
        set_channels(model.conv1, cfg)
    elif cfg.model.name.startswith('resnest'):
        set_channels(model.conv1[0], cfg)
    elif cfg.model.name.startswith('ghostnet'):
        set_channels(model.features[0][0], cfg)


def get_head(cfg):
    head_modules = []
    
    for m in cfg.values():
        module = getattr(nn, m['name'])(**m['params'])
        head_modules.append(module)

    head_modules = nn.Sequential(*head_modules)
    
    return head_modules


def replace_fc(model, cfg):
    if cfg.model.metric:
        classes = 1000
    else:
        classes = cfg.model.n_classes

    if cfg.model.name.startswith('densenet'):
        model.classifier = get_head(cfg.model.head)
    elif cfg.model.name.startswith('efficientnet'):
        model._fc = get_head(cfg.model.head)
    elif cfg.model.name.startswith('mobilenet'):
        model.classifier[1] = get_head(cfg.model.head)
    elif cfg.model.name.startswith('se_resnext'):
        model.last_linear = get_head(cfg.model.head)
    elif (cfg.model.name.startswith('resnet') or
          cfg.model.name.startswith('resnex') or
          cfg.model.name.startswith('wide_resnet') or
          cfg.model.name.startswith('resnest')):
        model.fc = get_head(cfg.model.head)
    elif cfg.model.name.startswith('ghostnet'):
        model.classifier = get_head(cfg.model.head)

    return model


def replace_pool(model, cfg):
    avgpool = getattr(layer, cfg.model.avgpool.name)(**cfg.model.avgpool.params)
    if cfg.model.name.startswith('efficientnet'):
        model._avg_pooling = avgpool
    elif cfg.model.name.startswith('se_resnext'):
        model.avg_pool = avgpool
    elif (cfg.model.name.startswith('resnet') or
          cfg.model.name.startswith('resnex') or
          cfg.model.name.startswith('wide_resnet') or
          cfg.model.name.startswith('resnest')):
        model.avgpool = avgpool
    elif cfg.model.name.startswith('ghostnet'):
        model.squeeze[-1] = avgpool
    return model


class CustomCnn(nn.Module):
    def __init__(self, model):
        super(CustomCnn, self).__init__()
        self.model = model
        self.linear1 = nn.Linear(29, 128)
        self.linear2 = nn.Linear(256, 1)
        self.bn1 = nn.BatchNorm1d(29)

    def forward(self, x, feats):
        x = self.model(x)

        feats = self.bn1(feats)
        feats = self.linear1(feats)

        x = torch.cat([x, feats], axis=1)
        x = self.linear2(x)
        return x


def get_model(cfg, is_train=True):
    model = model_encoder[cfg.model.name](pretrained=cfg.model.pretrained)
    if cfg.model.n_channels != 3:
        replace_channels(model, cfg)
    model = replace_fc(model, cfg)
    model = CustomCnn(model)
    if cfg.model.avgpool:
        model = replace_pool(model, cfg)

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
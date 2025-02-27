import sys

import torch
import torch.nn as nn

sys.path.append('../src')
import layer
from . import efficientnet, resnet, resnest, senet, ghostnet


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
    'resnest50_frelu': resnest.resnest50_frelu,
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
        if hasattr(nn, m['name']):
            module = getattr(nn, m['name'])(**m['params'])
        elif hasattr(layer, m['name']):
            module = getattr(layer, m['name'])(**m['params'])
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


class CustomModel(nn.Module):
    def __init__(self, cfg):
        super(CustomModel, self).__init__()
        self.base_model = model_encoder[cfg.model.name](pretrained=cfg.model.pretrained)
        if cfg.model.n_channels != 3:
            replace_channels(self.base_model, cfg)
        if cfg.model.avgpool:
            self.base_model = replace_pool(self.base_model, cfg)
        self.model = replace_fc(self.base_model, cfg)

    def forward(self, x):
        x = self.model(x)

        return x
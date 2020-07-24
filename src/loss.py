import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bcel = nn.BCEWithLogitsLoss()

    def forward(self, yhat, y):
        return self.bcel(yhat, y)

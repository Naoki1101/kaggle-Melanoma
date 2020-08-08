import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bcel = nn.BCEWithLogitsLoss()

    def forward(self, yhat, y):
        return self.bcel(yhat, y)


# https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/162035#905556
class WeightedFocalLoss(nn.Module):
    "Weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
import numpy as np
import torch
import torch.nn as nn


def dice_score(pred, target, num_labels):
    eps = 0.0001
    dice = np.zeros(num_labels - 1)
    for label in range(1, num_labels):
        target_surface = target == label
        pred_surface = pred == label

        iflat = pred_surface.reshape(-1)
        tflat = target_surface.reshape(-1)
        intersection = (iflat * tflat).sum()
        union = iflat.sum() + tflat.sum()

        dice[label - 1] = (2.0 * intersection + eps) / (union + eps)
    return np.mean(dice)


class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()
        self.eps = 0.0001

    def forward(self, x, target):
        target = target.type(x.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)
        dice_loss = ((2. * intersection + self.eps) / (cardinality + self.eps)).mean()
        return -dice_loss

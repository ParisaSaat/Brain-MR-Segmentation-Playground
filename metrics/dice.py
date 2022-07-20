import torch.nn as nn


def dice_score(pred, target, num_labels=2):
    eps = 0.0001
    dice = 0
    for label in range(1, num_labels):
        if num_labels > 2:
            target_surface = target == label
            pred_surface = pred == label
        else:
            target_surface = target
            pred_surface = pred
        iflat = pred_surface.reshape(-1)
        tflat = target_surface.reshape(-1)
        intersection = (iflat * tflat).sum()
        union = iflat.sum() + tflat.sum()

        dice += (2.0 * intersection + eps) / (union + eps)
    return dice / (num_labels - 1)


def dice_scoree(pred, target, num_labels=2):
    eps = 0.0001
    iflat = pred.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()
        self.eps = 0.0001

    def forward(self, x, target, num_labels=2):
        #         target = target.type(x.type())
        #         dims = (0,) + tuple(range(2, target.ndimension()))
        #         intersection = torch.sum(x * target, dims)
        #         cardinality = torch.sum(x + target, dims)
        #         dice_loss = ((2. * intersection + self.eps) / (cardinality + self.eps)).mean()
        dice_loss = -dice_score(x, target, num_labels)
        return dice_loss

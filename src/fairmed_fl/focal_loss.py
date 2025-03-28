import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, pos_weight=None, reduction="mean"):
        """
        Multi-label Focal Loss for Binary Classification.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):

        inputs = torch.clamp(
            inputs, min=1e-6, max=1 - 1e-6
        )  # Replace extreme values [0, 1]

        bce_loss = nn.functional.binary_cross_entropy(
            inputs, targets, weight=self.pos_weight, reduction="none"
        )

        # Calculating the probability of the true class
        pt = torch.where(
            targets == 1, inputs, 1 - inputs
        )  # pt for positive and (1-pt) for negative examples
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = focal_weight * alpha_weight

        focal_loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # None
            return focal_loss

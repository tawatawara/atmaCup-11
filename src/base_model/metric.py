# -*- coding: utf-8 -*- #
"""metric functions."""
import typing as tp

import numpy as np

import torch
from torch import nn
from sklearn.metrics import roc_auc_score


class ROCAUC(nn.Module):
    """ROC AUC score"""

    def __init__(self, average="macro") -> None:
        """Initialize."""
        self.average = average
        super(ROCAUC, self).__init__()

    def forward(self, y, t) -> float:
        """Forward."""
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()

        return roc_auc_score(t, y, average=self.average)


class Accuracy(nn.Module):
    """Micro Accuracy for multi-class classification."""

    def __init__(self, ignored_class: int=-1) -> None:
        """Initialize."""
        self.ignored_class = ignored_class
        super(Accuracy, self).__init__()

    def forward(self, y, t):
        """Forward."""
        indices = torch.argmax(y, dim=1)
        mask = (t != self.ignored_class)
        mask &= (indices != self.ignored_class)
        indices = indices[mask]
        t = t[mask]
        correct = torch.eq(indices, t).view(-1)

        return torch.sum(correct).item() / correct.shape[0]


class RMSE(nn.Module):
    """Root Mean Squared Error"""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()

    def forward(self, y, t) -> float:
        """Forward."""
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()

        return np.sqrt(np.mean((t - y) ** 2))

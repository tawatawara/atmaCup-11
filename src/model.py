# -*- coding: utf-8 -*- #
"""Newly created model, loss, metric for competition"""
import numpy as np
import torch
from torch import nn


class RMSEWithLogits(nn.Module):
    """
    Root Mean Squared Error for multi-class classification
    
    * **this class only for metric**, not loss
    * apply softmax then perform weighted summation classes by `probs`
    """

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()

    def forward(self, y, t) -> float:
        """Forward."""
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()

        n_classes = y.shape[1]
        p = np.exp(y - y.max(axis=1, keepdims=True))
        p = p / p.sum(axis=1, keepdims=True)
        y = (p * np.arange(n_classes)).sum(axis=1)
        return np.sqrt(np.mean((t - y) ** 2))


class RMSEWithClip(nn.Module):
    """
    Root Mean Squared Error with clipping
    
    * **this class only for metric**, not loss
    * apply clipping min and max value, then calculate RMSE
    """

    def __init__(self, min_value: int, max_value: int) -> None:
        """Initialize."""
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, y, t) -> float:
        """Forward."""
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        y = np.clip(y, self.min_value, self.max_value)

        return np.sqrt(np.mean((t - y) ** 2))

"""
losses.py
"""

__all__ = [
    "KLDivLatentLoss",
    "WeightedMSELoss"
]

from typing import Sequence
import os
import numpy as np
import torch
from torch import nn
from src.attrs import PATHS, GLOBALS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KLDivLatentLoss(nn.Module):
    def __init__(self):
        super(KLDivLatentLoss, self).__init__()
    def forward(self, mean: torch.tensor, logvar: torch.tensor):
        loss = torch.mean(-0.5 * torch.sum(1. + logvar - logvar.exp() - mean**2, dim=1), dim=0)
        return loss

class WeightedMSELoss(nn.Module):
    def __init__(self, weights: Sequence = None):
        """
        weights are whatever should be applied to the original dataset,
        e.g. areas. 

        If weights are not specified, then weights are loaded from 
        `{PATHS['tensors']}/weights.pt`.
        """
        super(WeightedMSELoss, self).__init__()
        if weights is None:
            weights = torch.load(os.path.join(PATHS['tensors'], 'weights.pt'))
        self.weights = (
            weights
            .clone()
            .detach()
            .to(torch.float32)
            .to(DEVICE)
            .reshape(1, -1)
        )

    def forward(self, input_: torch.Tensor, target: torch.Tensor):
        weighted_sum_sq = torch.sum(self.weights**2 * (input_ - target)**2, axis=1)
        sample_mse = weighted_sum_sq / torch.sum(self.weights**2)
        weighted_mse = sample_mse.mean(axis=0)
        return weighted_mse

class WeightedMSE2dLoss(nn.Module):
    def __init__(self):
        """
        Weighted MSE loss function. Weighted MSE is computed over lat/lon, and
        then the simple average is taken over all channels and samples. (Uses 
        the convention that tensors are shape (N, C, H, W), where N is the 
        number of samples, C is the number of channels, H is the number of lats
        and W is the number of lons.
        """
        super(WeightedMSE2dLoss, self).__init__()
        
    def forward(self, input_: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        weighted_sum_sq = torch.sum(weights**2 * (input_ - target)**2, dim=(2,3))
        sample_mse = weighted_sum_sq / torch.sum(weights**2)
        weighted_mse = sample_mse.mean(dim=(0,1))
        return weighted_mse
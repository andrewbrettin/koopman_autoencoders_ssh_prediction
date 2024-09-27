"""
metrics.py
"""

__all__ = [
    "weighted_mse",
    "weighted_variance_explained",
    "wasserstein_error",
]

from typing import Union, Sequence
import os
import sys
from datetime import datetime, timedelta

import numpy as np
from scipy import linalg, stats
import pandas as pd
import xarray as xr
import xrft

import torch

from src.train import losses
from src.attrs import PATHS, GLOBALS


def weighted_mse(x: np.ndarray, y: np.ndarray, weights: np.ndarray = None):
    """
    Computes weighted MSE of x and y.

    Parameters:
        x: array-like
            Input array.
        y: array-like
            Target array.
        weights: torch.Tensor
            Weights.
    Returns:
        fracvar: float
            Fraction of variance explained.
    """
    if weights is None:
        weights = torch.load(os.path.join(PATHS['tensors'], 'weights.pt'))
        weights = weights.clone().detach().numpy()
    weighted_sum_sq = np.sum(weights**2 * (x - y)**2, axis=1)
    sample_mse = weighted_sum_sq / np.sum(weights**2)
    mse_weighted = sample_mse.mean(axis=0)
    return mse_weighted
    
def weighted_pattern_correlation(x, y, weights):
    cov = np.sum(weights**2 * x * y, axis=1)
    var_x = np.sum(weights**2 * x**2, axis=1)
    var_y = np.sum(weights**2 * y**2, axis=1)
    r = cov / np.sqrt(var_x * var_y)
    return r.mean(axis=0)
    
def weighted_variance_explained(X, X_pred, weights=None):
    """
    Computes weighted variance of X_pred relative to weighted variance of X.

    Parameters:
        X: array-like
            Target array.
        X_pred: array-like
            Reconstructed values.
        weights: torch.Tensor
            Weights.
    Returns:
        fracvar: float
            Fraction of variance explained.
    """
    if weights is None:
        weights = torch.load(os.path.join(PATHS['tensors'], 'weights.pt'))
    X = X.detach().numpy()
    X_pred = X_pred.detach().numpy()
    weights = weights.detach().numpy()
    pred_var = linalg.norm(X_pred * weights, ord='fro')**2
    full_var = linalg.norm(X * weights, ord='fro')**2
    fracvar = pred_var / full_var
    return fracvar

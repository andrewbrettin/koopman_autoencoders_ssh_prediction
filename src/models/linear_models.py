"""
linear_models.py

Linear prediction models.
"""

__all__ = [
    "PersistenceModel",
    "DampedPersistenceModel",
    "LinearInverseModel"
]

from typing import Union
import os
from datetime import timedelta

import numpy as np
from scipy import stats, linalg
import pandas as pd
import xarray as xr
import torch

import src
from src.attrs import PATHS, GLOBALS
from src.tools import comp, processing

class LinearModel():
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


class DampedPersistenceModel(LinearModel):
    def __init__(self):
        self.A = None
        self.nu = None
        self.mode_mean = 0

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        nu: int,
        remove_mean: bool = True
    ):
        """
        Fits the model by setting attributes nu and A.

        We need to account for the fact that latent modes might have a
        time-average non-zero mean. We do this by saving the time averages
        to self.mode_mean.

        NOTE: axis 0 of the datasets X and y are assumed to be along the 
        time/sample dimension.

        Parameters:
            X: np.ndarray, torch.Tensor
                Inputs with time dimension along axis 0.
            y: np.ndarray, torch.Tensor
                Target values with time dimension along axis 0.
            nu: int
                Time lag for fitting.
        
        Returns: self with attributes:
            A: np.ndarray
                Evolution matrix of autocorrelations at lag time nu.
            nu: int
                Number of timesteps used for fitting autocorrelations. 
                This is given the symbol `tau_0` in the paper.
            mode_mean: np.ndarray
                Time average of inputs.
        """
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        
        self.nu = nu

        # Remove time mean
        if remove_mean:
            self.mode_mean = X.mean(axis=0).reshape(1,-1)
            X = X - self.mode_mean
            y = y - self.mode_mean
        
        # Compute autocorrelation
        autocorr = comp.npcorr(X, y)
        A = np.diag(autocorr)

        # Enforce nonnegativity of entries
        self.A = np.maximum(A, np.zeros_like(A))

        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor], tau: int):
        """
        Predicts values at forecast time horizon tau.

        The propagator is computed by A**p, where p = tau/nu.

        If X is a DataArray, time coordinates are shifted by timedelta(tau).
        """
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        
        # Number of iterations
        p = tau / self.nu

        # Remove mean
        X = X - self.mode_mean
        
        # Compute propagator
        if p == 0:
            B = np.eye(X.shape[1])
        else:
            B = self.A**p

        # Make predictions
        pred = B @ X.T
        pred = pred.T
        pred = pred + self.mode_mean
        
        return pred


class LinearInverseModel(LinearModel):
    def __init__(self):
        self.A = None
        self.nu = None
        self.mode_mean = 0

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        nu: int,
        remove_mean: bool = True
    ):
        """
        Fits the model by setting attributes nu and A.

        We need to account for the fact that latent modes might have a
        time-average non-zero mean. We do this by saving the time averages
        to self.mode_mean.

        NOTE: axis 0 of the datasets X and y are assumed to be along the 
        time/sample dimension.

        Parameters:
            X: np.ndarray, torch.Tensor
                Inputs with time dimension along axis 0.
            y: np.ndarray, torch.Tensor
                Target values with time dimension along axis 0.
            nu: int
                Time lag for fitting.
        
        Returns: self with attributes:
            A: np.ndarray
                Evolution matrix of autocorrelations at lag time nu.
            nu: int
                Number of timesteps used for fitting autocorrelations. 
                This is given the symbol `tau_0` in the paper.
            mode_mean: np.ndarray
                Time average of inputs.
        """
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        
        self.nu = nu

        # Remove time mean
        if remove_mean:
            self.mode_mean = X.mean(axis=0).reshape(1,-1)
            X = X - self.mode_mean
            y = y - self.mode_mean
        
        # Compute response operator
        C_nu = y.T @ X
        C_0 = X.T @ X
        self.A = (1 / self.nu) * linalg.logm(C_nu @ linalg.inv(C_0))

        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor], tau: int):
        """
        Predicts values at forecast time horizon tau.
        """
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        # Remove mean
        X = X - self.mode_mean
        
        # Compute propagator
        if tau == 0:
            B = np.eye(X.shape[1])
        else:
            B = linalg.expm(tau * self.A)
            B = B.real

        # Make predictions
        pred = B @ X.T
        pred = pred.T
        pred = pred + self.mode_mean
        
        return pred
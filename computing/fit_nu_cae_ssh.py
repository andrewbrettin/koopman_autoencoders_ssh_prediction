"""
fit_nu.py

Loops over DP and LIM models trained using different fitted timescales nu,
and computes forecast MSE on the validation data.

The output of this is a netcdf saved to PATHS[SUBPROJECT]/fit_nu/<model>_val_mse.nc.
This netcdf has dimensions (nu, lag), where `nu` ranges from 1 to K and `lag` ranges from 
0 to K-1 (where K is the number of recurrent passes used for the koopman autoencoder).
"""

from typing import Sequence, Union
import os
import sys
import glob
import itertools
from datetime import datetime
from pprint import pprint
from tqdm import tqdm
import dask
import joblib

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import torch.distributions
import pytorch_lightning as pl

import src
from src.attrs import PATHS, GLOBALS
from src import utils

from src.data import loading
from src.train import datasets, losses
from src.models import base, cnn
from src.tools import plot

# Globals
REGION    = sys.argv[1]
SAMPLING  = sys.argv[2]
N_LATENTS = int(sys.argv[3])
BASELINE  = sys.argv[4]

SUBPROJECT = f'cnn_{REGION}_{SAMPLING}'
K = 20
MAX_SAMPLES = 10_000    # Only compute metrics over up to 5k samples for efficiency

# Runtime variables
START_TIME = datetime.now()
dask.config.set(scheduler='synchronous')
DEVICE = torch.device('cuda')
torch.set_grad_enabled(False)

# Functions
def load_prediction_baseline(baseline_type, n_latents, nu, path=PATHS[SUBPROJECT]):
    if baseline_type == 'dp':
        prediction_model = joblib.load(
            os.path.join(path, 'dp', f'dp_cae_ssh_d{n_latents}_nu{nu}.joblib')
        )
    elif baseline_type == 'lim':
        prediction_model = joblib.load(
            os.path.join(path, 'lim', f'lim_cae_ssh_d{n_latents}_nu{nu}.joblib')
        )
    return prediction_model

def weighted_mse(X, X_hat, weights):
    """
    Weighted MSE over all samples. Channels are preserved.
    """
    return torch.mean(
        torch.sum(weights**2 * (X - X_hat)**2, dim=(2,3)) / weights.sum(),
        dim=0
    )

def model_prediction(model, X, t, weights=None):
    """
    Returns prediction of model at timestep t

    Parameters:
        model: 'clim', cnn.CNNKoopmanAutoencoder, or list
            Dimensionality reduction and prediction models.
            If model is type CNNKoopmanAutoencoder, reduction and
            prediction steps are done simultaneously. Otherwise, `model` 
            should be a tuple of 2 items, a dimensionality reduction model
            (e.g., `PCA` or `CAE`) and a prediction model (`DP` or `LIM`).

            Need to handle PCA vs CAE differently.
        X: torch.Tensor
            Input data for prediction. Should be shape (N, C, H, W), where
            N is the number of samples, C is the number of channels, H is 
            the number of latitudes, and W is the number of longitudes.
        t: int
            Number of prediction timesteps. 
    """
    # CNN Koopman predictions
    if isinstance(model, cnn.CNNKoopmanAutoencoder):
        X_pred = torch.zeros_like(X)
        for i, sample in enumerate(X):
            sample = sample.to(DEVICE)
            sample = sample.view(1, *sample.shape)
            X_pred[i, :] = model.multistep_prediction(sample, t).cpu()
        return X_pred
    # Compression method + prediction method
    elif isinstance(model, Sequence):
        # Set compression and prediction models
        compression_model = model[0]
        prediction_model = model[1]
        
        if isinstance(compression_model, PCA):
            X_shape = X.shape
            X_weighted = X * weights
            X_weighted = X_weighted.view(X_weighted.shape[0], -1)
            z = compression_model.transform(X_weighted)
            # Prediction step
            z_pred = prediction_model.predict(z, t)
            X_pred = compression_model.inverse_transform(z_pred)
            # Convert to tensor, reshape, reweight, fill nans
            X_pred = torch.from_numpy(X_pred)
            X_pred = X_pred.view(*X_shape)
            X_pred = X_pred / weights
            X_pred = torch.where(weights != 0, X_pred, 0)
            return X_pred
            
        elif isinstance(compression_model, cnn.CNNAutoencoder):
            D = compression_model.configs['D']
            z_pred = torch.zeros((X.shape[0], D))
            for i, sample in enumerate(X):
                sample = sample.to(DEVICE)
                sample = sample.view(1, *sample.shape)
                z_pred[i, :] = compression_model.conv_encoder(sample).cpu()
            # Apply prediction timestep
            z_pred = prediction_model.predict(z_pred, t)
            z_pred = torch.from_numpy(z_pred.astype(np.float32))
            # Apply decoding
            X_pred = torch.zeros_like(X)
            for i, sample in enumerate(z_pred):
                sample = sample.to(DEVICE)
                sample = sample.view(1, *sample.shape)
                X_pred[i, :] = compression_model.conv_decoder(sample).cpu()
            return X_pred
            
    else:
        raise ValueError(f'Model {model} not valid')

def compute_mse(
    model: Union[cnn.CNNKoopmanAutoencoder, Sequence], 
    X: torch.Tensor, 
    mask: xr.DataArray, 
    weights: torch.Tensor,
    n_iter: int = K
):
    """
    Compute metrics for predictions at all timesteps
    """
    variables = ['SSH']
    
    global_mse_darray = xr.DataArray(
        np.empty((n_iter, len(variables))), 
        coords={'lag': np.arange(n_iter), 'variable': variables}
    )

    # Make predictions using mode
    for t in np.arange(0, n_iter):
        X_pred = model_prediction(model, X, t, weights=weights)

        # Compute slices
        pslice = slice(None, None) if t == 0 else slice(None, -t)
        global_mse_darray[t, :] = weighted_mse(X[t:, ...], X_pred[pslice, ...], weights)
    
    return global_mse_darray


def main():
    utils.log(f"Begin {__file__}")
    print(f"Region: {REGION}\nSampling: {SAMPLING}\nD={N_LATENTS}\nBaseline: {BASELINE}")

    utils.log("Load validation data", START_TIME)
    X_val = xr.open_dataarray(
        os.path.join(PATHS[SUBPROJECT], f'X_{GLOBALS["val_id"]}.nc')
    )
    X_val = X_val[0:MAX_SAMPLES, 0:1, ...]
    X_val = torch.from_numpy(X_val.values)
    weights = torch.load(os.path.join(PATHS[SUBPROJECT], 'weights.pt'))
    mask = xr.open_dataarray(os.path.join(PATHS[SUBPROJECT], 'mask.nc'))

    utils.log(f"Load CAE model for D={N_LATENTS}", START_TIME)
    cae = base.load_model_from_yaml(os.path.join(PATHS[SUBPROJECT], 'cae', f'cae_ssh.{N_LATENTS}'))
    cae = cae.to(DEVICE)

    utils.log(f"Computing MSE for {BASELINE}", START_TIME)
    mse_all = xr.DataArray(
        np.empty((K, K)),
        coords={
            'nu': np.arange(1, K+1),
            'lag': np.arange(K)
        }
    )
    
    for i, nu in enumerate(tqdm(np.arange(1, K+1))):
        prediction_model = load_prediction_baseline(
            BASELINE, N_LATENTS, nu, path=PATHS[SUBPROJECT]
        )
        models = (cae, prediction_model)
        mse = compute_mse(models, X_val, mask, weights, n_iter=K)
        mse_all[i, :] = mse.mean(dim='variable').values

    # Save as netcdf
    utils.log("Saving data", START_TIME)
    output_path = os.path.join(PATHS[SUBPROJECT], 'fit_nu')
    os.makedirs(output_path, exist_ok=True)
    mse_all.to_netcdf(
        os.path.join(output_path, f'cae_{BASELINE}_ssh_d{N_LATENTS}_val_mse.nc')
    )

    # Compute prediction error average over all lags
    mse_all_avg = mse_all.mean(dim='lag')
    optimal_nu = mse_all_avg.nu[mse_all_avg.argmin()].item()
    print(f"BEST BASELINE {BASELINE}: nu={optimal_nu}")
    
    utils.log("Saving best baseline", START_TIME)
    
    best_baseline = joblib.load(
        os.path.join(
            PATHS[SUBPROJECT],
            BASELINE,
            f'{BASELINE}_cae_ssh_d{N_LATENTS}_nu{optimal_nu}.joblib'
        )
    )
    
    joblib.dump(
        best_baseline, 
        os.path.join(
            PATHS[SUBPROJECT],
            BASELINE,
            f'{BASELINE}_cae_ssh_d{N_LATENTS}_best.joblib'
        )
    )

    utils.log("PROCESS COMPLETE", START_TIME)

    return 0

if __name__ == "__main__":
    main()


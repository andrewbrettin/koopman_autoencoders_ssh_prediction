"""
forecast_metrics.py

Computes metrics for forecasts for CNN experiments.

Note: this code is going to be a bit messy, and may require some modifications 
to the global variables as well as to main() as needed.
"""

from typing import Sequence, Union
import os
import sys
import glob
import itertools
from datetime import datetime
from pprint import pprint
from tqdm import tqdm
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
REGION     = sys.argv[1]
SAMPLING   = sys.argv[2]
N_LATENTS  = int(sys.argv[3])
K = 20
SUBPROJECT = f'cnn_{REGION}_{SAMPLING}'

if SAMPLING == 'monthly':
    N_ITER = 36
    DT = 1
elif SAMPLING == 'daily_subsampled':
    N_ITER = 120
    DT = 5
else:
    raise ValueError(f"Invalid sampling {SAMPLING}")

MAX_SAMPLES = None 
globals = {'region': REGION, 'sampling': SAMPLING, 'D': N_LATENTS, 'dt': DT}

# Runtime variables
START_TIME = datetime.now()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)

def sample_mse(X, X_hat):
    """
    Takes the MSE over all samples without aggregating over lat/lon.
    """
    return torch.mean((X - X_hat)**2, dim=0)

def weighted_mse(X, X_hat, weights):
    """
    Weighted MSE over all samples. Channels are preserved.
    """
    return torch.mean(
        torch.sum(weights**2 * (X - X_hat)**2, dim=(2,3)) / weights.sum(),
        dim=0
    )

def weighted_corr(X, X_hat, weights):
    """
    Weighted correlation over all samples. Channels are preserved.
    """
    return torch.mean(
        torch.sum(weights**2 * X * X_hat, dim=(2,3)) 
        / torch.sqrt(
            torch.sum((weights * X)**2, dim=(2,3)) 
            * torch.sum((weights * X_hat)**2, dim=(2,3))
        ),
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

def compute_metrics(
    model: Union[cnn.CNNKoopmanAutoencoder, Sequence], 
    X: torch.Tensor, 
    mask: xr.DataArray, 
    weights: torch.Tensor
):
    """
    Compute metrics for predictions at all timesteps
    """
    variables = ['SSH']

    sample_mse_darray = xr.DataArray(
        np.empty((N_ITER // DT, len(variables), *mask.shape)), 
        coords={'lag': np.arange(0, N_ITER, DT), 'variable': variables, **mask.coords}
    )
    
    global_mse_darray = xr.DataArray(
        np.empty((N_ITER // DT, len(variables))), 
        coords={'lag': np.arange(0, N_ITER, DT), 'variable': variables}
    )
    
    global_corr_darray = xr.DataArray(
        np.empty((N_ITER // DT, len(variables))), 
        coords={'lag': np.arange(0, N_ITER, DT), 'variable': variables}
    )

    # Make predictions using mode
    for i, t in enumerate(tqdm(np.arange(0, N_ITER, DT))):
        X_pred = model_prediction(model, X, t, weights=weights)

        # Compute slices
        pslice = slice(None, None) if t == 0 else slice(None, -t)
        sample_mse_darray[i, ...] = sample_mse(X[t:, ...], X_pred[pslice, ...])
        global_mse_darray[i, :] = weighted_mse(X[t:, ...], X_pred[pslice, ...], weights)
        global_corr_darray[i, :] = weighted_corr(X[t:, ...], X_pred[pslice, ...], weights)

    # Use mask to remove nans
    sample_mse_darray = xr.where(mask, sample_mse_darray, np.nan)
    
    return sample_mse_darray, global_mse_darray, global_corr_darray


def climatological_metrics(X, mask, weights):
    """
    Computes metrics for climatological predictions w.r.t. X.
    """
    variables = ['SSH']

    sample_mse_darray = xr.DataArray(
        np.empty((len(variables), *mask.shape)), 
        coords={'variable': variables, **mask.coords}
    )
    
    global_mse_darray = xr.DataArray(
        np.empty(len(variables)), 
        coords={'variable': variables}
    )
    
    global_corr_darray = xr.DataArray(
        np.empty(len(variables)), 
        coords={'variable': variables}
    )

    X_pred = torch.zeros_like(X)
    sample_mse_darray.values = sample_mse(X, X_pred)
    global_mse_darray.values = weighted_mse(X, X_pred, weights)
    global_corr_darray.values = weighted_corr(X, X_pred, weights)

    sample_mse_darray = xr.where(mask, sample_mse_darray, np.nan)

    return sample_mse_darray, global_mse_darray, global_corr_darray

def save_metrics(metrics, output_path, file_prefix):
    sample_mse, global_mse, global_corr = metrics
    sample_mse.to_netcdf(os.path.join(output_path, f'{file_prefix}_sample_mse.nc'))
    global_mse.to_netcdf(os.path.join(output_path, f'{file_prefix}_global_mse.nc'))
    global_corr.to_netcdf(os.path.join(output_path, f'{file_prefix}_global_corr.nc'))

    return None

def main():
    utils.log(f"Begin script {__file__}")
    utils.print_os_environ()
    print(SUBPROJECT)
    pprint(globals)
    output_path = os.path.join(PATHS[SUBPROJECT], 'forecast_metrics')
    os.makedirs(output_path, exist_ok=True)

    utils.log("Load data", START_TIME)
    X_test = xr.open_dataarray(os.path.join(PATHS[SUBPROJECT], 'X_8.nc'))
    X_test = X_test[0:MAX_SAMPLES, 0:1, ...]
    X_test = torch.from_numpy(X_test.values)
    weights = torch.load(os.path.join(PATHS[SUBPROJECT], 'weights.pt'))
    mask = xr.open_dataarray(os.path.join(PATHS[SUBPROJECT], 'mask.nc'))

    # Climatological MSE
    metrics = climatological_metrics(X_test, mask, weights)
    save_metrics(metrics, output_path, 'clim_ssh')

    # Koopman autoencoder predictions (copy to folder `kae` before computing metrics
    utils.log(f"Metrics for CNN Koopman D={N_LATENTS}, k={K}", START_TIME)
    model = base.load_model_from_yaml(os.path.join(PATHS[SUBPROJECT], 'kae', f'kae_ssh.{N_LATENTS}'))
    model = model.to(DEVICE)
    metrics = compute_metrics(model, X_test, mask, weights)
    save_metrics(metrics, output_path, f'kae_ssh_d{N_LATENTS}')
    
    # PCA + DP
    utils.log(f"Metrics for PCA + DP, D = {N_LATENTS}", START_TIME)
    pca_model = joblib.load(os.path.join(PATHS[SUBPROJECT], 'pca', f'pca_ssh_{N_LATENTS}.joblib'))
    dp_model = joblib.load(os.path.join(PATHS[SUBPROJECT], 'dp', f'dp_pca_ssh_d{N_LATENTS}_best.joblib'))
    models = (pca_model, dp_model)
    metrics = compute_metrics(models, X_test, mask, weights)
    save_metrics(metrics, output_path, f'dp_pca_ssh_d{N_LATENTS}_best')

    # PCA + LIM
    utils.log(f"Metrics for PCA + LIM, D = {N_LATENTS}", START_TIME)
    pca_model = joblib.load(os.path.join(PATHS[SUBPROJECT], 'pca', f'pca_ssh_{N_LATENTS}.joblib'))
    lim_model = joblib.load(os.path.join(PATHS[SUBPROJECT], 'lim', f'lim_pca_ssh_d{N_LATENTS}_best.joblib'))
    models = (pca_model, lim_model)
    metrics = compute_metrics(models, X_test, mask, weights)
    save_metrics(metrics, output_path, f'lim_pca_ssh_d{N_LATENTS}_best')
    
    # CAE + DP
    autoencoder = base.load_model_from_yaml(os.path.join(PATHS[SUBPROJECT], 'cae', f'cae_ssh.{N_LATENTS}'))
    autoencoder = autoencoder.to(DEVICE)
    dp_model = joblib.load(os.path.join(PATHS[SUBPROJECT], 'dp', f'dp_cae_ssh_d{N_LATENTS}_best.joblib'))
    models = (autoencoder, dp_model)
    metrics = compute_metrics(models, X_test, mask, weights)
    save_metrics(metrics, output_path, f'dp_cae_ssh_d{N_LATENTS}_best')

    # CAE + LIM
    utils.log(f"Metrics for CAE + LIM, D = {N_LATENTS}", START_TIME)
    autoencoder = base.load_model_from_yaml(os.path.join(PATHS[SUBPROJECT], 'cae', f'cae_ssh.{N_LATENTS}'))
    autoencoder = autoencoder.to(DEVICE)
    lim_model = joblib.load(os.path.join(PATHS[SUBPROJECT], 'lim', f'lim_cae_ssh_d{N_LATENTS}_best.joblib'))
    models = (autoencoder, lim_model)
    metrics = compute_metrics(models, X_test, mask, weights)
    save_metrics(metrics, output_path, f'lim_cae_ssh_d{N_LATENTS}_best')

    utils.log("PROCESS COMPLETED", START_TIME)

    return 0


if __name__ == "__main__":
    main()












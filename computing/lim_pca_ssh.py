"""
lim_pca.py

Fits Linear Inverse model to data from the CNN experiments
"""

import os
import sys
import itertools
from datetime import datetime, timedelta
import joblib
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr
import torch

import src
from src.attrs import PATHS, GLOBALS
from src import utils
from src.data import loading
from src.models import linear_models

# Globals
REGION = sys.argv[1]    # 'pacific'
SAMPLING = sys.argv[2]  # 'daily', 'monthly'
N_LATENTS = int(sys.argv[3])
LATENT_SIZES = [20]
C = 1                  # Number of channels
NU_MAX = 20  # Maximum number of reccurent passes for Koopman

# Runtime variables
START_TIME = datetime.now()

def main():
    utils.log(f"Begin script {__file__}")
    utils.print_os_environ()
    subproject_name = f'cnn_{REGION}_{SAMPLING}'
    print(subproject_name)

    # Load data
    utils.log(f"Load data for PCA", START_TIME)
    weights = torch.load(os.path.join(PATHS[subproject_name], 'weights.pt'))
    X_multipass = torch.load(
        os.path.join(PATHS[subproject_name], 'multipass', f'train_ssh_k{NU_MAX}.pt')
    )

    for nu in np.arange(1, NU_MAX+1):
        utils.log(f"Fitting LIM nu={nu}", START_TIME)
        # Select input and target from multipass and apply weights
        X_train = X_multipass[:, 0:C, :, :]
        y_train = X_multipass[:, nu*C:(nu+1)*C, :, :]
        
        X_train_weighted = X_train * weights
        y_train_weighted = y_train * weights
        del X_train, y_train
        
        # Reshape
        X_train_weighted = X_train_weighted.view(X_train_weighted.shape[0], -1)
        y_train_weighted = y_train_weighted.view(X_train_weighted.shape[0], -1)
        
        utils.log(f"Training LIM with D={N_LATENTS}", START_TIME)
        # Load PCA Model
        pca_model = joblib.load(
            os.path.join(PATHS[subproject_name], 'pca', f'pca_ssh_{N_LATENTS}.joblib')
        )
        z_train = pca_model.transform(X_train_weighted)
        z_train_next = pca_model.transform(y_train_weighted)

        # Fit damped persistence model
        lim = linear_models.LinearInverseModel()
        lim.fit(X=z_train, y=z_train_next, nu=nu)

        # Save damped persistence model
        utils.log(f"Saving LIM to {PATHS[subproject_name]}", START_TIME)
        path = os.path.join(PATHS[subproject_name], 'lim')
        os.makedirs(path, exist_ok=True)
        filename = f'lim_pca_ssh_d{N_LATENTS}_nu{nu}.joblib'
        joblib.dump(lim, os.path.join(path, filename))

    utils.log("PROCESS_COMPLETED", START_TIME)

    return 0

if __name__ == "__main__":
    main()

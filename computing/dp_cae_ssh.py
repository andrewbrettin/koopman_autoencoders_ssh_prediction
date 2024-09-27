"""
dp_cae_ssh.py

Fits Linear Inverse model to data from the CNN experiments
"""

import os
import sys
from tqdm import tqdm
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
from src.models import base, linear_models

# Globals
REGION = sys.argv[1]    # 'pacific'
SAMPLING = sys.argv[2]  # 'daily', 'monthly'
N_LATENTS = int(sys.argv[3])
C = 1                   # Number of channels
NU_MAX = 20

# Runtime variables
START_TIME = datetime.now()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)

def main():
    utils.log(f"Begin script {__file__}")
    utils.print_os_environ()
    subproject_name = f'cnn_{REGION}_{SAMPLING}'
    print(subproject_name)

    # Load data
    utils.log(f"Load data for CAE", START_TIME)
    weights = torch.load(os.path.join(PATHS[subproject_name], 'weights.pt'))
    X_multipass = torch.load(
        os.path.join(PATHS[subproject_name], 'multipass', f'train_ssh_k{NU_MAX}.pt')
    )

    # Load CAE
    utils.log("Load CAE", START_TIME)
    cae = base.load_model_from_yaml(os.path.join(PATHS[subproject_name], 'cae', f'cae_ssh.{N_LATENTS}'))
    cae = cae.to(DEVICE)

    # Compute z_train
    utils.log("Computing z_train", START_TIME)
    X_train = X_multipass[:, 0:C, :, :]
    z_train = torch.zeros((X_train.shape[0], N_LATENTS))
    for i, sample in enumerate(X_train):
        sample = sample.to(DEVICE)
        sample = sample.view(1, *sample.shape)
        z_train[i, :] = cae.conv_encoder(sample).cpu()

    for nu in tqdm(np.arange(1, NU_MAX+1)):
        utils.log(f"Training DP with NU={nu}", START_TIME)
        
        # Select input and target from multipass and apply weights
        y_train = X_multipass[:, nu*C:(nu+1)*C, :, :]
        z_train_next = torch.zeros((y_train.shape[0], N_LATENTS))
        for i, sample in enumerate(y_train):
            sample = sample.to(DEVICE)
            sample = sample.view(1, *sample.shape)
            z_train_next[i, :] = cae.conv_encoder(sample).cpu()

        # Fit damped persistence model
        dp = linear_models.DampedPersistenceModel()
        dp.fit(X=z_train, y=z_train_next, nu=nu)

        # Save damped persistence model
        utils.log(f"Saving DP to {PATHS[subproject_name]}", START_TIME)
        path = os.path.join(PATHS[subproject_name], 'dp')
        os.makedirs(path, exist_ok=True)
        filename = f'dp_cae_ssh_d{N_LATENTS}_nu{nu}.joblib'
        joblib.dump(dp, os.path.join(path, filename))

    utils.log("PROCESS_COMPLETED", START_TIME)

    return 0

if __name__ == "__main__":
    main()

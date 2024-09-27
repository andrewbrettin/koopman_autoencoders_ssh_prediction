"""
pca.py

Fits PCA model to data from the CNN experiments
"""

import os
import sys
from datetime import datetime
import joblib

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA

import torch

import src
from src.attrs import PATHS, GLOBALS
from src import utils

from src.data import loading

# Globals
REGION = sys.argv[1]    # 'pacific', 'north_atlantic'
SAMPLING = sys.argv[2]  # 'daily', 'monthly'
LATENT_SIZES = [30, 40, 50]
K = 10 if SAMPLING == 'monthly' else 20

# Runtime variables
START_TIME = datetime.now()

def main():
    utils.log("Begin script")
    utils.print_os_environ()
    subproject_name = f'cnn_{REGION}_{SAMPLING}'
    print(subproject_name)

    # Load data
    utils.log(f"Load data for PCA", START_TIME)
    
    X_train = torch.load(
        os.path.join(PATHS[subproject_name], 'multipass', f'train_ssh_k{K}.pt')
    )[:, 0:1, :, :]
    weights = torch.load(os.path.join(PATHS[subproject_name], 'weights.pt'))

    # Apply area weighting
    X_train_weighted = X_train * weights

    # Reshape 
    X_train_weighted = X_train_weighted.view(X_train.shape[0], -1)

    # Main loop
    for i, n_latents in enumerate(LATENT_SIZES):
        # Fit PCA model
        utils.log(f"Fitting PCA model r={n_latents}", START_TIME)
        pca_model = PCA(n_components=n_latents, svd_solver='randomized', random_state=0)
        pca_model.fit(X_train_weighted)
        
        # Save PCA model
        utils.log("Save model", START_TIME)
        os.makedirs(os.path.join(PATHS[subproject_name], 'pca'), exist_ok=True)
        model_name = os.path.join(PATHS[subproject_name], 'pca', f"pca_ssh_{n_latents}.joblib")
        joblib.dump(pca_model, model_name)

    utils.log("PROCESS COMPLETED", START_TIME)    
    return 0

if __name__ == "__main__":
    main()

"""
save_standardizer.py

Creates the standardizer for the tensors.
"""

import os
import sys
from datetime import datetime
from itertools import product
import warnings

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

import src
from src import utils
from src.data import loading
from src.attrs import PATHS, GLOBALS

# Ignorance is bliss
warnings.filterwarnings('ignore')

# Full suite
VARIABLES = ['SSH', 'SST', 'MLD', 'PSL', 'UBOT', 'VBOT']
START_TIME = datetime.now()

def get_train_members():
    all_members = list(product(GLOBALS['init_years'], GLOBALS['members']))
    train_members = [all_members[i] for i in GLOBALS['train_ids']]
    return train_members
    

def main():
    utils.log("Begin script")

    # Load mask
    utils.log("Loading mask", START_TIME)
    mask = loading.load_north_pacific_mask()
    
    # Get training members
    utils.log("Getting training dataset", START_TIME)
    X_train_data = []
    train_members = get_train_members()
    
    for i, (init_year, member) in enumerate(train_members):
        # Load ensemble_member dataset using inner join
        X_list = [loading.load_anomalies(var, init_year, member) for var in VARIABLES]
        X = xr.merge(X_list, join='inner')
        X = X.to_array(dim='variable')
        
        # Mask
        X = X.sel(**mask.coords)
        X = xr.where(mask, X, np.nan)
    
        # Variable names
        train_id = GLOBALS['train_ids'][i]
        X.name = f"X_{train_id}"
        X.attrs['init_year'] = init_year
        X.attrs['member'] = member
    
        X_train_data.append(X)
    
    X_train = xr.merge(X_train_data)
    X_train = X_train.to_array(dim='member_id')
    X_train['member_id'] = GLOBALS['train_ids']
    X_train = X_train.stack(sample=('member_id', 'time'))

    # Print information about X_train
    print('X_train:')
    print(X_train)

    # Get standardizer
    X_standardizer = xr.Dataset({
        'mean': X_train.mean(dim=('sample', 'lat', 'lon')),
        'std': X_train.std(dim=('sample', 'lat', 'lon'))
    })
    X_standardizer.attrs['description'] = (
        "Standardizer from training data. Standardization is done by variable"
        "(across times/ensemble members, lat, and lon). Values are masked "
        "according to `grid/north_pacific_mask.nc`."
    )
    X_standardizer.attrs['history'] = (
        f"Created on {datetime.now()} using {__file__}"
    )

    # Save standardizer
    utils.log("Saving standardizer", START_TIME)
    filename = os.path.join(PATHS['tensors'], 'X_standardizer.nc')

    with ProgressBar():
        X_standardizer.to_netcdf(filename, compute=True)

    utils.log("PROCESS COMPLETED", START_TIME)

    return 0

if __name__ == "__main__":
    main()
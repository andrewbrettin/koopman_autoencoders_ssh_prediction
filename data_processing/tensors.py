"""
tensors.py
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
SYS_IN = sys.argv[1]
START_TIME = datetime.now()

def get_feature_multiindex():
    """
    Gets feature coordinates for selecting stacked data computationally efficiently
    """
    mask = loading.load_north_pacific_mask()
    X_list = []
    for var in VARIABLES:
        X_list.append(
            loading
            .load_anomalies(var, 1251, '011')
            .isel(time=0)
            .drop_vars('time')
        )
    X = xr.merge(X_list, join='inner')
    X = X.to_array(dim='variable')
    X = xr.where(mask, X.sel(**mask.coords), np.nan)
    feature_multiindex = X.stack(feature=('variable', 'lat', 'lon')).dropna(dim='feature')['feature']
    return feature_multiindex

def main():
    utils.log("Begin script")

    # Load mask and standardizer
    utils.log("Loading mask and standardizer", START_TIME)
    mask = loading.load_north_pacific_mask()
    X_standardizer = loading.load_standardizer()

    # Mask nan points
    feature_multiindex = get_feature_multiindex()

    # Main loop
    utils.log("Main loop", START_TIME)
    all_members = list(product(GLOBALS['init_years'], GLOBALS['members']))

    for i, (init_year, member) in enumerate(all_members):
        utils.log(f"Init year {init_year}, member {member}", START_TIME)
        
        # Load ensemble_member dataset using inner join
        X_list = [loading.load_anomalies(var, init_year, member) for var in VARIABLES]
        X = xr.merge(X_list, join='inner')
        X = X.to_array(dim='variable')
        
        # Mask
        X = X.sel(**mask.coords)
        X = xr.where(mask, X, np.nan)
        
        # Standardization
        X = (X - X_standardizer['mean']) / X_standardizer['std']
        
        # Flatten array and drop nans
        X = (
            X
            .stack(feature=('variable', 'lat', 'lon'))
            .sel(feature=feature_multiindex)
        )

        # Set meta
        X.attrs['id'] = i
        X.attrs['init_year'] = init_year
        X.attrs['member'] = member
        X.attrs['description'] = (
            f"Standardized and flattened data tensor X_{i}."
            "Contains coordinates and values in double-precision float (float64)."
        )
        X.attrs['history'] = f"Created on {datetime.now()} using {__file__}"

        # Save tensor
        X = X.reset_index('feature')
        X['feature'] = np.arange(X.shape[1])
        X = X.drop_vars(['variable', 'lat', 'lon'])
        
        with ProgressBar():
            X.to_netcdf(os.path.join(PATHS['tensors'], f'X_{i}.nc'))

    # Save coordinate values as well
    utils.log("Save coordinate values", START_TIME)
    feature_coords_df = pd.DataFrame(
        {
            'variable': feature_multiindex['variable'].values,
            'lat': feature_multiindex['lat'].values,
            'lon': feature_multiindex['lon'].values
        },
    )
    feature_coords_df.name = 'feature'
    df_filename = os.path.join(PATHS['tensors'], 'feature_coords.h5')
    feature_coords_df.to_hdf(df_filename, key='df')

    # Also save time coordinates
    X['time'].to_netcdf(os.path.join(PATHS['tensors'], 'time.nc'))

    utils.log("PROCESS COMPLETED", START_TIME)

    return 0





def test():
    utils.log("Begin script in test configuration")

    # Load mask and standardizer
    utils.log("Loading mask and standardizer", START_TIME)
    mask = loading.load_north_pacific_mask()
    X_standardizer = loading.load_standardizer()

    # Mask nan points
    feature_multiindex = get_feature_multiindex()

    # Main loop
    utils.log("Main loop", START_TIME)
    all_members = list(product(GLOBALS['init_years'], GLOBALS['members']))

    i = 0
    init_year = 1251
    member = '011'
    utils.log(f"Init year {init_year}, member {member}", START_TIME)
    
    # Load ensemble_member dataset using inner join
    X_list = [loading.load_anomalies(var, init_year, member) for var in VARIABLES]
    X = xr.merge(X_list, join='inner')
    X = X.to_array(dim='variable')
    
    # Mask
    X = X.sel(**mask.coords)
    X = xr.where(mask, X, np.nan)
    
    # Standardization
    X = (X - X_standardizer['mean']) / X_standardizer['std']
    
    # Flatten array
    X = (
        X
        .stack(feature=('variable', 'lat', 'lon'))
        .sel(feature=feature_multiindex)
    )

    # Set meta
    X.attrs['id'] = i
    X.attrs['init_year'] = init_year
    X.attrs['member'] = member
    X.attrs['description'] = (
        f"Standardized and flattened data tensor X_{i}."
        "Contains coordinates and values in double-precision float (float64)."
    )
    X.attrs['history'] = f"Created on {datetime.now()} using {__file__}"

    # Save tensor
    X = X.reset_index('feature')
    X['feature'] = np.arange(X.shape[1])
    X = X.drop_vars(['variable', 'lat', 'lon'])
    
    with ProgressBar():
        X.to_netcdf(os.path.join(PATHS['tensors'], f'X_{i}.nc'))

    
    # Save coordinate values as well
    utils.log("Save coordinate values", START_TIME)
    feature_coords_df = pd.DataFrame(
        {
            'variable': feature_multiindex['variable'].values,
            'lat': feature_multiindex['lat'].values,
            'lon': feature_multiindex['lon'].values
        },
    )
    feature_coords_df.name = 'feature'
    df_filename = os.path.join(PATHS['tmp'], 'feature_coords.h5')
    feature_coords_df.to_hdf(df_filename, key='df')

    # Also save time coordinates
    X['time'].to_netcdf(os.path.join(PATHS['tensors'], 'time.nc'))

    utils.log("TEST PROCESS COMPLETED", START_TIME)

    return 0

if __name__ == "__main__":
    if SYS_IN == 'test':
        test()
    else:
        main()

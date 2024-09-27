import os
import sys
import json
from itertools import product
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
import dask
from dask.diagnostics import ProgressBar
from dask_jobqueue import PBSCluster

import src
from src import utils
from src.data import loading
from src.attrs import PATHS, GLOBALS

# Full suite
VARIABLES = ['SSH', 'zos', 'SST', 'MLD', 'PSL', 'UBOT', 'VBOT']
START_TIME = datetime.now()

def detrend(arr, dim='time', deg=5):
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        p = arr.polyfit(dim=dim, deg=deg)
    p['polyfit_coefficients'] = p['polyfit_coefficients'].chunk(
        {'lat': 48, 'lon': 48}
    )
    trend = xr.polyval(arr[dim], p['polyfit_coefficients'])
    detrended = arr - trend
    
    detrended.name = arr.name
    detrended.attrs['units'] = arr.attrs['units']
    detrended.attrs['description'] = (
        f"Detrended {arr.name} using a degree {deg} polynomial."
    )
    detrended.attrs['history'] = (
        f"Created on {datetime.now()} using {__file__}"
    )
    return detrended

def deseasonalize(arr):
    if len(arr['time']) == 0:
        return arr
    gb = arr.groupby('time.dayofyear')
    clim = gb.mean(dim='time')
    anom = (gb - clim).drop_vars('dayofyear')
    return anom

def test():
    utils.log("Beginning test script")
    init_year = 1251
    member = '011'
    var = 'PSL'
    
    utils.log(f"LE-{init_year}.{member}.{var}", START_TIME)
    
    # Load data
    arr = loading.load_dataset(var, init_year, member, chunkedby='time')
    
    # Detrend
    utils.log("Detrending data", START_TIME)
    detrended = detrend(arr)
    
    # Deseasonalize
    utils.log("Deseasonalizing data", START_TIME)
    anom = xr.map_blocks(deseasonalize, detrended, template=detrended)
    anom.name = arr.name
    anom.attrs['units'] = arr.attrs['units']
    anom.attrs['description'] = (
        f"Deseasonalized {arr.name} using daily mean climatology."
    )
    anom.attrs['history'] = (
        f"Created on {datetime.now()} using {__file__}"
    )
    
    # Create dataset and save
    utils.log("Saving data", START_TIME)
    
    ds = xr.Dataset({var: anom})
    path = os.path.join(
        PATHS['tmp'],
        f'LE2-{init_year}.{member}.{var}_anom.zarr'
    )
    with ProgressBar():
        ds.to_zarr(path)

    utils.log("Data saved", START_TIME)   
    
    utils.log("PROCESS COMPLETED", START_TIME)
    return 0

def main():
    utils.log("Beginning script")
    for var, init_year, member in product(
            VARIABLES, GLOBALS['init_years'], GLOBALS['members']):
        utils.log(f"LE-{init_year}.{member}.{var}", START_TIME)

        # Load data
        arr = loading.load_dataset(var, init_year, member, chunkedby='time')

        # Detrend
        utils.log("Detrending data", START_TIME)
        detrended = detrend(arr)

        # Deseasonalize
        utils.log("Deseasonalizing data", START_TIME)
        anom = xr.map_blocks(deseasonalize, detrended, template=detrended)
        anom.name = arr.name
        anom.attrs['units'] = arr.attrs['units']
        anom.attrs['description'] = (
            f"Deseasonalized {arr.name} using daily mean climatology."
        )
        anom.attrs['history'] = (
            f"Created on {datetime.now()} using {__file__}"
        )

        # Create dataset and save
        utils.log("Saving data", START_TIME)

        ds = xr.Dataset({var: anom})
        path = os.path.join(
            PATHS['detrended_deseasonalized'],
            f'LE2-{init_year}.{member}.{var}_anom.zarr'
        )
        with ProgressBar():
            ds.to_zarr(path)
        
        # Rinse and repeat
        utils.log("Data saved", START_TIME)
        del arr, detrended, anom, ds, path
        
        
    utils.log("PROCESS COMPLETED", START_TIME)
    return 0

if __name__ == "__main__":
    # test()
    main()


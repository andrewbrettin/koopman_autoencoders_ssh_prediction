"""
resample_monthly.py

"""

import os
import sys
import shutil
import warnings
import json
from datetime import datetime
from itertools import product
import numpy as np
import xarray as xr
from rechunker import rechunk
import dask
from dask.diagnostics import ProgressBar

import src
from src import utils
from src.data import loading
from src.attrs import PATHS, GLOBALS

VARIABLES = ['SSH', 'zos', 'SST', 'MLD', 'PSL', 'UBOT', 'VBOT']

START_TIME = datetime.now()

def main():
    utils
    utils.log("Beginning resampling")
    utils.print_os_environ()

    for (var, init_year, member) in product(
            VARIABLES, GLOBALS['init_years'], GLOBALS['members']):
        utils.log(f"LE-{init_year}.{member}.{var}", START_TIME)

        # Load data
        arr = loading.load_anomalies(var, init_year, member, chunkedby='space')

        # Apply resampling operation
        resampled = arr.resample(time='M').mean()

        # Attributes
        resampled.attrs['description'] = (
            "Monthly-meaned {arr.name} deseasonalized anomalies."
        )
        resampled.attrs['history'] = (
            f"Created on {datetime.now()} using {__file__}"
        )

        # Dataset
        ds = xr.Dataset({var: resampled})
        
        # Save data 
        path = os.path.join(
            PATHS['scratch'],
            'monthly_anom',
            f'LE2-{init_year}.{member}.{var}_anom.zarr'
        )

        with ProgressBar():
            ds.to_zarr(path)
        utils.log("Data saved", START_TIME)
        del arr, resampled, ds
        
    utils.log("PROCESS COMPLETED", START_TIME)

    return 0

if __name__ == "__main__":
    main()

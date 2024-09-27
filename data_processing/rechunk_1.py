__author__ = "@andrewbrettin"
__date__ = "02-11-24"

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

# VARIABLES = ['SSH', 'zos', 'SST', 'MLD', 'PSL', 'UBOT', 'VBOT']
VARIABLES = ['PSL', 'UBOT', 'VBOT']

START_TIME = datetime.now()

def rm_stores(*stores):
    for store in stores:
        if os.path.exists(store):
            shutil.rmtree(store)

def execute_rechunk(ds, target_store, temp_store):
    chunks_dict = {
        'time': -1,
        'lat': 48,
        'lon': 48
    }
    max_mem='8GB'
    
    array_plan = rechunk(
        ds, chunks_dict, max_mem, target_store, temp_store=temp_store
    )
    
    array_plan.execute()

def test():
    utils.log("Beginning test script")
    init_year = 1251
    member = '011'
    var = 'PSL'

    utils.log(f"LE-{init_year}.{member}.{var}", START_TIME)
        
    # Load data as a dataset
    array = loading.load_dataset(
        var, init_year, member, chunkedby='space')
    ds = xr.Dataset({var: array})
    ds = ds.chunk({'time': 3650})
    
    # Prepare paths for rechunking
    utils.log("Preparing zarr stores", START_TIME)
    target_store = os.path.join(
        PATHS['rechunked'],
        f'LE2-{init_year}.{member}.{var}_rechunked.zarr'
    )
    temp_store = os.path.join(PATHS['tmp'],'temp.zarr')
    rm_stores(target_store, temp_store)
    
    # Rechunk
    utils.log("Rechunking", START_TIME)
    with ProgressBar():
        execute_rechunk(ds, target_store, temp_store)
    
    # Repeat
    utils.log(f"Completed rechunk for LE-{init_year}.{member}.{var}", START_TIME)
    del array
    del ds
    del target_store
    del temp_store

def main():
    utils.log("Beginning script")
    
    for var, init_year, member in product(
            VARIABLES, GLOBALS['init_years'], GLOBALS['members']):
        utils.log(f"LE-{init_year}.{member}.{var}", START_TIME)
        
        # Load data as a dataset
        array = loading.load_dataset(
            var, init_year, member, chunkedby='space')
        ds = xr.Dataset({var: array})
        ds = ds.chunk({'time': 3650})
        
        # Prepare paths for rechunking
        utils.log("Preparing zarr stores", START_TIME)
        target_store = os.path.join(
            PATHS['rechunked'],
            f'LE2-{init_year}.{member}.{var}_rechunked.zarr'
        )
        temp_store = os.path.join(PATHS['tmp'],'temp.zarr')
        rm_stores(target_store, temp_store)
        
        # Rechunk
        utils.log("Rechunking", START_TIME)
        with ProgressBar():
            execute_rechunk(ds, target_store, temp_store)
        
        # Repeat
        utils.log(f"Completed rechunk for LE-{init_year}.{member}.{var}", START_TIME)
        del array
        del ds
        del target_store
        del temp_store
    
    utils.log("PROCESS_COMPLETED", START_TIME)
    
    return 0

if __name__ == "__main__":
    test()

"""
regrid.py

This script regrids the CESM2 ocean datasets of interest.

We use the regridder file $TMPDIR/grid/bilinear_384x320_192x288_peri.nc
which was created using make_gridfile.py.
"""

__author__ = "@andrewbrettin"
__all__ = [
    "load_regridder_weights",
    "load_initial_geocoords",
    "load_target_geocoords",
    "preprocess_input_dataset",
    "load_ocean_dataset",
    "simplified_name",
    "save_regridded_ds"
]

import os
import sys
import json
from datetime import datetime
from itertools import product
from tqdm import tqdm
import numpy as np
import xarray as xr
import xesmf as xe
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import dask

from src.attrs import PATHS, GLOBALS
from src import utils
    
#$ Globals
VARIABLES = ['SSH_2', 'SST', 'HMXL_DR_2']
VARIABLES = ['SST', 'HMXL_DR_2']

#$ Functions
def load_regridder_weights():
    """
    Loads the regridder weights, as computed in 
    data_processing/regridding.ipynb.
    """
    weights = xr.open_dataset(
        os.path.join(PATHS['grid'], 'bilinear_384x320_192x288_peri.nc')
    )
    return weights

def load_initial_geocoords():
    """
    Because not all of the ensemble members contain the geographical
    coordinates, this function loads the initial 'TLONG' and 'TLAT'
    2D arrays from one of the ensemble members which does. These
    coordinates can then be added to degenerate datasets using the
    `assign_coords` function.
    
    Returns:
        tlat: xr.DataArray
            2D DataArray of geographical latitudes for
            displaced-dipole ocean grid.
        tlon: xr.DataArray
            2D DataArray of geographical longitudes for
            displaced-dipole ocean grid.
    """
    ocn_ds_ex = xr.open_dataset(
        os.path.join(
            PATHS['ocn_daily'],
            'SSH_2',
            f'b.e21.BSSP370smbb.f09_g17.LE2-1301.020.pop.h.nday1.SSH_2.'
            '20950102-21001231.nc'
        )
    )
    tlat = ocn_ds_ex['TLAT']
    tlong = ocn_ds_ex['TLONG']
    return tlat, tlong

def load_target_geocoords():
    """
    This script loads the target coordinates used for the output dataset.
    """
    atmos_ds_ex = xr.open_dataset(
        os.path.join(
            PATHS['atm_daily'],
            'PSL',
            (f'b.e21.BSSP370smbb.f09_g17.LE2-1301.020.cam.h1.PSL.'
            '20950101-21001231.nc')
        )
    )
    ds_out = xr.Dataset({'lat': atmos_ds_ex.lat, 'lon': atmos_ds_ex.lon})
    return ds_out

def preprocess_input_dataset(ds_in):
    """
    If geographical coordinates are not in the dataset, adds them.
    Also, renames the dimensions TLONG and TLAT to 'lon' and 'lat',
    as required by xESMF.
    
    Parameters:
        ds_in: xr.DataArray
            DataArray to process.
    Returns:
        ds_in_processed: xr.DataArray
            Same dataset with geographical coordinates 'lat' and 'lon'.
    """
    if 'TLONG' not in ds_in.coords or 'TLAT' not in ds_in.coords:
        tlat, tlong = load_initial_geocoords()
        ds_in.coords['TLAT'] = tlat
        ds_in.coords['TLONG'] = tlong
    
    #$ Rename coordinates
    ds_processed = ds_in.rename({'TLAT': 'lat', 'TLONG': 'lon'})
    
    return ds_processed

def load_ocean_dataset(variable, member, init_year):
    """
    Loads the original ocean datasets on the gx1v7 displaced-dipole grid.
    
    Parameters:
        variable: str
            Variable name to load (SSH_2 or SST only right now)
        member: str
            Ensemble member number to use
        init_year: int or string
            Initialization year for ensemble member.
    
    Returns:
        ds : xr.Dataset
            Ocean dataset on the gx1v7 displaced dipole grid.
    """
        
    ds = xr.open_mfdataset(
        os.path.join(
            PATHS['ocn_daily'],
            variable,
            f'b.e21*.f09_g17.LE2-{init_year}.{member}.pop.h.nday1.{variable}.*.nc'
        ),
        combine='by_coords'
    )
    
    #$ Return a subsetted dataset with just the one variable
    ds_copy = xr.Dataset({variable: ds[variable]})
    ds_copy.attrs = ds.attrs
    
    return ds_copy

def simplified_name(var):
    """
    Returns simplified variable names which are less confusing than
    defaults given by CESM2.
    
    Renaming of the dynamic sea level variable SSH_2 is handled in
    compute_sea_level_vars.ipynb.
    """
    if var == "SHF_2":
        var_simple = "SHF"
    elif var == "HMXL_DR_2":
        var_simple = "MLD"
    else:
        var_simple = var
    return var_simple

def save_regridded_ds(ds_regridded, var, member, iy, 
                      path=PATHS['regridded']):
    """
    Saves regridded dataset. Output filenames are saved as 
    `LE2-{init year}.{member}.{var}.nc`; for example:
    /glade/scratch/abrettin/regridded/LE2-1301.020.SSH_2.nc.
    
    Parameters:
        ds_regridded: xr.Dataset
            Regridded dataset to save.
        var: str
            Variable name.
        member: str
            Ensemble member number.
        iy: int or str
            Initialization year.
        path:
            Output path for saving the regridded dataset.
    """
    filename = f'LE2-{iy}.{member}.{var}.nc'
    ds_regridded.to_netcdf(os.path.join(path, filename))
    
    return 0

def main():
    #$ Initialization
    START_TIME = datetime.now()
    utils.log(f"Beginning script {__file__}")
    
    #$ Load weights and output dataset
    weights = load_regridder_weights()
    ds_out = load_target_geocoords()
    
    #$ Main loop
    for var, member, iy in product(
            VARIABLES, GLOBALS['members'], GLOBALS['init_years']):
        utils.log(f"Regridding {var} {member}.{iy}", START_TIME)
        
        
        #$ Load and preprocess input dataset
        ds_in = load_ocean_dataset(var, member, iy)
        ds_in = preprocess_input_dataset(ds_in)
        
        #$ Load regridder
        regridder = xe.Regridder(
            ds_in, ds_out, 'bilinear', periodic=True, weights=weights
        )
        
        #$ Regrid
        ds_regridded = regridder(ds_in)
        
        #$ Match metadata and rename to less stupid names
        ds_regridded[var].attrs['long_name'] = ds_in[var].attrs['long_name']
        ds_regridded[var].attrs['units'] = ds_in[var].attrs['units']
        
        arr = ds_regridded[var]
        varname_simple = simplified_name(var)
        arr.name = varname_simple
        ds_regridded = xr.Dataset({varname_simple: arr})
        
        #$ Save
        save_regridded_ds(ds_regridded, varname_simple, member, iy)
        
    
    #$ End
    utils.log("PROCESS COMPLETED\n", START_TIME)
    return 0

if __name__ == "__main__":
    main()
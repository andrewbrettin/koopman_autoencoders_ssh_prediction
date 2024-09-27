"""
make_gridfile.py

Rough script to make the regridder file and compute the area cells.
"""

__author__ = "@andrewbrettin"
__date__ = "01-19-2023"


import os
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
import xarray as xr
import xesmf as xe
import matplotlib.pyplot as plt
import dask
from dask.diagnostics import ProgressBar

from src import utils
from src.attrs import PATHS

# Globals
INIT_YEAR = 1301
MEMBER = '013'
START_RUNTIME = datetime.now()

def load_ssh():
    ocn_ds = xr.open_mfdataset(
        os.path.join(
            PATHS['ocn_daily'],
            'SSH_2',
            f'b.e21*.f09_g17.LE2-{INIT_YEAR}.{MEMBER}.pop.h.nday1.SSH_2.*.nc'
        ),
        combine='by_coords'
    )
    ssh = ocn_ds['SSH_2']
    return ssh

def load_slp():
    atm_ds = xr.open_mfdataset(
        os.path.join(
            PATHS['atm_daily'],
            'PSL',
            f'b.e21*.f09_g17.LE2-{INIT_YEAR}.{MEMBER}.cam.h1.PSL.*.nc'
        ),
        combine='by_coords'
    )
    slp = atm_ds['PSL']
    return slp

def load_regridded_ssh():
    ds = xr.open_dataset(
        os.path.join(
            PATHS['regridded'],
            f'LE-{INIT_YEAR}.{MEMBER}.SSH_2.nc'
        ),
        chunks={'time': 365}
    )
    ssh = ds['SSH_2']
    return ssh

def make_regridder(ssh, slp, path, fname='bilinear_384x320_192x288_peri.nc'):
    # Input grid
    ds = xr.Dataset({'SSH': ssh})
    ds = ds.rename({'TLONG': 'lon', 'TLAT': 'lat'})
    
    # Output grid
    ds_out = xr.Dataset({'lat': slp.lat, 'lon': slp.lon})
    
    filename = os.path.join(path, fname)
    
    regridder = xe.Regridder(
        ds, ds_out, 'bilinear', periodic=True, filename=filename
    )
    return regridder

def create_areas(slp):
    """Uses the slp lat-lon coordinates to compute areas"""
    lat = slp.coords['lat']
    lon = slp.coords['lon']
    dlats = lat[1:].values - lat[:-1].values
    dlons = lon[1:].values - lon[:-1].values
    dlat = dlats.mean()
    dlon = dlons.mean()
    
    R_e = 6378.1
    dx = 2 * np.pi * R_e * np.cos(lat * np.pi/180) * dlon / 360
    dy = 2 * np.pi * R_e * dlat / 360 * xr.ones_like(lon)
    
    areas = dx * dy
    areas.name = 'area'
    areas.attrs['units'] = 'km^2'
    areas.attrs['long_name'] = 'grid_cell_areas'
    areas.attrs['history'] = (
        f"Created on {datetime.now()} using {__file__}"
    )
    return areas

def create_land_mask(ssh_regridded):
    ssh_0 = ssh_regridded.isel(time=0)
    mask = xr.where((ssh_0 == 0) + np.isnan(ssh_0), 0, 1)
    mask.name = 'land_mask'
    mask = mask.drop('time')
    mask.load()
    return mask

def main():
    # Make regridder file
    utils.log("Begin script")

    ssh = load_ssh()
    slp = load_slp()
    # ssh_regridded = load_regridded_ssh()
    make_regridder(ssh, slp, path=PATHS['grid'])

    # Make area file and save
    areas = create_areas(slp)
    areas.to_netcdf(os.path.join(PATHS['grid'], 'areas.nc'))

    utils.log("PROCESS COMPLETED", START_RUNTIME)
    return 0


if __name__ == "__main__":
    main()
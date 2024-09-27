"""
loading.py

Functions for loading data.
"""
__author__ = "@andrewbrettin"

__all__ = [
    "load_areas",
    "load_land_mask",
    "load_north_pacific_mask",
    "load_north_atlantic_mask",
    "load_dataset",
    "load_anomalies",
    "load_standardizer",
    "load_feature_coords",
    "load_features",
    "load_autoencoder_tensors",
    "load_multipass_tensors"
]

from typing import Union
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import dask

import torch

from ..attrs import GLOBALS, PATHS
from ..tools import processing


def compute_areas(lats, lons):
    """
    Creates area files based on given latitudes and longitudes.
    
    Parameters:
        lats: xr.DataArray
            Target latitudes
        lons: xr.DataArray
            Target longitudes
            
    Returns:
        areas: xr.DataArray
            Grid cell areas.
    """
    #$ Latitude and longitude spacing, in degrees
    dlats = lats[1:].values - lats[:-1].values
    dlons = lons[1:].values - lons[:-1].values
    dlat = dlats.mean()
    dlons = dlons.mean()
    
    R_e = 6378.1
    dx = np.pi * R_e * np.cos(lats * np.pi/180) * dlons / 360
    dy = 2 * np.pi * R_e * dlat / 360 * xr.ones_like(lons)
    areas = dx * dy

    return areas

def load_areas():
    """Area file for CESM2 grid"""
    areas = xr.open_dataset(os.path.join(PATHS['grid'], 'areas.nc'))['area']
    return areas

def load_land_mask():
    mask = xr.open_dataset(os.path.join(PATHS['grid'], 'mask.nc'))['land_mask']
    return mask

def load_north_pacific_mask():
    mask = xr.open_dataarray(os.path.join(PATHS['grid'], 'north_pacific_mask.nc'))
    return mask

def load_north_atlantic_mask():
    mask = xr.open_dataarray(os.path.join(PATHS['grid'], 'north_atlantic_mask.nc'))
    return mask

def load_tropics_mask():
    mask = xr.open_dataarray(os.path.join(PATHS['grid'], 'tropics_mask.nc'))
    return mask

def load_dataset(var, init_year, member, full_ds=False, chunkedby='space'):
    """
    Loads a given dataarray on the lat-lon grid.
    For ocean variables, the land points are masked by np.nan.
    For atmospheric variables, there for some reason is a one-day
    overlap on 01-01-2015 where fields for both the BHIST and SSP370
    forcing scenarios occur. Therefore, we need to properly select
    dates 01-01-1850 through 12-31-2014 for the historical emissions
    and dates 01-01-2015 through 12-31-2100 for SSP370.
    
    Parameters:
        var: str
            Variable to load. Should be one of the following: 'SSH', 'zos', 
            'SSH_2', 'SST', 'PSL', 'UBOT', 'VBOT'.
        member: str
            Ensemble member.
        init_year: int or str
            Initialization year.
        full_ds: bool
            If full_ds is True, then returns an xr.Dataset object.
            Otherwise, it returns the xr.DataArray object.
        chunkedby: str, 'space' or 'time'
            Indicates whether to load datasets chunked by space or by time.
    Returns
        arr: xr.Dataset or xr.DataArray
            DataArray object dataset on the lat-lon grid. If var is
            an ocean variable, land points are masked with np.nan.
    """

    if chunkedby == 'time':
        path = os.path.join(
            PATHS['rechunked'], 
            f'LE2-{init_year}.{member}.{var}_rechunked.zarr'
        )
        ds = xr.open_zarr(path, consolidated=False)
        arr = ds[var]
    elif chunkedby == 'space':
        if var in ['PSL', 'UBOT', 'VBOT']:
            path = os.path.join(
                PATHS['atm_daily'], var,
                f'b.e21*.f09_g17.LE2-{init_year}.{member}.cam.h1.{var}.*.nc'
            )
            ds = xr.open_mfdataset(path, combine='by_coords')
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                ds = ds.drop_duplicates(dim='time', keep='last')
            ds = ds.chunk({'time': 3650})
            arr = ds[var]

        elif var in ['SSH_2', 'SST', 'MLD']:
            path = os.path.join(
                PATHS['regridded'], f'LE2-{init_year}.{member}.{var}.nc'
            )
            ds = xr.open_dataset(path, chunks={'time':3650})
            mask = load_land_mask()
            arr = xr.where(mask, ds[var], np.nan)
            arr.attrs = ds[var].attrs
            ds[var] = arr

        elif var in ['zos', 'SSH']:
            path = os.path.join(
                PATHS['sea_level'], f'LE2-{init_year}.{member}.{var}.zarr'
            )
            ds = xr.open_zarr(path)
            mask = load_land_mask()
            arr = xr.where(mask, ds[var], np.nan)
            arr.attrs = ds[var].attrs
            ds[var] = arr

        else:
            raise ValueError(
                f"Incorrect variable {var}. Should be one of: "
                "['SSH', 'zos', 'SSH_2', 'SST', 'PSL', 'UBOT', 'VBOT', MLD]"
            )
    else:
        raise ValueError(
            "Parameter 'chunkedby' must take values 'space' or 'time'"
        )
        
    if full_ds:
        return ds
    else:
        return arr
    
def load_anomalies(var, init_year, member, full_ds=False, chunkedby='space'):
    if chunkedby == 'time':
        path = os.path.join(
            PATHS['detrended_deseasonalized'], 
            f'LE2-{init_year}.{member}.{var}_anom.zarr'
        )
        ds = xr.open_zarr(path, consolidated=False)
    elif chunkedby == 'space':
        path = os.path.join(
            PATHS['anom_spatial'], 
            f'LE2-{init_year}.{member}.{var}_anom.zarr'
        )
        ds = xr.open_zarr(path, consolidated=False)

    if full_ds:
        return ds
    else:
        return ds[var]

def load_standardizer(path=PATHS['tensors']):
    """
    Loads standardizer for tensors.
    """
    ds = xr.open_dataset(os.path.join(path, 'X_standardizer.nc'))
    return ds


def load_feature_coords(data_path: str = PATHS['tensors']):
    """
    Loads the feature coordinate from HDF5 format.

    Returns:
        feature_coords: xr.DataArray
            DataArray of multiindex dimensions ('varible', 'lat', 'lon').
    """
    filename = os.path.join(data_path, 'feature_coords.h5')
    feature_df = pd.read_hdf(os.path.join(data_path, 'feature_coords.h5'), key='df')
    feature_coords = pd.MultiIndex.from_frame(feature_df)
    feature_coords.name = 'feature'
    feature_coords = xr.DataArray(feature_coords)
    return feature_coords

def load_time_coord(
    data_path: str = PATHS['tensors'],
    lag: timedelta = None,
    subsample: int = 10
):
    """
    Loads time coordinate with values shifted by time lag and subsampled.
    Useful for adding coordinate data for tensors loaded using 
    `load_autoencoder_tensors`.

    Parameters:
        data_path: str
            Path to time coordinates.
        lag: datetime.timedelta
            
    """
    times = xr.open_dataarray(os.path.join(data_path, 'time.nc'))
    times = processing.shift_times(times, lag)
    times = times.isel(time=slice(0, None, subsample))
    return times

def load_features(
    member_id: int,
    data_path: str = PATHS['tensors'],
    unstacked: bool=False
):
    """
    Loads xarray features with the correct coordinate info.

    Parameters:
        member_id: int
            Integer from 0-8 indicating the ensemble member (see attrs.GLOBALS)
        data_path: str
            Path to feature data.
        unstacked: bool
            If True, unstacks the feature coordinate into separate dimensions
            'variable', 'lat', and 'lon'. Note that this is computationally 
            memory intensive

    Returns:
        X: xr.DataArray
            Feature data.
    """
    filename = os.path.join(data_path, f'X_{member_id}.nc')

    X = xr.open_dataarray(filename)
    feature_coords = load_feature_coords(data_path)
    X['feature'] = feature_coords

    if unstacked:
        X = X.unstack('feature')

    return X

def load_autoencoder_tensors(
    datatype: str = 'train',
    lag: Union[int, timedelta, None] = None,
    truncation: Union[int, timedelta, None] = None,
    subsample: int = 10,
    data_path: str = PATHS['tensors']
):
    """
    Loads training tensors. Specifically, does the following:
    * Loads dataarrays for specific member using load_features()
    * Shifts values by time lag, if given
    * Truncates the values by time lag, if given
    * Subsamples the data
    * Converts to single-precision pytorch tensor
    For 'test' or 'val' datatypes, it returns the pytorch tensor;
    for 'train' datatype, it concatenates the data along the time dimension and
    returns it as a single tensor.

    Params:
        datatype: 'train', 'val', or 'test'
            Datatype of loaded tensors.
        data_path: str
            Path to netcdf files.
        lag: int or timedelta
            Time lag for loaded tensors.
        truncation: int or timedelta
            Number of samples to truncate from the end.
        subsample: int
            Level of subsampling of original dataset.

    For Koopman autoencoders, the input and target data are the same, but for the
    target data being shifted by a time lag. In order to ensure that the input and 
    target datasets have the same number of samples, we need to truncate the input 
    datasets to be the same length. This is handled by the `truncation` parameter.

    In the past, the hacky way I accomplished this was to apply `lag` with 
    the negative amount for the target tensor:
    ```
    X = load_autoencoder_tensors('train', lag=timedelta(days=-5), subsample=20)
    y = load_autoencoder_tensors('train', lag=timedelta(days=5), subsample=20)
    ```
    """
    # Handle different datatypes for truncation parameter
    if truncation is None:
        truncation = timedelta(days=0)
    elif isinstance(truncation, Union[int, np.integer]):
        truncation = timedelta(days=truncation)
    else:
        assert isinstance(truncation, timedelta)
    
    if datatype == 'train':
        member_ids = GLOBALS['train_ids']
        X_list = []
        for member_id in member_ids:
            X = load_features(member_id=member_id, data_path=data_path)
            X = processing.shift_times(X, lag)
            end_time = X.time[-1].item() - truncation
            X = X.sel(time=slice(None, end_time))
            X = X.isel(time=slice(0, None, subsample))
            X = X.values.astype(np.float32)
            X_list.append(X)
        # Concatenate along the time dimension
        X = np.concatenate(X_list, axis=0)
    elif datatype == 'val':
        member_id = GLOBALS['val_id']
        X = load_features(member_id=member_id, data_path=data_path)
        X = processing.shift_times(X, lag)
        end_time = X.time[-1].item() - truncation
        X = X.sel(time=slice(None, end_time))
        X = X.isel(time=slice(0, None, subsample))
        X = X.values.astype(np.float32)
    elif datatype == 'test':
        member_id = GLOBALS['test_id']
        X = load_features(member_id=member_id, data_path=data_path)
        X = processing.shift_times(X, lag)
        end_time = X.time[-1].item() - truncation
        X = X.sel(time=slice(None, end_time))
        X = X.isel(time=slice(0, None, subsample))
        X = X.values.astype(np.float32)
    else:
        raise ValueError(
            f"Invalid datatype {datatype}; must be one of ['train', 'val', 'test']"
        )
    X = torch.from_numpy(X)
    return X

def load_multipass_tensors(
    datatype: str = 'train',
    k: int = 3,
    nu: int = None,
    subsample: int = None,
    data_path: str = os.path.join(PATHS['tensors'], 'multipass'),
):
    """
    Loads tensors for MultipassKoopmanAutoencoder.

    Assumes data has already been created using data_processing/multipass_tensors.py.

    Params:
        datatype: 'train', 'val', or 'test'
            Datatype of loaded tensors.
        k: int > 0
            Number of recurrent passes.
        nu: int
            Time lag for loaded tensors.
        subsample: int
            Level of subsampling of original dataset.
        data_path: str
            Path to netcdf files.

    See multipass_tensors.py for the default assumptions behind nu
    """
    filename = f"{datatype}_k{k}"
    if nu is not None:
        filename += f"_nu{nu}"
    if subsample is not None:
        filename += f"_s{subsample}"
    filename += ".pt"
    filename = os.path.join(data_path, filename)
    X = torch.load(filename, weights_only=True)
    return X

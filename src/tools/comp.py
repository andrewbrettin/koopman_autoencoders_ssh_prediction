"""
comp.py

Computational utilities.
"""

__all__ = [
    "get_spectrum",
]

import numpy as np
import pandas as pd
import xarray as xr
from xrft import xrft

def npcorr(x: np.ndarray, y: np.ndarray, axis=0) -> np.ndarray:
    """
    Pearson correlation coefficient between two numpy arrays.
    """
    x_anom = x - x.mean(axis=axis)
    y_anom = y - y.mean(axis=axis)
    cov_xy = (x_anom * y_anom).sum(axis=axis)
    var_x = (x_anom**2).sum(axis=axis)
    var_y = (y_anom**2).sum(axis=axis)
    return cov_xy / np.sqrt(var_x * var_y)

def corr(x: xr.DataArray, y: xr.DataArray, dim='time') -> xr.DataArray:
    """
    Pearson correlation between two DataArrays x and y.
    """
    x_anom = (x - x.mean(dim=dim))
    y_anom = (y - y.mean(dim=dim))
    cov_xy = (x_anom * y_anom).sum(dim=dim)
    var_x = (x_anom**2).sum(dim=dim)
    var_y = (y_anom**2).sum(dim=dim)
    return cov_xy / np.sqrt(var_x * var_y)

def get_spectrum(x: xr.DataArray, dim: str = 'time', n_chunks: int = 10):
    """
    x: Array to compute fft on.
    dim: Dimension to apply FFT on.
    n_chunks: For smoothing, average FFT over this many chunks.
    """
    
    chunk_length = len(x[dim]) // n_chunks
    n_time = chunk_length * n_chunks
    x_chunked = x.isel(time=slice(0, n_time)).chunk({dim: chunk_length})
    
    spectrum = xrft.power_spectrum(
        x_chunked,
        dim=dim,
        window='hann',
        window_correction=True,
        detrend='linear',
        chunks_to_segments=True
    ).mean(dim=f'{dim}_segment')
    
    return spectrum
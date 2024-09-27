"""
processing.py
"""

__all__ = [
    "standardize",
    "unstandardize",
    "shift_times"
]

from typing import Union, Sequence
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import xrft

def standardize(darray, standardizer):
    return (darray - standardizer['mean']) / standardizer['std']

def unstandardize(darray, standardizer):
    return standardizer['mean'] + standardizer['std'] * darray

def shift_times(darray: xr.DataArray, lag: Union[int, timedelta, None] = None):
    """
    Selects values from darray with times shifted by timedelta 'lag'.
    Similar to 'shift' but supports arbitrary temporal resolution.

    Params:
        darray: xr.DataArray
    """
    # Datatype handling
    if lag is None:
        return darray
    elif isinstance(lag, int):
        lag = timedelta(days=lag)
    elif not isinstance(lag, timedelta):
        raise ValueError(f'lag must be type None, int, or datetime.timedelta')
    
    time = darray.indexes['time']
    times_shifted = time + lag
    # Subset to valid times for indexing
    times_shifted = np.intersect1d(time, times_shifted)
    # Set as CFTimeIndex
    times_shifted = xr.CFTimeIndex(times_shifted)
    
    return darray.sel(time=times_shifted)
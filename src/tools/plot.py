"""
plot.py
"""

__all__ = [
    'map_darray',
    'plot_loss'
]
import os
import pandas as pd
import numpy as np
import xarray as xr
import xrft

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cmocean as cmo

from src.attrs import PATHS, GLOBALS

def map_darray(
    darray, 
    ax=None,
    grid=True, 
    land_color='tab:gray',
    draw_labels=False, 
    add_colorbar=False,
    cbar_kwargs={}, 
    **kwargs
):
    """
    Creates cartopy map for the North Pacific
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Mollweide(202.5)})
        ax.set_extent((100, 300, -25, 65), crs=ccrs.PlateCarree())

    if grid:
        ax.gridlines(draw_labels=draw_labels)
    ax.add_feature(cartopy.feature.LAND, color=land_color)

    cax = darray.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, **kwargs)

    if add_colorbar:
        cb = plt.colorbar(cax, ax=ax, **cbar_kwargs)

    return ax

def plot_loss(
    model_name, 
    path=PATHS['networks'],
    ax=None,
    loss='val/mse',
    normalization=None,
    ax_kwargs={},
    **kwargs
):
    """
    Plots losses for specific experiment.
    """
    if ax is None:
        fig, ax = plt.subplots()

    log_path = os.path.join(path, model_name, 'logs')
    for version_name in np.sort(os.listdir(log_path)):
        df = pd.read_csv(os.path.join(log_path, version_name, 'metrics.csv'))
        df = df.groupby('epoch').mean()
    
        if normalization is not None:
            ax.plot(normalization * df[loss], **kwargs)
        else:
            ax.plot(df[loss], **kwargs)

    ax.set(xlabel='Epoch', ylabel='Loss')
    ax.set(**ax_kwargs)
    
    return ax
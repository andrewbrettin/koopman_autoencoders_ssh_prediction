"""
datasets.py
"""

__all__ = [
    "AutoregressionDataset",
    "RegressionDataset"
]

import os
import numpy as np
import xarray as xr
import dask

import torch
from torch.utils.data import Dataset, DataLoader

dask.config.set(scheduler='synchronous')

class AutoregressionDataset(Dataset):
    """
    Torch dataset for data that only includes regressors X
    """
    def __init__(self, X: torch.tensor):
        super(AutoregressionDataset, self).__init__()
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx, :]

class RegressionDataset(Dataset):
    """
    Torch dataset for data that includes regressors X and regressands y.
    """
    def __init__(self, X: torch.tensor, y: torch.tensor):
        super(RegressionDataset, self).__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx, :]

class FileAutoregressionDataset(Dataset):
    def __init__(self, data_path, filename, engine='h5netcdf', chunks={'sample': 1}):
        super(FileAutoregressionDataset, self).__init__()
        filename = os.path.join(data_path, f'{filename}.h5')
        self.X = xr.open_dataarray(filename, engine=engine, chunks=chunks)

    def __len__(self):
        return len(self.X['sample'])

    def __getitem__(self, idx):
        sample = self.X.isel(sample=idx)
        sample = torch.from_numpy(sample.values)
        return sample


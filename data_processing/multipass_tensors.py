import os
import sys
from datetime import datetime
from itertools import product
import warnings

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import torch

import src
from src import utils
from src.data import loading
from src.attrs import PATHS, GLOBALS

KS = [1, 20]
START_TIME = datetime.now()

REGION = sys.argv[1]
SAMPLING = sys.argv[2]
SUBPROJECT = f'cnn_{REGION}_{SAMPLING}'

def make_multipass_cnn_member(member_id, k):
    """
    Creates multipass tensor from a single ensemble member. These multipass 
    tensors are later concatenated using make_autoencoder_tensors_monthly.
    """
    subsample = 20 if 'subsampled' in SAMPLING else None
    X_darray = xr.open_dataarray(os.path.join(PATHS[SUBPROJECT], f'X_{member_id}.nc'))
    X_list = []
    for i in np.arange(k+1):
        i_last = -(k+1)+i
        X_npk = X_darray.isel(time=slice(i, i_last, subsample)).values
        X_npk = torch.from_numpy(X_npk)
        X_list.append(X_npk)
    X = torch.cat(X_list, dim=1)
    return X

def make_multipass_tensor(datatype: str, k: int):
    """
    See loading.data.load_autoencoder tensors to see how we've adapted it for monthly means here.
    """
    assert datatype in ['train', 'val', 'test']
    if datatype == 'train':
        member_ids = GLOBALS['train_ids']
        X_list = []
        for member_id in member_ids:
            X_list.append(make_multipass_cnn_member(member_id, k))
        # Concatenate members along sample axis
        X = torch.cat(X_list, dim=0)
    elif datatype == 'val':
        member_id = GLOBALS['val_id']
        X = make_multipass_cnn_member(member_id, k)
    else:
        member_id = GLOBALS['test_id']
        X = make_multipass_cnn_member(member_id, k)
    return X

def main():
    utils.log(f"Begin script {__file__}")
    utils.print_os_environ()

    os.makedirs(os.path.join(PATHS[SUBPROJECT], 'multipass'), exist_ok=True)

    for k in KS:
        for datatype in ['train', 'val', 'test']:
            utils.log(f"Creating {SUBPROJECT} {datatype} data, k={k}", START_TIME)
            X = make_multipass_tensor(datatype, k)
            filename = f'{datatype}_k{k}.pt'
            torch.save(X, os.path.join(PATHS[SUBPROJECT], 'multipass', filename))
    return 0

if __name__ == "__main__":
    main()
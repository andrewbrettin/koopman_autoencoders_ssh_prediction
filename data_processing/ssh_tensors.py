"""
ssh_tensors.py

"""


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

# Globals
REGION = sys.argv[1]
SAMPLING = sys.argv[2]
SUBPROJECT = f'cnn_{REGION}_{SAMPLING}'
K = 20

# Runtime variables
START_TIME = datetime.now()

def main():
    utils.log(f"Begin script {__file__}")
    utils.print_os_environ()
    print(sys.argv)

    for datatype in ['train', 'val', 'test']:
        utils.log(f"Creating {SUBPROJECT} {datatype} SSH-only data, k={K}", START_TIME)
        path = os.path.join(PATHS[SUBPROJECT], 'multipass')
        
        X = torch.load(
            os.path.join(path, f'{datatype}_k{K}.pt')
        )
        print(X.shape)
        X = X[:, 0::6, :, :].clone()
        print(X.shape)
        
        torch.save(X, os.path.join(path, f'{datatype}_ssh_k{K}.pt'))
        del X

    utils.log("PROCESS COMPLETED", START_TIME)
    
    return 0

if __name__ == "__main__":
    main()

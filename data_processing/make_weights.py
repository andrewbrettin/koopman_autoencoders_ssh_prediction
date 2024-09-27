import os
from datetime import datetime
import numpy as np
import xarray as xr
import torch

from src.attrs import PATHS, GLOBALS
from src import utils
from src.data import loading

START_TIME = datetime.now()

def main():
    utils.log("Begin script")
    
    # Load data
    areas = xr.open_dataarray(os.path.join(PATHS['grid'], 'areas.nc'))
    feature_coords = loading.load_feature_coords()

    # Select areas according to feature coordinates
    areas_flattened = areas.sel(lat=feature_coords.lat, lon=feature_coords.lon)

    # Scale weights by average
    weights = (areas_flattened / areas_flattened.mean()).values

    # Save as tensor
    weights = torch.from_numpy(weights).to(torch.float32)
    filename = os.path.join(PATHS['tensors'], 'weights.pt')
    torch.save(weights, filename)

    utils.log("PROCESS COMPLETE", START_TIME)
    
    return 0

if __name__ == "__main__":
    main()
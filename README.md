# Code for "Learning Propagators for Sea Surface Height Forecasts Using Koopman Autoencoders"

This is the code for the submitted paper "Learning Propagators for Sea Surface Height Forecasts Using Koopman Autoencoders," available here: https://doi.org/10.22541/essoar.172801453.37313793/v1.

---

Contents:
* [Repository structure](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/tree/master?tab=readme-ov-file#repository-structure)
* [Package structure](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/tree/master?tab=readme-ov-file#package-structure)
* [Installing packages](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/tree/master?tab=readme-ov-file#installing-packages)

## Repository structure
```bash
koopman_autoencoders_ssh_prediction/
├─ README.md
├─ LICENSE.md
├─ src/
├─ data_processing/
├─ train/
├─ computing/
├─ qsub/
├─ figures/
├─ setup.py
├─ install_packages.sh
├─ jobqueue.yaml
├─ development_log.md
```
|name|description|
|----|-----------|
| `src` | Source code for python package.|
| `data_processing` | Data processing pipeline.|
| `train` | Directory for training networks.|
| `computing` | Directory for miscellaneous computation. |
| `qsub` | PBS scripts and logs for batch jobs.|
| `figures` | Scripts for creating figures.|

## Package structure
```bash
src/
├─ attrs.py
├─ settings.py
├─ data/
│  └─ loading.py
├─ tools/
│  ├─ processing.py
│  ├─ metrics.py
│  └─ comp.py
├─ models/
│  ├─ autoencoder.py
│  ├─ base.py
│  ├─ cnn.py
│  └─ linear_models.py
├─ train/
│  ├─ datasets.py
│  └─ losses.py
└─ utils.py
```

* [data/](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/tree/master/src/data): utilities for loading data.
    * [loading.py](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/blob/master/src/data/loading.py)
* [tools/](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/tree/master/src/tools): 
	* [processing.py](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/blob/master/src/tools/processing.py) Processing tools, e.g., standardization, reshaping data, shifting times, etc.
    * [metrics.py](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/blob/master/src/tools/metrics.py) Metrics, like MSE and weighted variance explained.
    * [comp.py](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/blob/master/src/tools/comp.py) Computational tools, like autocorrelation, spectrum.
* [models/](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/tree/master/src/models):
	* [base.py](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/blob/master/src/models/base.py) Base class for neural networks (adds save functionality)
  * [cnn.py](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/blob/master/src/models/cnn.py) Module for CNN autoencoder and CNN Koopman Autoencoder classes
  * [linear_models.py](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/blob/master/src/models/linear_models.py) Baselines, like PCA, DP, and LIM.
* [train/](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/tree/master/src/train)
	* [datasets.py](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/blob/master/src/train/datasets.py) Pytorch dataset classes for regression.
	* [losses.py](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/blob/master/src/train/losses.py) Loss functions.
* [attrs.py](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/blob/master/src/attrs.py) Project globals, e.g. file path names and constants.
* [utils.py](https://github.com/andrewbrettin/koopman_autoencoders_ssh_prediction/blob/master/src/utils.py) Various utility functions, like logging outputs with timestamps and for printing script configurations.


## Installing packages
```
conda create -n koopman python=3.11
conda activate koopman

conda install -c conda-forge xesmf gcm_filters dask netCDF4 -y
conda install -c conda-forge numpy scipy pandas xarray -y

conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

conda install -c conda-forge ipykernel ipywidgets tqdm -y
conda install -c conda-forge distributed dask-jobqueue joblib cython bottleneck -y
conda install -c conda-forge zarr cftime nc-time-axis -y
conda install -c conda-forge xrft scikit-learn scikit-image lightning -y
conda install -c conda-forge -c pyviz matplotlib seaborn cartopy cmocean bokeh hvplot -y

which pip
pip install --upgrade pip
pip install -e .
pip install pytest
pytest -v --pyargs xesmf

conda install -c conda-forge rechunker -y
conda install -c conda-forge pytables
pip install wandb
```

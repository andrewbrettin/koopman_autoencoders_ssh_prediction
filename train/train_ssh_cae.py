"""
train_ssh_cae.py

Configurations:

Project name:    "koopman_autoencoders"
Subproject name: "cnn_{region}_{sampling}"
Experiment name: "{network type}_{variables}"
Model name:      "{network type}_{variables}/{region}_{sampling}.{n_latents}"


Example: CAE, SSH only, in Pacific, on daily timescales
Project name: "koopman_autoencoders"
Subproject:   "cnn_pacific_daily_subsampled"
Experiment:   "cae_ssh"
Model name:   "cnn_pacific_daily_subsampled/cae_ssh.20"

"""


import os
import sys
import itertools
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
import xarray as xr
import dask

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

import src
from src.attrs import PATHS, GLOBALS
from src import utils

from src.data import loading
from src.train import datasets, losses
from src.models import base, koopman_autoencoder, cnn

REGION = sys.argv[1]
SAMPLING = sys.argv[2]
N_LATENTS = int(sys.argv[3])

SUBPROJECT = f'cnn_{REGION}_{SAMPLING}'
EXPERIMENT = 'cae_ssh'
MODEL_NAME = f"{SUBPROJECT}/{EXPERIMENT}.{N_LATENTS}"
MODEL_PATH = os.path.join(PATHS['networks'], MODEL_NAME)
DATA_PATH = PATHS[SUBPROJECT]

# Runtime configurations
dask.config.set(scheduler='synchronous')
NCPUS = len(os.sched_getaffinity(0))
DEVICES = utils.get_available_devices()
START_TIME = datetime.now()

def set_subconfigs(configs):
    if REGION == 'pacific':
        configs['H'] = 80
        configs['W'] = 144
    else:
        configs['H'] = 64
        configs['W'] = 96
    configs['D'] = N_LATENTS
    return configs

def initialize_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name and 'conv' in name:
            torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
    return model

def set_callbacks(configs):
    callbacks = []
    if configs['early_stopping']:
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor='val/loss', patience=configs['patience'], mode='min'
        )
    # Save model with best reconstruction error
    best_model_callback = pl.callbacks.ModelCheckpoint(
        dirpath=MODEL_PATH,
        filename='best',
        monitor='val/loss',
        save_top_k=1,
    )
    # Save models at epochs 10 and 25
    epoch_callback_10 = pl.callbacks.ModelCheckpoint(
        dirpath=MODEL_PATH,
        filename='{epoch}',
        monitor='epoch',
        mode='min',
        save_top_k=1,
        every_n_epochs=10,
    )
    
    epoch_callback_25 = pl.callbacks.ModelCheckpoint(
        dirpath=MODEL_PATH,
        filename='{epoch}',
        monitor='epoch',
        mode='min',
        save_top_k=1,
        every_n_epochs=25,
    )

    callbacks.append(early_stopping_callback)
    callbacks.append(best_model_callback)
    callbacks.append(epoch_callback_10)
    callbacks.append(epoch_callback_25)
    
    return callbacks


def main():
    utils.log(f"Begin training {__file__}")
    print(f"Experiment {MODEL_NAME}")
    utils.print_os_environ()

    # Set configurations
    utils.log("Set configurations", START_TIME)
    configs = utils.load_configs(EXPERIMENT)
    configs = set_subconfigs(configs)

    # Initialize WandB
    wandb.init(
        project="koopman_autoencoders",
        name=MODEL_NAME,
        config=configs,
        save_code=True,
        dir=os.path.join(PATHS['scratch'], 'wandb'),
    )
    pprint(configs)

    # Load data
    utils.log("Load tensors, datasets, and dataloaders", START_TIME)
    X_train = loading.load_multipass_tensors(
        datatype='train',
        k=configs['k'],
        data_path=os.path.join(DATA_PATH, 'multipass')
    )[:, 0:configs['C'], :, :]
    X_val = loading.load_multipass_tensors(
        datatype='val', 
        k=configs['k'],
        data_path=os.path.join(DATA_PATH, 'multipass')
    )[:, 0:configs['C'], :, :]

    # Datasets
    X_train = datasets.AutoregressionDataset(X_train)
    X_val = datasets.AutoregressionDataset(X_val)

    # Dataloaders
    train_dataloader = DataLoader(
        X_train,
        batch_size=configs['batch_size'],
        num_workers=2,
        shuffle=True
        
    )
    val_dataloader = DataLoader(
        X_val,
        batch_size=configs['batch_size'],
        num_workers=2,
    )

    # Weights
    weights = torch.load(os.path.join(DATA_PATH, 'weights.pt'))
    configs['weights'] = weights

    # Model
    utils.log("Init model", START_TIME)
    model = cnn.CNNAutoencoder(configs)
    model = initialize_weights(model)
    print(model)

    # Logger, callbacks, trainer
    utils.log("Init loggers, callbacks, and trainer", START_TIME)
    # Logger
    logger = pl.loggers.WandbLogger()
    # Callbacks
    callbacks = set_callbacks(configs)
    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        # strategy='ddp',
        devices='auto',
        logger=logger,
        callbacks=callbacks,
        max_epochs=configs['epochs'],
        enable_checkpointing=True,
        enable_progress_bar=False
    )

    # Training loop
    utils.log("Begin training", START_TIME)
    trainer.fit(
        model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader
    )

    # Save model with best loss
    utils.log("Save model", START_TIME)
    state_dict = torch.load(
        os.path.join(MODEL_PATH, 'best.ckpt'),
        map_location=torch.device('cpu')
    )['state_dict']
    model.load_state_dict(state_dict)
    
    model.save(
        model_path=MODEL_PATH,
        name=MODEL_NAME,
        date=datetime.now(),
        script_path=__file__,
        data_path=os.path.join(DATA_PATH, 'multipass'),
        dataset=type(X_train),
    )

    utils.log("PROCESS COMPLETE", START_TIME)
    wandb.finish()

    return 0

if __name__ == "__main__":
    main()
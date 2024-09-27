"""
train_ssh_kae.py

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
from src.train import callbacks as custom_callbacks
from src.models import base, koopman_autoencoder, cnn


REGION = sys.argv[1]
SAMPLING = sys.argv[2]
N_LATENTS = int(sys.argv[3])

SUBPROJECT = f'cnn_{REGION}_{SAMPLING}'
EXPERIMENT = 'kae_ssh'
MODEL_NAME = f"{SUBPROJECT}/{EXPERIMENT}.{N_LATENTS}_A"
MODEL_PATH = os.path.join(PATHS['networks'], MODEL_NAME)
DATA_PATH = PATHS[SUBPROJECT]

# Runtime configurations
dask.config.set(scheduler='synchronous')
NCPUS = len(os.sched_getaffinity(0))
DEVICES = utils.get_available_devices()
START_TIME = datetime.now()
pl.seed_everything(0)  # Needed to get WandB to run

def set_subconfigs(configs):
    if REGION == 'pacific':
        configs['H'] = 80
        configs['W'] = 144
    else:
        configs['H'] = 64
        configs['W'] = 96
    if 'monthly' in SUBPROJECT:
        configs['l2'] = 0.01
    configs['D'] = N_LATENTS
    
    return configs

def load_cae_from_checkpoint(configs, model_path, checkpoint_name='epoch=9'):
    cae = cnn.CNNAutoencoder(configs)

    # Get checkpoint
    checkpoint_name = checkpoint_name + '.ckpt'
    state_dict = torch.load(
        os.path.join(model_path, checkpoint_name),
        map_location=torch.device('cpu')
    )['state_dict']
    cae.load_state_dict(state_dict)

    return cae

def initialize_weights_from_cae(kae, cae):
    """
    Initializes weights of Koopman autoencoder `kae` to have the same encoder/decoder 
    as a pre-trained autoencoder `cae`. Furthermore, propagator matrix is set to the
    identity matrix.

    Koopman autoencoder and CNN autoencoder should have the same configs.
    """
    state_dict = cae.state_dict()

    with torch.no_grad():
        for name, param in kae.named_parameters():
            if name == 'linear_embedding.weight':
                param.data = 0.5 * torch.eye(20)
            else:
                param.copy_(state_dict[name])

    return kae

def kaiming_initialization(kae):
    for name, param in kae.named_parameters():
        if name == 'linear_embedding.weight':
            torch.nn.init.eye_(param)
        elif 'weight' in name and 'conv' in name:
            torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
    return kae

def set_callbacks(configs):
    callbacks = []
    # Early stopping
    if configs['early_stopping']:
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor='val/loss', patience=configs['patience'], mode='min'
        )
        callbacks.append(early_stopping_callback)

    # Last model
    last_model_callback = pl.callbacks.ModelCheckpoint(
        dirpath=MODEL_PATH,
        filename='last',
        monitor='epoch',
        save_top_k=1,
        mode='max'
    )
    callbacks.append(last_model_callback)

    # Last valid model
    valid_model_callback = custom_callbacks.ValidPropagatorModelCheckpoint(
        dirpath=MODEL_PATH,
        filename='last_valid'
    )
    callbacks.append(valid_model_callback)
    
    # Save model with best overall loss
    best_model_callback = pl.callbacks.ModelCheckpoint(
        dirpath=MODEL_PATH,
        filename='best',
        monitor='val/loss',
        save_top_k=1,
    )
    callbacks.append(best_model_callback)

    
    # Save model with best prediction loss
    pred_callback = pl.callbacks.ModelCheckpoint(
        dirpath=MODEL_PATH,
        filename='best_pred-{epoch}',
        monitor='val/prediction_mse',
        save_top_k=50,
    )
    callbacks.append(pred_callback)

    # Periodic checkpoints every 25 epochs
    periodic_callback = pl.callbacks.ModelCheckpoint(
        dirpath=MODEL_PATH,
        filename='{epoch}',
        monitor='epoch',
        every_n_epochs=25,
        save_top_k=-1
    )
    callbacks.append(periodic_callback)
    
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
        group=MODEL_NAME,
        config=configs,
        save_code=True,
        dir=os.path.join(PATHS['scratch'], 'wandb'),
    )
    pprint(configs)

    # Load data
    utils.log("Load tensors, datasets, and dataloaders", START_TIME)
    X_train = torch.load(
        os.path.join(DATA_PATH, 'multipass', f'train_ssh_k{configs["k"]}.pt')
    )
    X_val = torch.load(
        os.path.join(DATA_PATH, 'multipass', f'val_ssh_k{configs["k"]}.pt')
    )

    # Datasets
    X_train = datasets.AutoregressionDataset(X_train)
    X_val = datasets.AutoregressionDataset(X_val)

    # Dataloaders
    num_workers = NCPUS
    if hasattr(DEVICES, "__len__"):
        num_workers = num_workers // len(DEVICES)
    train_dataloader = DataLoader(
        X_train,
        batch_size=configs['batch_size'],
        num_workers=num_workers
    )
    val_dataloader = DataLoader(
        X_val,
        batch_size=configs['batch_size'],
        num_workers=num_workers
    )

    # Weights
    weights = torch.load(os.path.join(DATA_PATH, 'weights.pt'))
    configs['weights'] = weights

    # Model
    utils.log("Init model", START_TIME)
    model = cnn.CNNKoopmanAutoencoder(configs)
    
    # CAE weights
    cae_model_name = f"{SUBPROJECT}/cae_ssh.{N_LATENTS}"
    cae = load_cae_from_checkpoint(
        configs,
        os.path.join(PATHS['networks'], cae_model_name),
        checkpoint_name='epoch=9'
    )
    
    # Initialize weights
    model = initialize_weights_from_cae(model, cae)
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
        strategy='ddp',
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

    # # Save model with best params
    # utils.log("Save model", START_TIME)
    # state_dict = torch.load(
    #     os.path.join(MODEL_PATH, 'best.ckpt'),
    #     map_location=torch.device('cpu')
    # )['state_dict']
    # model.load_state_dict(state_dict)
    
    # model.save(
    #     model_path=MODEL_PATH,
    #     name=MODEL_NAME,
    #     date=datetime.now(),
    #     script_path=__file__,
    #     data_path=os.path.join(DATA_PATH, 'multipass'),
    #     dataset=type(X_train),
    #     device=DEVICES,
    # )

    utils.log("PROCESS COMPLETE", START_TIME)
    wandb.finish()

    return 0

if __name__ == "__main__":
    main()

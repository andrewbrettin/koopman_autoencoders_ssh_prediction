"""
base.py
"""

from typing import Dict
import os
import yaml
from abc import abstractmethod
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

from src.attrs import PATHS, GLOBALS


class BaseModule:
    def save(
        self,
        model_path: str,
        name: str = 'model',
        date: datetime = None,
        script_path: str = None,
        data_path: str = None,
        dataset: Dataset = None,
        n_inputs: int = None,
        n_samples: int = None,
        nu: int = None,
        k: int = None,
        subsample: int = None,
        description: str = None,
        **kwargs
    ):
        """
        Saves model parameters and run.yaml configuration file.

        Parameters:
            model_path:
                Model directory
        """
        
        # Make directory
        os.makedirs(model_path, exist_ok=True)

        # Save network
        self = self.cpu()
        torch.save(self.state_dict(), os.path.join(model_path, f"model.pt"))

        # Make yaml
        if date is None:
            date = datetime.now()
        if isinstance(dataset, Dataset):
            dataset = type(dataset)

        d = {
            # Basic attributes
            'name': name,
            'date': date,
            'script_path': script_path,
            # Model attributes
            'model': {
                'model_path': model_path,
                'class': type(self),
                'configs': self.configs,
            },
            # Data attributes
            'data': {
                'data_path': data_path,
                'dataset': dataset,
                'n_inputs': n_inputs,
                'n_samples': n_samples,
                'subsample': subsample,
                'nu': nu,
                'k': k,
            },
            # Environment settings
            'environ': {
                'PBS_JOBNAME': os.environ['PBS_JOBNAME'],
                'PBS_JOBID': os.environ['PBS_JOBID'],
                'PBS_O_WORKDIR': os.environ['PBS_O_WORKDIR'],
                'NCPUS': len(os.sched_getaffinity(0)),
                'NGPUS': os.environ['NGPUS'],
            },
            # Miscellaneous
            'misc': kwargs,
            'description': description,
        }

        yaml_path = os.path.join(model_path, 'run.yaml')
        with open(yaml_path, mode="wt", encoding="utf-8") as file:
            yaml.dump(
                d, stream=file, sort_keys=False, explicit_end=True, 
                explicit_start=True
            )


def load_model_from_yaml(model_path: str):
    with open(os.path.join(model_path, 'run.yaml')) as f:
        d = yaml.unsafe_load(f)
    
    model_dict = d['model']
    configs = model_dict['configs']
    model_class = model_dict['class']
    model = model_class(configs)
    state_dict_path = os.path.join(model_path, "model.pt")
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()

    return model

def load_model_from_checkpoint(model_class, model_path, configs):
    """
    Loads model from checkpoint.

    model_class: cnn.CNNKoopmanAutoencoder, cnn.CNNAutoencoder, etc
    model_path: Path to model.
    configs: Model configs.

    Returns model.
    """
    # Create model instance
    model = model_class(configs)

    # Get checkpoint path
    log_path = os.path.join(model_path, 'logs')
    last_version = np.sort(os.listdir(log_path))[-1]
    checkpoint_path = os.path.join(log_path, last_version, 'checkpoints')
    last_checkpoint = os.listdir(checkpoint_path)[-1]
    checkpoint_path = os.path.join(checkpoint_path, last_checkpoint)

    # Load state dict from checkpoint path
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(state_dict)
    
    return model

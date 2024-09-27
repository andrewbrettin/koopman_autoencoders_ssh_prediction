"""
utils.py
"""

all = ['log', 'load_configs']

from typing import Sequence, Dict
import os
import yaml
from datetime import datetime
from .attrs import PATHS

import torch


def log(statement: str, start_time: datetime = None) -> None:
    """
    Prints message with runtime timestamp. If `start_time` is not given,
    prints the current time. Otherwise, prints the elapsed time since
    `start_time`.
    """
    if start_time is None:
        print('='*80)
        print(datetime.now())
        print(statement)
    else:
        print("{}\t {}".format(datetime.now() - start_time, statement))

def print_os_environ():
    """
    Prints job environment settings, like jobname, job ID, #CPU and GPU.
    """
    settings = [
        'PBS_JOBNAME',
        'PBS_JOBID',
        'PBS_ARRAY_INDEX',
        'MEM_LIMIT',
        'CUDA_VISIBLE_DEVICES'
    ]
    print("NCPUS:\t", len(os.sched_getaffinity(0)))
    for setting in settings:
        if setting in dict(os.environ).keys():
            if setting == 'MEM_LIMIT':
                print("{:16s}   {:6.4f} GB".format(setting + ':', float(os.environ['MEM_LIMIT']) / 1e9))
            else:
                print("{:16s}   {:s}".format(setting + ':', os.environ[setting]))
    print()

def get_available_devices():
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        if n_devices == 1:
            return torch.device('cuda')
        else:
            devices = [torch.device('cuda', index=i) for i in range(n_devices)]
            return devices
    else:
        return torch.device('cpu')
    return device

def load_configs(experiment):
    """
    Loads configuration dictionary for a specific experiment.
    """
    with open(os.path.join(PATHS['src'], 'settings.yaml')) as f:
        configs = yaml.safe_load(f)[experiment]
    return configs



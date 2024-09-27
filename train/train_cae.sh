#!/bin/bash
#PBS -N train_cae
#PBS -A UNYU0010
#PBS -l select=1:ncpus=2:mem=64GB:ngpus=1
#PBS -l gpu_type=v100
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -m ea
#PBS -M aeb783@nyu.edu

### Activate environment
module load conda
conda activate koopman

### Run
python -u ~/src/train/train_ssh_cae.py 'pacific' 'daily_subsampled' 40

printf "\n=================================================\n"
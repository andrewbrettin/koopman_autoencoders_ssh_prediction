#!/bin/bash
#PBS -N pca_ssh
#PBS -A UNYU0010
#PBS -l select=1:ncpus=2:mem=128GB
#PBS -l walltime=02:00:00
#PBS -q casper
#PBS -m ea
#PBS -M aeb783@nyu.edu

### Activate environment
module load conda
conda activate koopman

### Run

python -u ~/src/computing/pca_ssh.py 'pacific' 'daily_subsampled'

printf "\n=================================================\n"

#!/bin/bash
#PBS -N lim_cae
#PBS -A UNYU0010
#PBS -l select=1:ncpus=2:mem=64GB:ngpus=1
#PBS -l gpu_type=v100
#PBS -l walltime=02:00:00
#PBS -q casper
#PBS -m ea
#PBS -M aeb783@nyu.edu

### Activate environment
module load conda
conda activate koopman

### Run
python -u ~/src/computing/lim_cae_ssh.py 'north_atlantic' 'monthly' 20

printf "\n=================================================\n"

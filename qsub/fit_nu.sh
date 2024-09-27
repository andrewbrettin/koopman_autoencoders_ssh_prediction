#!/bin/bash
#PBS -N fit_nu
#PBS -A UNYU0010
#PBS -l select=1:ncpus=8:mem=64GB:ompthreads=8
#PBS -l walltime=02:00:00
#PBS -q casper
#PBS -m ea
#PBS -M aeb783@nyu.edu

### Activate environment
module load conda
conda activate koopman

### Run
# python -u ~/src/computing/fit_nu_pca.py 'north_atlantic' 'daily_subsampled' 20 'lim'
python -u ~/src/computing/fit_nu_pca_ssh.py 'pacific' 'daily_subsampled' 40 'lim'

printf "\n=================================================\n"

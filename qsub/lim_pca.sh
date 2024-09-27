#!/bin/bash
#PBS -N lim
#PBS -A UNYU0010
##PBS -l select=1:ncpus=4:mem=256GB:ompthreads=4
#PBS -l select=1:ncpus=4:mem=64GB:ompthreads=4
#PBS -l walltime=2:00:00
#PBS -q casper
#PBS -m ea
#PBS -M aeb783@nyu.edu

### Activate environment
module load conda
conda activate koopman

### Run
# python -u ~/src/computing/lim_pca.py 'north_atlantic' 'daily_subsampled' 20
python -u ~/src/computing/lim_pca_ssh.py 'pacific' 'daily_subsampled' 40

printf "\n=================================================\n"

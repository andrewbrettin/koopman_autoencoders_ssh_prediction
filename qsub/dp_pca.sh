#!/bin/bash
#PBS -N dp_pca
#PBS -A UNYU0010
#PBS -l select=1:ncpus=4:mem=128GB:ompthreads=4
#PBS -l walltime=02:00:00
#PBS -q casper
#PBS -m ea
#PBS -M aeb783@nyu.edu

### Activate environment
module load conda
conda activate koopman

### Run
# python -u ~/src/computing/dp_pca.py 'pacific' 'monthly'
python -u ~/src/computing/dp_pca_ssh.py 'pacific' 'monthly' 20
python -u ~/src/computing/dp_pca_ssh.py 'north_atlantic' 'monthly' 20
printf "\n=================================================\n"

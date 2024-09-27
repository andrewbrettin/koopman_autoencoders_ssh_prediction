#!/bin/bash
#PBS -N pca
#PBS -A UNYU0010
#PBS -l select=1:ncpus=4:mem=128GB
#PBS -l walltime=02:00:00
#PBS -q casper
#PBS -m ea
#PBS -M aeb783@nyu.edu

### Echo configurations
echo "PBS Job Name:     ${PBS_JOBNAME}"
echo "PBS Job ID:       ${PBS_JOBID}"

### Activate environment
module load conda
conda activate koopman

### Run
# python -u ~/src/computing/pca.py 'pacific' 'daily_subsampled'
# python -u ~/src/computing/pca.py 'north_atlantic' 'monthly'
python -u ~/src/computing/pca.py 'north_atlantic' 'daily_subsampled'

printf "\n=================================================\n"

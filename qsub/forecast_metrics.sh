#!/bin/bash
#PBS -N metrics
#PBS -A UNYU0010
#PBS -l select=1:ncpus=2:mem=128GB:ngpus=1
#PBS -l gpu_type=v100
##PBS -l select=1:ncpus=8:mem=128GB:ompthreads=8
#PBS -l walltime=2:00:00
#PBS -q casper
#PBS -m ea
#PBS -M aeb783@nyu.edu

### Activate environment
module load conda
conda activate koopman

### Run
python -u ~/src/computing/forecast_metrics.py 'north_atlantic' 'daily_subsampled' 20


printf "\n=================================================\n"

#!/bin/bash
#PBS -N multipass
#PBS -A UNYU0010
#PBS -l select=1:ncpus=4:mem=196GB
#PBS -l walltime=02:00:00
#PBS -q casper
#PBS -m ea
#PBS -M aeb783@nyu.edu

### Activate environment
module load conda
conda activate koopman

### Run
python -u ~/src/subprojects/tensors/multipass_tensors.py 'pacific' 'monthly'

printf "\n=================================================\n"
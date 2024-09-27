#!/bin/bash
#PBS -N ssh_tensors
#PBS -A UNYU0010
#PBS -l select=1:ncpus=4:mem=128GB
#PBS -l walltime=02:00:00
#PBS -q casper
#PBS -m ea
#PBS -M aeb783@nyu.edu

### Activate environment
module load conda
conda activate koopman

### Run
python -u ~/src/tensors/ssh_tensors.py "north_atlantic" "monthly"

printf "\n=================================================\n"
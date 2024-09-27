#!/bin/bash
#PBS -N resume_kae
#PBS -A UNYU0010
#PBS -l select=1:ncpus=4:mem=128GB:ngpus=2
#PBS -l gpu_type=v100
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -m ea
#PBS -M aeb783@nyu.edu

### Activate environment
module load conda
conda activate koopman

### Run
## python -u ~/src/train/train_ssh_kae.py 'north_atlantic' 'daily_subsampled' 20
python -u ~/src/train/train_ssh_kae_resume.py 'north_atlantic' 'daily_subsampled' 20 '7ds5es1a'

printf "\n=================================================\n"
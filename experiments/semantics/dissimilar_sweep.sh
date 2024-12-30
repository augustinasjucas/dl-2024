#!/bin/bash
#SBATCH --job-name=dissimilarity_sweep
#SBATCH --account=dl_jobs
#SBATCH --partition=standard
#SBATCH --time=03:00:00

# Activate env
. ~/jupyter/bin/activate

# go to where script should be run
cd /home/tsiebert/dl-2024/experiments/semantics

# run wandb sweep
wandb login $(cat /home/tsiebert/dl-2024/tavis_wandb_key)
wandb agent "$1"
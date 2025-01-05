#!/bin/bash
#SBATCH --job-name=dissimilarity_sweep
#SBATCH --account=dl_jobs
#SBATCH --partition=standard
#SBATCH --time=06:00:00

# Activate env
. ~/jupyter/bin/activate

# go to where script should be run
cd /home/tsiebert/dl-2024/

# run wandb sweep
wandb login $(cat /home/tsiebert/dl-2024/tavis_wandb_key)

python varying_num_tasks.py --run_num='1' --num_tasks=1 & python varying_num_tasks.py --run_num='1' --num_tasks=4
python varying_num_tasks.py --run_num='1' --num_tasks=2 & python varying_num_tasks.py --run_num='1' --num_tasks=3
python varying_num_tasks.py --run_num='1' --num_tasks=5

# cleanup
rm -r wandb
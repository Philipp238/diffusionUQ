#!/bin/bash
#SBATCH --job-name=diff_t2m
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint=RTXA6000
#SBATCH --time=72:00:00

set -euo pipefail


# Activate the conda environment.
source /home/groups/ai/buelte/anaconda3/etc/profile.d/conda.sh
conda activate pfno

echo "Job:         $SLURM_JOB_ID on $(hostname)"
echo "Python:      $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python main.py -c T2M/normal.ini


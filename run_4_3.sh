#!/bin/bash
#SBATCH --job-name="Task_4_3"
#SBATCH --account=ml4h
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=task_4_3_%j.out

module load cuda/12.6
source /cluster/courses/ml4h/jupyter/bin/activate

# --- Updated Search Path ---
export PYTHONPATH=$PYTHONPATH:$(pwd):/home/lfrigoli/chronos_models

# Helps prevent CUDA Out of Memory by managing memory fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

python3 Scripts/Task_4_3.py
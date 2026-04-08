#!/bin/bash
#SBATCH --job-name="Task_2_3b"
#SBATCH --account=ml4h
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --output=task_2_3b_%j.out

module load cuda/12.6
source /cluster/courses/ml4h/jupyter/bin/activate

# Ensure Python can see your 'utilities' folder from the Scripts directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

python3 Scripts/Task_2_3b.py
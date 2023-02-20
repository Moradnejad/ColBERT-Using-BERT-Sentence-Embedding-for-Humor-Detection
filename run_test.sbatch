#!/bin/bash
#SBATCH --job-name=my_numpy_task
#SBATCH --error=reddit.err
#SBATCH --output=reddit.log
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=40
#SBATCH --gpus=0

module purge
module load CUDA/11.2 Python/Anaconda_v10.2019

source deactivate
source activate my_py_env 

srun python run_test.py
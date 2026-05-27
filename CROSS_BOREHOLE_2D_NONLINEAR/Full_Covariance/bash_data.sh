#!/bin/bash
#SBATCH -J Generate_data_2D
#SBATCH -o Generate_data_2D.o%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p short
#SBATCH -t 7-00:00:00

set -euo pipefail

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate ml_env

# Reduce TF noise
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=""

# Quick sanity check
python -c "import numpy as np, scipy, matplotlib; print('Environment OK')"

python generate_data.py >> generate_data.txt
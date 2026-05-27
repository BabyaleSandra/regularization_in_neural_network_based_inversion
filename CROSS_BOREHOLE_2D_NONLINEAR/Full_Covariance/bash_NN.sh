#!/bin/bash
    #SBATCH -J NN_2D-full
#SBATCH -o NN.o%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p short
#SBATCH -t 7-00:00:00

set -euo pipefail

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate ml_env

# Reduce TF noise
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=""          # force CPU (optional)
# export TF_ENABLE_ONEDNN_OPTS=0        # only if you want to disable oneDNN

# Quick sanity check
python -c "import numpy as np, pandas as pd, scipy, sklearn, tensorflow as tf; print('TF', tf.__version__)"

python NN.py >> NN.txt
# Regularization in Neural Network-Based Inversion

This repository contains numerical experiments for neural network-based inverse problems with implicit regularization induced by prior sampling distributions and covariance structure choices.

The experiments investigate how Gaussian, Laplace, Total Variation (TV), and Uniform priors influence learned inverse operators using standard neural networks (NNs) and Physics-Informed Neural Networks (PINNs).

The repository currently includes:

- 1D linear Wing inverse problem
- 1D nonlinear gravity anomaly inversion
- 2D nonlinear cross-borehole seismic tomography

---

# Repository Structure

```text
.
├── WING_1D_LINEAR/
├── GRAVITY_ANOMALY_1D_NONLINEAR/
└── CROSS_BOREHOLE_2D_NONLINEAR/
```

Each experiment contains:

```text
Experiment/
├── Constant_Covariance/
├── Full_Covariance/
└── Figures/
```

---

# Experiment Organization

## Constant_Covariance

Experiments using a scaled identity covariance matrix

\[
\mathbf{C}_m = \sigma^2 \mathbf{I}.
\]

Contains:

- `generate_data.py`  
  Generates synthetic training and testing data.

- `NN.py`  
  Neural network inversion experiments.

- `PINN.py`  
  Physics-informed neural network inversion experiments.

- `functions.py`  
  Utility and helper functions used by the experiments.

- `bash_data.sh`  
  SLURM/bash script for data generation on HPC systems.

- `bash_NN.sh`  
  SLURM/bash script for NN experiments.

- `bash_PINN.sh`  
  SLURM/bash script for PINN experiments.

- `DATA/`  
  Synthetic observations, true models, prior samples, and Gauss-Newton baseline solutions.

- `RESULTS_NN/`
  - prediction outputs
  - training histories
  - evaluation metrics for NN experiments

- `RESULTS_PINN/`
  - prediction outputs
  - training histories
  - evaluation metrics for PINN experiments

---

## Full_Covariance

Experiments using correlated covariance operators.

Contains:

- `generate_data.py`  
  Generates synthetic training and testing data.

- `NN.py`  
  Neural network inversion experiments.

- `PINN.py`  
  Physics-informed neural network inversion experiments.

- `functions.py`  
  Utility and helper functions used by the experiments.

- `bash_data.sh`  
  SLURM/bash script for data generation on HPC systems.

- `bash_NN.sh`  
  SLURM/bash script for NN experiments.

- `bash_PINN.sh`  
  SLURM/bash script for PINN experiments.

- `DATA/`  
  Synthetic observations, true models, prior samples, and Gauss-Newton baseline solutions.

- `RESULTS_NN/`
  - prediction outputs
  - training histories
  - evaluation metrics for NN experiments

- `RESULTS_PINN/`
  - prediction outputs
  - training histories
  - evaluation metrics for PINN experiments

---

## Figures

Contains:

- `generate_figures.ipynb`  
  Notebook used to reproduce manuscript figures.

- manuscript-quality figures
- summary visualizations

---

# Neural Architectures

The experiments include:

- MLP
- CNN / CNN2D
- DeepONet with Fourier trunk features

---

# Prior Distributions

The following prior distributions are studied:

- Gaussian prior
- Laplace prior
- Total Variation (TV) prior
- Uniform prior

---

# Methods

The repository compares:

- Neural Networks (NN)
- Physics-Informed Neural Networks (PINN)
- Gauss-Newton / MAP estimation baselines

under different covariance structures and prior sampling distributions.

---

# Running Experiments

## Generate data

```bash
python generate_data.py
```

---

## Run neural network experiments

```bash
python NN.py
```

---

## Run PINN experiments

```bash
python PINN.py
```

---

## HPC execution

Example:

```bash
sbatch bash_NN.sh
```

---

# Figures

Figures can be reproduced using:

```text
Figures/generate_figures.ipynb
```

Generated figures include:

- architecture comparisons
- NN vs PINN comparisons
- covariance comparisons
- prior comparisons
- reconstruction visualizations

---

# Main Dependencies

- numpy
- scipy
- pandas
- matplotlib
- tensorflow
- keras
- scikit-learn

---

# Research Context

This repository accompanies research on:

- learned inverse operators
- implicit regularization
- prior-induced regularization
- neural inverse problems
- physics-informed learning
- ill-posed inverse problems

---

# Author

Sandra R. Babyale  
Computational Mathematics, Science and Engineering  
Boise State University


# GNN-HGM-CLUSTER-CALIBRATION

A Graph Neural Network (GNN) extension to the **SALT (PyTorch Lightning)** framework for calorimeter cluster energy calibration using a **Heterogeneous Gaussian Mixture (HGM)** regression head.

## Overview

Accurate energy reconstruction is essential in high-energy physics experiments, where calorimeter measurements are used to estimate the energy of particles. Conventional regression models may struggle to capture complex, non-Gaussian residual distributions and energy-dependent reconstruction biases.

This project integrates a Gaussian mixture-based probabilistic regression head into the GNN pipeline to model the conditional distribution of reconstructed cluster energy and estimate predictive uncertainty.

## Key Features

- **Gaussian Mixture Regression:** Uses three Gaussian components (K = 3) to represent complex energy distributions.
- **Uncertainty Quantification:** Predicts mixture weights, mean energies, and variances for each component.
- **Probabilistic Training:** Uses Negative Log-Likelihood (NLL) as the training objective.
- **Numerical Stability:** Incorporates stable likelihood computation and variance constraints.
- **Model Evaluation:** Provides tools to analyze energy response, reconstruction bias, resolution, and residual distributions.
- **Dataset Processing:** Supports HDF5-based data handling and validation workflows.

## Repository Structure

```text
salt/
├── models/task/                 # Regression task heads
├── gnn_new_k3.yaml              # HGM training configuration
├── process_dataset.py           # Dataset preprocessing
├── verify_gmm.py                # Distribution and pipeline checks
├── compare_regression_models.py # Model comparison
└── final_evaluation.py           # Final evaluation and metrics
```

## Installation

```bash
git clone https://github.com/Samhitha-03/GNN-HGM-CLUSTER-CALIBRATION.git
cd GNN-HGM-CLUSTER-CALIBRATION

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Training

Train the model using the K = 3 HGM configuration:

```bash
python -m salt.train --config salt/gnn_new_k3.yaml
```

## Evaluation

Verify the predicted Gaussian mixture distributions and compare regression models:

```bash
python salt/verify_gmm.py
python salt/compare_regression_models.py
```

Additional evaluation tools are available in `salt/final_evaluation.py`.

## Project Goals

- Improve the representation of non-Gaussian energy response distributions.
- Investigate systematic biases in calorimeter energy reconstruction.
- Quantify predictive uncertainty using probabilistic regression.
- Compare HGM-based regression with conventional regression approaches.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

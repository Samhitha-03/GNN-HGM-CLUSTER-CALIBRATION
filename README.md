
# GNN-HGM-CLUSTER-CALIBRATION

A Graph Neural Network (GNN) extension to the **SALT (PyTorch Lightning)** framework for calorimeter cluster energy calibration using a **Heterogeneous Gaussian Mixture (HGM)** regression head.

## Overview

Accurate energy reconstruction is essential in high-energy physics experiments, where calorimeter measurements are used to estimate particle energies. Conventional regression models may struggle to capture complex, non-Gaussian residual distributions and energy-dependent reconstruction biases.

This project integrates a Gaussian Mixture-based probabilistic regression head into the GNN pipeline to model the conditional distribution of calorimeter cluster energy and estimate predictive uncertainty.

The approach explores the use of multiple Gaussian components to represent complex energy response distributions, rather than relying on a single deterministic prediction or a single Gaussian distribution.

## Key Features

- **Gaussian Mixture Regression:** Uses multiple Gaussian components to model complex energy distributions.
- **Uncertainty Quantification:** Predicts mixture weights, mean energies, and variances for each component.
- **Probabilistic Training:** Uses Negative Log-Likelihood (NLL) as the training objective.
- **Numerical Stability:** Incorporates stable likelihood computation and variance constraints.
- **Model Evaluation:** Analyzes energy response, reconstruction bias, resolution, and residual distributions through visualization and comparison scripts.
- **Dataset Processing:** Supports HDF5-based data handling and validation workflows using `process_dataset2.py`.
- **Baseline Comparison:** Supports comparisons between standard regression, Gaussian regression, and HGM-based regression.

## Repository Structure

```text
GNN-HGM-CLUSTER-CALIBRATION/
│
├── salt/
│   ├── after_plots.py                 # Post-training visualization
│   ├── before_plots.py                # Pre-training distribution analysis
│   ├── dataset.py                     # Dataset loading and batching
│   ├── model.py                       # GNN architecture and task definitions
│   └── process_dataset2.py            # Dataset preprocessing
│
├── .gitignore                         # Excluded files and build artifacts
├── README.md                          # Project documentation
├── class_dict.yaml                    # Class mapping definitions
├── gaussian_regression.yaml           # Gaussian regression configuration
├── gnn_k3.yaml                        # K=3 HGM training configuration
├── norm_dict.yaml                     # Feature normalization parameters
├── plots_overlay.py                   # Model comparison plotting utility
├── requirements.txt                   # Python dependencies
└── standard_regression.yaml           # Standard regression configuration
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Samhitha-03/GNN-HGM-CLUSTER-CALIBRATION.git
cd GNN-HGM-CLUSTER-CALIBRATION
```

### 2. Set Up the Environment

Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Dataset Preprocessing

Prepare the calorimeter cluster data using the HDF5 preprocessing pipeline:

```bash
python salt/process_dataset2.py
```

**Note:** Ensure that the dataset paths and preprocessing parameters in the configuration files match your local or cluster directory structure.

## Training

Train the HGM regression model using the K = 3 configuration:

```bash
python -m salt.train --config gnn_k3.yaml
```

Alternatively, train using the baseline regression configurations:

**Gaussian Regression**

```bash
python -m salt.train --config gaussian_regression.yaml
```

**Standard Regression**

```bash
python -m salt.train --config standard_regression.yaml
```

## Evaluation and Visualization

The repository includes scripts for analyzing data distributions before and after training and comparing model outputs.

**Pre-training Distribution Analysis**

```bash
python salt/before_plots.py
```

**Post-training Analysis**

```bash
python salt/after_plots.py
```

**Model Comparison and Overlay Plots**

```bash
python plots_overlay.py
```

## Project Goals

- Improve the representation of non-Gaussian calorimeter energy response distributions.
- Investigate and reduce systematic biases in energy reconstruction.
- Quantify predictive uncertainty through probabilistic regression.
- Evaluate the effectiveness of Gaussian mixture-based regression.
- Compare HGM performance against standard and Gaussian regression baselines.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

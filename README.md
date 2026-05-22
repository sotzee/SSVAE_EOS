# SSVAE

This repository implements a Semi-Supervised Variational Autoencoder (SSVAE) framework for generating and reconstructing neutron star equations of state (EOS). The model learns latent representations of EOSs while incorporating astrophysical observables such as the maximum neutron star mass and the radius of a 1.4 solar mass neutron star.

This repository accompanies the manuscript:

**A Semi-Supervised Variational Autoencoder for Neutron Star Equation-of-State Generation**

Tianqi Zhao, Fanglida Yan, Alex Ross, and James M. Lattimer

---

## Overview

The SSVAE framework aims to:

- Learn compact latent representations of neutron star EOSs
- Generate physically plausible EOSs
- Incorporate supervised astrophysical observables
- Reconstruct EOSs with controllable physical properties
- Explore latent space sensitivity to neutron star observables

The input EOS representation consists of:

- Sound speed squared profile: \(c_s^2(P)\)
- EOS boundary quantities:
  - \(n_{B,\mathrm{cc}}\)
  - \(\epsilon_{\mathrm{cc}}\)
  - \(P_{\mathrm{cc}}\)
  - \(n_{B,\max}\)
  - \(\epsilon_{\max}\)
  - \(P_{\max}\)

Supervised observables:

- Maximum neutron star mass \(M_{\max}\)
- Radius at \(1.4 M_\odot\): \(R_{1.4}\)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your_username/SSVAE_EOS.git
cd SSVAE_EOS
```

Create a Python environment:

```bash
conda create -n ssvae python=3.11
conda activate ssvae
```

Install required packages:

```bash
pip install tensorflow
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install joblib
```

---

## Input data format

The code expects three input files:

### data_obs.txt

Observed quantities:

```text
M_max     R_1.4
```

Example:

```text
2.18   12.7
2.23   12.9
...
```

---

### data_cs2.txt

Sound speed profiles:

```text
c_s^2(P_1) c_s^2(P_2) ... c_s^2(P_101)
```

---

### data_boundary.txt

Boundary quantities:

```text
nB_cc
eps_cc
pressure_cc
nB_max
eps_max
pressure_max
```

---

## Training

To train a single model:

Modify in `main.py`

```python
single_run=True

parameters=[2,0.01,5]
```

where:

```text
parameters = [
    latent_dim_variational,
    eta,
    kappa
]
```

Then run:

```bash
python main.py output_directory ./data_skyrme/
```

---

## Hyperparameter sweep

Multiple hyperparameter combinations can be trained automatically.

Current default grid:

```python
latent_dim_list=[1,2,3]

eta_list=[0.0001,0.001,0.01,0.1]

kappa_list=[1,2,5,10]
```

Run:

```bash
bash run_train_batch.bash
```

---

## Parallel execution

Parallel execution can be enabled through:

```python
parallel_run=True
```

The code will automatically distribute the parameter grid.

---

## Model evaluation

Evaluate reconstructed EOS predictions:

```bash
bash run_MRtest_batch.bash
```

This computes astrophysical observables for reconstructed EOSs on the test dataset.

---

### Figure generation

The notebooks below reproduce the figures used in the manuscript:

- `paper_fig_model_picked.ipynb`: generates figures related to the selected SSVAE model, including latent space analysis and reconstruction results.
- `paper_fig_skyrme.ipynb`: generates figures related to the Skyrme EOS dataset and associated neutron star properties.
  
---

## Citation

If you use this repository in your work, please cite:

```bibtex
@article{Zhao2026SSVAE,
  author={Tianqi Zhao and Fanglida Yan and Alex Ross and James M. Lattimer},
  title={A Semi-Supervised Variational Autoencoder for Neutron Star Equation-of-State Generation},
  journal={Machine Learning: Science and Technology},
  year={2026}
}
```

---
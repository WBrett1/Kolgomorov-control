# Data-driven control of extreme events in turbulent flows through latent space clustering

**Authors:** Youssef Shehata, Kevin Schuurman, Pablo Dominguez Estevez, Nguyen Anh Khoa Doan 

This repository contains the official implementation of our paper ["Data-driven control of extreme events in turbulent flows through latent space clustering"](https://journals.aps.org/prfluids/abstract/10.1103/jbgr-wsnn), published in _Physical Review Fluids_.

## Abstract

Turbulent flows can exhibit extreme events, which are characterized by sudden bursts in the system observables. These events pose significant challenges for prediction and control owing to their intermittent, high-dimensional, and strongly nonlinear nature. In this work, we present a data-driven control framework for the suppression of extreme events in turbulent flows, leveraging reduced-order modeling for nonlinear compression of high-dimensional flow fields and a data-driven clustering algorithm for the identification of precursors to extreme events. A control law is defined in the low-dimensional latent space and is optimized to efficiently mitigate extreme events through actuation values associated with the preidentified clusters. To address issues in the latent space structure that arise from invariance transformations present in many nonlinear systems, leading to an inflated latent space, we employ symmetry-aware autoencoders to establish a more structured and compact latent space. By analyzing high-fidelity simulations of a canonical chaotic flow (Kolmogorov flow), we demonstrate how the resulting closed-loop dynamics exhibit a substantial reduction (ca. 99.4%) in the frequency and intensity of extreme events. The ability to incorporate controller limitations, including actuator latency relevant to practical applications of the framework, has also been demonstrated. This study highlights the efficacy of data-driven methods that require little to no prior knowledge of the underlying system dynamics to achieve effective flow control, thereby providing a pathway for general real-time suppression of extreme events in turbulent flows.

## Installation

We recommend using Anaconda to set up the environment. You can create the required environment using the provided configuration file:

```bash
conda env create -f cbc.yml
conda activate cbc
```

## Dataset

The dataset used in this project can be downloaded from [here](https://doi.org/10.5281/zenodo.21853176) (3.15 GB). Unzip the downloaded file to create the directory `/datasets`.

## Running the Code

Our pipeline consists of three main steps:

### Step 1: Training the Autoencoder

The training of the different types of autoencoder (linear/POD, nonlinear, and symmetry-aware) is carried out in the following Notebook: ``autoencoders/train_autoencoder.ipynb``. The pre-trained autoencoders used in the paper can be found at ``autoencoders/pretrained_models``.

### Step 2: Obtaining clusters in the latent space

Run the clustering algorithm through the following command, after modifying the parameters at the beginning of the file:

```bash
python clustering/main_with_loop_only_features_with_evaluation.py
```

### Step 3: Optimization of the control law

To start the optimization and obtain a set of optimized actuation values $b_k$, run the following Python script:

```bash
python control_optimization/control_law_optimization.py
```
To evaluate the performance of the optimized control law, run the following Python script:

```bash
python control_optimization/control_law_evaluation.py
```

Update the parameters at the beginning of both files in advance to try different configurations during optimization and/or evaluation.

## Citation

If you find this work useful, please cite our paper:

```
@article{jbgr-wsnn,
  title = {Data-driven control of extreme events in turbulent flows through latent space clustering},
  author = {Shehata, Youssef and Schuurman, Kevin and Est{\'e}vez, Pablo Dom{\'{\i}}nguez and Doan, Nguyen Anh Khoa},
  journal = {Phys. Rev. Fluids},
  volume = {11},
  issue = {9},
  pages = {093902},
  numpages = {31},
  year = {2026},
  month = {Sep},
  publisher = {American Physical Society},
  doi = {10.1103/jbgr-wsnn},
  url = {https://link.aps.org/doi/10.1103/jbgr-wsnn}
}
```

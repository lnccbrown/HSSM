<div style="position: relative; width: 100%;">
  <img src="docs/images/mainlogo.png" style="width: 250px;">
  <a href="https://ccbs.carney.brown.edu/brainstorm" style="position: absolute; right: 0; top: 50%; transform: translateY(-50%);">
    <img src="docs/images/Brain-Bolt-%2B-Circuits.gif" style="width: 100px;">
  </a>
</div>

## HSSM - Hierarchical Sequential Sampling Modeling

![PyPI](https://img.shields.io/pypi/v/hssm)
![PyPI - Downloads](https://img.shields.io/pypi/dm/HSSM?link=https%3A%2F%2Fpypi.org%2Fproject%2Fhssm%2F)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hssm)
![GitHub pull requests](https://img.shields.io/github/issues-pr/lnccbrown/HSSM)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/lnccbrown/HSSM/run_tests.yml)
![GitHub Repo stars](https://img.shields.io/github/stars/lnccbrown/HSSM)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

### Overview

HSSM is a Python toolbox that provides a seamless combination of state-of-the-art likelihood approximation methods with the wider ecosystem of probabilistic programming languages. It facilitates flexible hierarchical model building and inference via modern MCMC samplers. HSSM is user-friendly and provides the ability to rigorously estimate the impact of neural and other trial-by-trial covariates through parameter-wise mixed-effects models for a large variety of cognitive process models. HSSM is a <a href="https://ccbs.carney.brown.edu/brainstorm">BRAINSTORM</a> project in collaboration with the Center for Computation and Visualization and the Center for Computational Brain Science within the Carney Institute at Brown University.

- Allows approximate hierarchical Bayesian inference via various likelihood approximators.
- Estimate impact of neural and other trial-by-trial covariates via native hierarchical mixed-regression support.
- Extensible for users to add novel models with corresponding likelihoods.
- Built on PyMC with support from the Python Bayesian ecosystem at large.
- Incorporates Bambi's intuitive `lmer`-like regression parameter specification for within- and between-subject effects.
- Native ArviZ support for plotting and other convenience functions to aid the Bayesian workflow.
- Utilizes the ONNX format for translation of differentiable likelihood approximators across backends.

### [Official documentation](https://lnccbrown.github.io/HSSM/).

## Installation

**Update**: From HSSM 0.2.1 on, we recommend `conda` as the virtual environment manager for HSSM. We will also gradually make HSSM available directly through `conda-forge` in the near future. For now, please follow the instruction below to install HSSM:

### Step 1: Create a conda environment

If you haven't already, please follow the [Anaconda official website](https://www.anaconda.com/download) to install anaconda. We assume that you already have one of [Anaconda](https://www.anaconda.com/download), [Miniconda](https://docs.anaconda.com/free/miniconda/index.html), [miniforge](https://github.com/conda-forge/miniforge/releases), or [mambaforge](https://github.com/conda-forge/miniforge/releases) installed on your system and have access to either `conda` or `mamba` available on your command line.

To create a conda environment, use the following command. Substitute `mamba` for `conda` if `mamba` is available:

```bash
conda create -n <your-env-name> python=3.11
conda activate <your-env-name>
```

Substitute `<your-env-name>` with the name of the virtual environment that you choose. HSSM 0.2.0 and above supports Python versions 3.10 and 3.11.

### Step 2: Install PyMC through conda-forge

Installation through `conda-forge` is the official way of installing PyMC. This will also install other libraries such as `libblas` that PyMC requires to run properly.

```bash
conda install -c conda-forge pymc
```

As of HSSM 0.2.1, HSSM supports PyMC 5.10.4. If a future newer version of PyMC causes compatibility issues, please specify the version of PyMC:

```bash
conda install -c conda-forge pymc=5.10
```

### Step 3: Install `hssm` through `pip`

In the same environment, install `hssm` through `pip`.

```bash
pip install hssm
```

## Advanced Installation

`hssm` is also available through PyPI. You can directly install it with pip into any virtual environment via:

```bash
pip install hssm
```

You can also install the bleeding-edge version of `hssm` directly from this repo:

```bash
pip install git+https://github.com/lnccbrown/HSSM.git
```

While this installation is much simpler, you might need optional dependencies to use JAX-based samplers and to produce model graphs. You might also encounter this warning message `WARNING (pytensor.tensor.blas): Using NumPy C-API based implementation for BLAS functions.` Please refer to our [advanced installation guide](https://lnccbrown.github.io/HSSM/getting_started/installation/) for more details.

**Note**: Possible solutions to any issues with installations with hssm can be located
[here](https://github.com/lnccbrown/HSSM/discussions). We recommend leveraging an
environment manager with Python 3.10~3.11 to prevent any problems with dependencies
during the installation process. Please note that hssm is tested for python 3.10,
3.11. As of HSSM v0.2.0, support for Python 3.9 is dropped. Use other python
versions with caution.

## Example

Here is a simple example of how to use HSSM:

```python
import hssm

# Set float type to float32 to avoid a current bug in PyMC
# This will not be necessary in the future
hssm.set_floatX("float32")

# Load a package-supplied dataset
cav_data = hssm.load_data('cavanagh_theta')

# Define a basic hierarchical model with trial-level covariates
model = hssm.HSSM(
    model="ddm",
    data=cav_data,
    include=[
        {
            "name": "v",
            "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
                "theta": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
            },
            "formula": "v ~ theta + (1|participant_id)",
            "link": "identity",
        },
    ],
)

# Sample from the posterior for this model
model.sample()
```

To quickly get started with HSSM, please follow [this tutorial](https://lnccbrown.github.io/HSSM/getting_started/getting_started/).
For a deeper dive into HSSM, please follow [our main tutorial](https://lnccbrown.github.io/HSSM/tutorials/main_tutorial/).

## License

HSSM is licensed under [Copyright 2023, Brown University, Providence, RI](LICENSE)

## Support

For questions, please feel free to [open a discussion](https://github.com/lnccbrown/HSSM/discussions).

For bug reports and feature requests, please feel free to [open an issue](https://github.com/lnccbrown/HSSM/issues) using the corresponding template.

## Contribution

If you want to contribute to this project, please follow our [contribution guidelines](docs/CONTRIBUTING.md).

## Acknowledgements

We would like to extend our gratitude to the following individuals for their valuable contributions to the development of the HSSM package:

- [Bambi](https://github.com/bambinos/bambi) - A special thanks to the Bambi project for providing inspiration, guidance, and support throughout the development process. [Tomás Capretto](https://github.com/tomicapretto), a key contributor to Bambi, provided invaluable assistance in the development of the HSSM package.

Those contributions have greatly enhanced the functionality and quality of the HSSM.

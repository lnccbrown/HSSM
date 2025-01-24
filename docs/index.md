<div>
    <a href="https://ccbs.carney.brown.edu/brainstorm" style="display: block; float: right; padding: 10px">
        <img src="images/Brain-Bolt-%2B-Circuits.gif" style="width: 100px;">
    </a>
    <img src="images/mainlogo.png" style="width: 250px;">
</div>

![PyPI](https://img.shields.io/pypi/v/hssm)
![PyPI - Downloads](https://img.shields.io/pypi/dm/HSSM?link=https%3A%2F%2Fpypi.org%2Fproject%2Fhssm%2F)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hssm)
![GitHub pull requests](https://img.shields.io/github/issues-pr/lnccbrown/HSSM)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/lnccbrown/HSSM/run_slow_tests.yml)
![GitHub Repo stars](https://img.shields.io/github/stars/lnccbrown/HSSM)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**HSSM** (Hierarchical Sequential Sampling Modeling) is a modern Python toolbox
that provides state-of-the-art likelihood approximation methods within the
Python Bayesian ecosystem. It facilitates hierarchical model building and
inference via fast and robust MCMC samplers. User-friendly, extensible, and
flexible, HSSM can rigorously estimate the impact of neural and other
trial-by-trial covariates through parameter-wise mixed-effects models for a
large variety of cognitive process models.

HSSM is a [BRAINSTORM](https://ccbs.carney.brown.edu/brainstorm) project in
collaboration with the
[Center for Computation and Visualization (CCV)](https://ccv.brown.edu/) and the
[Center for Computational Brain Science](https://ccbs.carney.brown.edu/) within
the [Carney Institute at Brown University](https://www.brown.edu/carney/).

## Citation

Fengler, A., Xu, Y., Bera, K., Omar, A., Frank, M.J. (in preparation). HSSM: A
generalized toolbox for hierarchical bayesian estimation of computational
models in cognitive neuroscience.

## Features

- Allows approximate hierarchical Bayesian inference via various likelihood
  approximators.
- Estimate impact of neural and other trial-by-trial covariates via native
  hierarchical mixed-regression support.
- Extensible for users to add novel models with corresponding likelihoods.
- Built on PyMC with support from the Python Bayesian ecosystem at large.
- Incorporates Bambi's intuitive `lmer`-like regression parameter specification
  for within- and between-subject effects.
- Native ArviZ support for plotting and other convenience functions to aid the
  Bayesian workflow.
- Utilizes the ONNX format for translation of differentiable likelihood
  approximators across backends.

## Example

Here is a simple example of how to use HSSM:

```python
import hssm

# Load a package-supplied dataset
cav_data = hssm.load_data("cavanagh_theta")

# Define a basic hierarchical model with trial-level covariates
model = hssm.HSSM(
    model="ddm",
    data=cav_data,
    include=[
        {
            "name": "v",
            "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                "theta": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
            },
            "formula": "v ~ theta + (1|participant_id)",
            "link": "identity",
        },
    ],
)

# Sample from the posterior for this model
model.sample()
```

To quickly get started with HSSM, please follow
[this tutorial](getting_started/getting_started.ipynb). For a deeper dive into
HSSM, please follow [our main tutorial](tutorials/main_tutorial.ipynb).

## Installation

HSSM can be directly installed into your conda environment on Linux and MacOS.
Installing HSSM on windows takes only one more simple step. We have a more
detailed
[installation guide](https://lnccbrown.github.io/HSSM/getting_started/installation/)
for users with more specific setups.

**Important Update:** From HSSM 0.2.2, the official recommended way to install
HSSM is through conda.

### Install HSSM on Linux and MacOS (CPU only)

Use the following command to install HSSM into your virtual environment:

```bash
conda install -c conda-forge hssm
```

### Install HSSM on Linux and MacOS (with GPU Support)

If you need to sample with GPU, please install JAX with GPU support before
installing HSSM:

```bash
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
conda install -c conda-forge hssm
```

### Install HSSM on Windows (CPU only)

On Windows, we need to install pymc through conda, and then HSSM through pip:

```bash
conda install -c conda-forge pymc
pip install hssm
```

### Install HSSM on Windows (with GPU support)

You simply need to install JAX with GPU support before getting HSSM:

```bash
conda install -c conda-forge pymc
pip install hssm[cuda12]
```

### Support for Apple Silicon, AMD, and other GPUs

JAX also has support other GPUs. Please follow the
[Official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
to install the correct version of JAX before installing HSSM.

## Advanced Installation

### Install HSSM directly with Pip

HSSM is also available through PyPI. You can directly install it with pip into
any virtual environment via:

```bash
pip install hssm
```

!!! note

    While this installation is much simpler, you might encounter this warning message `WARNING (pytensor.tensor.blas): Using NumPy C-API based implementation for BLAS functions.` Please refer to our [advanced installation guide](https://lnccbrown.github.io/HSSM/getting_started/installation/) for more details.

### Install the dev version of HSSM

You can install the dev version of `hssm` directly from this repo:

```bash
pip install git+https://github.com/lnccbrown/HSSM.git
```

### Install HSSM on Google Colab

Google Colab comes with PyMC and JAX pre-configured. That holds true even if you
are using the GPU and TPU backend, so you simply need to install HSSM via pip on
Colab regardless of the backend you are using:

```bash
!pip install hssm
```

## Troubleshooting

!!! note

    Possible solutions to any issues with installations with hssm can be located
    [here](https://github.com/lnccbrown/HSSM/discussions). Also feel free to start a new
    discussion thread if you don't find answers there. We recommend installing HSSM into
    a new conda environment with Python 3.10 or 3.11 to prevent any problems with dependencies
    during the installation process. Please note that hssm is only tested for python 3.10,
    3.11. As of HSSM v0.2.0, support for Python 3.9 is dropped. Use unsupported python
    versions with caution.

## License

HSSM is licensed under
[Copyright 2023, Brown University, Providence, RI](LICENSE)

## Support

For questions, please feel free to
[open a discussion](https://github.com/lnccbrown/HSSM/discussions).

For bug reports and feature requests, please feel free to
[open an issue](https://github.com/lnccbrown/HSSM/issues) using the
corresponding template.

## Contributing

If you want to contribute to this project, please follow our
[contribution guidelines](CONTRIBUTING.md).

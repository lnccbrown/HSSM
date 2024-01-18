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
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/lnccbrown/HSSM/run_tests.yml)
![GitHub Repo stars](https://img.shields.io/github/stars/lnccbrown/HSSM)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

**HSSM** (Hierarchical Sequential Sampling Modeling) is a modern Python toolbox that provides state-of-the-art likelihood approximation methods within the Python Bayesian ecosystem. It facilitates hierarchical model building and inference via fast and robust MCMC samplers. User-friendly, extensible, and flexible, HSSM can rigorously estimate the impact of neural and other trial-by-trial covariates through parameter-wise mixed-effects models for a large variety of cognitive process models.

HSSM is a [BRAINSTORM](https://ccbs.carney.brown.edu/brainstorm) project in collaboration with the [Center for Computation and Visualization (CCV)](https://ccv.brown.edu/) and the [Center for Computational Brain Science](https://ccbs.carney.brown.edu/) within the [Carney Institute at Brown University](https://www.brown.edu/carney/).

## Features

- Allows approximate hierarchical Bayesian inference via various likelihood approximators.
- Estimate impact of neural and other trial-by-trial covariates via native hierarchical mixed-regression support.
- Extensible for users to add novel models with corresponding likelihoods.
- Built on PyMC with support from the Python Bayesian ecosystem at large.
- Incorporates Bambi's intuitive `lmer`-like regression parameter specification for within- and between-subject effects.
- Native ArviZ support for plotting and other convenience functions to aid the Bayesian workflow.
- Utilizes the ONNX format for translation of differentiable likelihood approximators across backends.

## Installation

`hssm` is available through PyPI. You can install it with pip via:

```bash
pip install hssm
```

You can also install the bleeding edge version of `hssm` directly from this repo:

```bash
pip install git+https://github.com/lnccbrown/HSSM.git
```

For more detailed guidance, please check out our [installation guide](getting_started/installation.md).

!!! note

    Possible solutions to any issues with installations with hssm can be located
    [here](https://github.com/lnccbrown/HSSM/discussions). We recommend leveraging an
    environment manager with Python 3.10~3.11 to prevent any problems with dependencies
    during the installation process. Please note that hssm is tested for python 3.10,
    3.11. As of HSSM v0.2.0, support for Python 3.9 is dropped. Use other python
    versions with caution.

### Setting global float type

Using the analytical DDM (Drift Diffusion Model) likelihood in PyMC without forcing float type to `"float32"` in PyTensor may result in warning messages during sampling, which is a known bug in PyMC v5.6.0 and earlier versions. We can use `hssm.set_floatX("float32")` to get around this for now.

```python
hssm.set_floatX("float32")
```

## Example

Here is a simple example of how to use HSSM:

```python
import hssm

# Set float type to float32 to avoid a current bug in PyMC mentioned above
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
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                "theta": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
            },
            "formula": "v ~ (1|participant_id) + theta",
            "link": "identity",
        },
    ],
)

# Sample from the posterior for this model
model.sample()
```

To quickly get started with HSSM, please follow [this tutorial](getting_started/getting_started.ipynb).
For a deeper dive into HSSM, please follow [our main tutorial](tutorials/main_tutorial.ipynb).

## License

HSSM is licensed under [Copyright 2023, Brown University, Providence, RI](LICENSE)

## Support

For questions, please feel free to [open a discussion](https://github.com/lnccbrown/HSSM/discussions).

For bug reports and feature requests, please feel free to [open an issue](https://github.com/lnccbrown/HSSM/issues) using the corresponding template.

## Contributing

If you want to contribute to this project, please follow our [contribution guidelines](CONTRIBUTING.md).

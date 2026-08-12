<div>
    <a href="https://ccbs.carney.brown.edu/brainstorm" style="display: block; float: right; padding: 10px">
        <img src="images/Brain-Bolt-%2B-Circuits.gif" style="width: 100px;">
    </a>
    <img src="images/mainlogo.png" style="width: 175px;">
</div>

![PyPI](https://img.shields.io/pypi/v/hssm)
![PyPI - Downloads](https://img.shields.io/pypi/dm/HSSM?link=https%3A%2F%2Fpypi.org%2Fproject%2Fhssm%2F)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hssm)
![GitHub pull requests](https://img.shields.io/github/issues-pr/lnccbrown/HSSM)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/lnccbrown/HSSM/run_slow_tests.yml)
![GitHub Repo stars](https://img.shields.io/github/stars/lnccbrown/HSSM)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**HSSM** (Hierarchical Sequential Sampling Modeling) is a modern open-source
Python toolbox for computational modeling in cognitive neuroscience. It supports
a broad range of sequential sampling models used to study decision-making,
learning, and other cognitive processes — from basic research to the analysis of
clinical effects. HSSM provides state-of-the-art likelihood approximation
methods within the Python Bayesian ecosystem and facilitates hierarchical model
building and inference via fast and robust MCMC samplers. User-friendly,
extensible, and flexible, it can rigorously estimate the impact of neural and
other trial-by-trial covariates through parameter-wise mixed-effects models.

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
- (💥 New in HSSM 0.4.0) Support for reinforcement learning sequential sampling models.
- Native ArviZ support for plotting and other convenience functions to aid the
  Bayesian workflow.
- Utilizes the ONNX format for translation of differentiable likelihood
  approximators across backends.
- Broad ecosystem support for differentiable likelihoods sourced from the sbi and BayesFlow libraries.

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
[the quickstart](getting_started/getting_started.ipynb). For the full guided
introduction, please follow [the HSSM tutorial](tutorials/main_tutorial.ipynb).

## Installation

```bash
pip install hssm        # or: uv add hssm
```

HSSM installs on all platforms with Python 3.12–3.14. For GPU sampling (CUDA
extras), Colab, the dev version, optional dependencies, and troubleshooting,
see the [Installation guide](getting_started/installation.md).

## License

HSSM is licensed under
[Copyright 2023, Brown University, Providence, RI](license.md)

## Support

For questions, please feel free to
[open a discussion](https://github.com/lnccbrown/HSSM/discussions).

For bug reports and feature requests, please feel free to
[open an issue](https://github.com/lnccbrown/HSSM/issues) using the
corresponding template.

## Contributing

If you want to contribute to this project, please follow our
[contribution guidelines](CONTRIBUTING.md).

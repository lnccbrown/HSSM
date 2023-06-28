<div id="wrapper">
  <div id="main-logo"><img src="images/HSSM2_Logo_Transparent1200.png" width="250"></div>
  <div id="main-title"><h1>Hierarchical Sequential <br />Sampling Models</h1></div>
</div>

HSSM is a Python toolbox that provides a seamless combination of state-of-the-art likelihood approximation methods with the wider ecosystem of probabilistic programming languages. It facilitates flexible hierarchical model building and inference via modern MCMC samplers. HSSM is user-friendly and provides the ability to rigorously estimate the impact of neural and other trial-by-trial covariates through parameter-wise mixed-effects models for a large variety of cognitive process models.

**Authors**: Alexander Fengler, Aisulu Omar, Paul Xu, Krishn Bera, Michael J. Frank

**Contacts**: alexander_fengler@brown.edu

**Github**: https://github.com/lnccbrown/HSSM

**Copyright**: This document has been placed in the public domain.

**License**: HSSM is licensed under [Copyright 2023, Brown University, Providence, RI](../LICENSE)

**Version**: 0.1.0

## Features

- Allows approximate hierarchical Bayesian inference via various likelihood approximators
- Rigorously estimates the impact of neural and other trial-by-trial covariates
- Extensible for users to add novel models with corresponding likelihoods
- Uses the PyMC python package for construction of probabilistic models
- Incorporates the Bambi python package for specifying regression-based trial-by-trial parameters
- Utilizes the Arviz package for sophisticated plots for all stages of the Bayesian analysis workflow
- Utilizes the ONNX format for translation of differentiable likelihood approximators across backends

## Installation

```
pip install hssm
```

## Example

Here is a simple example of how to use HSSM:

```python
# Import the package
import hssm

# Load a package-supplied dataset
cav_data = hssm.load_csv(hssm.__path__[0] + '/examples/cavanagh_theta.csv')

# Define a basic hierarchical model with trial-level covariates
model = hssm.HSSM(
    model="ddm",
    data=cav_data,
    include=[
    {
    "name": "v",
    "prior": {
        "Intercept": {"name": "Uniform",
                      "lower": -3.0,
                      "upper": 3.0},
        "theta": {"name": "Uniform",
                  "lower": -1.0,
                  "upper": 1.0},
    },
    "formula": "v  ̃ (1|subj_idx) + theta",
    "link": "identity",
    },
    ]
)

# Sample from the posterior for this model
model.sample()
```

## Contributing
If you want to contribute to this project, please familiarize yourself with our [contribution guidelines](CONTRIBUTING.md).

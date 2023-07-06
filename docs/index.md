<div id="wrapper">
  <div id="main-logo"><img src="images/mainlogo.png" width="250"></div>
  <div id="main-title"><h1>Hierarchical Sequential <br />Sampling Models</h1></div>
</div>

HSSM is a Python toolbox that provides a seamless combination of state-of-the-art likelihood approximation methods with the wider ecosystem of probabilistic programming languages. It facilitates flexible hierarchical model building and inference via modern MCMC samplers. HSSM is user-friendly and provides the ability to rigorously estimate the impact of neural and other trial-by-trial covariates through parameter-wise mixed-effects models for a large variety of cognitive process models.

**Authors**: Alexander Fengler, Aisulu Omar, Paul Xu, Krishn Bera, Michael J. Frank

**Contacts**: alexander_fengler@brown.edu

**Github**: https://github.com/lnccbrown/HSSM

**Copyright**: This document has been placed in the public domain.

**License**: HSSM is licensed under [Copyright 2023, Brown University, Providence, RI](../LICENSE)

**Version**: 0.1.0

- Allows approximate hierarchical Bayesian inference via various likelihood approximators.
- Estimate impact of neural and other trial-by-trial covariates via native hierarchical mixed-regression support.
- Extensible for users to add novel models with corresponding likelihoods.
- Built on PyMC with support from the Python Bayesian ecosystem at large.
- Incorporates Bambi's intuitive `lmer`-like regression parameter specification for within- and between-subject effects.
- Native ArviZ support for plotting and other convenience functions to aid the Bayesian workflow.
- Utilizes the ONNX format for translation of differentiable likelihood approximators across backends.

## Installation

`hssm` is available through PyPI. You can install it with Pip via:

```
pip install hssm
```
You can also install the bleeding edge version of `hssm` directly from this repo:

```
pip install git+https://github.com/lnccbrown/hssm.git
```

### Optional Installation

**Dependency for graph() Function**

Note: In addition to the installation of the main hssm class, there is an optional dependency for the graph() function. This dependency requires graphviz, which can be installed conveniently using conda with the following command:

```
conda install -c conda-forge python-graphviz
```
Alternatively, you have the option to install the graphviz binaries manually and then install the Python bindings using pip with the following command:

```
pip install graphviz
```
**Dependency for sampler="nuts_numpyro"**

To utilize the nuts_numpyro sampler, please follow these steps:

1. Install numpyro by executing the following command:
```
pip install numpyro
```
2. Import the necessary modules and configure the required settings:
```
import numpyro
from jax.config import config

numpyro.set_host_device_count(jax.local_device_count())
config.update("jax_enable_x64", False)
```
For more information please refer to [jax documentation](https://jax.readthedocs.io/en/latest/installation.html).

### Setting a global float32 
Using the analytical DDM (Drift Diffusion Model) likelihood in PyMC without setting pytensor.config.floatX = "float32" may result in warning messages during sampling, which is a known bug in PyMC v5.6.0 and earlier versions. Including pytensor.config.floatX = "float32" in the code can temporarily avoid these warnings.

```
pytensor.config.floatX = "float32"
```

## Example

Here is a simple example of how to use HSSM:

```python
import hssm
from hssm import load_data

# Load a package-supplied dataset
cav_data = load_data('cavanagh_theta')

# Define a basic hierarchical model with trial-level covariates
model = hssm.HSSM(
    model="ddm",
    data=cav_data,
    include=[
        {
            "name": "v",
            "prior": {
                "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
                "theta": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
            },
            "formula": "v ~ (1|subj_idx) + theta",
            "link": "identity",
        },
    ],
)

# Sample from the posterior for this model
model.sample()
```

## Example
HSSM is licensed under [Copyright 2023, Brown University, Providence, RI](LICENSE)

## Support
For questions, bug reports, or other unexpected issues, please open an issue on the GitHub repository.

## Contributing
If you want to contribute to this project, please familiarize yourself with our [contribution guidelines](CONTRIBUTING.md).

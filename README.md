<p align="left">
  <img src="docs/images/mainlogo.png" width="250">
</p>

## HSSM - Hierarchical Sequential Sampling Modeling

### Overview
HSSM is a Python toolbox that provides a seamless combination of state-of-the-art likelihood approximation methods with the wider ecosystem of probabilistic programming languages. It facilitates flexible hierarchical model building and inference via modern MCMC samplers. HSSM is user-friendly and provides the ability to rigorously estimate the impact of neural and other trial-by-trial covariates through parameter-wise mixed-effects models for a large variety of cognitive process models.

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
pip install git+https://github.com/lnccbrown/HSSM.git
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

## Contribution
If you want to contribute to this project, please familiarize yourself with our [contribution guidelines](docs/CONTRIBUTING.md).

## Acknowledgements

We would like to extend our gratitude to the following individuals for their valuable contributions to the development of the HSSM package:

- [Bambi](https://github.com/bambinos/bambi) - A special thanks to the Bambi project for providing inspiration, guidance, and support throughout the development process. [Tomás Capretto](https://github.com/tomicapretto), a key contributor to Bambi, provided invaluable assistance in the development of the HSSM package.

Those contributions have greatly enhanced the functionality and quality of the HSSM.

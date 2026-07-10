<div style="position: relative; width: 100%;">
  <img src="docs/images/mainlogo.png" alt="HSSM logo" style="width: 175px;">
  <a href="https://ccbs.carney.brown.edu/brainstorm" style="position: absolute; right: 0; top: 50%; transform: translateY(-50%);">
    <img src="docs/images/Brain-Bolt-%2B-Circuits.gif" alt="BRAINSTORM logo" style="width: 100px;">
  </a>
</div>

# HSSM - Hierarchical Sequential Sampling Modeling

[![Paper DOI](https://img.shields.io/badge/paper-10.64898%2F2026.06.05.730398-blue)](https://doi.org/10.64898/2026.06.05.730398)
[![PyPI](https://img.shields.io/pypi/v/hssm)](https://pypi.org/project/hssm/)
[![Downloads](https://static.pepy.tech/badge/hssm/month)](https://pepy.tech/projects/hssm)
[![GitHub stars](https://img.shields.io/github/stars/lnccbrown/HSSM)](https://github.com/lnccbrown/HSSM/stargazers)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)
[![Run tests](https://github.com/lnccbrown/HSSM/actions/workflows/run_tests.yml/badge.svg)](https://github.com/lnccbrown/HSSM/actions/workflows/run_tests.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/github/license/lnccbrown/HSSM)](LICENSE)
[![codecov](https://codecov.io/gh/lnccbrown/HSSM/branch/main/graph/badge.svg)](https://codecov.io/gh/lnccbrown/HSSM)

HSSM is a Python toolbox for hierarchical Bayesian neurocognitive modeling of
choice, response time, and covariate-rich trial data.

It supports a broad family of sequential-sampling models, reinforcement
learning sequential-sampling models (RLSSMs), behavioral, neural,
eye-tracking, and SCR covariates, hierarchical regression, posterior
diagnostics, posterior predictive checks, and Bayesian model comparison.

HSSM is a [BRAINSTORM](https://ccbs.carney.brown.edu/brainstorm) project in
collaboration with the Center for Computation and Visualization and the Center
for Computational Brain Science within the Carney Institute at Brown University.

## At a Glance

| Use HSSM to... | What this means in practice |
| --- | --- |
| Fit sequential-sampling models | Build DDM, LBA, race, angle, and related choice/RT models from a high-level Python API. |
| Model behavioral, neural, and other trial-wise covariates | Regress model parameters on trial-level predictors such as EEG, eye-tracking, SCR, task condition, or stimulus features. |
| Use hierarchical and mixed-effects parameter models | Estimate participant-level variation and within- or between-subject effects with Bambi-style formulas. |
| Work with RLSSMs | Combine reinforcement-learning dynamics with sequential-sampling decision models. |
| Diagnose and compare Bayesian models | Use ArviZ summaries, trace diagnostics, posterior predictive checks, and model-comparison workflows. |
| Extend models with custom likelihoods | Register new models and likelihoods when built-in model definitions are not enough. |

## Installation

Install HSSM in a fresh virtual environment with Python 3.12, 3.13, or 3.14.

### CPU

```bash
pip install hssm
```

You can also add HSSM to a `uv` project:

```bash
uv add hssm
```

### CUDA

For NVIDIA GPUs, install HSSM with the CUDA 12 extra:

```bash
pip install "hssm[cuda12]"
```

or with `uv`:

```bash
uv add "hssm[cuda12]"
```

### Apple Silicon, AMD, and Other Accelerators

JAX supports several accelerator backends. Follow the
[official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
for your platform, then install HSSM in the same environment.

### Development Version

Install the current development version directly from GitHub:

```bash
pip install git+https://github.com/lnccbrown/HSSM.git
```

With `uv`:

```bash
uv add git+https://github.com/lnccbrown/HSSM.git
```

### Google Colab

Google Colab already includes PyMC and JAX, so a standard pip install is enough:

```bash
!pip install hssm
```

### Troubleshooting

HSSM is tested on Python 3.12, 3.13, and 3.14. If installation fails, start from
a fresh virtual environment using one of those Python versions and check
[GitHub Discussions](https://github.com/lnccbrown/HSSM/discussions) for known
platform-specific fixes.

## Quick Start

```python
import arviz as az
import hssm

# Load a package-supplied choice/response-time dataset.
data = hssm.load_data("cavanagh_theta")

# Build a basic drift-diffusion model.
model = hssm.HSSM(data=data, model="ddm")

# Draw posterior samples. Increase draws/tune/chains for publication analyses.
idata = model.sample(draws=1000, tune=1000, chains=4)

# Inspect convergence and posterior summaries with ArviZ.
az.summary(idata)
az.plot_trace_dist(idata)
```

For diagnostics, interpretation, regression formulas, and model comparison, see
the [getting started guide](https://lnccbrown.github.io/HSSM/getting_started/getting_started/)
and the [main tutorial](https://lnccbrown.github.io/HSSM/tutorials/main_tutorial/).

## Where HSSM Fits

HSSM is the user-facing modeling layer in the HSSM ecosystem. It builds on
[PyMC](https://www.pymc.io/) for Bayesian inference,
[Bambi](https://bambinos.github.io/bambi/) for formula-based and hierarchical
regression, [ArviZ](https://python.arviz.org/) for diagnostics and model
comparison, and JAX/PyTensor for computation.

Within the broader ecosystem,
[ssm-simulators](https://github.com/lnccbrown/ssm-simulators) supplies simulator
and model definitions, while
[LANfactory](https://github.com/lnccbrown/LANfactory) supports likelihood
approximation workflows used to develop new likelihoods.

## Citation

Please cite the current HSSM paper:

Fengler, A., Xu, Y., Bera, K., Paniagua, C., Omar, A., and Frank, M. J.
HSSM: A Widely Applicable Toolbox for Hierarchical Bayesian Neurocognitive
Modeling. bioRxiv 2026.06.05.730398.

- DOI: [https://doi.org/10.64898/2026.06.05.730398](https://doi.org/10.64898/2026.06.05.730398)
- bioRxiv: [https://www.biorxiv.org/content/10.1101/2026.06.05.730398v1](https://www.biorxiv.org/content/10.1101/2026.06.05.730398v1)
- Software archive DOI, when needed for reproducibility:
  [https://doi.org/10.5281/zenodo.17247695](https://doi.org/10.5281/zenodo.17247695)

## Next Steps

- [Documentation](https://lnccbrown.github.io/HSSM/)
- [Getting started](https://lnccbrown.github.io/HSSM/getting_started/getting_started/)
- [Main tutorial](https://lnccbrown.github.io/HSSM/tutorials/main_tutorial/)
- [Scientific workflow tutorial](https://lnccbrown.github.io/HSSM/tutorials/scientific_workflow_hssm/)
- [RLSSM basic tutorial](https://lnccbrown.github.io/HSSM/tutorials/rlssm_basic/)
- [RLSSM custom models](https://lnccbrown.github.io/HSSM/tutorials/rlssm_advanced/)
- [Plotting and model checking](https://lnccbrown.github.io/HSSM/tutorials/plotting/)
- [GitHub Discussions](https://github.com/lnccbrown/HSSM/discussions)
- [Contribution guide](docs/CONTRIBUTING.md)

## Support

For questions, please
[open a discussion](https://github.com/lnccbrown/HSSM/discussions).

For bug reports and feature requests, please
[open an issue](https://github.com/lnccbrown/HSSM/issues) using the
corresponding template.

## Contribution

If you want to contribute to this project, please follow our
[contribution guidelines](docs/CONTRIBUTING.md).

## License

HSSM is licensed under
[Copyright 2023, Brown University, Providence, RI](LICENSE).

## Acknowledgements

We are grateful to the Bambi project for inspiration, guidance, and support.
[Tomas Capretto](https://github.com/tomicapretto), a key contributor to Bambi,
provided invaluable assistance during HSSM development.

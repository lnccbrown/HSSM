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

## Installation

```bash
pip install hssm        # or: uv add hssm
```

HSSM installs on all platforms with Python 3.12–3.14. For GPU sampling (CUDA
extras), Colab, the dev version, optional dependencies, and troubleshooting,
see the [Installation guide](getting_started/installation.md).

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

## Start here

<div class="grid cards" markdown>

-   __Fit your first model__

    ---

    From `pip install` to a result you can defend. Six steps, each building on
    the last:

    1. [Installation](getting_started/installation.md) — a working environment, GPU extras included.
    2. [Quickstart](getting_started/getting_started.ipynb) — simulate, fit, and check a DDM in about 15 minutes.
    3. [The HSSM tutorial](tutorials/main_tutorial.ipynb) — the guided introduction: model choice, priors, diagnostics, predictive checks, comparison.
    4. [Hierarchical modeling](getting_started/hierarchical_modeling.ipynb) — `lmer`-style formulas on any model parameter.
    5. [Hierarchical DDM regressions](tutorials/ddm_hierarchical_tutorial.ipynb) — map your actual design onto a formula, and recover it.
    6. [Compare and interpret models](how_to/compare_models.ipynb) — rank candidates, and recognise when the data cannot separate them.

    **Capstone:** [A complete scientific workflow](tutorials/scientific_workflow_hssm.ipynb) — one dataset, start to finish.

    *Used HDDM before?* [Coming from HDDM](explanations/coming_from_hddm.md) maps what you know onto HSSM.

-   __Bring your own likelihood or model__

    ---

    For models HSSM does not ship, or likelihoods you trained yourself:

    1. [Likelihood kinds in HSSM](explanations/likelihoods.md) — analytical, `approx_differentiable`, and blackbox, and when each applies.
    2. [The ONNX likelihood contract](how_to/custom_onnx_likelihoods.md) — the exact rules an approximate differentiable ONNX file must satisfy.
    3. [Bring your own likelihood](how_to/external_trainers.md) — the route table for networks trained in sbi or BayesFlow.
    4. Then the walkthrough for your route — [sbi NRE](tutorials/sbi_nre_integration.ipynb), [BayesFlow NLE](tutorials/bayesflow_nle_onnx_integration.ipynb), [BayesFlow LRE](tutorials/bayesflow_lre_integration.ipynb), or [JAX callables](tutorials/jax_callable_contribution_onnx_example.ipynb).
    5. [Use the low-level API with PyMC](tutorials/pymc.ipynb) — when the formula interface is the constraint.

</div>

Beyond the paths, the docs are organised by what you are doing:
[Learn](learn/index.md) for guided material,
[How-to guides](how_to/index.md) for a specific task, **Explanations** for the
reasoning behind a choice, and [Reference](reference/index.md) for exact APIs
and project metadata.

## Part of a larger toolchain

HSSM is the inference layer of a larger ecosystem: `ssm-simulators` supplies
the models and simulated data, LANfactory trains the likelihood networks that
make otherwise-intractable models estimable, and HSSM consumes them. Most users
never need the other repositories. See the canonical
[HSSM ecosystem map](ecosystem/index.md) for
ownership boundaries, contributor routes, and the complete site directory.

## What HSSM gives you

- Hierarchical Bayesian inference for a broad family of sequential sampling
  models, including those with no analytical likelihood.
- Parameter-wise mixed-effects regressions in `lmer`-like syntax, so neural and
  trial-by-trial covariates can enter any model parameter.
- Reinforcement learning sequential sampling models (RLSSMs), where the
  decision parameters are driven by a learning process.
- Custom models and likelihoods: bring an ONNX network, a JAX callable, or a
  black-box Python function.
- Built on PyMC, Bambi, and ArviZ, so the wider Python Bayesian ecosystem
  applies directly.

## Citation

Fengler, A., Xu, Y., Bera, K., Paniagua, C., Omar, A., and Frank, M. J. HSSM: A
Widely Applicable Toolbox for Hierarchical Bayesian Neurocognitive Modeling.
bioRxiv 2026.06.05.730398.

- DOI: [https://doi.org/10.64898/2026.06.05.730398](https://doi.org/10.64898/2026.06.05.730398)
- bioRxiv: [https://www.biorxiv.org/content/10.1101/2026.06.05.730398v1](https://www.biorxiv.org/content/10.1101/2026.06.05.730398v1)

## Community

- **Questions and modeling advice** —
  [open a discussion](https://github.com/lnccbrown/HSSM/discussions).
- **Bugs and feature requests** —
  [open an issue](https://github.com/lnccbrown/HSSM/issues) using the
  corresponding template.
- **Contributing** — see the
  [contribution guidelines](CONTRIBUTING.md).

## License

HSSM is licensed under
[Copyright 2023, Brown University, Providence, RI](license.md)

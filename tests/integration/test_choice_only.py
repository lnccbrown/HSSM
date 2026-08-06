"""Integration tests for choice-only (softmax inverse temperature) models.

This file used to run a 9-row sampler grid against each of 4 model shapes (36
tests). Most of that grid re-tested `HSSMBase.sample` machinery that
`test_mcmc.py` already covers: for analytical likelihoods `backend="pytensor"`
and `backend=None` resolve to the identical code path
(`distribution_utils/dist.py:807-816`), and `test_mcmc.py` runs every sampler
against it.

What is *not* covered elsewhere is `analytical` + `backend="jax"`, which wraps
the callable in a JAX logp Op instead of using it directly -- `test_mcmc.py`
always passes `backend=None` for analytical likelihoods. That axis is kept in
full here, along with the choice-only model shapes themselves.

The grid is therefore a covering array over (backend x sampler/step) and
(backend x model shape). Sampler and model shape are deliberately *not* crossed:
the sampler only ever sees a logp graph and a set of free RVs, so whether those
came from a `beta` or a `logit1` regression is invisible to it.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import hssm

hssm.set_floatX("float32", update_jax=True)


def _reg(param):
    """Return a regression spec for `param` on the synthetic `x` column."""
    return dict(
        formula=f"{param} ~ x",
        prior={
            "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "x": {"name": "Normal", "mu": 0, "sigma": 1},
        },
    )


# Model shape -> (model name, extra HSSM kwargs).
MODEL_SHAPES = {
    "default": ("softmax_inv_temperature_2", {}),
    "beta_reg": ("softmax_inv_temperature_2", {"beta": _reg("beta")}),
    "logit_reg": ("softmax_inv_temperature_2", {"logit1": _reg("logit1")}),
    "multiple_reg": (
        "softmax_inv_temperature_3",
        {"beta": _reg("beta"), "logit1": _reg("logit1")},
    ),
}

# Covering array over:
#   B = backend:      jax / pytensor
#   S = (sampler, step): pymc / pymc+slice / numpyro
#   M = model shape:  default / beta_reg / logit_reg / multiple_reg
#
# 8 rows cover every (B, S) pair and every (B, M) pair. All three samplers run
# against the jax-wrapped analytical logp, which is the path unique to this file.
PARAMETER_NAMES = "backend,sampler,step,shape"
COVERING_ARRAY = [
    ("jax", "pymc", None, "default"),
    ("jax", "pymc", "slice", "beta_reg"),
    ("jax", "numpyro", None, "logit_reg"),
    ("jax", "pymc", None, "multiple_reg"),
    ("pytensor", "pymc", None, "default"),
    ("pytensor", "pymc", "slice", "beta_reg"),
    ("pytensor", "numpyro", None, "logit_reg"),
    ("pytensor", "pymc", None, "multiple_reg"),
]


@pytest.fixture(scope="module")
def synthetic_data():
    """Generate synthetic choice-only dataset for testing."""
    n_samples = 100
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        {
            "response": rng.choice([-1, 1], size=n_samples),
            "x": rng.standard_normal(n_samples),
        }
    )


def sample(model, sampler, step):
    """Sample the model briefly, optionally with a Slice step sampler."""
    if step == "slice":
        return model.sample(
            sampler=sampler,
            cores=1,
            chains=1,
            tune=10,
            draws=10,
            step=pm.Slice(model=model.pymc_model),
            progressbar=False,
        )
    return model.sample(
        sampler=sampler,
        cores=1,
        chains=1,
        tune=10,
        draws=10,
        progressbar=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, COVERING_ARRAY)
def test_choice_only(synthetic_data, backend, sampler, step, shape):
    """Sample each covered (backend, sampler/step, model shape) cell."""
    model_name, params = MODEL_SHAPES[shape]
    model = hssm.HSSM(
        data=synthetic_data,
        model=model_name,
        loglik_kind="analytical",
        model_config={"backend": backend},
        # HSSM mutates the param dicts it is handed, so hand it a fresh copy.
        **deepcopy(params),
    )

    idata = sample(model, sampler, step)

    assert isinstance(idata, xr.DataTree)

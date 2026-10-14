"""Integration tests for MCMC sampling of missing-data and deadline models.

This file used to be two near-identical files -- `test_missing_data_mcmc.py` and
`test_missing_data_and_deadline_mcmc.py` -- each running the same 20-row grid
against a simple and a regression model (80 tests total). Their 20-row grids and
their `sample`/`run_sample` helpers were byte-identical; only the data fixture,
the ONNX network and the `deadline` flag differed.

They are merged here because `deadline` is not a separate mechanism but a
parameter of the same one (`missing_data_mixin.py:88-95`):

    if not missing_data:
        return MissingDataNetwork.NONE
    network = MissingDataNetwork.OPN if deadline else MissingDataNetwork.CPN

Both cases set `missing_data=True`; `deadline` only selects CPN vs OPN, which
downstream is `params_only=True` vs `False`. So the network becomes a `mode`
column in one covering array rather than a second copy of the grid.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pymc as pm
import pytest
import xarray as xr

import hssm

from tests.integration.regression_priors import V_REG
from hssm.utils import _rearrange_data

hssm.set_floatX("float32", update_jax=True)

FIXTURES = Path(__file__).parent.parent / "fixtures"

# Missing-data mode -> ONNX network. "cpn" is missing_data only; "opn" adds a
# deadline column and censors RTs past the deadline.
NETWORKS = {
    "cpn": FIXTURES / "ddm_cpn.onnx",
    "opn": FIXTURES / "ddm_opn.onnx",
}

# Model shape -> (data fixture name, extra HSSM kwargs).
MODEL_SHAPES = {
    "simple": ("data_ddm", {}),
    "reg": ("data_ddm_reg", {"v": V_REG}),
}


def prepare_data(base, mode):
    """Return a copy of `base` with missing RTs injected for the given mode."""
    rng = np.random.default_rng(seed=42)
    data = base.copy()
    if mode == "opn":
        data["deadline"] = data["rt"] + rng.normal(0, 0.1, data.shape[0])
        missing_indices = data["rt"] > data["deadline"]
    else:
        missing_indices = rng.choice(data.shape[0], 50, replace=False)
    data.iloc[missing_indices, 0] = -999.0
    return _rearrange_data(data)


def build_model(request, shape, mode, loglik_kind, backend, **kwargs):
    """Build a missing-data HSSM model of the given shape and mode."""
    data_fixture, params = MODEL_SHAPES[shape]
    return hssm.HSSM(
        prepare_data(request.getfixturevalue(data_fixture), mode),
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        missing_data=True,
        deadline=(mode == "opn"),
        loglik_missing_data=NETWORKS[mode],
        # HSSM mutates the param dicts it is handed, so hand it a fresh copy.
        **deepcopy(params),
        **kwargs,
    )


def sample(model, sampler, step):
    """Sample the model briefly, optionally with a Slice step sampler."""
    if step == "slice":
        model.sample(
            sampler=sampler,
            cores=1,
            chains=1,
            tune=10,
            draws=10,
            step=pm.Slice(model=model.pymc_model),
            progressbar=False,
        )
    else:
        model.sample(
            sampler=sampler, cores=1, chains=1, tune=10, draws=10, progressbar=False
        )


def sample_and_verify(model, sampler, step):
    """Sample, then check that recomputing the log-likelihood reproduces it."""
    sample(model, sampler, step)
    assert isinstance(model.traces, xr.DataTree)

    traces_copy = deepcopy(model.traces)
    del traces_copy["log_likelihood"]

    model.log_likelihood(traces_copy, inplace=True)
    assert isinstance(traces_copy, xr.DataTree)
    assert "log_likelihood" in traces_copy
    for group_ in traces_copy:
        xr.testing.assert_equal(traces_copy[group_], model.traces[group_])


# Pairwise covering array over four factors:
#   B = (loglik_kind, backend): analytical / approx+pytensor / approx+jax / blackbox
#   S = (sampler, step):        pymc / pymc+slice / numpyro
#   M = model shape:            simple / reg
#   D = missing-data mode:      cpn (missing data) / opn (deadline)
#
# 12 rows cover every valid (B,S), and every (B,M), (B,D), (S,M), (S,D) and
# (M,D) pair. blackbox has no valid numpyro cell, so it repeats `pymc` to reach
# both shapes and both modes.
PARAMETER_NAMES = "loglik_kind,backend,sampler,step,shape,mode"
COVERING_ARRAY = [
    ("analytical", None, "pymc", None, "simple", "cpn"),
    ("analytical", None, "pymc", "slice", "reg", "opn"),
    ("analytical", None, "numpyro", None, "simple", "opn"),
    ("approx_differentiable", "pytensor", "pymc", None, "reg", "opn"),
    ("approx_differentiable", "pytensor", "pymc", "slice", "simple", "cpn"),
    ("approx_differentiable", "pytensor", "numpyro", None, "reg", "cpn"),
    ("approx_differentiable", "jax", "pymc", None, "simple", "opn"),
    ("approx_differentiable", "jax", "pymc", "slice", "reg", "cpn"),
    ("approx_differentiable", "jax", "numpyro", None, "simple", "cpn"),
    ("blackbox", None, "pymc", None, "reg", "cpn"),
    ("blackbox", None, "pymc", "slice", "simple", "opn"),
    ("blackbox", None, "pymc", None, "reg", "opn"),
]

# Rejected combinations, all of which raise ValueError. These raise in `sample()`
# before any sampling happens and depend on neither the model shape nor the
# missing-data mode, so one cheap cell covers each and they are not slow.
ERROR_NAMES = "loglik_kind,backend,sampler,step"
ERROR_GRID = [
    ("analytical", None, "numpyro", "slice"),
    ("approx_differentiable", "pytensor", "numpyro", "slice"),
    ("approx_differentiable", "jax", "numpyro", "slice"),
    ("blackbox", None, "numpyro", "slice"),
    ("blackbox", None, "numpyro", None),
]


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, COVERING_ARRAY)
def test_missing_data_matrix(request, loglik_kind, backend, sampler, step, shape, mode):
    """Sample each covered (likelihood/backend, sampler/step, shape, mode) cell."""
    model = build_model(request, shape, mode, loglik_kind, backend)
    assert np.all(
        [val_ is None for key_, val_ in model.pymc_model.rvs_to_initial_values.items()]
    )
    sample_and_verify(model, sampler, step)


@pytest.mark.parametrize(ERROR_NAMES, ERROR_GRID)
def test_rejected_sampler_combinations(request, loglik_kind, backend, sampler, step):
    """Unsupported sampler/step combinations are rejected before sampling."""
    model = build_model(request, "simple", "cpn", loglik_kind, backend)
    with pytest.raises(ValueError):
        sample(model, sampler, step)


def test_deadline_requires_missing_data(request):
    """`deadline=True` without `missing_data=True` is rejected at construction.

    `tests/test_data_sanity.py::test_deadline_missing_data_false` covers the
    same error for the non-deadline case; this pins the deadline route, which
    censors RTs to -999.0 itself and so always trips the check.
    """
    data = prepare_data(request.getfixturevalue("data_ddm"), "opn")
    with pytest.raises(
        ValueError,
        match="Missing data provided as False. \n"
        "However, you have RTs of -999.0 in your dataset!",
    ):
        hssm.HSSM(
            data,
            loglik_kind="analytical",
            deadline=True,
            loglik_missing_data=NETWORKS["opn"],
        )


@pytest.mark.slow
def test_default_sampler_end_to_end(request):
    """The default sampling path works end to end for a missing-data model.

    Which sampler `sampler=None` resolves to is asserted without sampling in
    `test_mcmc.py::test_default_sampler_resolution`.
    """
    model = build_model(request, "simple", "cpn", "analytical", None)
    sample_and_verify(model, None, None)

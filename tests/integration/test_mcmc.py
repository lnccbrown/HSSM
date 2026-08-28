"""Integration tests for `HSSM.sample()` across likelihood/backend/sampler combos.

The sampling matrix used to be a full 3-way cross of (loglik_kind, backend) x
(sampler, step) x model shape, run as three near-identical 20-row grids. Every
failure mode these tests actually catch is pairwise -- a backend either supports
a sampler or it does not, a model shape either builds or it does not -- so the
cross is replaced by a pairwise covering array (`COVERING_ARRAY` below), and the
checks that never depended on the sampler at all were lifted out into their own
tests.
"""

from copy import deepcopy

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import hssm

from tests.integration.regression_priors import A_REG, V_REG

hssm.set_floatX("float32", update_jax=True)

# AF-TODO: Include more tests that use different link functions!


# Model shape -> (data fixture name, extra HSSM kwargs).
MODEL_SHAPES = {
    "simple": ("data_ddm", {}),
    "reg_v": ("data_ddm_reg", {"v": V_REG}),
    "reg_va": ("data_ddm_reg_va", {"v": V_REG, "a": A_REG}),
}


def build_model(request, shape, loglik_kind, backend):
    """Build an HSSM model of the given shape, pulling its data fixture lazily."""
    data_fixture, params = MODEL_SHAPES[shape]
    return hssm.HSSM(
        request.getfixturevalue(data_fixture),
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        # HSSM mutates the param dicts it is handed, so hand it a fresh copy.
        **deepcopy(params),
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
            sampler=sampler,
            cores=1,
            chains=1,
            tune=10,
            draws=10,
            progressbar=False,
        )


def sample_and_verify(model, sampler, step):
    """Sample, then check that recomputing the log-likelihood reproduces it."""
    sample(model, sampler, step)
    assert isinstance(model.traces, xr.DataTree)

    # make sure log_likelihood computations check out
    traces_copy = deepcopy(model.traces)
    del traces_copy["log_likelihood"]

    # recomputing log-likelihood yields same results?
    model.log_likelihood(traces_copy, inplace=True)
    assert isinstance(traces_copy, xr.DataTree)
    assert "log_likelihood" in traces_copy
    for group_ in traces_copy:
        xr.testing.assert_equal(traces_copy[group_], model.traces[group_])


# Pairwise covering array over three factors:
#   B = (loglik_kind, backend): analytical / approx+pytensor / approx+jax / blackbox
#   S = (sampler, step):        pymc / pymc+slice / numpyro
#   M = model shape:            simple / reg_v / reg_va
#
# 12 rows cover every valid (B, S) pair, every (B, M) pair and every (S, M) pair
# -- rows 1-9 are an exact Latin square, and blackbox (which has no valid numpyro
# cell) repeats `pymc` to reach all three shapes. Because the 12 rows hit each
# (backend, shape) cell exactly once, the log_likelihood recompute check in
# `sample_and_verify` still gets full coverage while running 12 times instead of 45.
#
# approx/jax + pymc+slice is deliberately paired with `simple`: PyMC's
# Python-level Slice sampler calls the JAX logp once per variable per draw, which
# costs ~2.6s on `simple` but ~198s on `reg_va` (75x, from 4 extra scalar RVs).
MATRIX_NAMES = "loglik_kind,backend,sampler,step,shape"
COVERING_ARRAY = [
    ("analytical", None, "pymc", None, "simple"),
    ("analytical", None, "pymc", "slice", "reg_v"),
    ("analytical", None, "numpyro", None, "reg_va"),
    ("approx_differentiable", "pytensor", "pymc", None, "reg_v"),
    ("approx_differentiable", "pytensor", "pymc", "slice", "reg_va"),
    ("approx_differentiable", "pytensor", "numpyro", None, "simple"),
    ("approx_differentiable", "jax", "pymc", None, "reg_va"),
    ("approx_differentiable", "jax", "pymc", "slice", "simple"),
    ("approx_differentiable", "jax", "numpyro", None, "reg_v"),
    ("blackbox", None, "pymc", None, "simple"),
    ("blackbox", None, "pymc", "slice", "reg_v"),
    ("blackbox", None, "pymc", None, "reg_va"),
]

# Rejected combinations, all of which raise ValueError. These raise in `sample()`
# before any sampling happens and do not depend on the model shape, so one cheap
# model shape covers them and they are not marked slow.
ERROR_NAMES = "loglik_kind,backend,sampler,step"
ERROR_GRID = [
    ("analytical", None, "numpyro", "slice"),
    ("approx_differentiable", "pytensor", "numpyro", "slice"),
    ("approx_differentiable", "jax", "numpyro", "slice"),
    ("blackbox", None, "numpyro", "slice"),
    ("blackbox", None, "numpyro", None),
]

# `sampler=None` resolves to an explicit sampler in `HSSMBase.sample`; these are
# the expected resolutions, asserted without sampling.
DEFAULT_SAMPLER_NAMES = "loglik_kind,backend,expected_sampler"
DEFAULT_SAMPLER_GRID = [
    ("analytical", None, "pymc"),
    ("approx_differentiable", "pytensor", "pymc"),
    ("approx_differentiable", "jax", "numpyro"),
    ("blackbox", None, "pymc"),
]


@pytest.mark.slow
@pytest.mark.parametrize(MATRIX_NAMES, COVERING_ARRAY)
def test_mcmc_matrix(request, loglik_kind, backend, sampler, step, shape):
    """Sample each covered (likelihood/backend, sampler/step, model shape) cell."""
    model = build_model(request, shape, loglik_kind, backend)
    assert np.all(
        [val_ is None for key_, val_ in model.pymc_model.rvs_to_initial_values.items()]
    )
    sample_and_verify(model, sampler, step)


@pytest.mark.parametrize(ERROR_NAMES, ERROR_GRID)
def test_rejected_sampler_combinations(data_ddm, loglik_kind, backend, sampler, step):
    """Unsupported sampler/step combinations are rejected before sampling."""
    model = hssm.HSSM(
        data_ddm, loglik_kind=loglik_kind, model_config={"backend": backend}
    )
    with pytest.raises(ValueError):
        sample(model, sampler, step)


class _FitCalled(Exception):
    """Sentinel raised in place of an actual `bambi.Model.fit` call."""


@pytest.mark.parametrize(DEFAULT_SAMPLER_NAMES, DEFAULT_SAMPLER_GRID)
def test_default_sampler_resolution(
    data_ddm, monkeypatch, loglik_kind, backend, expected_sampler
):
    """`sampler=None` resolves to the right sampler for each likelihood/backend."""
    model = hssm.HSSM(
        data_ddm, loglik_kind=loglik_kind, model_config={"backend": backend}
    )

    captured = {}

    def fake_fit(**kwargs):
        captured["inference_method"] = kwargs["inference_method"]
        raise _FitCalled

    monkeypatch.setattr(model.model, "fit", fake_fit)

    with pytest.raises(_FitCalled):
        sample(model, None, None)

    assert captured["inference_method"] == expected_sampler


def test_default_blackbox_slice_uses_cvm(data_ddm, monkeypatch):
    """Black-box callbacks bypass Numba's closure-serialization path."""
    model = hssm.HSSM(data_ddm, loglik_kind="blackbox")
    captured = {}
    sentinel_step = object()

    def fake_slice(*, model, compile_kwargs):
        captured["slice_model"] = model
        captured["compile_kwargs"] = compile_kwargs
        return sentinel_step

    def fake_fit(**kwargs):
        captured["step"] = kwargs["step"]
        raise _FitCalled

    monkeypatch.setattr(pm, "Slice", fake_slice)
    monkeypatch.setattr(model.model, "fit", fake_fit)

    with pytest.raises(_FitCalled):
        model.sample()

    assert captured["slice_model"] is model.pymc_model
    assert captured["compile_kwargs"] == {"mode": "cvm"}
    assert captured["step"] is sentinel_step


@pytest.mark.slow
def test_default_sampler_end_to_end(data_ddm):
    """The default sampling path works end to end, not just in resolution."""
    model = hssm.HSSM(data_ddm, loglik_kind="analytical")
    sample_and_verify(model, None, None)


@pytest.fixture(scope="module")
def fitted_analytical(request):
    """Return an analytical model of the requested shape, sampled with defaults.

    Module-scoped and shape-parametrized so the post-processing tests below share
    one fit per shape instead of re-sampling for every assertion.
    """
    model = build_model(request, request.param, "analytical", None)
    sample(model, None, None)
    return model


@pytest.mark.slow
@pytest.mark.parametrize("fitted_analytical", ["simple"], indirect=True)
def test_post_processing_simple(fitted_analytical):
    """Traces of a non-regression model omit the trial-wise parent parameter."""
    model = fitted_analytical

    # Traces should be post-processed to NOT include
    # the trial wise parameters if they are not
    # associated with an actual regression
    deterministics = model._get_deterministic_var_names(model.traces)
    assert f"~{model._parent}_mean" not in deterministics
    # test summary:
    summary = az.summary(model.traces)
    assert summary.shape[0] == 4

    az.plot_trace_dist(model.traces)
    fig = plt.gcf()
    assert len(fig.axes) // 2 == 4


@pytest.mark.slow
@pytest.mark.parametrize("fitted_analytical", ["reg_v"], indirect=True)
def test_post_processing_reg(fitted_analytical):
    """Traces of a v-regression model keep `v` as a deterministic."""
    model = fitted_analytical

    assert model._get_deterministic_var_names(model.traces) == ["~v"]
    # test summary:
    summary = az.summary(model.traces)
    assert summary.shape[0] == 6

    az.plot_trace_dist(model.traces)
    fig = plt.gcf()
    assert len(fig.axes) // 2 == 6


@pytest.mark.slow
@pytest.mark.parametrize("fitted_analytical", ["reg_va"], indirect=True)
def test_post_processing_reg_v_a(fitted_analytical):
    """`var_names` filtering behaves as expected with two regressions."""
    model = fitted_analytical

    assert len(model._get_deterministic_var_names(model.traces)) == len(["~a", "~v"])
    assert set(model._get_deterministic_var_names(model.traces)) == set(["~a", "~v"])
    # test summary:
    summary = az.summary(model.traces)
    assert summary.shape[0] == 8

    summary = az.summary(model.traces, var_names=["~a"])
    assert summary.shape[0] == 8

    summary = az.summary(model.traces, var_names=["~t"])
    assert summary.shape[0] == 7

    summary = az.summary(model.traces, var_names=["~a", "~t"])
    assert summary.shape[0] == 7

    az.plot_trace_dist(model.traces)
    fig = plt.gcf()
    assert len(fig.axes) // 2 == 8

    az.plot_trace_dist(model.traces, var_names=["~a"])
    fig = plt.gcf()
    assert len(fig.axes) // 2 == 8

    az.plot_trace_dist(model.traces, var_names=["~t"])
    fig = plt.gcf()
    assert len(fig.axes) // 2 == 7

    az.plot_trace_dist(model.traces, var_names=["~a", "~t"])
    fig = plt.gcf()
    assert len(fig.axes) // 2 == 7


# Basic tests for LBA likelihood
@pytest.mark.slow
def test_lba_sampling():
    """Test if sampling works for available lba models."""
    lba2_data_out = hssm.simulate_data(
        model="lba2", theta=dict(A=0.2, b=0.5, v0=1.0, v1=1.0), size=500
    )

    lba3_data_out = hssm.simulate_data(
        model="lba3", theta=dict(A=0.2, b=0.5, v0=1.0, v1=1.0, v2=1.0), size=500
    )

    lba2_model = hssm.HSSM(model="lba2", data=lba2_data_out)

    lba3_model = hssm.HSSM(model="lba3", data=lba3_data_out)

    lba4_data = pd.DataFrame(
        [[rt, choice] for rt in (0.35, 0.50) for choice in range(4)],
        columns=["rt", "response"],
    )
    lba4_model = hssm.HSSM(model="lba4", data=lba4_data, p_outlier=0)

    traces_2 = lba2_model.sample(sampler="numpyro", draws=10, tune=10, chains=1)
    traces_3 = lba3_model.sample(sampler="numpyro", draws=10, tune=10, chains=1)
    traces_4 = lba4_model.sample(
        sampler="numpyro",
        draws=10,
        tune=10,
        chains=1,
        cores=1,
        progressbar=False,
    )

    assert isinstance(traces_2, xr.DataTree)
    assert isinstance(traces_3, xr.DataTree)
    assert isinstance(traces_4, xr.DataTree)

import pytest

import arviz as az
import hssm
import numpy as np
import pymc as pm
from copy import deepcopy
import xarray as xr

from hssm.utils import _rearrange_data

hssm.set_floatX("float32", update_jax=True)


@pytest.fixture
def data_ddm_missing(data_ddm):
    data = data_ddm.copy()
    missing_indices = np.random.choice(data_ddm.shape[0], 50, replace=False)
    data.iloc[missing_indices, 0] = -999.0

    return _rearrange_data(data)


@pytest.fixture
def data_ddm_reg_missing(data_ddm_reg):
    data = data_ddm_reg.copy()
    missing_indices = np.random.choice(data_ddm_reg.shape[0], 50, replace=False)
    data.iloc[missing_indices, 0] = -999.0

    return _rearrange_data(data)


@pytest.fixture
def fixture_path():
    from pathlib import Path

    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def cpn(fixture_path):
    return fixture_path / "ddm_cpn.onnx"


PARAMETER_NAMES = "loglik_kind,backend,sampler,step,expected"
PARAMETER_GRID = [
    ("analytical", None, None, None, True),  # Defaults should work
    ("analytical", None, "mcmc", None, True),
    ("analytical", None, "mcmc", "slice", True),
    ("analytical", None, "nuts_numpyro", None, True),
    ("analytical", None, "nuts_numpyro", "slice", ValueError),
    ("approx_differentiable", "pytensor", None, None, True),  # Defaults should work
    ("approx_differentiable", "pytensor", "mcmc", None, True),
    ("approx_differentiable", "pytensor", "mcmc", "slice", True),
    ("approx_differentiable", "pytensor", "nuts_numpyro", None, True),
    ("approx_differentiable", "pytensor", "nuts_numpyro", "slice", ValueError),
    ("approx_differentiable", "jax", None, None, True),  # Defaults should work
    ("approx_differentiable", "jax", "mcmc", None, True),
    ("approx_differentiable", "jax", "mcmc", "slice", True),
    ("approx_differentiable", "jax", "nuts_numpyro", None, True),
    ("approx_differentiable", "jax", "nuts_numpyro", "slice", ValueError),
    ("blackbox", None, None, None, True),  # Defaults should work
    ("blackbox", None, "mcmc", None, True),
    ("blackbox", None, "mcmc", "slice", True),
    ("blackbox", None, "nuts_numpyro", None, ValueError),
    ("blackbox", None, "nuts_numpyro", "slice", ValueError),
]


def sample(model, sampler, step):
    if step == "slice":
        model.sample(
            sampler=sampler,
            cores=1,
            chains=1,
            tune=10,
            draws=10,
            step=pm.Slice(model=model.pymc_model),
        )
    else:
        model.sample(
            sampler=sampler,
            cores=1,
            chains=1,
            tune=10,
            draws=10,
        )


def run_sample(model, sampler, step, expected):
    if expected is True:
        sample(model, sampler, step)
        assert isinstance(model.traces, az.InferenceData)

        traces_copy = deepcopy(model.traces)
        del traces_copy["log_likelihood"]

        model.log_likelihood(traces_copy, inplace=True)
        assert isinstance(traces_copy, az.InferenceData)
        assert "log_likelihood" in traces_copy.groups()
        for group_ in traces_copy.groups():
            xr.testing.assert_equal(traces_copy[group_], model.traces[group_])
    else:
        with pytest.raises(expected):
            sample(model, sampler, step)


# AF-TODO: CPN / GONOGO part has to be rethought
@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
# @pytest.mark.xfail(reason="Needs to be reactivated, CPN logic needs to be revised")
def test_simple_models_missing_data(
    data_ddm_missing, loglik_kind, backend, sampler, step, expected, cpn
):
    model = hssm.HSSM(
        data_ddm_missing,
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        missing_data=True,
        loglik_missing_data=cpn,
        deadline=False,
    )
    assert np.all(
        [val_ is None for key_, val_ in model.pymc_model.rvs_to_initial_values.items()]
    )
    run_sample(model, sampler, step, expected)


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
# @pytest.mark.xfail(reason="Needs to be reactivated, CPN logic needs to be revised")
def test_reg_models_missing_data(
    data_ddm_reg_missing, loglik_kind, backend, sampler, step, expected, cpn
):
    param_reg = dict(
        formula="v ~ 1 + x + y",
        prior={
            "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
            "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
        },
    )
    model = hssm.HSSM(
        data_ddm_reg_missing,
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        v=param_reg,
        missing_data=True,
        loglik_missing_data=cpn,
        deadline=False,
    )
    assert np.all(
        [val_ is None for key_, val_ in model.pymc_model.rvs_to_initial_values.items()]
    )
    run_sample(model, sampler, step, expected)

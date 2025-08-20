import pytest

import arviz as az
import matplotlib.pyplot as plt
import hssm
import numpy as np
import pymc as pm
from copy import deepcopy
import xarray as xr


hssm.set_floatX("float32", update_jax=True)

# AF-TODO: Include more tests that use different link functions!


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
    """Run the sample function and check if the expected error is raised."""
    if expected is True:
        sample(model, sampler, step)
        assert isinstance(model.traces, az.InferenceData)

        # make sure log_likelihood computations check out
        traces_copy = deepcopy(model.traces)
        del traces_copy["log_likelihood"]

        # recomputing log-likelihood yields same results?
        model.log_likelihood(traces_copy, inplace=True)
        assert isinstance(traces_copy, az.InferenceData)
        assert "log_likelihood" in traces_copy.groups()
        for group_ in traces_copy.groups():
            xr.testing.assert_equal(traces_copy[group_], model.traces[group_])

    else:
        with pytest.raises(expected):
            sample(model, sampler, step)


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

    traces_2 = lba2_model.sample(sampler="nuts_numpyro", draws=100, tune=100, chains=1)
    traces_3 = lba3_model.sample(sampler="nuts_numpyro", draws=100, tune=100, chains=1)

    assert isinstance(traces_2, az.InferenceData)
    assert isinstance(traces_3, az.InferenceData)


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_simple_models(data_ddm, loglik_kind, backend, sampler, step, expected):
    """Test simple models."""
    print("PYMC VERSION: ")
    print(pm.__version__)
    print("TEST INPUTS WERE: ")
    print("REPORTING FROM SIMPLE MODELS TEST")
    print(loglik_kind, backend, sampler, step, expected)

    model = hssm.HSSM(
        data_ddm, loglik_kind=loglik_kind, model_config={"backend": backend}
    )
    assert np.all(
        [val_ is None for key_, val_ in model.pymc_model.rvs_to_initial_values.items()]
    )
    run_sample(model, sampler, step, expected)

    # Only runs once
    if loglik_kind == "analytical" and sampler is None:
        # Traces should be post-processed to NOT include
        # the trial wise parameters if they are not
        # associated with an actual regression
        assert not (
            f"~{model._parent}_mean" in model._get_deterministic_var_names(model.traces)
        )
        # test summary:
        summary = model.summary()
        assert summary.shape[0] == 4

        model.plot_trace(show=False)
        fig = plt.gcf()
        assert len(fig.axes) // 2 == 4


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_reg_models(data_ddm_reg, loglik_kind, backend, sampler, step, expected):
    """Test regression models."""
    print("PYMC VERSION: ")
    print(pm.__version__)
    print("TEST INPUTS WERE: ")
    print("REPORTING FROM REG MODELS TEST")
    print(loglik_kind, backend, sampler, step, expected)

    param_reg = dict(
        formula="v ~ 1 + x + y",
        prior={
            "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
            "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
        },
    )
    model = hssm.HSSM(
        data_ddm_reg,
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        v=param_reg,
    )
    assert np.all(
        [val_ is None for key_, val_ in model.pymc_model.rvs_to_initial_values.items()]
    )
    run_sample(model, sampler, step, expected)

    # Only runs once
    if loglik_kind == "analytical" and sampler is None:
        assert model._get_deterministic_var_names(model.traces) == ["~v"]
        # test summary:
        summary = model.summary()
        assert summary.shape[0] == 6

        model.plot_trace(show=False)
        fig = plt.gcf()
        assert len(fig.axes) // 2 == 6


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_reg_models_v_a(data_ddm_reg_va, loglik_kind, backend, sampler, step, expected):
    """Test regression models with multiple parameters (v, a)."""
    print("PYMC VERSION: ")
    print(pm.__version__)
    print("TEST INPUTS WERE: ")
    print("REPORTING FROM REG MODELS V_A TEST")
    print(loglik_kind, backend, sampler, step, expected)
    param_reg_v = dict(
        formula="v ~ 1 + x + y",
        prior={
            "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
            "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
        },
    )
    param_reg_a = dict(
        formula="a ~ 1 + m + n",
        prior={
            "Intercept": {
                "name": "Normal",
                "mu": 1.0,
                "sigma": 0.5,
            },
            "m": {"name": "Uniform", "lower": 0.0, "upper": 0.2},
            "n": {"name": "Uniform", "lower": 0.0, "upper": 0.2},
        },
        link="identity",
    )

    model = hssm.HSSM(
        data_ddm_reg_va,
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        v=param_reg_v,
        a=param_reg_a,
    )
    assert np.all(
        [val_ is None for key_, val_ in model.pymc_model.rvs_to_initial_values.items()]
    )
    print(model.params["a"])
    run_sample(model, sampler, step, expected)

    # Only runs once
    if loglik_kind == "analytical" and sampler is None:
        assert len(model._get_deterministic_var_names(model.traces)) == len(
            ["~a", "~v"]
        )
        assert set(model._get_deterministic_var_names(model.traces)) == set(
            ["~a", "~v"]
        )
        # test summary:
        summary = model.summary()
        assert summary.shape[0] == 8

        summary = model.summary(var_names=["~a"])
        assert summary.shape[0] == 8

        summary = model.summary(var_names=["~t"])
        assert summary.shape[0] == 7

        summary = model.summary(var_names=["~a", "~t"])
        assert summary.shape[0] == 7

        model.plot_trace(show=False)
        fig = plt.gcf()
        assert len(fig.axes) // 2 == 8

        model.plot_trace(show=False, var_names=["~a"])
        fig = plt.gcf()
        assert len(fig.axes) // 2 == 8

        model.plot_trace(show=False, var_names=["~t"])
        fig = plt.gcf()
        assert len(fig.axes) // 2 == 7

        model.plot_trace(show=False, var_names=["~a", "~t"])
        fig = plt.gcf()
        assert len(fig.axes) // 2 == 7

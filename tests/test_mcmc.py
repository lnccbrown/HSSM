import pytest

import arviz as az
import matplotlib.pyplot as plt
import hssm
import pymc as pm

hssm.set_floatX("float32")

parameter_names = "loglik_kind,backend,sampler,step,expected"
parameter_grid = [
    ("analytical", None, None, None, True),  # Defaults should work
    ("analytical", None, "mcmc", None, True),
    ("analytical", None, "mcmc", "slice", True),
    ("analytical", None, "nuts_numpyro", None, True),
    ("analytical", None, "nuts_numpyro", "slice", TypeError),
    ("approx_differentiable", "pytensor", None, None, True),  # Defaults should work
    ("approx_differentiable", "pytensor", "mcmc", None, True),
    ("approx_differentiable", "pytensor", "mcmc", "slice", True),
    ("approx_differentiable", "pytensor", "nuts_numpyro", None, True),
    ("approx_differentiable", "pytensor", "nuts_numpyro", "slice", TypeError),
    ("approx_differentiable", "jax", None, None, True),  # Defaults should work
    ("approx_differentiable", "jax", "mcmc", None, True),
    ("approx_differentiable", "jax", "mcmc", "slice", True),
    ("approx_differentiable", "jax", "nuts_numpyro", None, True),
    ("approx_differentiable", "jax", "nuts_numpyro", "slice", TypeError),
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
    if expected == True:
        sample(model, sampler, step)
        assert isinstance(model.traces, az.InferenceData)
    else:
        with pytest.raises(expected):
            sample(model, sampler, step)


@pytest.mark.parametrize(parameter_names, parameter_grid)
def test_simple_models(data_ddm, loglik_kind, backend, sampler, step, expected):
    model = hssm.HSSM(
        data_ddm, loglik_kind=loglik_kind, model_config={"backend": backend}
    )
    run_sample(model, sampler, step, expected)

    # Only runs once
    if loglik_kind == "analytical" and sampler is None:
        assert not model._get_deterministic_var_names(model.traces)
        # test summary:
        summary = model.summary()
        assert summary.shape[0] == 4

        model.plot_trace(show=False)
        fig = plt.gcf()
        assert len(fig.axes) // 2 == 4


@pytest.mark.parametrize(parameter_names, parameter_grid)
def test_reg_models(data_ddm_reg, loglik_kind, backend, sampler, step, expected):
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
    run_sample(model, sampler, step, expected)

    # Only runs once
    if loglik_kind == "analytical" and sampler is None:
        assert not model._get_deterministic_var_names(model.traces)
        # test summary:
        summary = model.summary()
        assert summary.shape[0] == 6

        model.plot_trace(show=False)
        fig = plt.gcf()
        assert len(fig.axes) // 2 == 6


@pytest.mark.parametrize(parameter_names, parameter_grid)
def test_reg_models_v_a(data_ddm_reg, loglik_kind, backend, sampler, step, expected):
    param_reg_v = dict(
        formula="v ~ 1 + x + y",
        prior={
            "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
            "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
        },
    )
    param_reg_a = dict(
        formula="v ~ 1 + x + y",
        prior={
            "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
            "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
        },
        link="log",
    )

    model = hssm.HSSM(
        data_ddm_reg,
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        v=param_reg_v,
        a=param_reg_a,
    )
    run_sample(model, sampler, step, expected)

    # Only runs once
    if loglik_kind == "analytical" and sampler is None:
        assert model._get_deterministic_var_names(model.traces) == ["~a"]
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

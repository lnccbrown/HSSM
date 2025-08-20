import pytest

import arviz as az
import hssm
import pymc as pm

hssm.set_floatX("float32", update_jax=True)

# AF-TODO: Include more tests that use different link functions!


PARAMETER_NAMES = "loglik_kind, backend, method, expected"
PARAMETER_GRID = [
    # ("analytical", "pytensor", "advi", True),  # TODO: Gradients still broken
    # ("analytical", "pytensor", "fullrank_advi", True), # TODO Gradients still broken
    ("blackbox", "pytensor", "advi", ValueError),  # Defaults should work
    ("blackbox", "pytensor", "fullrank_advi", ValueError),
    ("approx_differentiable", "pytensor", "advi", True),  # Defaults should work
    ("approx_differentiable", "pytensor", "fullrank_advi", True),
    ("approx_differentiable", "jax", "advi", True),  # Defaults should work
    ("approx_differentiable", "jax", "fullrank_advi", True),
]


def run_vi(model, method, expected):
    if expected == True:
        model.vi(method, niter=1000)
        assert isinstance(model.vi_idata, az.InferenceData)
        assert isinstance(model.vi_approx, pm.Approximation)
    else:
        with pytest.raises(expected):
            model.vi(method, niter=1000)


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_simple_models(data_ddm, loglik_kind, backend, method, expected):
    print("PYMC VERSION: ")
    print(pm.__version__)
    print("TEST INPUTS WERE: ")
    print("REPORTING FROM SIMPLE MODELS TEST")
    print(loglik_kind, backend, method, expected)

    model = hssm.HSSM(
        data_ddm, loglik_kind=loglik_kind, model_config={"backend": backend}
    )

    run_vi(model, method, expected)


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_reg_models(data_ddm_reg, loglik_kind, backend, method, expected):
    print("PYMC VERSION: ")
    print(pm.__version__)
    print("TEST INPUTS WERE: ")
    print("REPORTING FROM REG MODELS TEST")
    print(loglik_kind, backend, method, expected)

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
    run_vi(model, method, expected)


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_reg_models_v_a(data_ddm_reg_va, loglik_kind, backend, method, expected):
    print("PYMC VERSION: ")
    print(pm.__version__)
    print("TEST INPUTS WERE: ")
    print("REPORTING FROM REG MODELS V_A TEST")
    print(loglik_kind, backend, method, expected)
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
    run_vi(model, method, expected)

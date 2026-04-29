import pytest

import arviz as az
import hssm
import numpy as np
import pandas as pd
import pymc as pm


hssm.set_floatX("float32", update_jax=True)

PARAMETER_NAMES = "loglik_kind,sampler,step,backend,expected"
PARAMETER_GRID = [
    ("analytical", None, None, "pytensor", True),  # Defaults should work
    ("analytical", "pymc", None, "pytensor", True),
    ("analytical", "pymc", "slice", "pytensor", True),
    ("analytical", None, None, "jax", True),  # Defaults should work
    ("analytical", "pymc", None, "jax", True),
    ("analytical", "pymc", "slice", "jax", True),
    ("analytical", "numpyro", None, "jax", True),
    ("analytical", "numpyro", None, "pytensor", ValueError),
    ("analytical", "numpyro", "slice", "jax", ValueError),
]


@pytest.fixture(scope="module")
def generate_synthetic_data():
    """Generate synthetic dataset for testing."""
    n_samples = 100
    data = pd.DataFrame(
        {
            "response": np.random.choice([-1, 1], size=n_samples),
            "x": np.random.randn(n_samples),
        }
    )
    return data


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_choice_only_default_params(
    loglik_kind, sampler, step, backend, expected, generate_synthetic_data
):
    """Test choice-only model configuration."""
    model = hssm.HSSM(
        data=generate_synthetic_data,
        model="softmax_inv_temperature_2",
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
    )

    if expected is ValueError:
        with pytest.raises(ValueError):
            model.sample(
                sampler=sampler,
                step=pm.Slice(model=model.pymc_model),
                cores=1,
                chains=1,
                tune=10,
                draws=10,
            )

        return

    if step is None:
        idata = model.sample(
            sampler=sampler,
            cores=1,
            chains=1,
            tune=10,
            draws=10,
        )
    else:
        idata = model.sample(
            sampler=sampler,
            step=pm.Slice(model=model.pymc_model),
            cores=1,
            chains=1,
            tune=10,
            draws=10,
        )

    assert isinstance(idata, az.InferenceData)


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_choice_only_beta_reg(
    loglik_kind, sampler, step, backend, expected, generate_synthetic_data
):
    """Test choice-only model configuration."""
    model = hssm.HSSM(
        data=generate_synthetic_data,
        model="softmax_inv_temperature_2",
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        beta=dict(
            formula="beta ~ x",
            prior={
                "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
                "x": {"name": "Normal", "mu": 0, "sigma": 1},
            },
        ),
    )

    if expected is ValueError:
        with pytest.raises(ValueError):
            model.sample(
                sampler=sampler,
                step=pm.Slice(model=model.pymc_model),
                cores=1,
                chains=1,
                tune=10,
                draws=10,
            )

        return

    if step is None:
        idata = model.sample(
            sampler=sampler,
            cores=1,
            chains=1,
            tune=10,
            draws=10,
        )
    else:
        idata = model.sample(
            sampler=sampler,
            step=pm.Slice(model=model.pymc_model),
            cores=1,
            chains=1,
            tune=10,
            draws=10,
        )

    assert isinstance(idata, az.InferenceData)


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_choice_only_logit_reg(
    loglik_kind, sampler, step, backend, expected, generate_synthetic_data
):
    """Test choice-only model configuration."""
    model = hssm.HSSM(
        data=generate_synthetic_data,
        model="softmax_inv_temperature_2",
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        logit1=dict(
            formula="logit1 ~ x",
            prior={
                "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
                "x": {"name": "Normal", "mu": 0, "sigma": 1},
            },
        ),
    )

    if expected is ValueError:
        with pytest.raises(ValueError):
            model.sample(
                sampler=sampler,
                step=pm.Slice(model=model.pymc_model),
                cores=1,
                chains=1,
                tune=10,
                draws=10,
            )

        return

    if step is None:
        idata = model.sample(
            sampler=sampler,
            cores=1,
            chains=1,
            tune=10,
            draws=10,
        )
    else:
        idata = model.sample(
            sampler=sampler,
            step=pm.Slice(model=model.pymc_model),
            cores=1,
            chains=1,
            tune=10,
            draws=10,
        )

    assert isinstance(idata, az.InferenceData)


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_choice_only_multiple_reg(
    loglik_kind, sampler, step, backend, expected, generate_synthetic_data
):
    """Test choice-only model configuration."""
    model = hssm.HSSM(
        data=generate_synthetic_data,
        model="softmax_inv_temperature_3",
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        beta=dict(
            formula="beta ~ x",
            prior={
                "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
                "x": {"name": "Normal", "mu": 0, "sigma": 1},
            },
        ),
        logit1=dict(
            formula="logit1 ~ x",
            prior={
                "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
                "x": {"name": "Normal", "mu": 0, "sigma": 1},
            },
        ),
    )

    if expected is ValueError:
        with pytest.raises(ValueError):
            model.sample(
                sampler=sampler,
                step=pm.Slice(model=model.pymc_model),
                cores=1,
                chains=1,
                tune=10,
                draws=10,
            )

        return

    if step is None:
        idata = model.sample(
            sampler=sampler,
            cores=1,
            chains=1,
            tune=10,
            draws=10,
        )
    else:
        idata = model.sample(
            sampler=sampler,
            step=pm.Slice(model=model.pymc_model),
            cores=1,
            chains=1,
            tune=10,
            draws=10,
        )

    assert isinstance(idata, az.InferenceData)

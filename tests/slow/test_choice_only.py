import pytest

import arviz as az
import hssm
import numpy as np
import pandas as pd
import pymc as pm


hssm.set_floatX("float32", update_jax=True)

PARAMETER_NAMES = "loglik_kind,sampler,step,expected"
PARAMETER_GRID = [
    ("analytical", None, None, True),  # Defaults should work
    ("analytical", "pymc", None, True),
    ("analytical", "pymc", "slice", True),
    ("analytical", "numpyro", None, True),
    ("analytical", "numpyro", "slice", ValueError),
]


@pytest.fixture(scope="module")
def generate_synthetic_data():
    """Generate synthetic dataset for testing."""
    n_samples = 100
    data = pd.DataFrame(
        {
            "response": np.random.choice([-1, 1], size=n_samples),
        }
    )
    return data


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_choice_only(loglik_kind, sampler, step, expected, generate_synthetic_data):
    """Test choice-only model configuration."""
    model = hssm.HSSM(
        data=generate_synthetic_data,
        model="softmax_inv_temperature_2",
        loglik_kind=loglik_kind,
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

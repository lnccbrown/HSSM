import pytest

import arviz as az
import hssm
import numpy as np
import pandas as pd
import pymc as pm

from hssm.likelihoods.analytical import softmax_inv_temperature


hssm.set_floatX("float32", update_jax=True)

# ---------------------------------------------------------------------------
# Tests for softmax_inv_temperature
# ---------------------------------------------------------------------------

_N = 10
_rng = np.random.default_rng(42)

# data is always a 1-D float32 vector
_DATA_BINARY = _rng.choice([-1, 1], size=_N).astype(np.float32)  # 2-choice
_DATA_TERNARY = _rng.choice([0, 1, 2], size=_N).astype(np.float32)  # 3-choice

_SCALAR_BETA = np.float32(1.5)
_VECTOR_BETA = np.full(_N, 1.5, dtype=np.float32)

_SCALAR_LOGIT = np.float32(0.5)
_VECTOR_LOGIT = np.full(_N, 0.5, dtype=np.float32)


@pytest.mark.parametrize(
    "beta", [_SCALAR_BETA, _VECTOR_BETA], ids=["scalar_beta", "vector_beta"]
)
@pytest.mark.parametrize(
    "logit", [_SCALAR_LOGIT, _VECTOR_LOGIT], ids=["scalar_logit", "vector_logit"]
)
def test_softmax_inv_temperature_shape_2choice(beta, logit):
    """Output is a length-n 1-D array for all scalar/vector combos (2-choice)."""
    result = softmax_inv_temperature(_DATA_BINARY, beta, logit)
    evaluated = result.eval()
    assert evaluated.shape == (_N,)


@pytest.mark.parametrize(
    "beta", [_SCALAR_BETA, _VECTOR_BETA], ids=["scalar_beta", "vector_beta"]
)
@pytest.mark.parametrize(
    "logit1", [_SCALAR_LOGIT, _VECTOR_LOGIT], ids=["scalar_logit1", "vector_logit1"]
)
@pytest.mark.parametrize(
    "logit2", [_SCALAR_LOGIT, _VECTOR_LOGIT], ids=["scalar_logit2", "vector_logit2"]
)
def test_softmax_inv_temperature_shape_3choice(beta, logit1, logit2):
    """Output is a length-n 1-D array for all scalar/vector combos (3-choice)."""
    result = softmax_inv_temperature(_DATA_TERNARY, beta, logit1, logit2)
    evaluated = result.eval()
    assert evaluated.shape == (_N,)


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

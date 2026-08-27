"""End-to-end construction tests for the JEAM model registration."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jeam")

import jax
import pytensor

import hssm
from hssm.integrations.jeam import simulate_circular_diffusion


@pytest.fixture(scope="module", autouse=True)
def _pin_fixed_cdm_x64():
    """Construct analytical models in JEAM's supported precision mode."""
    original_floatx = pytensor.config.floatX
    original_jax_x64 = jax.config.jax_enable_x64
    hssm.set_floatX("float64")
    yield
    pytensor.config.floatX = original_floatx
    jax.config.update("jax_enable_x64", original_jax_x64)


def test_simulated_circular_data_builds_an_ordinary_hssm_model():
    """The fixed CDM should use HSSM's standard config-driven lifecycle."""
    observations = simulate_circular_diffusion(
        theta=np.array([0.70, -0.35, 0.40, 0.08]),
        random_state=1947,
        n_replicas=12,
    )
    data = pd.DataFrame(observations, columns=["rt", "response"])

    model = hssm.HSSM(
        data=data,
        model="circular_diffusion",
        p_outlier=None,
    )

    assert type(model) is hssm.HSSM
    assert model.model_name == "circular_diffusion"
    assert model.response_kind == "circular"
    assert model.response_bounds == {"response": (-np.pi, np.pi)}
    assert model.choices is None
    assert model.n_choices is None
    assert model.list_params == ["v_x", "v_y", "a", "t"]
    assert model.model_config.rv is simulate_circular_diffusion


def test_differentiable_circular_data_builds_an_ordinary_hssm_model():
    """The opt-in JAX path should use the same class and config lifecycle."""
    observations = simulate_circular_diffusion(
        theta=np.array([0.70, -0.35, 0.40, 0.08]),
        random_state=1947,
        n_replicas=12,
    )
    data = pd.DataFrame(observations, columns=["rt", "response"])

    model = hssm.HSSM(
        data=data,
        model="circular_diffusion",
        loglik_kind="analytical",
        p_outlier=None,
    )

    assert type(model) is hssm.HSSM
    assert model.loglik_kind == "analytical"
    assert model.model_config.backend == "jax"
    assert model.response_kind == "circular"
    assert model.list_params == ["v_x", "v_y", "a", "t"]
    assert model.model_config.rv is simulate_circular_diffusion

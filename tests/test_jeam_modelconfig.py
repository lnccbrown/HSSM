"""Tests for the experimental built-in JEAM model configuration."""

import numpy as np
import pandas as pd
import pytest

import hssm
from hssm.config import Config
from hssm.integrations.jeam import (
    logp_circular_diffusion,
    logp_circular_diffusion_jax,
    simulate_circular_diffusion,
)
from hssm.modelconfig import get_default_model_config
from hssm.plotting.model_cartoon import _require_categorical_choices


def test_circular_diffusion_is_a_supported_model():
    """The prototype should be discoverable through the public model list."""
    assert "circular_diffusion" in hssm.list_models()


def test_get_circular_diffusion_config():
    """The raw built-in config should expose the complete model contract."""
    config = get_default_model_config("circular_diffusion")

    assert config["response"] == ["rt", "response"]
    assert config["response_kind"] == "circular"
    assert config["response_bounds"] == {"response": (-np.pi, np.pi)}
    assert config["choices"] is None
    assert config["list_params"] == ["v_x", "v_y", "a", "t"]
    assert config["rv"] is simulate_circular_diffusion
    assert config["default_loglik_kind"] == "blackbox"

    blackbox = config["likelihoods"]["blackbox"]
    assert blackbox["loglik"] is logp_circular_diffusion
    assert blackbox["backend"] is None
    assert blackbox["bounds"] == {
        "v_x": (-3.0, 3.0),
        "v_y": (-3.0, 3.0),
        "a": (0.1, 3.0),
        "t": (0.0, 2.0),
    }
    assert blackbox["default_priors"] == {"t": {"name": "HalfNormal", "sigma": 2.0}}
    assert blackbox["extra_fields"] is None

    analytical = config["likelihoods"]["analytical"]
    assert analytical["loglik"] is logp_circular_diffusion_jax
    assert analytical["backend"] == "jax"
    assert analytical["bounds"] == blackbox["bounds"]
    assert analytical["default_priors"] == blackbox["default_priors"]
    assert analytical["extra_fields"] is None


def test_circular_diffusion_from_defaults_preserves_domain_and_rv():
    """Lazy registration should produce a validated non-categorical Config."""
    config = Config.from_defaults("circular_diffusion", None)

    assert config.model_name == "circular_diffusion"
    assert config.loglik_kind == "blackbox"
    assert config.response == ["rt", "response"]
    assert config.response_kind == "circular"
    assert config.response_bounds == {"response": (-np.pi, np.pi)}
    assert config.choices is None
    assert config.list_params == ["v_x", "v_y", "a", "t"]
    assert config.loglik is logp_circular_diffusion
    assert config.rv is simulate_circular_diffusion
    config.validate()


def test_circular_diffusion_analytical_likelihood_is_explicitly_selectable():
    """The semi-analytical JAX path remains opt-in until its promotion gate passes."""
    config = Config.from_defaults("circular_diffusion", "analytical")

    assert config.loglik_kind == "analytical"
    assert config.backend == "jax"
    assert config.loglik is logp_circular_diffusion_jax
    assert config.rv is simulate_circular_diffusion
    config.validate()


@pytest.mark.parametrize("p_outlier", [0.0, 0.05])
def test_non_categorical_model_requires_explicitly_disabled_lapse(p_outlier):
    """Continuous-response density measures cannot use HSSM's categorical lapse."""
    data = pd.DataFrame({"rt": [0.4, 0.8], "response": [-0.2, 0.3]})

    with pytest.raises(
        ValueError,
        match=r"`p_outlier` is not supported.*Set `p_outlier=None`",
    ):
        hssm.HSSM(
            data=data,
            model="circular_diffusion",
            p_outlier=p_outlier,
        )


def test_model_cartoon_rejects_non_categorical_choices_directly():
    """Categorical-only plotting should fail before trying to enumerate angles."""
    with pytest.raises(ValueError, match="Model cartoons.*non-categorical responses"):
        _require_categorical_choices(None, "circular_diffusion")

"""Contracts for the experimental built-in fixed-PSDM model configuration."""

import numpy as np
import pandas as pd
import pytest

import hssm
from hssm.config import Config
from hssm.integrations.jeam import (
    logp_projected_spherical_diffusion,
    simulate_projected_spherical_diffusion,
)
from hssm.modelconfig import get_default_model_config

MODEL_NAME = "projected_spherical_diffusion"

pytestmark = pytest.mark.xfail(
    MODEL_NAME not in hssm.list_models(),
    reason="the fixed-PSDM built-in model is not registered yet",
    strict=True,
)


def test_projected_spherical_diffusion_is_a_supported_model():
    """The prototype should be discoverable through the public model list."""
    assert MODEL_NAME in hssm.list_models()


def test_get_projected_spherical_diffusion_config():
    """The raw built-in config should expose the complete fixed-model contract."""
    config = get_default_model_config(MODEL_NAME)

    assert config["response"] == ["rt", "response"]
    assert config["response_kind"] == "continuous"
    assert config["response_bounds"] == {"response": (0.0, np.pi)}
    assert config["choices"] is None
    assert config["list_params"] == ["v_x", "v_y", "a", "t"]
    assert config["rv"] is simulate_projected_spherical_diffusion
    assert config["default_loglik_kind"] == "blackbox"
    assert set(config["likelihoods"]) == {"blackbox"}

    blackbox = config["likelihoods"]["blackbox"]
    assert blackbox["loglik"] is logp_projected_spherical_diffusion
    assert blackbox["backend"] is None
    assert blackbox["bounds"] == {
        "v_x": (-3.0, 3.0),
        "v_y": (0.0, 3.0),
        "a": (0.1, 3.0),
        "t": (0.0, 2.0),
    }
    assert blackbox["default_priors"] == {"t": {"name": "HalfNormal", "sigma": 2.0}}
    assert blackbox["extra_fields"] is None


def test_projected_spherical_defaults_preserve_domain_and_rv():
    """Lazy registration should produce a validated continuous-response Config."""
    config = Config.from_defaults(MODEL_NAME, None)

    assert config.model_name == MODEL_NAME
    assert config.loglik_kind == "blackbox"
    assert config.backend is None
    assert config.response == ["rt", "response"]
    assert config.response_kind == "continuous"
    assert config.response_bounds == {"response": (0.0, np.pi)}
    assert config.choices is None
    assert config.list_params == ["v_x", "v_y", "a", "t"]
    assert config.loglik is logp_projected_spherical_diffusion
    assert config.rv is simulate_projected_spherical_diffusion
    config.validate()


@pytest.mark.parametrize("p_outlier", [0.0, 0.05])
def test_projected_spherical_model_requires_disabled_lapse(p_outlier):
    """A continuous coordinate density cannot use the categorical lapse model."""
    data = pd.DataFrame({"rt": [0.4, 0.8], "response": [0.2, 1.4]})

    with pytest.raises(
        ValueError,
        match=r"`p_outlier` is not supported.*Set `p_outlier=None`",
    ):
        hssm.HSSM(
            data=data,
            model=MODEL_NAME,
            p_outlier=p_outlier,
        )


@pytest.mark.parametrize("response", [0.0, np.pi])
def test_projected_spherical_config_accepts_both_polar_endpoints(response):
    """Continuous bounds should retain the model's closed polar support."""
    model = hssm.HSSM(
        data=pd.DataFrame({"rt": [0.4, 0.8], "response": [response, 1.4]}),
        model=MODEL_NAME,
        p_outlier=None,
    )

    assert model.response_bounds == {"response": (0.0, np.pi)}

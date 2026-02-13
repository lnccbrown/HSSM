import numpy as np
import pytest

from hssm.modelconfig import get_default_model_config
from hssm.modelconfig._softmax_inv_temperature_config import (
    softmax_inv_temperature_config,
)
import hssm


def test_get_ddm_sdv_config():
    ddm_sdv_model_config = get_default_model_config("ddm_sdv")
    assert ddm_sdv_model_config["response"] == ["rt", "response"]
    assert ddm_sdv_model_config["choices"] == [-1, 1]
    assert ddm_sdv_model_config["list_params"] == ["v", "a", "z", "t", "sv"]

    likelihoods = ddm_sdv_model_config["likelihoods"]
    lk_analytical = likelihoods["analytical"]
    assert lk_analytical["bounds"] == {
        "v": (-float("inf"), float("inf")),
        "a": (0.0, float("inf")),
        "z": (0.0, 1.0),
        "t": (0.0, float("inf")),
        "sv": (0.0, float("inf")),
    }

    assert lk_analytical["default_priors"] == {
        "t": {"name": "HalfNormal", "sigma": 2.0}
    }


def test_get_levy_config():
    levy_model_config = get_default_model_config("levy")
    assert levy_model_config["response"] == ["rt", "response"]
    assert levy_model_config["choices"] == [-1, 1]
    assert levy_model_config["list_params"] == ["v", "a", "z", "alpha", "t"]

    likelihoods = levy_model_config["likelihoods"]
    lk_approx_differentiable = likelihoods["approx_differentiable"]
    assert lk_approx_differentiable["bounds"] == {
        "v": (-3.0, 3.0),
        "a": (0.3, 3.0),
        "z": (0.1, 0.9),
        "alpha": (1.0, 2.0),
        "t": (1e-3, 2.0),
    }


def test_get_weibull_config():
    weibull_model_config = get_default_model_config("weibull")
    assert weibull_model_config["response"] == ["rt", "response"]
    assert weibull_model_config["choices"] == [-1, 1]
    assert weibull_model_config["list_params"] == ["v", "a", "z", "t", "alpha", "beta"]

    likelihood = weibull_model_config["likelihoods"]["approx_differentiable"]
    likelihood["loglik"] == "weibull.onnx"
    assert likelihood["bounds"] == {
        "v": (-2.5, 2.5),
        "a": (0.3, 2.5),
        "z": (0.2, 0.8),
        "t": (1e-3, 2.0),
        "alpha": (0.31, 4.99),
        "beta": (0.31, 6.99),
    }


def test_get_ddm_seq2_no_bias_config():
    ddm_seq2_no_bias_model_config = get_default_model_config("ddm_seq2_no_bias")
    assert ddm_seq2_no_bias_model_config["response"] == ["rt", "response"]
    assert ddm_seq2_no_bias_model_config["choices"] == [0, 1, 2, 3]
    assert ddm_seq2_no_bias_model_config["list_params"] == [
        "vh",
        "vl1",
        "vl2",
        "a",
        "t",
    ]

    likelihood = ddm_seq2_no_bias_model_config["likelihoods"]["approx_differentiable"]
    assert likelihood["loglik"] == "ddm_seq2_no_bias.onnx"
    assert likelihood["bounds"] == {
        "vh": (-4.0, 4.0),
        "vl1": (-4.0, 4.0),
        "vl2": (-4.0, 4.0),
        "a": (0.3, 2.5),
        "t": (0.0, 2.0),
    }


def test_get_ornstein_config():
    ornstein_model_config = get_default_model_config("ornstein")
    assert ornstein_model_config["response"] == ["rt", "response"]
    assert ornstein_model_config["choices"] == [-1, 1]
    assert ornstein_model_config["list_params"] == ["v", "a", "z", "g", "t"]

    likelihood = ornstein_model_config["likelihoods"]["approx_differentiable"]
    likelihood["loglik"] == "ornstein.onnx"
    assert likelihood["bounds"] == {
        "v": (-2.0, 2.0),
        "a": (0.3, 3.0),
        "z": (0.1, 0.9),
        "g": (-1.0, 1.0),
        "t": (1e-3, 2.0),
    }


@pytest.mark.parametrize("model", hssm.HSSM.supported_models)
def test_load_all_supported_model_configs(model):
    assert isinstance(get_default_model_config(model), dict)


def test_get_default_model_config_invalid():
    with pytest.raises(ValueError):
        get_default_model_config("invalid_model")


def test_inv_softmax_temperature_default():
    """Test inv_softmax_temperature with default n_logits=2."""
    config = softmax_inv_temperature_config()

    assert config["response"] == ["response"]
    assert config["choices"] == [-1, 1]
    assert config["list_params"] == ["beta", "logit1"]
    assert (
        config["description"] == "The Inverse Softmax Temperature Model with 2 logits"
    )

    likelihoods = config["likelihoods"]
    lk_analytical = likelihoods["analytical"]

    assert lk_analytical["backend"] is None
    assert lk_analytical["extra_fields"] is None

    # Test bounds
    assert lk_analytical["bounds"]["beta"] == (0.0, np.inf)
    assert lk_analytical["bounds"]["logit1"] == (-np.inf, np.inf)

    # Test default priors
    assert lk_analytical["default_priors"]["beta"] == {
        "name": "HalfNormal",
        "mu": 0.0,
        "sigma": 1.0,
    }
    assert lk_analytical["default_priors"]["logit1"] == {
        "name": "Normal",
        "mu": 0.0,
        "sigma": 1.0,
    }


def test_inv_softmax_temperature_3_logits():
    """Test inv_softmax_temperature with n_logits=3."""
    config = softmax_inv_temperature_config(n_logits=3)

    assert config["response"] == ["response"]
    assert config["choices"] == [0, 1, 2]
    assert config["list_params"] == ["beta", "logit1", "logit2"]
    assert (
        config["description"] == "The Inverse Softmax Temperature Model with 3 logits"
    )

    likelihoods = config["likelihoods"]
    lk_analytical = likelihoods["analytical"]

    # Test bounds
    assert lk_analytical["bounds"]["beta"] == (0.0, np.inf)
    assert lk_analytical["bounds"]["logit1"] == (-np.inf, np.inf)
    assert lk_analytical["bounds"]["logit2"] == (-np.inf, np.inf)

    # Test default priors
    assert lk_analytical["default_priors"]["beta"] == {
        "name": "HalfNormal",
        "mu": 0.0,
        "sigma": 1.0,
    }
    assert lk_analytical["default_priors"]["logit1"] == {
        "name": "Normal",
        "mu": 0.0,
        "sigma": 1.0,
    }
    assert lk_analytical["default_priors"]["logit2"] == {
        "name": "Normal",
        "mu": 0.0,
        "sigma": 1.0,
    }

"""Test the model registration functionality."""

import pytest

import hssm
from hssm.config import Config
from hssm.defaults import (
    default_model_config as registered_models,
)
from hssm.modelconfig import get_default_model_config
from hssm.register import get_model_info, list_registered_models, register_model


def test_register_model_preserves_non_categorical_metadata():
    """Registration should retain response-domain and simulator metadata."""
    name = "registered_circular_test_model"

    def logp(data, v):
        return data[:, 1] * 0.0 + v * 0.0

    def simulator(theta, random_state, n_replicas):
        del theta, random_state, n_replicas

    try:
        register_model(
            name=name,
            response=["rt", "response"],
            list_params=["v"],
            choices=None,
            response_kind="circular",
            response_bounds={"response": (-3.14, 3.14)},
            rv=simulator,
            default_loglik_kind="blackbox",
            description="A registration-contract test model",
            likelihoods={
                "blackbox": {
                    "loglik": logp,
                    "backend": None,
                    "bounds": {"v": (-1.0, 1.0)},
                    "default_priors": {},
                    "extra_fields": None,
                }
            },
        )

        config = Config.from_defaults(name, None)

        assert config.choices is None
        assert config.response_kind == "circular"
        assert config.response_bounds == {"response": (-3.14, 3.14)}
        assert config.rv is simulator
        config.validate()
    finally:
        registered_models.pop(name, None)


def test_register_model_rejects_an_unregistered_default_likelihood():
    """An explicit default must identify one of the supplied likelihood configs."""
    with pytest.raises(ValueError, match="is not registered"):
        register_model(
            name="invalid_default_likelihood_test_model",
            response=["rt", "response"],
            list_params=["v"],
            choices=[-1, 1],
            description=None,
            default_loglik_kind="approx_differentiable",
            likelihoods={
                "blackbox": {
                    "loglik": lambda data, v: data[:, 0] * 0.0 + v * 0.0,
                    "backend": None,
                    "bounds": {"v": (-1.0, 1.0)},
                    "default_priors": {},
                    "extra_fields": None,
                }
            },
        )


@pytest.mark.slow
def test_register_model():
    """Test registering a custom model."""
    # Test data for registration
    name = "custom_model"
    # get some model metadata for testing purposes
    config = get_default_model_config("ddm")

    # Register the model
    register_model(name=name, **config)

    # Test listing models
    list_registered_models()
    assert name in registered_models

    # Verify model can be instantiated
    data = hssm.load_data("cavanagh_theta")
    model = hssm.HSSM(
        model=name,
        data=data,
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
                    "theta": {"name": "Normal", "mu": 0.0, "sigma": 0.1},
                },
                "formula": "v ~ theta + (1|participant_id)",
                "link": "identity",
            },
        ],
    )
    assert isinstance(model, hssm.HSSM)

    # Test model info (should print but not return anything)
    assert get_model_info(name) is None

    # Test error cases
    with pytest.raises(ValueError):
        # Try to register model with existing name
        register_model(name=name, **config)

        # Try to get info for non-existent model
        get_model_info("non_existent_model")

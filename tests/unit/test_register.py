"""Test the model registration functionality."""

import pytest

import hssm
from hssm.register import register_model, list_registered_models, get_model_info
from hssm.modelconfig import get_default_model_config
from hssm.defaults import (
    default_model_config as registered_models,
)


@pytest.mark.slow
def test_register_model():
    """Test registering a custom model"""

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

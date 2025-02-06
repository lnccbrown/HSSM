"""Test the model registration functionality."""

import pytest

import hssm
from hssm.register import register_model, list_registered_models, get_model_info
from hssm.defaults import default_model_config


def test_register_model():
    """Test registering a custom model"""

    my_custom_model_name = "my_custom_model"
    config = default_model_config["ddm"]

    # Register the model with a new name
    register_model(my_custom_model_name, config)

    # Verify registration
    assert my_custom_model_name in default_model_config

    # Verify model can be instantiated
    data = hssm.load_data("cavanagh_theta")
    model = hssm.HSSM(
        model=my_custom_model_name,
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

    assert get_model_info(my_custom_model_name) is None

    with pytest.raises(ValueError):
        get_model_info("non_existent_model")

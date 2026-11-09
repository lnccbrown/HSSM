"""Softmax Inverse Temperature Model with 4 choices configuration."""

from .._types import DefaultConfig
from ._softmax_inv_temperature_config import softmax_inv_temperature_config


def get_softmax_inv_temperature_4_config() -> DefaultConfig:
    """
    Get the default config for the Inverse Softmax Temperature Model with 4 choices.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the
        Inverse Softmax Temperature Model with 4 choices, including response variables,
        model parameters, choices, description, and likelihood specifications.
    """
    return softmax_inv_temperature_config(n_choices=4)

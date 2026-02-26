"""Softmax Inverse Temperature Model with 3 logits configuration."""

from .._types import DefaultConfig
from ._softmax_inv_temperature_config import softmax_inv_temperature_config


def get_softmax_inv_temperature_3_config() -> DefaultConfig:
    """
    Get the default config for the Inverse Softmax Temperature Model with 3 logits.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the
        Inverse Softmax Temperature Model with 3 logits, including response variables,
        model parameters, choices, description, and likelihood specifications.
    """
    return softmax_inv_temperature_config(n_logits=3)

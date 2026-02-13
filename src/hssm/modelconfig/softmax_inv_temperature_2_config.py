"""Softmax Inverse Temperature Model with 2 logits configuration."""

from .._types import DefaultConfig
from ._softmax_inv_temperature import inv_softmax_temperature


def get_softmax_inv_temperature_2_config() -> DefaultConfig:
    """
    Get the default config for the Inverse Softmax Temperature Model with 2 logits.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the
        Inverse Softmax Temperature Model with 2 logits, including response variables,
        model parameters, choices, description, and likelihood specifications.
    """
    return inv_softmax_temperature(n_logits=2)

from .._types import DefaultConfig  # noqa: D100
from ..likelihoods.analytical import (
    lba3_bounds,
    lba3_params,
    logp_lba3,
)


def get_lba3_config() -> DefaultConfig:
    """
    Get the default configuration for the Levy-Beard Model 3 (LBA3).

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the LBA3,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": lba3_params,
        "choices": [0, 1, 2],
        "description": "Linear Ballistic Accumulator 3 Choices (LBA3)",
        "likelihoods": {
            "analytical": {
                "loglik": logp_lba3,
                "backend": None,
                "default_priors": {},
                "bounds": lba3_bounds,
                "extra_fields": None,
            }
        },
    }

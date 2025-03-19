from .._types import DefaultConfig  # noqa: D100
from ..likelihoods.analytical import (
    lba2_bounds,
    lba2_params,
    logp_lba2,
)


def get_lba2_config() -> DefaultConfig:
    """
    Get the default configuration for the Levy-Beard Model 2 (LBA2).

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the LBA2,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": lba2_params,
        "choices": [0, 1],
        "description": "Linear Ballistic Accumulator 2 Choices (LBA2)",
        "likelihoods": {
            "analytical": {
                "loglik": logp_lba2,
                "backend": None,
                "default_priors": {},
                "bounds": lba2_bounds,
                "extra_fields": None,
            }
        },
    }

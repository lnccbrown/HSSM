from .._types import DefaultConfig  # noqa: D100
from ..likelihoods.analytical import (
    lba4_bounds,
    lba4_params,
    logp_lba4,
)


def get_lba4_config() -> DefaultConfig:
    """
    Get the default configuration for the Linear Ballistic Accumulator 4 (LBA4).

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the LBA4,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": lba4_params,
        "choices": [0, 1, 2, 3],
        "description": "Linear Ballistic Accumulator 4 Choices (LBA4)",
        "likelihoods": {
            "analytical": {
                "loglik": logp_lba4,
                "backend": "jax",
                "default_priors": {},
                "bounds": lba4_bounds,
                "extra_fields": None,
            }
        },
    }

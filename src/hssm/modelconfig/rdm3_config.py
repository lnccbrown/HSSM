from .._types import DefaultConfig  # noqa: D100
from ..likelihoods.analytical import (
    rdm3_bounds,
    rdm3_params,
    logp_rdm3,
)


def get_rdm3_config() -> DefaultConfig:
    """
    Get the default configuration for the Racing Diffusion Model 3 (RDM3).

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the RDM3,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": rdm3_params,
        "choices": [0, 1, 2],
        "description": "Racing Diffusion Model 3 Choices (RDM3)",
        "likelihoods": {
            "analytical": {
                "loglik": logp_rdm3,
                "backend": None,
                "default_priors": {},
                "bounds": rdm3_bounds,
                "extra_fields": None,
            }
        },
    }

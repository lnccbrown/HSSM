import numpy as np  # noqa: D100

from .._types import DefaultConfig
from ..likelihoods.analytical import (
    ddm_sdv_bounds,
)
from ..likelihoods.blackbox import logp_full_ddm


def get_full_ddm_config() -> DefaultConfig:
    """
    Get the default configuration for the Full Drift Diffusion Model (FDDM).

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the FDDM,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ["v", "a", "z", "t", "sz", "sv", "st"],
        "choices": [-1, 1],
        "description": "The full Drift Diffusion Model (DDM)",
        "likelihoods": {
            "blackbox": {
                "loglik": logp_full_ddm,
                "backend": None,
                "bounds": ddm_sdv_bounds | {"sz": (0, np.inf), "st": (0, np.inf)},
                "default_priors": {
                    "t": {
                        "name": "HalfNormal",
                        "sigma": 2.0,
                    },
                },
                "extra_fields": None,
            }
        },
    }

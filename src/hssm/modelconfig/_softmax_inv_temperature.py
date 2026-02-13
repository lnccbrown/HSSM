"""The default configuration for the Inverse Softmax Temperature Model with 2 logits."""

import numpy as np

from .._types import DefaultConfig, ParamSpec
from ..likelihoods.analytical import softmax_inv_temperature


def get_inv_softmax_temperature_config(n_logits: int = 2) -> DefaultConfig:
    """
    Get the default config for the Inverse Softmax Temperature Model with 2 logits.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the DDM,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    bounds = {"beta": (0.0, np.inf)}
    bounds.update({f"logit{i}": (-np.inf, np.inf) for i in range(1, n_logits)})
    default_priors: dict[str, ParamSpec] = {
        "beta": {"name": "HalfNormal", "mu": 0.0, "sigma": 1.0},
    }
    default_priors.update(
        {
            f"logit{i}": {"name": "Normal", "mu": 0.0, "sigma": 1.0}
            for i in range(1, n_logits)
        }
    )

    return {
        "response": ["response"],
        "list_params": ["beta"] + [f"logit{i}" for i in range(1, n_logits)],
        "choices": [-1, 1] if n_logits == 2 else list(range(n_logits)),
        "description": "The Inverse Softmax Temperature Model",
        "likelihoods": {
            "analytical": {
                "loglik": softmax_inv_temperature,
                "backend": None,
                "bounds": bounds,
                "default_priors": default_priors,
                "extra_fields": None,
            },
        },
    }

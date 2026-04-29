"""The default configuration for the Inverse Softmax Temperature Model with 2 logits."""

import numpy as np

from .._types import DefaultConfig, ParamSpec
from ..likelihoods.analytical import softmax_inv_temperature


def softmax_inv_temperature_config(n_choices: int = 2) -> DefaultConfig:
    """
    Get the default config for the Softmax Inv. Temperature Model.

    Parameters
    ----------
    n_choices : optional
        The number of choices in the model. Must be at least 2. The number of logits
        will be n_choices - 1. Default is 2 (i.e., 1 logit).

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the
        Inverse Softmax Temperature Model, including response variables, model
        parameters, choices, description, and likelihood specifications.
    """
    if n_choices < 2:
        raise ValueError("n_choices must be at least 2.")

    bounds = {"beta": (0.0, np.inf)}
    bounds.update({f"logit{i}": (-np.inf, np.inf) for i in range(1, n_choices)})
    default_priors: dict[str, ParamSpec] = {
        "beta": {
            "name": "Gamma",
            "alpha": 2.0,
            "beta": 0.5,
        },
    }
    default_priors.update(
        {
            f"logit{i}": {"name": "Normal", "mu": 0.0, "sigma": 1.0}
            for i in range(1, n_choices)
        }
    )

    return {
        "response": ["response"],
        "list_params": ["beta"] + [f"logit{i}" for i in range(1, n_choices)],
        "choices": [-1, 1] if n_choices == 2 else list(range(n_choices)),
        "description": f"The Softmax Inv. Temperature Model with {n_choices} choices",
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

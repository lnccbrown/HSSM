from .._types import DefaultConfig  # noqa: D100
from ..likelihoods.analytical import logp_exgauss

def get_exgauss_config() -> DefaultConfig:
    """
    Get the default configuration for the Ex-Gaussian model.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the ex-gaussian model,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ["mu", "sigma", "tau", "p"],
        "choices": [-1, 1],
        "description": "The Ex-Gaussian Model",
        "likelihoods": {
            "analytical": {
                "loglik": logp_exgauss,
                "backend": "None",
                "default_priors": {},
                "bounds": {
                    "mu": (0.0, 50.0),
                    "sigma": (0.0, 50.0),
                    "tau": (0.0, 50.0),
                    "p": (0.0, 1.0),
                },
                "extra_fields": None,
            },
        },
    }

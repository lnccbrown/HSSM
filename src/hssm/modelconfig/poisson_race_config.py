from .._types import DefaultConfig  # noqa: D100
from ..likelihoods.analytical import poisson_race_bounds, poisson_race_params, logp_poisson_race


def get_poisson_race_config() -> DefaultConfig:
    """
    Get the default configuration for the Poisson Race Model.

    Returns
    -------
    DefaultConfig
        A dict containing the default configuration settings for the Poisson Race Model
    """
    return {
        "response": ["rt", "response"],
        "list_params": poisson_race_params,
        "choices": [-1, 1],
        "description": "The Poisson Race Model",
        "likelihoods": {
            "analytical": {
                "loglik": logp_poisson_race,
                "backend": None,
                "bounds": poisson_race_bounds,
                "default_priors": {
                    "t": {
                        "name": "HalfNormal",
                        "sigma": 2.0,
                    },
                    "r1": {
                        "name": "HalfNormal",
                        "sigma": 5.0,
                    },
                    "r2": {
                        "name": "HalfNormal",
                        "sigma": 5.0,
                    },
                    "k1": {
                        "name": "HalfNormal",
                        "sigma": 20.0,
                    },
                    "k2": {
                        "name": "HalfNormal",
                        "sigma": 20.0,
                    },
                },
                "extra_fields": None,
            },
        },
    }

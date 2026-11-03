from .._types import DefaultConfig  # noqa: D100
from ..likelihoods.analytical import (
    lba4_bounds,
    lba4_params,
    logp_lba4,
    logp_lba_jax,
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
                "jax_loglik": logp_lba_jax,
                "backend": "jax",
                "default_priors": {
                    # The bounds-derived prior, started below b's default start
                    # (1.2): the likelihood rejects b <= A, and A's own default
                    # start (2.0) would make log p -inf.
                    "A": {"name": "HalfNormal", "sigma": 2.0, "initval": 0.5},
                },
                "bounds": lba4_bounds,
                "extra_fields": None,
            }
        },
    }

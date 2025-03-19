from .._types import DefaultConfig  # noqa: D100


def get_weibull_config() -> DefaultConfig:
    """
    Get the default configuration for the weibull model.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the
        weibull model, including response variables, model parameters,
        choices, description, and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ["v", "a", "z", "t", "alpha", "beta"],
        "choices": [-1, 1],
        "description": None,
        "likelihoods": {
            "approx_differentiable": {
                "loglik": "weibull.onnx",
                "backend": "jax",
                "default_priors": {},
                "bounds": {
                    "v": (-2.5, 2.5),
                    "a": (0.3, 2.5),
                    "z": (0.2, 0.8),
                    "t": (1e-3, 2.0),
                    "alpha": (0.31, 4.99),
                    "beta": (0.31, 6.99),
                },
                "extra_fields": None,
            },
        },
    }

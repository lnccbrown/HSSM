from .._types import DefaultConfig  # noqa: D100


def get_levy_config() -> DefaultConfig:
    """
    Get the default configuration for the levy model.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the levy model,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ["v", "a", "z", "alpha", "t"],
        "choices": [-1, 1],
        "description": None,
        "likelihoods": {
            "approx_differentiable": {
                "loglik": "levy.onnx",
                "backend": "jax",
                "default_priors": {},
                "bounds": {
                    "v": (-3.0, 3.0),
                    "a": (0.3, 3.0),
                    "z": (0.1, 0.9),
                    "alpha": (1.0, 2.0),
                    "t": (1e-3, 2.0),
                },
                "extra_fields": None,
            },
        },
    }

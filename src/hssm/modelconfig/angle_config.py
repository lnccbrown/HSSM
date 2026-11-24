from .._types import DefaultConfig  # noqa: D100


def get_angle_config() -> DefaultConfig:
    """
    Get the default configuration for the angle model.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the angle model,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ["v", "a", "z", "t", "theta"],
        "choices": [-1, 1],
        "description": None,
        "likelihoods": {
            "approx_differentiable": {
                "loglik": "angle.onnx",
                "backend": "jax",
                "default_priors": {},
                "bounds": {
                    "v": (-3.0, 3.0),
                    "a": (0.3, 3.0),
                    "z": (0.1, 0.9),
                    "t": (0.001, 2.0),
                    "theta": (-0.1, 1.3),
                },
                "extra_fields": None,
            },
        },
    }

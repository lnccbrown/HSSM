from .._types import DefaultConfig  # noqa: D100


def get_race_no_bias_angle_4_config() -> DefaultConfig:
    """
    Get the default configuration for the race model.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the race model,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ["v0", "v1", "v2", "v3", "a", "z", "t", "theta"],
        "choices": [0, 1, 2, 3],
        "description": None,
        "likelihoods": {
            "approx_differentiable": {
                "loglik": "race_no_bias_angle_4.onnx",
                "backend": "jax",
                "default_priors": {},
                "bounds": {
                    "v0": (0.0, 2.5),
                    "v1": (0.0, 2.5),
                    "v2": (0.0, 2.5),
                    "v3": (0.0, 2.5),
                    "a": (1.0, 3.0),
                    "z": (0.0, 0.9),
                    "t": (0.0, 2.0),
                    "theta": (-0.1, 1.45),
                },
                "extra_fields": None,
            },
        },
    }

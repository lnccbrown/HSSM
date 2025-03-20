from .._types import DefaultConfig  # noqa: D100


def get_ddm_seq2_no_bias_config() -> DefaultConfig:
    """
    Get the default configuration for the ddm_seq2_no_bias model.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the
        ddm_seq2_no_bias model, including response variables, model choices,
        description, and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ["vh", "vl1", "vl2", "a", "t"],
        "choices": [0, 1, 2, 3],
        "description": None,
        "likelihoods": {
            "approx_differentiable": {
                "loglik": "ddm_seq2_no_bias.onnx",
                "backend": "jax",
                "default_priors": {},
                "bounds": {
                    "vh": (-4.0, 4.0),
                    "vl1": (-4.0, 4.0),
                    "vl2": (-4.0, 4.0),
                    "a": (0.3, 2.5),
                    "t": (0.0, 2.0),
                },
                "extra_fields": None,
            },
        },
    }

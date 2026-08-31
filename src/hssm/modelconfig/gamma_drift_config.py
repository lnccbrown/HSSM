from .._types import DefaultConfig  # noqa: D100


def get_gamma_drift_config() -> DefaultConfig:
    """
    Get the default configuration for the gamma_drift model.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for
        the gamma_drift model, including response variables, model
        parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ["v", "a", "z", "t", "shape", "scale", "c"],
        "choices": [-1, 1],
        "description": (
            "DDM with a gamma-shaped drift component added to a constant "
            "drift, as in conflict-task models: `shape` and `scale` set the "
            "bump's time course and `c` its signed peak amplitude."
        ),
        "likelihoods": {
            "approx_differentiable": {
                "loglik": "gamma_drift.onnx",
                "backend": "jax",
                "default_priors": {},
                # The network's training box: the full ssm-simulators bounds
                # for gamma_drift, which the production training data was
                # sampled from (LAN_pipeline_minimal
                # configs/production_gamma_drift/).
                "bounds": {
                    "v": (-3.0, 3.0),
                    "a": (0.3, 3.0),
                    "z": (0.1, 0.9),
                    "t": (0.001, 2.0),
                    "shape": (2.0, 10.0),
                    "scale": (0.01, 1.0),
                    "c": (-3.0, 3.0),
                },
                "extra_fields": None,
            },
        },
    }

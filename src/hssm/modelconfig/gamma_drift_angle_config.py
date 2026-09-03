from .._types import DefaultConfig  # noqa: D100


def get_gamma_drift_angle_config() -> DefaultConfig:
    """
    Get the default configuration for the gamma_drift_angle model.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for
        the gamma_drift_angle model, including response variables, model
        parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        # Order is load-bearing: it is the column order of the ONNX input, and
        # it must match ssms.config.model_config["gamma_drift_angle"]["params"]
        # element for element. `theta` sits at index 4, between `t` and
        # `shape` -- not appended at the end.
        "list_params": ["v", "a", "z", "t", "theta", "shape", "scale", "c"],
        "choices": [-1, 1],
        "description": (
            "`gamma_drift` with an angled, collapsing decision bound: a "
            "gamma-shaped drift bump (`shape`, `scale`, signed peak `c`) on "
            "top of a constant drift, with `theta` setting the bound's "
            "collapse angle."
        ),
        "likelihoods": {
            "approx_differentiable": {
                "loglik": "gamma_drift_angle.onnx",
                "backend": "jax",
                "default_priors": {},
                # The network's training box: the full ssm-simulators bounds
                # for gamma_drift_angle, which the production training data
                # was sampled from (LAN_pipeline_minimal
                # configs/production_gamma_drift_angle/).
                "bounds": {
                    "v": (-3.0, 3.0),
                    "a": (0.3, 3.0),
                    "z": (0.1, 0.9),
                    "t": (0.001, 2.0),
                    "theta": (-0.1, 1.3),
                    "shape": (2.0, 10.0),
                    "scale": (0.01, 1.0),
                    "c": (-3.0, 3.0),
                },
                "extra_fields": None,
            },
        },
    }

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
        # Order is load-bearing: it is the column order of the ONNX input, and
        # it must match ssms.config.model_config["angle"]["params"] element
        # for element.
        "list_params": ["v", "a", "z", "t", "theta"],
        "choices": [-1, 1],
        "description": (
            "The angle model -- constant drift with a linearly collapsing "
            "decision bound whose collapse angle is `theta`."
        ),
        "likelihoods": {
            "approx_differentiable": {
                # The wider-range network, trained on `v` in (-6, 6). Up to
                # 0.5.0 it was the model `angle_extended`, and `angle` loaded
                # `angle.onnx`, trained on `v` in (-3, 3). The Hub file keeps
                # its name.
                "loglik": "angle_extended.onnx",
                "backend": "jax",
                "default_priors": {},
                # The network's training box: the ssm-simulators bounds for
                # angle, which the training data was sampled from
                # (LAN_pipeline_minimal configs/production_angle_extended/).
                "bounds": {
                    "v": (-6.0, 6.0),
                    "a": (0.3, 3.0),
                    "z": (0.1, 0.9),
                    "t": (0.001, 2.0),
                    "theta": (-0.1, 1.3),
                },
                "extra_fields": None,
            },
        },
    }

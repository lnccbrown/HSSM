from .._types import DefaultConfig  # noqa: D100
from ..likelihoods.analytical import ddm_bounds, ddm_params, logp_ddm
from ..likelihoods.blackbox import logp_ddm_bbox


def get_ddm_config() -> DefaultConfig:
    """
    Get the default configuration for the Drift Diffusion Model (DDM).

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the DDM,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ddm_params,
        "choices": [-1, 1],
        "description": "The Drift Diffusion Model (DDM)",
        "likelihoods": {
            "analytical": {
                "loglik": logp_ddm,
                "backend": None,
                "bounds": ddm_bounds,
                "default_priors": {
                    "t": {
                        "name": "HalfNormal",
                        "sigma": 2.0,
                    },
                },
                "extra_fields": None,
            },
            "approx_differentiable": {
                "loglik": "ddm.onnx",
                "backend": "jax",
                "default_priors": {
                    "t": {
                        "name": "HalfNormal",
                        "sigma": 2.0,
                    },
                },
                "bounds": {
                    "v": (-3.0, 3.0),
                    "a": (0.3, 2.5),
                    "z": (0.0, 1.0),
                    "t": (0.0, 2.0),
                },
                "extra_fields": None,
            },
            "blackbox": {
                "loglik": logp_ddm_bbox,
                "backend": None,
                "bounds": ddm_bounds,
                "default_priors": {
                    "t": {
                        "name": "HalfNormal",
                        "sigma": 2.0,
                    },
                },
                "extra_fields": None,
            },
        },
    }

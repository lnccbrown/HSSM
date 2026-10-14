from .._types import DefaultConfig  # noqa: D100
from ..likelihoods.analytical import (
    ddm_sdv_bounds,
    ddm_sdv_params,
    logp_ddm_sdv,
)
from ..likelihoods.blackbox import logp_ddm_sdv_bbox


def get_ddm_sdv_config() -> DefaultConfig:
    """
    Get the default configuration for the DDM with standard deviation for v.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the model,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ddm_sdv_params,
        "choices": [-1, 1],
        "description": "The Drift Diffusion Model (DDM) with standard deviation for v",
        "likelihoods": {
            "analytical": {
                "loglik": logp_ddm_sdv,
                "backend": None,
                "bounds": ddm_sdv_bounds,
                "default_priors": {
                    "t": {
                        "name": "HalfNormal",
                        "sigma": 2.0,
                    },
                },
                "extra_fields": None,
            },
            "approx_differentiable": {
                "loglik": "ddm_sdv.onnx",
                "backend": "jax",
                "default_priors": {
                    "t": {
                        "name": "HalfNormal",
                        "sigma": 2.0,
                    },
                },
                # The network's training box (ssms `ddm_sdv` param_bounds), which
                # is what these bounds are supposed to describe: outside it the
                # LAN extrapolates and returns a finite but wrong density.
                # `sv` was (0.0, 1.0) here while training covered (0.001, 2.5) —
                # written before any ddm_sdv.onnx existed, so it described no
                # actual network. The published network's density gate samples
                # `sv` up to 2.25 and passes there (total mass within 0.15% of 1,
                # Hellinger at or below the KDE sampling floor).
                "bounds": {
                    "v": (-3.0, 3.0),
                    "a": (0.3, 2.5),
                    "z": (0.1, 0.9),
                    "t": (0.0, 2.0),
                    "sv": (0.0, 2.5),
                },
                "extra_fields": None,
            },
            "blackbox": {
                "loglik": logp_ddm_sdv_bbox,
                "backend": None,
                "bounds": ddm_sdv_bounds,
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

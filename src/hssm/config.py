"""Provide default configurations for models in the HSSM class."""
from typing import Any, Literal

from .likelihoods.analytical import (
    ddm_bounds,
    ddm_params,
    ddm_sdv_params,
    logp_ddm,
    logp_ddm_sdv,
)

SupportedModels = Literal[
    "ddm",
    "ddm_sdv",
    "angle",
    "levy",
    "ornstein",
    "weibull",
    "race_no_bias_angle_4",
    "ddm_seq2_no_bias",
]

LoglikKind = Literal["analytical", "approx_differentiable", "blackbox"]

ConfigParams = Literal[
    "loglik",
    "list_params",
    "default_priors",
    "backend",
    "bounds",
    "rv",
]

Config = dict[ConfigParams, Any]

default_model_config: dict[SupportedModels, dict[Literal[LoglikKind], Config]] = {
    "ddm": {
        "analytical": {
            "loglik": logp_ddm,
            "bounds": ddm_bounds,
            "default_priors": {
                "v": {"name": "Uniform", "lower": -10.0, "upper": 10.0},
                "a": {"name": "HalfNormal", "sigma": 2.0},
                "t": {"name": "Uniform", "lower": 0.0, "upper": 2.0, "initval": 0.1},
            },
        },
        "approx_differentiable": {
            "loglik": "ddm.onnx",
            "backend": "jax",
            "bounds": {
                "v": (-3.0, 3.0),
                "a": (0.3, 2.5),
                "z": (0.0, 1.0),
                "t": (0.0, 2.0),
            },
        },
    },
    "ddm_sdv": {
        "analytical": {
            "loglik": logp_ddm_sdv,
            "bounds": ddm_bounds,
            "default_priors": {
                "v": {"name": "Uniform", "lower": -10.0, "upper": 10.0},
                "sv": {"name": "HalfNormal", "sigma": 2.0},
                "a": {"name": "HalfNormal", "sigma": 2.0},
                "t": {"name": "Uniform", "lower": 0.0, "upper": 5.0, "initval": 0.0},
            },
        },
        "approx_differentiable": {
            "loglik": "ddm_sdv.onnx",
            "backend": "jax",
            "bounds": {
                "v": (-3.0, 3.0),
                "sv": (0.0, 1.0),
                "a": (0.3, 2.5),
                "z": (0.1, 0.9),
                "t": (0.0, 2.0),
            },
        },
    },
    "angle": {
        "approx_differentiable": {
            "loglik": "angle.onnx",
            "backend": "jax",
            "bounds": {
                "v": (-3.0, 3.0),
                "a": (0.3, 3.0),
                "z": (0.1, 0.9),
                "t": (0.001, 2.0),
                "theta": (-0.1, 1.3),
            },
        },
    },
    "levy": {
        "approx_differentiable": {
            "loglik": "levy.onnx",
            "backend": "jax",
            "bounds": {
                "v": (-3.0, 3.0),
                "a": (0.3, 3.0),
                "z": (0.1, 0.9),
                "alpha": (1.0, 2.0),
                "t": (1e-3, 2.0),
            },
        },
    },
    "ornstein": {
        "approx_differentiable": {
            "loglik": "ornstein.onnx",
            "backend": "jax",
            "bounds": {
                "v": (-2.0, 2.0),
                "a": (0.3, 3.0),
                "z": (0.1, 0.9),
                "g": (-1.0, 1.0),
                "t": (1e-3, 2.0),
            },
        },
    },
    "weibull": {
        "approx_differentiable": {
            "loglik": "weibull.onnx",
            "backend": "jax",
            "bounds": {
                "v": (-2.5, 2.5),
                "a": (0.3, 2.5),
                "z": (0.2, 0.8),
                "t": (1e-3, 2.0),
                "alpha": (0.31, 4.99),
                "beta": (0.31, 6.99),
            },
        },
    },
    "race_no_bias_angle_4": {
        "approx_differentiable": {
            "loglik": "race_no_bias_angle_4.onnx",
            "backend": "jax",
            "bounds": {
                "v0": (0.0, 2.5),
                "v1": (0.0, 2.5),
                "v2": (0.0, 2.5),
                "v3": (0.0, 2.5),
                "a": (1.0, 3.0),
                "z": (0.0, 0.9),
                "ndt": (0.0, 2.0),
                "theta": (-0.1, 1.45),
            },
        },
    },
    "ddm_seq2_no_bias": {
        "approx_differentiable": {
            "loglik": "ddm_seq2_no_bias.onnx",
            "backend": "jax",
            "bounds": {
                "vh": (-4.0, 4.0),
                "vl1": (-4.0, 4.0),
                "vl2": (-4.0, 4.0),
                "a": (0.3, 2.5),
                "t": (0.0, 2.0),
            },
        },
    },
}

default_params: dict[SupportedModels, list[str]] = {
    "ddm": ddm_params,
    "ddm_sdv": ddm_sdv_params,
    "angle": ["v", "a", "z", "t", "theta"],
    "levy": ["v", "a", "z", "alpha", "t"],
    "ornstein": ["v", "a", "z", "g", "t"],
    "weibull": ["v", "a", "z", "t", "alpha", "beta"],
    "race_no_bias_angle_4": ["v0", "v1", "v2", "v3", "a", "z", "ndt", "theta"],
    "ddm_seq2_no_bias": ["vh", "vl1", "vl2", "a", "t"],
}

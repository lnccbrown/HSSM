"""Module provides default configuration settings for various HSSM package models.

Each model configuration includes response variables, model parameters, choices,
description.

Functions
---------
get_ddm_config() -> DefaultConfig

get_ddm_svd_config() -> DefaultConfig

get_full_ddm_config() -> DefaultConfig

get_lba2_config() -> DefaultConfig

get_lba3_config() -> DefaultConfig

get_angle_config() -> DefaultConfig

get_levy_config() -> DefaultConfig

get_ornstein_config() -> DefaultConfig

get_weibull_config() -> DefaultConfig

get_race_no_bias_angle_4_config() -> DefaultConfig
"""

import functools

import numpy as np

from ._types import DefaultConfig, SupportedModels
from .likelihoods.analytical import (
    ddm_bounds,
    ddm_params,
    ddm_sdv_bounds,
    ddm_sdv_params,
    lba2_bounds,
    lba2_params,
    lba3_bounds,
    lba3_params,
    logp_ddm,
    logp_ddm_sdv,
    logp_lba2,
    logp_lba3,
)
from .likelihoods.blackbox import logp_ddm_bbox, logp_ddm_sdv_bbox, logp_full_ddm


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


def get_ddm_svd_config() -> DefaultConfig:
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
                "bounds": {
                    "v": (-3.0, 3.0),
                    "a": (0.3, 2.5),
                    "z": (0.1, 0.9),
                    "t": (0.0, 2.0),
                    "sv": (0.0, 1.0),
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


def get_full_ddm_config() -> DefaultConfig:
    """
    Get the default configuration for the Full Drift Diffusion Model (FDDM).

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the FDDM,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ["v", "a", "z", "t", "sz", "sv", "st"],
        "choices": [-1, 1],
        "description": "The full Drift Diffusion Model (DDM)",
        "likelihoods": {
            "blackbox": {
                "loglik": logp_full_ddm,
                "backend": None,
                "bounds": ddm_sdv_bounds | {"sz": (0, np.inf), "st": (0, np.inf)},
                "default_priors": {
                    "t": {
                        "name": "HalfNormal",
                        "sigma": 2.0,
                    },
                },
                "extra_fields": None,
            }
        },
    }


def get_lba2_config() -> DefaultConfig:
    """
    Get the default configuration for the Levy-Beard Model 2 (LBA2).

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the LBA2,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": lba2_params,
        "choices": [0, 1],
        "description": "Linear Ballistic Accumulator 2 Choices (LBA2)",
        "likelihoods": {
            "analytical": {
                "loglik": logp_lba2,
                "backend": None,
                "default_priors": {},
                "bounds": lba2_bounds,
                "extra_fields": None,
            }
        },
    }


def get_lba3_config() -> DefaultConfig:
    """
    Get the default configuration for the Levy-Beard Model 3 (LBA3).

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the LBA3,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": lba3_params,
        "choices": [0, 1, 2],
        "description": "Linear Ballistic Accumulator 3 Choices (LBA3)",
        "likelihoods": {
            "analytical": {
                "loglik": logp_lba3,
                "backend": None,
                "default_priors": {},
                "bounds": lba3_bounds,
                "extra_fields": None,
            }
        },
    }


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


def get_ornstein_config() -> DefaultConfig:
    """
    Get the default configuration for the ornstein model.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for
        the ornstein model, including response variables, model
        parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ["v", "a", "z", "g", "t"],
        "choices": [-1, 1],
        "description": None,
        "likelihoods": {
            "approx_differentiable": {
                "loglik": "ornstein.onnx",
                "backend": "jax",
                "default_priors": {},
                "bounds": {
                    "v": (-2.0, 2.0),
                    "a": (0.3, 3.0),
                    "z": (0.1, 0.9),
                    "g": (-1.0, 1.0),
                    "t": (1e-3, 2.0),
                },
                "extra_fields": None,
            },
        },
    }


def get_weibull_config() -> DefaultConfig:
    """
    Get the default configuration for the weibull model.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the
        weibull model, including response variables, model parameters,
        choices, description, and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ["v", "a", "z", "t", "alpha", "beta"],
        "choices": [-1, 1],
        "description": None,
        "likelihoods": {
            "approx_differentiable": {
                "loglik": "weibull.onnx",
                "backend": "jax",
                "default_priors": {},
                "bounds": {
                    "v": (-2.5, 2.5),
                    "a": (0.3, 2.5),
                    "z": (0.2, 0.8),
                    "t": (1e-3, 2.0),
                    "alpha": (0.31, 4.99),
                    "beta": (0.31, 6.99),
                },
                "extra_fields": None,
            },
        },
    }


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


@functools.lru_cache
def get_default_model_meta(model_name: SupportedModels) -> DefaultConfig:
    """
    Get the default configuration for a given model name.

    Parameters
    ----------
    model_name
        The name of the model.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the given model
        name,
        including response variables, model parameters, choices, description,
        and likelihood specifications.

    """
    _map = {
        "ddm": get_ddm_config,
        "ddm_sdv": get_ddm_svd_config,
        "full_ddm": get_full_ddm_config,
        "angle": get_angle_config,
        "levy": get_levy_config,
        "ornstein": get_ornstein_config,
        "weibull": get_weibull_config,
        "race_no_bias_angle_4": get_race_no_bias_angle_4_config,
        "ddm_seq2_no_bias": get_ddm_seq2_no_bias_config,
        "lba3": get_lba3_config,
        "lba2": get_lba2_config,
    }
    try:
        return _map[model_name]()
    except KeyError:
        raise ValueError(f"Model {model_name} not currently registered in HSSM.")

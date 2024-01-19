"""Provide default configurations for models in the HSSM class."""
from os import PathLike
from typing import Callable, Literal, Optional, TypedDict, Union

import bambi as bmb
import numpy as np
from pymc import Distribution
from pytensor.graph.op import Op

from .likelihoods.analytical import (
    ddm_bounds,
    ddm_params,
    ddm_sdv_bounds,
    ddm_sdv_params,
    logp_ddm,
    logp_ddm_sdv,
)
from .likelihoods.blackbox import logp_ddm_bbox, logp_ddm_sdv_bbox, logp_full_ddm
from .param import ParamSpec, _make_default_prior

LogLik = Union[str, PathLike, Callable, Op, type[Distribution]]

SupportedModels = Literal[
    "ddm",
    "ddm_sdv",
    "full_ddm",
    "angle",
    "levy",
    "ornstein",
    "weibull",
    "race_no_bias_angle_4",
    "ddm_seq2_no_bias",
]

LoglikKind = Literal["analytical", "approx_differentiable", "blackbox"]


class LoglikConfig(TypedDict):
    """Type for the value of LoglikConfig."""

    loglik: LogLik
    backend: Optional[Literal["jax", "pytensor"]]
    default_priors: dict[str, ParamSpec]
    bounds: dict[str, tuple[float, float]]
    extra_fields: Optional[list[str]]


LoglikConfigs = dict[LoglikKind, LoglikConfig]


class DefaultConfig(TypedDict):
    """Type for the value of DefaultConfig."""

    list_params: list[str]
    description: Optional[str]
    likelihoods: LoglikConfigs


DefaultConfigs = dict[SupportedModels, DefaultConfig]

default_model_config: DefaultConfigs = {
    "ddm": {
        "list_params": ddm_params,
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
                        "initval": 0.1,
                    },
                },
                "extra_fields": None,
            },
            "approx_differentiable": {
                "loglik": "ddm.onnx",
                "backend": "jax",
                "default_priors": {},
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
                        "initval": 0.1,
                    },
                },
                "extra_fields": None,
            },
        },
    },
    "ddm_sdv": {
        "list_params": ddm_sdv_params,
        "description": "The Drift Diffusion Model (DDM) with standard deviation for v",
        "likelihoods": {
            "analytical": {
                "loglik": logp_ddm_sdv,
                "backend": None,
                "bounds": ddm_bounds,
                "default_priors": {
                    "t": {
                        "name": "HalfNormal",
                        "sigma": 2.0,
                        "initval": 0.1,
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
                        "initval": 0.1,
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
                        "initval": 0.1,
                    },
                },
                "extra_fields": None,
            },
        },
    },
    "full_ddm": {
        "list_params": ["v", "a", "z", "t", "sv", "sz", "st"],
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
                        "initval": 0.1,
                    },
                },
                "extra_fields": None,
            }
        },
    },
    "angle": {
        "list_params": ["v", "a", "z", "t", "theta"],
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
    },
    "levy": {
        "list_params": ["v", "a", "z", "alpha", "t"],
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
    },
    "ornstein": {
        "list_params": ["v", "a", "z", "g", "t"],
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
    },
    "weibull": {
        "list_params": ["v", "a", "z", "t", "alpha", "beta"],
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
    },
    "race_no_bias_angle_4": {
        "list_params": ["v0", "v1", "v2", "v3", "a", "z", "t", "theta"],
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
    },
    "ddm_seq2_no_bias": {
        "list_params": ["vh", "vl1", "vl2", "a", "t"],
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
    },
}


def show_defaults(model: SupportedModels, loglik_kind=Optional[LoglikKind]) -> str:
    """Show the defaults for supported models.

    Parameters
    ----------
    model
        One of the supported model strings.
    loglik_kind : optional
        The kind of likelihood function, by default None, in which case the defaults for
        all likelihoods will be shown.

    Returns
    -------
    str
        A nicely organized printout for the defaults of provided model.
    """
    if model not in default_model_config:
        raise ValueError(f"{model} does not currently have defaults in HSSM.")

    model_config = default_model_config[model]

    output = []
    output.append("Default model config:")
    output.append(f"Model: {model}")

    if model_config["description"] is not None:
        output.append("Description:")
        output.append(f"    {model_config['description']}")

    output.append(f"Default parameters: {model_config['list_params']}")
    output.append("")

    if loglik_kind is not None:
        if loglik_kind not in model_config["likelihoods"]:
            raise ValueError(
                f"{model} does not currently have defaults for `{loglik_kind}` "
                + "log-likelihoods in HSSM."
            )

        output += _show_defaults_helper(model, loglik_kind)

    else:
        for loglik_kind in model_config["likelihoods"].keys():
            output += _show_defaults_helper(model, loglik_kind)
            output.append("")

        output = output[:-1]

    return "\r\n".join(output)


def _show_defaults_helper(model: SupportedModels, loglik_kind: LoglikKind) -> list[str]:
    """Show the defaults for supported models.

    Parameters
    ----------
    model
        One of the supported model strings.
    loglik_kind
        The kind of likelihood function.

    Returns
    -------
    list[str]
        A list of nicely organized printout for the defaults of provided model.
    """
    output = []
    params = default_model_config[model]["list_params"]
    model_defaults = default_model_config[model]["likelihoods"][loglik_kind]

    output.append(f"Log-likelihood kind: {loglik_kind}")
    output.append(f"Log-likelihood: {model_defaults['loglik']}")
    if loglik_kind == "approx_differentiable":
        output.append("Default backend: jax")
    output.append("Default priors:")

    default_priors = model_defaults.get("default_priors", {})
    default_bounds = model_defaults.get("bounds", {})

    for param in params:
        prior = default_priors.get(param, None)
        if prior is None:
            bounds = default_bounds.get(param, None)
            prior = _make_default_prior(bounds) if bounds is not None else None
        else:
            if isinstance(prior, dict):
                prior = bmb.Prior(**prior)

        output.append(f"    {param} ~ {prior}")

    output.append("Default bounds:")

    for param in params:
        output.append(f"    {param}: {default_bounds.get(param, None)}")

    return output

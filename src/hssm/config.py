"""Provide default configurations for models in the HSSM class."""
from typing import Any, Literal, Optional

import bambi as bmb

from .likelihoods.analytical import (
    ddm_bounds,
    ddm_params,
    ddm_sdv_params,
    logp_ddm,
    logp_ddm_sdv,
)
from .param import _make_default_prior

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
                "t": {"name": "Uniform", "lower": 0.0, "upper": 2.0, "initval": 0.1},
            },
        },
        "approx_differentiable": {
            "loglik": "ddm.onnx",
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
                "t": {"name": "Uniform", "lower": 0.0, "upper": 2.0, "initval": 0.1},
            },
        },
        "approx_differentiable": {
            "loglik": "ddm_sdv.onnx",
            "bounds": {
                "v": (-3.0, 3.0),
                "a": (0.3, 2.5),
                "z": (0.1, 0.9),
                "t": (0.0, 2.0),
                "sv": (0.0, 1.0),
            },
        },
    },
    "angle": {
        "approx_differentiable": {
            "loglik": "angle.onnx",
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

    # Will implement later
    # if "description" in model_config:
    #     output.append("Description:")
    #     output.append(f"    {model_config['description']}")

    output.append(f"Default parameters: {default_params[model]}")
    output.append("")

    if loglik_kind is not None:
        if loglik_kind not in model_config:
            raise ValueError(
                f"{model} does not currently have defaults for `{loglik_kind}` "
                + "log-likelihoods in HSSM."
            )

        output += _show_defaults_helper(model, loglik_kind)

    else:
        for loglik_kind in model_config.keys():
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
    params = default_params[model]
    model_defaults = default_model_config[model][loglik_kind]

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

"""Provide default configurations for models in the HSSM class."""

from enum import Enum

import bambi as bmb

from ._types import (
    DefaultConfigs,
    LoglikKind,
    Optional,
    SupportedModels,
)
from .modelconfig import get_default_model_config
from .param.utils import _make_default_prior


class MissingDataNetwork(Enum):
    """Enum for the missing data network."""

    NONE = 0
    CPN = 1
    GONOGO = 2
    OPN = 3


missing_data_networks_suffix = {
    MissingDataNetwork.NONE: "",
    MissingDataNetwork.CPN: "_cpn",
    MissingDataNetwork.GONOGO: "_gonogo",
    MissingDataNetwork.OPN: "_opn",
}


default_model_config: DefaultConfigs = {"ddm": get_default_model_config("ddm")}

# TODO: Initval settings could be specified directly in model config as well.
INITVAL_SETTINGS = {
    # logit link function case
    # should never use priors with bounds,
    # so no need to take care of _log__, and _interval__ variables
    "log_logit": {
        "t": -4.0,
        "t_Intercept": -4.0,
        "v": 0.0,
        "a": 0.0,
        "a_Intercept": 0.0,
        "v_Intercept": 0.0,
        "p_outlier": -5.0,
    },
    # identity link function case,
    # need to take care of_log__ and _interval__ variables
    None: {
        "t": 0.025,
        "t_Intercept": 0.025,
        "a": 1.5,
        "a_Intercept": 1.5,
        "v_Intercept": 0.0,
        "v": 0.0,
        "p_outlier": 0.001,
    },
}

INITVAL_JITTER_SETTINGS = {
    "jitter_epsilon": 0.01,
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
        for loglik_kind_ in model_config["likelihoods"]:
            output += _show_defaults_helper(model, loglik_kind_)
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

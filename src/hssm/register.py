"""Module for registering custom models in HSSM."""

from pprint import pformat
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .defaults import (
    DefaultConfig,
    LogLik,
    LoglikConfig,
    LoglikKind,
    ParamSpec,
    default_model_config,
)
from .hssm import HSSM


def register_model(
    name: str,
    response: List[str] = ["rt", "response"],
    list_params: List[str] = None,
    choices: List[int] = [-1, 1],
    description: Optional[str] = None,
    loglik: Optional[LogLik] = None,
    loglik_kind: Optional[LoglikKind] = None,
    backend: Optional[str] = None,
    default_priors: Optional[Dict[str, ParamSpec]] = None,
    bounds: Optional[Dict[str, tuple[float, float]]] = None,
    extra_fields: Optional[List[str]] = None,
    **kwargs: Any,
) -> None:
    """Register a new model in HSSM.

    This function allows you to register a new model that can be used with the `HSSM`
    constructor.

    The new model configuration will be stored in `defaults.default_model_config`
    under the `name` key.

    Parameters
    ----------
    name : str
        Name of the model to register
    response : List[str], optional
        List of response variables, by default ["rt", "response"]
    list_params : List[str], optional
        List of parameters for the model
    choices : List[int], optional
        List of possible choices, by default [-1, 1]
    description : Optional[str], optional
        Description of the model
    loglik : Optional[LogLik], optional
        Log-likelihood function
    loglik_kind : Optional[LoglikKind], optional
        Kind of log-likelihood function
    backend : Optional[str], optional
        Backend to use ("jax" or "pytensor")
    default_priors : Optional[Dict[str, ParamSpec]], optional
        Dictionary of default priors for parameters
    bounds : Optional[Dict[str, tuple[float, float]]], optional
        Dictionary of bounds for parameters
    extra_fields : Optional[List[str]], optional
        List of extra fields required by the model
    **kwargs : Any
        Additional keyword arguments

    Raises
    ------
    ValueError
        If required parameters are missing or invalid
    """
    if list_params is None:
        raise ValueError("list_params must be provided")

    if loglik_kind is None:
        raise ValueError("loglik_kind must be provided")

    # Create the loglik config
    loglik_config: LoglikConfig = {
        "loglik": loglik,
        "backend": backend,
        "default_priors": default_priors or {},
        "bounds": bounds or {},
        "extra_fields": extra_fields,
    }

    # Create the model config
    model_config: DefaultConfig = {
        "response": response,
        "list_params": list_params,
        "choices": choices,
        "description": description,
        "likelihoods": {loglik_kind: loglik_config},
    }

    # Register the model
    default_model_config[name] = model_config

    # Validate the model by attempting to create an instance
    try:
        # Create minimal test data
        n_trials = 2
        data = pd.DataFrame(
            {
                "rt": np.abs(np.random.normal(0.8, 0.2, n_trials)),
                "response": [-1, 1],
            }
        )

        # Try to create a model instance
        HSSM(model=name, data=data, choices=choices)
    except Exception as e:
        # If model creation fails, remove the model and raise the error
        del default_model_config[name]
        raise ValueError(f"Failed to validate model '{name}': {str(e)}")



def list_registered_models() -> Dict[str, str]:
    """List all registered models and their descriptions.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping model names to their descriptions
    """
    return {
        name: config.get("description", "No description")
        for name, config in default_model_config.items()
    }


def get_model_info(name: str) -> str:
    """Get detailed information about a registered model.

    Parameters
    ----------
    name : str
        Name of the model to get information about

    Returns
    -------
    str
        Formatted string containing model information

    Raises
    ------
    ValueError
        If the model name is not found
    """
    if name not in default_model_config:
        raise ValueError(f"Model '{name}' not found")

    return f"Model: {name}\n" + pformat(default_model_config[name], indent=2)

"""Module for registering custom models in HSSM."""

from os import PathLike
from pprint import pformat
from typing import Any, Callable, Dict, Literal

import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
import pytensor

from hssm import model_meta
from hssm.config import ModelConfig
from hssm.param.param import Param

from .defaults import (
    INITVAL_JITTER_SETTINGS,
    LoglikKind,
    default_model_config,
)
from .hssm import HSSM


def register_model(
    name: str = "ddm",
    choices: list[int] | None = None,
    include: list[dict[str, Any] | Param] | None = None,
    model_config: ModelConfig | dict | None = None,
    loglik: (
        str | PathLike | Callable | pytensor.graph.Op | type[pm.Distribution] | None
    ) = None,
    loglik_kind: LoglikKind | None = None,
    p_outlier: float | dict | bmb.Prior | None = 0.05,
    lapse: dict | bmb.Prior | None = bmb.Prior("Uniform", lower=0.0, upper=20.0),
    global_formula: str | None = None,
    link_settings: Literal["log_logit"] | None = None,
    prior_settings: Literal["safe"] | None = "safe",
    extra_namespace: dict[str, Any] | None = None,
    missing_data: bool | float = False,
    deadline: bool | str = False,
    loglik_missing_data: (str | PathLike | Callable | pytensor.graph.Op | None) = None,
    process_initvals: bool = True,
    initval_jitter: float = INITVAL_JITTER_SETTINGS["jitter_epsilon"],
    **kwargs,
) -> None:
    """Register a new model in HSSM."""
    # Register the model
    model_metadata = {k: v for k, v in locals().items() if k not in ["kwargs"]}
    if kwargs:
        model_metadata.update(kwargs)
    default_model_config[name] = model_metadata

    # Validate the model by attempting to create an instance
    # try:
    #     # Create minimal test data
    #     n_trials = 2
    #     data = pd.DataFrame(
    #         {
    #             "rt": np.abs(np.random.normal(0.8, 0.2, n_trials)),
    #             "response": [-1, 1],
    #         }
    #     )

    #     # Try to create a model instance
    #     HSSM(model=name, data=data, choices=choices)
    # except Exception as e:
    #     # If model creation fails, remove the model and raise the error
    #     del default_model_config[name]
    #     raise ValueError(f"Failed to validate model '{name}': {str(e)}")


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

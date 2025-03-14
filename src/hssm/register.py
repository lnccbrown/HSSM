"""Module for registering custom models in HSSM."""

from pprint import pformat, pp
from typing import cast

from ._types import (
    DefaultConfig,
    LoglikConfigs,
    SupportedModels,
)
from .defaults import (
    default_model_config as registered_models,
)


def register_model(
    name: SupportedModels,
    response: list[str],
    list_params: list[str],
    choices: list[int],
    likelihoods: LoglikConfigs,
    description: str | None,
) -> None:
    """Register a new model in HSSM.

    Parameters
    ----------
    name : str
        Name of the model to register
    response : list[str]
        List of response variables
    list_params : list[str]
        List of parameters
    choices : list[int]
        List of possible choices
    description : str
        Description of the model
    likelihoods : LoglikConfigs
        Dictionary of likelihood configurations

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the model name already exists
    """
    # Ensure no collisions with existing models
    if name in registered_models:
        raise ValueError(f"Model '{name}' already exists")

    _config = {k: v for k, v in locals().items() if k != "name"}
    config = cast("DefaultConfig", _config)

    # TODO: validate provided configs?

    # Register the model
    registered_models[name] = config


def list_registered_models() -> None:
    """List all registered models and their descriptions."""
    pp(
        {
            name: config.get("description") or "No description"
            for name, config in registered_models.items()
        },
        sort_dicts=True,
    )


def get_model_info(name: SupportedModels | str) -> None:
    """Get detailed information about a registered model.

    Parameters
    ----------
    name : SupportedModels | str
        Name of the model to get information about

    Returns
    -------
    str
        Formatted string containing detailed model configuration metadata

    Raises
    ------
    ValueError
        If the model name is not found in the registered models
    """
    if name not in registered_models:
        raise ValueError(f"Model '{name}' not found")

    name = cast("SupportedModels", name)
    print(f"Model: {name}\n" + pformat(registered_models[name], indent=2))

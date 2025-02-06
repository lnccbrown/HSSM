"""Module for registering custom models in HSSM."""

from pprint import pformat

from .defaults import (
    DefaultConfig,
    SupportedModels,
    default_model_config,
)


def register_model(name: SupportedModels, config: DefaultConfig) -> None:
    """Register a new model in HSSM.

    Parameters
    ----------
    name : str
        Name of the model to register
    config : DefaultConfig
        Model configuration dictionary with the following structure:
        {
            "response": list[str],  # e.g. ["rt", "response"]
            "list_params": list[str],  # e.g. ["v", "a", "z", "t"]
            "choices": list[int],  # e.g. [-1, 1]
            "description": str,  # Model description
            "likelihoods": {
                "kind": {  # One of: analytical, approx_differentiable, blackbox
                    "loglik": Callable,  # Log-likelihood function
                    "backend": str | None,  # Optional, e.g. "jax" or "pytensor"
                    "bounds": dict,  # Parameter bounds
                    "default_priors": dict,  # Default priors for parameters
                    "extra_fields": list | None  # Optional extra fields
                }
            }
        }
    """
    # Validate required keys
    required_keys = set(DefaultConfig.__annotations__.keys())
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys in config: {missing_keys}")

    # TODO: validate provided configs?

    # Register the model
    default_model_config[name] = config


def list_registered_models() -> dict[SupportedModels, str]:
    """List all registered models and their descriptions.

    Returns
    -------
    dict[SupportedModels, str]
        Dictionary mapping model names to their descriptions
    """
    return {
        name: config.get("description") or "No description"
        for name, config in default_model_config.items()
    }


def get_model_info(name: SupportedModels | str) -> str:
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
    if name not in default_model_config:
        raise ValueError(f"Model '{name}' not found")

    return f"Model: {name}\n" + pformat(default_model_config[name], indent=2)

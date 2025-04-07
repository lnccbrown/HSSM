"""Module provides default configuration settings for various HSSM package models.

Each model configuration includes response variables, model parameters, choices,
description.

Functions
---------
get_ddm_config() -> DefaultConfig

get_ddm_sdv_config() -> DefaultConfig

get_full_ddm_config() -> DefaultConfig

get_lba2_config() -> DefaultConfig

get_lba3_config() -> DefaultConfig

get_angle_config() -> DefaultConfig

get_levy_config() -> DefaultConfig

get_ornstein_config() -> DefaultConfig

get_weibull_config() -> DefaultConfig

get_race_no_bias_angle_4_config() -> DefaultConfig

get_default_model_config(model_name: SupportedModels) -> DefaultConfig
"""

import importlib

from .._types import DefaultConfig, SupportedModels


def get_default_model_config(model_name: SupportedModels) -> DefaultConfig:
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
    try:
        # Construct the module name and function name based on the model name
        module_name = f"{model_name}_config"
        function_name = f"get_{module_name}"

        # Dynamically import the module
        module = importlib.import_module(f".{module_name}", package="hssm.modelconfig")

        # Retrieve the function from the module
        function = getattr(module, function_name)

        # Call and return the function
        return function()
    except ModuleNotFoundError:
        raise ValueError(f"Module for model {model_name} not found in HSSM.")
    except AttributeError:
        raise ValueError(f"Function {function_name} not found in module {module_name}.")

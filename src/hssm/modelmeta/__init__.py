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

from .._types import DefaultConfig, SupportedModels
from .angle_config import get_angle_config
from .ddm_config import get_ddm_config
from .ddm_seq2_no_bias_config import get_ddm_seq2_no_bias_config
from .ddm_svd_config import get_ddm_svd_config
from .full_ddm_config import get_full_ddm_config
from .lba2_config import get_lba2_config
from .lba3_config import get_lba3_config
from .levy_config import get_levy_config
from .ornstein_config import get_ornstein_config
from .race_no_bias_angle_4_config import get_race_no_bias_angle_4_config
from .weibull_config import get_weibull_config


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

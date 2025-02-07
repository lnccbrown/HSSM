"""Utilities for working with model configuration JSON files."""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..likelihoods.analytical import (
    logp_ddm,
    logp_ddm_sdv,
    logp_lba2,
    logp_lba3,
)
from ..likelihoods.blackbox import logp_ddm_bbox, logp_ddm_sdv_bbox, logp_full_ddm

# Map of loglik strings to their actual functions/objects
LOGLIK_MAP = {
    "logp_ddm": logp_ddm,
    "logp_ddm_sdv": logp_ddm_sdv,
    "logp_lba2": logp_lba2,
    "logp_lba3": logp_lba3,
    "logp_ddm_bbox": logp_ddm_bbox,
    "logp_ddm_sdv_bbox": logp_ddm_sdv_bbox,
    "logp_full_ddm": logp_full_ddm,
}

def convert_json_values(value: Any) -> Any:
    """Convert special JSON string values back to Python types.

    Handles:
    - 'inf', 'Infinity', '"inf"', '"Infinity"' -> float('inf')
    - '-inf', '-Infinity', '"-inf"', '"-Infinity"' -> float('-inf')
    """
    if isinstance(value, str):
        # Remove any quotes
        value = value.strip('"')
        if value.lower() in ('inf', 'infinity'):
            return np.float_('inf')
        elif value.lower() in ('-inf', '-infinity'):
            return np.float_('-inf')
    return value

def convert_json_dict(data: Dict) -> Dict:
    """Recursively convert all values in a dictionary."""
    result = {}
    for key, value in data.items():
        if key == "loglik" and isinstance(value, str):
            # Convert loglik string to actual function/object
            if value in LOGLIK_MAP:
                result[key] = LOGLIK_MAP[value]
            else:
                # For ONNX models or other paths, keep as string
                result[key] = value
        elif key == "bounds" and isinstance(value, dict):
            # Convert bounds lists to tuples
            result[key] = {param: tuple(bounds) for param, bounds in value.items()}
        elif isinstance(value, dict):
            result[key] = convert_json_dict(value)
        elif isinstance(value, list):
            result[key] = [convert_json_values(item) for item in value]
        else:
            result[key] = convert_json_values(value)
    return result

def load_model_config(model_name: str) -> Dict:
    """Load a model configuration from its JSON file.

    Parameters
    ----------
    model_name : str
        Name of the model whose configuration to load

    Returns
    -------
    dict
        The model configuration with proper Python types

    Raises
    ------
    FileNotFoundError
        If the model configuration file does not exist
    """
    config_path = Path(__file__).parent / f"{model_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No configuration file found for model: {model_name}")

    with open(config_path) as f:
        config = json.load(f)

    return convert_json_dict(config)

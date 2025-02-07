"""Split default_model_config into separate JSON files."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from hssm.defaults import default_model_config

def convert_to_json_serializable(obj: Any) -> Any:
    """Convert objects to JSON serializable format."""
    if callable(obj):
        return obj.__name__
    elif hasattr(obj, '__dict__'):
        return str(obj)
    elif isinstance(obj, float) and obj == np.float_('inf'):
        return 'inf'
    elif isinstance(obj, float) and obj == np.float_('-inf'):
        return '-inf'
    return obj

def make_json_serializable(data: dict) -> dict:
    """Recursively convert dictionary values to JSON serializable format."""
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = make_json_serializable(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [convert_to_json_serializable(item) for item in value]
        else:
            result[key] = convert_to_json_serializable(value)
    return result

def main():
    """Split and save configs."""
    config_dir = Path(__file__).parent
    config_dir.mkdir(exist_ok=True)

    for model_name, config in default_model_config.items():
        # Convert config to JSON serializable format
        json_config = make_json_serializable(config)

        # Save to file
        output_file = config_dir / f"{model_name}.json"
        with open(output_file, 'w') as f:
            json.dump(json_config, f, indent=2)

if __name__ == "__main__":
    main()

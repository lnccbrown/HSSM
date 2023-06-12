from hssm.wfpt.config import default_model_config
from pathlib import Path


def test_model_paths():
    """Ensures all default model onnx files exist in the correct path."""
    for model, config in default_model_config.items():
        if config["loglik_kind"] == "approx_differentiable":
            if model == "custom_angle":
                assert (
                    config["loglik"] is None
                ), "The `loglik` should be None for `custom_angle` model."
            else:
                assert Path(config["loglik"]).exists()

from hssm.wfpt.config import default_model_config


def test_model_paths():
    """Ensures all default model onnx files exist in the correct path."""
    for model, config in default_model_config.items():
        if config["loglik_kind"] == "approx_differentiable":
            if model == "custom_angle":
                assert (
                    config["loglik_path"] is None
                ), "The `loglik_path` should be None for `custom_angle` model."
            else:
                assert config["loglik_path"].exists()

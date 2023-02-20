default_model_config = {
    "analytical": {
        "model": "base",
        "list_params": ["v", "sv", "a", "z", "t"],
        "backend": "pytensor",
        "formula": "c(rt,response)  ~ 1",
        "priors": {
            "v": {"name": "Uniform", "upper": -3.0, "lower": 3.0},
            "sv": {"name": "Uniform", "upper": 0.0, "lower": 1.2},
            "a": {"name": "Uniform", "upper": 0.5, "lower": 2.0},
            "z": {"name": "Uniform", "upper": 0.1, "lower": 0.9},
            "t": {"name": "Uniform", "upper": 0.0, "lower": 2.0},
        },
    },
    "lan": {
        "model": "test.onnx",
        "list_params": ["v", "sv", "a", "z", "theta"],
        "backend": "jax",
        "formula": "c(rt,response)  ~ 1",
        "priors": {
            "v": {"name": "Uniform", "upper": -3.0, "lower": 3.0},
            "sv": {"name": "Uniform", "upper": 0.0, "lower": 1.2},
            "a": {"name": "Uniform", "upper": 0.5, "lower": 2.0},
            "z": {"name": "Uniform", "upper": 0.1, "lower": 0.9},
            "theta": {"name": "Uniform", "upper": 0.0, "lower": 2.0},
        },
    },
}

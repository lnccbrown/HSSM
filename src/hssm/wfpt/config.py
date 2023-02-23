default_model_config = {
    "analytical": {
        "model": "base",
        "list_params": ["v", "sv", "a", "z", "t"],
        "backend": "pytensor",
        "formula": "c(rt,response)  ~ 1",
        "prior": {
            "v": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "sv": {"name": "Uniform", "lower": 0.0, "upper": 1.2},
            "a": {"name": "Uniform", "lower": 0.5, "upper": 2.01},
            "z": {"name": "Uniform", "lower": 0.1, "upper": 0.9},
            "t": {"name": "Uniform", "lower": 0.0, "upper": 2.01},
        },
        "link": {param: "identity" for param in ["v", "sv", "a", "z", "t"]},
    },
    "lan": {
        "model": "test.onnx",
        "list_params": ["v", "sv", "a", "z", "theta"],
        "backend": "jax",
        "formula": "c(rt,response)  ~ 1",
        "prior": {
            "v": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "sv": {"name": "Uniform", "lower": 0.0, "upper": 1.2},
            "a": {"name": "Uniform", "lower": 0.5, "upper": 2.01},
            "z": {"name": "Uniform", "lower": 0.1, "upper": 0.9},
            "theta": {"name": "Uniform", "lower": 0.0, "upper": 2.01},
        },
        "link": {param: "identity" for param in ["v", "sv", "a", "z", "theta"]},
    },
}

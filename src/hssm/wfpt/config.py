model_config = {
    "analytical": {
        "model": "base",
        "list_params": ["v", "sv", "a", "z", "t"],
        "backend": "pytensor",
        "formula": "c(rt,response)  ~ 1",
        "priors": {
            "v": ["Uniform", -3, 3],
            "sv": ["Uniform", 0.0, 1.2],
            "a": ["Uniform", 0.5, 2.0],
            "z": ["Uniform", 0.1, 0.9],
            "t": ["Uniform", 0.0, 2.0],
        },
    },
    "lan": {
        "model": "test.onnx",
        "list_params": ["v", "sv", "a", "z", "theta"],
        "backend": "jax",
        "formula": "c(rt,response)  ~ 1",
        "priors": {
            "v": ["Uniform", -3, 3],
            "sv": ["Uniform", 0.0, 1.2],
            "a": ["Uniform", 0.5, 2.0],
            "z": ["Uniform", 0.1, 0.9],
            "theta": ["Uniform", 0.0, 2.0],
        },
    },
}

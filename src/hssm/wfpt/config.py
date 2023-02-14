model_config = {
    "analytical": {
        "model": "base",
        "list_params": ["v", "sv", "a", "z", "t"],
        "backend": "pytensor",
        "formula": "c(rt,response)  ~ 1",
        "priors": {
            "v": [-3.0, 3.0],
            "sv": [0.0, 1.2],
            "a": [0.5, 2.0],
            "z": [0.1, 0.9],
            "t": [0.0, 2.0],
        },
    },
    "lan": {
        "model": "test.onnx",
        "list_params": ["v", "sv", "a", "z", "theta"],
        "backend": "jax",
        "formula": "c(rt,response)  ~ 1",
        "priors": {
            "v": [-3.0, 3.0],
            "sv": [0.0, 1.2],
            "a": [0.5, 2.0],
            "z": [0.1, 0.9],
            "theta": [0.0, 2.0],
        },
    },
}

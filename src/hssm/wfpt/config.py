"""
Add specification
"""


default_model_config = {
    "ddm": {
        "loglik_kind": "analytical",
        "list_params": ["v", "sv", "a", "z", "t"],
        "backend": "pytensor",
        "formula": "c(rt,response)  ~ 1",
        "prior": {
            "v": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "sv": {"name": "Uniform", "lower": 0.3, "upper": 2.5},
            "a": {"name": "Uniform", "lower": 0.1, "upper": 0.9},
            "z": {"name": "Uniform", "lower": 0.1, "upper": 0.9},
            "t": {"name": "Uniform", "lower": 0.0, "upper": 2.0},
        },
    },
    "angle": {
        "loglik_kind": "approx_differentiable",
        "loglik_path": "test.onnx",
        "list_params": ["v", "a", "z", "t", "theta"],
        "backend": "jax",
        "formula": "c(rt,response)  ~ 1",
        "prior": {
            "v": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "a": {"name": "Uniform", "lower": 0.3, "upper": 3.0},
            "z": {"name": "Uniform", "lower": 0.2, "upper": 0.9},
            "t": {"name": "Uniform", "lower": 0.001, "upper": 2.0},
            "theta": {"name": "Uniform", "lower": -0.1, "upper": 1.3},
        },
    },
}

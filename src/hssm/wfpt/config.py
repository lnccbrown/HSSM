"""
Default configurations for models in HSSM class
"""

from pathlib import Path

# Ensure the absolute paths of the models are correct
ONNX_Model_Path = Path(__file__).parent / "lan/onnx_models"

default_model_config: dict = {
    "ddm": {
        "loglik_kind": "analytical",
        "list_params": ["v", "sv", "a", "z", "t"],
        "backend": "pytensor",
        "default_prior": {
            "v": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "sv": 0,
            "a": {"name": "Uniform", "lower": 0.30, "upper": 2.5},
            "z": {"name": "Uniform", "lower": 0.10, "upper": 0.9},
            "t": {"name": "Uniform", "lower": 0.0, "upper": 2.0},
        },
        "default_boundaries": {
            "v": (-3.0, 3.0),
            "sv": (0.0, 1.0),
            "a": (0.3, 2.5),
            "z": (0.1, 0.9),
            "t": (0.0, 2.0),
        },
    },
    "angle": {
        "loglik_kind": "approx_differentiable",
        "loglik_path": ONNX_Model_Path / "angle.onnx",
        "list_params": ["v", "a", "z", "t", "theta"],
        "backend": "jax",
        "default_prior": {
            "v": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "a": {"name": "Uniform", "lower": 0.3, "upper": 3.0},
            "z": {"name": "Uniform", "lower": 0.1, "upper": 0.9},
            "t": {"name": "Uniform", "lower": 0.001, "upper": 2.0},
            "theta": {"name": "Uniform", "lower": -0.1, "upper": 1.3},
        },
        "default_boundaries": {
            "v": (-3.0, 3.0),
            "a": (0.3, 3.0),
            "z": (0.1, 0.9),
            "t": (0.001, 2.0),
            "theta": (-0.1, 1.3),
        },
    },
    "levy": {
        "loglik_kind": "approx_differentiable",
        "loglik_path": ONNX_Model_Path / "levy.onnx",
        "list_params": ["v", "a", "z", "alpha", "t"],
        "backend": "jax",
        "default_prior": {
            "v": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "a": {"name": "Uniform", "lower": 0.3, "upper": 3.0},
            "z": {"name": "Uniform", "lower": 0.1, "upper": 0.9},
            "alpha": {"name": "Uniform", "lower": 1.0, "upper": 2.0},
            "t": {"name": "Uniform", "lower": 1e-3, "upper": 2.0},
        },
        "default_boundaries": {
            "v": (-3.0, 3.0),
            "a": (0.3, 3.0),
            "z": (0.1, 0.9),
            "alpha": (1.0, 2.0),
            "t": (1e-3, 2.0),
        },
    },
    "ornstein": {
        "loglik_kind": "approx_differentiable",
        "loglik_path": ONNX_Model_Path / "ornstein.onnx",
        "list_params": ["v", "a", "z", "g", "t"],
        "backend": "jax",
        "default_prior": {
            "v": {"name": "Uniform", "lower": -2.0, "upper": 2.0},
            "a": {"name": "Uniform", "lower": 0.3, "upper": 3.0},
            "z": {"name": "Uniform", "lower": 0.1, "upper": 0.9},
            "g": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
            "t": {"name": "Uniform", "lower": 1e-3, "upper": 2.0},
        },
        "default_boundaries": {
            "v": (-2.0, 2.0),
            "a": (0.3, 3.0),
            "z": (0.1, 0.9),
            "g": (-1.0, 1.0),
            "t": (1e-3, 2.0),
        },
    },
    "weibull": {
        "loglik_kind": "approx_differentiable",
        "loglik_path": ONNX_Model_Path / "weibull.onnx",
        "list_params": ["v", "a", "z", "t", "alpha", "beta"],
        "backend": "jax",
        "default_prior": {
            "v": {"name": "Uniform", "lower": -2.5, "upper": 2.5},
            "a": {"name": "Uniform", "lower": 0.3, "upper": 2.5},
            "z": {"name": "Uniform", "lower": 0.2, "upper": 0.8},
            "t": {"name": "Uniform", "lower": 1e-3, "upper": 2.0},
            "alpha": {"name": "Uniform", "lower": 0.31, "upper": 4.99},
            "beta": {"name": "Uniform", "lower": 0.31, "upper": 6.99},
        },
        "default_boundaries": {
            "v": (-2.5, 2.5),
            "a": (0.3, 2.5),
            "z": (0.2, 0.8),
            "t": (1e-3, 2.0),
            "alpha": (0.31, 4.99),
            "beta": (0.31, 6.99),
        },
    },
    "race_no_bias_angle_4": {
        "loglik_kind": "approx_differentiable",
        "loglik_path": ONNX_Model_Path / "race_no_bias_angle_4.onnx",
        "list_params": ["v0", "v1", "v2", "v3", "a", "z", "ndt", "theta"],
        "backend": "jax",
        "default_prior": {
            "v0": {"name": "Uniform", "lower": 0.0, "upper": 2.5},
            "v1": {"name": "Uniform", "lower": 0.0, "upper": 2.5},
            "v2": {"name": "Uniform", "lower": 0.0, "upper": 2.5},
            "v3": {"name": "Uniform", "lower": 0.0, "upper": 2.5},
            "a": {"name": "Uniform", "lower": 1.0, "upper": 3.0},
            "z": {"name": "Uniform", "lower": 0.0, "upper": 0.9},
            "ndt": {"name": "Uniform", "lower": 0.0, "upper": 2.0},
            "theta": {"name": "Uniform", "lower": -0.1, "upper": 1.45},
        },
        "default_boundaries": {
            "v0": (0.0, 2.5),
            "v1": (0.0, 2.5),
            "v2": (0.0, 2.5),
            "v3": (0.0, 2.5),
            "a": (1.0, 3.0),
            "z": (0.0, 0.9),
            "ndt": (0.0, 2.0),
            "theta": (-0.1, 1.45),
        },
    },
    "ddm_seq2_no_bias": {
        "loglik_kind": "approx_differentiable",
        "loglik_path": ONNX_Model_Path / "ddm_seq2_no_bias.onnx",
        "list_params": ["vh", "vl1", "vl2", "a", "t"],
        "backend": "jax",
        "default_prior": {
            "vh": {"name": "Uniform", "lower": -4.0, "upper": 4.0},
            "vl1": {"name": "Uniform", "lower": -4.0, "upper": 4.0},
            "vl2": {"name": "Uniform", "lower": -4.0, "upper": 4.0},
            "a": {"name": "Uniform", "lower": 0.3, "upper": 2.5},
            "t": {"name": "Uniform", "lower": 0.0, "upper": 2.0},
        },
        "default_boundaries": {
            "vh": (-4.0, 4.0),
            "vl1": (-4.0, 4.0),
            "vl2": (-4.0, 4.0),
            "a": (0.3, 2.5),
            "t": (0.0, 2.0),
        },
    },
}

"""
Default configurations for models in HSSM class
"""
from huggingface_hub import hf_hub_download
from typing import Any, Literal

from hssm import wfpt

REPO_ID = "Aisulu/hssm_onnx_models"


def download_hf(path: str):
    return hf_hub_download(repo_id=REPO_ID, filename=path)


SupportedModels = Literal[
    "ddm",
    "angle",
    "levy",
    "ornstein",
    "weibull",
    "race_no_bias_angle_4",
    "ddm_seq2_no_bias",
    "custom",
]

ConfigParams = Literal[
    "loglik",
    "loglik_kind",
    "loglik_path",
    "list_params",
    "backend",
    "bounds",
]

Config = dict[ConfigParams, Any]

default_model_config: dict[SupportedModels, Config] = {
    "custom_analytical": {
        "loglik": None,
        "loglik_kind": "analytical",
        "list_params": ["v", "sv", "a", "z", "t"],
        "backend": "pytensor",
        "bounds": {
            "v": (-3.0, 3.0),
            "sv": (0.0, 1.0),
            "a": (0.3, 2.5),
            "z": (0.1, 0.9),
            "t": (0.0, 2.0),
        },
    },
    "custom_angle": {
        "loglik_kind": "approx_differentiable",
        "loglik_path": None,
        "list_params": ["v", "a", "z", "t", "theta"],
        "backend": "jax",
        "bounds": {
            "v": (-3.0, 3.0),
            "a": (0.3, 3.0),
            "z": (0.1, 0.9),
            "t": (0.001, 2.0),
            "theta": (-0.1, 1.3),
        },
    },
    "ddm": {
        "loglik": wfpt.WFPT,
        "loglik_kind": "analytical",
        "list_params": ["v", "sv", "a", "z", "t"],
        "backend": "pytensor",
        "bounds": {
            "v": (-3.0, 3.0),
            "sv": (0.0, 1.0),
            "a": (0.3, 2.5),
            "z": (0.1, 0.9),
            "t": (0.0, 2.0),
        },
    },
    "angle": {
        "loglik_kind": "approx_differentiable",
        "loglik_path": download_hf("angle.onnx"),
        "list_params": ["v", "a", "z", "t", "theta"],
        "backend": "jax",
        "bounds": {
            "v": (-3.0, 3.0),
            "a": (0.3, 3.0),
            "z": (0.1, 0.9),
            "t": (0.001, 2.0),
            "theta": (-0.1, 1.3),
        },
    },
    "levy": {
        "loglik_kind": "approx_differentiable",
        "loglik_path": download_hf("levy.onnx"),
        "list_params": ["v", "a", "z", "alpha", "t"],
        "backend": "jax",
        "bounds": {
            "v": (-3.0, 3.0),
            "a": (0.3, 3.0),
            "z": (0.1, 0.9),
            "alpha": (1.0, 2.0),
            "t": (1e-3, 2.0),
        },
    },
    "ornstein": {
        "loglik_kind": "approx_differentiable",
        "loglik_path": download_hf("ornstein.onnx"),
        "list_params": ["v", "a", "z", "g", "t"],
        "backend": "jax",
        "bounds": {
            "v": (-2.0, 2.0),
            "a": (0.3, 3.0),
            "z": (0.1, 0.9),
            "g": (-1.0, 1.0),
            "t": (1e-3, 2.0),
        },
    },
    "weibull": {
        "loglik_kind": "approx_differentiable",
        "loglik_path": download_hf("weibull.onnx"),
        "list_params": ["v", "a", "z", "t", "alpha", "beta"],
        "backend": "jax",
        "bounds": {
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
        "loglik_path": download_hf("race_no_bias_angle_4.onnx"),
        "list_params": ["v0", "v1", "v2", "v3", "a", "z", "ndt", "theta"],
        "backend": "jax",
        "bounds": {
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
        "loglik_path": download_hf("ddm_seq2_no_bias.onnx"),
        "list_params": ["vh", "vl1", "vl2", "a", "t"],
        "backend": "jax",
        "bounds": {
            "vh": (-4.0, 4.0),
            "vl1": (-4.0, 4.0),
            "vl2": (-4.0, 4.0),
            "a": (0.3, 2.5),
            "t": (0.0, 2.0),
        },
    },
}

"""Native fixed-radius circular diffusion, with RT in seconds and angle in radians."""

from math import inf, pi

from .._types import DefaultConfig
from ..likelihoods.circular import logp_circular_diffusion
from ..simulator import _sample_circular_diffusion


def get_circular_diffusion_config() -> DefaultConfig:
    """Register the centered, unit-noise disk model and its analytical likelihood."""
    return {
        "response": ["rt", "response"],
        "choices": None,
        "response_domains": {"response": {"kind": "circular", "bounds": (-pi, pi)}},
        "list_params": ["v_x", "v_y", "a", "t"],
        "description": "Centered diffusion to a fixed circle with Cartesian drift",
        "likelihoods": {
            "analytical": {
                "loglik": logp_circular_diffusion,
                "rv": _sample_circular_diffusion,
                "backend": "jax",
                "bounds": {
                    "v_x": (-inf, inf),
                    "v_y": (-inf, inf),
                    "a": (0, inf),
                    "t": (0, inf),
                },
                "default_priors": {
                    "v_x": {"name": "Normal", "mu": 0, "sigma": 1},
                    "v_y": {"name": "Normal", "mu": 0, "sigma": 1},
                    "a": {"name": "HalfNormal", "sigma": 1, "initval": 1.0},
                    "t": {"name": "HalfNormal", "sigma": 0.2, "initval": 0.05},
                },
                "extra_fields": None,
            },
        },
    }

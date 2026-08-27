"""Default configuration for the experimental JEAM circular diffusion model."""

from math import pi

from .._types import DefaultConfig
from ..integrations.jeam import (
    logp_circular_diffusion,
    logp_circular_diffusion_jax,
    simulate_circular_diffusion,
)


def get_circular_diffusion_config() -> DefaultConfig:
    """Return the fixed-boundary JEAM circular diffusion configuration.

    HSSM parameters ``v_x`` and ``v_y`` form JEAM's Cartesian drift vector, ``a``
    maps to its fixed threshold, and ``t`` maps to nondecision time. The prototype
    fixes diffusion scale to one and drift/nondecision-time variability to zero.
    """
    return {
        "response": ["rt", "response"],
        "response_kind": "circular",
        "response_bounds": {"response": (-pi, pi)},
        "list_params": ["v_x", "v_y", "a", "t"],
        "choices": None,
        "rv": simulate_circular_diffusion,
        "description": (
            "Experimental fixed-boundary JEAM circular diffusion model with "
            "Cartesian drift"
        ),
        "default_loglik_kind": "blackbox",
        "likelihoods": {
            "analytical": {
                "loglik": logp_circular_diffusion_jax,
                "backend": "jax",
                "requires_jax_x64": True,
                "default_priors": {
                    "t": {
                        "name": "HalfNormal",
                        "sigma": 2.0,
                    },
                },
                "bounds": {
                    "v_x": (-3.0, 3.0),
                    "v_y": (-3.0, 3.0),
                    "a": (0.1, 3.0),
                    "t": (0.0, 2.0),
                },
                "extra_fields": None,
                "supported_samplers": ("pymc", "numpyro"),
            },
            "blackbox": {
                "loglik": logp_circular_diffusion,
                "backend": None,
                "default_priors": {
                    "t": {
                        "name": "HalfNormal",
                        "sigma": 2.0,
                    },
                },
                "bounds": {
                    "v_x": (-3.0, 3.0),
                    "v_y": (-3.0, 3.0),
                    "a": (0.1, 3.0),
                    "t": (0.0, 2.0),
                },
                "extra_fields": None,
                "supported_samplers": ("pymc",),
            },
        },
    }

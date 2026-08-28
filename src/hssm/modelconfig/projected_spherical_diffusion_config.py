"""Default configuration for the experimental JEAM fixed-PSDM model."""

from math import pi

from .._types import DefaultConfig
from ..integrations.jeam import (
    logp_projected_spherical_diffusion,
    simulate_projected_spherical_diffusion,
)


def get_projected_spherical_diffusion_config() -> DefaultConfig:
    """Return the fixed-boundary JEAM projected-spherical configuration.

    HSSM parameter ``v_x`` maps to JEAM's axial drift, ``v_y`` maps to its
    nonnegative projected-radial drift, ``a`` maps to the fixed threshold, and ``t``
    maps to nondecision time. Diffusion scale is fixed to one and drift/NDT variability
    to zero.
    """
    return {
        "response": ["rt", "response"],
        "response_kind": "continuous",
        "response_bounds": {"response": (0.0, pi)},
        "list_params": ["v_x", "v_y", "a", "t"],
        "choices": None,
        "rv": simulate_projected_spherical_diffusion,
        "description": (
            "Experimental fixed-boundary JEAM projected spherical diffusion model"
        ),
        "default_loglik_kind": "blackbox",
        "likelihoods": {
            "blackbox": {
                "loglik": logp_projected_spherical_diffusion,
                "backend": None,
                "default_priors": {
                    "t": {
                        "name": "HalfNormal",
                        "sigma": 2.0,
                    },
                },
                "bounds": {
                    "v_x": (-3.0, 3.0),
                    "v_y": (0.0, 3.0),
                    "a": (0.1, 3.0),
                    "t": (0.0, 2.0),
                },
                "extra_fields": None,
                "supported_samplers": ("pymc",),
            },
        },
    }

"""Native fixed-radius circular diffusion, with RT in seconds and angle in radians."""

from math import inf

from ssms.config import get_model_registry

from .._types import DefaultConfig, ResponseDomainSpec
from ..likelihoods.circular import _ANGLE_SUPPORT, logp_circular_diffusion
from ..simulator import _sample_circular_diffusion


def get_circular_diffusion_config() -> DefaultConfig:
    """Register the centered, unit-noise disk model and its analytical likelihood."""
    # Resolve on demand: other HSSM models don't require the circular producer.
    try:
        producer = get_model_registry().get("circular_diffusion")
        version = producer["observation_schema_version"]
        schema = producer["observation_schema"]
    except KeyError as exc:
        raise ImportError(
            "circular_diffusion requires an ssm-simulators build that declares "
            "its observation schema. Install matching versions of the two packages."
        ) from exc

    # This density takes positive RT in seconds, then the angle in signed radians.
    # Fail on an incompatible producer; never treat degrees as if they were radians.
    match schema:
        case (
            {
                "name": "rt",
                "kind": "continuous",
                "lower": 0,
                "lower_inclusive": False,
            } as rt,
            {"name": "response", "kind": "circular", "lower": lo, "upper": hi} as angle,
        ) if version == 1 and "upper" not in rt and (lo, hi) == _ANGLE_SUPPORT:
            response = [rt["name"], angle["name"]]
            domain: ResponseDomainSpec = {"kind": angle["kind"], "bounds": (lo, hi)}
        case _:
            raise ValueError(
                "Incompatible ssm-simulators observation schema for circular_diffusion "
                f"(version {version!r}): {schema!r}. The likelihood expects a positive "
                "rt followed by a circular response in signed radians."
            )
    return {
        "response": response,
        "choices": None,
        "response_domains": {angle["name"]: domain},
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

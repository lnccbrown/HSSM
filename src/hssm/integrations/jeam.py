"""Adapters for the experimental JEAM circular diffusion integration."""

from collections.abc import Callable
from typing import Any

import numpy as np


def _load_circular_diffusion_model() -> Callable[..., Any]:
    """Load JEAM's circular model without making it a core HSSM dependency."""
    try:
        from jeam.Models.Circular import CircularDiffusionModel
    except ModuleNotFoundError as exc:
        if exc.name == "jeam" or (
            exc.name is not None and exc.name.startswith("jeam.")
        ):
            raise ImportError(
                "The experimental JEAM integration requires the "
                "`jeam-prototype` dependency group. Install it with "
                "`uv sync --group jeam-prototype`."
            ) from exc
        raise

    return CircularDiffusionModel


def _broadcast_parameter(
    value: float | np.ndarray, name: str, n_observations: int
) -> np.ndarray:
    """Return one finite float64 value per observation."""
    try:
        values = np.asarray(value, dtype=np.float64)
        broadcast = np.broadcast_to(values, (n_observations,))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Parameter {name!r} must be scalar or broadcastable to "
            f"{n_observations} observations."
        ) from exc

    if not np.all(np.isfinite(broadcast)):
        raise ValueError(f"Parameter {name!r} must contain only finite values.")

    return broadcast


def _validate_observations(data: np.ndarray) -> np.ndarray:
    """Validate and normalize ``[rt, response]`` observations."""
    try:
        observations = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "JEAM observations must be a numeric two-column array."
        ) from exc

    if observations.ndim != 2 or observations.shape[1] != 2:
        raise ValueError(
            "JEAM observations must have shape (n_observations, 2) with columns "
            "[rt, response]."
        )
    if not np.all(np.isfinite(observations)):
        raise ValueError("JEAM observations must contain only finite values.")

    angles = observations[:, 1]
    if np.any((angles < -np.pi) | (angles >= np.pi)):
        raise ValueError("JEAM circular responses must lie in [-pi, pi).")

    return observations


def logp_circular_diffusion(
    data: np.ndarray,
    v_x: float | np.ndarray,
    v_y: float | np.ndarray,
    a: float | np.ndarray,
    t: float | np.ndarray,
) -> np.ndarray:
    """Evaluate the fixed-boundary JEAM circular diffusion log density.

    The two data columns are response time and response angle in radians. Angles use
    the half-open interval ``[-pi, pi)``. The adapter fixes JEAM's diffusion scale to
    one and both drift and nondecision-time variability to zero.

    Parameters may be scalars or arrays broadcastable to the number of observations.
    JEAM currently evaluates a single threshold value per call, so observations with
    different trial-wise thresholds are partitioned without altering their order.
    """
    observations = _validate_observations(data)
    n_observations = observations.shape[0]
    v_x_values = _broadcast_parameter(v_x, "v_x", n_observations)
    v_y_values = _broadcast_parameter(v_y, "v_y", n_observations)
    threshold_values = _broadcast_parameter(a, "a", n_observations)
    ndt_values = _broadcast_parameter(t, "t", n_observations)

    model_type = _load_circular_diffusion_model()
    model = model_type(threshold_dynamic="fixed")
    logp = np.empty(n_observations, dtype=np.float64)
    thresholds, threshold_groups = np.unique(threshold_values, return_inverse=True)

    for group, threshold in enumerate(thresholds):
        selection = threshold_groups == group
        drift = np.column_stack((v_x_values[selection], v_y_values[selection]))
        group_logp = model.joint_lpdf(
            rt=observations[selection, 0],
            theta=observations[selection, 1],
            drift_vec=drift,
            ndt=ndt_values[selection],
            threshold=float(threshold),
            decay=0.0,
            threshold_function=None,
            dt_threshold_function=None,
            s_v=0.0,
            s_t=0.0,
            sigma=1.0,
        )
        group_logp = np.asarray(group_logp, dtype=np.float64)
        expected_shape = (int(np.count_nonzero(selection)),)
        if group_logp.shape != expected_shape:
            raise RuntimeError(
                "JEAM returned an unexpected pointwise log-density shape: "
                f"expected {expected_shape}, got {group_logp.shape}."
            )
        logp[selection] = group_logp

    return logp


__all__ = ["logp_circular_diffusion"]

"""Adapters for experimental JEAM diffusion-model integrations."""

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


def _load_projected_spherical_diffusion_model() -> Callable[..., Any]:
    """Load JEAM's projected spherical model without a core dependency."""
    try:
        from jeam.Models.Spherical import ProjectedSphericalDiffusionModel
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

    return ProjectedSphericalDiffusionModel


def _load_fixed_cdm_logpdf() -> Callable[..., Any]:
    """Load JEAM's batched JAX kernel without importing it from base HSSM paths."""
    try:
        from jeam.likelihoods.jax_fixed import fixed_cdm_logpdf
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
    except ImportError as exc:
        raise ImportError(
            "The installed JEAM revision does not provide the differentiable "
            "fixed-CDM likelihood. Refresh it with "
            "`uv sync --group jeam-prototype --locked`."
        ) from exc

    return fixed_cdm_logpdf


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


def _validate_projected_spherical_observations(data: np.ndarray) -> np.ndarray:
    """Validate and normalize fixed-PSDM ``[rt, response]`` observations."""
    try:
        observations = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "JEAM projected spherical observations must be a numeric two-column array."
        ) from exc

    if observations.ndim != 2 or observations.shape[1] != 2:
        raise ValueError(
            "JEAM projected spherical observations must have shape "
            "(n_observations, 2) with columns [rt, response]."
        )
    if not np.all(np.isfinite(observations)):
        raise ValueError(
            "JEAM projected spherical observations must contain only finite values."
        )

    angles = observations[:, 1]
    if np.any((angles < 0.0) | (angles > np.pi)):
        raise ValueError("JEAM projected spherical responses must lie in [0, pi].")

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


def logp_projected_spherical_diffusion(
    data: np.ndarray,
    v_x: float | np.ndarray,
    v_y: float | np.ndarray,
    a: float | np.ndarray,
    t: float | np.ndarray,
) -> np.ndarray:
    """Evaluate the fixed-boundary JEAM PSDM log density.

    The two data columns are response time and polar response angle. The angle uses the
    closed interval ``[0, pi]``. ``v_x`` is the axial drift and ``v_y`` is JEAM's
    nonnegative projected-radial drift. The adapter fixes diffusion scale to one and
    drift/NDT variability to zero.

    Parameters may be scalars or arrays broadcastable to the observation count. Because
    JEAM evaluates one threshold per call, trial-wise thresholds are partitioned without
    changing pointwise order.
    """
    observations = _validate_projected_spherical_observations(data)
    n_observations = observations.shape[0]
    v_x_values = _broadcast_parameter(v_x, "v_x", n_observations)
    v_y_values = _broadcast_parameter(v_y, "v_y", n_observations)
    threshold_values = _broadcast_parameter(a, "a", n_observations)
    ndt_values = _broadcast_parameter(t, "t", n_observations)

    model_type = _load_projected_spherical_diffusion_model()
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


def logp_circular_diffusion_jax(
    data: Any,
    v_x: Any,
    v_y: Any,
    a: Any,
    t: Any,
) -> Any:
    """Evaluate strict fixed-CDM log densities through JEAM's JAX kernel.

    ``data`` contains HSSM observations in ``[rt, response]`` order. JEAM's public
    batched evaluator broadcasts scalar and regression-generated parameters across the
    observations. This lets HSSM classify the semi-analytical likelihood as
    ``analytical`` instead of routing it through the LAN-oriented single-trial vmap
    path. The diffusion scale is fixed to one, matching
    :func:`logp_circular_diffusion`; no finite density floor is applied.
    """
    fixed_cdm_logpdf = _load_fixed_cdm_logpdf()
    return fixed_cdm_logpdf(
        data[:, 0],
        data[:, 1],
        v_x,
        v_y,
        a,
        t,
        sigma=1.0,
    )


def _normalize_simulator_theta(theta: np.ndarray) -> np.ndarray:
    """Normalize HSSM simulator parameters to one ``[v_x, v_y, a, t]`` row."""
    try:
        parameters = np.asarray(theta, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "JEAM simulator parameters must be a numeric array with columns "
            "[v_x, v_y, a, t]."
        ) from exc

    if parameters.ndim == 1:
        if parameters.shape != (4,):
            raise ValueError(
                "JEAM simulator parameters must have shape (4,) or "
                "(n_trials, 4) with columns [v_x, v_y, a, t]."
            )
        parameters = parameters[np.newaxis, :]
    elif parameters.ndim != 2 or parameters.shape[1] != 4 or parameters.shape[0] == 0:
        raise ValueError(
            "JEAM simulator parameters must have shape (4,) or (n_trials, 4) "
            "with columns [v_x, v_y, a, t]."
        )

    if not np.all(np.isfinite(parameters)):
        raise ValueError("JEAM simulator parameters must contain only finite values.")

    return parameters


def simulate_circular_diffusion(
    theta: np.ndarray,
    random_state: int | np.random.Generator | None,
    n_replicas: int,
) -> np.ndarray:
    """Simulate fixed-boundary JEAM circular diffusion observations for HSSM.

    ``theta`` follows HSSM's ``[v_x, v_y, a, t]`` parameter order. Replicas are
    contiguous within each trial so ``ssms.hssm_support.rng_fn`` can reshape the
    returned rows to ``(..., n_trials, n_replicas, 2)`` without changing their
    association. The final two columns are always ``[rt, response]``.
    """
    parameters = _normalize_simulator_theta(theta)
    if (
        isinstance(n_replicas, (bool, np.bool_))
        or not isinstance(n_replicas, (int, np.integer))
        or n_replicas < 1
    ):
        raise ValueError("n_replicas must be a positive integer.")

    replicated = np.repeat(parameters, int(n_replicas), axis=0)
    model_type = _load_circular_diffusion_model()
    model = model_type(threshold_dynamic="fixed")
    simulated = model.simulate(
        drift_vec=replicated[:, :2],
        ndt=replicated[:, 3],
        threshold=replicated[:, 2],
        decay=0.0,
        threshold_function=None,
        s_v=0.0,
        s_t=0.0,
        sigma=1.0,
        n_sample=replicated.shape[0],
        random_state=random_state,
    )

    try:
        observations = np.asarray(
            simulated.loc[:, ["rt", "response"]], dtype=np.float64
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "JEAM simulator output must expose numeric columns [rt, response]."
        ) from exc

    expected_shape = (replicated.shape[0], 2)
    if observations.shape != expected_shape:
        raise RuntimeError(
            "JEAM returned an unexpected simulator output shape: "
            f"expected {expected_shape}, got {observations.shape}."
        )
    if not np.all(np.isfinite(observations)):
        raise RuntimeError("JEAM simulator output must contain only finite values.")
    if np.any(observations[:, 0] <= 0.0):
        raise RuntimeError("JEAM simulator response times must be positive.")
    angles = observations[:, 1]
    if np.any((angles < -np.pi) | (angles >= np.pi)):
        raise RuntimeError("JEAM simulator responses must lie in [-pi, pi).")

    return observations


def simulate_projected_spherical_diffusion(
    theta: np.ndarray,
    random_state: int | np.random.Generator | None,
    n_replicas: int,
) -> np.ndarray:
    """Simulate fixed-boundary JEAM PSDM observations for HSSM.

    ``theta`` follows HSSM's ``[v_x, v_y, a, t]`` parameter order, where ``v_y`` is
    the nonnegative projected-radial drift. Replicas remain contiguous within trials.
    The returned columns are ``[rt, response]`` with response in closed ``[0, pi]``.
    JEAM's decision-time horizon is fixed to 20 seconds; an omitted trial raises a
    targeted error because the current HSSM random-variable contract has no omission
    mask.
    """
    parameters = _normalize_simulator_theta(theta)
    if np.any(parameters[:, 1] < 0.0):
        raise ValueError(
            "JEAM projected spherical simulator parameter 'v_y' must be non-negative."
        )
    if np.any(parameters[:, 2] <= 0.0):
        raise ValueError(
            "JEAM projected spherical simulator parameter 'a' must be positive."
        )
    if np.any(parameters[:, 3] < 0.0):
        raise ValueError(
            "JEAM projected spherical simulator parameter 't' must be non-negative."
        )
    if (
        isinstance(n_replicas, (bool, np.bool_))
        or not isinstance(n_replicas, (int, np.integer))
        or n_replicas < 1
    ):
        raise ValueError("n_replicas must be a positive integer.")

    replicated = np.repeat(parameters, int(n_replicas), axis=0)
    model_type = _load_projected_spherical_diffusion_model()
    model = model_type(threshold_dynamic="fixed")
    simulated = model.simulate(
        drift_vec=replicated[:, :2],
        ndt=replicated[:, 3],
        threshold=replicated[:, 2],
        decay=0.0,
        threshold_function=None,
        s_v=0.0,
        s_t=0.0,
        sigma=1.0,
        n_sample=replicated.shape[0],
        max_time=20.0,
        random_state=random_state,
    )

    try:
        observations = np.asarray(
            simulated.loc[:, ["rt", "response"]], dtype=np.float64
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "JEAM projected spherical simulator output must expose numeric columns "
            "[rt, response]."
        ) from exc

    expected_shape = (replicated.shape[0], 2)
    if observations.shape != expected_shape:
        raise RuntimeError(
            "JEAM returned an unexpected simulator output shape: "
            f"expected {expected_shape}, got {observations.shape}."
        )

    omitted = np.all(np.isnan(observations), axis=1)
    if np.any(omitted):
        raise RuntimeError(
            "JEAM projected spherical simulator omitted "
            f"{int(np.count_nonzero(omitted))} trial(s) before max_time=20.0; "
            "HSSM predictive simulation currently requires complete output."
        )
    if not np.all(np.isfinite(observations)):
        raise RuntimeError(
            "JEAM projected spherical simulator output must contain only finite values."
        )
    if np.any(observations[:, 0] <= 0.0):
        raise RuntimeError("JEAM simulator response times must be positive.")
    angles = observations[:, 1]
    if np.any((angles < 0.0) | (angles > np.pi)):
        raise RuntimeError(
            "JEAM projected spherical simulator responses must lie in [0, pi]."
        )

    return observations


setattr(simulate_circular_diffusion, "model_name", "circular_diffusion")
# Required by ssms.hssm_support for compatibility. Circular responses are continuous,
# so there are deliberately no categorical choices to enumerate.
setattr(simulate_circular_diffusion, "choices", [])
setattr(simulate_circular_diffusion, "obs_dim", 2)

setattr(
    simulate_projected_spherical_diffusion,
    "model_name",
    "projected_spherical_diffusion",
)
setattr(simulate_projected_spherical_diffusion, "choices", [])
setattr(simulate_projected_spherical_diffusion, "obs_dim", 2)


__all__ = [
    "logp_circular_diffusion",
    "logp_circular_diffusion_jax",
    "logp_projected_spherical_diffusion",
    "simulate_circular_diffusion",
    "simulate_projected_spherical_diffusion",
]

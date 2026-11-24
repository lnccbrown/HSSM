# efpt NumPy reference backend -- vendored TEST ORACLE (not shipped in the package).
# Source: efficient-fpt @ d97a451479141acef845195610f0f9d85824844e
#         MIT License, Copyright (c) 2025 Sicheng Liu.
# Vendored verbatim except import retargeting to flatten the backend into this
# self-contained  package. Kept at efpt's ORIGINAL DEFAULT_TRUNC_NUM=100
# so it is an INDEPENDENT numerical reference for the HSSM-vendored jax kernel
# (do not sync to that kernel's local modifications).

"""Input validation for efficient-fpt model parameters."""

from __future__ import annotations

import warnings
from numbers import Number

import numpy as np


def _ensure_finite_real(name: str, value) -> float:
    """Return *value* as float if it is finite; otherwise raise."""
    if not isinstance(value, Number):
        raise ValueError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def _ensure_nonnegative_finite(name: str, value) -> float:
    """Return *value* as a finite nonnegative float; otherwise raise."""
    value = _ensure_finite_real(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _ensure_finite_array(name: str, value) -> np.ndarray:
    """Return *value* as float64 array if every entry is finite; otherwise raise."""
    arr = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_single_stage_boundaries(a: float, b: float) -> tuple[float, float]:
    """Validate symmetric single-stage boundary parameters."""
    b = _ensure_nonnegative_finite("b", b)
    if not isinstance(a, Number):
        raise ValueError(f"a must be a real number, got {type(a).__name__}")
    a = float(a)
    if np.isnan(a) or a <= 0:
        raise ValueError(f"a (boundary intercept) must be positive, got {a}")
    return a, b


def _validate_numeric_x0(x0, lower: float, upper: float) -> None:
    """Validate a numeric x0 against the initial boundaries."""
    x0 = _ensure_finite_real("x0", x0)
    if np.isfinite(lower) and np.isfinite(upper) and not (lower < x0 < upper):
        raise ValueError(
            f"x0 must be within the initial boundary ({lower}, {upper}), got {x0}"
        )


def _validate_x0(
    x0: float | dict,
    lower: float,
    upper: float,
    *,
    distribution_hint: str,
) -> None:
    """Validate x0 while allowing distribution-valued inputs."""
    if isinstance(x0, dict):
        warnings.warn(
            f"x0 is a distribution — {distribution_hint}",
            UserWarning,
            stacklevel=2,
        )
        return
    if isinstance(x0, Number):
        _validate_numeric_x0(x0, lower, upper)
        return
    raise ValueError(
        f"x0 must be a number or a distribution dict, got {type(x0).__name__}"
    )


def check_single_stage_params(
    mu: float,
    sigma: float,
    a: float,
    b: float,
    x0: float | dict,
) -> None:
    """Validate SingleStageModel constructor parameters.

    Infinite boundary values are allowed only through ``a = +inf``. When the
    initial boundaries are infinite, ``x0`` must still be finite, but the
    interval containment check is skipped.
    """
    _ensure_finite_real("mu", mu)
    sigma = _ensure_nonnegative_finite("sigma", sigma)
    del sigma  # validated for side effect only
    a, _ = _validate_single_stage_boundaries(a, b)
    _validate_x0(
        x0,
        -a,
        a,
        distribution_hint=(
            "boundary validation skipped. Ensure sampled x0 values fall within "
            "the initial interval when the boundaries are finite."
        ),
    )


def check_addm_params(
    eta: float,
    kappa: float,
    sigma: float,
    a: float,
    b: float,
    x0: float | dict,
) -> None:
    """Validate aDDModel constructor parameters.

    Raises
    ------
    ValueError
        If any parameter is out of its valid range.
    """
    _ensure_finite_real("eta", eta)
    _ensure_finite_real("kappa", kappa)
    _ensure_nonnegative_finite("sigma", sigma)
    a, _ = _validate_single_stage_boundaries(a, b)
    _validate_x0(
        x0,
        -a,
        a,
        distribution_hint=(
            "boundary validation skipped. Ensure sampled x0 values fall within "
            "(-a, a) when the initial boundaries are finite."
        ),
    )


def check_multistage_params(
    mu_array,
    node_array,
    sigma_array,
    a1,
    b1_array,
    a2,
    b2_array,
    x0=None,
    *,
    allow_infinite_boundaries: bool = False,
):
    """Check the validity of the parameters for the multi-stage model.

    Parameters
    ----------
    allow_infinite_boundaries : bool, optional
        If True, allow ``a1 = +inf`` and/or ``a2 = -inf`` as model-level edge
        cases. This is used by :class:`MultiStageModel`. Numerical quadrature
        code should keep the default ``False`` unless it explicitly supports
        infinite boundaries.
    """
    mu_array = _ensure_finite_array("mu_array", mu_array)
    node_array = _ensure_finite_array("node_array", node_array)
    sigma_array = np.asarray(sigma_array, dtype=np.float64)
    if np.any(np.isnan(sigma_array)) or np.any(np.isinf(sigma_array)):
        raise ValueError("sigma_array must contain only finite values")
    if np.any(sigma_array < 0):
        raise ValueError("sigma_array must be non-negative")
    b1_array = _ensure_finite_array("b1_array", b1_array)
    b2_array = _ensure_finite_array("b2_array", b2_array)

    d = len(mu_array)
    if d == 0:
        raise ValueError("mu_array must contain at least one stage")
    if len(node_array) != d:
        raise ValueError(f"node_array length {len(node_array)} != mu_array length {d}")
    if len(sigma_array) != d:
        raise ValueError(
            f"sigma_array length {len(sigma_array)} != mu_array length {d}"
        )
    if len(b1_array) != d:
        raise ValueError(f"b1_array length {len(b1_array)} != mu_array length {d}")
    if len(b2_array) != d:
        raise ValueError(f"b2_array length {len(b2_array)} != mu_array length {d}")
    if not all(np.diff(node_array) > 0):
        raise ValueError("node_array must be strictly increasing")
    if node_array[0] != 0:
        raise ValueError("node_array[0] must be 0")
    if d >= 2 and node_array[1] <= 0:
        raise ValueError("node_array[1] must be positive")

    if not isinstance(a1, Number):
        raise ValueError(f"a1 must be a real number, got {type(a1).__name__}")
    if not isinstance(a2, Number):
        raise ValueError(f"a2 must be a real number, got {type(a2).__name__}")
    a1 = float(a1)
    a2 = float(a2)
    if np.isnan(a1) or np.isnan(a2):
        raise ValueError("boundary intercepts must not be NaN")
    if allow_infinite_boundaries:
        if not (np.isfinite(a1) or np.isposinf(a1)):
            raise ValueError("a1 must be finite or +inf")
        if not (np.isfinite(a2) or np.isneginf(a2)):
            raise ValueError("a2 must be finite or -inf")
    else:
        if not np.isfinite(a1) or not np.isfinite(a2):
            raise ValueError("boundary intercepts must be finite")

    if np.isfinite(a1) and np.isfinite(a2) and not (a1 > a2):
        raise ValueError("initial upper boundary must be greater than lower boundary")

    if x0 is not None:
        _validate_x0(
            x0,
            a2,
            a1,
            distribution_hint=(
                "boundary validation skipped. Ensure sampled x0 values fall within "
                "the initial interval when both boundaries are finite."
            ),
        )

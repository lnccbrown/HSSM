# efpt NumPy reference backend -- vendored TEST ORACLE (not shipped in the package).
# Source: efficient-fpt @ d97a451479141acef845195610f0f9d85824844e
#         MIT License, Copyright (c) 2025 Sicheng Liu.
# Vendored verbatim except import retargeting to flatten the backend into this
# self-contained  package. Kept at efpt's ORIGINAL DEFAULT_TRUNC_NUM=100
# so it is an INDEPENDENT numerical reference for the HSSM-vendored jax kernel
# (do not sync to that kernel's local modifications).

"""Generic numerical utilities for efficient-fpt."""

from __future__ import annotations

import warnings

import numpy as np

from ._defaults import (
    DEFAULT_LAST_QUAD_ORDER,
    DEFAULT_MID_QUAD_ORDER,
)


def adaptive_interpolation(
    f,
    x_range,
    error_threshold,
    max_iterations=1000,
    initial_points=10,
    num_eval_points=1000,
):
    """
    Adaptive linear interpolation of a function `f` over a specified range `x_range`.
    The function iteratively refines the interpolation points until the maximum error
    on the evaluation grid is below the specified `error_threshold` or the maximum number of iterations is reached.
    The function returns the x-coordinates and corresponding y-coordinates of the interpolation.
    """
    x_points = np.linspace(x_range[0], x_range[1], initial_points)
    y_points = f(x_points)
    xi = np.linspace(x_range[0], x_range[1], num_eval_points)
    yi = np.interp(xi, x_points, y_points)
    iteration = 0
    while iteration < max_iterations:
        f_actual = f(xi)
        errors = np.abs(yi - f_actual)
        max_error = np.max(errors)
        if max_error <= error_threshold:
            break
        max_error_idx = np.argmax(errors)
        new_x = xi[max_error_idx]
        if np.any(np.isclose(x_points, new_x, atol=1e-12)):
            break
        new_y = f(new_x)
        idx = np.searchsorted(x_points, new_x)
        x_points = np.insert(x_points, idx, new_x)
        y_points = np.insert(y_points, idx, new_y)
        yi = np.interp(xi, x_points, y_points)
        iteration += 1
    else:
        warnings.warn(
            "Maximum iterations reached before meeting error threshold.",
            RuntimeWarning,
            stacklevel=2,
        )
    return x_points, y_points


def resolve_quadrature_orders(order_mid, order_last, order=None):
    """Resolve the repo's split quadrature-order configuration.

    Parameters
    ----------
    order_mid : int
        Order used for intermediate-stage ``q_single`` quadrature.
    order_last : int
        Order used for final-stage ``fptd_single`` quadrature.
    order : int or None, optional
        Legacy compatibility alias. When provided on its own, it maps to both
        ``order_mid`` and ``order_last``.

    Returns
    -------
    tuple[int, int]
        Resolved ``(order_mid, order_last)``.

    Notes
    -----
    The compatibility policy is:

    - ``order`` only: map to both split orders
    - split orders only: use them directly
    - mixing ``order`` with non-default split orders: raise ``ValueError``
    """
    if order is not None:
        if order_mid != DEFAULT_MID_QUAD_ORDER or order_last != DEFAULT_LAST_QUAD_ORDER:
            raise ValueError(
                "pass either legacy order or split order_mid/order_last, not both"
            )
        order_mid = order
        order_last = order

    order_mid = int(order_mid)
    order_last = int(order_last)
    if order_mid <= 0 or order_last <= 0:
        raise ValueError(
            f"quadrature orders must be positive, got order_mid={order_mid}, "
            f"order_last={order_last}"
        )
    return order_mid, order_last


def get_cpu_name():
    """Return the CPU model name as a string.

    Reads from ``/proc/cpuinfo`` on Linux, falls back to
    :func:`platform.processor` on other systems.
    """
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    import platform

    return platform.processor() or platform.machine() or "unknown"

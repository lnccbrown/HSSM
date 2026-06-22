# Vendored from efficient-fpt @ d97a451; do not edit in place — re-vendor instead.
# HSSM-authored additions to the vendored copy: ``resolve_quadrature_orders`` is
# folded in here from efficient_fpt/utils.py so the jax/ subpackage stays
# self-contained (re-vendor = one directory). Quadrature imports are repointed
# at the sibling ``_quadrature`` module.
"""Utility functions and constants for JAX implementation of efficient-fpt.

Wraps shared Gauss-Legendre quadrature constants as JAX arrays.
Provides a precision switch (float64 by default, configurable to float32).
"""

import numpy as np
import jax
import jax.numpy as jnp
from ._defaults import (
    DEFAULT_LAST_QUAD_ORDER,
    DEFAULT_MID_QUAD_ORDER,
)
from ._quadrature import (
    GAUSS_LEGENDRE_30_X as _NP_GL30_X,
    GAUSS_LEGENDRE_30_W as _NP_GL30_W,
    lgwt_lookup_table as _np_lgwt,
)


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
        if (
            order_mid != DEFAULT_MID_QUAD_ORDER
            or order_last != DEFAULT_LAST_QUAD_ORDER
        ):
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

# ---------------------------------------------------------------------------
# Precision control
# ---------------------------------------------------------------------------

def _current_jax_dtype():
    """Return the active JAX floating-point dtype without mutating config."""
    return jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


_dtype = _current_jax_dtype()

# ---------------------------------------------------------------------------
# Numerical safety constants for stage-duration computations
# ---------------------------------------------------------------------------

_DUMMY_STAGE_DURATION = 1.0  # placeholder duration for invalid/padding stages


def _refresh_quadrature_constants():
    """Refresh cached quadrature constants to match the current dtype."""
    global GAUSS_LEGENDRE_30_X, GAUSS_LEGENDRE_30_W
    GAUSS_LEGENDRE_30_X = jnp.array(_NP_GL30_X, dtype=_dtype)
    GAUSS_LEGENDRE_30_W = jnp.array(_NP_GL30_W, dtype=_dtype)


def set_jax_precision(use_x64: bool = True):
    """Set the floating-point precision for all JAX FPT computations.

    Parameters
    ----------
    use_x64 : bool, optional (default=True)
        If True, use float64 (recommended for accuracy).
        If False, use float32 (faster on GPU, lower precision).

    Notes
    -----
    This helper updates the process-wide JAX x64 flag and refreshes the cached
    quadrature constants used by this package. Call it explicitly before
    compiling or benchmarking JAX workloads when a specific precision mode is
    required.
    """
    global _dtype, _QUAD_CACHE
    jax.config.update("jax_enable_x64", use_x64)
    _dtype = _current_jax_dtype()
    _QUAD_CACHE.clear()
    _refresh_quadrature_constants()


def get_jax_dtype():
    """Return the current floating-point dtype used by JAX FPT functions."""
    return _dtype


def positive_log(values):
    """Return log(x) for x > 0, -inf for x <= 0, and nan when x is nan."""
    values = jnp.asarray(values, dtype=_dtype)
    safe = jnp.where(values > 0.0, values, 1.0)
    logs = jnp.where(values > 0.0, jnp.log(safe), -jnp.inf)
    return jnp.where(jnp.isnan(values), jnp.nan, logs)


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature
# ---------------------------------------------------------------------------

# These module-level constants are JAX arrays, which are immutable by design
# (item assignment raises TypeError), so no additional write-protection is needed.
GAUSS_LEGENDRE_30_X = None
GAUSS_LEGENDRE_30_W = None
_refresh_quadrature_constants()

_QUAD_CACHE = {}


def get_gauss_legendre_ref(order: int):
    """Return (x_ref, w_ref) on [-1, 1] as JAX arrays for the given order.

    The shared cache stores NumPy reference arrays, not JAX arrays. This is
    important because ``get_gauss_legendre_ref(...)`` is called from jitted /
    transformed code paths. Storing ``jnp.array(...)`` results in global state
    during tracing can leak tracers and break later JAX transforms.

    Precision changes are handled at the conversion boundary by casting the
    cached NumPy arrays to the current JAX dtype on return.

    Parameters
    ----------
    order : int
        Quadrature order (any order supported by the shared cache)

    Returns
    -------
    x_ref : jnp.ndarray of shape (order,)
        Reference nodes on [-1, 1]
    w_ref : jnp.ndarray of shape (order,)
        Reference weights on [-1, 1]
    """
    if order not in _QUAD_CACHE:
        x_np, w_np = _np_lgwt(order, -1.0, 1.0)
        _QUAD_CACHE[order] = (
            np.asarray(x_np, dtype=np.float64),
            np.asarray(w_np, dtype=np.float64),
        )
    x_np, w_np = _QUAD_CACHE[order]
    return jnp.asarray(x_np, dtype=_dtype), jnp.asarray(w_np, dtype=_dtype)


def lgwt_lookup_table(order: int, a: float, b: float):
    """Return scaled Gauss-Legendre nodes and weights for interval [a, b] as JAX arrays.

    Parameters
    ----------
    order : int
        Order of quadrature (any order supported by the shared cache)
    a : float
        Lower bound of integration interval
    b : float
        Upper bound of integration interval

    Returns
    -------
    x : jnp.ndarray
        Quadrature nodes scaled to [a, b]
    w : jnp.ndarray
        Quadrature weights scaled for [a, b]
    """
    x_np, w_np = _np_lgwt(order, a, b)
    return jnp.array(x_np, dtype=_dtype), jnp.array(w_np, dtype=_dtype)

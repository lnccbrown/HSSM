# efpt NumPy reference backend -- vendored TEST ORACLE (not shipped in the package).
# Source: efficient-fpt @ d97a451479141acef845195610f0f9d85824844e
#         MIT License, Copyright (c) 2025 Sicheng Liu.
# Vendored verbatim except import retargeting to flatten the backend into this
# self-contained  package. Kept at efpt's ORIGINAL DEFAULT_TRUNC_NUM=100
# so it is an INDEPENDENT numerical reference for the HSSM-vendored jax kernel
# (do not sync to that kernel's local modifications).

"""Gauss-Legendre quadrature — single source of truth for all backends.

Pre-computes and caches Gauss-Legendre nodes and weights. The order-30
reference arrays are exposed as module-level constants for the common case.
"""

import numpy as np


# Cache keyed by order; pre-populated with commonly used orders.
_GAUSS_LEGENDRE_CACHE = {
    n: np.polynomial.legendre.leggauss(n)
    for n in (1, 2, 3, 4, 5, 6, 10, 15, 20, 25, 30, 35, 40, 100)
}


def lgwt_lookup_table(order, a, b):
    """Gauss-Legendre quadrature nodes and weights on the interval [a, b].

    Parameters
    ----------
    order : int
        The order of the Gauss-Legendre quadrature.
    a, b : float
        Integration limits.

    Returns
    -------
    x : np.ndarray
        Nodes on [a, b].
    w : np.ndarray
        Weights on [a, b].
    """
    if order not in _GAUSS_LEGENDRE_CACHE:
        _GAUSS_LEGENDRE_CACHE[order] = np.polynomial.legendre.leggauss(order)
    x, w = _GAUSS_LEGENDRE_CACHE[order]
    # Map from [-1, 1] to [a, b]
    x = x * (b - a) / 2 + (b + a) / 2
    w = w * (b - a) / 2
    return x, w


# Convenience references for the default order-30 on [-1, 1].
_x30, _w30 = _GAUSS_LEGENDRE_CACHE[30]
GAUSS_LEGENDRE_30_X = np.ascontiguousarray(_x30, dtype=np.float64)
GAUSS_LEGENDRE_30_W = np.ascontiguousarray(_w30, dtype=np.float64)

# efpt NumPy reference backend -- vendored TEST ORACLE (not shipped in the package).
# Source: efficient-fpt @ d97a451479141acef845195610f0f9d85824844e
#         MIT License, Copyright (c) 2025 Sicheng Liu.
# Vendored verbatim except import retargeting to flatten the backend into this
# self-contained  package. Kept at efpt's ORIGINAL DEFAULT_TRUNC_NUM=100
# so it is an INDEPENDENT numerical reference for the HSSM-vendored jax kernel
# (do not sync to that kernel's local modifications).

from numpy import exp, sqrt, pi
import numpy as np

from ._defaults import DEFAULT_TRUNC_NUM, DEFAULT_THRESHOLD
from ._numpy_utils import positive_log

"""
Reference:
Hall, W. J. (1997). The distribution of Brownian motion on linear stopping boundaries. Sequential analysis, 16(4), 345-352.
"""


def _scalar_or_array(out):
    """Cast to float64 and unwrap 0-d arrays to Python float."""
    out = np.asarray(out, dtype=np.float64)
    return float(out) if out.ndim == 0 else out

def fptd_basic(
    t,
    mu,
    a1,
    b1,
    a2,
    b2,
    bdy,
    *,
    trunc_num=DEFAULT_TRUNC_NUM,
    threshold=DEFAULT_THRESHOLD,
    adaptive_stopping=True,
):
    """
    First passage time density of Brownian motion with drift starting at x0 = 0 to
    the upper boundary u(t) = a1 + b1 * t and the lower boundary l(t) = a2 + b2 * t
    where a1 > 0 > a2, b1 < 0 < b2
    Note: set a1 = -a2 = a, b1 = b2 = -b to recover the symmetric case
    t shoud be in (0, -(a1 - a2) / (b1 - b2)), otherwise the density is 0

    Parameters
    ----------
    adaptive_stopping : bool, optional (default=True)
        If True, stop early when the current term is smaller than `threshold`.
        If False, always compute exactly `trunc_num` terms.
    """
    a_bar = (a1 + a2) / 2
    b = (b2 - b1) / 2
    c = a1 - a2
    t = np.asarray(t, dtype=np.float64)
    valid = t > 0
    if b2 > b1:
        t_max = c / (b2 - b1)
        valid = valid & (t < t_max)
    if not np.any(valid):
        return _scalar_or_array(np.zeros_like(np.asarray(t, dtype=np.float64)))
    t_safe = np.where(valid, t, 1.0)

    if bdy == 1:
        delta = mu - b1
        factor = (
            t_safe ** (-1.5)
            / sqrt(2 * pi)
            * exp(-b / c * a1**2 + a1 * delta - 0.5 * delta**2 * t_safe)
        )
    elif bdy == -1:
        delta = -mu + b2
        factor = (
            t_safe ** (-1.5)
            / sqrt(2 * pi)
            * exp(-b / c * a2**2 - a2 * delta - 0.5 * delta**2 * t_safe)
        )
    else:
        raise ValueError("bdy must be 1 or -1")

    result = 0
    sign = 1.0
    for j in range(trunc_num):
        rj = (j + 0.5) * c + bdy * sign * a_bar
        term = sign * rj * exp((b / c - 1 / (2 * t_safe)) * rj**2)
        if adaptive_stopping and np.max(np.abs(term)) < threshold:
            break
        result += term
        sign = -sign

    return _scalar_or_array(np.where(valid, result * factor, 0.0))


def q_basic(
    x,
    mu,
    a1,
    b1,
    a2,
    b2,
    T,
    *,
    trunc_num=DEFAULT_TRUNC_NUM,
    threshold=DEFAULT_THRESHOLD,
    adaptive_stopping=True,
):
    """
    density of Brownian motion with drift at time T starting at x0 = 0
    given that it hasn't hit the upper boundary u(t) = a1 + b1 * t or the lower boundary l(t) = a2 + b2 * t
    upper boundary: u(t) = a1 + b1 * t
    lower boundary: l(t) = a2 + b2 * t
    vertical boundary: v(x) = T
    where a1 > 0 > a2, b1 < 0 < b2, T > 0
    Note: set a1 = -a2 = a, b1 = b2 = -b to recover the symmetric case
    x shoud be in (l(T), u(T)), otherwise the density is 0

    Parameters
    ----------
    adaptive_stopping : bool, optional (default=True)
        If True, stop early when the current term is smaller than `threshold`.
        If False, always compute exactly `trunc_num` terms.
    """
    if T <= 0:
        return _scalar_or_array(np.zeros_like(np.asarray(x, dtype=np.float64)))
    a_bar = (a1 + a2) / 2
    b = (b2 - b1) / 2
    b_bar = (b1 + b2) / 2
    c = a1 - a2
    x = np.asarray(x, dtype=np.float64)
    upper_T = a1 + b1 * T
    lower_T = a2 + b2 * T
    valid = (x > lower_T) & (x < upper_T)
    if b2 > b1:
        t_max = c / (b2 - b1)
        valid = valid & (T < t_max)
    if not np.any(valid):
        return _scalar_or_array(np.zeros_like(x))
    x_safe = np.where(valid, x, 0.0)
    y = x_safe - b_bar * T
    factor = exp((mu - b_bar) * x_safe - 0.5 * (mu**2 - b_bar**2) * T) / sqrt(T)
    result = 1 / sqrt(2 * pi) * exp(-(y**2) / (2 * T))
    for j in range(1, trunc_num):
        t1 = 4 * b * j * (j * c - a_bar) - (y - 2 * j * c) ** 2 / (2 * T)
        t2 = 4 * b * j * (j * c + a_bar) - (y + 2 * j * c) ** 2 / (2 * T)
        t3 = 2 * b * (2 * j - 1) * (j * c - a1) - (y + 2 * j * c - 2 * a1) ** 2 / (
            2 * T
        )
        t4 = 2 * b * (2 * j - 1) * (j * c + a2) - (y - 2 * j * c - 2 * a2) ** 2 / (
            2 * T
        )
        term = exp(t1) + exp(t2) - exp(t3) - exp(t4)
        if adaptive_stopping and np.max(np.abs(term)) < threshold:
            break
        result += term / sqrt(2 * pi)
    return _scalar_or_array(np.where(valid, result * factor, 0.0))


def fptd_single(
    t,
    mu,
    sigma,
    a1,
    b1,
    a2,
    b2,
    x0,
    bdy,
    *,
    trunc_num=DEFAULT_TRUNC_NUM,
    threshold=DEFAULT_THRESHOLD,
    adaptive_stopping=True,
):
    """
    First passage time density with sigma scaling.

    Parameters
    ----------
    adaptive_stopping : bool, optional (default=True)
        If True, stop early when the current term is smaller than `threshold`.
        If False, always compute exactly `trunc_num` terms.
    """
    mu = mu / sigma
    a1 = (a1 - x0) / sigma
    b1 = b1 / sigma
    a2 = (a2 - x0) / sigma
    b2 = b2 / sigma
    return fptd_basic(
        t,
        mu,
        a1,
        b1,
        a2,
        b2,
        bdy,
        trunc_num=trunc_num,
        threshold=threshold,
        adaptive_stopping=adaptive_stopping,
    )


def q_single(
    x,
    mu,
    sigma,
    a1,
    b1,
    a2,
    b2,
    T,
    x0,
    *,
    trunc_num=DEFAULT_TRUNC_NUM,
    threshold=DEFAULT_THRESHOLD,
    adaptive_stopping=True,
):
    """
    Non-exit probability density with sigma scaling.

    Parameters
    ----------
    adaptive_stopping : bool, optional (default=True)
        If True, stop early when the current term is smaller than `threshold`.
        If False, always compute exactly `trunc_num` terms.
    """
    x = (x - x0) / sigma
    mu = mu / sigma
    a1 = (a1 - x0) / sigma
    b1 = b1 / sigma
    a2 = (a2 - x0) / sigma
    b2 = b2 / sigma
    return (
        q_basic(
            x,
            mu,
            a1,
            b1,
            a2,
            b2,
            T,
            trunc_num=trunc_num,
            threshold=threshold,
            adaptive_stopping=adaptive_stopping,
        )
        / sigma
    )


def log_fptd_basic(
    t,
    mu,
    a1,
    b1,
    a2,
    b2,
    bdy,
    *,
    trunc_num=DEFAULT_TRUNC_NUM,
    threshold=DEFAULT_THRESHOLD,
    adaptive_stopping=True,
):
    """Safe log of :func:`fptd_basic`."""
    return positive_log(
        fptd_basic(
            t,
            mu,
            a1,
            b1,
            a2,
            b2,
            bdy,
            trunc_num=trunc_num,
            threshold=threshold,
            adaptive_stopping=adaptive_stopping,
        )
    )


def log_q_basic(
    x,
    mu,
    a1,
    b1,
    a2,
    b2,
    T,
    *,
    trunc_num=DEFAULT_TRUNC_NUM,
    threshold=DEFAULT_THRESHOLD,
    adaptive_stopping=True,
):
    """Safe log of :func:`q_basic`."""
    return positive_log(
        q_basic(
            x,
            mu,
            a1,
            b1,
            a2,
            b2,
            T,
            trunc_num=trunc_num,
            threshold=threshold,
            adaptive_stopping=adaptive_stopping,
        )
    )


def log_fptd_single(
    t,
    mu,
    sigma,
    a1,
    b1,
    a2,
    b2,
    x0,
    bdy,
    *,
    trunc_num=DEFAULT_TRUNC_NUM,
    threshold=DEFAULT_THRESHOLD,
    adaptive_stopping=True,
):
    """Safe log of :func:`fptd_single`."""
    return positive_log(
        fptd_single(
            t,
            mu,
            sigma,
            a1,
            b1,
            a2,
            b2,
            x0,
            bdy,
            trunc_num=trunc_num,
            threshold=threshold,
            adaptive_stopping=adaptive_stopping,
        )
    )


def log_q_single(
    x,
    mu,
    sigma,
    a1,
    b1,
    a2,
    b2,
    T,
    x0,
    *,
    trunc_num=DEFAULT_TRUNC_NUM,
    threshold=DEFAULT_THRESHOLD,
    adaptive_stopping=True,
):
    """Safe log of :func:`q_single`."""
    return positive_log(
        q_single(
            x,
            mu,
            sigma,
            a1,
            b1,
            a2,
            b2,
            T,
            x0,
            trunc_num=trunc_num,
            threshold=threshold,
            adaptive_stopping=adaptive_stopping,
        )
    )

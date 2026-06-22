# Vendored from efficient-fpt @ d97a451; do not edit in place — re-vendor instead.
"""
JAX implementation of single-stage first-passage time density computation.

All functions support broadcasting on spatial arguments (x, x0, t) to enable
efficient batch computation without nested vmap.

JAX uses a fixed-length series controlled only by ``trunc_num``. There is no
``adaptive_stopping`` or ``threshold`` option in this backend.

Reference:
Hall, W. J. (1997). The distribution of Brownian motion on linear stopping
boundaries. Sequential analysis, 16(4), 345-352.
"""

import jax.numpy as jnp
from .utils import get_jax_dtype, positive_log


def fptd_basic(t, mu, a1, b1, a2, b2, bdy, *, trunc_num=100):
    """
    First passage time density of Brownian motion with drift.

    Computes the density for hitting boundary `bdy` at time `t`, starting
    from x0 = 0, with upper boundary u(t) = a1 + b1*t and lower boundary
    l(t) = a2 + b2*t.

    Supports broadcasting: t can be scalar or array of any shape.
    Series terms are computed vectorized and summed over the last axis.

    Parameters
    ----------
    t : float or array
        Time(s) at which to evaluate density. Must be in (0, -(a1-a2)/(b1-b2)).
    mu : float
        Drift coefficient
    a1 : float or array (broadcastable with t)
        Upper boundary intercept (a1 > 0)
    b1 : float or array (broadcastable with t)
        Upper boundary slope (b1 < 0 for collapsing)
    a2 : float or array (broadcastable with t)
        Lower boundary intercept (a2 < 0)
    b2 : float or array (broadcastable with t)
        Lower boundary slope (b2 > 0 for collapsing)
    bdy : int
        Boundary indicator: 1 for upper, -1 for lower
    trunc_num : int
        Number of series terms (fixed, no early termination)

    Returns
    -------
    density : same shape as t
        First passage time density value(s)
    """
    # Ensure inputs are arrays for consistent broadcasting
    dtype = get_jax_dtype()
    t = jnp.asarray(t, dtype=dtype)
    a1 = jnp.asarray(a1, dtype=dtype)
    b1 = jnp.asarray(b1, dtype=dtype)
    a2 = jnp.asarray(a2, dtype=dtype)
    b2 = jnp.asarray(b2, dtype=dtype)

    a_bar = (a1 + a2) / 2
    b = (b2 - b1) / 2
    c = a1 - a2

    # Validity mask: t must be in (0, t_max) where t_max = c / (b2 - b1)
    valid = t > 0
    t_max = c / jnp.maximum(b2 - b1, 1e-30)  # avoid division by zero
    has_max = b2 > b1
    valid = jnp.where(has_max, valid & (t < t_max), valid)
    t_safe = jnp.where(valid, t, 1.0)  # use 1.0 to avoid 0**-1.5

    # Handle both boundary cases with jnp.where
    delta = jnp.where(bdy == 1, mu - b1, -mu + b2)
    a_bdy = jnp.where(bdy == 1, a1, a2)
    sign_factor = jnp.where(bdy == 1, 1.0, -1.0)

    # Compute factor (handles broadcasting on t)
    factor = (
        (t_safe**-1.5)
        / jnp.sqrt(2 * jnp.pi)
        * jnp.exp(
            -b / c * a_bdy**2 + sign_factor * a_bdy * delta - 0.5 * delta**2 * t_safe
        )
    )

    # Vectorized series computation
    # Expand all parameters to add series dimension: (...,) -> (..., 1)
    t_expanded = jnp.expand_dims(t_safe, axis=-1)
    c_expanded = jnp.expand_dims(c, axis=-1)
    a_bar_expanded = jnp.expand_dims(a_bar, axis=-1)
    b_expanded = jnp.expand_dims(b, axis=-1)

    j = jnp.arange(trunc_num)  # (trunc_num,)
    sign_j = jnp.where(j % 2 == 0, 1.0, -1.0)

    # rj computation with proper broadcasting
    rj = (j + 0.5) * c_expanded + bdy * sign_j * a_bar_expanded  # (..., trunc_num)

    # Compute series terms: shape (..., trunc_num)
    exponent = (b_expanded / c_expanded - 1 / (2 * t_expanded)) * rj**2
    terms = sign_j * rj * jnp.exp(exponent)

    # Sum over series dimension
    result = jnp.sum(terms, axis=-1) * factor

    return jnp.where(valid, result, 0.0)


def q_basic(x, mu, a1, b1, a2, b2, T, *, trunc_num=100):
    """
    Non-exit probability density (transition density).

    Computes the density of being at position x at time T, starting from x0 = 0,
    given no boundary crossing. Upper boundary u(t) = a1 + b1*t, lower boundary
    l(t) = a2 + b2*t.

    Supports broadcasting on x: x can be any shape, enabling efficient batch
    evaluation for transition matrix computation.

    Parameters
    ----------
    x : float or array
        Position(s) at which to evaluate density. Must be in (l(T), u(T)).
    mu : float
        Drift coefficient
    a1 : float or array (broadcastable with x)
        Upper boundary intercept
    b1 : float or array (broadcastable with x)
        Upper boundary slope
    a2 : float or array (broadcastable with x)
        Lower boundary intercept
    b2 : float or array (broadcastable with x)
        Lower boundary slope
    T : float
        Time duration
    trunc_num : int
        Number of series terms (fixed, no early termination)

    Returns
    -------
    density : same shape as x
        Non-exit probability density value(s)
    """
    # Ensure inputs are arrays for consistent broadcasting
    dtype = get_jax_dtype()
    x = jnp.asarray(x, dtype=dtype)
    a1 = jnp.asarray(a1, dtype=dtype)
    b1 = jnp.asarray(b1, dtype=dtype)
    a2 = jnp.asarray(a2, dtype=dtype)
    b2 = jnp.asarray(b2, dtype=dtype)

    a_bar = (a1 + a2) / 2
    b = (b2 - b1) / 2
    b_bar = (b1 + b2) / 2
    c = a1 - a2

    # Validity mask: T > 0 and x in (lower_T, upper_T)
    upper_T = a1 + b1 * T
    lower_T = a2 + b2 * T
    valid = (T > 0) & (x > lower_T) & (x < upper_T)
    t_max = c / jnp.maximum(b2 - b1, 1e-30)
    has_max = b2 > b1
    valid = jnp.where(has_max, valid & (T < t_max), valid)
    x_safe = jnp.where(valid, x, 0.0)
    T_safe = jnp.maximum(T, 1e-30)  # avoid sqrt(0) and division by 0

    y = x_safe - b_bar * T_safe

    factor = jnp.exp(
        (mu - b_bar) * x_safe - 0.5 * (mu**2 - b_bar**2) * T_safe
    ) / jnp.sqrt(T_safe)

    # Base term (j=0)
    result = 1 / jnp.sqrt(2 * jnp.pi) * jnp.exp(-(y**2) / (2 * T_safe))

    # Series terms j=1..trunc_num-1
    # Expand all parameters for broadcasting with series index
    # y, c, a_bar, a1, a2, b have shape (...), need (..., 1) for broadcasting with j
    y_expanded = jnp.expand_dims(y, axis=-1)
    c_expanded = jnp.expand_dims(c, axis=-1)
    a_bar_expanded = jnp.expand_dims(a_bar, axis=-1)
    a1_expanded = jnp.expand_dims(a1, axis=-1)
    a2_expanded = jnp.expand_dims(a2, axis=-1)
    b_expanded = jnp.expand_dims(b, axis=-1)

    j = jnp.arange(1, trunc_num)  # (trunc_num-1,)

    # Compute exponents for all terms
    # Each t1, t2, t3, t4 has shape (..., trunc_num-1)
    t1 = 4 * b_expanded * j * (j * c_expanded - a_bar_expanded) - (
        y_expanded - 2 * j * c_expanded
    ) ** 2 / (2 * T_safe)
    t2 = 4 * b_expanded * j * (j * c_expanded + a_bar_expanded) - (
        y_expanded + 2 * j * c_expanded
    ) ** 2 / (2 * T_safe)
    t3 = 2 * b_expanded * (2 * j - 1) * (j * c_expanded - a1_expanded) - (
        y_expanded + 2 * j * c_expanded - 2 * a1_expanded
    ) ** 2 / (2 * T_safe)
    t4 = 2 * b_expanded * (2 * j - 1) * (j * c_expanded + a2_expanded) - (
        y_expanded - 2 * j * c_expanded - 2 * a2_expanded
    ) ** 2 / (2 * T_safe)

    terms = jnp.exp(t1) + jnp.exp(t2) - jnp.exp(t3) - jnp.exp(t4)

    # Sum over series dimension
    result = result + jnp.sum(terms, axis=-1) / jnp.sqrt(2 * jnp.pi)

    return jnp.where(valid, result * factor, 0.0)


def fptd_single(t, mu, sigma, a1, b1, a2, b2, x0, bdy, *, trunc_num=100):
    """
    First passage time density with sigma scaling.

    Supports broadcasting on BOTH t and x0, enabling efficient computation
    of FPTD for multiple starting positions.

    Parameters
    ----------
    t : float or array
        Time(s) at which to evaluate density
    mu : float
        Drift coefficient
    sigma : float
        Diffusion coefficient
    a1 : float
        Upper boundary intercept
    b1 : float
        Upper boundary slope
    a2 : float
        Lower boundary intercept
    b2 : float
        Lower boundary slope
    x0 : float or array (broadcastable with t)
        Starting position(s)
    bdy : int
        Boundary indicator: 1 for upper, -1 for lower
    trunc_num : int
        Number of series terms (fixed, no early termination)

    Returns
    -------
    density : shape broadcast(t, x0)
        First passage time density value(s)

    Example
    -------
    For computing FPTD from multiple starting positions:
        x0_array = jnp.linspace(-0.5, 0.5, 20)
        fptds = fptd_single(t, mu, sigma, a1, b1, a2, b2, x0_array, bdy)
        # Returns array of shape (20,)
    """
    # Scale parameters
    mu_scaled = mu / sigma
    a1_scaled = (a1 - x0) / sigma
    b1_scaled = b1 / sigma
    a2_scaled = (a2 - x0) / sigma
    b2_scaled = b2 / sigma

    return fptd_basic(
        t,
        mu_scaled,
        a1_scaled,
        b1_scaled,
        a2_scaled,
        b2_scaled,
        bdy,
        trunc_num=trunc_num,
    )


def q_single(x, mu, sigma, a1, b1, a2, b2, T, x0, *, trunc_num=100):
    """
    Non-exit probability density with sigma scaling.

    **Key for efficiency**: Supports broadcasting on BOTH x and x0, enabling
    efficient transition matrix computation without nested vmap.

    Parameters
    ----------
    x : float or array
        Destination position(s)
    mu : float
        Drift coefficient
    sigma : float
        Diffusion coefficient
    a1 : float
        Upper boundary intercept
    b1 : float
        Upper boundary slope
    a2 : float
        Lower boundary intercept
    b2 : float
        Lower boundary slope
    T : float
        Time duration
    x0 : float or array (broadcastable with x)
        Starting position(s)
    trunc_num : int
        Number of series terms (fixed, no early termination)

    Returns
    -------
    density : shape broadcast(x, x0)
        Non-exit probability density value(s)

    Example: Transition Matrix Computation
    --------------------------------------
    For computing the full transition matrix P[i,j] = q(xs[i] | xs_prev[j]):

        xs = jnp.linspace(-0.5, 0.5, 30)      # destinations, shape (30,)
        xs_prev = jnp.linspace(-0.6, 0.6, 30) # sources, shape (30,)

        # Use broadcasting: xs[:, None] @ xs_prev[None, :]
        P = q_single(
            xs[:, None],      # shape (30, 1)
            mu, sigma, a1, b1, a2, b2, T,
            xs_prev[None, :], # shape (1, 30)
            trunc_num
        )  # Returns shape (30, 30)
    """
    # Scale parameters - these will broadcast with x and x0
    x_scaled = (x - x0) / sigma
    mu_scaled = mu / sigma
    a1_scaled = (a1 - x0) / sigma
    b1_scaled = b1 / sigma
    a2_scaled = (a2 - x0) / sigma
    b2_scaled = b2 / sigma

    return (
        q_basic(
            x_scaled,
            mu_scaled,
            a1_scaled,
            b1_scaled,
            a2_scaled,
            b2_scaled,
            T,
            trunc_num=trunc_num,
        )
        / sigma
    )


def log_fptd_basic(t, mu, a1, b1, a2, b2, bdy, *, trunc_num=100):
    """Safe log of :func:`fptd_basic`."""
    return positive_log(
        fptd_basic(t, mu, a1, b1, a2, b2, bdy, trunc_num=trunc_num)
    )


def log_q_basic(x, mu, a1, b1, a2, b2, T, *, trunc_num=100):
    """Safe log of :func:`q_basic`."""
    return positive_log(q_basic(x, mu, a1, b1, a2, b2, T, trunc_num=trunc_num))


def log_fptd_single(t, mu, sigma, a1, b1, a2, b2, x0, bdy, *, trunc_num=100):
    """Safe log of :func:`fptd_single`."""
    return positive_log(
        fptd_single(t, mu, sigma, a1, b1, a2, b2, x0, bdy, trunc_num=trunc_num)
    )


def log_q_single(x, mu, sigma, a1, b1, a2, b2, T, x0, *, trunc_num=100):
    """Safe log of :func:`q_single`."""
    return positive_log(
        q_single(x, mu, sigma, a1, b1, a2, b2, T, x0, trunc_num=trunc_num)
    )

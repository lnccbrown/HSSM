"""Joint exit-time and angle density of a centered, fixed-radius circular DDM."""

import jax.numpy as jnp
from scipy.special import j1, jn_zeros

# Positive zeros of J0. Compute once in SciPy; the model parameters never enter
# this calculation. Factor the slowest exponential out before taking the log.
_ROOTS = jn_zeros(0, 16)
_RATES = _ROOTS**2 / 2
_WEIGHTS = _ROOTS / j1(_ROOTS)


def _disk_logpdf(u):
    """Log first-exit time density for a zero-drift Brownian motion on a unit disk."""
    # The alternating Fourier-Bessel sum catastrophically cancels at early
    # times. The unit-disk exit time has Laplace transform 1/I0(sqrt(2s)).
    # Matching its large-s expansion to f(u) = exp(-1/(2u)) h(u)/u**2 gives
    # h(u) = 1-u/2-u**3/8+u**4/4-45*u**5/32+37*u**6/4 + O(u**7).
    # Its [3/3] Padé approximant stays accurate into the spectral overlap.
    lo, hi = (0.04, 0.05) if u.dtype == jnp.float64 else (0.09, 0.11)
    us = jnp.minimum(u, hi)
    p = ((-87 * us - 9) * us + 190) * us + 28
    q = ((-37 * us + 93) * us + 204) * us + 28
    early = -0.5 / us - 2 * jnp.log(us) + jnp.log(p / q)

    # Evaluate each formula only where its log and its derivative are safe.
    ul = jnp.maximum(u, lo)
    gaps = jnp.asarray(_RATES - _RATES[0], dtype=u.dtype)
    weights = jnp.asarray(_WEIGHTS, dtype=u.dtype)
    late = jnp.log(jnp.sum(weights * jnp.exp(-ul[..., None] * gaps), axis=-1))
    late -= ul * _RATES[0].item()
    # C1 transition so a NUTS trajectory sees no numerical seam.
    w = jnp.clip((u - lo) / (hi - lo), 0.0, 1.0)
    w = w**2 * (3 - 2 * w)
    return (1 - w) * early + w * late


def logp_circular_diffusion(data, v_x, v_y, a, t):
    """Return per-trial log density with respect to seconds and radians.

    The two columns are total RT and exit angle in [-pi, pi). Drift components
    `v_x`, `v_y`, radius `a` > 0 and nondecision time `t` >= 0 can be
    scalars or trial vectors. Start is centered and diffusion scale is one.
    See Smith (2016), https://doi.org/10.1037/rev0000023 (Eq. 23).
    """
    data = jnp.atleast_2d(data)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Circular diffusion observations must be [rt, response] rows.")
    rt, theta = data.T
    v_x, v_y, a, t = (
        jnp.broadcast_to(jnp.ravel(p), rt.shape) for p in (v_x, v_y, a, t)
    )
    valid = (
        jnp.isfinite(rt)
        & (rt > t)
        & (theta >= -jnp.pi)
        & (theta < jnp.pi)
        & jnp.isfinite(v_x)
        & jnp.isfinite(v_y)
        & jnp.isfinite(a)
        & (a > 0)
        & (t >= 0)
    )
    # NUTS may visit invalid proposals: safe operands keep inactive branches
    # from leaking NaNs through reverse-mode differentiation on other trials.
    dt, a, v_x, v_y, theta = (
        jnp.where(valid, x, safe)
        for x, safe in ((rt - t, 1.0), (a, 1.0), (v_x, 0.0), (v_y, 0.0), (theta, 0.0))
    )
    # Scaling the radius rescales exit time by a². Uniform zero-drift angle
    # contributes 1/(2π); the last two terms are the drift change of measure.
    logp = (
        _disk_logpdf(dt / a**2)
        - 2 * jnp.log(a)
        - jnp.log(2 * jnp.pi)
        + a * (v_x * jnp.cos(theta) + v_y * jnp.sin(theta))
        - 0.5 * dt * (v_x**2 + v_y**2)
    )
    return jnp.where(valid, logp, -jnp.inf)

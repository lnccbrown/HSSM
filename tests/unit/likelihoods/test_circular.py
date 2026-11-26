"""Independent numerical oracles for the fixed circular diffusion likelihood."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import i0

from hssm.likelihoods.circular import logp_circular_diffusion

# Unit-disk first-exit density f and its time derivative. Generated independently
# of the implementation by Cohen inversion at 85 decimal digits (mpmath):
# F(s) = 1 / besseli(0, sqrt(2*s)); f = invertlaplace(F,u);
# dlogf = invertlaplace(lambda s: s*F(s),u) / f.
# Covers the early tail, both blend regions, and the first-mode-dominated tail.
REFERENCE = np.array(
    [
        [0.005, -89.405868412634, 19599.498737582],
        [0.042, -5.5858230341029, 235.31631757724],
        [0.05, -4.0338680296184, 159.48630552822],
        [0.095, -0.60416479564967, 33.821028844951],
        [0.105, -0.30837570839057, 25.772414716119],
        [0.25, 0.63725274014602, -0.59394452018685],
        [1.1, -1.6477121099365, -2.8915381850613],
        [4.0, -10.03332731712, -2.8915929814734],
        [30.0, -85.214744835428, -2.8915929814734],
    ]
)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_points_and_gradients_against_inverse_laplace(dtype):
    """Check the measured joint density and all four trial-wise parameter scores."""
    with jax.enable_x64(dtype == "float64"):
        u, logf, dlogf = REFERENCE.T
        # Varied parameters catch time/angle/threshold conventions and broadcasting.
        vx = np.linspace(-1.1, 0.7, len(u))
        vy = np.linspace(0.1, -1.7, len(u))
        a = np.linspace(0.8, 1.6, len(u))
        t = np.linspace(0.05, 0.23, len(u))
        dt = a**2 * u
        theta = np.linspace(-2.9, 2.7, len(u))
        data = jnp.asarray(np.column_stack([dt + t, theta]), dtype=dtype)
        params = [jnp.asarray(p, dtype=dtype) for p in (vx, vy, a, t)]
        logp = jax.jit(logp_circular_diffusion)(data, *params)
        projection = vx * np.cos(theta) + vy * np.sin(theta)
        v2 = vx**2 + vy**2
        expected = logf - np.log(2 * np.pi * a**2) + a * projection - dt * v2 / 2
        grads = jax.grad(
            lambda *p: logp_circular_diffusion(data, *p).sum(), argnums=(0, 1, 2, 3)
        )(*params)
        scores = [
            a * np.cos(theta) - dt * vx,
            a * np.sin(theta) - dt * vy,
            projection - 2 * (u * dlogf + 1) / a,
            v2 / 2 - dlogf / a**2,
        ]
        assert logp.shape == (len(u),)
        assert logp.dtype == np.dtype(dtype)
        atol = 2e-4 if dtype == "float32" else 3e-8
        np.testing.assert_allclose(logp, expected, rtol=0, atol=atol)
        np.testing.assert_allclose(grads, scores, rtol=atol, atol=atol)


def test_support_and_seam():
    """Impossible trials have no density; adjacent valid angles join continuously."""
    with jax.enable_x64(True):
        obs = jnp.array(
            [
                [0.4, -np.pi],
                [0.4, np.pi - 1e-9],
                [0.4, np.pi],
                [0.2, 0.0],
                [0.1, 0.0],
                [0.4, np.nan],
                [np.inf, 0.0],
            ]
        )
        evaluate = lambda vx, vy, a, t: logp_circular_diffusion(obs, vx, vy, a, t)
        result = evaluate(-0.9, 0.8, 1.0, 0.2)
        assert np.isfinite(result[:2]).all()
        np.testing.assert_allclose(result[0], result[1], atol=1e-8, rtol=0)
        assert np.isneginf(result[2:]).all()
        for invalid in [
            (np.nan, 0.8, 1.0, 0.2),
            (-0.9, np.inf, 1.0, 0.2),
            (-0.9, 0.8, 0.0, 0.2),
            (-0.9, 0.8, 1.0, -0.1),
        ]:
            assert np.isneginf(evaluate(*invalid)).all()
        # Invalid rows do not contaminate the valid-trial gradients.
        score = jax.grad(lambda a: evaluate(-0.9, 0.8, a, 0.2).sum())(1.0)
        good_score = jax.grad(
            lambda a: logp_circular_diffusion(obs[:2], -0.9, 0.8, a, 0.2).sum()
        )(1.0)
        np.testing.assert_allclose(score, good_score)


def test_integrated_density_is_von_mises():
    """The joint density integrated over RT is a normalized von Mises density."""
    with jax.enable_x64(True):
        vx, vy, a, t = -1.3, 0.6, 1.7, 0.2
        nodes, weights = np.polynomial.legendre.leggauss(128)
        # Gauss-Legendre with decision time (1+x)/(1-x) maps to (0, infinity).
        dt = (1 + nodes) / (1 - nodes)
        for theta in [-2.6, 0.2, 1.4]:
            data = np.column_stack([t + dt, np.full_like(dt, theta)])
            values = np.exp(logp_circular_diffusion(data, vx, vy, a, t))
            integral = np.sum(values * weights * 2 / (1 - nodes) ** 2)
            expected = np.exp(a * (vx * np.cos(theta) + vy * np.sin(theta))) / (
                2 * np.pi * i0(a * np.hypot(vx, vy))
            )
            np.testing.assert_allclose(integral, expected, rtol=1e-7)

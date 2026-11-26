"""Parity, gradient and precision tests for `logp_ddm_jax` against `logp_ddm`.

Runs in float64 (the module fixture), unlike the float32-pinned
test_analytical.py; the float32 tests switch precision locally.
"""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

import hssm
from hssm.likelihoods.analytical import (
    LOGP_LB,
    _jax_k_large,
    _jax_k_small,
    compare_k,
    logp_ddm,
    logp_ddm_jax,
)


@contextmanager
def _precision(dtype):
    """Switch PyTensor and JAX to ``dtype``, then restore the previous settings."""
    prev_floatx = pytensor.config.floatX
    prev_jax_x64 = jax.config.read("jax_enable_x64")
    hssm.set_floatX(dtype)
    try:
        yield
    finally:
        pytensor.config.floatX = prev_floatx
        jax.config.update("jax_enable_x64", prev_jax_x64)


@pytest.fixture(scope="module", autouse=True)
def _float64_precision_guard():
    """Pin float64 for this module and restore prior precision settings."""
    with _precision("float64"):
        yield


def _compiled_logp_ddm(dtype, k_terms=20):
    """Return HSSM's PyTensor `logp_ddm`, compiled with array inputs of ``dtype``."""
    data = pt.matrix("data", dtype=dtype)
    params = [pt.vector(name, dtype=dtype) for name in "vazt"]
    logp = logp_ddm(data, *params, k_terms=k_terms)
    return pytensor.function([data, *params], logp)


def _grid(n, seed):
    """Random trials: |v| <= 10, rt - t from 1e-6 s to 10 s, and 10% rt < t."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(-10, 10, n)
    a = rng.uniform(0.3, 2.5, n)
    z = rng.uniform(0.05, 0.95, n)
    t = rng.uniform(0, 1, n)
    after_t = t + 10 ** rng.uniform(-6, 1, n)
    rt = np.where(rng.random(n) < 0.1, t * rng.uniform(0, 1, n), after_t)
    response = rng.choice([-1.0, 1.0], n)
    return np.column_stack([rt, response]), v, a, z, t


@pytest.mark.parametrize("k_terms", [20, 7])
def test_matches_logp_ddm(k_terms):
    """Per-trial parity with the PyTensor likelihood, including floor and rt < t."""
    data, v, a, z, t = _grid(6000, seed=0)
    expected = _compiled_logp_ddm("float64", k_terms)(data, v, a, z, t)

    result = np.asarray(logp_ddm_jax(data, v, a, z, t, k_terms=k_terms))

    assert result.shape == (6000,)
    assert (data[:, 0] < t).sum() > 300
    assert (expected < -60).sum() > 500  # near or at the LOGP_LB floor
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-10)


def test_response_coding_is_invariant():
    """0/1 and -1/1 responses give identical values (response > 0 is upper)."""
    data, v, a, z, t = _grid(500, seed=1)
    zero_one = data.copy()
    zero_one[:, 1] = (data[:, 1] > 0).astype(float)

    np.testing.assert_array_equal(
        logp_ddm_jax(zero_one, v, a, z, t), logp_ddm_jax(data, v, a, z, t)
    )


def test_negative_rt_is_read_as_absolute():
    """A negative response time is read as its absolute value, as in `logp_ddm`."""
    data, v, a, z, t = _grid(500, seed=7)

    np.testing.assert_array_equal(
        logp_ddm_jax(data * [-1, 1], v, a, z, t), logp_ddm_jax(data, v, a, z, t)
    )


def test_series_switch_matches_compare_k():
    """The small/large-time series choice is the one `compare_k` makes."""
    tt = np.logspace(-6, 2, 4001)
    tt_var = pt.vector("tt", dtype="float64")
    expected = pytensor.function([tt_var], compare_k(tt_var, 1e-15))(tt)

    result = _jax_k_small(jnp.asarray(tt), 1e-15) < _jax_k_large(jnp.asarray(tt), 1e-15)

    np.testing.assert_array_equal(np.asarray(result), expected)


def test_gradients_match_finite_differences():
    """jax.grad agrees with central differences away from the floor and switch."""
    data, v, a, z, t = _grid(6000, seed=2)
    logp = _compiled_logp_ddm("float64")(data, v, a, z, t)
    tt = (data[:, 0] - t) / (2 * a) ** 2
    smooth = (
        (data[:, 0] - t > 0.05) & (logp > -30) & (np.abs(tt - 1 / (8 * np.pi)) > 0.005)
    )
    idx = np.flatnonzero(smooth)[:500]
    data, params = data[idx], [jnp.asarray(p[idx]) for p in (v, a, z, t)]

    grads = jax.grad(lambda *p: jnp.sum(logp_ddm_jax(data, *p)), argnums=(0, 1, 2, 3))(
        *params
    )

    h = 1e-6
    for i, grad in enumerate(grads):
        up, down = list(params), list(params)
        up[i], down[i] = params[i] + h, params[i] - h
        fd = (logp_ddm_jax(data, *up) - logp_ddm_jax(data, *down)) / (2 * h)
        np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-6)


def test_float32_matches_logp_ddm_float32():
    """float32 values match PyTensor's float32 `logp_ddm`.

    Both carry the same float32 error near the floor (up to about 3e-3 relative
    to float64), so the reference is the existing likelihood at the same precision.
    """
    data, v, a, z, t = (x.astype(np.float32) for x in _grid(2000, seed=3))
    with _precision("float32"):
        expected = _compiled_logp_ddm("float32")(data, v, a, z, t)
        result = np.asarray(logp_ddm_jax(data, v, a, z, t))

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-3)


def test_float32_gradients_are_finite():
    """No NaN or inf gradients in float32, at the floor, near it and for rt < t."""
    data, *params = (x.astype(np.float32) for x in _grid(6000, seed=0))

    with _precision("float32"):
        grads = jax.grad(
            lambda *p: jnp.sum(logp_ddm_jax(data, *p)), argnums=(0, 1, 2, 3)
        )(*params)

    for grad in grads:
        assert np.isfinite(np.asarray(grad)).all()


@pytest.mark.parametrize(
    ("a", "z"), [(-1.0, 0.5), (1.0, -0.1), (1.0, 1.1)], ids=["a<0", "z<0", "z>1"]
)
def test_invalid_parameters_give_negative_infinity(a, z):
    """Out-of-support parameters give -inf, as `check_parameters` would."""
    data, *_ = _grid(5, seed=5)

    assert np.all(np.asarray(logp_ddm_jax(data, 0.5, a, z, 0.0)) == -np.inf)


@pytest.mark.parametrize("z", [0.0, 1.0])
def test_boundary_z_is_in_support(z):
    """z = 0 and z = 1 are valid, as in `logp_ddm`."""
    data, *_ = _grid(5, seed=5)

    assert np.all(np.isfinite(np.asarray(logp_ddm_jax(data, 0.5, 1.0, z, 0.0))))


def test_jit_and_broadcasting():
    """Scalar and per-trial parameters agree, eagerly and under jit."""
    data, *_ = _grid(50, seed=6)
    data[:, 0] = np.abs(data[:, 0]) + 0.3  # every rt beyond t = 0.2
    scalars = (0.7, 1.1, 0.4, 0.2)
    arrays = [np.full(50, p) for p in scalars]

    eager = np.asarray(logp_ddm_jax(data, *scalars))
    jitted = np.asarray(jax.jit(logp_ddm_jax)(data, *arrays))

    assert eager.shape == (50,)
    assert np.all(eager > LOGP_LB)
    np.testing.assert_allclose(jitted, eager, rtol=0, atol=1e-12)

from pathlib import Path

import pytest
import numpy as np
import jax.numpy as jnp

from hssm.distribution_utils.jax import make_vmap_func
from hssm.distribution_utils.func_utils import make_vjp_func


@pytest.fixture
def fixture_path():
    return Path(__file__).parent.parent / "fixtures"


def test_make_vmap_func(fixture_path):
    def fake_jax_callable(data, param1, param2):
        """A fake JAX callable that computes a simple operation."""
        return data * param1 * param2

    nojit_funcs = make_vmap_func(
        fake_jax_callable,
        in_axes=(0, None, None),
        params_only=False,
        return_jit=False,
    )
    assert len(nojit_funcs) == 2

    jit_funcs = make_vmap_func(
        fake_jax_callable,
        in_axes=(0, None, None),
        params_only=False,
        return_jit=True,
    )

    assert len(jit_funcs) == 3


def test_make_vjp_func_basic():
    # Simple function: f(x, y) = x * y
    def logp(data, x, y):
        return x * y

    vjp_func = make_vjp_func(logp)
    x = 2.0
    y = 3.0
    gz = 1.0  # Upstream gradient
    grads = vjp_func(0, x, y, gz=gz)
    # The gradient of x*y w.r.t x is y, w.r.t y is x
    assert np.allclose(grads[0], y * gz)
    assert np.allclose(grads[1], x * gz)

    def logp(x, y):
        return x * y

    # Test with params_only=True (should return all outputs)
    vjp_func_params = make_vjp_func(logp, params_only=True)
    grads_params = vjp_func_params(x, y, gz=gz)
    assert np.allclose(grads_params[0], y * gz)
    assert np.allclose(grads_params[1], x * gz)

    # Test with array inputs
    def logp_vec(data, x, y):
        return jnp.sum(x * y)

    vjp_func_vec = make_vjp_func(logp_vec)
    x_arr = jnp.array([1.0, 2.0, 3.0])
    y_arr = jnp.array([4.0, 5.0, 6.0])
    gz = 1.0
    grads_vec = vjp_func_vec(0, x_arr, y_arr, gz=gz)
    assert np.allclose(grads_vec[0], y_arr)
    assert np.allclose(grads_vec[1], x_arr)

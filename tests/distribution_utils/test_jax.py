from pathlib import Path

import numpy as np
import onnx
import pytensor
import pytensor.tensor as pt
import pytest

import hssm
from hssm.distribution_utils.jax import (
    make_jax_logp_ops,
    make_jax_logp_funcs_from_callable,
)
from hssm.distribution_utils.onnx import (
    make_jax_logp_funcs_from_onnx,
    make_pytensor_logp_from_onnx,
)

DECIMAL = 4
hssm.set_floatX("float32")


@pytest.fixture
def fixture_path():
    return Path(__file__).parent.parent / "fixtures"


def test_make_jax_logp_ops(fixture_path):
    """Tests whether the logp Op returned from make_jax_logp_ops with different backends
    work the same way.
    """
    model = onnx.load(fixture_path / "angle.onnx")

    jax_logp_op = make_jax_logp_ops(
        *make_jax_logp_funcs_from_onnx(model, params_is_reg=[False] * 5)
    )
    pytensor_logp = make_pytensor_logp_from_onnx(model)

    data = np.random.rand(10, 2)
    params_all_scalars = np.random.rand(5).astype(np.float32)

    jax_loglik = jax_logp_op(data, *params_all_scalars)
    pt_loglik = pytensor_logp(data, *params_all_scalars)

    np.testing.assert_array_almost_equal(
        np.asarray(jax_loglik.eval()), pt_loglik.eval(), decimal=DECIMAL
    )

    v = pt.as_tensor_variable(np.random.rand())

    params_with_v = [v, *params_all_scalars[1:]]
    data = data.astype(np.float32)

    jax_loglik = jax_logp_op(data, *params_with_v)
    pt_loglik = pytensor_logp(data, *params_with_v)

    np.testing.assert_array_almost_equal(
        pytensor.grad(jax_loglik.sum(), wrt=v).eval(),
        pytensor.grad(pt_loglik.sum(), wrt=v).eval(),
        decimal=DECIMAL,
    )

    jax_logp_op = make_jax_logp_ops(
        *make_jax_logp_funcs_from_onnx(model, params_is_reg=[True] + [False] * 4)
    )
    pytensor_logp = make_pytensor_logp_from_onnx(model)

    v = np.random.rand(10)

    jax_loglik = jax_logp_op(data, v, *params_all_scalars[1:])
    pt_loglik = pytensor_logp(data, v, *params_all_scalars[1:])

    np.testing.assert_array_almost_equal(
        jax_loglik.eval(), pt_loglik.eval(), decimal=DECIMAL
    )

    v = pt.as_tensor_variable(np.random.rand(10).astype(np.float32))

    params_with_v = params_all_scalars[1:]
    data = data.astype(np.float32)

    jax_loglik = jax_logp_op(data, v, *params_with_v)
    pt_loglik = pytensor_logp(data, v, *params_with_v)

    np.testing.assert_array_almost_equal(
        pytensor.grad(jax_loglik.sum(), wrt=v).eval(),
        pytensor.grad(pt_loglik.sum(), wrt=v).eval(),
        decimal=DECIMAL,
    )


def test_make_jax_logp_funcs_from_callable():
    import jax
    import jax.numpy as jnp

    # A fake JAX callable to test the conversion
    def jax_callable(data, param1, param2):
        return param1 * param2

    data = jnp.array([1.0, 2.0, 3.0])
    param1 = 2.0
    param2 = 3.0
    expected = param1 * param2

    # Test vmap=False, params_only=False
    nojit_funcs = make_jax_logp_funcs_from_callable(
        jax_callable,
        vmap=False,
        params_is_reg=None,
        params_only=False,
        return_jit=False,
    )
    assert len(nojit_funcs) == 2
    f, _ = nojit_funcs
    out = f(data, param1, param2)
    assert jnp.allclose(out, expected)
    grad_val = jax.grad(lambda p1: f(data, p1, param2).sum())(param1)
    assert jnp.allclose(grad_val, param2)

    # Test vmap=True, params_only=False
    nojit_funcs = make_jax_logp_funcs_from_callable(
        jax_callable,
        vmap=True,
        params_is_reg=[False, False],
        params_only=False,
        return_jit=False,
    )
    assert len(nojit_funcs) == 2
    f, _ = nojit_funcs
    out = f(data, param1, param2)
    assert jnp.allclose(out, expected)

    # Test return_jit=True
    jit_funcs = make_jax_logp_funcs_from_callable(
        jax_callable,
        vmap=True,
        params_is_reg=[False, False],
        params_only=False,
        return_jit=True,
    )
    assert len(jit_funcs) == 3
    f_jit, _, f_nojit = jit_funcs
    out_jit = f_jit(data, param1, param2)
    assert jnp.allclose(out_jit, expected)
    out_nojit = f_nojit(data, param1, param2)
    assert jnp.allclose(out_nojit, expected)

    # Test params_only=True
    with pytest.raises(ValueError, match="No vmap is needed"):
        make_jax_logp_funcs_from_callable(
            jax_callable,
            vmap=True,
            params_is_reg=[False, False],
            params_only=True,
            return_jit=False,
        )

    # Test error if vmap=True and params_is_reg=None
    with pytest.raises(
        ValueError, match="If `vmap` is True, `params_is_reg` must be provided"
    ):
        make_jax_logp_funcs_from_callable(
            jax_callable,
            vmap=True,
            params_is_reg=None,
            params_only=False,
            return_jit=False,
        )

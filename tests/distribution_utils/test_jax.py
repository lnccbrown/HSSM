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


def test_lan_logp_vjp_op_jax_linker_data_params(fixture_path):
    """Gradient graphs containing LANLogpVJPOp compile under the JAX linker.

    Regression test for #1056: VI (``pm.fit``) differentiates LANLogpOp
    symbolically, emitting LANLogpVJPOp into the compiled graph, so the VJP
    Op needs its own ``jax_funcify`` registration. Compares gradients from a
    ``mode="JAX"`` function against the default backend (which runs
    ``perform``) for the standard data + mixed scalar/regression case.
    """
    model = onnx.load(fixture_path / "angle.onnx")
    logp_op = make_jax_logp_ops(
        *make_jax_logp_funcs_from_onnx(model, params_is_reg=[True] + [False] * 4)
    )

    rng = np.random.default_rng(42)
    n = 10
    data_np = np.stack(
        [rng.uniform(0.4, 1.5, n), rng.choice([-1.0, 1.0], n)], axis=1
    ).astype(np.float32)
    v_np = rng.normal(0.5, 0.1, n).astype(np.float32)

    data = pt.matrix("data")
    v = pt.vector("v")
    a, z, t, theta = (pt.scalar(name) for name in ["a", "z", "t", "theta"])

    grads = pytensor.grad(
        logp_op(data, v, a, z, t, theta).sum(), wrt=[v, a, z, t, theta]
    )
    inputs = [data, v, a, z, t, theta]
    values = [data_np, v_np, *(np.float32(x) for x in (1.2, 0.5, 0.2, 0.1))]

    f_default = pytensor.function(inputs, grads)
    f_jax = pytensor.function(inputs, grads, mode="JAX")

    for g_default, g_jax in zip(f_default(*values), f_jax(*values)):
        np.testing.assert_array_almost_equal(
            np.asarray(g_jax), np.asarray(g_default), decimal=DECIMAL
        )


def test_lan_logp_vjp_op_jax_linker_scalars_only(fixture_path):
    """The scalars-only CPN branch (``gz=None``) compiles under the JAX linker."""
    model = onnx.load(fixture_path / "ddm_cpn.onnx")
    logp_op = make_jax_logp_ops(
        *make_jax_logp_funcs_from_onnx(
            model, params_is_reg=[False] * 4, params_only=True
        )
    )

    v, a, z, t = (pt.scalar(name) for name in ["v", "a", "z", "t"])
    grads = pytensor.grad(logp_op(None, v, a, z, t).sum(), wrt=[v, a, z, t])
    values = [np.float32(x) for x in (0.5, 1.2, 0.5, 0.2)]

    f_default = pytensor.function([v, a, z, t], grads)
    f_jax = pytensor.function([v, a, z, t], grads, mode="JAX")

    for g_default, g_jax in zip(f_default(*values), f_jax(*values)):
        np.testing.assert_array_almost_equal(
            np.asarray(g_jax), np.asarray(g_default), decimal=DECIMAL
        )


def test_lan_logp_vjp_op_jax_linker_single_output():
    """A single differentiable parameter yields a single-output VJP node.

    The funcified VJP must return a bare array (not a 1-tuple) in this case,
    because PyTensor's generated code assigns without tuple unpacking.
    """

    def jax_callable(data, v):
        # Single-trial signature: data has shape (2,), v is a scalar.
        return -((data[..., 0] - v) ** 2)

    logp_op = make_jax_logp_ops(
        *make_jax_logp_funcs_from_callable(
            jax_callable, vmap=True, params_is_reg=[True]
        )
    )

    rng = np.random.default_rng(7)
    n = 10
    data_np = np.stack(
        [rng.uniform(0.4, 1.5, n), rng.choice([-1.0, 1.0], n)], axis=1
    ).astype(np.float32)
    v_np = rng.normal(0.5, 0.1, n).astype(np.float32)

    data = pt.matrix("data")
    v = pt.vector("v")
    grad_v = pytensor.grad(logp_op(data, v).sum(), wrt=v)

    f_default = pytensor.function([data, v], grad_v)
    f_jax = pytensor.function([data, v], grad_v, mode="JAX")

    np.testing.assert_array_almost_equal(
        np.asarray(f_jax(data_np, v_np)),
        np.asarray(f_default(data_np, v_np)),
        decimal=DECIMAL,
    )


def test_lan_logp_vjp_op_jax_linker_n_params():
    """Extra-field inputs are excluded from the VJP outputs (n_params contract).

    Mirrors the RLDM builder: an already-vectorized callable, a VJP built
    with ``n_params``, and an Op whose gradient is undefined for the extra
    trial-wise field.
    """
    import jax
    from pytensor.gradient import NullTypeGradError

    from hssm.distribution_utils.func_utils import make_vjp_func

    n_params = 2

    def jax_callable(data, p1, p2, extra):
        # Already vectorized over trials; extra is a trial-wise field.
        return -((data[:, 0] - p1) ** 2) * p2 + extra

    vjp_callable = make_vjp_func(jax_callable, params_only=False, n_params=n_params)
    logp_op = make_jax_logp_ops(
        logp=jax.jit(jax_callable),
        logp_vjp=jax.jit(vjp_callable),
        logp_nojit=jax_callable,
        n_params=n_params,
    )

    rng = np.random.default_rng(11)
    n = 10
    data_np = np.stack(
        [rng.uniform(0.4, 1.5, n), rng.choice([-1.0, 1.0], n)], axis=1
    ).astype(np.float32)
    extra_np = rng.normal(0.0, 1.0, n).astype(np.float32)

    data = pt.matrix("data")
    p1, p2 = (pt.scalar(name) for name in ["p1", "p2"])
    extra = pt.vector("extra")

    loglik = logp_op(data, p1, p2, extra)
    grads = pytensor.grad(loglik.sum(), wrt=[p1, p2])
    inputs = [data, p1, p2, extra]
    values = [data_np, np.float32(0.5), np.float32(1.5), extra_np]

    f_default = pytensor.function(inputs, grads)
    f_jax = pytensor.function(inputs, grads, mode="JAX")

    for g_default, g_jax in zip(f_default(*values), f_jax(*values)):
        np.testing.assert_array_almost_equal(
            np.asarray(g_jax), np.asarray(g_default), decimal=DECIMAL
        )

    # The extra field sits past n_params, so its gradient is undefined.
    with pytest.raises(NullTypeGradError):
        pytensor.grad(loglik.sum(), wrt=extra)

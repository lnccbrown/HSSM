import warnings
from pathlib import Path

import jax
import numpy as np
import onnx
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.gradient import NullTypeGradError

import hssm
from hssm.distribution_utils.func_utils import make_vjp_func
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


def _rt_choice_data(seed: int, n: int = 10) -> np.ndarray:
    """Build a two-column [rt, choice] array for gradient tests."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.uniform(0.4, 1.5, n), rng.choice([-1.0, 1.0], n)], axis=1
    ).astype(np.float32)


def _assert_jax_matches_default(inputs, grads, values):
    """Assert mode="JAX" gradients match the default (perform) backend."""
    f_default = pytensor.function(inputs, grads)
    f_jax = pytensor.function(inputs, grads, mode="JAX")
    for g_default, g_jax in zip(f_default(*values), f_jax(*values)):
        np.testing.assert_array_almost_equal(
            np.asarray(g_jax), np.asarray(g_default), decimal=DECIMAL
        )


def test_lan_logp_vjp_op_jax_linker_data_params(fixture_path):
    """Gradient graphs containing LANLogpVJPOp compile under the JAX linker.

    Regression test for #1056: VI (``pm.fit``) differentiates LANLogpOp
    symbolically, emitting LANLogpVJPOp into the compiled graph, so the VJP
    Op needs its own ``jax_funcify`` registration. This covers the standard
    data + mixed scalar/regression case.
    """
    model = onnx.load(fixture_path / "angle.onnx")
    logp_op = make_jax_logp_ops(
        *make_jax_logp_funcs_from_onnx(model, params_is_reg=[True] + [False] * 4)
    )

    data = pt.matrix("data")
    v = pt.vector("v")
    a, z, t, theta = (pt.scalar(name) for name in ["a", "z", "t", "theta"])

    grads = pytensor.grad(
        logp_op(data, v, a, z, t, theta).sum(), wrt=[v, a, z, t, theta]
    )
    rng = np.random.default_rng(42)
    values = [
        _rt_choice_data(42),
        rng.normal(0.5, 0.1, 10).astype(np.float32),
        *(np.float32(x) for x in (1.2, 0.5, 0.2, 0.1)),
    ]

    _assert_jax_matches_default([data, v, a, z, t, theta], grads, values)


def test_lan_logp_vjp_op_jax_linker_scalars_only(fixture_path):
    """The scalars-only CPN branch compiles under the JAX linker."""
    model = onnx.load(fixture_path / "ddm_cpn.onnx")
    logp_op = make_jax_logp_ops(
        *make_jax_logp_funcs_from_onnx(
            model, params_is_reg=[False] * 4, params_only=True
        )
    )

    v, a, z, t = (pt.scalar(name) for name in ["v", "a", "z", "t"])
    grads = pytensor.grad(logp_op(None, v, a, z, t).sum(), wrt=[v, a, z, t])
    values = [np.float32(x) for x in (0.5, 1.2, 0.5, 0.2)]

    _assert_jax_matches_default([v, a, z, t], grads, values)


def test_lan_logp_vjp_op_jax_linker_params_only_regression(fixture_path):
    """The no-data + regression-params CPN branch compiles under the JAX linker."""
    model = onnx.load(fixture_path / "ddm_cpn.onnx")
    logp_op = make_jax_logp_ops(
        *make_jax_logp_funcs_from_onnx(
            model, params_is_reg=[True] + [False] * 3, params_only=True
        )
    )

    v = pt.vector("v")
    a, z, t = (pt.scalar(name) for name in ["a", "z", "t"])
    grads = pytensor.grad(logp_op(None, v, a, z, t).sum(), wrt=[v, a, z, t])
    values = [
        np.linspace(0.2, 0.8, 5).astype(np.float32),
        *(np.float32(x) for x in (1.2, 0.5, 0.2)),
    ]

    _assert_jax_matches_default([v, a, z, t], grads, values)


@pytest.mark.parametrize("mode", [None, "JAX"])
def test_lan_logp_op_cotangent_flows(fixture_path, mode):
    """Non-unit cotangents are applied by the VJP (chain rule holds).

    Guards against the VJP silently dropping the upstream gradient: the
    gradient of ``(2 * logp).sum()`` must be exactly twice the gradient of
    ``logp.sum()``, on every backend and for both the data-based and the
    scalars-only CPN paths. An equivalence test alone cannot catch this,
    because both backends would drop the cotangent identically.
    """
    angle = onnx.load(fixture_path / "angle.onnx")
    angle_op = make_jax_logp_ops(
        *make_jax_logp_funcs_from_onnx(angle, params_is_reg=[True] + [False] * 4)
    )
    cpn = onnx.load(fixture_path / "ddm_cpn.onnx")
    cpn_op = make_jax_logp_ops(
        *make_jax_logp_funcs_from_onnx(cpn, params_is_reg=[False] * 4, params_only=True)
    )

    data = pt.matrix("data")
    v = pt.vector("v")
    a, z, t, theta = (pt.scalar(name) for name in ["a", "z", "t", "theta"])
    rng = np.random.default_rng(0)
    angle_inputs = [data, v, a, z, t, theta]
    angle_values = [
        _rt_choice_data(0),
        rng.normal(0.5, 0.1, 10).astype(np.float32),
        *(np.float32(x) for x in (1.2, 0.5, 0.2, 0.1)),
    ]

    vs, as_, zs, ts = (pt.scalar(name) for name in ["vs", "as", "zs", "ts"])
    cpn_inputs = [vs, as_, zs, ts]
    cpn_values = [np.float32(x) for x in (0.5, 1.2, 0.5, 0.2)]

    for logp, inputs, values, wrt in [
        (angle_op(data, v, a, z, t, theta), angle_inputs, angle_values, v),
        (cpn_op(None, vs, as_, zs, ts), cpn_inputs, cpn_values, vs),
    ]:
        g1 = pytensor.grad(logp.sum(), wrt=wrt)
        g2 = pytensor.grad((2.0 * logp).sum(), wrt=wrt)
        kwargs = {} if mode is None else {"mode": mode}
        f = pytensor.function(inputs, [g1, g2], **kwargs)
        out1, out2 = f(*values)
        np.testing.assert_array_almost_equal(
            np.asarray(out2), 2.0 * np.asarray(out1), decimal=DECIMAL
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
    data = pt.matrix("data")
    v = pt.vector("v")
    grads = pytensor.grad(logp_op(data, v).sum(), wrt=[v])
    values = [_rt_choice_data(7), rng.normal(0.5, 0.1, 10).astype(np.float32)]

    _assert_jax_matches_default([data, v], grads, values)


def test_lan_logp_vjp_op_jax_linker_n_params():
    """Extra-field inputs are excluded from the VJP outputs (n_params contract).

    Mirrors the RLDM builder: an already-vectorized callable, a VJP built
    with ``n_params``, and an Op whose gradient is undefined for the extra
    trial-wise field.
    """
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
    data = pt.matrix("data")
    p1, p2 = (pt.scalar(name) for name in ["p1", "p2"])
    extra = pt.vector("extra")

    loglik = logp_op(data, p1, p2, extra)
    grads = pytensor.grad(loglik.sum(), wrt=[p1, p2])
    values = [
        _rt_choice_data(11),
        np.float32(0.5),
        np.float32(1.5),
        rng.normal(0.0, 1.0, 10).astype(np.float32),
    ]

    _assert_jax_matches_default([data, p1, p2, extra], grads, values)

    # The extra field sits past n_params, so its gradient is undefined.
    with pytest.raises(NullTypeGradError):
        pytensor.grad(loglik.sum(), wrt=extra)


def test_lan_logp_op_pullback_no_future_warning():
    """Differentiating LANLogpOp emits no PyTensor deprecation FutureWarning.

    Guards the grad -> pullback migration: if the Op reverts to the
    deprecated ``grad``/``L_op`` hooks, PyTensor 3 warns on every symbolic
    differentiation.
    """

    def jax_callable(data, v):
        return -((data[..., 0] - v) ** 2)

    logp_op = make_jax_logp_ops(
        *make_jax_logp_funcs_from_callable(
            jax_callable, vmap=True, params_is_reg=[True]
        )
    )

    data = pt.matrix("data")
    v = pt.vector("v")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        pytensor.grad(logp_op(data, v).sum(), wrt=v)

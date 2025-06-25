from math import dist
from pathlib import Path

import jax
import numpy as np
import onnx
import onnxruntime
import pytensor
import pytensor.tensor as pt
import pytest

import hssm
from hssm.distribution_utils.onnx_utils import *
from hssm.distribution_utils.onnx import (
    make_jax_logp_funcs_from_onnx,
    make_pytensor_logp,
)
from hssm.distribution_utils.jax import make_jax_logp_ops
from hssm.distribution_utils.onnx import make_jax_logp_funcs_from_jax_callable

hssm.set_floatX("float32")
DECIMAL = 4


@pytest.fixture
def fixture_path():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def onnx_session(fixture_path):
    model_path = str(fixture_path / "angle.onnx")

    return onnxruntime.InferenceSession(
        model_path, None, providers=["CPUExecutionProvider"]
    )


def test_interpret_onnx(onnx_session, fixture_path):
    """Tests whether both versions of interpret_onnx return similar values as does the
    ONNX runtime.
    """
    data = np.random.rand(1, 7).astype(np.float32)

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    result_onnx = onnx_session.run([output_name], {input_name: data})[0]

    model = onnx.load(fixture_path / "angle.onnx")
    result_jax = np.asarray(interpret_onnx(model.graph, data)[0])
    result_pytensor = pt_interpret_onnx(model.graph, data)[0].eval()

    np.testing.assert_almost_equal(result_jax, result_onnx, decimal=DECIMAL)
    # For some reason pytensor and onnx (jax) version results are slightly different
    # This could be due to precision.
    np.testing.assert_almost_equal(result_pytensor, result_onnx, decimal=DECIMAL)


def test_make_jax_logp_funcs_from_onnx(fixture_path):
    """Tests whether the jax logp functions returned from jax_logp_funcs from onnx
    returns the same values to interpret_onnx.
    """
    model = onnx.load(fixture_path / "angle.onnx")

    jax_logp, _, jax_logp_nojit = make_jax_logp_funcs_from_onnx(
        model, params_is_reg=[False] * 5
    )

    data = np.random.rand(10, 2)
    params_all_scalars = np.random.rand(5)

    result_boxed_function = jax_logp(data, *params_all_scalars)

    # ensures it returns a vector
    assert len(result_boxed_function) == 10

    input_matrix = np.hstack([np.broadcast_to(params_all_scalars, (10, 5)), data])
    np.testing.assert_array_almost_equal(
        result_boxed_function,
        interpret_onnx(model.graph, input_matrix)[0].squeeze(),
        decimal=DECIMAL,
    )

    np.testing.assert_array_almost_equal(
        result_boxed_function,
        interpret_onnx(model.graph, input_matrix)[0].squeeze(),
        decimal=DECIMAL,
    )

    v = np.random.rand(10)
    input_matrix[:, 0] = v

    jax_logp, _, jax_logp_nojit = make_jax_logp_funcs_from_onnx(
        model, params_is_reg=[True] + [False] * 4
    )

    np.testing.assert_array_almost_equal(
        jax_logp(data, v, *params_all_scalars[1:]),
        interpret_onnx(model.graph, input_matrix)[0].squeeze(),
        decimal=DECIMAL,
    )

    np.testing.assert_array_almost_equal(
        jax_logp_nojit(data, v, *params_all_scalars[1:]),
        interpret_onnx(model.graph, input_matrix)[0].squeeze(),
        decimal=DECIMAL,
    )


def test_make_jax_logp_ops(fixture_path):
    """Tests whether the logp Op returned from make_jax_logp_ops with different backends
    work the same way.
    """
    model = onnx.load(fixture_path / "angle.onnx")

    jax_logp_op = make_jax_logp_ops(
        *make_jax_logp_funcs_from_onnx(model, params_is_reg=[False] * 5)
    )
    pytensor_logp = make_pytensor_logp(model)

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
    pytensor_logp = make_pytensor_logp(model)

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


def test_make_jax_logp_ops_with_extra_fields(fixture_path):
    # Load an arbitrary ONNX model
    def jax_logp_func(data, a, b, e1, e2):
        """A simple JAX logp function for testing."""
        return data + a + b + e1 + e2

    vmap_jax_logp_func = jax.vmap(jax_logp_func, in_axes=(0, None, None, 0, 0))

    def vjp_vmap_func(*inputs, gz):
        """A simple JAX vjp function for testing."""
        _, vjp_fn = jax.vjp(vmap_jax_logp_func, *inputs)
        return vjp_fn(gz)[1:]

    grad_logp_func = jax.jit(vjp_vmap_func)
    jit_logp_func = jax.jit(jax_logp_func)

    jax_logp_op = make_jax_logp_ops(
        jit_logp_func, grad_logp_func, jit_logp_func, n_params=2
    )

    data = np.random.rand(10).astype(np.float32)
    a, b = np.random.rand(2).astype(np.float32)
    var_a = pt.as_tensor_variable(a)
    e1 = np.random.rand(10).astype(np.float32)
    e2 = np.random.rand(10).astype(np.float32)

    params1 = [a, b, e1, e2]
    params2 = [var_a, b, e1, e2]

    jax_loglik = jax_logp_op(data, *params1)
    np.testing.assert_array_almost_equal(
        jax_loglik.eval(),
        jax_logp_func(data, *params1).astype(np.float32),
        decimal=DECIMAL,
    )
    jax_loglik = jax_logp_op(data, *params2)
    assert pytensor.grad(jax_loglik.sum(), wrt=var_a).eval().size == 1

    with pytest.raises(pytensor.gradient.NullTypeGradError):
        e1 = pt.as_tensor_variable(np.random.rand(10).astype(np.float32))
        params3 = [a, b, e1, e2]
        jax_loglik = jax_logp_op(data, *params3)
        assert np.isscalar(pytensor.grad(jax_loglik.sum(), wrt=e1).eval())

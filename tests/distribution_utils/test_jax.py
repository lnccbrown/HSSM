from pathlib import Path

import numpy as np
import onnx
import pytensor
import pytensor.tensor as pt
import pytest

import hssm
from hssm.distribution_utils.jax import make_jax_logp_ops
from hssm.distribution_utils.onnx import (
    make_jax_logp_funcs_from_onnx,
    make_pytensor_logp,
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

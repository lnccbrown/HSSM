from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pytest

import hssm
from hssm.distribution_utils.onnx import (
    make_jax_logp_funcs_from_onnx,
    make_jax_matrix_logp_funcs_from_onnx,
)
from hssm.distribution_utils.onnx_utils import *
from hssm.distribution_utils.onnx_utils.onnx2jax import make_jax_func

hssm.set_floatX("float32")
DECIMAL = 4


@pytest.fixture
def fixture_path():
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def onnx_session(fixture_path):
    model_path = str(fixture_path / "angle.onnx")

    return onnxruntime.InferenceSession(
        model_path, None, providers=["CPUExecutionProvider"]
    )


@pytest.mark.flaky(reruns=2, reruns_delay=1)
@pytest.mark.slow
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

    nojit_funcs = make_jax_logp_funcs_from_onnx(
        model, params_is_reg=[False] * 5, params_only=True, return_jit=False
    )

    assert len(nojit_funcs) == 2

    nojit_funcs = make_jax_logp_funcs_from_onnx(
        model, params_is_reg=[False] * 5, params_only=True, return_jit=False
    )

    assert len(nojit_funcs) == 2

    jit_funcs = make_jax_logp_funcs_from_onnx(
        model, params_is_reg=[False] * 5, params_only=False, return_jit=True
    )

    assert len(jit_funcs) == 3

    jit_funcs = make_jax_logp_funcs_from_onnx(
        model, params_is_reg=[False] * 5, params_only=False, return_jit=True
    )

    assert len(jit_funcs) == 3
    jax_logp, _, jax_logp_nojit = jit_funcs

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


def test_make_simple_jax_logp_funcs_from_onnx(fixture_path):
    """Tests whether the simple jax logp functions returned from onnx
    returns the same values to interpret_onnx.
    """
    model = onnx.load(fixture_path / "angle.onnx")

    jax_logp, _, _ = make_jax_logp_funcs_from_onnx(
        model, params_is_reg=[False] * 5, params_only=False, return_jit=True
    )

    data = np.random.rand(10, 2)
    params_all_scalars = np.random.rand(5)

    result_boxed_function = jax_logp(data, *params_all_scalars)

    jax_logp_simple = make_jax_matrix_logp_funcs_from_onnx(model)

    input_matrix = np.hstack(
        [
            np.broadcast_to(params_all_scalars, (10, 5)),
            data,
        ]
    )

    result_simple = jax_logp_simple(input_matrix)

    np.testing.assert_array_almost_equal(
        result_boxed_function,
        result_simple,
        decimal=DECIMAL,
    )


# ---------------------------------------------------------------------------
# Guards introduced when removing the auto-x64 flip in onnx2jax.py.
# ---------------------------------------------------------------------------


def _make_minimal_onnx(input_shape):
    """Construct a minimal Identity-graph ONNX model with the requested shape.

    ``input_shape`` is an iterable of ``int`` (concrete) or ``str``/``None``
    (symbolic / dynamic).
    """
    from onnx import TensorProto, helper

    dims = list(input_shape)
    in_tensor = helper.make_tensor_value_info("x", TensorProto.FLOAT, dims)
    out_tensor = helper.make_tensor_value_info("y", TensorProto.FLOAT, dims)
    node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    graph = helper.make_graph([node], "minimal", [in_tensor], [out_tensor])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_make_jax_func_rejects_dynamic_input_dim():
    """Dynamic input dims would silently produce wrong outputs -- must raise."""
    model = _make_minimal_onnx(["batch", 5])  # symbolic batch dim
    with pytest.raises(ValueError, match="dynamic"):
        make_jax_func(model)


def test_make_jax_func_accepts_concrete_input_dim():
    """Concrete-shape models pass the guard and produce a callable."""
    model = _make_minimal_onnx([1, 5])
    fn = make_jax_func(model)
    out = np.asarray(fn(np.ones((1, 5), dtype=np.float32)))
    assert out.shape == (5,) or out.shape == (1, 5)  # squeeze may drop the 1


def test_make_jax_func_runs_int64_shape_graph():
    """Loader runs graphs carrying int64 shape ops without the removed recast.

    LAN/sbi exports carry int64 ``Reshape``/``Slice`` metadata. After dropping
    the int64->int32 pre-cast (it corrupted flow exports by wrapping an
    ``INT64_MAX`` sentinel to -1), ``make_jax_func`` must still load and run
    such graphs correctly. This module runs under ``float32`` (x64 off), where
    small int64 index/shape values truncate losslessly — the supported case.
    """
    from onnx import TensorProto, helper, numpy_helper

    # y = Reshape(x, [5]): x is (1, 5) -> y is (5,); the shape arg is int64.
    shape = numpy_helper.from_array(np.array([5], dtype=np.int64), "shape")
    in_tensor = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 5])
    out_tensor = helper.make_tensor_value_info("y", TensorProto.FLOAT, [5])
    node = helper.make_node("Reshape", inputs=["x", "shape"], outputs=["y"])
    graph = helper.make_graph(
        [node], "reshape", [in_tensor], [out_tensor], initializer=[shape]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    fn = make_jax_func(model)
    inp = np.arange(5, dtype=np.float32).reshape(1, 5)
    np.testing.assert_array_almost_equal(
        np.asarray(fn(inp)), np.arange(5), decimal=DECIMAL
    )


def test_make_jax_func_guards_int64_max_under_float32():
    """x64-off must reject int64 constants that would truncate and corrupt.

    Flow exports (sbi nflows, bayesflow ``CouplingFlow``) carry an ``INT64_MAX``
    open-ended-slice sentinel. Under ``float32`` (x64 off) JAX would truncate it
    to -1 and silently corrupt the likelihood, so ``make_jax_func`` must raise.
    Under ``float64`` (x64 on, HSSM's default) the same graph loads fine.
    """
    from onnx import TensorProto, helper, numpy_helper

    sentinel = numpy_helper.from_array(
        np.array([2**63 - 1], dtype=np.int64), "sentinel"
    )
    in_tensor = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 5])
    out_tensor = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 5])
    node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node], "int64_max", [in_tensor], [out_tensor], initializer=[sentinel]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    try:
        hssm.set_floatX("float32")  # x64 off -> the guard must fire
        with pytest.raises(ValueError, match="int64"):
            make_jax_func(model)

        hssm.set_floatX("float64")  # x64 on (HSSM default) -> loads without raising
        make_jax_func(model)
    finally:
        hssm.set_floatX("float32")  # restore the module default for later tests

from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pytest

import hssm
from hssm.distribution_utils.onnx_utils import *
from hssm.distribution_utils.onnx import (
    make_jax_logp_funcs_from_onnx,
    make_jax_matrix_logp_funcs_from_onnx,
)
from hssm.distribution_utils.onnx_utils.onnx2jax import (
    _recast_int64_to_int32,
    make_jax_func,
)

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
    from onnx import helper, TensorProto

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


def test_recast_int64_to_int32_rewrites_initializers():
    """The int64 -> int32 pre-cast must rewrite int64 initializers in place."""
    from onnx import helper, numpy_helper, TensorProto

    init = numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64), "shape")
    in_tensor = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    out_tensor = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Reshape", inputs=["x", "shape"], outputs=["y"])
    graph = helper.make_graph(
        [node], "minimal", [in_tensor], [out_tensor], initializer=[init]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    assert model.graph.initializer[0].data_type == TensorProto.INT64
    n = _recast_int64_to_int32(model)
    assert n == 1
    assert model.graph.initializer[0].data_type == TensorProto.INT32
    # Idempotent: a second call is a no-op.
    assert _recast_int64_to_int32(model) == 0

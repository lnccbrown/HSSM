from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pytensor
import pytest

from src.hssm.wfpt.lan import LAN
from src.hssm.wfpt.onnx2pt import pt_interpret_onnx
from src.hssm.wfpt.onnx2xla import interpret_onnx

pytensor.config.floatX = "float32"


@pytest.fixture(scope="module")
def fixture_path():
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def onnx_session():

    fixture_dir = Path(__file__).parent / "fixtures"
    model_path = str(fixture_dir / "test.onnx")

    return onnxruntime.InferenceSession(model_path, None)


def test_interpret_onnx(onnx_session, fixture_path):

    data = np.random.rand(1, 7).astype(np.float32)

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    result_onnx = onnx_session.run([output_name], {input_name: data})[0]

    model = onnx.load(fixture_path / "test.onnx")
    result_jax = np.asarray(interpret_onnx(model.graph, data)[0])
    result_pytensor = pt_interpret_onnx(model.graph, data)[0].eval()

    np.testing.assert_almost_equal(result_jax, result_onnx, decimal=4)
    # For some reason pytensor and onnx (jax) version results are slightly different
    np.testing.assert_almost_equal(result_pytensor, result_onnx, decimal=4)

    jax_logp, jax_logp_grad, jax_logp_nojit = LAN.make_jax_logp_funcs_from_onnx(
        model, n_params=5
    )

    jax_logp_op = LAN.make_jax_logp_ops(jax_logp, jax_logp_grad, jax_logp_nojit)
    pytensor_logp = LAN.make_pytensor_logp(model)

    random_rt = np.random.choice([1.0, -1.0], size=1, replace=True)
    params = data[:, :5]
    input_onnx = np.ones([1, 7], dtype=np.float32)
    input_onnx[0, :5] = params
    input_onnx[0, 6] = random_rt
    rts_choices = input_onnx[0, 5:]
    list_params = np.squeeze(params).tolist()

    jax_func_result = jax_logp(rts_choices, *list_params)

    result_onnx_rt = np.squeeze(
        np.array(onnx_session.run([output_name], {input_name: input_onnx})[0])
    )

    np.testing.assert_almost_equal(jax_func_result, result_onnx_rt, 4)
    np.testing.assert_almost_equal(
        jax_logp_nojit(rts_choices, *list_params), result_onnx_rt, 4
    )
    np.testing.assert_almost_equal(
        jax_logp_op(rts_choices, *list_params).eval(), result_onnx_rt, 4
    )
    np.testing.assert_almost_equal(
        pytensor_logp(rts_choices, *list_params).eval(), result_onnx_rt, 4
    )

"""Regression tests for arbitrary Python likelihood callbacks."""

from pathlib import Path

import numpy as np
import onnxruntime
import pymc as pm
import pytensor.tensor as pt

from hssm.distribution_utils.blackbox import make_blackbox_op


def test_cvm_executes_blackbox_callback_with_native_session():
    """CVM executes a callback whose ONNX Runtime session is not picklable."""
    model_path = Path(__file__).parents[1] / "fixtures" / "ddm.onnx"
    session = onnxruntime.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )

    def logp(data):
        # Retaining the session in this closure reproduces the native resource
        # that PyMC 6's default Numba linker attempts and fails to cloudpickle.
        session.get_inputs()
        return np.zeros(data.shape[0], dtype=np.float64)

    data = pt.matrix("data")
    blackbox_op = make_blackbox_op(logp)
    compiled_logp = pm.compile([data], blackbox_op(data), mode="cvm")

    result = compiled_logp(np.ones((2, 2), dtype=np.float32))

    np.testing.assert_array_equal(result, np.zeros(2))

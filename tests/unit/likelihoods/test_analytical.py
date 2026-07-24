"""Unit tests for analytical likelihood helpers and DDM-SDV stability."""

import pytest

import numpy as np

import hssm
from hssm.likelihoods.analytical import logp_ddm_sdv, softmax_inv_temperature

hssm.set_floatX("float32")


_N = 10
_rng = np.random.default_rng(42)

_DATA_BINARY = _rng.choice([-1, 1], size=_N).astype(np.float32)
_DATA_TERNARY = _rng.choice([0, 1, 2], size=_N).astype(np.float32)

_SCALAR_BETA = np.float32(1.5)
_VECTOR_BETA = np.full(_N, 1.5, dtype=np.float32)

_SCALAR_LOGIT = np.float32(0.5)
_VECTOR_LOGIT = np.full(_N, 0.5, dtype=np.float32)


class TestSoftmaxInvTemperature:
    @pytest.mark.parametrize(
        "beta", [_SCALAR_BETA, _VECTOR_BETA], ids=["scalar_beta", "vector_beta"]
    )
    @pytest.mark.parametrize(
        "logit", [_SCALAR_LOGIT, _VECTOR_LOGIT], ids=["scalar_logit", "vector_logit"]
    )
    def test_shape_2choice(self, beta, logit):
        result = softmax_inv_temperature(_DATA_BINARY, beta, logit)
        evaluated = result.eval()
        assert evaluated.shape == (_N,)

    @pytest.mark.parametrize(
        "beta", [_SCALAR_BETA, _VECTOR_BETA], ids=["scalar_beta", "vector_beta"]
    )
    @pytest.mark.parametrize(
        "logit1", [_SCALAR_LOGIT, _VECTOR_LOGIT], ids=["scalar_logit1", "vector_logit1"]
    )
    @pytest.mark.parametrize(
        "logit2", [_SCALAR_LOGIT, _VECTOR_LOGIT], ids=["scalar_logit2", "vector_logit2"]
    )
    def test_shape_3choice(self, beta, logit1, logit2):
        result = softmax_inv_temperature(_DATA_TERNARY, beta, logit1, logit2)
        evaluated = result.eval()
        assert evaluated.shape == (_N,)


class TestDdmSdvStability:
    names = ["a", "t", "v"]
    values = [2.5, 3.0, 3.0]
    parameters = [
        (name, np.arange(value, 5.1, 0.5)) for name, value in zip(names, values)
    ]

    @pytest.fixture
    def shared_params(self):
        return {
            "v": 1,
            "sv": 0,
            "a": 0.5,
            "z": 0.5,
            "t": 0.5,
            "err": 1e-7,
            "epsilon": 1e-15,
        }

    @pytest.mark.parametrize("param_name, param_values", parameters)
    def test_no_inf_values(self, data_ddm, shared_params, param_name, param_values):
        for value in param_values:
            params = shared_params | {param_name: value}
            logp = logp_ddm_sdv(data_ddm, **params).eval()
            assert logp.ndim == 1, "logp_ddm_sdv() returned wrong number of dimensions."
            assert np.all(np.isfinite(logp)), (
                f"log_pdf_sv() returned non-finite values for {param_name} = {value}."
            )

import pytest

import numpy as np

from hssm.likelihoods.analytical import softmax_inv_temperature


_N = 10
_rng = np.random.default_rng(42)

_DATA_BINARY = _rng.choice([-1, 1], size=_N).astype(np.float32)
_DATA_TERNARY = _rng.choice([0, 1, 2], size=_N).astype(np.float32)

_SCALAR_BETA = np.float32(1.5)
_VECTOR_BETA = np.full(_N, 1.5, dtype=np.float32)

_SCALAR_LOGIT = np.float32(0.5)
_VECTOR_LOGIT = np.full(_N, 0.5, dtype=np.float32)


@pytest.mark.parametrize(
    "beta", [_SCALAR_BETA, _VECTOR_BETA], ids=["scalar_beta", "vector_beta"]
)
@pytest.mark.parametrize(
    "logit", [_SCALAR_LOGIT, _VECTOR_LOGIT], ids=["scalar_logit", "vector_logit"]
)
def test_softmax_inv_temperature_shape_2choice(beta, logit):
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
def test_softmax_inv_temperature_shape_3choice(beta, logit1, logit2):
    result = softmax_inv_temperature(_DATA_TERNARY, beta, logit1, logit2)
    evaluated = result.eval()
    assert evaluated.shape == (_N,)

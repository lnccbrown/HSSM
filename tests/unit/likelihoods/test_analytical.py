"""Unit tests for analytical likelihood helpers and DDM-SDV stability."""

from itertools import product

import pytest

import jax
import numpy as np
import pymc as pm
import pytensor

import hssm
from hssm.likelihoods.analytical import (
    LOGP_LB,
    logp_ddm_sdv,
    logp_lba2,
    logp_lba3,
    logp_lba4,
    logp_poisson_race,
    logp_rdm3,
    softmax_inv_temperature,
)


@pytest.fixture(scope="module", autouse=True)
def _float32_precision_guard():
    """Pin float32 for this module and restore prior precision settings."""
    prev_floatx = pytensor.config.floatX
    prev_jax_x64 = jax.config.read("jax_enable_x64")

    hssm.set_floatX("float32")
    yield

    pytensor.config.floatX = prev_floatx
    jax.config.update("jax_enable_x64", prev_jax_x64)


_N = 10
_rng = np.random.default_rng(42)

_DATA_BINARY = _rng.choice([-1, 1], size=_N).astype(np.float32)
_DATA_TERNARY = _rng.choice([0, 1, 2], size=_N).astype(np.float32)

_SCALAR_BETA = np.float32(1.5)
_VECTOR_BETA = np.full(_N, 1.5, dtype=np.float32)

_SCALAR_LOGIT = np.float32(0.5)
_VECTOR_LOGIT = np.full(_N, 0.5, dtype=np.float32)

CLOSE_TOLERANCE = 1e-4


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


class TestLbaLikelihood:
    theta_lba2 = dict(A=0.2, b=0.5, v0=1.0, v1=1.0)
    theta_lba3 = theta_lba2 | {"v2": 1.0}
    theta_lba4 = theta_lba3 | {"v3": 1.0}
    lba_parameters = [
        (logp_lba2, theta_lba2, 2),
        (logp_lba3, theta_lba3, 3),
    ]

    @staticmethod
    def _filter_theta(theta, exclude_keys=("A", "b")):
        return {k: v for k, v in theta.items() if k not in exclude_keys}

    @staticmethod
    def _vectorize_param(theta, param, size):
        return {
            k: (np.full(size, v, dtype=np.float32) if k == param else v)
            for k, v in theta.items()
        }

    @staticmethod
    def _make_lba_data(n_choices):
        rows = [[0.35, choice] for choice in range(n_choices)] + [
            [0.5, choice] for choice in range(n_choices)
        ]
        return np.array(rows, dtype=np.float32)

    @pytest.mark.parametrize(
        "logp_func, theta, n_choices",
        lba_parameters,
    )
    def test_lba_synthetic_data(self, logp_func, theta, n_choices):
        lba_data = self._make_lba_data(n_choices)
        size = lba_data.shape[0]

        out_base = logp_func(lba_data, **theta).eval()
        assert out_base.shape == (size,)
        assert np.isfinite(out_base).all()

        for param in theta:
            param_vec = self._vectorize_param(theta, param, size)
            out_vec = logp_func(lba_data, **param_vec).eval()
            assert np.allclose(out_vec, out_base, atol=CLOSE_TOLERANCE)

        for A, b in product([np.full(size, 0.6), 0.6], [np.full(size, 0.5), 0.5]):
            with pytest.raises(pm.logprob.utils.ParameterValueError):
                logp_func(
                    lba_data,
                    A=A,
                    b=b,
                    **self._filter_theta(theta, ("A", "b")),
                ).eval()

    def test_lba4_jax_likelihood_supports_jit_vectors_and_gradients(self):
        """LBA4 should be a native JAX likelihood with differentiable parameters."""
        lba_data = self._make_lba_data(4)
        expected = np.array(
            [
                0.65798534,
                0.65798534,
                0.65798534,
                0.65798534,
                -6.08265287,
                -6.08265287,
                -6.08265287,
                -6.08265287,
            ],
            dtype=np.float32,
        )

        output = np.asarray(jax.jit(logp_lba4)(lba_data, **self.theta_lba4))
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)

        for param in self.theta_lba4:
            vectorized = np.asarray(
                logp_lba4(
                    lba_data,
                    **self._vectorize_param(self.theta_lba4, param, len(lba_data)),
                )
            )
            np.testing.assert_allclose(vectorized, expected, rtol=1e-5, atol=1e-5)

            def summed_logp(value):
                params = self.theta_lba4 | {param: value}
                return logp_lba4(lba_data, **params).sum()

            gradient = np.asarray(jax.grad(summed_logp)(self.theta_lba4[param]))
            assert np.isfinite(gradient).all()

        invalid = np.asarray(
            logp_lba4(
                lba_data,
                A=0.6,
                b=0.5,
                v0=1.0,
                v1=1.0,
                v2=1.0,
                v3=1.0,
            )
        )
        assert np.isneginf(invalid).all()


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


class TestPoissonRaceLikelihood:
    theta_poisson_race = dict(r1=2.5, r2=3.5, k1=1.3, k2=1.6, t=0.05)

    @staticmethod
    def _vectorize_param(theta, param, size):
        return {
            k: (np.full(size, v, dtype=np.float32) if k == param else v)
            for k, v in theta.items()
        }

    @staticmethod
    def _assert_parameter_value_error(data, theta, **bad_values):
        with pytest.raises(pm.logprob.utils.ParameterValueError):
            logp_poisson_race(data, **(theta | bad_values)).eval()

    @pytest.fixture
    def poisson_race_data(self):
        return np.array(
            [
                (0.45, 1.0),
                (0.55, -1.0),
                (0.65, -1.0),
                (0.75, 1.0),
            ],
            dtype=np.float32,
        )

    def test_vectorization(self, poisson_race_data):
        """Per-parameter vectorization should preserve log-likelihood values."""
        size = poisson_race_data.shape[0]
        base = logp_poisson_race(poisson_race_data, **self.theta_poisson_race).eval()

        for param in self.theta_poisson_race:
            param_vec = self._vectorize_param(self.theta_poisson_race, param, size)
            out_vec = logp_poisson_race(poisson_race_data, **param_vec).eval()
            assert np.allclose(out_vec, base, atol=CLOSE_TOLERANCE)

    def test_matches_exponential_case(self):
        """With k1 = k2 = 1, the race reduces to competing exponentials."""
        data = np.array(
            [
                (0.4, 1.0),
                (0.5, -1.0),
                (0.8, -1.0),
                (0.9, 1.0),
            ],
            dtype=np.float32,
        )
        theta = dict(r1=1.5, r2=2.0, k1=1.0, k2=1.0, t=0.0)

        logp = logp_poisson_race(data, **theta).eval()

        def _compute_exponential_logp(rt, response, winner_rate, loser_rate):
            log_pdf = np.log(winner_rate) - winner_rate * rt
            log_survival = -loser_rate * rt
            return log_pdf + log_survival

        expected = [
            _compute_exponential_logp(
                rt,
                response,
                theta["r2"] if response > 0 else theta["r1"],
                theta["r1"] if response > 0 else theta["r2"],
            )
            for rt, response in data
        ]

        np.testing.assert_allclose(np.asarray(logp), expected, atol=1e-6)

    @pytest.mark.parametrize(
        "param,bad_value",
        [
            ("r1", 0.0),
            ("r2", -0.1),
            ("k1", 0.0),
            ("k2", -1.0),
            ("t", -0.2),
        ],
    )
    def test_parameter_validation(self, poisson_race_data, param, bad_value):
        self._assert_parameter_value_error(
            poisson_race_data, self.theta_poisson_race, **{param: bad_value}
        )

    def test_negative_rt_returns_logp_lb(self):
        """Trials with rt <= t should clip to LOGP_LB."""
        data = np.array([(0.02, 1.0)], dtype=np.float32)
        theta = dict(r1=2.0, r2=3.0, k1=1.2, k2=1.4, t=0.05)
        logp = logp_poisson_race(data, **theta).eval()
        assert np.allclose(logp, LOGP_LB)

    def test_tiny_parameters(self):
        """Tiny positive parameters should still produce finite log-likelihoods."""
        data = np.array([(0.5, 1.0), (0.6, -1.0)], dtype=np.float32)
        theta = dict(r1=1e-8, r2=1e-8, k1=1e-8, k2=1e-8, t=0.0)
        logp = logp_poisson_race(data, **theta).eval()
        assert np.all(np.isfinite(logp))

    def test_t_equals_zero(self):
        """t=0 should be valid and return finite values."""
        data = np.array(
            [(0.3, 1.0), (0.5, -1.0), (0.7, 1.0)],
            dtype=np.float32,
        )
        theta = dict(r1=2.0, r2=3.0, k1=1.5, k2=1.5, t=0.0)
        logp = logp_poisson_race(data, **theta).eval()
        assert np.all(np.isfinite(logp))
        assert logp.shape == (3,)


class TestRdmLikelihood:
    def test_negative_rt_returns_logp_lb_and_is_finite(self):
        """Trials with rt <= t should return LOGP_LB and remain finite."""
        data = np.array(
            [
                (0.02, 0.0),
                (0.60, 1.0),
            ],
            dtype=np.float32,
        )
        theta = dict(v0=1.0, v1=1.2, v2=1.4, b=2.0, A=1.0, t=0.05)

        logp = np.asarray(logp_rdm3(data, **theta))

        assert np.isclose(logp[0], LOGP_LB)
        assert np.all(np.isfinite(logp))
        assert not np.any(np.isnan(logp))

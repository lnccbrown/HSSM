"""Unit tests for the Poisson race likelihood."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest
from ssms.config import model_config

import hssm
from hssm.likelihoods.analytical import LOGP_LB, logp_poisson_race
from hssm.simulator import simulate_data

hssm.set_floatX("float32")

CLOSE_TOLERANCE = 1e-4


def vectorize_param(theta, param, size):
    """Return a new parameter dict with one parameter repeated across trials."""
    return {
        k: (np.full(size, v, dtype="float32") if k == param else v)
        for k, v in theta.items()
    }


def assert_parameter_value_error(data, theta, **bad_values):
    """Assert that invalid parameter values raise ParameterValueError."""
    with pytest.raises(pm.logprob.utils.ParameterValueError):
        logp_poisson_race(data, **(theta | bad_values)).eval()


@pytest.fixture
def poisson_race_data():
    """Fixture providing example data."""
    return np.array(
        [
            (0.45, 1.0),
            (0.55, -1.0),
            (0.65, -1.0),
            (0.75, 1.0),
        ],
        dtype="float32",
    )


theta_poisson_race = dict(r1=2.5, r2=3.5, k1=1.3, k2=1.6, t=0.05)


def test_poisson_race_vectorization(poisson_race_data):
    """Check that per-parameter vectorization produces identical log-likelihoods."""
    size = poisson_race_data.shape[0]
    base = logp_poisson_race(poisson_race_data, **theta_poisson_race).eval()

    for param in theta_poisson_race:
        param_vec = vectorize_param(theta_poisson_race, param, size)
        out_vec = logp_poisson_race(poisson_race_data, **param_vec).eval()
        assert np.allclose(out_vec, base, atol=CLOSE_TOLERANCE)


def test_poisson_race_matches_exponential_case():
    """When k1 = k2 = 1, the likelihood reduces to two competing exponentials."""
    data = np.array(
        [
            (0.4, 1.0),
            (0.5, -1.0),
            (0.8, -1.0),
            (0.9, 1.0),
        ],
        dtype="float32",
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
def test_poisson_race_parameter_validation(poisson_race_data, param, bad_value):
    """Invalid parameter values should produce a ParameterValueError."""
    assert_parameter_value_error(
        poisson_race_data, theta_poisson_race, **{param: bad_value}
    )


def test_poisson_race_negative_rt_returns_logp_lb():
    """Trials with rt <= t should clip to LOGP_LB."""
    data = np.array([(0.02, 1.0)], dtype="float32")
    theta = dict(r1=2.0, r2=3.0, k1=1.2, k2=1.4, t=0.05)
    logp = logp_poisson_race(data, **theta).eval()
    assert np.allclose(logp, LOGP_LB)


def test_poisson_race_tiny_parameters():
    """Very small positive parameter values should produce finite log-likelihoods."""
    data = np.array([(0.5, 1.0), (0.6, -1.0)], dtype="float32")
    theta = dict(r1=1e-8, r2=1e-8, k1=1e-8, k2=1e-8, t=0.0)
    logp = logp_poisson_race(data, **theta).eval()
    assert np.all(np.isfinite(logp))


def test_poisson_race_t_equals_zero():
    """t=0 is valid; log-likelihoods should be finite and match the no-shift case."""
    data = np.array(
        [(0.3, 1.0), (0.5, -1.0), (0.7, 1.0)],
        dtype="float32",
    )
    theta = dict(r1=2.0, r2=3.0, k1=1.5, k2=1.5, t=0.0)
    logp = logp_poisson_race(data, **theta).eval()
    assert np.all(np.isfinite(logp))
    assert logp.shape == (3,)


@pytest.mark.skipif(
    "poisson_race" not in model_config,
    reason="poisson_race not available in installed ssms",
)
class TestPoissonRaceSimulator:
    """Tests for the smooth_unif override in simulate_data for poisson_race."""

    _theta = {"r1": 2.5, "r2": 3.0, "k1": 1.3, "k2": 1.5, "t": 0.2}

    def test_smooth_unif_defaults_to_false(self):
        """simulate_data should set smooth_unif=False for poisson_race by default."""
        df = simulate_data("poisson_race", theta=self._theta, size=20, random_state=42)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20
        assert set(df.columns) == {"rt", "response"}

    def test_smooth_unif_explicit_override_respected(self):
        """An explicit smooth_unif=True passed by the caller should be honoured."""
        df = simulate_data(
            "poisson_race",
            theta=self._theta,
            size=20,
            random_state=42,
            smooth_unif=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20

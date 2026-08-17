"""Dependency-free contract tests for the experimental JEAM simulator adapter."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from ssms.hssm_support import validate_simulator_fun

from hssm.integrations import jeam as jeam_integration


class _RecordingCircularModel:
    calls: list[dict] = []

    def __init__(self, threshold_dynamic):
        assert threshold_dynamic == "fixed"

    def simulate(self, **kwargs):
        self.calls.append(kwargs)
        n_sample = kwargs["n_sample"]
        return pd.DataFrame(
            {
                "rt": np.arange(n_sample, dtype=np.float64) + 0.25,
                "response": np.linspace(-1.0, 1.0, n_sample),
                "ignored": np.full(n_sample, 99.0),
            }
        )


@pytest.fixture
def _fake_jeam(monkeypatch):
    _RecordingCircularModel.calls = []
    monkeypatch.setattr(
        jeam_integration,
        "_load_circular_diffusion_model",
        lambda: _RecordingCircularModel,
    )


def test_simulator_advertises_the_hssm_callable_contract():
    """HSSM should recognize two continuous observation columns."""
    model_name, choices, obs_dim = validate_simulator_fun(
        jeam_integration.simulate_circular_diffusion
    )

    assert model_name == "circular_diffusion"
    assert choices == []
    assert obs_dim == 2


def test_simulator_maps_and_replicates_one_parameter_row(_fake_jeam):
    """A one-dimensional theta should produce contiguous JEAM replicas."""
    observed = jeam_integration.simulate_circular_diffusion(
        theta=np.array([0.45, -0.30, 1.20, 0.08]),
        random_state=1947,
        n_replicas=3,
    )

    np.testing.assert_array_equal(
        observed,
        [[0.25, -1.0], [1.25, 0.0], [2.25, 1.0]],
    )
    assert observed.dtype == np.float64
    assert len(_RecordingCircularModel.calls) == 1
    call = _RecordingCircularModel.calls[0]
    np.testing.assert_array_equal(call["drift_vec"], [[0.45, -0.30]] * 3)
    np.testing.assert_array_equal(call["threshold"], [1.20] * 3)
    np.testing.assert_array_equal(call["ndt"], [0.08] * 3)
    assert call["decay"] == 0.0
    assert call["threshold_function"] is None
    assert call["s_v"] == 0.0
    assert call["s_t"] == 0.0
    assert call["sigma"] == 1.0
    assert call["n_sample"] == 3
    assert call["random_state"] == 1947


def test_simulator_preserves_trial_major_replica_order(_fake_jeam):
    """Each trial's replicas should remain adjacent for HSSM's reshape."""
    theta = np.array(
        [
            [0.45, -0.30, 1.20, 0.08],
            [-0.15, 0.55, 0.90, 0.12],
        ]
    )

    observed = jeam_integration.simulate_circular_diffusion(
        theta=theta, random_state=519, n_replicas=2
    )

    assert observed.shape == (4, 2)
    call = _RecordingCircularModel.calls[0]
    np.testing.assert_array_equal(
        call["drift_vec"],
        [[0.45, -0.30], [0.45, -0.30], [-0.15, 0.55], [-0.15, 0.55]],
    )
    np.testing.assert_array_equal(call["threshold"], [1.20, 1.20, 0.90, 0.90])
    np.testing.assert_array_equal(call["ndt"], [0.08, 0.08, 0.12, 0.12])


@pytest.mark.parametrize(
    "theta",
    [
        np.ones(3),
        np.ones(5),
        np.ones((2, 3)),
        np.ones((1, 2, 4)),
        np.empty((0, 4)),
    ],
)
def test_simulator_rejects_invalid_theta_shapes(theta):
    """The adapter accepts exactly four ordered parameters per trial."""
    with pytest.raises(ValueError, match=r"shape \(4,\) or \(n_trials, 4\)"):
        jeam_integration.simulate_circular_diffusion(theta, 11, 1)


@pytest.mark.parametrize(
    "theta",
    [
        np.array([0.1, 0.2, np.nan, 0.05]),
        np.array([[0.1, np.inf, 1.0, 0.05]]),
        ["not", "numeric", "parameters", "here"],
    ],
)
def test_simulator_rejects_invalid_theta_values(theta):
    """Invalid parameter values should fail before loading JEAM."""
    with pytest.raises(ValueError, match="numeric|finite"):
        jeam_integration.simulate_circular_diffusion(theta, 11, 1)


@pytest.mark.parametrize("n_replicas", [0, -1, 1.5, True])
def test_simulator_rejects_invalid_replica_counts(n_replicas):
    """Replica counts follow the positive-integer ssms contract."""
    with pytest.raises(ValueError, match="positive integer"):
        jeam_integration.simulate_circular_diffusion(
            np.array([0.1, 0.2, 1.0, 0.05]), 11, n_replicas
        )


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"rt": [0.2], "choice": [0.1]}), "columns"),
        (pd.DataFrame({"rt": [0.2, 0.3], "response": [0.1, 0.2]}), "shape"),
        (pd.DataFrame({"rt": [np.nan], "response": [0.1]}), "finite"),
        (pd.DataFrame({"rt": [0.0], "response": [0.1]}), "positive"),
        (pd.DataFrame({"rt": [0.2], "response": [np.pi]}), r"\[-pi, pi\)"),
    ],
)
def test_simulator_rejects_producer_contract_drift(monkeypatch, frame, message):
    """Malformed JEAM output should fail at the integration boundary."""
    monkeypatch.setattr(
        jeam_integration,
        "_load_circular_diffusion_model",
        lambda: lambda **kwargs: SimpleNamespace(simulate=lambda **kwargs: frame),
    )

    with pytest.raises(RuntimeError, match=message):
        jeam_integration.simulate_circular_diffusion(
            np.array([0.1, 0.2, 1.0, 0.05]), 11, 1
        )

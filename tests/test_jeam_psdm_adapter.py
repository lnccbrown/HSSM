"""Dependency-free contracts for the projected-spherical JEAM adapters."""

import builtins
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from ssms.hssm_support import validate_simulator_fun

from hssm.integrations import jeam as jeam_integration

pytestmark = pytest.mark.xfail(
    not hasattr(jeam_integration, "logp_projected_spherical_diffusion"),
    reason="projected-spherical JEAM adapters are not implemented yet",
    raises=AttributeError,
    strict=True,
)


class _RecordingProjectedSphericalModel:
    likelihood_calls: list[dict] = []
    simulator_calls: list[dict] = []
    likelihood_error: Exception | None = None

    def __init__(self, threshold_dynamic):
        assert threshold_dynamic == "fixed"

    def joint_lpdf(self, **kwargs):
        self.likelihood_calls.append(kwargs)
        if self.likelihood_error is not None:
            raise self.likelihood_error
        return (
            kwargs["rt"]
            + kwargs["theta"]
            + kwargs["drift_vec"][:, 0]
            + kwargs["drift_vec"][:, 1]
            + kwargs["ndt"]
            + kwargs["threshold"]
        )

    def simulate(self, **kwargs):
        self.simulator_calls.append(kwargs)
        n_sample = kwargs["n_sample"]
        return pd.DataFrame(
            {
                "rt": np.arange(n_sample, dtype=np.float64) + 0.25,
                "response": np.linspace(0.0, np.pi, n_sample),
                "ignored": np.full(n_sample, 99.0),
            }
        )


@pytest.fixture
def _fake_jeam(monkeypatch):
    _RecordingProjectedSphericalModel.likelihood_calls = []
    _RecordingProjectedSphericalModel.simulator_calls = []
    _RecordingProjectedSphericalModel.likelihood_error = None
    monkeypatch.setattr(
        jeam_integration,
        "_load_projected_spherical_diffusion_model",
        lambda: _RecordingProjectedSphericalModel,
    )


def test_likelihood_maps_scalars_and_fixed_settings(_fake_jeam):
    """Scalar parameters should reach PSDM as fixed trial-wise settings."""
    data = np.array([[0.2, 0.0], [0.5, 0.25], [0.9, np.pi]])

    observed = jeam_integration.logp_projected_spherical_diffusion(
        data,
        v_x=0.4,
        v_y=0.7,
        a=1.1,
        t=0.05,
    )

    expected = data.sum(axis=1) + 0.4 + 0.7 + 1.1 + 0.05
    np.testing.assert_allclose(observed, expected)
    assert observed.dtype == np.float64
    call = _RecordingProjectedSphericalModel.likelihood_calls[0]
    np.testing.assert_array_equal(call["drift_vec"], [[0.4, 0.7]] * 3)
    np.testing.assert_array_equal(call["ndt"], [0.05] * 3)
    assert call["threshold"] == 1.1
    assert call["decay"] == 0.0
    assert call["threshold_function"] is None
    assert call["dt_threshold_function"] is None
    assert call["s_v"] == 0.0
    assert call["s_t"] == 0.0
    assert call["sigma"] == 1.0


def test_likelihood_preserves_pointwise_order_with_trialwise_thresholds(_fake_jeam):
    """Threshold grouping should preserve the original observation order."""
    data = np.array([[0.2, 0.1], [0.4, 1.2], [0.6, 2.5]])
    v_x = np.array([0.1, 0.2, 0.3])
    v_y = np.array([0.4, 0.5, 0.6])
    threshold = np.array([1.2, 0.8, 1.2])
    ndt = np.array([0.01, 0.02, 0.03])

    observed = jeam_integration.logp_projected_spherical_diffusion(
        data,
        v_x=v_x,
        v_y=v_y,
        a=threshold,
        t=ndt,
    )

    np.testing.assert_allclose(
        observed,
        data.sum(axis=1) + v_x + v_y + threshold + ndt,
    )
    assert [
        call["threshold"] for call in _RecordingProjectedSphericalModel.likelihood_calls
    ] == [0.8, 1.2]


@pytest.mark.parametrize(
    "data",
    [np.array([0.2, 0.1]), np.ones((2, 1)), np.ones((2, 3))],
)
def test_likelihood_rejects_invalid_data_shapes(data):
    """Only the two-column HSSM observation contract should be accepted."""
    with pytest.raises(ValueError, match=r"shape \(n_observations, 2\)"):
        jeam_integration.logp_projected_spherical_diffusion(
            data,
            0.1,
            0.2,
            1.0,
            0.05,
        )


@pytest.mark.parametrize(
    "angle",
    [np.nextafter(0.0, -np.inf), np.nextafter(np.pi, np.inf)],
)
def test_likelihood_rejects_angles_outside_closed_polar_interval(angle):
    """Polar observations should use the declared closed interval."""
    with pytest.raises(ValueError, match=r"\[0, pi\]"):
        jeam_integration.logp_projected_spherical_diffusion(
            np.array([[0.2, angle]]),
            0.1,
            0.2,
            1.0,
            0.05,
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("v_x", np.ones(3)),
        ("v_y", np.array([np.inf, 0.2])),
        ("a", np.array([1.0, np.nan])),
        ("t", "not-a-number"),
    ],
)
def test_likelihood_rejects_invalid_parameter_arrays(parameter, value):
    """Parameters must provide one finite value for every observation."""
    parameters = {"v_x": 0.1, "v_y": 0.2, "a": 1.0, "t": 0.05}
    parameters[parameter] = value

    with pytest.raises(ValueError, match=parameter):
        jeam_integration.logp_projected_spherical_diffusion(
            np.array([[0.2, 0.1], [0.3, 0.2]]),
            **parameters,
        )


def test_likelihood_preserves_exact_producer_errors(_fake_jeam):
    """Scientific producer failures should cross the adapter unchanged."""
    expected = FloatingPointError("PSDM first-passage calculation failed")
    _RecordingProjectedSphericalModel.likelihood_error = expected

    with pytest.raises(FloatingPointError, match="first-passage") as caught:
        jeam_integration.logp_projected_spherical_diffusion(
            np.array([[0.2, 0.1]]),
            0.1,
            0.2,
            1.0,
            0.05,
        )

    assert caught.value is expected


def test_likelihood_rejects_an_unexpected_producer_shape(monkeypatch):
    """A pointwise-shape regression should fail at the integration boundary."""
    bad_model = SimpleNamespace(joint_lpdf=lambda **kwargs: np.ones((1, 1)))
    monkeypatch.setattr(
        jeam_integration,
        "_load_projected_spherical_diffusion_model",
        lambda: lambda **kwargs: bad_model,
    )

    with pytest.raises(RuntimeError, match="unexpected pointwise log-density shape"):
        jeam_integration.logp_projected_spherical_diffusion(
            np.array([[0.2, 0.1]]),
            0.1,
            0.2,
            1.0,
            0.05,
        )


def test_loader_explains_how_to_install_the_optional_dependency(monkeypatch):
    """A missing JEAM install should report the prototype-group command."""
    real_import = builtins.__import__

    def import_without_jeam(name, *args, **kwargs):
        if name.startswith("jeam"):
            raise ModuleNotFoundError("No module named 'jeam'", name="jeam")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_jeam)

    with pytest.raises(ImportError, match="uv sync --group jeam-prototype"):
        jeam_integration._load_projected_spherical_diffusion_model()


def test_simulator_advertises_the_hssm_callable_contract():
    """HSSM should recognize two continuous observation columns."""
    model_name, choices, obs_dim = validate_simulator_fun(
        jeam_integration.simulate_projected_spherical_diffusion
    )

    assert model_name == "projected_spherical_diffusion"
    assert choices == []
    assert obs_dim == 2


def test_simulator_maps_parameters_replicas_and_fixed_settings(_fake_jeam):
    """One parameter row should produce contiguous fixed-model replicas."""
    observed = jeam_integration.simulate_projected_spherical_diffusion(
        theta=np.array([0.45, 0.70, 1.20, 0.08]),
        random_state=1947,
        n_replicas=3,
    )

    np.testing.assert_array_equal(
        observed,
        [[0.25, 0.0], [1.25, np.pi / 2], [2.25, np.pi]],
    )
    call = _RecordingProjectedSphericalModel.simulator_calls[0]
    np.testing.assert_array_equal(call["drift_vec"], [[0.45, 0.70]] * 3)
    np.testing.assert_array_equal(call["threshold"], [1.20] * 3)
    np.testing.assert_array_equal(call["ndt"], [0.08] * 3)
    assert call["decay"] == 0.0
    assert call["threshold_function"] is None
    assert call["s_v"] == 0.0
    assert call["s_t"] == 0.0
    assert call["sigma"] == 1.0
    assert call["n_sample"] == 3
    assert call["max_time"] == 20.0
    assert call["random_state"] == 1947


def test_simulator_preserves_trial_major_replica_order(_fake_jeam):
    """Each trial's replicas should remain adjacent for HSSM's reshape."""
    theta = np.array([[0.45, 0.70, 1.20, 0.08], [-0.15, 0.55, 0.90, 0.12]])

    observed = jeam_integration.simulate_projected_spherical_diffusion(
        theta=theta,
        random_state=519,
        n_replicas=2,
    )

    assert observed.shape == (4, 2)
    call = _RecordingProjectedSphericalModel.simulator_calls[0]
    np.testing.assert_array_equal(
        call["drift_vec"],
        [[0.45, 0.70], [0.45, 0.70], [-0.15, 0.55], [-0.15, 0.55]],
    )
    np.testing.assert_array_equal(call["threshold"], [1.20, 1.20, 0.90, 0.90])
    np.testing.assert_array_equal(call["ndt"], [0.08, 0.08, 0.12, 0.12])


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"rt": [0.2], "choice": [0.1]}), "columns"),
        (pd.DataFrame({"rt": [0.2, 0.3], "response": [0.1, 0.2]}), "shape"),
        (pd.DataFrame({"rt": [np.nan], "response": [0.1]}), "finite"),
        (pd.DataFrame({"rt": [0.0], "response": [0.1]}), "positive"),
        (pd.DataFrame({"rt": [0.2], "response": [-1e-8]}), r"\[0, pi\]"),
    ],
)
def test_simulator_rejects_producer_contract_drift(monkeypatch, frame, message):
    """Malformed producer output should fail at the integration boundary."""
    monkeypatch.setattr(
        jeam_integration,
        "_load_projected_spherical_diffusion_model",
        lambda: lambda **kwargs: SimpleNamespace(simulate=lambda **kwargs: frame),
    )

    with pytest.raises(RuntimeError, match=message):
        jeam_integration.simulate_projected_spherical_diffusion(
            np.array([0.1, 0.2, 1.0, 0.05]),
            11,
            1,
        )


def test_simulator_reports_producer_omissions_explicitly(monkeypatch):
    """A finite-horizon omission should not silently become predictive data."""
    omitted = pd.DataFrame({"rt": [np.nan], "response": [np.nan]})
    monkeypatch.setattr(
        jeam_integration,
        "_load_projected_spherical_diffusion_model",
        lambda: lambda **kwargs: SimpleNamespace(simulate=lambda **kwargs: omitted),
    )

    with pytest.raises(RuntimeError, match=r"omitted 1 trial.*max_time=20\.0"):
        jeam_integration.simulate_projected_spherical_diffusion(
            np.array([0.1, 0.2, 1.0, 0.05]),
            11,
            1,
        )

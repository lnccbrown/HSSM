"""Dependency-free contract tests for the experimental JEAM adapter."""

import builtins
from types import SimpleNamespace

import numpy as np
import pytest

from hssm.integrations import jeam as jeam_integration


class _RecordingCircularModel:
    calls: list[dict] = []

    def __init__(self, threshold_dynamic):
        assert threshold_dynamic == "fixed"

    def joint_lpdf(self, **kwargs):
        self.calls.append(kwargs)
        return (
            kwargs["rt"]
            + kwargs["theta"]
            + kwargs["drift_vec"][:, 0]
            + kwargs["drift_vec"][:, 1]
            + kwargs["ndt"]
            + kwargs["threshold"]
        )


@pytest.fixture
def _fake_jeam(monkeypatch):
    _RecordingCircularModel.calls = []
    monkeypatch.setattr(
        jeam_integration,
        "_load_circular_diffusion_model",
        lambda: _RecordingCircularModel,
    )


def test_adapter_broadcasts_scalars_and_fixes_jeam_settings(_fake_jeam):
    """Scalar parameters should reach JEAM as fixed trial-wise settings."""
    data = np.array([[0.2, -np.pi], [0.5, 0.25], [0.9, np.pi - 1e-12]])

    observed = jeam_integration.logp_circular_diffusion(
        data, v_x=0.4, v_y=-0.2, a=1.1, t=0.05
    )

    expected = data[:, 0] + data[:, 1] + 0.4 - 0.2 + 1.1 + 0.05
    np.testing.assert_allclose(observed, expected)
    assert observed.dtype == np.float64

    assert len(_RecordingCircularModel.calls) == 1
    call = _RecordingCircularModel.calls[0]
    np.testing.assert_array_equal(call["drift_vec"], [[0.4, -0.2]] * 3)
    np.testing.assert_array_equal(call["ndt"], [0.05] * 3)
    assert call["threshold"] == 1.1
    assert call["decay"] == 0.0
    assert call["threshold_function"] is None
    assert call["dt_threshold_function"] is None
    assert call["s_v"] == 0.0
    assert call["s_t"] == 0.0
    assert call["sigma"] == 1.0


def test_adapter_preserves_order_with_trialwise_thresholds(_fake_jeam):
    """Threshold grouping should preserve the original observation order."""
    data = np.array([[0.2, -0.5], [0.4, 0.0], [0.6, 0.5]])
    v_x = np.array([0.1, 0.2, 0.3])
    v_y = np.array([-0.4, -0.5, -0.6])
    thresholds = np.array([1.2, 0.8, 1.2])
    ndt = np.array([0.01, 0.02, 0.03])

    observed = jeam_integration.logp_circular_diffusion(
        data, v_x=v_x, v_y=v_y, a=thresholds, t=ndt
    )

    expected = data.sum(axis=1) + v_x + v_y + thresholds + ndt
    np.testing.assert_allclose(observed, expected)
    assert [call["threshold"] for call in _RecordingCircularModel.calls] == [0.8, 1.2]


@pytest.mark.parametrize(
    "data",
    [
        np.array([0.2, 0.1]),
        np.ones((2, 1)),
        np.ones((2, 3)),
    ],
)
def test_adapter_rejects_invalid_data_shapes(data):
    """Only the HSSM two-column observation contract should be accepted."""
    with pytest.raises(ValueError, match=r"shape \(n_observations, 2\)"):
        jeam_integration.logp_circular_diffusion(data, 0.1, 0.2, 1.0, 0.05)


@pytest.mark.parametrize(
    "data",
    [
        np.array([[0.2, np.nan]]),
        np.array([[np.inf, 0.1]]),
    ],
)
def test_adapter_rejects_nonfinite_data(data):
    """Nonfinite RTs and angles should fail before JEAM evaluation."""
    with pytest.raises(ValueError, match="only finite values"):
        jeam_integration.logp_circular_diffusion(data, 0.1, 0.2, 1.0, 0.05)


@pytest.mark.parametrize("angle", [np.pi, np.nextafter(-np.pi, -np.inf)])
def test_adapter_rejects_angles_outside_half_open_interval(angle):
    """Circular responses should use the declared half-open radian interval."""
    with pytest.raises(ValueError, match=r"\[-pi, pi\)"):
        jeam_integration.logp_circular_diffusion(
            np.array([[0.2, angle]]), 0.1, 0.2, 1.0, 0.05
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
def test_adapter_rejects_invalid_parameter_arrays(parameter, value):
    """Parameters must provide one finite value for every observation."""
    parameters = {"v_x": 0.1, "v_y": 0.2, "a": 1.0, "t": 0.05}
    parameters[parameter] = value

    with pytest.raises(ValueError, match=parameter):
        jeam_integration.logp_circular_diffusion(
            np.array([[0.2, -0.1], [0.3, 0.1]]), **parameters
        )


def test_adapter_rejects_an_unexpected_jeam_output_shape(monkeypatch):
    """A producer shape-contract regression should fail explicitly."""
    bad_model = SimpleNamespace(joint_lpdf=lambda **kwargs: np.ones((1, 1)))
    monkeypatch.setattr(
        jeam_integration,
        "_load_circular_diffusion_model",
        lambda: lambda **kwargs: bad_model,
    )

    with pytest.raises(RuntimeError, match="unexpected pointwise log-density shape"):
        jeam_integration.logp_circular_diffusion(
            np.array([[0.2, 0.1]]), 0.1, 0.2, 1.0, 0.05
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
        jeam_integration._load_circular_diffusion_model()

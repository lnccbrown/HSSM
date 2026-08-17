"""Numerical handshake tests against the pinned JEAM implementation."""

import numpy as np
import pytest

pytest.importorskip("jeam")

from jeam.Models.Circular import CircularDiffusionModel

from hssm.distribution_utils.dist import (
    LOGP_LB,
    make_distribution,
    make_likelihood_callable,
)
from hssm.integrations.jeam import logp_circular_diffusion

PARAMETERS = ["v_x", "v_y", "a", "t"]
BOUNDS = {
    "v_x": (-3.0, 3.0),
    "v_y": (-3.0, 3.0),
    "a": (0.1, 3.0),
    "t": (0.0, 2.0),
}
# HSSM test modules may select float32 globally during collection. Allow several
# float32 ULPs while keeping the direct adapter comparisons at float64 precision.
COMPILED_RTOL = 5e-7
COMPILED_ATOL = 5e-7


@pytest.fixture(scope="module")
def circular_distribution():
    """Build the real HSSM blackbox distribution around the JEAM adapter."""
    likelihood = make_likelihood_callable(
        loglik=logp_circular_diffusion,
        loglik_kind="blackbox",
        backend="pytensor",
    )
    return make_distribution(
        rv="circular_diffusion",
        loglik=likelihood,
        list_params=PARAMETERS.copy(),
        bounds=BOUNDS,
    )


def _direct_trialwise_jeam(
    data: np.ndarray,
    v_x: np.ndarray,
    v_y: np.ndarray,
    a: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Evaluate JEAM row by row when every parameter may be trial-wise."""
    model = CircularDiffusionModel(threshold_dynamic="fixed")
    return np.array(
        [
            model.joint_lpdf(
                rt=data[index : index + 1, 0],
                theta=data[index : index + 1, 1],
                drift_vec=np.array([[v_x[index], v_y[index]]]),
                ndt=np.array([t[index]]),
                threshold=float(a[index]),
                decay=0.0,
                threshold_function=None,
                dt_threshold_function=None,
                s_v=0.0,
                s_t=0.0,
                sigma=1.0,
            )[0]
            for index in range(data.shape[0])
        ],
        dtype=np.float64,
    )


def test_adapter_matches_direct_jeam_at_asymmetric_scalar_parameters():
    """The adapter should preserve JEAM values across RT and angle locations."""
    data = np.array([[0.18, -2.1], [0.37, 0.35], [0.82, 2.4]], dtype=np.float64)
    v_x = 0.45
    v_y = -0.30
    threshold = 1.20
    ndt = 0.08
    model = CircularDiffusionModel(threshold_dynamic="fixed")

    expected = model.joint_lpdf(
        rt=data[:, 0],
        theta=data[:, 1],
        drift_vec=np.column_stack(
            (np.full(data.shape[0], v_x), np.full(data.shape[0], v_y))
        ),
        ndt=np.full(data.shape[0], ndt),
        threshold=threshold,
        decay=0.0,
        threshold_function=None,
        dt_threshold_function=None,
        s_v=0.0,
        s_t=0.0,
        sigma=1.0,
    )
    observed = logp_circular_diffusion(data, v_x, v_y, threshold, ndt)

    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
    assert observed.shape == (3,)
    assert observed.dtype == np.float64


def test_adapter_matches_direct_jeam_with_trialwise_parameters():
    """All four HSSM parameters should support one value per observation."""
    data = np.array([[0.21, -1.7], [0.46, 0.2], [0.91, 2.2]], dtype=np.float64)
    v_x = np.array([0.55, -0.20, 0.35])
    v_y = np.array([-0.25, 0.45, -0.15])
    threshold = np.array([1.00, 1.25, 1.00])
    ndt = np.array([0.04, 0.11, 0.17])

    expected = _direct_trialwise_jeam(data, v_x=v_x, v_y=v_y, a=threshold, t=ndt)
    observed = logp_circular_diffusion(data, v_x=v_x, v_y=v_y, a=threshold, t=ndt)

    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_adapter_preserves_jeam_impossible_rt_support_value():
    """An RT at or below NDT should retain JEAM's pointwise support result."""
    data = np.array([[0.05, 0.2], [0.10, -0.4]], dtype=np.float64)
    v_x = np.array([0.45, -0.10])
    v_y = np.array([-0.30, 0.25])
    threshold = np.array([1.20, 1.20])
    ndt = np.array([0.10, 0.10])

    expected = _direct_trialwise_jeam(data, v_x=v_x, v_y=v_y, a=threshold, t=ndt)
    observed = logp_circular_diffusion(data, v_x=v_x, v_y=v_y, a=threshold, t=ndt)

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(observed, np.log(1e-14), rtol=0.0, atol=1e-12)


def test_compiled_hssm_distribution_matches_adapter_and_direct_jeam(
    circular_distribution,
):
    """The compiled HSSM blackbox path should preserve pointwise JEAM values."""
    data = np.array([[0.18, -2.1], [0.37, 0.35], [0.82, 2.4]])
    v_x = np.full(3, 0.45)
    v_y = np.full(3, -0.30)
    threshold = np.full(3, 1.20)
    ndt = np.full(3, 0.08)

    direct = _direct_trialwise_jeam(data, v_x=v_x, v_y=v_y, a=threshold, t=ndt)
    adapted = logp_circular_diffusion(data, 0.45, -0.30, 1.20, 0.08)
    compiled = np.asarray(
        circular_distribution.logp(data, 0.45, -0.30, 1.20, 0.08).eval()
    )

    np.testing.assert_allclose(adapted, direct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(compiled, direct, rtol=COMPILED_RTOL, atol=COMPILED_ATOL)


def test_compiled_hssm_distribution_accepts_a_trialwise_drift(
    circular_distribution,
):
    """A regression-shaped drift should remain pointwise through compilation."""
    data = np.array([[0.21, -1.7], [0.46, 0.2], [0.91, 2.2]])
    v_x = np.array([0.55, -0.20, 0.35])
    v_y = np.full(3, -0.25)
    threshold = np.full(3, 1.15)
    ndt = np.full(3, 0.07)

    direct = _direct_trialwise_jeam(data, v_x=v_x, v_y=v_y, a=threshold, t=ndt)
    adapted = logp_circular_diffusion(data, v_x, -0.25, 1.15, 0.07)
    compiled = np.asarray(
        circular_distribution.logp(data, v_x, -0.25, 1.15, 0.07).eval()
    )

    np.testing.assert_allclose(adapted, direct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(compiled, direct, rtol=COMPILED_RTOL, atol=COMPILED_ATOL)


def test_hssm_distribution_applies_standard_ndt_support_mask(circular_distribution):
    """HSSM should replace JEAM's finite floor when nondecision time reaches RT."""
    data = np.array([[0.18, -0.4], [0.37, 0.6], [0.82, 1.2]])
    ndt = np.array([0.08, 0.37, 0.90])

    adapted = logp_circular_diffusion(data, 0.45, -0.30, 1.20, ndt)
    compiled = np.asarray(
        circular_distribution.logp(data, 0.45, -0.30, 1.20, ndt).eval()
    )
    lower_bound = float(np.asarray(LOGP_LB))

    assert compiled[0] == pytest.approx(
        adapted[0], rel=COMPILED_RTOL, abs=COMPILED_ATOL
    )
    np.testing.assert_array_equal(compiled[1:], lower_bound)
    np.testing.assert_allclose(adapted[1:], np.log(1e-14), atol=1e-12)


def test_hssm_distribution_applies_parameter_bounds_pointwise(
    circular_distribution,
):
    """Only trials whose parameter values violate HSSM bounds should be masked."""
    data = np.array([[0.18, -0.4], [0.37, 0.6], [0.82, 1.2]])
    v_x = np.array([0.45, 3.01, -0.25])

    adapted = logp_circular_diffusion(data, v_x, -0.30, 1.20, 0.08)
    compiled = np.asarray(
        circular_distribution.logp(data, v_x, -0.30, 1.20, 0.08).eval()
    )
    lower_bound = float(np.asarray(LOGP_LB))

    np.testing.assert_allclose(
        compiled[[0, 2]],
        adapted[[0, 2]],
        rtol=COMPILED_RTOL,
        atol=COMPILED_ATOL,
    )
    assert compiled[1] == lower_bound

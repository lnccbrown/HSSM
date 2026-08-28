"""Real-producer tests for the JEAM fixed-PSDM adapter handshake."""

import numpy as np
import pytest

pytest.importorskip("jeam")

from jeam.Models.Spherical import ProjectedSphericalDiffusionModel

from hssm.distribution_utils.dist import make_hssm_rv
from hssm.integrations.jeam import (
    logp_projected_spherical_diffusion,
    simulate_projected_spherical_diffusion,
)

PARAMETERS = ["v_x", "v_y", "a", "t"]


def _assert_projected_observations(observations: np.ndarray) -> None:
    """Check the fixed two-column predictive-data contract."""
    assert observations.shape[-1] == 2
    assert observations.dtype == np.float64
    assert np.all(np.isfinite(observations))
    assert np.all(observations[..., 0] > 0.0)
    assert np.all(observations[..., 1] >= 0.0)
    assert np.all(observations[..., 1] <= np.pi)


def test_likelihood_adapter_matches_direct_trialwise_jeam():
    """Every mapped row should equal direct fixed-PSDM evaluation."""
    data = np.array([[0.28, 0.2], [0.53, 1.4], [0.91, 2.7]])
    v_x = np.array([0.55, -0.20, 0.35])
    v_y = np.array([0.25, 0.45, 0.15])
    threshold = np.array([1.00, 1.25, 1.00])
    ndt = np.array([0.04, 0.11, 0.17])
    model = ProjectedSphericalDiffusionModel(threshold_dynamic="fixed")
    expected = np.array(
        [
            model.joint_lpdf(
                rt=data[index : index + 1, 0],
                theta=data[index : index + 1, 1],
                drift_vec=np.array([[v_x[index], v_y[index]]]),
                ndt=np.array([ndt[index]]),
                threshold=float(threshold[index]),
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

    observed = logp_projected_spherical_diffusion(
        data,
        v_x=v_x,
        v_y=v_y,
        a=threshold,
        t=ndt,
    )

    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
    assert observed.shape == (3,)
    assert observed.dtype == np.float64


def test_likelihood_adapter_preserves_projected_drift_error():
    """A negative projected-radial drift should retain JEAM's exact policy."""
    with pytest.raises(
        ValueError,
        match=(
            r"^drift_vec second component must be non-negative for the projected "
            r"spherical model\.$"
        ),
    ):
        logp_projected_spherical_diffusion(
            np.array([[0.3, 0.5]]),
            v_x=0.4,
            v_y=-0.2,
            a=1.0,
            t=0.05,
        )


def test_simulator_adapter_matches_direct_seeded_jeam():
    """The adapter should only map parameters, replicas, and output columns."""
    theta = np.array(
        [[0.70, 0.35, 0.40, 0.08], [-0.20, 0.55, 0.32, 0.12]],
        dtype=np.float64,
    )
    replicated = np.repeat(theta, 3, axis=0)
    model = ProjectedSphericalDiffusionModel(threshold_dynamic="fixed")
    expected = model.simulate(
        drift_vec=replicated[:, :2],
        ndt=replicated[:, 3],
        threshold=replicated[:, 2],
        decay=0.0,
        threshold_function=None,
        s_v=0.0,
        s_t=0.0,
        sigma=1.0,
        n_sample=replicated.shape[0],
        max_time=20.0,
        random_state=1947,
    )[["rt", "response"]].to_numpy(dtype=np.float64)

    observed = simulate_projected_spherical_diffusion(
        theta,
        random_state=1947,
        n_replicas=3,
    )

    np.testing.assert_array_equal(observed, expected)
    _assert_projected_observations(observed)


def test_hssm_random_variable_reproduces_trialwise_replicas():
    """HSSM seed and reshape behavior should preserve trial identity."""
    random_variable = make_hssm_rv(
        simulate_projected_spherical_diffusion,
        PARAMETERS.copy(),
    )
    v_x = np.array([0.70, -0.20, 0.35])
    v_y = np.array([0.35, 0.55, 0.10])
    threshold = np.array([0.40, 0.32, 0.36])
    ndt = np.array([0.08, 0.12, 0.05])

    first = random_variable.rng_fn(
        np.random.default_rng(997),
        v_x,
        v_y,
        threshold,
        ndt,
        size=6,
    )
    matching = random_variable.rng_fn(
        np.random.default_rng(997),
        v_x,
        v_y,
        threshold,
        ndt,
        size=6,
    )
    different = random_variable.rng_fn(
        np.random.default_rng(998),
        v_x,
        v_y,
        threshold,
        ndt,
        size=6,
    )

    np.testing.assert_array_equal(first, matching)
    assert not np.array_equal(first, different)
    assert first.shape == (3, 2, 2)
    _assert_projected_observations(first)

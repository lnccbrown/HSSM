"""Real-producer tests for the JEAM-to-HSSM simulator handshake."""

import numpy as np
import pytest

pytest.importorskip("jeam")

from jeam.Models.Circular import CircularDiffusionModel

from hssm.distribution_utils.dist import make_hssm_rv
from hssm.integrations.jeam import simulate_circular_diffusion

PARAMETERS = ["v_x", "v_y", "a", "t"]


def _assert_circular_observations(observations: np.ndarray) -> None:
    """Check the fixed two-column predictive-data contract."""
    assert observations.shape[-1] == 2
    assert observations.dtype == np.float64
    assert np.all(np.isfinite(observations))
    assert np.all(observations[..., 0] > 0.0)
    assert np.all(observations[..., 1] >= -np.pi)
    assert np.all(observations[..., 1] < np.pi)


def test_simulator_adapter_matches_direct_seeded_jeam():
    """The adapter should only map parameters, replicas, and output columns."""
    theta = np.array(
        [
            [0.70, -0.35, 0.40, 0.08],
            [-0.20, 0.55, 0.32, 0.12],
        ],
        dtype=np.float64,
    )
    replicated = np.repeat(theta, 3, axis=0)
    model = CircularDiffusionModel(threshold_dynamic="fixed")

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
        random_state=1947,
    )[["rt", "response"]].to_numpy(dtype=np.float64)
    observed = simulate_circular_diffusion(theta, random_state=1947, n_replicas=3)

    np.testing.assert_array_equal(observed, expected)
    _assert_circular_observations(observed)


def test_hssm_random_variable_repeats_scalar_draws_exactly_by_seed():
    """An HSSM RNG seed should deterministically control a scalar-parameter draw."""
    random_variable = make_hssm_rv(simulate_circular_diffusion, PARAMETERS.copy())

    first = random_variable.rng_fn(
        np.random.default_rng(519), 0.70, -0.35, 0.40, 0.08, size=6
    )
    second = random_variable.rng_fn(
        np.random.default_rng(519), 0.70, -0.35, 0.40, 0.08, size=6
    )
    different_seed = random_variable.rng_fn(
        np.random.default_rng(520), 0.70, -0.35, 0.40, 0.08, size=6
    )

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different_seed)
    assert first.shape == (6, 2)
    _assert_circular_observations(first)


def test_hssm_random_variable_shapes_trialwise_replicas():
    """Trial-wise parameters should retain a separate replica axis."""
    random_variable = make_hssm_rv(simulate_circular_diffusion, PARAMETERS.copy())
    v_x = np.array([0.70, -0.20, 0.35])
    v_y = np.array([-0.35, 0.55, -0.10])
    threshold = np.array([0.40, 0.32, 0.36])
    ndt = np.array([0.08, 0.12, 0.05])

    first = random_variable.rng_fn(
        np.random.default_rng(997), v_x, v_y, threshold, ndt, size=6
    )
    second = random_variable.rng_fn(
        np.random.default_rng(997), v_x, v_y, threshold, ndt, size=6
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 2, 2)
    _assert_circular_observations(first)

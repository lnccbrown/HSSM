"""Tests for the two-armed-bandit Rescorla-Wagner drift learners."""

import jax.numpy as jnp
import numpy as np
import pytest

from hssm.likelihoods import rldm_optimized
from hssm.rl.likelihoods import two_armed_bandit

RL_ALPHA, SCALER = 0.3, 3.0


def _reference_drifts(arms: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    """Rescorla-Wagner drift before each trial's update, starting from Q = 0.5."""
    q = np.array([0.5, 0.5])
    drifts = []
    for arm, reward in zip(arms, rewards):
        drifts.append((q[1] - q[0]) * SCALER)
        q[arm] += RL_ALPHA * (reward - q[arm])
    return np.array(drifts)


@pytest.mark.parametrize("module", [two_armed_bandit, rldm_optimized])
@pytest.mark.parametrize("lower_code", [0.0, -1.0], ids=["0/1", "-1/1"])
def test_response_coding_selects_the_updated_arm(module, lower_code):
    """Both response codings update arm 0 for the lower and arm 1 for the upper."""
    rng = np.random.default_rng(0)
    arms = rng.integers(0, 2, size=50)
    rewards = rng.integers(0, 2, size=50).astype(float)
    responses = np.where(arms == 1, 1.0, lower_code)
    subj_trials = jnp.column_stack(
        [jnp.full(50, RL_ALPHA), jnp.full(50, SCALER), responses, rewards]
    )

    drifts = module.compute_v_subject_wise(subj_trials)

    # float32-safe tolerance (other RL test modules switch JAX to float32); the
    # mis-indexing this guards against is off by O(1).
    np.testing.assert_allclose(drifts, _reference_drifts(arms, rewards), atol=1e-5)

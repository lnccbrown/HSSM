"""Unit tests for the generic N-action choice-only softmax likelihood.

These exercise the pure JAX functions in ``hssm.rl.likelihoods.softmax`` directly
(no PyMC), covering: equivalence to the legacy 2AFC formula, correctness and
normalisation for N > 2 actions, and the single-pass Q-value scan.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from hssm.rl.likelihoods.softmax import (
    make_compute_q_values_subject_wise,
    softmax_logp_func,
)


def _legacy_binary_logp(params_matrix: np.ndarray) -> np.ndarray:
    """Compute the pre-refactor 2AFC softmax formula (mask-based) for regression."""
    beta = params_matrix[:, 0]
    q0 = params_matrix[:, 1]
    q1 = params_matrix[:, 2]
    response = params_matrix[:, 3]
    logits = np.stack((beta * q0, beta * q1), axis=1)
    chosen = logits[:, 1] * (response == 1) + logits[:, 0] * (response == 0)
    return chosen - logsumexp(logits, axis=1)


def test_softmax_logp_matches_legacy_binary():
    """For 2 actions the generic logp must equal the old masked formula exactly."""
    rng = np.random.default_rng(0)
    n = 64
    beta = rng.uniform(0.1, 5.0, n)
    q0 = rng.normal(size=n)
    q1 = rng.normal(size=n)
    response = rng.integers(0, 2, n).astype(float)
    pm = np.stack([beta, q0, q1, response], axis=1)

    new = np.asarray(softmax_logp_func(jnp.asarray(pm)))
    legacy = _legacy_binary_logp(pm)
    np.testing.assert_allclose(new, legacy, rtol=1e-5, atol=1e-6)


def test_softmax_logp_three_actions_matches_manual():
    """For 3 actions the logp equals beta*q[chosen] - logsumexp(beta*q)."""
    rng = np.random.default_rng(1)
    n = 40
    beta = rng.uniform(0.1, 5.0, n)
    q = rng.normal(size=(n, 3))
    response = rng.integers(0, 3, n).astype(float)
    pm = np.concatenate([beta[:, None], q, response[:, None]], axis=1)

    out = np.asarray(softmax_logp_func(jnp.asarray(pm)))
    logits = beta[:, None] * q
    expected = logits[np.arange(n), response.astype(int)] - logsumexp(logits, axis=1)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
    assert np.all(out <= 1e-6)  # log-probabilities are non-positive


@pytest.mark.parametrize("n_actions", [2, 3, 4])
def test_softmax_logp_normalizes_over_actions(n_actions):
    """For one trial, probabilities over all possible chosen actions sum to 1."""
    beta = np.array([2.0])
    q = (np.arange(n_actions, dtype=float) * 0.3)[None, :]
    probs = []
    for action in range(n_actions):
        pm = np.concatenate([beta[:, None], q, np.array([[float(action)]])], axis=1)
        logp = float(np.asarray(softmax_logp_func(jnp.asarray(pm)))[0])
        probs.append(np.exp(logp))
    np.testing.assert_allclose(sum(probs), 1.0, rtol=1e-6)


def test_compute_q_values_single_scan_n_actions():
    """The N-action Q scan returns the full (n_trials, N) trajectory in one pass."""
    n_actions = 3
    compute_q = make_compute_q_values_subject_wise(n_actions)
    alpha = 0.5
    # columns: [rl_alpha, response (action index), feedback (reward)]
    trials = jnp.asarray(
        [
            [alpha, 0.0, 1.0],
            [alpha, 1.0, 0.0],
            [alpha, 0.0, 0.0],
        ]
    )
    q = np.asarray(compute_q(trials))
    assert q.shape == (3, n_actions)
    # Hand-rolled Rescorla-Wagner (init 0.5, alpha 0.5), reporting pre-update Q:
    #  trial0 Q=[.5,.5,.5];  update action0, reward1 -> q0 = .5 + .5*(1-.5)=.75
    #  trial1 Q=[.75,.5,.5]; update action1, reward0 -> q1 = .5 + .5*(0-.5)=.25
    #  trial2 Q=[.75,.25,.5];update action0, reward0 -> (post-update, unused here)
    expected = np.array(
        [
            [0.5, 0.5, 0.5],
            [0.75, 0.5, 0.5],
            [0.75, 0.25, 0.5],
        ]
    )
    np.testing.assert_allclose(q, expected, rtol=1e-6)

"""Softmax decision process and Q-learning functions for choice-only RLSSM models.

This module provides:
- Q-value learning functions (Rescorla-Wagner update, no scaler)
- A JAX softmax log-likelihood function for 2-alternative forced choice tasks

Unlike the DDM-based learning functions in ``two_armed_bandit.py``, these
functions operate on choice-only data (no RT column) and compute the
log-probability of observed choices under a softmax decision rule.
"""

import jax.numpy as jnp
from jax import nn as jax_nn
from jax.lax import scan


def compute_q_values_trial_wise(
    q_val: jnp.ndarray, inputs: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute trial-wise Q-values and update them.

    Used with ``jax.lax.scan`` to process each trial. Takes current Q-values
    and per-trial inputs, returns the computed Q-values used by the softmax
    decision rule, and updates Q-values via the Rescorla-Wagner learning rule.

    Parameters
    ----------
    q_val
        A length-2 array of current Q-values for the two alternatives.
        Carried forward across trials.
    inputs
        A 1D array of ``[rl_alpha, action, reward]`` for the current trial.

    Returns
    -------
    tuple
        Updated Q-values and the computed action values ``[Q[0], Q[1]]``.
    """
    rl_alpha, action, reward = inputs
    action = jnp.astype(action, jnp.int32)

    computed_q_values = q_val

    # Rescorla-Wagner update
    delta_RL = reward - q_val[action]
    q_val = q_val.at[action].set(q_val[action] + rl_alpha * delta_RL)

    return q_val, computed_q_values


def _compute_q_values_subject_wise(subj_trials: jnp.ndarray) -> jnp.ndarray:
    """Compute trial-wise Q-values for one subject.

    Parameters
    ----------
    subj_trials
        Array of shape ``(n_trials, 3)`` with columns
        ``[rl_alpha, response, feedback]``.

    Returns
    -------
    jnp.ndarray
        Action values for each trial, shape ``(n_trials, 2)``.
    """
    _, q_values = scan(
        compute_q_values_trial_wise,
        jnp.ones(2) * 0.5,  # uniform initial Q-values
        subj_trials,
    )
    return q_values


def compute_q0_subject_wise(subj_trials: jnp.ndarray) -> jnp.ndarray:
    """Compute the first action's trial-wise Q-values for one subject."""
    return _compute_q_values_subject_wise(subj_trials)[:, 0]


def compute_q1_subject_wise(subj_trials: jnp.ndarray) -> jnp.ndarray:
    """Compute the second action's trial-wise Q-values for one subject."""
    return _compute_q_values_subject_wise(subj_trials)[:, 1]


def softmax_logp_func(params_matrix: jnp.ndarray) -> jnp.ndarray:
    """Compute the softmax log-likelihood for 2AFC choice-only data.

    Inputs (columns of ``params_matrix``) are resolved by the builder from
    the ``.inputs`` annotation: ``["beta", "q0", "q1", "response"]``.

    Parameters
    ----------
    params_matrix
        Array of shape ``(n_trials, 4)`` with columns
        ``[beta, q0, q1, response]``.

    Returns
    -------
    jnp.ndarray
        Log-probability of each observed choice, shape ``(n_trials,)``.
    """
    beta = params_matrix[:, 0]
    q0 = params_matrix[:, 1]
    q1 = params_matrix[:, 2]
    response = params_matrix[:, 3]

    logits = jnp.stack((beta * q0, beta * q1), axis=1)
    chosen_logits = logits[:, 1] * (response == 1) + logits[:, 0] * (response == 0)
    logp = chosen_logits - jax_nn.logsumexp(logits, axis=1)
    return logp

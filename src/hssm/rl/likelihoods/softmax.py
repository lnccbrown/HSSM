"""Softmax decision process and Q-learning functions for choice-only RLSSM models.

This module provides:
- Q-value difference learning functions (Rescorla-Wagner update, no scaler)
- A JAX softmax log-likelihood function for 2-alternative forced choice tasks

Unlike the DDM-based learning functions in ``two_armed_bandit.py``, these
functions operate on choice-only data (no RT column) and compute the
log-probability of observed choices under a softmax decision rule.
"""

import jax.numpy as jnp
from jax import nn as jax_nn
from jax.lax import scan


def compute_q_diff_trial_wise(
    q_val: jnp.ndarray, inputs: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute trial-wise Q-value difference and update Q-values.

    Used with ``jax.lax.scan`` to process each trial. Takes current Q-values
    and per-trial inputs, computes the Q-value difference (used as the logit
    scaling input for the softmax decision), and updates Q-values via the
    Rescorla-Wagner learning rule.

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
        Updated Q-values and the Q-value difference ``Q[1] - Q[0]``.
    """
    rl_alpha, action, reward = inputs
    action = jnp.astype(action, jnp.int32)

    # Q-value difference: passed to the softmax (scaled by beta there)
    q_diff = q_val[1] - q_val[0]

    # Rescorla-Wagner update
    delta_RL = reward - q_val[action]
    q_val = q_val.at[action].set(q_val[action] + rl_alpha * delta_RL)

    return q_val, q_diff


def compute_q_diff_subject_wise(subj_trials: jnp.ndarray) -> jnp.ndarray:
    """Compute trial-wise Q-value differences for one subject.

    Parameters
    ----------
    subj_trials
        Array of shape ``(n_trials, 3)`` with columns
        ``[rl_alpha, response, feedback]``.

    Returns
    -------
    jnp.ndarray
        Q-value differences ``Q[1] - Q[0]`` for each trial, shape ``(n_trials,)``.
    """
    _, q_diff = scan(
        compute_q_diff_trial_wise,
        jnp.ones(2) * 0.5,  # uniform initial Q-values
        subj_trials,
    )
    return q_diff


def softmax_logp_func(params_matrix: jnp.ndarray) -> jnp.ndarray:
    """Compute the softmax log-likelihood for 2AFC choice-only data.

    Inputs (columns of ``params_matrix``) are resolved by the builder from
    the ``.inputs`` annotation: ``["beta", "q_diff", "response"]``.

    Parameters
    ----------
    params_matrix
        Array of shape ``(n_trials, 3)`` with columns
        ``[beta, q_diff, response]``.

    Returns
    -------
    jnp.ndarray
        Log-probability of each observed choice, shape ``(n_trials,)``.
    """
    beta = params_matrix[:, 0]
    q_diff = params_matrix[:, 1]
    response = params_matrix[:, 2]

    # p(choice=1) = sigmoid(beta * q_diff)
    logp = jnp.where(
        response == 1,
        jax_nn.log_sigmoid(beta * q_diff),
        jax_nn.log_sigmoid(-beta * q_diff),
    )
    return logp

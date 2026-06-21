"""Softmax decision process and Q-learning functions for choice-only RLSSM models.

This module provides:
- Q-value learning functions (Rescorla-Wagner update, no scaler)
- A JAX softmax log-likelihood function for N-alternative choice tasks

Unlike the DDM-based learning functions in ``two_armed_bandit.py``, these
functions operate on choice-only data (no RT column) and compute the
log-probability of observed choices under a softmax decision rule.

The Q-value scan emits the full per-action Q-value array in a single pass and a
single generic softmax consumes it, so the same code supports any number of
actions (responses are 0-based action indices ``0..n_actions-1``).
"""

from collections.abc import Callable

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
        A length-``n_actions`` array of current Q-values for the alternatives.
        Carried forward across trials.
    inputs
        A 1D array of ``[rl_alpha, action, reward]`` for the current trial.

    Returns
    -------
    tuple
        Updated Q-values and the computed action values ``[Q[0], ..., Q[N-1]]``.
    """
    rl_alpha, action, reward = inputs
    action = jnp.astype(action, jnp.int32)

    computed_q_values = q_val

    # Rescorla-Wagner update
    delta_RL = reward - q_val[action]
    q_val = q_val.at[action].set(q_val[action] + rl_alpha * delta_RL)

    return q_val, computed_q_values


def make_compute_q_values_subject_wise(
    n_actions: int,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build the subject-wise Q-value scan for ``n_actions`` alternatives.

    The returned function maps one subject's ``(n_trials, 3)`` input array with
    columns ``[rl_alpha, response, feedback]`` to its ``(n_trials, n_actions)``
    trajectory of pre-update Q-values. The Rescorla-Wagner scan runs **once** and
    emits all per-action values, so the builder computes every ``q_i`` column in a
    single pass (it is registered for ``q0..q_{n_actions-1}`` under one function).

    Parameters
    ----------
    n_actions
        Number of choice alternatives (length of the Q-value carrier).

    Returns
    -------
    Callable
        ``subj_trials -> q_values`` of shape ``(n_trials, n_actions)``.
    """

    def compute_q_values_subject_wise(subj_trials: jnp.ndarray) -> jnp.ndarray:
        # Build the initial carrier inside the traced function so its dtype
        # matches the active floatX (the scan carry input and output dtypes must
        # agree); building it at factory time would freeze the import-time dtype.
        initial_q_values = jnp.ones(n_actions) * 0.5  # uniform initial Q-values
        _, q_values = scan(
            compute_q_values_trial_wise,
            initial_q_values,
            subj_trials,
        )
        return q_values

    return compute_q_values_subject_wise


def softmax_logp_func(params_matrix: jnp.ndarray) -> jnp.ndarray:
    """Compute the softmax log-likelihood for N-alternative choice-only data.

    Inputs (columns of ``params_matrix``) are resolved by the builder from the
    ``.inputs`` annotation ``["beta", "q0", "q1", ..., "q_{N-1}", "response"]``;
    the number of actions ``N`` is inferred from the width (the Q-value columns
    are everything between ``beta`` and ``response``).

    The decision rule is ``logits = beta * q`` followed by a softmax over actions,
    so the chosen action's log-probability is ``logit[response] - logsumexp(logits)``.

    Parameters
    ----------
    params_matrix
        Array of shape ``(n_trials, N + 2)`` with columns
        ``[beta, q0, ..., q_{N-1}, response]``. ``response`` holds 0-based action
        indices.

    Returns
    -------
    jnp.ndarray
        Log-probability of each observed choice, shape ``(n_trials,)``.
    """
    beta = params_matrix[:, 0]
    q_values = params_matrix[:, 1:-1]  # (n_trials, n_actions)
    response = params_matrix[:, -1]

    logits = beta[:, None] * q_values  # (n_trials, n_actions)
    chosen = jnp.astype(response, jnp.int32)
    chosen_logits = logits[jnp.arange(logits.shape[0]), chosen]
    return chosen_logits - jax_nn.logsumexp(logits, axis=1)

"""The log-likelihood function for the RLDM model."""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import scan

from hssm.distribution_utils.func_utils import make_vjp_func

from ..distribution_utils.jax import make_jax_logp_ops
from ..distribution_utils.onnx import make_simple_jax_logp_funcs_from_onnx

# Obtain the angle log-likelihood function from an ONNX model.
angle_logp_jax_func = make_simple_jax_logp_funcs_from_onnx(
    model="angle.onnx",
)


def compute_v_inner(
    q_val: jnp.ndarray, inputs: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the drift rate and updates the q-values for each trial.

    This function is used with `jax.lax.scan` to process each trial. It takes the
    current q-values and the RL parameters (rl_alpha, scaler), action (response),
    and reward (feedback) for the current trial, computes the drift rate, and
    updates the q-values. The q_values are updated in each iteration and carried
    forward to the next one.

    Parameters
    ----------
    carry
        A tuple containing the current q-values and log likelihood.
    inputs
        A jnp array containing the RL parameters (rl_alpha, scaler), action (response),
        and reward (feedback) for the current trial.

    Returns
    -------
    tuple
        A tuple containing the updated q-values and the computed drift rate (v).
    """
    rl_alpha, scaler, action, reward = inputs
    action = jnp.astype(action, jnp.int32)

    # drift rate on each trial depends on difference in expected rewards for
    # the two alternatives:
    # drift rate = (q_up - q_low) * scaler where
    # the scaler parameter describes the weight to put on the difference in
    # q-values.
    computed_v = (q_val[1] - q_val[0]) * scaler

    # compute the reward prediction error
    delta_RL = reward - q_val[action]

    # update the q-values using the RL learning rule (here, simple TD rule)
    q_val = q_val.at[action].set(q_val[action] + rl_alpha * delta_RL)

    return q_val, computed_v


def compute_v(
    subj_trials: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the log likelihood for a given subject using the RLDM model.

    Parameters
    ----------
    subj_trials:
        A jnp array of dimension (n_trials, 4) containing rl_alpha, scaler,
        action (response), and reward (feedback) for each trial of the subject.

    Returns
    -------
    jnp.ndarray
        The log likelihoods for the RLDM model for the given subject.
    """
    _, v = scan(
        compute_v_inner,
        jnp.ones(2) * 0.5,  # initial q-values for the two alternatives
        subj_trials,
    )

    return v


compute_v_vmapped = jax.vmap(compute_v, in_axes=0)


def make_rldm_logp_func(n_participants: int, n_trials: int) -> Callable:
    """Create a log likelihood function for the RLDM model.

    Parameters
    ----------
    n_participants : int
        The number of participants in the dataset.
    n_trials : int
        The number of trials per participant.

    Returns
    -------
    callable
        A function that computes the log likelihood for the RLDM model.
    """

    def logp(data, *dist_params) -> np.ndarray:
        """Compute the log likelihood for the RLDM model.

        Parameters
        ----------
        data:
            A 2D numpy array of shape (n_trials * n_participants, 2) containing the
            response and reaction time for each trial.
        dist_params:
            A list of parameters for the RLDM model, including:
            - rl_alpha: learning rate for the RL model.
            - scaler: scaling factor for the drift rate.
            - a: boundary separation.
            - z: starting point.
            - t: non-decision time.
            - theta: lapse rate.
            - feedback: feedback for each trial.

        Returns
        -------
        np.ndarray
            The log likelihoods for each subject.
        """
        action = data[:, 1]
        rl_alpha = dist_params[0]
        scaler = dist_params[1]
        feedback = dist_params[-1]

        # Reshape subj_trials into a 3D array of shape (n_participants, n_trials, 4)
        # so we can vmap the compute_v function over its first axis.
        subj_trials = jnp.stack((rl_alpha, scaler, action, feedback), axis=1).reshape(
            n_participants, n_trials, -1
        )

        # Use the compute_v function to get the drift rates (v)
        v = compute_v_vmapped(subj_trials).reshape((-1, 1))

        # create parameter arrays to be passed to the likelihood function
        ddm_params_matrix = jnp.stack(dist_params[2:6], axis=1)
        lan_matrix = jnp.concatenate((v, ddm_params_matrix, data), axis=1)

        return angle_logp_jax_func(lan_matrix)

    return logp


def make_rldm_logp_op(n_participants: int, n_trials: int) -> Callable:
    """Create a pytensor Op for the likelihood function of RLDM model.

    Parameters
    ----------
    n_participants : int
        The number of participants in the dataset.
    n_trials : int
        The number of trials per participant.

    Returns
    -------
    callable
        A function that computes the log likelihood for the RLDM model.
    """
    logp = make_rldm_logp_func(n_participants, n_trials)
    vjp_logp = make_vjp_func(logp, params_only=False)

    return make_jax_logp_ops(
        logp=jax.jit(logp),
        logp_vjp=jax.jit(vjp_logp),
        logp_nojit=logp,
        n_params=6,  # rl_alpha, scaler, a, z, t, theta
    )

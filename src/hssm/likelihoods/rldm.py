"""The log-likelihood function for the RLDM model."""

from typing import Callable

import jax
import jax.numpy as jnp
from jax.lax import dynamic_slice, scan

from hssm.distribution_utils.func_utils import make_vjp_func

from ..distribution_utils.jax import make_jax_logp_ops
from ..distribution_utils.onnx import make_simple_jax_logp_funcs_from_onnx

# Obtain the angle log-likelihood function from an ONNX model.
angle_logp_jax_func = make_simple_jax_logp_funcs_from_onnx(
    model="angle.onnx",
)


def compute_v(q_val: jnp.ndarray, inputs: jnp.ndarray) -> tuple:
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


def rldm_logp_inner_func(
    subj,
    ntrials_subj,
    data,
    rl_alpha,
    scaler,
    a,
    z,
    t,
    theta,
    feedback,
):
    """Compute the log likelihood for a given subject using the RLDM model."""
    rt = data[:, 0]
    response = data[:, 1]

    subj = jnp.astype(subj, jnp.int32)

    compute_v_input = jnp.stack([rl_alpha, scaler, response, feedback], axis=1)
    subj_trials = dynamic_slice(
        compute_v_input, [subj * ntrials_subj, 0], [ntrials_subj, 4]
    )
    _, v = scan(
        compute_v,
        jnp.ones(2) * 0.5,  # initial q-values for the two alternatives
        subj_trials,
    )

    lan_input = jnp.stack(
        [a, z, t, theta, rt, response], axis=1
    )  # Combine all parameters and data into a single input matrix
    lan_matrix = jnp.concatenate(
        [
            v.reshape((-1, 1)),
            dynamic_slice(lan_input, [subj * ntrials_subj, 0], [ntrials_subj, 6]),
        ],
        axis=1,
    )
    return angle_logp_jax_func(lan_matrix)


rldm_logp_inner_func_vmapped = jax.vmap(
    rldm_logp_inner_func,
    # Only the first argument (subj), needs to be vectorized
    in_axes=[0] + [None] * 9,
)


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

    def logp(data, *dist_params) -> jnp.ndarray:
        """Compute the log likelihood for the RLDM model.

        Parameters
        ----------
        args : tuple
            A tuple containing the subject index, number of trials per subject,
            data, and model parameters
            (rl_alpha, scaler, a, z, t, theta, feedback).

        Returns
        -------
        jnp.ndarray
            The log likelihoods for each subject.
        """
        participant_id = dist_params[6]
        feedback = dist_params[7]

        subj = jnp.unique(participant_id, size=n_participants).astype(jnp.int32)

        # create parameter arrays to be passed to the likelihood function
        rl_alpha, scaler, a, z, t, theta = dist_params[:6]

        return rldm_logp_inner_func_vmapped(
            subj,
            n_trials,
            data,
            rl_alpha,
            scaler,
            a,
            z,
            t,
            theta,
            feedback,
        ).ravel()

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

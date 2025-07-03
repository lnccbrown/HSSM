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
    trial,
    feedback,
):
    """Compute the log likelihood for a given subject using the RLDM model."""
    rt = data[:, 0]
    response = data[:, 1]

    subj = jnp.astype(subj, jnp.int32)

    # Extracting the parameters and data for the specific subject
    subj_rl_alpha = dynamic_slice(rl_alpha, [subj * ntrials_subj], [ntrials_subj])
    subj_scaler = dynamic_slice(scaler, [subj * ntrials_subj], [ntrials_subj])
    subj_a = dynamic_slice(a, [subj * ntrials_subj], [ntrials_subj])
    subj_z = dynamic_slice(z, [subj * ntrials_subj], [ntrials_subj])
    subj_t = dynamic_slice(t, [subj * ntrials_subj], [ntrials_subj])
    subj_theta = dynamic_slice(theta, [subj * ntrials_subj], [ntrials_subj])

    subj_trial = dynamic_slice(trial, [subj * ntrials_subj], [ntrials_subj])
    subj_response = dynamic_slice(response, [subj * ntrials_subj], [ntrials_subj])
    subj_rt = dynamic_slice(rt, [subj * ntrials_subj], [ntrials_subj])
    subj_feedback = dynamic_slice(feedback, [subj * ntrials_subj], [ntrials_subj])

    # Initialize the LAN matrix that will hold the trial-by-trial data
    # The matrix will have 7 columns: data (choice, rt) and parameters of
    # the angle model (v, a, z, t, theta)
    # The number of rows is equal to the number of trials for the subject
    q_val = jnp.ones(2) * 0.5
    LAN_matrix_init = jnp.zeros((ntrials_subj, 7))

    # function to process each trial
    def process_trial(carry, inputs):
        q_val, loglik, LAN_matrix, t = carry
        state, action, rt, reward = inputs
        state = jnp.astype(state, jnp.int32)
        action = jnp.astype(action, jnp.int32)

        # drift rate on each trial depends on difference in expected rewards for
        # the two alternatives:
        # drift rate = (q_up - q_low) * scaler where
        # the scaler parameter describes the weight to put on the difference in
        # q-values.
        computed_v = (q_val[1] - q_val[0]) * subj_scaler[t]

        # compute the reward prediction error
        delta_RL = reward - q_val[action]

        # update the q-values using the RL learning rule (here, simple TD rule)
        q_val = q_val.at[action].set(q_val[action] + subj_rl_alpha[t] * delta_RL)

        # update the LAN_matrix with the current trial data
        # The first column is the drift rate, followed by
        # the parameters a, z, t, theta, rt, and action
        segment_result = jnp.array(
            [computed_v, subj_a[t], subj_z[t], subj_t[t], subj_theta[t], rt, action]
        )
        LAN_matrix = LAN_matrix.at[t, :].set(segment_result)

        return (q_val, loglik, LAN_matrix, t + 1), None

    trials = (
        subj_trial,
        subj_response,
        subj_rt,
        subj_feedback,
    )
    (q_val, LL, LAN_matrix, _), _ = scan(
        process_trial, (q_val, 0.0, LAN_matrix_init, 0), trials
    )

    # forward pass through the LAN to compute log likelihoods
    LL = angle_logp_jax_func(LAN_matrix)

    return LL


rldm_logp_inner_func_vmapped = jax.vmap(
    rldm_logp_inner_func,
    # Only the first argument (subj), needs to be vectorized
    in_axes=[0] + [None] * 10,
)


def vec_logp(*args):
    """Parallelize (vectorize) the likelihood computation across subjects.

    'subj_index' arg to the JAX likelihood should be vectorized.
    """
    output = rldm_logp_inner_func_vmapped(*args).ravel()

    return output


def make_logp_func(n_participants: int, n_trials: int) -> Callable:
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
            (rl_alpha, scaler, a, z, t, theta, trial, feedback).

        Returns
        -------
        jnp.ndarray
            The log likelihoods for each subject.
        """
        participant_id = dist_params[6]
        trial = dist_params[7]
        feedback = dist_params[8]

        subj = jnp.unique(participant_id, size=n_participants).astype(jnp.int32)

        # create parameter arrays to be passed to the likelihood function
        rl_alpha, scaler, a, z, t, theta = dist_params[:6]

        return vec_logp(
            subj,
            n_trials,
            data,
            rl_alpha,
            scaler,
            a,
            z,
            t,
            theta,
            trial,
            feedback,
        )

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
    logp = make_logp_func(n_participants, n_trials)
    vjp_logp = make_vjp_func(logp, params_only=False)

    return make_jax_logp_ops(
        logp=jax.jit(logp),
        logp_vjp=jax.jit(vjp_logp),
        logp_nojit=logp,
        n_params=6,  # rl_alpha, scaler, a, z, t, theta
    )

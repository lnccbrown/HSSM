"""The log-likelihood function for the RLDM model."""

import jax
import jax.numpy as jnp
from jax.lax import dynamic_slice, scan

from ..distribution_utils.onnx import make_jax_logp_funcs_from_onnx

# Obtain an angle log-likelihood function from an ONNX model.
angle_logp_func, _ = make_jax_logp_funcs_from_onnx(
    model="angle.onnx",
    params_is_reg=[True] * 5,
    params_only=False,
    return_jit=False,
)


def jax_call_LAN(LAN_matrix):
    """Forward pass through the LAN to compute log likelihoods."""
    net_input = jnp.array(LAN_matrix)
    LL = angle_logp_func(
        net_input[:, 5:7],
        net_input[:, 0],
        net_input[:, 1],
        net_input[:, 2],
        net_input[:, 3],
        net_input[:, 4],
    )

    return LL


def jax_LL_inner(
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
        q_val, LL, LAN_matrix, t = carry
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
        LAN_matrix = jnp.zeros((ntrials_subj, 7))
        q_val = q_val.at[action].set(q_val[action] + subj_rl_alpha[t] * delta_RL)

        # update the LAN_matrix with the current trial data
        # The first column is the drift rate, followed by
        # the parameters a, z, t, theta, rt, and action
        segment_result = jnp.array(
            [computed_v, subj_a[t], subj_z[t], subj_t[t], subj_theta[t], rt, action]
        )

        return (q_val, LL, LAN_matrix, t + 1), segment_result

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
    LL = jax_call_LAN(LAN_matrix)

    return LL


jax_LL_inner_vmapped = jax.vmap(
    jax_LL_inner,
    # Only the first argument (subj), needs to be vectorized
    in_axes=(0, None, None, None, None, None, None, None, None, None, None),
)


def vec_logp(*args):
    """Parallelize (vectorize) the likelihood computation across subjects.

    'subj_index' arg to the JAX likelihood should be vectorized.
    """
    # return jnp.sum(vmap(*args))
    res_LL = jax_LL_inner(*args).ravel()
    res_LL = jnp.reshape(res_LL, (len(res_LL), 1))

    return res_LL


def logp(data, *dist_params) -> jnp.ndarray:
    """Compute the log likelihood for the RLDM model.

    Parameters
    ----------
    args : tuple
        A tuple containing the subject index, number of trials per subject,
        data, and model parameters (rl_alpha, scaler, a, z, t, theta, trial, feedback).

    Returns
    -------
    jnp.ndarray
        The log likelihoods for each subject.
    """
    participant_id = dist_params[5]
    trial = dist_params[6]
    feedback = dist_params[7]

    subj = jnp.unique(participant_id).astype(jnp.int32)
    num_subj = subj.size
    ntrials = (trial.size / num_subj).astype(jnp.int32)

    # create parameter arrays to be passed to the likelihood function
    rl_alpha, scaler, a, z, t, theta = dist_params[:5]

    return vec_logp(
        subj,
        ntrials,
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

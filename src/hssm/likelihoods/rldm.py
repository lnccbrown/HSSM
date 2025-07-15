"""The log-likelihood function for the RLDM model."""

from typing import Callable

import jax
import jax.numpy as jnp
from jax.lax import dynamic_slice, scan

from ..distribution_utils.jax import make_jax_logp_ops
from ..distribution_utils.func_utils import make_vjp_func
from ..distribution_utils.onnx import make_jax_logp_funcs_from_onnx
# from ..onnx_utils.model import download_hf
from ..distribution_utils.onnx import make_jax_matrix_logp_funcs_from_onnx

rlssm_model_config_list = {
    "rlssm1": {
        "name": "rlssm1", 
        "description": "Custom RLSSM with special features", 
        "n_params": 6, 
        "n_extra_fields": 3, 
        "list_params": ["rl.alpha", "scaler", "a", "Z", "t", "theta"], 
        "extra_fields": ["participant_id", "trial_id", "feedback"], 
        "decision_model": "LAN", 
        "LAN": "angle", 
    }
}

MODEL_NAME = "rlssm1"
MODEL_CONFIG = rlssm_model_config_list[MODEL_NAME]
num_params = MODEL_CONFIG["n_params"]
total_params = MODEL_CONFIG["n_params"] + MODEL_CONFIG["n_extra_fields"]

# lan_onnx = download_hf("angle.onnx")

# # Obtain the angle log-likelihood function from an ONNX model.
# lan_logp_jax_func, _ = make_jax_logp_funcs_from_onnx(
#     model=lan_onnx,
#     params_is_reg=[True] * 5,
#     params_only=False,
#     return_jit=False,
# )

lan_logp_jax_func = make_jax_matrix_logp_funcs_from_onnx(
    model="angle.onnx",
)


# def jax_lan_wrapper(input_matrix):
#     """Forward pass through the LAN to compute log likelihoods.

#     This function is just a wrapper that changes the column order of the input matrix
#     to match the expected input for the angle log likelihood function.
#     """
#     net_input = jnp.array(input_matrix)
#     lan_loglik = lan_logp_jax_func(
#         net_input[:, 5:7],
#         net_input[:, 0],
#         net_input[:, 1],
#         net_input[:, 2],
#         net_input[:, 3],
#         net_input[:, 4],
#     )

#     return lan_loglik


def rlssm1_logp_inner_func(
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
    LL = lan_logp_jax_func(LAN_matrix)

    return LL.ravel()


def rlssm2_logp_inner_func(
    subj,
    ntrials_subj,
    data,
    rl_alpha,
    rl_alpha_neg,
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
    subj_rl_alpha_neg = dynamic_slice(rl_alpha_neg, [subj * ntrials_subj], [ntrials_subj])
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

        # if delta_RL < 0, use learning rate subj_rl_alpha_neg[t] else use subj_rl_alpha[t]
        rl_alpha_t = jnp.where(delta_RL < 0, subj_rl_alpha_neg[t], subj_rl_alpha[t])

        # update the q-values using the RL learning rule (here, simple TD rule)
        q_val = q_val.at[action].set(q_val[action] + rl_alpha_t * delta_RL)

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
    LL = lan_logp_jax_func(LAN_matrix)

    return LL.ravel()


rldm_logp_inner_func_vmapped = jax.vmap(
    rlssm1_logp_inner_func,  
    in_axes=[0] + [None] * (total_params + 1),
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
        dist_params
            A tuple containing the subject index, number of trials per subject,
            data, and model parameters. In this case, it is expected to be
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
        rl_alpha, scaler, a, z, t, theta = dist_params[:num_params]

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


def make_rldm_logp_op(n_participants: int, n_trials: int, n_params: int) -> Callable:
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
    #vjp_logp = make_vjp_logp_func(logp, n_params)
    vjp_logp = make_vjp_func(logp, params_only=False, n_params=n_params)

    return make_jax_logp_ops(
        logp=jax.jit(logp),
        logp_vjp=jax.jit(vjp_logp),
        logp_nojit=logp,
        n_params=n_params,
    )

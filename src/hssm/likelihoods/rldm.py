"""The log-likelihood function for the RLDM model."""

from typing import Callable

import jax
import jax.numpy as jnp
from jax.lax import dynamic_slice, scan

from ..distribution_utils.jax import make_jax_logp_ops
from ..distribution_utils.onnx import make_jax_logp_funcs_from_onnx
from ..utils import download_hf

##########################
### CUSTOM RLSSM MODEL ###
##########################
# Add your custom RLSSM model configurations here.

rlssm_model_config_list = {
    "rlssm1": {
        "name": "rlssm1",
        "description": "2-armed bandit Rescorla-Wagner model with Angle decision model.",
        "n_params": 6,
        "n_extra_fields": 2, 
        "list_params": [
            "rl_alpha",
            "scaler",
            "a",
            "z",
            "t",
            "theta",
        ],
        "extra_fields": ["trial", "feedback"],
        "decision_model": "LAN",
        "LAN": "angle",
    },

    "rlssm2": {
        "name": "rlssm2",
        "description": "2-armed bandit Rescorla-Wagner model with Angle decision model. \
            Uses different learning rates for positive and negative prediction errors.",
        "n_params": 7,
        "n_extra_fields": 2, 
        "list_params": [
            "rl_alpha",
            "rl_alpha_neg",
            "scaler",
            "a",
            "z",
            "t",
            "theta",
        ],
        "extra_fields": ["trial", "feedback"],
        "decision_model": "LAN",
        "LAN": "angle",
    }
}

##########################
### CUSTOM RLSSM MODEL ###
##########################
# Specify the model name to use.
# This should match one of the keys in the rlssm_model_config_list dictionary.

MODEL_NAME = "rlssm2"
MODEL_CONFIG = rlssm_model_config_list[MODEL_NAME]

total_params = MODEL_CONFIG["n_params"] + MODEL_CONFIG["n_extra_fields"] + 2 # + 2 for data columns (rt, response)

angle_onnx = download_hf("angle.onnx")

# Obtain the angle log-likelihood function from an ONNX model.
LAN_logp_jax_func, _ = make_jax_logp_funcs_from_onnx(
    model=angle_onnx,
    params_is_reg=[True] * 5,
    params_only=False,
    return_jit=False,
)

def jax_lan_wrapper(input_matrix):
    """Forward pass through the LAN to compute log likelihoods.

    This function is just a wrapper that changes the column order of the input matrix
    to match the expected input for the angle log likelihood function.
    """
    net_input = jnp.array(input_matrix)
    angle_loglik = LAN_logp_jax_func(
        net_input[:, 5:7],
        net_input[:, 0],
        net_input[:, 1],
        net_input[:, 2],
        net_input[:, 3],
        net_input[:, 4],
    )

    return angle_loglik


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
    LL = jax_lan_wrapper(LAN_matrix)

    return LL.ravel()

##########################
### CUSTOM RLSSM MODEL ###
##########################
# Add the likelihood function for your custom RLSSM model here.
# This function should have the following signature:
# def rlssm2_logp_inner_func(
#     subj,
#     ntrials_subj,
#     data,
#     *model_params,
#     *extra_fields,
# ):
# This function output should be a 1D array of log likelihoods for each trial for the given subject.
# The function implementation should be differentiable in order to compute gradients for sampling/optimization.

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

        # Use different learning rates for postive and negative prediction errors
        rl_alpha_t = jnp.where(delta_RL > 0, subj_rl_alpha[t], subj_rl_alpha_neg[t])

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
    LL = jax_lan_wrapper(LAN_matrix)

    return LL.ravel()


##########################
### CUSTOM RLSSM MODEL ###
##########################
# Update the name of the likelihood function to match your custom RLSSM model.
rldm_logp_inner_func_vmapped = jax.vmap(
    rlssm2_logp_inner_func, # this function name should match your custom RLSSM model
    # Only the first argument (subj), needs to be vectorized
    in_axes=[0] + [None] * total_params,
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
        ##########################
        ### CUSTOM RLSSM MODEL ###
        ##########################
        # Unpack the distribution parameters and adjust the parameter array.
        # Update the return statement to match your custom RLSSM model signature.

        participant_id = dist_params[7]
        trial = dist_params[8]
        feedback = dist_params[9]

        subj = jnp.unique(participant_id, size=n_participants).astype(jnp.int32)

        # create parameter arrays to be passed to the likelihood function
        rl_alpha, rl_alpha_neg, scaler, a, z, t, theta = dist_params[:MODEL_CONFIG["n_params"]]

        return vec_logp(
            subj,
            n_trials,
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
        )

    return logp


def make_vjp_logp_func(logp: Callable) -> Callable:
    """Create a vector-Jacobian product (VJP) function for the RLDM log likelihood.

    Parameters
    ----------
    logp : callable
        The log likelihood function.

    Returns
    -------
    callable
        A function that computes the VJP of the log likelihood for the RLDM model.
    """

    def vjp_logp(*inputs, gz):
        """Compute the vector-Jacobian product (VJP) of the log likelihood function.

        Parameters
        ----------
        inputs : list
            A list of data and parameters used in the likelihood computation.
            Also includes the extra_fields data columns.
            Eg. [data, rl_alpha, scaler, a, z, t, theta, trial, feedback].
        gz: jnp.ndarray
            The vector to multiply with the Jacobian of the log likelihood.

        Returns
        -------
        jnp.ndarray
            The VJP of the log likelihoods for each subject.
        """
        _, vjp_logp = jax.vjp(logp, *inputs)
        return vjp_logp(gz)[1:MODEL_CONFIG["n_params"] + 1]  # Exclude the data and the extra fields

    return vjp_logp


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
    vjp_logp = make_vjp_logp_func(logp)

    return make_jax_logp_ops(
        logp=jax.jit(logp),
        logp_vjp=jax.jit(vjp_logp),
        logp_nojit=logp,
        n_params=MODEL_CONFIG["n_params"],  
    )

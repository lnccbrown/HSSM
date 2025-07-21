"""The log-likelihood function for the RLDM model."""

from typing import Callable

import jax
import jax.numpy as jnp
from jax.lax import dynamic_slice, scan
from jax.scipy.special import logsumexp

from ..distribution_utils.func_utils import make_vjp_func
from ..distribution_utils.jax import make_jax_logp_ops

# from ..onnx_utils.model import download_hf
from ..distribution_utils.onnx import (
    make_jax_logp_funcs_from_onnx,
    make_jax_matrix_logp_funcs_from_onnx,
)

rlssm_model_config_list = {
    "rlssm1": {
        "name": "rlssm1",
        "description": "RLSSM model where the learning process is a simple  \
                        Rescorla-Wagner model in a 2-armed bandit task. \
                        The decision process is a collapsing bound DDM (angle model).",
        "n_params": 6,
        "n_extra_fields": 3,
        "list_params": ["rl.alpha", "scaler", "a", "Z", "t", "theta"],
        "extra_fields": ["participant_id", "trial_id", "feedback"],
        "decision_model": "LAN",
        "LAN": "angle",
    },
    "rlssm2": {
        "name": "rlssm2",
        "description": "RLSSM model where the learning process is a simple \
                        Rescorla-Wagner model in a 2-armed bandit task. \
                        The decision process is a collapsing bound DDM (angle model). \
                        Same as rlssm1, but with dual learning rates for positive and \
                         negative prediction errors. \
                        This model is meant to serve as a tutorial for showing how to  \
                         implement a custom RLSSM model in HSSM.",
        "n_params": 7,
        "n_extra_fields": 3,
        "list_params": ["rl.alpha", "rl.alpha_neg", "scaler", "a", "Z", "t", "theta"],
        "extra_fields": ["participant_id", "trial_id", "feedback"],
        "decision_model": "LAN",
        "LAN": "angle",
    },
    "rlwmssm_v2": {
        "name": "rlwmssm_v2",
        "description": "RLSSM model where the learning process is the RLWM model \
                        (see Collins & Frank, 2012 for details).  \
                        The decision process is a collapsing bound LBA \
                            (LBA angle model). ",
        "n_params": 10,
        "n_extra_fields": 6,
        "list_params": [
            "a",
            "z",
            "theta",
            "alpha",
            "phi",
            "rho",
            "gamma",
            "epsilon",
            "C",
            "eta",
        ],
        "extra_fields": [
            "participant_id",
            "set_size",
            "stimulus_id",
            "feedback",
            "new_block_start",
            "unidim_mask",
        ],
        "decision_model": "LAN",
        "LAN": "dev_lba_angle_3_v2",
    },
}

MODEL_NAME = "rlssm1"
MODEL_CONFIG = rlssm_model_config_list[MODEL_NAME]

if not isinstance(MODEL_CONFIG["n_extra_fields"], int) or not isinstance(
    MODEL_CONFIG["n_params"], int
):
    raise ValueError(
        f"Expected 'n_extra_fields' to be an int, \
            got {type(MODEL_CONFIG['n_extra_fields'])}."
    )
num_params = int(MODEL_CONFIG["n_params"])
num_extra_fields = int(MODEL_CONFIG["n_extra_fields"])
total_params = num_params + num_extra_fields


lan_logp_jax_func = make_jax_matrix_logp_funcs_from_onnx(
    model="angle.onnx",
)

jax_LAN_logp = make_jax_logp_funcs_from_onnx(
    "../../tests/fixtures/dev_lba_angle_3_v2.onnx",
    [True] * 6,
)[0]


# RLSSM model likelihood function
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


# RLSSM model liklihood function
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
    subj_rl_alpha_neg = dynamic_slice(
        rl_alpha_neg, [subj * ntrials_subj], [ntrials_subj]
    )
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

        # if delta_RL < 0, use learning rate subj_rl_alpha_neg[t]
        # else use subj_rl_alpha[t]
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


# auxiliary function for the RLWMSSM model
def jax_call_LAN(LAN_matrix, unidim_mask):
    """
    Call the LAN log likelihood function with the LAN matrix and unidim_mask.

    The unidim_mask is used to mask out the log likelihoods for the
    flagged unidimensional trials.
    """
    net_input = jnp.array(LAN_matrix)
    LL = jax_LAN_logp(
        net_input[:, 6:8],
        net_input[:, 0],
        net_input[:, 1],
        net_input[:, 2],
        net_input[:, 3],
        net_input[:, 4],
        net_input[:, 5],
    )

    LL = jnp.multiply(LL, (1 - unidim_mask))

    return LL


# auxiliary function for the RLWMSSM model
def jax_softmax(q_values, beta):
    """Compute the softmax of q_values with temperature beta."""
    return jnp.exp(beta * q_values - logsumexp(beta * q_values))


# RLSSM model likelihood function
def rlwmssm_v2_inner_func(
    subj,
    ntrials_subj,
    data,
    a,
    z,
    theta,
    alpha,
    phi,
    rho,
    gamma,
    epsilon,
    C,
    eta,
    participant_id,
    set_size,
    stimulus_id,
    feedback,
    new_block_start,
    unidim_mask,
):
    """Compute the log likelihood for a given subject using the RLWMSSM model."""
    rt = data[:, 0]
    response = data[:, 1]

    num_actions = 3
    beta = 100
    subj = jnp.astype(subj, jnp.int32)

    def init_block(bl_set_size, subj_rho, subj_C):
        max_set_size = 5
        set_size_mask = jnp.arange(max_set_size) >= bl_set_size
        set_size_mask = set_size_mask[:, None]

        q_RL = jnp.ones((max_set_size, num_actions)) / num_actions
        q_RL = jnp.where(set_size_mask, -1000.0, q_RL)

        q_WM = jnp.ones((max_set_size, num_actions)) / num_actions
        q_WM = jnp.where(set_size_mask, -1000.0, q_WM)

        weight = subj_rho * jnp.minimum(1, subj_C / bl_set_size)

        return q_RL, q_WM, weight

    def update_q_values(carry, inputs):
        q_RL, q_WM, alpha, gamma, phi, num_actions = carry
        state, action, reward = inputs

        delta_RL = reward - q_RL[state, action]
        delta_WM = reward - q_WM[state, action]

        RL_alpha_factor = jnp.where(reward == 1, alpha, gamma * alpha)
        WM_alpha_factor = jnp.where(reward == 1, 1.0, gamma)

        q_RL = q_RL.at[state, action].set(
            q_RL[state, action] + RL_alpha_factor * delta_RL
        )
        q_WM = q_WM.at[state, action].set(
            q_WM[state, action] + WM_alpha_factor * delta_WM
        )

        q_WM = q_WM + phi * ((1 / num_actions) - q_WM)

        return q_RL, q_WM

    # Extracting the parameters for the specific subject
    subj_a = dynamic_slice(a, [subj * ntrials_subj], [ntrials_subj])
    subj_z = dynamic_slice(z, [subj * ntrials_subj], [ntrials_subj])
    subj_theta = dynamic_slice(theta, [subj * ntrials_subj], [ntrials_subj])
    subj_alpha = dynamic_slice(alpha, [subj * ntrials_subj], [ntrials_subj])
    subj_alpha = jnp.exp(subj_alpha)
    subj_phi = dynamic_slice(phi, [subj * ntrials_subj], [ntrials_subj])
    subj_rho = dynamic_slice(rho, [subj * ntrials_subj], [ntrials_subj])
    subj_gamma = dynamic_slice(gamma, [subj * ntrials_subj], [ntrials_subj])
    subj_epsilon = dynamic_slice(epsilon, [subj * ntrials_subj], [ntrials_subj])
    subj_C = dynamic_slice(C, [subj * ntrials_subj], [ntrials_subj])
    subj_eta = dynamic_slice(eta, [subj * ntrials_subj], [ntrials_subj])

    # Extracting the data for the specific subject
    subj_set_size = dynamic_slice(set_size, [subj * ntrials_subj], [ntrials_subj])
    subj_stimulus_id = dynamic_slice(stimulus_id, [subj * ntrials_subj], [ntrials_subj])
    subj_response = dynamic_slice(response, [subj * ntrials_subj], [ntrials_subj])
    subj_feedback = dynamic_slice(feedback, [subj * ntrials_subj], [ntrials_subj])
    subj_new_block_start = dynamic_slice(
        new_block_start, [subj * ntrials_subj], [ntrials_subj]
    )
    subj_unidim_mask = dynamic_slice(unidim_mask, [subj * ntrials_subj], [ntrials_subj])
    subj_rt = dynamic_slice(rt, [subj * ntrials_subj], [ntrials_subj])

    LAN_matrix_init = jnp.zeros((ntrials_subj, 8))

    def process_trial(carry, inputs):
        q_RL, q_WM, weight, LL, LAN_matrix, t = carry
        bl_set_size, state, action, rt, reward, new_block = inputs
        state = jnp.astype(state, jnp.int32)
        action = jnp.astype(action, jnp.int32)

        q_RL, q_WM, weight = jax.lax.cond(
            new_block == 1,
            lambda _: init_block(bl_set_size, subj_rho[t], subj_C[t]),
            lambda _: (q_RL, q_WM, weight),
            None,
        )

        pol_RL = jax_softmax(q_RL[state, :], beta)
        pol_WM = jax_softmax(q_WM[state, :], beta)

        pol = weight * pol_WM + (1 - weight) * pol_RL
        pol = (
            subj_epsilon[t] * (jnp.ones_like(pol) * 1 / num_actions)
            + (1 - subj_epsilon[t]) * pol
        )
        pol_final = pol * subj_eta[t]

        q_RL, q_WM = update_q_values(
            (q_RL, q_WM, subj_alpha[t], subj_gamma[t], subj_phi[t], num_actions),
            (state, action, reward),
        )

        LAN_matrix = LAN_matrix.at[t, :].set(
            jnp.array(
                [
                    pol_final[0],
                    pol_final[1],
                    pol_final[2],
                    subj_a[t],
                    subj_z[t],
                    subj_theta[t],
                    rt,
                    action,
                ]
            )
        )

        return (q_RL, q_WM, weight, LL, LAN_matrix, t + 1), None

    q_RL, q_WM, weight = init_block(5, 0, 5)
    trials = (
        subj_set_size,
        subj_stimulus_id,
        subj_response,
        subj_rt,
        subj_feedback,
        subj_new_block_start,
    )
    (q_RL, q_WM, weight, LL, LAN_matrix, _), _ = jax.lax.scan(
        process_trial, (q_RL, q_WM, weight, 0.0, LAN_matrix_init, 0), trials
    )

    LL = jax_call_LAN(LAN_matrix, subj_unidim_mask)

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

    # Ensure parameters are correctly extracted and passed to your custom function.
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
        # Extract extra fields (adjust indices based on your model)
        participant_id = dist_params[num_params]
        trial = dist_params[num_params + 1]
        feedback = dist_params[num_params + 2]

        subj = jnp.unique(participant_id, size=n_participants).astype(jnp.int32)

        # create parameter arrays to be passed to the likelihood function
        rl_alpha, scaler, a, z, t, theta = dist_params[:num_params]

        # pass the parameters and data to the likelihood function
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
    vjp_logp = make_vjp_func(logp, params_only=False, n_params=n_params)

    return make_jax_logp_ops(
        logp=jax.jit(logp),
        logp_vjp=jax.jit(vjp_logp),
        logp_nojit=logp,
        n_params=n_params,
    )

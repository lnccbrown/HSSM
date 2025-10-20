"""The log-likelihood function for the RLDM model."""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import scan
from pytensor.graph import Op

from hssm.distribution_utils.func_utils import make_vjp_func

from ..distribution_utils.jax import make_jax_logp_ops
from ..distribution_utils.onnx import make_jax_matrix_logp_funcs_from_onnx

# Obtain the angle log-likelihood function from an ONNX model.
angle_logp_jax_func = make_jax_matrix_logp_funcs_from_onnx(
    model="angle.onnx",
)


# Inner function to compute the drift rate and update q-values for each trial.
# This function is used with `jax.lax.scan` to process each trial in the RLDM model.
def compute_v_trial_wise(
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
    q_val
        A length-2 jnp array containing the current q-values for the two alternatives.
        These values are updated in each iteration and carried forward to the next
        trial.
    inputs
        A 2D jnp array containing the RL parameters (rl_alpha, scaler),
        action (response), and reward (feedback) for the current trial.

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


# This function computes the drift rates (v) for each subject by processing
# their trials one by one. It uses `jax.lax.scan` to efficiently iterate over
# the trials and compute the drift rates based on the RL parameters, actions,
# and rewards for each trial.
def compute_v_subject_wise(
    subj_trials: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the drift rates (v) for a given subject.

    Parameters
    ----------
    subj_trials:
        A jnp array of dimension (n_trials, 4) containing rl_alpha, scaler,
        action (response), and reward (feedback) for each trial of the subject.

    Returns
    -------
    jnp.ndarray
        The computed drift rates (v) for the RLDM model for the given subject.
    """
    _, v = scan(
        compute_v_trial_wise,
        jnp.ones(2) * 0.5,  # initial q-values for the two alternatives
        subj_trials,
    )

    return v


def _validate_columns(
    data_cols: list[str],
    dist_params: list[str] | None = None,
    extra_fields: list[str] | None = None,
) -> list[str]:
    dist_params = dist_params or []
    extra_fields = extra_fields or []
    all_cols = [*dist_params, *extra_fields]
    missing_cols = set(all_cols) - set(data_cols)
    if missing_cols:
        raise ValueError(
            f"The following columns are missing from data_cols: {missing_cols}"
        )


def _get_column_indices(
    data_cols: list[str],
    dist_params: list[str] | None = None,
    extra_fields: list[str] | None = None,
) -> list[int]:
    col2idx = {col: idx for idx, col in enumerate(data_cols)} if data_cols else {}
    dist_params_idxs = [col2idx[c] for c in (dist_params or [])]
    extra_fields_idxs = [col2idx[c] for c in (extra_fields or [])]
    return dist_params_idxs + extra_fields_idxs


def make_rl_logp_func(
    subject_wise_func: Callable,
    n_participants: int,
    n_trials: int,
    data_cols: list[str] | None = None,
    dist_params: list[str] | None = None,
    extra_fields: list[str] | None = None,
) -> Callable:
    """Create a function to compute the drift rates (v) for the RLDM model.

    Parameters
    ----------
    subject_wise_func : Callable
        Function that computes drift rates for a subject's trials.
    n_participants : int
        Number of participants in the dataset.
    n_trials : int
        Number of trials per participant.
    data_cols : list[str] | None
        List of column names in the data array.
    dist_params : list[str] | None
        List of distribution parameter names required by the RL model.
    extra_fields : list[str] | None
        List of extra field names required by the RL model.

    Returns
    -------
    Callable
        A function that computes drift rates (v) for all subjects given their trial data
        and RLDM parameters.
    """
    # Vectorized version of  subject_wise_func to handle multiple subjects.
    subject_wise_vmapped = jax.vmap(subject_wise_func, in_axes=0)

    def logp(*args) -> np.ndarray:
        """Compute the log likelihood for the specified RL model.

        Parameters
        ----------
        *args:
            Variable number of arguments containing trial data and model parameters.
            Arguments should be provided in the order they will be stacked, and can
            include any combination of:
            - rl_alpha: learning rate for the RL model.
            - scaler: scaling factor for the drift rate.
            - a: boundary separation.
            - z: starting point.
            - t: non-decision time.
            - theta: lapse rate.
            - feedback: feedback for each trial.
            - any other relevant parameter as needed.
            Each argument should be a 1D array of length (n_trials * n_participants).

        Returns
        -------
        np.ndarray
            The computed drift rates for each trial, reshaped as a 2D array.
        """
        # Reshape subj_trials into a 3D array of shape
        # (n_participants, n_trials, len(args))
        # so we can act on this object with the vmapped version of the mapping function
        subj_trials = jnp.stack((*args,), axis=1).reshape(n_participants, n_trials, -1)

        # Use the compute_v function to get the drift rates (v)
        drift_rates = subject_wise_vmapped(subj_trials).reshape((-1, 1))

        return drift_rates

    return logp


# TODO[CP]: Adapt this function given the changes to make_rl_logp_func
# pragma: no cover
def make_rldm_logp_op(
    subject_wise_func: Callable, n_participants: int, n_trials: int, n_params: int
) -> Op:
    """Create a pytensor Op for the likelihood function of RLDM model.

    Parameters
    ----------
    n_participants : int
        The number of participants in the dataset.
    n_trials : int
        The number of trials per participant.

    Returns
    -------
    Op
        A pytensor Op that computes the log likelihood for the RLDM model.
    """
    logp = make_rl_logp_func(subject_wise_func, n_participants, n_trials)
    vjp_logp = make_vjp_func(logp, params_only=False, n_params=n_params)

    return make_jax_logp_ops(
        logp=jax.jit(logp),
        logp_vjp=jax.jit(vjp_logp),
        logp_nojit=logp,
        n_params=n_params,  # rl_alpha, scaler, a, z, t, theta
    )

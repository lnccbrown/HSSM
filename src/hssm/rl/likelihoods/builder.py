"""The log-likelihood function for the RLDM model."""

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import scan
from pytensor.graph import Op

from hssm.distribution_utils.func_utils import make_vjp_func
from hssm.distribution_utils.jax import make_jax_logp_ops


class AnnotatedFunction(Protocol):
    """A protocol for functions annotated with metadata about their parameters.

    This protocol defines the interface for functions that have been annotated with
    the `hssm.rl.likelihoods.utils.annotate_function` decorator. It is a key abstraction
    in the RLDM (Reinforcement Learning Drift-Diffusion Model) likelihood system,
    enabling automatic parameter resolution and composition of complex likelihood
    functions.

    The protocol ensures that functions carry metadata about:
    - Which parameters they require as inputs
    - Which values they produce as outputs
    - Which of their inputs need to be computed by other annotated functions

    This metadata-driven approach allows `make_rl_logp_func` to automatically:
    1. Identify which parameters come from data columns vs. model parameters
    2. Determine which parameters need to be computed by other functions
    3. Build a complete computational graph for the likelihood function

    Attributes
    ----------
    inputs : list[str]
        Names of all input parameters required by the function. These can include:
        - Data columns (e.g., "rt", "response", "feedback")
        - Model parameters (e.g., "rl.alpha", "scaler", "a", "z", "t")
        - Computed parameters that will be provided by other functions (e.g., "v")
    outputs : list[str]
        Names of values produced by the function. For parameter computation
        functions (e.g., `compute_v_subject_wise`), this typically contains
        the name of the computed parameter (e.g., ["v"]). For likelihood
        functions, this is often empty or contains ["logp"].
    computed : dict[str, AnnotatedFunction]
        Mapping from parameter names to the AnnotatedFunction instances that
        compute them. For example, if drift rate "v" is computed by a
        reinforcement learning model, this would be:
        `{"v": compute_v_subject_wise_annotated}`.
        This creates a dependency graph that the system uses to determine
        the order of computation.

    Examples
    --------
    Create an annotated computation function for drift rates:

    >>> @annotate_function(
    ...     inputs=["rl.alpha", "scaler", "response", "feedback"], outputs=["v"]
    ... )
    ... def compute_v_subject_wise(subj_trials):
    ...     # Computation logic here
    ...     return v_values

    Create an annotated likelihood function that depends on computed parameters:

    >>> @annotate_function(
    ...     inputs=["v", "a", "z", "t", "theta", "rt", "response"],
    ...     computed={"v": compute_v_subject_wise_annotated},
    ... )
    ... def angle_logp_func(params_matrix):
    ...     # Likelihood computation logic here
    ...     return logp

    Notes
    -----
    The `@annotate_function` decorator dynamically attaches the `inputs`, `outputs`,
    and `computed` attributes to functions at runtime via `setattr()`. This Protocol
    provides a type hint interface for static type checkers.

    See Also
    --------
    hssm.rl.likelihoods.utils.annotate_function : Decorator that creates
        AnnotatedFunction instances
    make_rl_logp_func : Factory that uses AnnotatedFunction metadata to build
        likelihoods
    _get_column_indices_with_computed : Helper that resolves parameter sources
    """

    inputs: list[str]
    outputs: list[str]
    computed: dict[str, "AnnotatedFunction"]

    # Added to satisfy static type checkers
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: D102


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


def _get_column_indices(
    cols_to_look_up: list[str],
    data_cols: list[str],
    list_params: list[str] | None,
    extra_fields: list[str] | None,
) -> dict[str, tuple[str, int]]:
    """Return indices for required columns.

    Parameters
    ----------
    cols_to_look_up : list[str]
        Columns to find indices for
    data_cols : list[str]
        Available data columns
    list_params : list[str] | None
        Available list parameters
    extra_fields : list[str] | None
        Available extra fields

    Returns
    -------
    dict[str, tuple[str, int]]
        Mapping of column names to (source, index) tuples where source is
        "data" for data columns or "args" for list_params/extra_fields.

    See Also
    --------
    _get_column_indices_with_computed : Extended version that handles computed
        parameters
    _collect_cols_arrays : Uses these indices to extract arrays
    """
    list_params = list_params or []
    extra_fields = extra_fields or []
    list_params_extra_fields = list_params + extra_fields
    colidxs = {}
    for col in cols_to_look_up:
        if col in data_cols:
            colidxs[col] = ("data", data_cols.index(col))
        elif col in list_params_extra_fields:
            colidxs[col] = ("args", list_params_extra_fields.index(col))
        else:
            raise ValueError(
                f"Column '{col}' not found in any of `data`, `list_params`, "
                f"or `extra_fields`."
            )
    return colidxs


@dataclass
class ColumnLookupResult:
    """Result of column lookup containing indices and computed parameter info."""

    colidxs: dict[str, tuple[str, int]]
    computed: list[str]


def _get_column_indices_with_computed(
    func_with_inputs: AnnotatedFunction,
    data_cols: list[str],
    list_params: list[str] | None,
    extra_fields: list[str] | None,
) -> ColumnLookupResult:
    """Get column indices and identify computed parameters from function inputs.

    Parameters
    ----------
    func_with_inputs : Callable
        Function object with .inputs attribute containing required columns
    data_cols : list[str]
        Available data columns
    list_params : list[str] | None
        Available list parameters
    extra_fields : list[str] | None
        Available extra fields

    Returns
    -------
    ColumnLookupResult
        Object with colidxs dict and computed list
    """
    if not hasattr(func_with_inputs, "inputs"):
        raise TypeError("func_with_inputs must have an 'inputs' attribute")
    inputs = func_with_inputs.inputs

    # Create sets for efficient lookup
    inputs_set = set(inputs)
    list_params = list_params or []
    extra_fields = extra_fields or []
    all_available_set = set(data_cols + list_params + extra_fields)

    # Find which inputs are available and which are computed
    found_set = inputs_set & all_available_set
    computed_set = inputs_set - all_available_set

    # Convert back to lists maintaining original order
    found_inputs = [inp for inp in inputs if inp in found_set]
    computed = [inp for inp in inputs if inp in computed_set]

    colidxs = _get_column_indices(found_inputs, data_cols, list_params, extra_fields)

    return ColumnLookupResult(colidxs=colidxs, computed=computed)


def _collect_cols_arrays(data, _args, colidxs):
    """Extract arrays from data or args based on column index mappings.

    Parameters
    ----------
    data : np.ndarray
        2D array containing trial data (e.g., rt, response).
    _args : tuple
        Model parameters and extra fields passed as separate arrays.
    colidxs : dict[str, tuple[str, int]]
        Mapping of column names to (source, index) tuples where source is
        either "data" or "args".

    Returns
    -------
    list
        Arrays extracted from their respective sources in the order of colidxs keys.
    """

    def _extract_array(source, idx):
        return data[:, idx] if source == "data" else _args[idx]

    collected = [_extract_array(source, idx) for source, idx in colidxs.values()]
    return collected


def make_rl_logp_func(
    ssm_logp_func: AnnotatedFunction,
    n_participants: int,
    n_trials: int,
    data_cols: list[str] | None = None,
    list_params: list[str] | None = None,
    extra_fields: list[str] | None = None,
) -> Callable:
    """Create a log-likelihood function for models with computed parameters.

    Factory function that builds a complete log-likelihood for sequential sampling
    models where some parameters are computed by other models (e.g., drift rates
    computed from reinforcement learning).

    Parameters
    ----------
    ssm_logp_func : AnnotatedFunction
        Log-likelihood function for the sequential sampling model, decorated with
        `@annotate_function`. Must have `.inputs`, `.outputs`, and `.computed`
        attributes specifying parameter dependencies.
    n_participants : int
        Number of participants in the dataset.
    n_trials : int
        Number of trials per participant.
    data_cols : list[str] | None, optional
        Column names in the data array. Defaults to `["rt", "response"]`.
    list_params : list[str] | None, optional
        Model parameter names passed as separate arrays in `*args`. Defaults to None.
    extra_fields : list[str] | None, optional
        Additional fields (e.g., 'feedback') required by computation functions.
        Defaults to None.

    Returns
    -------
    Callable
        Log-likelihood function with signature `logp(data, *args) -> np.ndarray`.
        Automatically computes dependent parameters and evaluates the SSM likelihood.
    """
    data_cols = data_cols or ["rt", "response"]
    list_params = list_params or []
    extra_fields = extra_fields or []

    vmapped_func = jax.vmap(ssm_logp_func.computed["v"], in_axes=0)

    def logp(data, *args) -> np.ndarray:
        """Compute the log-likelihood for the RLDM model for each trial.

        This function computes the full log-likelihood for a reinforcement learning
        drift-diffusion model (RLDM). It first computes any required intermediate
        parameters (such as drift rates) for each trial, and then evaluates the
        sequential sampling model (SSM) likelihood using these parameters.
        Parameters.
        ----------
        data : np.ndarray
            A 2D array containing trial data.
        *args :
            Model parameters included in list_params and extra_fields.

        Returns
        -------
        np.ndarray
            The computed log-likelihoods for each trial, reshaped as a 2D array.
        """
        # Reshape subj_trials into a 3D array of shape
        # (n_participants, n_trials, len(args))
        # so we can act on this object with the vmapped version of the mapping function
        # TODO: Generalize to handle every member in computed
        # Get column indices for SSM logp function
        ssm_logp_func_colidxs = _get_column_indices_with_computed(
            ssm_logp_func,
            data_cols,
            list_params,
            extra_fields,
        )
        computed_colidxs1 = _get_column_indices(
            ssm_logp_func.computed["v"].inputs, data_cols, list_params, extra_fields
        )
        computed_colidxs1_data = _collect_cols_arrays(data, args, computed_colidxs1)
        subj_trials = jnp.stack(computed_colidxs1_data, axis=1)
        subj_trials = subj_trials.reshape(n_participants, n_trials, -1)
        computed_arg = vmapped_func(subj_trials)
        computed_arg = computed_arg.reshape((-1, 1))

        non_computed_args = _collect_cols_arrays(
            data, args, ssm_logp_func_colidxs.colidxs
        )
        # create parameter arrays to be passed to the likelihood function
        ddm_params_matrix = jnp.stack(non_computed_args, axis=1)
        lan_matrix = jnp.concatenate((computed_arg, ddm_params_matrix), axis=1)
        return ssm_logp_func(lan_matrix)

    return logp


# TODO[CP]: Adapt this function given the changes to make_rl_logp_func (#857)
# pragma: no cover
def make_rl_logp_op(
    ssm_logp_func: AnnotatedFunction,
    n_participants: int,
    n_trials: int,
    data_cols: list[str] | None = None,
    list_params: list[str] | None = None,
    extra_fields: list[str] | None = None,
) -> Op:
    """Create a log-likelihood pytensor Op for models with computed parameters.

    Factory function that builds a complete log-likelihood Op for reinforcement learning
    models where some parameters are computed by other models (e.g., drift rates
    computed from reinforcement learning).

    Parameters
    ----------
    ssm_logp_func : AnnotatedFunction
        Log-likelihood function for the sequential sampling model, decorated with
        `@annotate_function`. It must have `.inputs`, `.outputs`, and `.computed`
        attributes specifying parameter dependencies.
    n_participants : int
        Number of participants in the dataset.
    n_trials : int
        Number of trials per participant.
    data_cols : list[str] | None, optional
        Column names in the data array. Defaults to `["rt", "response"]`.
    list_params : list[str] | None, optional
        Model parameter names passed as separate arrays in `*args`. Defaults to None.
    extra_fields : list[str] | None, optional
        Additional fields (e.g., 'feedback') required by computation functions.
        Defaults to None.

    Returns
    -------
    Op
        A PyTensor Op that wraps the log-likelihood computation, automatically computing
        dependent parameters and evaluating the SSM likelihood.
    """
    logp = make_rl_logp_func(
        ssm_logp_func,
        n_participants,
        n_trials,
        data_cols,
        list_params,
        extra_fields,
    )
    n_params = len(list_params or [])
    vjp_logp = make_vjp_func(logp, params_only=False, n_params=n_params)

    return make_jax_logp_ops(
        logp=jax.jit(logp),
        logp_vjp=jax.jit(vjp_logp),
        logp_nojit=logp,
        n_params=n_params,
    )

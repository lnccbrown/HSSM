"""The log-likelihood function for the RLDM model."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.lax import scan

from hssm.distribution_utils.func_utils import make_vjp_func
from hssm.distribution_utils.jax import make_jax_logp_ops

if TYPE_CHECKING:
    from pytensor.graph import Op


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
                "or `extra_fields`."
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


def _collect_cols_arrays(
    data: np.ndarray, _args: tuple, colidxs: dict[str, tuple[str, int]]
) -> list[np.ndarray]:
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

    def _extract_array(source: str, idx: int) -> np.ndarray:
        return data[:, idx] if source == "data" else _args[idx]

    collected = [_extract_array(source, idx) for source, idx in colidxs.values()]
    return collected


def _validate_computed_parameters(
    ssm_logp_func: AnnotatedFunction,
    computed_params: list[str],
) -> None:
    """Validate that all computed parameters have corresponding compute functions.

    Parameters
    ----------
    ssm_logp_func : AnnotatedFunction
        The SSM log-likelihood function that should contain compute functions
        for computed parameters.
    computed_params : list[str]
        List of parameter names identified as computed (not available in
        data_cols, list_params, or extra_fields).

    Raises
    ------
    ValueError
        If computed parameters are identified but no compute functions are
        provided, or if some computed parameters lack compute functions.
    """
    if not computed_params:
        return

    if not hasattr(ssm_logp_func, "computed") or not ssm_logp_func.computed:
        raise ValueError(
            f"Parameters {computed_params} are not available "
            "in data_cols, list_params, or extra_fields, but no compute "
            "functions are provided in ssm_logp_func.computed"
        )

    missing_compute_funcs = set(computed_params) - ssm_logp_func.computed.keys()
    if missing_compute_funcs:
        raise ValueError(
            f"Parameters {sorted(missing_compute_funcs)} are identified as "
            "computed but no compute functions are provided in "
            "ssm_logp_func.computed"
        )


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
        A non-jitted JAX log-likelihood function for the sequential sampling model,
        decorated with `@annotate_function`. It must have `.inputs`, `.outputs`, and
        `.computed` attributes specifying parameter dependencies.
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

    # Pre-compute vmapped versions of all compute functions
    vmapped_compute_funcs = (
        {
            param_name: jax.vmap(compute_func, in_axes=0)
            for param_name, compute_func in ssm_logp_func.computed.items()
        }
        if hasattr(ssm_logp_func, "computed") and ssm_logp_func.computed
        else {}
    )

    def _prepare_subj_trials(
        compute_func: AnnotatedFunction, data: np.ndarray, args: tuple
    ) -> jnp.ndarray:
        """Extract and reshape data for a computation function."""
        # Get column indices for inputs needed by computation function
        colidxs = _get_column_indices(
            compute_func.inputs, data_cols, list_params, extra_fields
        )
        # Extract and organize data for this computation
        cols_data = _collect_cols_arrays(data, args, colidxs)
        # Reshape into 3D array (n_participants, n_trials, n_inputs)
        subj_trials = jnp.stack(cols_data, axis=1)
        return subj_trials.reshape(n_participants, n_trials, -1)

    def compute_parameter(
        param_name: str, data: np.ndarray, args: tuple
    ) -> jnp.ndarray:
        """Compute a single parameter."""
        compute_func = ssm_logp_func.computed[param_name]
        vmapped_func = vmapped_compute_funcs[param_name]

        # Prepare data for computation
        subj_trials = _prepare_subj_trials(compute_func, data, args)

        # Apply pre-vmapped function to compute parameter values
        computed_values = vmapped_func(subj_trials)

        # Reshape back to 2D (total_trials, 1) for concatenation
        return computed_values.reshape((-1, 1))

    def logp(data, *args) -> Array:
        """Compute the log-likelihood for the RLDM model for each trial.

        This function computes the full log-likelihood for a reinforcement learning
        drift-diffusion model (RLDM). It first computes any required intermediate
        parameters (such as drift rates) for each trial, and then evaluates the
        sequential sampling model (SSM) likelihood using these parameters.

        Parameters
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
        # Get column indices for SSM logp function
        # Identifies computed vs available params
        ssm_logp_func_colidxs = _get_column_indices_with_computed(
            ssm_logp_func,
            data_cols,
            list_params,
            extra_fields,
        )

        # Validate that all computed parameters have compute functions
        _validate_computed_parameters(ssm_logp_func, ssm_logp_func_colidxs.computed)

        computed_param_values = (
            {
                param_name: compute_parameter(param_name, data, args)
                for param_name in ssm_logp_func_colidxs.computed
            }
            if hasattr(ssm_logp_func, "computed") and ssm_logp_func.computed
            else {}
        )

        # Extract non-computed parameters
        non_computed_args = _collect_cols_arrays(
            data, args, ssm_logp_func_colidxs.colidxs
        )

        # Build final parameter matrix maintaining order from ssm_logp_func.inputs
        # This ensures parameters appear in the sequence expected by the SSM likelihood
        def get_param_array(param_name: str) -> Array | np.ndarray:
            """Get parameter array from computed values or non-computed args."""
            if param_name in computed_param_values:
                return computed_param_values[param_name]
            # Use non-computed value (from data or args)
            # Find its position in non_computed_args based on colidxs order
            colidxs_keys = list(ssm_logp_func_colidxs.colidxs.keys())
            idx = colidxs_keys.index(param_name)
            return non_computed_args[idx].reshape((-1, 1))

        param_arrays = [get_param_array(name) for name in ssm_logp_func.inputs]

        # Stack all parameters into final matrix
        lan_matrix = jnp.concatenate(param_arrays, axis=1)
        return ssm_logp_func(lan_matrix)

    return logp


def make_rl_logp_op(
    ssm_logp_func: AnnotatedFunction,
    n_participants: int,
    n_trials: int,
    data_cols: list[str] | None = None,
    list_params: list[str] | None = None,
    extra_fields: list[str] | None = None,
) -> "Op":
    """Create a log-likelihood pytensor Op for models with computed parameters.

    Factory function that builds a complete log-likelihood Op for reinforcement learning
    models where some parameters are computed by other models (e.g., drift rates
    computed from reinforcement learning).

    Parameters
    ----------
    ssm_logp_func : AnnotatedFunction
        A non-jitted JAX log-likelihood function for the sequential sampling model,
        decorated with `@annotate_function`. It must have `.inputs`, `.outputs`, and
        `.computed` attributes specifying parameter dependencies.
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

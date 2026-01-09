"""The log-likelihood function for the RLDM model.

Column Ordering Contract
------------------------
The builder functions (`make_rl_logp_func` and `make_rl_logp_op`) require that data
and parameter arrays follow a strict ordering contract:

1. **data_cols**: Column names in the data array, in exact order
   - data[:, i] corresponds to data_cols[i]
   - Example: if data_cols=["rt", "response"], then:
     * data[:, 0] contains rt values
     * data[:, 1] contains response values

2. **list_params**: Model parameter names in *args, in exact order
   - args[i] corresponds to list_params[i]
   - Example: if list_params=["a", "z", "t"], then:
     * args[0] contains 'a' values
     * args[1] contains 'z' values
     * args[2] contains 't' values

3. **extra_fields**: Additional field names in *args after list_params, in exact order
   - args[len(list_params) + i] corresponds to extra_fields[i]
   - Example: if list_params=["a", "z"] and extra_fields=["feedback", "stimulus"]:
     * args[0] = 'a' (list_params[0])
     * args[1] = 'z' (list_params[1])
     * args[2] = 'feedback' (extra_fields[0])
     * args[3] = 'stimulus' (extra_fields[1])

Together, data_cols + list_params + extra_fields form the complete ordered column
metadata that the builder uses to look up available data sources and map them to
function inputs.

This ordering is validated at runtime by `_validate_inputs` and its component
validation functions.
"""

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

    This function implements the column ordering contract defined at module level,
    mapping column names to their sources and indices in either the data array or
    the args tuple.

    Parameters
    ----------
    cols_to_look_up : list[str]
        Columns to find indices for
    data_cols : list[str]
        Available data columns (data[:, i] maps to data_cols[i])
    list_params : list[str] | None
        Available list parameters (args[i] maps to list_params[i])
    extra_fields : list[str] | None
        Available extra fields (args[len(list_params)+i] maps to extra_fields[i])

    Returns
    -------
    dict[str, tuple[str, int]]
        Mapping of column names to (source, index) tuples where source is
        "data" for data columns or "args" for list_params/extra_fields.
        The index specifies position in either data array or args tuple.

    Raises
    ------
    ValueError
        If any column in cols_to_look_up is not found in data_cols, list_params,
        or extra_fields.

    Notes
    -----
    This function enforces the column ordering contract where data_cols,
    list_params, and extra_fields together form the complete ordered metadata
    for available data sources. See module docstring for full contract details.

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


def _validate_data_shape(
    data: np.ndarray,
    data_cols: list[str],
) -> None:
    """Validate that data array has correct number of columns.

    Parameters
    ----------
    data : np.ndarray
        Data array to validate.
    data_cols : list[str]
        Expected column names.

    Raises
    ------
    ValueError
        If data doesn't have expected number of columns or is not 2D.
    """
    if data.ndim != 2:
        raise ValueError(
            f"Data array must be 2D, but got shape {data.shape} with "
            f"{data.ndim} {'dimension' if data.ndim == 1 else 'dimensions'}."
        )

    if data.shape[1] != len(data_cols):
        raise ValueError(
            f"Data array has {data.shape[1]} columns but {len(data_cols)} "
            f"columns were specified in data_cols: {data_cols}."
        )


def _validate_args_length(
    args: tuple,
    list_params: list[str],
    extra_fields: list[str],
) -> None:
    """Validate that args tuple has correct number of elements.

    Parameters
    ----------
    args : tuple
        Arguments tuple containing parameter arrays.
    list_params : list[str]
        Expected list parameters.
    extra_fields : list[str]
        Expected extra fields.

    Raises
    ------
    ValueError
        If args doesn't have expected number of elements.
    """
    expected_n_args = len(list_params) + len(extra_fields)
    if len(args) != expected_n_args:
        raise ValueError(
            f"Expected {expected_n_args} argument arrays "
            f"({len(list_params)} list_params + {len(extra_fields)} extra_fields), "
            f"but got {len(args)}."
        )


def _validate_uniform_trials(
    data: np.ndarray,
    n_participants: int,
    n_trials: int,
) -> None:
    """Validate that all participants have the same number of trials.

    Parameters
    ----------
    data : np.ndarray
        Data array containing all trials.
    n_participants : int
        Number of participants.
    n_trials : int
        Expected number of trials per participant.

    Raises
    ------
    ValueError
        If total number of trials doesn't match n_participants * n_trials.
    """
    total_trials = data.shape[0]
    expected_trials = n_participants * n_trials

    if total_trials != expected_trials:
        raise ValueError(
            f"Data has {total_trials} total trials, but with {n_participants} "
            f"participants and {n_trials} trials per participant, expected "
            f"{expected_trials} trials. All participants must have the same "
            "number of trials."
        )


def _validate_args_array_shapes(
    args: tuple,
    expected_length: int,
    list_params: list[str],
    extra_fields: list[str],
) -> None:
    """Validate that all arrays in args have the same length.

    Parameters
    ----------
    args : tuple
        Argument arrays to validate.
    expected_length : int
        Expected length for all arrays.
    list_params : list[str]
        Names of list parameters for error messages.
    extra_fields : list[str]
        Names of extra fields for error messages.

    Raises
    ------
    ValueError
        If any array in args has incorrect length.
    """
    all_params = list_params + extra_fields
    for i, (arg, param_name) in enumerate(zip(args, all_params)):
        if not hasattr(arg, "shape") or not hasattr(arg, "__len__"):
            raise ValueError(
                f"Argument {i} ('{param_name}') is not an array-like object."
            )

        if len(arg) != expected_length:
            raise ValueError(
                f"Argument {i} ('{param_name}') has length {len(arg)}, "
                f"but expected {expected_length} (matching data.shape[0])."
            )


def _validate_inputs(
    data: np.ndarray,
    args: tuple,
    n_participants: int,
    n_trials: int,
    data_cols: list[str],
    list_params: list[str],
    extra_fields: list[str],
) -> None:
    """Validate all inputs to the log-likelihood function.

    Performs comprehensive validation of data array and parameter arrays
    to ensure they have correct shapes and sizes before likelihood computation.

    Parameters
    ----------
    data : np.ndarray
        Data array containing trial data.
    args : tuple
        Argument arrays containing parameter values.
    n_participants : int
        Number of participants.
    n_trials : int
        Number of trials per participant.
    data_cols : list[str]
        Expected data column names.
    list_params : list[str]
        Expected list parameter names.
    extra_fields : list[str]
        Expected extra field names.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    _validate_data_shape(data, data_cols)
    _validate_args_length(args, list_params, extra_fields)
    _validate_uniform_trials(data, n_participants, n_trials)
    _validate_args_array_shapes(args, data.shape[0], list_params, extra_fields)


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
        Column names in the data array, in the exact order they appear as columns
        in the `data` array passed to the returned logp function. For example, if
        `data_cols=["rt", "response"]`, then the data array must have exactly 2
        columns where column 0 contains rt values and column 1 contains response
        values. Defaults to `["rt", "response"]`.
    list_params : list[str] | None, optional
        Model parameter names passed as separate arrays in `*args`, in the exact
        order they appear in the `*args` tuple. For example, if
        `list_params=["a", "z", "t"]`, then args[0] contains 'a' values,
        args[1] contains 'z' values, and args[2] contains 't' values.
        Together with `data_cols` and `extra_fields`, these form the complete
        ordered column metadata for looking up available data. Defaults to None.
    extra_fields : list[str] | None, optional
        Additional fields (e.g., 'feedback') required by computation functions,
        passed as separate arrays in `*args` after list_params, in the exact order
        they appear. For example, if `list_params=["a", "z"]` and
        `extra_fields=["feedback", "stimulus"]`, then args[0]='a', args[1]='z',
        args[2]='feedback', args[3]='stimulus'. Together with `data_cols` and
        `list_params`, these form the complete ordered column metadata.
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
            A 2D array of shape (n_total_trials, n_data_cols) containing trial data.
            Columns must be in the exact order specified in data_cols. For example,
            if data_cols=["rt", "response"], then data[:, 0] contains rt values and
            data[:, 1] contains response values.
        *args :
            Model parameters and extra fields as separate arrays, in the order:
            first all list_params arrays, then all extra_fields arrays. Each array
            must have length n_total_trials.

        Returns
        -------
        np.ndarray
            The computed log-likelihoods for each trial, reshaped as a 2D array.
        """
        # Validate inputs
        _validate_inputs(
            data, args, n_participants, n_trials, data_cols, list_params, extra_fields
        )

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
        Column names in the data array, in the exact order they appear as columns
        in the `data` array passed to the returned logp function. For example, if
        `data_cols=["rt", "response"]`, then the data array must have exactly 2
        columns where column 0 contains rt values and column 1 contains response
        values. Defaults to `["rt", "response"]`.
    list_params : list[str] | None, optional
        Model parameter names passed as separate arrays in `*args`, in the exact
        order they appear in the `*args` tuple. For example, if
        `list_params=["a", "z", "t"]`, then args[0] contains 'a' values,
        args[1] contains 'z' values, and args[2] contains 't' values.
        Together with `data_cols` and `extra_fields`, these form the complete
        ordered column metadata for looking up available data. Defaults to None.
    extra_fields : list[str] | None, optional
        Additional fields (e.g., 'feedback') required by computation functions,
        passed as separate arrays in `*args` after list_params, in the exact order
        they appear. For example, if `list_params=["a", "z"]` and
        `extra_fields=["feedback", "stimulus"]`, then args[0]='a', args[1]='z',
        args[2]='feedback', args[3]='stimulus'. Together with `data_cols` and
        `list_params`, these form the complete ordered column metadata.
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

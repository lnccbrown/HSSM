from pathlib import Path
from typing import NamedTuple
import pytest

import numpy as np
import pytensor
import pytensor.tensor as pt
import jax
import jax.numpy as jnp

import hssm
from hssm.rl.likelihoods.builder import (
    make_rl_logp_func,
    make_rl_logp_op,
    compute_v_subject_wise,
    annotate_function,
    _get_column_indices,
    _get_column_indices_with_computed,
    _collect_cols_arrays,
)
from hssm.distribution_utils.func_utils import make_vjp_func

from hssm.distribution_utils.onnx import make_jax_matrix_logp_funcs_from_onnx
# Obtain the angle log-likelihood function from an ONNX model.

angle_logp_jax_func = make_jax_matrix_logp_funcs_from_onnx(
    model="angle.onnx",
)

hssm.set_floatX("float32")

DECIMAL = 2


class ModelConfig(NamedTuple):
    """Model configuration constants."""

    list_params: list[str]
    extra_fields: list[str]
    data_cols: list[str]


class RLDMData(NamedTuple):
    """RLDM data and metadata."""

    data: np.ndarray
    participant_id: np.ndarray
    trial: np.ndarray
    subj: np.ndarray
    total_trials: int
    n_participants: int
    n_trials_per_participant: int


class ParamArrays(NamedTuple):
    """Parameter arrays for testing."""

    rl_alpha: np.ndarray
    scaler: np.ndarray
    a: np.ndarray
    z: np.ndarray
    t: np.ndarray
    theta: np.ndarray
    feedback: np.ndarray


class RLDMSetup(NamedTuple):
    """Named tuple for RLDM test setup data."""

    data: np.ndarray
    values: np.ndarray
    logp_fn: callable
    total_trials: int
    args: list


@pytest.fixture
def fixture_path():
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def model_config():
    """Shared model configuration constants."""
    return {
        "list_params": ["rl.alpha", "scaler", "a", "Z", "t", "theta"],
        "extra_fields": ["feedback"],
        "data_cols": ["rt", "response"],
    }


@pytest.fixture
def rldm_data(fixture_path):
    """Load RLDM data and extract metadata."""
    data = np.load(fixture_path / "rldm_data.npy", allow_pickle=True).item()["data"]
    participant_id = data["participant_id"].values
    trial = data["trial"].values
    subj = np.unique(participant_id).astype(np.int32)
    total_trials = trial.size

    return {
        "data": data,
        "participant_id": participant_id,
        "trial": trial,
        "subj": subj,
        "total_trials": total_trials,
        "n_participants": len(subj),
        "n_trials_per_participant": total_trials // len(subj),
    }


@pytest.fixture
def param_arrays(rldm_data):
    """Create parameter arrays for testing."""
    total_trials = rldm_data["total_trials"]
    feedback = rldm_data["data"]["feedback"].values

    return {
        "rl_alpha": np.ones(total_trials) * 0.60,
        "scaler": np.ones(total_trials) * 3.2,
        "a": np.ones(total_trials) * 1.2,
        "z": np.ones(total_trials) * 0.1,
        "t": np.ones(total_trials) * 0.1,
        "theta": np.ones(total_trials) * 0.1,
        "feedback": feedback,
    }


@pytest.fixture
def annotated_ssm_logp_func():
    """Create the annotated SSM log-likelihood function for RLDM models."""
    compute_v_subject_wise_annotated = annotate_function(
        inputs=["rl.alpha", "scaler", "response", "feedback"],
        outputs=["v"],
    )(compute_v_subject_wise)
    ssm_logp_func = annotate_function(
        inputs=["v", "a", "Z", "t", "theta", "rt", "response"],
        computed={"v": compute_v_subject_wise_annotated},
    )(angle_logp_jax_func)
    return ssm_logp_func


@pytest.fixture
def rldm_setup(rldm_data, model_config, param_arrays, annotated_ssm_logp_func):
    """Create complete RLDM setup using shared fixtures."""
    logp_fn = make_rl_logp_func(
        annotated_ssm_logp_func,
        n_participants=rldm_data["n_participants"],
        n_trials=rldm_data["n_trials_per_participant"],
        data_cols=model_config["data_cols"],
        list_params=model_config["list_params"],
        extra_fields=model_config["extra_fields"],
    )

    # Map parameter names to param_arrays keys
    param_key_map = {
        "rl.alpha": "rl_alpha",
        "scaler": "scaler",
        "a": "a",
        "Z": "z",
        "t": "t",
        "theta": "theta",
    }
    args = [param_arrays[param_key_map[p]] for p in model_config["list_params"]] + [
        param_arrays["feedback"]
    ]

    return RLDMSetup(
        data=rldm_data["data"],
        values=rldm_data["data"].values,
        logp_fn=logp_fn,
        total_trials=rldm_data["total_trials"],
        args=args,
    )


class TestGetDataColumnsFromDataArgs:
    def test_get_column_indices(self):
        """Test the column indexing and data collection functionality.

        This test verifies that _get_column_indices correctly maps column names to their
        sources (either 'data' matrix or 'args' parameters) and that _collect_cols_arrays
        properly extracts and organizes the data for further processing.

        The test simulates the RLDM likelihood scenario where we need to collect:
        - Data columns from the input data matrix (rt, response)
        - Parameter arrays from the args list (model parameters like rl.alpha, scaler, etc.)
        - Extra fields that are passed as additional parameters (feedback)
        """
        # Define the structure of available data columns
        # These represent columns in the input data matrix (e.g., reaction time, response)
        data_cols = ["rt", "response"]

        # Define model parameters that will be passed as separate arrays
        list_params = ["rl.alpha", "scaler", "a", "Z", "t", "theta"]

        # Define extra fields that are needed for RL computation but not core DDM params
        extra_fields = ["feedback"]

        # Combine parameters and extra fields to get the full args list structure
        list_params_extra_fields = list_params + extra_fields

        # Define which columns we want to extract for this specific computation
        # This represents the subset needed for the RL likelihood computation
        cols_to_look_up = ["rl.alpha", "scaler", "response", "feedback"]

        # Create mock data matrix: 10 trials with 2 columns [rt=1.0, response=2.0]
        data = np.array([1, 2]) * np.ones((10, 2))

        # Create mock parameter arrays: each parameter gets a different constant value
        # _args[0] = [3, 3, 3, ...] (rl.alpha), _args[1] = [4, 4, 4, ...] (scaler), etc.
        _args = [
            np.ones(10) * i for i, _ in enumerate(list_params_extra_fields, start=3)
        ]

        ssm_logp_func = angle_logp_jax_func
        ssm_logp_func.inputs = ["v", "a", "z", "t", "theta", "rt", "response"]
        # Call the function under test: get indices for where to find each column
        indices = _get_column_indices(
            cols_to_look_up, data_cols, list_params, extra_fields
        )

        # Expected mapping: each column name maps to (source, index) tuple
        # - "rl.alpha": found in args at index 0
        # - "scaler": found in args at index 1
        # - "response": found in data at column index 1
        # - "feedback": found in args at index 6 (len(list_params) = 6)
        expected_indices = {
            "rl.alpha": ("args", 0),
            "scaler": ("args", 1),
            "response": ("data", 1),
            "feedback": ("args", 6),
        }

        # Use the indices to collect the actual data arrays
        collected_arrays = _collect_cols_arrays(data, _args, indices)
        # Stack the collected arrays into a matrix for easier verification
        stacked_arrays = np.stack(collected_arrays, axis=1)

        # Verify the indices mapping is correct
        assert indices == expected_indices

        # Verify the collected data values are correct:
        # - rl.alpha: _args[0] = [3, 3, 3, ...]
        # - scaler: _args[1] = [4, 4, 4, ...]
        # - response: data[:, 1] = [2, 2, 2, ...]
        # - feedback: _args[6] = [9, 9, 9, ...]
        expected_stacked_arrays = np.array([[3.0, 4.0, 2.0, 9.0]] * 10)
        np.testing.assert_array_equal(expected_stacked_arrays, stacked_arrays)

        with pytest.raises(ValueError, match="Column 'non_existent_param' not found"):
            _get_column_indices(
                ["non_existent_param"],
                data_cols,
                list_params,
                extra_fields,
            )

    def test_get_column_indices_with_computed(self):
        """Test the column indexing with computed parameters functionality.

        1. Identifies which inputs from a function's .inputs attribute can be found
           in available data sources (data_cols, list_params, extra_fields)
        2. Marks inputs that cannot be found as "computed" parameters
        3. Returns both the column indices for found inputs and the list of computed parameters
        """
        # Define the structure of available data columns and parameters
        data_cols = ["rt", "response"]
        list_params = ["rl.alpha", "scaler", "a", "Z", "t", "theta"]
        extra_fields = ["feedback"]

        # Create a mock function object with .inputs attribute
        # This simulates an SSM log-likelihood function that needs both
        # regular parameters and computed parameters (like drift rates from RL)

        class MockSSMLogpFunc:
            inputs = ["v", "a", "Z", "t", "theta", "rt", "response"]

        result = _get_column_indices_with_computed(
            MockSSMLogpFunc(), data_cols, list_params, extra_fields
        )

        # Expected results:
        # - "rt", "response" in data_cols
        # - "a", "Z", "t", "theta" in list_params
        # - "v" NOT found, should be marked as computed
        expected_colidxs = {
            "rt": ("data", 0),
            "response": ("data", 1),
            "a": ("args", 2),
            "Z": ("args", 3),
            "t": ("args", 4),
            "theta": ("args", 5),
        }
        expected_computed = ["v"]

        assert result.colidxs == expected_colidxs
        assert result.computed == expected_computed

        # Test with a function that has all parameters available (no computed)
        class MockFuncAllFound:
            inputs = ["rt", "response", "rl.alpha", "feedback"]

        result_all_found = _get_column_indices_with_computed(
            MockFuncAllFound(), data_cols, list_params, extra_fields
        )

        expected_colidxs_all_found = {
            "rt": ("data", 0),
            "response": ("data", 1),
            "rl.alpha": ("args", 0),
            "feedback": ("args", 6),
        }

        assert result_all_found.colidxs == expected_colidxs_all_found
        assert result_all_found.computed == []  # No computed parameters

        # Test with a function that has all computed parameters
        class MockFuncAllComputed:
            inputs = ["computed_param1", "computed_param2"]

        result_all_computed = _get_column_indices_with_computed(
            MockFuncAllComputed(), data_cols, list_params, extra_fields
        )

        assert result_all_computed.colidxs == {}  # No found parameters
        assert result_all_computed.computed == ["computed_param1", "computed_param2"]

        # Test with a function missing the inputs attribute
        with pytest.raises(
            TypeError, match="func_with_inputs must have an 'inputs' attribute"
        ):
            _get_column_indices_with_computed(
                lambda x: 0, data_cols, list_params, extra_fields
            )


class TestAnnotateFunction:
    def test_decorator_adds_attributes(self):
        @annotate_function(inputs=["input1", "input2"], outputs=["output1"], other=42)
        def sample_function():
            return

        assert sample_function.inputs == ["input1", "input2"]
        assert sample_function.outputs == ["output1"]
        assert sample_function.other == 42


class TestRldmLikelihoodAbstraction:
    def test_make_rl_logp_func(self, rldm_setup):
        result = rldm_setup.logp_fn(rldm_setup.values, *rldm_setup.args)
        assert result.shape[0] == rldm_setup.total_trials
        np.testing.assert_almost_equal(result.sum(), -39215.64, decimal=DECIMAL)

    def test_make_rl_logp_op(
        self, rldm_setup, rldm_data, model_config, param_arrays, annotated_ssm_logp_func
    ):
        """Test that make_rl_logp_op creates a working PyTensor Op.

        This test verifies that:
        1. The Op can be called and produces correct log-likelihood values
        2. The Op can compute gradients with respect to parameters
        3. The Op produces the same results as make_rl_logp_func
        """
        # Create the Op
        logp_op = make_rl_logp_op(
            annotated_ssm_logp_func,
            n_participants=rldm_data["n_participants"],
            n_trials=rldm_data["n_trials_per_participant"],
            data_cols=model_config["data_cols"],
            list_params=model_config["list_params"],
            extra_fields=model_config["extra_fields"],
        )

        # Test 1: Op produces correct output
        result_op = logp_op(rldm_setup.values, *rldm_setup.args)
        result_eval = result_op.eval()

        assert result_eval.shape[0] == rldm_data["total_trials"]
        np.testing.assert_almost_equal(result_eval.sum(), -39215.64, decimal=DECIMAL)

        # Test 2: Op produces same results as make_rl_logp_func
        result_func = rldm_setup.logp_fn(rldm_setup.values, *rldm_setup.args)
        np.testing.assert_array_almost_equal(result_eval, result_func, decimal=DECIMAL)

        # Test 3: Op can compute gradients
        rl_alpha_var = pt.as_tensor_variable(
            param_arrays["rl_alpha"].astype(np.float32)
        )
        args_float32 = [arr.astype(np.float32) for arr in rldm_setup.args]
        args_float32[0] = rl_alpha_var  # Replace first arg with variable

        logp_with_var = logp_op(rldm_setup.values.astype(np.float32), *args_float32)
        grad_rl_alpha = pytensor.grad(logp_with_var.sum(), wrt=rl_alpha_var)
        grad_eval = grad_rl_alpha.eval()

        assert grad_eval.shape == param_arrays["rl_alpha"].shape
        assert not np.allclose(grad_eval, 0.0), "Gradient should not be all zeros"

    def test_rl_logp_func_vjp_jitted(
        self, rldm_setup, rldm_data, model_config, param_arrays
    ):
        """Test that the VJP of rl_logp_func can be jitted and produces correct gradients.

        This test verifies that:
        1. The VJP function can be successfully JIT-compiled
        2. The jitted VJP produces the same results as the non-jitted version
        3. Gradients are computed correctly with respect to parameters
        """
        # Create VJP and jitted versions
        n_params = len(model_config["list_params"])
        vjp_fn = make_vjp_func(rldm_setup.logp_fn, params_only=False, n_params=n_params)
        logp_fn_jit = jax.jit(rldm_setup.logp_fn)
        vjp_fn_jit = jax.jit(vjp_fn)

        # Prepare JAX inputs
        data_values = rldm_setup.values.astype(np.float32)
        jax_args = tuple(jnp.asarray(arg, dtype=jnp.float32) for arg in rldm_setup.args)

        # Test 1: Jitted logp produces same results as non-jitted
        logp_result = rldm_setup.logp_fn(data_values, *jax_args)
        logp_result_jit = logp_fn_jit(data_values, *jax_args)
        np.testing.assert_array_almost_equal(
            logp_result, logp_result_jit, decimal=DECIMAL
        )

        # Test 2: Jitted VJP produces same results as non-jitted VJP
        gz = jnp.ones_like(logp_result)
        vjp_result = vjp_fn(data_values, *jax_args, gz=gz)
        vjp_result_jit = vjp_fn_jit(data_values, *jax_args, gz=gz)

        assert len(vjp_result) == n_params
        assert len(vjp_result_jit) == n_params
        for i, (grad_nojit, grad_jit) in enumerate(zip(vjp_result, vjp_result_jit)):
            param_name = model_config["list_params"][i]
            np.testing.assert_array_almost_equal(
                grad_nojit,
                grad_jit,
                decimal=DECIMAL,
                err_msg=f"Gradient mismatch for parameter {i} ({param_name})",
            )

        # Test 3: Gradients are not all zeros
        for i, grad in enumerate(vjp_result):
            param_name = model_config["list_params"][i]
            assert not np.allclose(grad, 0.0), (
                f"Gradient for {param_name} should not be all zeros"
            )

    def test_rl_logp_func_vjp_consistency(self, rldm_setup, model_config):
        """Test that VJP computed via make_vjp_func matches JAX's automatic differentiation.

        This test verifies that the custom VJP implementation produces the same
        gradients as JAX's built-in grad function for a subset of parameters.
        """
        # Create VJP function
        n_params = len(model_config["list_params"])
        vjp_fn = make_vjp_func(rldm_setup.logp_fn, params_only=False, n_params=n_params)

        # Prepare JAX inputs
        data_values = rldm_setup.values.astype(np.float32)
        jax_args = tuple(jnp.asarray(arg, dtype=jnp.float32) for arg in rldm_setup.args)

        # Compute VJP using make_vjp_func
        logp_result = rldm_setup.logp_fn(data_values, *jax_args)
        gz = jnp.ones_like(logp_result)
        vjp_result = vjp_fn(data_values, *jax_args, gz=gz)

        # Compute gradient using JAX's grad for the first parameter (rl_alpha)
        def sum_logp_first_param(first_param_val):
            return rldm_setup.logp_fn(data_values, first_param_val, *jax_args[1:]).sum()

        grad_first_param_jax = jax.grad(sum_logp_first_param)(jax_args[0])

        # Compare: VJP result should match JAX's grad
        np.testing.assert_array_almost_equal(
            vjp_result[0],
            grad_first_param_jax,
            decimal=DECIMAL,
            err_msg=f"VJP gradient for {model_config['list_params'][0]} doesn't match JAX's grad",
        )

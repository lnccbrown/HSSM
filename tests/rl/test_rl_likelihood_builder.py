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
    _validate_computed_parameters,
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
    return ModelConfig(
        list_params=["rl.alpha", "scaler", "a", "Z", "t", "theta"],
        extra_fields=["feedback"],
        data_cols=["rt", "response"],
    )


@pytest.fixture
def rldm_data(fixture_path):
    """Load RLDM data and extract metadata."""
    data = np.load(fixture_path / "rldm_data.npy", allow_pickle=True).item()["data"]
    participant_id = data["participant_id"].values
    trial = data["trial"].values
    subj = np.unique(participant_id).astype(np.int32)
    total_trials = trial.size

    return RLDMData(
        data=data,
        participant_id=participant_id,
        trial=trial,
        subj=subj,
        total_trials=total_trials,
        n_participants=len(subj),
        n_trials_per_participant=total_trials // len(subj),
    )


@pytest.fixture
def param_arrays(rldm_data):
    """Create parameter arrays for testing."""
    total_trials = rldm_data.total_trials
    feedback = rldm_data.data["feedback"].values

    return ParamArrays(
        rl_alpha=np.ones(total_trials) * 0.60,
        scaler=np.ones(total_trials) * 3.2,
        a=np.ones(total_trials) * 1.2,
        z=np.ones(total_trials) * 0.1,
        t=np.ones(total_trials) * 0.1,
        theta=np.ones(total_trials) * 0.1,
        feedback=feedback,
    )


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
        n_participants=rldm_data.n_participants,
        n_trials=rldm_data.n_trials_per_participant,
        data_cols=model_config.data_cols,
        list_params=model_config.list_params,
        extra_fields=model_config.extra_fields,
    )

    # Map parameter names to param_arrays attributes
    param_key_map = {
        "rl.alpha": param_arrays.rl_alpha,
        "scaler": param_arrays.scaler,
        "a": param_arrays.a,
        "Z": param_arrays.z,
        "t": param_arrays.t,
        "theta": param_arrays.theta,
    }
    args = [param_key_map[p] for p in model_config.list_params] + [
        param_arrays.feedback
    ]

    return RLDMSetup(
        data=rldm_data.data,
        values=rldm_data.data.values,
        logp_fn=logp_fn,
        total_trials=rldm_data.total_trials,
        args=args,
    )


class TestGetDataColumnsFromDataArgs:
    def test_get_column_indices(self):
        """Test column indexing and data collection from data matrix and args.

        Verifies _get_column_indices maps column names to their sources
        and _collect_cols_arrays extracts the data correctly.
        """
        data_cols = ["rt", "response"]
        list_params = ["rl.alpha", "scaler", "a", "Z", "t", "theta"]
        extra_fields = ["feedback"]
        list_params_extra_fields = list_params + extra_fields
        cols_to_look_up = ["rl.alpha", "scaler", "response", "feedback"]

        data = np.array([1, 2]) * np.ones((10, 2))
        _args = [
            np.ones(10) * i for i, _ in enumerate(list_params_extra_fields, start=3)
        ]

        indices = _get_column_indices(
            cols_to_look_up, data_cols, list_params, extra_fields
        )

        expected_indices = {
            "rl.alpha": ("args", 0),
            "scaler": ("args", 1),
            "response": ("data", 1),
            "feedback": ("args", 6),
        }

        collected_arrays = _collect_cols_arrays(data, _args, indices)
        stacked_arrays = np.stack(collected_arrays, axis=1)

        assert indices == expected_indices
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
        """Test column indexing that identifies computed parameters.

        Verifies that inputs not found in data sources are marked as computed.
        """
        # Define available data columns and parameters
        data_cols = ["rt", "response"]
        list_params = ["rl.alpha", "scaler", "a", "Z", "t", "theta"]
        extra_fields = ["feedback"]

        class MockSSMLogpFunc:
            inputs = ["v", "a", "Z", "t", "theta", "rt", "response"]

        result = _get_column_indices_with_computed(
            MockSSMLogpFunc(), data_cols, list_params, extra_fields
        )

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
        assert result_all_found.computed == []

        class MockFuncAllComputed:
            inputs = ["computed_param1", "computed_param2"]

        result_all_computed = _get_column_indices_with_computed(
            MockFuncAllComputed(), data_cols, list_params, extra_fields
        )

        assert result_all_computed.colidxs == {}
        assert result_all_computed.computed == ["computed_param1", "computed_param2"]

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


class TestValidateComputedParameters:
    """Tests for _validate_computed_parameters function."""

    def test_no_computed_params_passes(self):
        """Test that validation passes when there are no computed parameters."""

        @annotate_function(inputs=["a", "b"], outputs=["c"])
        def mock_func():
            pass

        # Should not raise any errors
        _validate_computed_parameters(mock_func, [])

    def test_valid_computed_params_passes(self):
        """Test that validation passes when all computed params have functions."""

        @annotate_function(inputs=["a", "b"], outputs=["c"])
        def compute_a():
            pass

        @annotate_function(inputs=["x", "y"], outputs=["z"])
        def compute_b():
            pass

        @annotate_function(
            inputs=["a", "b", "rt", "response"],
            computed={"a": compute_a, "b": compute_b},
        )
        def mock_ssm_func():
            pass

        # Should not raise any errors
        _validate_computed_parameters(mock_ssm_func, ["a", "b"])

    def test_missing_computed_attribute_raises(self):
        """Test error when computed params exist but no computed attribute."""

        @annotate_function(inputs=["v", "a", "z"])
        def mock_func():
            pass

        with pytest.raises(
            ValueError,
            match=r"Parameters \['v'\] are not available.*no compute functions",
        ):
            _validate_computed_parameters(mock_func, ["v"])

    def test_empty_computed_dict_raises(self):
        """Test error when computed params exist but computed dict is empty."""

        @annotate_function(inputs=["v", "a", "z"], computed={})
        def mock_func():
            pass

        with pytest.raises(
            ValueError,
            match=r"Parameters \['v'\] are not available.*no compute functions",
        ):
            _validate_computed_parameters(mock_func, ["v"])

    def test_partial_missing_compute_funcs_raises(self):
        """Test error when some computed params lack compute functions."""

        @annotate_function(inputs=["x"], outputs=["a"])
        def compute_a():
            pass

        @annotate_function(inputs=["a", "b", "c", "rt"], computed={"a": compute_a})
        def mock_func():
            pass

        with pytest.raises(ValueError, match=r"Parameters.*are identified as computed"):
            _validate_computed_parameters(mock_func, ["a", "b", "c"])

    def test_single_missing_compute_func_raises(self):
        """Test error when a single computed param lacks compute function."""

        @annotate_function(inputs=["v"], outputs=["a"])
        def compute_a():
            pass

        @annotate_function(inputs=["v", "a"], computed={"a": compute_a})
        def mock_func():
            pass

        with pytest.raises(ValueError, match=r"Parameters.*are identified as computed"):
            _validate_computed_parameters(mock_func, ["v", "a"])

    def test_integration_with_make_rl_logp_func(self, rldm_data, model_config):
        """Test that validation is triggered in make_rl_logp_func."""

        @annotate_function(
            inputs=["v", "a", "Z", "t", "theta", "rt", "response"], computed={}
        )
        def mock_ssm_func(params):
            return jnp.zeros(len(params))

        # Create the logp function
        logp_fn = make_rl_logp_func(
            mock_ssm_func,
            n_participants=rldm_data.n_participants,
            n_trials=rldm_data.n_trials_per_participant,
            data_cols=model_config.data_cols,
            list_params=["a", "Z", "t", "theta"],
            extra_fields=model_config.extra_fields,
        )

        # Validation happens when logp is called, not when created
        # This should raise because 'v' is not in data/params but no compute func
        with pytest.raises(
            ValueError,
            match=r"Parameters \['v'\] are not available.*no compute functions",
        ):
            # Create dummy data and args
            data = rldm_data.data.values
            a_vals = np.ones(rldm_data.total_trials)
            z_vals = np.ones(rldm_data.total_trials) * 0.5
            t_vals = np.ones(rldm_data.total_trials) * 0.1
            theta_vals = np.ones(rldm_data.total_trials) * 0.3
            feedback = rldm_data.data["feedback"].values

            logp_fn(data, a_vals, z_vals, t_vals, theta_vals, feedback)


class TestRldmLikelihoodAbstraction:
    def test_make_rl_logp_func(self, rldm_setup):
        result = rldm_setup.logp_fn(rldm_setup.values, *rldm_setup.args)
        assert result.shape[0] == rldm_setup.total_trials
        np.testing.assert_almost_equal(result.sum(), -39215.64, decimal=DECIMAL)

    def test_make_rl_logp_op(
        self, rldm_setup, rldm_data, model_config, param_arrays, annotated_ssm_logp_func
    ):
        """Test that make_rl_logp_op creates a working PyTensor Op.

        Verifies Op execution, gradient computation, and consistency with make_rl_logp_func.
        """
        # Create the Op
        logp_op = make_rl_logp_op(
            annotated_ssm_logp_func,
            n_participants=rldm_data.n_participants,
            n_trials=rldm_data.n_trials_per_participant,
            data_cols=model_config.data_cols,
            list_params=model_config.list_params,
            extra_fields=model_config.extra_fields,
        )

        # Test 1: Op produces correct output
        result_op = logp_op(rldm_setup.values, *rldm_setup.args)
        result_eval = result_op.eval()

        assert result_eval.shape[0] == rldm_data.total_trials
        np.testing.assert_almost_equal(result_eval.sum(), -39215.64, decimal=DECIMAL)

        # Test 2: Op produces same results as make_rl_logp_func
        result_func = rldm_setup.logp_fn(rldm_setup.values, *rldm_setup.args)
        np.testing.assert_array_almost_equal(result_eval, result_func, decimal=DECIMAL)

        # Verify gradient computation
        rl_alpha_var = pt.as_tensor_variable(param_arrays.rl_alpha.astype(np.float32))
        args_float32 = [arr.astype(np.float32) for arr in rldm_setup.args]
        args_float32[0] = rl_alpha_var  # Replace first arg with variable

        logp_with_var = logp_op(rldm_setup.values.astype(np.float32), *args_float32)
        grad_rl_alpha = pytensor.grad(logp_with_var.sum(), wrt=rl_alpha_var)
        grad_eval = grad_rl_alpha.eval()

        assert grad_eval.shape == param_arrays.rl_alpha.shape
        assert not np.allclose(grad_eval, 0.0), "Gradient should not be all zeros"

    def test_rl_logp_func_vjp_jitted(
        self, rldm_setup, rldm_data, model_config, param_arrays
    ):
        """Test VJP JIT compilation and gradient correctness."""
        # Create VJP and jitted versions
        n_params = len(model_config.list_params)
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
            param_name = model_config.list_params[i]
            np.testing.assert_array_almost_equal(
                grad_nojit,
                grad_jit,
                decimal=DECIMAL,
                err_msg=f"Gradient mismatch for parameter {i} ({param_name})",
            )

        # Test 3: Gradients are not all zeros
        for i, grad in enumerate(vjp_result):
            param_name = model_config.list_params[i]
            assert not np.allclose(grad, 0.0), (
                f"Gradient for {param_name} should not be all zeros"
            )

    def test_rl_logp_func_vjp_consistency(self, rldm_setup, model_config):
        """Test VJP consistency with JAX automatic differentiation."""
        # Create VJP function
        n_params = len(model_config.list_params)
        vjp_fn = make_vjp_func(rldm_setup.logp_fn, params_only=False, n_params=n_params)

        # Prepare JAX inputs
        data_values = rldm_setup.values.astype(np.float32)
        jax_args = tuple(jnp.asarray(arg, dtype=jnp.float32) for arg in rldm_setup.args)

        # Compute VJP using make_vjp_func
        logp_result = rldm_setup.logp_fn(data_values, *jax_args)
        gz = jnp.ones_like(logp_result)
        vjp_result = vjp_fn(data_values, *jax_args, gz=gz)

        def sum_logp_first_param(first_param_val):
            return rldm_setup.logp_fn(data_values, first_param_val, *jax_args[1:]).sum()

        grad_first_param_jax = jax.grad(sum_logp_first_param)(jax_args[0])

        # VJP result should match JAX's grad
        np.testing.assert_array_almost_equal(
            vjp_result[0],
            grad_first_param_jax,
            decimal=DECIMAL,
            err_msg=f"VJP gradient for {model_config.list_params[0]} doesn't match JAX's grad",
        )


class TestMultipleComputedParameters:
    """Test suite for models with multiple computed parameters."""

    def test_two_computed_parameters(self, rldm_data, model_config, param_arrays):
        """Test model with two computed parameters (v and a)."""

        # Create computation function for threshold 'a' from arousal
        def compute_a_subject_wise(subj_trials: jnp.ndarray) -> jnp.ndarray:
            """Compute threshold 'a' from arousal."""
            feedback = subj_trials[:, 0]
            mean_feedback = jnp.mean(feedback)
            return jnp.ones(len(feedback)) * (1.0 + 0.2 * mean_feedback)

        compute_v_annotated = annotate_function(
            inputs=["rl.alpha", "scaler", "response", "feedback"],
            outputs=["v"],
        )(compute_v_subject_wise)

        compute_a_annotated = annotate_function(
            inputs=["feedback"],
            outputs=["a"],
        )(compute_a_subject_wise)

        ssm_logp_func = annotate_function(
            inputs=["v", "a", "Z", "t", "theta", "rt", "response"],
            computed={"v": compute_v_annotated, "a": compute_a_annotated},
        )(angle_logp_jax_func)

        test_list_params = ["rl.alpha", "scaler", "Z", "t", "theta"]
        logp_fn = make_rl_logp_func(
            ssm_logp_func,
            n_participants=rldm_data.n_participants,
            n_trials=rldm_data.n_trials_per_participant,
            data_cols=model_config.data_cols,
            list_params=test_list_params,
            extra_fields=model_config.extra_fields,
        )

        # Prepare arguments (exclude 'a' from args since it's computed)
        param_key_map = {
            "rl.alpha": param_arrays.rl_alpha,
            "scaler": param_arrays.scaler,
            "Z": param_arrays.z,
            "t": param_arrays.t,
            "theta": param_arrays.theta,
        }
        args = [param_key_map[p] for p in test_list_params] + [param_arrays.feedback]

        result = logp_fn(rldm_data.data.values, *args)

        # Verify shape and successful computation
        assert result.shape[0] == rldm_data.total_trials
        assert not np.isnan(result).any(), "Result contains NaN values"
        assert not np.isinf(result).any(), "Result contains infinite values"

    def test_parameter_ordering(self, rldm_data, model_config, param_arrays):
        """Test that computed and non-computed parameters are ordered correctly."""
        compute_v_annotated = annotate_function(
            inputs=["rl.alpha", "scaler", "response", "feedback"],
            outputs=["v"],
        )(compute_v_subject_wise)

        # Test with computed param NOT first in the input list
        ssm_logp_func = annotate_function(
            inputs=["a", "v", "Z", "t", "theta", "rt", "response"],
            computed={"v": compute_v_annotated},
        )(angle_logp_jax_func)

        logp_fn = make_rl_logp_func(
            ssm_logp_func,
            n_participants=rldm_data.n_participants,
            n_trials=rldm_data.n_trials_per_participant,
            data_cols=model_config.data_cols,
            list_params=model_config.list_params,
            extra_fields=model_config.extra_fields,
        )

        param_key_map = {
            "rl.alpha": param_arrays.rl_alpha,
            "scaler": param_arrays.scaler,
            "a": param_arrays.a,
            "Z": param_arrays.z,
            "t": param_arrays.t,
            "theta": param_arrays.theta,
        }
        args = [param_key_map[p] for p in model_config.list_params] + [
            param_arrays.feedback
        ]

        result = logp_fn(rldm_data.data.values, *args)

        # Should still compute successfully despite different parameter order
        assert result.shape[0] == rldm_data.total_trials
        assert not np.isnan(result).any()

    def test_computed_params_different_inputs(
        self, rldm_data, model_config, param_arrays
    ):
        """Test multiple computed params with different input requirements."""

        def compute_v_annotated_func(subj_trials: jnp.ndarray) -> jnp.ndarray:
            return compute_v_subject_wise(subj_trials)

        def compute_z_from_bias(subj_trials: jnp.ndarray) -> jnp.ndarray:
            """Compute starting point z from feedback."""
            feedback = subj_trials[:, 0]
            return 0.5 + 0.1 * (feedback - 0.5)

        compute_v_annotated = annotate_function(
            inputs=["rl.alpha", "scaler", "response", "feedback"],
            outputs=["v"],
        )(compute_v_annotated_func)

        compute_z_annotated = annotate_function(
            inputs=["feedback"],  # Different input set than v
            outputs=["Z"],
        )(compute_z_from_bias)

        ssm_logp_func = annotate_function(
            inputs=["v", "a", "Z", "t", "theta", "rt", "response"],
            computed={"v": compute_v_annotated, "Z": compute_z_annotated},
        )(angle_logp_jax_func)

        # Exclude 'Z' from list_params since it's computed
        test_list_params = ["rl.alpha", "scaler", "a", "t", "theta"]
        logp_fn = make_rl_logp_func(
            ssm_logp_func,
            n_participants=rldm_data.n_participants,
            n_trials=rldm_data.n_trials_per_participant,
            data_cols=model_config.data_cols,
            list_params=test_list_params,
            extra_fields=model_config.extra_fields,
        )
        param_key_map = {
            "rl.alpha": param_arrays.rl_alpha,
            "scaler": param_arrays.scaler,
            "a": param_arrays.a,
            "t": param_arrays.t,
            "theta": param_arrays.theta,
        }
        args = [param_key_map[p] for p in test_list_params] + [param_arrays.feedback]

        result = logp_fn(rldm_data.data.values, *args)

        assert result.shape[0] == rldm_data.total_trials
        assert not np.isnan(result).any()

    def test_no_computed_parameters(self, rldm_data, model_config, param_arrays):
        """Test edge case with no computed parameters."""
        ssm_logp_func = annotate_function(
            inputs=["v", "a", "Z", "t", "theta", "rt", "response"],
            computed={},
        )(angle_logp_jax_func)

        logp_fn = make_rl_logp_func(
            ssm_logp_func,
            n_participants=rldm_data.n_participants,
            n_trials=rldm_data.n_trials_per_participant,
            data_cols=model_config.data_cols,
            list_params=["v", "a", "Z", "t", "theta"],
            extra_fields=model_config.extra_fields,
        )

        # Provide explicit v values instead of computing them
        v_values = np.ones(rldm_data.total_trials) * 0.5
        args = [
            v_values,
            param_arrays.a,
            param_arrays.z,
            param_arrays.t,
            param_arrays.theta,
            param_arrays.feedback,
        ]

        result = logp_fn(rldm_data.data.values, *args)

        assert result.shape[0] == rldm_data.total_trials
        assert not np.isnan(result).any()

    def test_all_parameters_computed(self, rldm_data, model_config, param_arrays):
        """Test edge case where all SSM parameters are computed.

        This verifies the system can handle scenarios where no parameters
        come directly from args (except for the RL/computation inputs).
        """
        # Create computation functions for all SSM parameters
        compute_v_annotated = annotate_function(
            inputs=["rl.alpha", "scaler", "response", "feedback"],
            outputs=["v"],
        )(compute_v_subject_wise)

        def make_constant_computer(value):
            """Create a function that computes constant parameter values."""

            def compute_func(subj_trials):
                return jnp.ones(len(subj_trials)) * value

            return compute_func

        compute_a_annotated = annotate_function(inputs=["feedback"], outputs=["a"])(
            make_constant_computer(1.2)
        )

        compute_z_annotated = annotate_function(inputs=["feedback"], outputs=["Z"])(
            make_constant_computer(0.5)
        )

        compute_t_annotated = annotate_function(inputs=["feedback"], outputs=["t"])(
            make_constant_computer(0.1)
        )

        compute_theta_annotated = annotate_function(
            inputs=["feedback"], outputs=["theta"]
        )(make_constant_computer(0.3))

        ssm_logp_func = annotate_function(
            inputs=["v", "a", "Z", "t", "theta", "rt", "response"],
            computed={
                "v": compute_v_annotated,
                "a": compute_a_annotated,
                "Z": compute_z_annotated,
                "t": compute_t_annotated,
                "theta": compute_theta_annotated,
            },
        )(angle_logp_jax_func)

        # Only RL parameters in list_params, all SSM params are computed
        test_list_params = ["rl.alpha", "scaler"]
        logp_fn = make_rl_logp_func(
            ssm_logp_func,
            n_participants=rldm_data.n_participants,
            n_trials=rldm_data.n_trials_per_participant,
            data_cols=model_config.data_cols,
            list_params=test_list_params,
            extra_fields=model_config.extra_fields,
        )

        args = [
            param_arrays.rl_alpha,
            param_arrays.scaler,
            param_arrays.feedback,
        ]

        result = logp_fn(rldm_data.data.values, *args)

        assert result.shape[0] == rldm_data.total_trials
        assert not np.isnan(result).any()

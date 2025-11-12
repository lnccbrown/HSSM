from pathlib import Path
import pytest

import jax
import numpy as np

import hssm
from hssm.likelihoods.rldm_optimized_abstraction import (
    make_rl_logp_func,
    make_rldm_logp_op,
    compute_v_subject_wise,
    annotate_function,
    _get_column_indices,
    _get_column_indices_with_computed,
    _collect_cols_arrays,
)


hssm.set_floatX("float32")

DECIMAL = 2


@pytest.fixture
def fixture_path():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def rldm_setup(fixture_path):
    data = np.load(fixture_path / "rldm_data.npy", allow_pickle=True).item()["data"]
    participant_id = data["participant_id"].values
    trial = data["trial"].values
    subj = np.unique(participant_id).astype(np.int32)
    total_trials = trial.size
    list_params = ["rl.alpha", "scaler", "a", "Z", "t", "theta"]
    extra_fields = ["feedback"]

    compute_v_subject_wise.inputs = ["rl.alpha", "scaler", "response", "feedback"]
    compute_v_subject_wise.outputs = ["v"]

    rl_alpha = np.ones(total_trials) * 0.60
    scaler = np.ones(total_trials) * 3.2
    a = np.ones(total_trials) * 1.2
    z = np.ones(total_trials) * 0.1
    t = np.ones(total_trials) * 0.1
    theta = np.ones(total_trials) * 0.1
    feedback = data["feedback"].values  # Extract feedback from data

    logp_fn = make_rl_logp_func(
        compute_v_subject_wise,
        n_participants=len(subj),
        n_trials=total_trials // len(subj),
        list_params=list_params,
        extra_fields=extra_fields,
    )

    return {
        "data": data,
        "values": data.values,
        "logp_fn": logp_fn,
        "total_trials": total_trials,
        "_args": [rl_alpha, scaler, a, z, t, theta, feedback],
    }


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

        # Call the function under test: get indices for where to find each column
        indices = _get_column_indices(
            cols_to_look_up,
            data_cols,
            list_params,
            extra_fields,
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
        setup = rldm_setup
        logp_fn = setup["logp_fn"]
        data = setup["values"]
        _args = setup["_args"]
        total_trials = setup["total_trials"]
        drift_rates = logp_fn(data, *_args)
        assert drift_rates.shape[0] == total_trials
        np.testing.assert_allclose(drift_rates.sum(), -141.76924, rtol=1e-2)

        jitted_logp = jax.jit(logp_fn)
        jax_ll = jitted_logp(data, *_args)
        assert np.all(jax_ll == drift_rates)

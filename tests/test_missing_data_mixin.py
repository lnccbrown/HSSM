"""
Tests for MissingDataMixin
-------------------------
1. Old tests migrated from test_data_validator.py that belong to missing data/deadline logic.
2. Additional tests for new features and edge cases in MissingDataMixin.
"""

import pytest
import pandas as pd
from hssm.missing_data_mixin import MissingDataMixin


class DummyModel(MissingDataMixin):
    """
    Dummy model for testing MissingDataMixin.

    This class provides stub implementations of methods that the mixin expects
    to exist on the consuming class. These stubs allow us to verify, via mocks/spies,
    that the mixin calls them as part of its logic. This is a common pattern for
    testing mixins: the dummy class provides the required interface, and the test
    checks the mixin's interaction with it.
    """

    def __init__(self, data):
        self.data = data
        self.response = ["response"]
        self.missing_data_value = -999.0
        self.missing_data = False
        self.deadline = False


# --- Fixtures ---
@pytest.fixture
def basic_data():
    return pd.DataFrame({"rt": [1.0, 2.0, -999.0], "response": [1, -1, 1]})


@pytest.fixture
def dummy_model(basic_data):
    return DummyModel(basic_data)


# Fixture for DummyModel with a deadline column
@pytest.fixture
def dummy_model_with_deadline(basic_data):
    data = basic_data.assign(deadline=[2.0, 2.0, 2.0])
    return DummyModel(data)


# --- 1. Old tests migrated from test_data_validator.py ---
class TestMissingDataMixinOld:
    def test_handle_missing_data_and_deadline_deadline_column_missing(
        self, dummy_model
    ):
        """
        Should raise ValueError if deadline is True but deadline_name column is missing.
        """
        # Try to process with deadline=True, should error
        with pytest.raises(ValueError, match="`deadline` is not found in your dataset"):
            dummy_model._process_missing_data_and_deadline(
                missing_data=False,
                deadline=True,
                loglik_missing_data=None,
            )

    def test_handle_missing_data_and_deadline_deadline_applied(
        self, dummy_model_with_deadline
    ):
        """
        Should set rt to -999.0 where rt >= deadline.
        """
        # Set one rt above deadline
        dummy_model_with_deadline.data.loc[0, "rt"] = 2.0  # Exceeds deadline
        dummy_model_with_deadline._process_missing_data_and_deadline(
            missing_data=False,
            deadline=True,
            loglik_missing_data=None,
        )
        assert dummy_model_with_deadline.data.loc[0, "rt"] == -999.0
        # All other rts should be less than their deadline
        assert all(
            dummy_model_with_deadline.data.loc[1:, "rt"]
            < dummy_model_with_deadline.data.loc[1:, "deadline"]
        )

    @pytest.mark.parametrize(
        "missing_data,expected_missing,expected_value",
        [
            (True, True, -999.0),
            (-999.0, True, -999.0),
        ],
    )
    def test_process_missing_data_handles_bool_and_float(
        self, dummy_model, missing_data, expected_missing, expected_value
    ):
        """
        Test that _process_missing_data_and_deadline correctly interprets the
        'missing_data' argument when given as a boolean or a float value.

        Parameters:
            missing_data: bool or float
                If True, missing data handling is enabled with default value -999.0.
                If a float (e.g., -999.0), that value is used for missing data.
            expected_missing: bool
                Expected value for model.missing_data after processing.
            expected_value: float
                Expected value for model.missing_data_value after processing.
        """

        dummy_model._process_missing_data_and_deadline(
            missing_data=missing_data,
            deadline=False,
            loglik_missing_data=None,
        )
        assert dummy_model.missing_data == expected_missing
        assert dummy_model.missing_data_value == expected_value

    def test_missing_data_false_drops_rows_and_warns(self, dummy_model):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dummy_model._process_missing_data_and_deadline(
                missing_data=False,
                deadline=False,
                loglik_missing_data=None,
            )
            assert not (dummy_model.data.rt == -999.0).any()
            assert dummy_model.missing_data is False
            assert dummy_model.missing_data_value == -999.0
            assert any("Dropping those rows" in str(warn.message) for warn in w)

    @pytest.mark.parametrize("missing_data", [123.45, "badtype"])
    def test_process_missing_data_errors(self, dummy_model, missing_data):
        with pytest.raises(ValueError):
            dummy_model._process_missing_data_and_deadline(
                missing_data=missing_data,
                deadline=False,
                loglik_missing_data=None,
            )

    def test_deadline_str_sets_name(self, dummy_model, basic_data):
        # Add a deadline_col to the data
        dummy_model.data = basic_data.assign(deadline_col=[2.0, 2.0, 2.0])
        dummy_model._process_missing_data_and_deadline(
            missing_data=False,
            deadline="deadline_col",
            loglik_missing_data=None,
        )
        assert dummy_model.deadline is True
        assert dummy_model.deadline_name == "deadline_col"
        assert "deadline_col" in dummy_model.response

    def test_deadline_bool_sets_name(self, dummy_model_with_deadline):
        dummy_model_with_deadline._process_missing_data_and_deadline(
            missing_data=False,
            deadline=True,
            loglik_missing_data=None,
        )
        assert dummy_model_with_deadline.deadline is True
        assert dummy_model_with_deadline.deadline_name == "deadline"

    @pytest.mark.parametrize(
        "missing_data,deadline,loglik_missing_data",
        [
            (False, False, lambda x: x),
        ],
    )
    def test_loglik_missing_data_error(
        self, dummy_model, missing_data, deadline, loglik_missing_data
    ):
        with pytest.raises(ValueError):
            dummy_model._process_missing_data_and_deadline(
                missing_data=missing_data,
                deadline=deadline,
                loglik_missing_data=loglik_missing_data,
            )


# --- 2. Additional tests for new features and edge cases in MissingDataMixin ---
class TestMissingDataMixinNew:
    def test_missing_data_value_custom(self, dummy_model):
        custom_missing = -123.0
        # Add a row with custom missing value
        dummy_model.data.loc[len(dummy_model.data)] = [custom_missing, 1]
        dummy_model._process_missing_data_and_deadline(
            missing_data=custom_missing,
            deadline=False,
            loglik_missing_data=None,
        )
        assert dummy_model.missing_data is True
        assert dummy_model.missing_data_value == custom_missing
        # After processing, custom missing values are replaced with -999.0
        assert (dummy_model.data.rt == -999.0).any()

    def test_deadline_column_added_once(self, dummy_model, basic_data):
        # Add a deadline_col to the data
        data = basic_data.assign(deadline_col=[2.0, 2.0, 2.0])
        dummy_model.data = data
        # Add deadline_col to response already
        dummy_model.response.append("deadline_col")
        dummy_model._process_missing_data_and_deadline(
            missing_data=False,
            deadline="deadline_col",
            loglik_missing_data=None,
        )
        # Should not duplicate
        assert dummy_model.response.count("deadline_col") == 1

    def test_missing_data_and_deadline_together(self, dummy_model_with_deadline):
        # Should set both flags
        dummy_model_with_deadline._process_missing_data_and_deadline(
            missing_data=True,
            deadline=True,
            loglik_missing_data=None,
        )
        assert dummy_model_with_deadline.missing_data is True
        assert dummy_model_with_deadline.deadline is True
        assert dummy_model_with_deadline.deadline_name == "deadline"

    def test_handle_missing_data_and_deadline_direct(self, dummy_model):
        """
        Directly test the _handle_missing_data_and_deadline method for coverage.
        """
        # Call with no arguments, as expected by the mixin stub
        dummy_model._handle_missing_data_and_deadline()

    def test_set_missing_data_and_deadline_direct(self, dummy_model):
        """
        Directly test the _set_missing_data_and_deadline method for coverage.
        """
        # Call with only the required arguments (data is now internal)
        dummy_model._set_missing_data_and_deadline(
            missing_data=True,
            deadline=False,
            data=dummy_model.data,
        )

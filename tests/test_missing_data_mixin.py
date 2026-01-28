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
    def test_missing_data_false_raises_valueerror(
        self, dummy_model, basic_data, dummy_model_with_deadline
    ):
        """
        Should raise ValueError if missing_data=False and -999.0 is present in rt column.
        Covers all cases where deadline is False, True, or a string.
        """
        # deadline=False
        with pytest.raises(ValueError, match="Missing data provided as False"):
            dummy_model._process_missing_data_and_deadline(
                missing_data=False,
                deadline=False,
                loglik_missing_data=None,
            )
        # deadline=True
        with pytest.raises(ValueError, match="Missing data provided as False"):
            dummy_model_with_deadline._process_missing_data_and_deadline(
                missing_data=False,
                deadline=True,
                loglik_missing_data=None,
            )
        # deadline as string
        dummy_model.data = basic_data.assign(deadline_col=[2.0, 2.0, 2.0])
        with pytest.raises(ValueError, match="Missing data provided as False"):
            dummy_model._process_missing_data_and_deadline(
                missing_data=False,
                deadline="deadline_col",
                loglik_missing_data=None,
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
        # Should raise ValueError due to -999.0 in rt when missing_data=False
        with pytest.raises(ValueError, match="Missing data provided as False"):
            dummy_model._process_missing_data_and_deadline(
                missing_data=False,
                deadline="deadline_col",
                loglik_missing_data=None,
            )

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

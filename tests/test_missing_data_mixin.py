"""
Tests for MissingDataMixin
-------------------------
1. Old tests migrated from test_data_validator.py that belong to missing data/deadline logic.
2. Additional tests for new features and edge cases in MissingDataMixin.
"""

import pytest
import pandas as pd

from hssm.missing_data_mixin import MissingDataMixin
from hssm.defaults import MissingDataNetwork


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
    def test_missing_data_network_set(self, dummy_model):
        # missing_data True, deadline False
        dummy_model._process_missing_data_and_deadline(
            missing_data=True, deadline=False, loglik_missing_data=None
        )
        assert dummy_model.missing_data_network == MissingDataNetwork.CPN

        # missing_data True, deadline True
        dummy_model.data["deadline"] = 2.0
        dummy_model._process_missing_data_and_deadline(
            missing_data=True, deadline=True, loglik_missing_data=None
        )
        assert dummy_model.missing_data_network == MissingDataNetwork.OPN

        # missing_data False, deadline False (should raise ValueError due to -999.0 present)
        with pytest.raises(ValueError, match="Missing data provided as False"):
            dummy_model._process_missing_data_and_deadline(
                missing_data=False, deadline=False, loglik_missing_data=None
            )

    def test_response_appended_with_deadline_name(self, dummy_model):
        # Should append deadline_name to response if not present
        dummy_model.data["deadline"] = 2.0
        dummy_model.response = ["response"]
        dummy_model._process_missing_data_and_deadline(
            missing_data=True, deadline="deadline", loglik_missing_data=None
        )
        assert "deadline" in dummy_model.response

    def test_data_mutation_missing_data_false(self, dummy_model):
        # Should drop rows with -999.0 if missing_data is False
        n_before = len(dummy_model.data)
        dummy_model._process_missing_data_and_deadline(
            missing_data=False, deadline=False, loglik_missing_data=None
        )
        n_after = len(dummy_model.data)
        assert n_after < n_before
        assert not (-999.0 in dummy_model.data.rt.values)

    def test_data_mutation_missing_data_true(self, dummy_model):
        # Should replace -999.0 with -999.0 (idempotent) if missing_data is True
        dummy_model._process_missing_data_and_deadline(
            missing_data=True, deadline=False, loglik_missing_data=None
        )
        assert -999.0 in dummy_model.data.rt.values

    def test_data_mutation_deadline(self, dummy_model):
        # Should set rt to -999.0 if above deadline
        # Set up so that the second RT is above its deadline
        dummy_model.data["rt"] = [1.0, 3.0, -999.0]  # 3.0 > 2.5
        dummy_model.data["deadline"] = [1.5, 2.5, 2.5]
        dummy_model._process_missing_data_and_deadline(
            missing_data=True, deadline="deadline", loglik_missing_data=None
        )
        # The first row rt=1.0 < 1.5, so not -999.0; second should be -999.0; third is already -999.0
        assert dummy_model.data.rt.iloc[0] == 1.0
        assert dummy_model.data.rt.iloc[1] == -999.0
        assert dummy_model.data.rt.iloc[2] == -999.0

    def test_loglik_missing_data_error(self, dummy_model):
        # Should raise if loglik_missing_data is set but both missing_data and deadline are False
        dummy_model.data.rt = [1.0, 2.0, 3.0]  # No -999.0 present
        with pytest.raises(
            ValueError,
            match="loglik_missing_data function, but you have not set the missing_data or deadline flag to True",
        ):
            dummy_model._process_missing_data_and_deadline(
                missing_data=False, deadline=False, loglik_missing_data=lambda x: x
            )

    def test_process_missing_data_and_deadline_updates_attributes(self, dummy_model):
        """
        Test that _process_missing_data_and_deadline updates missing_data, deadline, deadline_name, and loglik_missing_data.
        """

        # Set up a custom loglik function
        def custom_loglik(x):
            return x

        # Add a custom_deadline column to the data to satisfy the mixin's requirements
        dummy_model.data["custom_deadline"] = 2.0
        # Call with missing_data True, deadline as string, and custom loglik
        dummy_model._process_missing_data_and_deadline(
            missing_data=True,
            deadline="custom_deadline",
            loglik_missing_data=custom_loglik,
        )
        assert dummy_model.missing_data is True
        assert dummy_model.deadline is True
        assert dummy_model.deadline_name == "custom_deadline"
        assert dummy_model.loglik_missing_data is custom_loglik

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

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


# --- Fixtures ---
@pytest.fixture
def basic_data():
    return pd.DataFrame({"rt": [1.0, 2.0, -999.0], "response": [1, -1, 1]})


# --- 1. Old tests migrated from test_data_validator.py ---
class TestMissingDataMixinOld:
    def test_handle_missing_data_and_deadline_deadline_column_missing(self, basic_data):
        """
        Should raise ValueError if deadline is True but deadline_name column is missing.
        """
        model = DummyModel(basic_data)
        # Try to process with deadline=True, should error
        with pytest.raises(ValueError, match="`deadline` is not found in your dataset"):
            model._process_missing_data_and_deadline(
                missing_data=False,
                deadline=True,
                loglik_missing_data=None,
            )

    def test_handle_missing_data_and_deadline_deadline_applied(self, basic_data):
        """
        Should set rt to -999.0 where rt >= deadline.
        """
        # Add a deadline column and set one rt above deadline
        basic_data = basic_data.assign(deadline=[1.5, 2.0, 2.0])
        basic_data.loc[0, "rt"] = 2.0  # Exceeds deadline
        model = DummyModel(basic_data)
        model._process_missing_data_and_deadline(
            missing_data=False,
            deadline=True,
            loglik_missing_data=None,
        )
        assert model.data.loc[0, "rt"] == -999.0
        # All other rts should be less than their deadline
        assert all(model.data.loc[1:, "rt"] < model.data.loc[1:, "deadline"])

    @pytest.mark.parametrize(
        "missing_data,expected_missing,expected_value",
        [
            (True, True, -999.0),
            (-999.0, True, -999.0),
        ],
    )
    def test_process_missing_data_handles_bool_and_float(
        self, basic_data, missing_data, expected_missing, expected_value
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

        model = DummyModel(basic_data)
        model._process_missing_data_and_deadline(
            missing_data=missing_data,
            deadline=False,
            loglik_missing_data=None,
        )
        assert model.missing_data == expected_missing
        assert model.missing_data_value == expected_value

    def test_missing_data_false_drops_rows_and_warns(self, basic_data):
        import warnings

        model = DummyModel(basic_data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model._process_missing_data_and_deadline(
                missing_data=False,
                deadline=False,
                loglik_missing_data=None,
            )
            assert not (model.data.rt == -999.0).any()
            assert model.missing_data is False
            assert model.missing_data_value == -999.0
            assert any("Dropping those rows" in str(warn.message) for warn in w)

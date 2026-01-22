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


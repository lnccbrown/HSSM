import pytest
import pandas as pd
import numpy as np
from hssm.data_validator import DataValidator


# @pytest.fixture
def base_data():
    return pd.DataFrame(
        {
            "rt": [0.5, 0.7, 0.9, 1.1],
            "response": [1, 0, 1, 0],
            "deadline": [1.0, 1.0, 1.0, 1.0],
            "extra": [10, 20, 30, 40],
        }
    )


# @pytest.fixture
def base_data_with_missing():
    return pd.DataFrame(
        {
            "rt": [0.5, 0.7, -999.0, 1.1],
            "response": [1, 0, -999.0, 0],
            "deadline": [1.0, 1.0, 1.0, 1.0],
            "extra": [10, 20, -999.0, 40],
        }
    )


def base_data_nan_missing():
    return pd.DataFrame(
        {
            "rt": [-999.0, 0.7, np.nan, 1.1],
            "response": [1, 0, np.nan, 0],
            "deadline": [1.0, 1.0, 1.0, 1.0],
            "extra": [10, 20, np.nan, 40],
        }
    )


# @pytest.fixture
def dv_instance(data_factory: pd.DataFrame) -> DataValidator:
    return DataValidator(
        data=data_factory(),
        response=["rt", "response"],
        choices=[0, 1],
        n_choices=2,
        extra_fields=["extra"],
        deadline=True,
        deadline_name="deadline",
        missing_data=False,
        missing_data_value=-999.0,
    )


def test_constructor():
    data = base_data()
    dv = dv_instance(base_data)

    assert isinstance(dv, DataValidator)
    assert dv.data.equals(data)
    assert dv.response == ["rt", "response"]
    assert dv.choices == [0, 1]
    assert dv.n_choices == 2
    assert dv.extra_fields == ["extra"]
    assert dv.deadline is True
    assert dv.deadline_name == "deadline"
    assert dv.missing_data is False
    assert dv.missing_data_value == -999.0


def test_check_extra_fields():
    dv = dv_instance(base_data)
    # Should not raise an exception
    assert dv._check_extra_fields()

    # Test with missing extra field
    dv.extra_fields = ["missing_field", "foo", "bar"]
    with pytest.raises(ValueError, match="Field.* not found in data."):
        dv._check_extra_fields()


def test_pre_check_data_sanity():
    dv = dv_instance(base_data)
    dv._pre_check_data_sanity()  # Should not raise any exceptions


def test_post_check_data_sanity_valid():
    dv = dv_instance(base_data_with_missing)
    dv._post_check_data_sanity()  # Should not raise any exceptions

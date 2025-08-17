import random
from typing import Callable

import pytest
import pandas as pd
import numpy as np
from hssm.data_validator import DataValidator
from hssm.defaults import MissingDataNetwork


def _base_data():
    return pd.DataFrame(
        {
            "rt": [0.5, 0.7, 0.9, 1.1],
            "response": [1, 0, 1, 0],
            "deadline": [1.0, 1.0, 1.0, 1.0],
            "extra": [10, 20, 30, 40],
        }
    )


@pytest.fixture
def base_data():
    return _base_data()


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


def dv_instance(
    data_factory: Callable = _base_data, deadline: bool = True
) -> DataValidator:
    return DataValidator(
        data=data_factory(),
        extra_fields=["extra"],
        deadline=deadline,
    )


def test_constructor(base_data):
    dv = DataValidator(
        data=base_data,
        extra_fields=["extra"],
        deadline=True,
    )

    assert isinstance(dv, DataValidator)
    assert dv.data.equals(_base_data())
    assert dv.response == ["rt", "response"]
    assert dv.choices == [0, 1]
    assert dv.n_choices == 2
    assert dv.extra_fields == ["extra"]
    assert dv.deadline is True
    assert dv.deadline_name == "deadline"
    assert dv.missing_data is False
    assert dv.missing_data_value == -999.0


def test_check_extra_fields():
    dv = dv_instance()
    # Should not raise an exception
    assert dv._check_extra_fields()

    # Test with missing extra field
    dv.extra_fields = ["missing_field", "foo", "bar"]
    with pytest.raises(ValueError, match="Field.* not found in data."):
        dv._check_extra_fields()


def test_pre_check_data_sanity():
    dv_instance()._pre_check_data_sanity()  # Should not raise any exceptions


def test_post_check_data_sanity_valid(base_data):
    dv = dv_instance(base_data_with_missing)
    dv._post_check_data_sanity()  # Should not raise any exceptions

    dv_instance_no_missing = dv_instance()
    with pytest.raises(ValueError, match="You have no missing data in your dataset"):
        dv_instance_no_missing._post_check_data_sanity()

    dv_instance_nan = dv_instance(base_data_nan_missing)
    with pytest.raises(ValueError, match="You have NaN response times in your dataset"):
        dv_instance_nan._post_check_data_sanity()

    dv_instance_no_missing.data = dv_instance_no_missing.data * -1
    dv_instance_no_missing.deadline = False
    with pytest.raises(
        ValueError, match="You have negative response times in your dataset"
    ):
        dv_instance_no_missing._post_check_data_sanity()

    dv_instance_no_missing = DataValidator(
        data=base_data,
        deadline=False,
        missing_data=False,
        choices=[0, 1, 2],
        n_choices=3,
    )

    invalid_response = random.choice(range(2, 100))
    dv_instance_no_missing.data.iloc[0, 1] = invalid_response
    with pytest.raises(ValueError, match=f"Invalid responses found in your dataset: "):
        dv_instance_no_missing._post_check_data_sanity()

    dv_instance_no_missing.data.iloc[0, 1] = 1  # Reset to valid response
    with pytest.warns(
        UserWarning,
        match=(r"missing from your dataset"),
    ):
        dv_instance_no_missing._post_check_data_sanity()


def test_handle_missing_data_and_deadline_deadline_column_missing(base_data):
    # Should raise ValueError if deadline is True but deadline_name column is missing
    data = base_data.drop(columns=["deadline"])
    dv = DataValidator(
        data=data,
        deadline=True,
    )
    with pytest.raises(ValueError, match="`deadline` is not found in your dataset"):
        dv._handle_missing_data_and_deadline()


def test_handle_missing_data_and_deadline_deadline_applied(base_data):
    # Should set rt to -999.0 where rt >= deadline
    base_data.loc[0, "rt"] = 2.0  # Exceeds deadline
    dv = DataValidator(
        data=base_data,
        deadline=True,
    )
    dv._handle_missing_data_and_deadline()
    assert dv.data.loc[0, "rt"] == -999.0
    assert all(dv.data.loc[1:, "rt"] < dv.data.loc[1:, "deadline"])


def test_update_extra_fields(monkeypatch):
    # Create a DataValidator with extra_fields
    data = pd.DataFrame(
        {
            "rt": [0.5, 0.7],
            "response": [1, 0],
            "deadline": [1.0, 1.0],
            "extra": [10, 20],
            "extra2": [100, 200],
        }
    )
    dv = DataValidator(
        data=data,
        extra_fields=["extra", "extra2"],
    )

    # Mock the model_distribution attribute
    class DummyModelDist:
        pass

    dv.model_distribution = DummyModelDist()

    # Call the method
    dv._update_extra_fields()

    # Check that extra_fields were updated correctly
    assert len(dv.model_distribution.extra_fields) == 2
    assert (dv.model_distribution.extra_fields[0] == data["extra"].values).all()
    for i, field in enumerate(dv.extra_fields):
        assert (dv.model_distribution.extra_fields[i] == data[field].values).all()


def test_set_missing_data_and_deadline():
    # No missing data and no deadline
    data = pd.DataFrame({"rt": [0.5, 0.7]})
    assert (
        DataValidator._set_missing_data_and_deadline(False, False, data)
        == MissingDataNetwork.NONE
    )
    # Missing data but no deadline
    data = pd.DataFrame({"rt": [0.5, -999.0]})
    assert (
        DataValidator._set_missing_data_and_deadline(True, False, data)
        == MissingDataNetwork.CPN
    )
    assert (
        DataValidator._set_missing_data_and_deadline(True, True, data)
        == MissingDataNetwork.OPN
    )
    # AF-TODO: I think GONOGO as a network category can go,
    # but needs a little more thought, out of scope for PR,
    # during which this was commented out.
    # assert (
    #     DataValidator._set_missing_data_and_deadline(True, True, data)
    #     == MissingDataNetwork.GONOGO
    # )


def test_set_missing_data_and_deadline_all_missing():
    data = pd.DataFrame({"rt": [-999.0, -999.0]})
    # cpn
    with pytest.raises(
        ValueError,
        match="`missing_data` is set to True, but you have no valid data in your "
        "dataset.",
    ):
        DataValidator._set_missing_data_and_deadline(True, False, data)

    # opn
    with pytest.raises(
        ValueError,
        match="`missing_data` is set to True, but you have no valid data in your "
        "dataset.",
    ):
        DataValidator._set_missing_data_and_deadline(True, True, data)

    # AF-TODO: GONOGO case not yet correctly implemented
    # gonogo
    # data = pd.DataFrame({"rt": [-999.0, -999.0]})
    # with pytest.raises(
    #     ValueError,
    #     match="`missing_data` is set to True, but you have no valid data in your "
    #     + "dataset.",
    # ):
    #     DataValidator._set_missing_data_and_deadline(True, True, data)

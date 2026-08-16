from typing import Callable

import pytest
import pandas as pd
import numpy as np
from hssm.data_validator import DataValidatorMixin
from hssm.defaults import MissingDataNetwork


class DataValidatorTester(DataValidatorMixin):
    """A concrete class for testing DataValidatorMixin."""

    def __init__(
        self,
        data: pd.DataFrame,
        extra_fields: list[str] | None = None,
        deadline: bool = False,
        missing_data: bool = False,
        choices: list[int] | None = None,
        n_choices: int | None = None,
        response_kind: str = "categorical",
        response_bounds: dict[str, tuple[float, float]] | None = None,
    ):
        self.data = data
        self.response = ["rt", "response"]
        self.response_kind = response_kind
        self.response_bounds = response_bounds or {}
        self.choices = (
            [0, 1] if choices is None and response_kind == "categorical" else choices
        )
        self.n_choices = (
            n_choices
            if n_choices is not None
            else (len(self.choices) if self.choices is not None else None)
        )
        self.extra_fields = extra_fields
        self.deadline = deadline
        self.deadline_name = "deadline"
        self.missing_data = missing_data
        self.missing_data_value = -999.0
        self.is_choice_only = False


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
) -> DataValidatorTester:
    return DataValidatorTester(
        data=data_factory(),
        extra_fields=["extra"],
        deadline=deadline,
    )


def test_constructor(base_data):
    dv = DataValidatorTester(
        data=base_data,
        extra_fields=["extra"],
        deadline=True,
    )

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

    dv_instance_no_missing = DataValidatorTester(
        data=base_data,
        deadline=False,
        missing_data=False,
        choices=[0, 1, 2],
        n_choices=3,
    )

    invalid_response = max(dv_instance_no_missing.choices) + 1
    dv_instance_no_missing.data.loc[0, "response"] = invalid_response
    with pytest.raises(ValueError, match=f"Invalid responses found in your dataset: "):
        dv_instance_no_missing._post_check_data_sanity()

    dv_instance_no_missing.data.loc[0, "response"] = 1  # Reset to valid response
    with pytest.warns(
        UserWarning,
        match=(r"missing from your dataset"),
    ):
        dv_instance_no_missing._post_check_data_sanity()


def test_update_extra_fields(monkeypatch):
    # Create a DataValidatorTester with extra_fields
    data = pd.DataFrame(
        {
            "rt": [0.5, 0.7],
            "response": [1, 0],
            "deadline": [1.0, 1.0],
            "extra": [10, 20],
            "extra2": [100, 200],
        }
    )
    dv = DataValidatorTester(
        data=data,
        extra_fields=["extra", "extra2"],
    )

    # Mock the model_distribution attribute
    class DummyModelDist:
        extra_fields: list

    dv.model_distribution = DummyModelDist()  # type: ignore[assignment]

    # Call the method
    dv._update_extra_fields()

    # Check that extra_fields were updated correctly
    assert len(dv.model_distribution.extra_fields) == 2
    assert (dv.model_distribution.extra_fields[0] == data["extra"].values).all()
    for i, field in enumerate(dv.extra_fields):  # type: ignore[union-attr]
        assert (dv.model_distribution.extra_fields[i] == data[field].values).all()


def test_validate_choices():
    # ====== Valid choices =====
    dv = DataValidatorTester(
        data=_base_data(),
        choices=[0, 1],
        n_choices=2,
    )
    dv._validate_choices()  # Should not raise an exception

    # ===== Invalid choices =====
    dv.choices = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="`choices` must be provided*."):
        dv._validate_choices()

    circular_dv = DataValidatorTester(
        data=_base_data(),
        response_kind="circular",
        response_bounds={"response": (-np.pi, np.pi)},
    )
    circular_dv._validate_choices()
    assert circular_dv.choices is None
    assert circular_dv.n_choices is None


def test_circular_responses_remain_fractional():
    responses = np.array([-np.pi, -1.25, 0.125, np.nextafter(np.pi, -np.inf)])
    data = _base_data()
    data["response"] = responses
    original = data["response"].to_numpy(copy=True)
    dv = DataValidatorTester(
        data=data,
        deadline=False,
        response_kind="circular",
        response_bounds={"response": (-np.pi, np.pi)},
    )

    dv._post_check_data_sanity()

    np.testing.assert_array_equal(dv.data["response"].to_numpy(), original)


@pytest.mark.parametrize("invalid_response", [np.pi, np.nextafter(-np.pi, -np.inf)])
def test_circular_responses_enforce_half_open_bounds(invalid_response):
    data = _base_data()
    data["response"] = data["response"].astype(float)
    data.loc[0, "response"] = invalid_response
    dv = DataValidatorTester(
        data=data,
        deadline=False,
        response_kind="circular",
        response_bounds={"response": (-np.pi, np.pi)},
    )

    with pytest.raises(ValueError, match=r"must lie in \[-3.*3.*\)"):
        dv._post_check_data_sanity()


@pytest.mark.parametrize("invalid_response", [np.nan, np.inf, "not-a-number"])
def test_non_categorical_responses_must_be_numeric_and_finite(invalid_response):
    data = _base_data()
    data["response"] = data["response"].astype(object)
    data.loc[0, "response"] = invalid_response
    if invalid_response != "not-a-number":
        data["response"] = pd.to_numeric(data["response"])
    dv = DataValidatorTester(
        data=data,
        deadline=False,
        response_kind="continuous",
    )

    with pytest.raises(ValueError, match="must contain numeric, finite values"):
        dv._post_check_data_sanity()


def test_continuous_response_bounds_are_closed():
    data = _base_data()
    data["response"] = [-1.0, -0.25, 0.25, 1.0]
    dv = DataValidatorTester(
        data=data,
        deadline=False,
        response_kind="continuous",
        response_bounds={"response": (-1.0, 1.0)},
    )
    dv._post_check_data_sanity()

    dv.data.loc[0, "response"] = np.nextafter(-1.0, -np.inf)
    with pytest.raises(ValueError, match=r"must lie in \[-1.0, 1.0\]"):
        dv._post_check_data_sanity()


def test_circular_responses_preserve_deadline_and_missing_row_handling():
    data = base_data_with_missing()
    data["response"] = [-2.5, 0.125, -999.0, 2.25]
    dv = DataValidatorTester(
        data=data,
        deadline=True,
        missing_data=True,
        response_kind="circular",
        response_bounds={"response": (-np.pi, np.pi)},
    )
    dv.response.append("deadline")

    dv._post_check_data_sanity()

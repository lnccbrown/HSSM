"""Tests for response-data validation."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from hssm._types import ResponseDomainSpec
from hssm.data_validator import DataValidatorMixin


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
        response: list[str] | None = None,
        response_domains: dict[str, ResponseDomainSpec] | None = None,
        is_choice_only: bool = False,
    ):
        self.data = data
        self.response = response or ["rt", "response"]
        if response_domains is None:
            self.choices = tuple(choices) if choices is not None else (0, 1)
            self.response_domains = {
                self.response[-1]: {
                    "kind": "categorical",
                    "values": self.choices,
                }
            }
        else:
            self.response_domains = response_domains
            only_domain = (
                next(iter(response_domains.values()))
                if len(response_domains) == 1
                else None
            )
            self.choices = (
                tuple(only_domain["values"])
                if only_domain is not None and only_domain["kind"] == "categorical"
                else None
            )
        self.n_choices = (
            n_choices
            if n_choices is not None
            else len(self.choices)
            if self.choices is not None
            else None
        )
        self.extra_fields = extra_fields
        self.deadline = deadline
        self.deadline_name = "deadline"
        self.missing_data = missing_data
        self.missing_data_value = -999.0
        self.is_choice_only = is_choice_only


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
    assert dv.choices == (0, 1)
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


def test_mixed_domains_validate_in_physical_column_order_without_mutation():
    """Mixed scalar columns validate against their own domains without coercion."""
    data = pd.DataFrame(
        {
            "rt": [0.2, 0.3, 0.4],
            "polar": [0.0, 1.0, 2.0],
            "azimuth": [-np.pi, 0.0, np.nextafter(np.pi, -np.inf)],
        }
    )
    original = data.copy(deep=True)
    validator = DataValidatorTester(
        data,
        response=["rt", "polar", "azimuth"],
        response_domains={
            "polar": {"kind": "continuous", "bounds": (0.0, 2.0)},
            "azimuth": {"kind": "circular", "bounds": (-np.pi, np.pi)},
        },
    )

    validator._post_check_data_sanity()

    pd.testing.assert_frame_equal(data, original)


def test_four_observation_columns_validate_independently():
    """Each scalar coordinate in a wider observation has its own domain."""
    validator = DataValidatorTester(
        pd.DataFrame(
            {
                "rt": [0.2, 0.3],
                "confidence": [0.0, 1.0],
                "angle": [-np.pi, 0.0],
                "choice": [0.0, 1.0],
            }
        ),
        response=["rt", "confidence", "angle", "choice"],
        response_domains={
            "confidence": {"kind": "continuous", "bounds": (0.0, 1.0)},
            "angle": {"kind": "circular", "bounds": (-np.pi, np.pi)},
            "choice": {"kind": "categorical", "values": (0, 1)},
        },
    )

    validator._post_check_data_sanity()


def test_first_invalid_physical_column_follows_declared_order():
    """Simultaneous failures report the first configured response coordinate."""
    validator = DataValidatorTester(
        pd.DataFrame({"rt": [0.2], "first": [2.0], "second": [2.0]}),
        response=["rt", "first", "second"],
        response_domains={
            "first": {"kind": "continuous", "bounds": (0.0, 1.0)},
            "second": {"kind": "continuous", "bounds": (0.0, 1.0)},
        },
    )

    with pytest.raises(ValueError, match="column 'first'.*bounds"):
        validator._post_check_data_sanity()


@pytest.mark.parametrize(
    ("kind", "bounds", "value", "passes"),
    [
        ("continuous", (0.0, 1.0), 0.0, True),
        ("continuous", (0.0, 1.0), 1.0, True),
        ("continuous", (0.0, 1.0), np.nextafter(0.0, -np.inf), False),
        ("continuous", (0.0, 1.0), np.nextafter(1.0, np.inf), False),
        ("circular", (-np.pi, np.pi), -np.pi, True),
        ("circular", (-np.pi, np.pi), np.nextafter(np.pi, -np.inf), True),
        ("circular", (-np.pi, np.pi), np.pi, False),
        ("circular", (-np.pi, np.pi), np.nextafter(-np.pi, -np.inf), False),
    ],
)
def test_continuous_and_circular_endpoint_semantics(
    kind: str, bounds: tuple[float, float], value: float, passes: bool
):
    """Continuous bounds are closed while circular upper bounds are excluded."""
    validator = DataValidatorTester(
        pd.DataFrame({"rt": [0.2], "coordinate": [value]}),
        response=["rt", "coordinate"],
        response_domains={
            "coordinate": {"kind": kind, "bounds": bounds}  # type: ignore[typeddict-item]
        },
    )

    if passes:
        validator._post_check_data_sanity()
    else:
        with pytest.raises(ValueError, match="column 'coordinate'.*bounds"):
            validator._post_check_data_sanity()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "1", True])
def test_domains_reject_nonfinite_or_nonnumeric_values(value):
    """Every domain rejects nonfinite, nonnumeric, and Boolean observations."""
    validator = DataValidatorTester(
        pd.DataFrame({"rt": [0.2], "coordinate": [value]}),
        response=["rt", "coordinate"],
        response_domains={"coordinate": {"kind": "continuous"}},
    )

    with pytest.raises(ValueError, match="column 'coordinate'.*finite numeric"):
        validator._post_check_data_sanity()


def test_categorical_membership_does_not_integer_cast_fractional_values():
    """A fractional category cannot pass by truncation to an integer label."""
    validator = DataValidatorTester(
        pd.DataFrame({"rt": [0.2, 0.3], "response": [0.0, 0.5]}),
        response_domains={"response": {"kind": "categorical", "values": (0, 1)}},
    )

    with pytest.raises(ValueError, match=r"Invalid responses.*\[0\.5\]"):
        validator._post_check_data_sanity()


def test_categorical_membership_preserves_large_integer_precision():
    """Adjacent integer labels above float precision remain distinguishable."""
    allowed = 2**53
    validator = DataValidatorTester(
        pd.DataFrame({"rt": [0.2], "response": [allowed + 1]}),
        response_domains={"response": {"kind": "categorical", "values": (allowed,)}},
    )

    with pytest.raises(ValueError, match=str(allowed + 1)):
        validator._post_check_data_sanity()


def test_categorical_membership_supports_arbitrary_python_integers():
    """Categorical labels are not constrained to NumPy fixed-width integers."""
    allowed = 10**100
    validator = DataValidatorTester(
        pd.DataFrame(
            {
                "rt": [0.2],
                "response": pd.Series([allowed], dtype=object),
            }
        ),
        response_domains={"response": {"kind": "categorical", "values": (allowed,)}},
    )

    validator._post_check_data_sanity()

    validator.data.loc[0, "response"] = allowed + 1
    with pytest.raises(ValueError, match=str(allowed + 1)):
        validator._post_check_data_sanity()


def test_multidomain_categorical_failure_names_physical_response_column():
    """A legacy-named column is still identified in a wider response."""
    validator = DataValidatorTester(
        pd.DataFrame({"rt": [0.2], "response": [2], "confidence": [0.5]}),
        response=["rt", "response", "confidence"],
        response_domains={
            "response": {"kind": "categorical", "values": (0, 1)},
            "confidence": {"kind": "continuous", "bounds": (0, 1)},
        },
    )

    with pytest.raises(ValueError, match="column 'response'"):
        validator._post_check_data_sanity()


def test_missing_rt_rows_are_omitted_but_observed_rows_remain_validated():
    """Missing-data sentinels exempt only their own response row."""
    data = pd.DataFrame({"rt": [-999.0, 0.3], "coordinate": [99.0, 0.5]})
    validator = DataValidatorTester(
        data,
        missing_data=True,
        response=["rt", "coordinate"],
        response_domains={"coordinate": {"kind": "continuous", "bounds": (0.0, 1.0)}},
    )
    validator._post_check_data_sanity()

    validator.data.loc[1, "coordinate"] = 99.0
    with pytest.raises(ValueError, match="column 'coordinate'.*bounds"):
        validator._post_check_data_sanity()


def test_custom_missing_marker_uses_processed_internal_sentinel():
    """A custom missing marker remains omitted after preprocessing to -999."""
    validator = DataValidatorTester(
        pd.DataFrame({"rt": [-999.0, 0.3], "coordinate": [99.0, 0.5]}),
        missing_data=True,
        response=["rt", "coordinate"],
        response_domains={"coordinate": {"kind": "continuous", "bounds": (0, 1)}},
    )
    validator.missing_data_value = -123.0

    validator._post_check_data_sanity()


def test_choice_only_domain_uses_the_shared_validation_loop():
    """A one-column choice-only response is checked by the canonical path."""
    validator = DataValidatorTester(
        pd.DataFrame({"choice": [0.0, 1.0, 0.5]}),
        response=["choice"],
        response_domains={"choice": {"kind": "categorical", "values": (0, 1)}},
        is_choice_only=True,
    )

    with pytest.raises(ValueError, match=r"column 'choice'.*\[0\.5\]"):
        validator._post_check_data_sanity()

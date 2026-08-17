"""Data validation and preprocessing utilities for HSSM behavioral models."""

import logging
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

_logger = logging.getLogger("hssm")


class DataValidatorMixin:
    """Provide validation and preprocessing methods for HSSM behavioral models."""

    data: pd.DataFrame
    response: list[str]
    choices: list[int] | tuple[int, ...] | None
    response_kind: str
    response_bounds: dict[str, tuple[float, float]]
    n_choices: int | None
    extra_fields: list[str] | None
    deadline: bool
    deadline_name: str
    missing_data: bool
    missing_data_value: float
    is_choice_only: bool

    @staticmethod
    def check_fields(a, b):
        """Check if all fields in a are in b."""
        missing = set(a) - set(b)
        if missing:  # there are leftover fields
            raise ValueError(f"Field(s) `{', '.join(missing)}` not found in data.")

    def _check_extra_fields(self, data: pd.DataFrame | None = None) -> bool:
        """Check if every field in self.extra_fields exists in data."""
        if not self.extra_fields:
            return False

        data = data if data is not None else self.data

        DataValidatorMixin.check_fields(self.extra_fields, data.columns)

        return True

    def _pre_check_data_sanity(self):
        """Check if the data is clean enough for the model."""
        DataValidatorMixin.check_fields(self.response, self.data.columns)
        self._check_extra_fields()

    def _post_check_data_sanity(self):
        """Check if the data is clean enough for the model."""
        if self.is_choice_only:
            if self.response_kind == "categorical":
                return
            valid_data = self.data
        else:
            if self.deadline or self.missing_data:
                if -999.0 not in self.data["rt"].unique():
                    raise ValueError(
                        "You have no missing data in your dataset, "
                        + "which is not allowed when `missing_data` or `deadline` is "
                        + "set to True."
                    )
                rt_filtered = self.data.rt[self.data.rt != -999.0]
            else:
                rt_filtered = self.data.rt

            if np.any(rt_filtered.isna(), axis=None):
                raise ValueError(
                    "You have NaN response times in your dataset, "
                    + "which is not allowed."
                )

            if not np.all(rt_filtered >= 0):
                raise ValueError(
                    "You have negative response times in your dataset, "
                    + "which is not allowed."
                )

            valid_data = self.data.loc[self.data["rt"] != -999.0]

        if self.response_kind != "categorical":
            self._validate_non_categorical_responses(valid_data)
            return

        assert self.choices is not None
        assert self.n_choices is not None
        valid_responses = valid_data["response"]
        unique_responses = valid_responses.unique().astype(int)

        if np.any(~np.isin(unique_responses, self.choices)):
            invalid_responses = sorted(
                unique_responses[~np.isin(unique_responses, self.choices)].tolist()
            )
            raise ValueError(
                f"Invalid responses found in your dataset: {invalid_responses}"
            )

        if len(unique_responses) != self.n_choices:
            missing_responses = sorted(
                np.setdiff1d(self.choices, unique_responses).tolist()
            )
            warnings.warn(
                (
                    f"You set choices to be {self.choices}, but {missing_responses} "
                    "are missing from your dataset."
                ),
                UserWarning,
                stacklevel=2,
            )

    def _validate_non_categorical_responses(self, data: pd.DataFrame) -> None:
        """Validate continuous or circular response columns without coercion."""
        response_columns = list(self.response)
        if self.deadline and self.deadline_name in response_columns:
            response_columns.remove(self.deadline_name)
        if not self.is_choice_only:
            response_columns = response_columns[1:]

        for column in response_columns:
            values = data[column]
            if is_bool_dtype(values.dtype) or not is_numeric_dtype(values.dtype):
                raise ValueError(
                    f"Response column {column!r} must contain numeric, finite values."
                )

            values_array = values.to_numpy()
            if not np.all(np.isfinite(values_array)):
                raise ValueError(
                    f"Response column {column!r} must contain numeric, finite values."
                )

            if column not in self.response_bounds:
                continue

            lower, upper = self.response_bounds[column]
            upper_invalid = (
                values_array >= upper
                if self.response_kind == "circular"
                else values_array > upper
            )
            invalid = (values_array < lower) | upper_invalid
            if np.any(invalid):
                interval_end = ")" if self.response_kind == "circular" else "]"
                invalid_values = np.unique(values_array[invalid]).tolist()
                raise ValueError(
                    f"{self.response_kind.capitalize()} responses in column "
                    f"{column!r} must lie in [{lower}, {upper}{interval_end}. "
                    f"Invalid values: {invalid_values}"
                )

    # AF-TODO: We probably want to incorporate some of the
    # remaining check on missing data
    # which are coming AFTER the data validation
    # in the HSSM class, into this function?

    def _update_extra_fields(self, new_data: pd.DataFrame | None = None):
        """Update the extra fields data in self.model_distribution.

        Parameters
        ----------
        new_data
            A DataFrame containing new data for update.
        """
        if new_data is None:
            new_data = self.data

        # The attribute 'model_distribution' is not defined in
        # DataValidatorMixin itself, but is expected to exist in subclasses
        # (e.g., HSSM).
        # The 'type: ignore[attr-defined]' comment tells mypy to ignore the missing
        # attribute error here and avoid moving this method to the HSSM class.
        if self.extra_fields is not None:
            self.model_distribution.extra_fields = [  # type: ignore[attr-defined]
                new_data[field].values for field in self.extra_fields
            ]

    def _validate_choices(self):
        """Validate that categorical response metadata includes choices."""
        if self.response_kind == "categorical" and self.choices is None:
            raise ValueError(
                "`choices` must be provided either in `model_config` or as an argument."
            )
        if self.response_kind != "categorical" and self.choices is not None:
            raise ValueError(
                "`choices` must be None for continuous or circular responses."
            )

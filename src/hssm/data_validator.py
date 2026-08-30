"""Data validation and preprocessing utilities for HSSM behavioral models."""

import logging
import warnings
from numbers import Integral, Real

import numpy as np
import pandas as pd

from ._types import ResponseDomainSpec

_logger = logging.getLogger("hssm")


class DataValidatorMixin:
    """Provide validation and preprocessing methods for HSSM behavioral models."""

    data: pd.DataFrame
    response: list[str]
    response_domains: dict[str, ResponseDomainSpec]
    choices: tuple[int, ...] | None
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
            valid_rows = np.ones(len(self.data), dtype=bool)
        else:
            if self.deadline or self.missing_data:
                if -999.0 not in self.data["rt"].unique():
                    raise ValueError(
                        "You have no missing data in your dataset, "
                        + "which is not allowed when `missing_data` or `deadline` "
                        "is set to True."
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
            valid_rows = self.data["rt"].to_numpy() != -999.0

        for column, domain in self.response_domains.items():
            observed = self.data.loc[valid_rows, column].to_numpy()
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or (not isinstance(value, Integral) and not np.isfinite(value))
                for value in observed
            ):
                raise ValueError(
                    f"Response column {column!r} must contain finite numeric values."
                )
            if domain["kind"] == "categorical":
                allowed = domain["values"]
                observed_values = set(observed.tolist())
                invalid = sorted(observed_values - set(allowed))
                if invalid:
                    invalid_responses = [
                        int(value)
                        if isinstance(value, Integral) or float(value).is_integer()
                        else float(value)
                        for value in invalid
                    ]
                    if column == "response" and len(self.response_domains) == 1:
                        raise ValueError(
                            "Invalid responses found in your dataset: "
                            f"{invalid_responses}"
                        )
                    raise ValueError(
                        f"Invalid responses found in column {column!r}: "
                        f"{invalid_responses}"
                    )

                missing = sorted(set(allowed) - observed_values)
                if missing:
                    if column == "response" and len(self.response_domains) == 1:
                        message = (
                            f"You set choices to be {allowed}, but {missing} "
                            "are missing from your dataset."
                        )
                    else:
                        message = (
                            f"Categorical response domain for {column!r} declares "
                            f"{allowed}, but {missing} are missing from your dataset."
                        )
                    warnings.warn(message, UserWarning, stacklevel=2)
                continue

            numeric = observed.astype(float, copy=False)
            bounds = domain.get("bounds")
            if bounds is None:
                continue
            lower, upper = bounds
            outside = (numeric < lower) | (
                numeric >= upper if domain["kind"] == "circular" else numeric > upper
            )
            if np.any(outside):
                interval = "half-open" if domain["kind"] == "circular" else "closed"
                raise ValueError(
                    f"Response column {column!r} has values outside its {interval} "
                    f"bounds {bounds}."
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

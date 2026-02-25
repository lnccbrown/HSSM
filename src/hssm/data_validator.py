"""Data validation and preprocessing utilities for HSSM behavioral models."""

import logging
import warnings

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401

from hssm.defaults import MissingDataNetwork  # noqa: F401

_logger = logging.getLogger("hssm")


class DataValidatorMixin:
    """Mixin providing validation and preprocessing methods for HSSM behavioral models.

    This class expects subclasses to define the following attributes:
    - data: pd.DataFrame
    - response: list[str]
    - choices: list[int]
    - n_choices: int
    - extra_fields: list[str] | None
    - deadline: bool
    - deadline_name: str
    - missing_data: bool
    - missing_data_value: float
    """

    def __init__(
        self,
        data: pd.DataFrame,
        response: list[str] | None = ["rt", "response"],
        choices: list[int] | None = [0, 1],
        n_choices: int = 2,
        extra_fields: list[str] | None = None,
        deadline: bool = False,
        deadline_name: str = "deadline",
        missing_data: bool = False,
        missing_data_value: float = -999.0,
    ):
        """Initialize the DataValidatorMixin.

        Init method kept for testing purposes.
        """
        self.data = data
        self.response = response
        self.choices = choices
        self.n_choices = n_choices
        self.extra_fields = extra_fields
        self.deadline = deadline
        self.deadline_name = deadline_name
        self.missing_data = missing_data
        self.missing_data_value = missing_data_value

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
        if self.deadline or self.missing_data:
            if -999.0 not in self.data["rt"].unique():
                raise ValueError(
                    "You have no missing data in your dataset, "
                    + "which is not allowed when `missing_data` or `deadline` is set to"
                    + " True."
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

        valid_responses = self.data.loc[self.data["rt"] != -999.0, "response"]
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
        """
        Ensure that `choices` is provided (not None).

        Raises ValueError if choices is None.
        """
        if self.choices is None:
            raise ValueError(
                "`choices` must be provided either in `model_config` or as an argument."
            )

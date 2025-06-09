"""Data validation and preprocessing utilities for HSSM behavioral models."""

import logging

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401

from hssm.defaults import MissingDataNetwork  # noqa: F401

_logger = logging.getLogger("hssm")


class DataValidator:
    """Class for validating and preprocessing behavioral data for HSSM models."""

    def __init__(
        self,
        data,
        response,
        choices,
        n_choices,
        extra_fields=None,
        deadline=False,
        deadline_name="deadline",
        missing_data=False,
        missing_data_value=-999.0,
    ):
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

        DataValidator.check_fields(self.extra_fields, data.columns)

        return True

    def _pre_check_data_sanity(self):
        """Check if the data is clean enough for the model."""
        DataValidator.check_fields(self.response, self.data.columns)
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
                unique_responses[~np.isin(unique_responses, self.choices)]
            )
            raise ValueError(
                f"Invalid responses found in your dataset: {invalid_responses}"
            )

        if len(unique_responses) != self.n_choices:
            missing_responses = sorted(np.setdiff1d(self.choices, unique_responses))
            _logger.warning(
                "You set choices to be %s, but %s are missing from your dataset.",
                self.choices,
                missing_responses,
            )

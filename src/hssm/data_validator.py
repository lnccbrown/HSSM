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

    def _pre_check_data_sanity(self):
        """Check if the data is clean enough for the model."""
        for field in self.response:
            if field not in self.data.columns:
                raise ValueError(f"Field {field} not found in data.")

        self._check_extra_fields()

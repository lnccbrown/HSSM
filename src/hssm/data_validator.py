"""Data validation and preprocessing utilities for HSSM behavioral models."""

import logging

import numpy as np
import pandas as pd

from hssm.defaults import MissingDataNetwork

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

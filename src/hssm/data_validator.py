"""Data validation and preprocessing utilities for HSSM behavioral models."""

import logging
import warnings

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401

from hssm.defaults import MissingDataNetwork  # noqa: F401

_logger = logging.getLogger("hssm")


class DataValidator:
    """Class for validating and preprocessing behavioral data for HSSM models."""

    def __init__(
        self,
        data,
        response=["rt", "response"],
        choices=[0, 1],
        n_choices=2,
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
    def _handle_missing_data_and_deadline(self):
        """Handle missing data and deadline."""
        if not self.missing_data and not self.deadline:
            # In the case where missing_data is set to False, we need to drop the
            # cases where rt = na_value
            if pd.isna(self.missing_data_value):
                na_dropped = self.data.dropna(subset=["rt"])
            else:
                na_dropped = self.data.loc[
                    self.data["rt"] != self.missing_data_value, :
                ]

            if len(na_dropped) != len(self.data):
                warnings.warn(
                    "`missing_data` is set to False, "
                    + "but you have missing data in your dataset. "
                    + "Missing data will be dropped.",
                    stacklevel=2,
                )
            self.data = na_dropped

        elif self.missing_data and not self.deadline:
            # In the case where missing_data is set to True, we need to replace the
            # missing data with a specified na_value

            # Create a shallow copy to avoid modifying the original dataframe
            if pd.isna(self.missing_data_value):
                self.data["rt"] = self.data["rt"].fillna(-999.0)
            else:
                self.data["rt"] = self.data["rt"].replace(
                    self.missing_data_value, -999.0
                )

        else:  # deadline = True
            if self.deadline_name not in self.data.columns:
                raise ValueError(
                    "You have specified that your data has deadline, but "
                    + f"`{self.deadline_name}` is not found in your dataset."
                )
            else:
                self.data.loc[:, "rt"] = np.where(
                    self.data["rt"] < self.data[self.deadline_name],
                    self.data["rt"],
                    -999.0,
                )

    def _update_extra_fields(self, new_data: pd.DataFrame | None = None):
        """Update the extra fields data in self.model_distribution.

        Parameters
        ----------
        new_data
            A DataFrame containing new data for update.
        """
        if not new_data:
            new_data = self.data

        # The attribute 'model_distribution' is not defined in DataValidator itself,
        # but is expected to exist in subclasses (e.g., HSSM).
        # The 'type: ignore[attr-defined]' comment tells mypy to ignore the missing
        # attribute error here and avoid moving this method to the HSSM class.
        self.model_distribution.extra_fields = [  # type: ignore[attr-defined]
            new_data[field].values for field in self.extra_fields
        ]

    @staticmethod
    def _set_missing_data_and_deadline(
        missing_data: bool, deadline: bool, data: pd.DataFrame
    ) -> MissingDataNetwork:
        """Set missing data and deadline."""
        network = MissingDataNetwork.NONE
        if not missing_data:
            return network
        if missing_data and not deadline:
            network = MissingDataNetwork.CPN
        elif missing_data and deadline:
            network = MissingDataNetwork.OPN
        # AF-TODO: GONOGO case not yet correctly implemented
        # else:
        #     # TODO: This won't behave as expected yet, GONOGO needs to be split
        #     # into a deadline case and a non-deadline case.
        #     network = MissingDataNetwork.GONOGO

        if np.all(data["rt"] == -999.0):
            if network in [MissingDataNetwork.CPN, MissingDataNetwork.OPN]:
                # AF-TODO: I think we should allow invalid-only datasets.
                raise ValueError(
                    "`missing_data` is set to True, but you have no valid data in your "
                    "dataset."
                )
            # AF-TODO: This one needs refinement for GONOGO case
            # elif network == MissingDataNetwork.OPN:
            #     raise ValueError(
            #         "`deadline` is set to True and `missing_data` is set to True, "
            #         "but ."
            #     )
            # else:
            #     raise ValueError(
            #         "`missing_data` and `deadline` are both set to True,
            #         "but you have "
            #         "no missing data and/or no rts exceeding the deadline."
            #     )
        return network

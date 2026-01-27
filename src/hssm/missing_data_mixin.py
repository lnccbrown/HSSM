"""Mixin module for handling missing data and deadline logic in HSSM models."""

import warnings

import numpy as np
import pandas as pd

from hssm.defaults import MissingDataNetwork  # noqa: F401


class MissingDataMixin:
    """Mixin for handling missing data and deadline logic in HSSM models.

    Parameters
    ----------
    missing_data : optional
        Specifies whether the model should handle missing data. Can be a `bool`
        or a `float`. If `False`, and if the `rt` column contains -999.0, the
        model will drop those rows and produce a warning. If `True`, the model
        will treat -999.0 as missing data. If a `float` is provided, it will be
        treated as the missing data value. Defaults to `False`.
    deadline : optional
        Specifies whether the model should handle deadline data. Can be a `bool`
        or a `str`. If `False`, the model will not act even if a deadline column
        is provided. If `True`, the model will treat the `deadline` column as
        deadline data. If a `str` is provided, it is treated as the name of the
        deadline column. Defaults to `False`.
    loglik_missing_data : optional
        A likelihood function for missing data. See the `loglik` parameter for
        details. If not provided, a default likelihood is used. Required only if
        either `missing_data` or `deadline` is not `False`.
    """

    def _handle_missing_data_and_deadline(self):
        """Handle missing data and deadline.

        Originally from DataValidatorMixin. Handles dropping, replacing, or masking
        missing data and deadline values in self.data based on the current settings.
        """
        import warnings

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

    def _process_missing_data_and_deadline(
        self, missing_data: float | bool, deadline: bool | str, loglik_missing_data
    ):
        """
        Process missing data and deadline logic for the model's data.

        This method sets up missing data and deadline handling for the model.
        It updates self.missing_data, self.missing_data_value, self.deadline,
        self.deadline_name, and self.loglik_missing_data based on the arguments.
        It also modifies self.data in-place to drop or replace missing/deadline
        values as appropriate, and sets self.missing_data_network.

        Parameters
        ----------
        missing_data : float or bool
            If True, treat -999.0 as missing data. If a float, use that value
            as the missing data marker. If False, drop missing data rows.
        deadline : bool or str
            If True, use the 'deadline' column for deadline logic. If a str,
            use that column name. If False, ignore deadline logic.
        loglik_missing_data : callable or None
            Optional custom likelihood function for missing data. If not None,
            must be used only when missing_data or deadline is True.
        """
        if isinstance(missing_data, float):
            if not ((self.data.rt == missing_data).any()):
                raise ValueError(
                    f"missing_data argument is provided as a float {missing_data}, "
                    f"However, you have no RTs of {missing_data} in your dataset!"
                )
            else:
                self.missing_data = True
                self.missing_data_value = missing_data
        elif isinstance(missing_data, bool):
            if missing_data:
                if not (self.data.rt == -999.0).any():
                    raise ValueError(
                        "missing_data argument is provided as True, "
                        " so RTs of -999.0 are treated as missing. \n"
                        "However, you have no RTs of -999.0 in your dataset!"
                    )
                self.missing_data = True
                self.missing_data_value = -999.0
            else:
                if (self.data.rt == -999.0).any():
                    warnings.warn(
                        "missing_data is False, but -999.0 found in rt column."
                        "Dropping those rows.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self.data = self.data[self.data.rt != -999.0].reset_index(drop=True)
                self.missing_data = False
                self.missing_data_value = -999.0
        else:
            raise ValueError(
                "missing_data argument must be a bool or a float! \n"
                f"You provided: {type(missing_data)}"
            )

        if isinstance(deadline, str):
            self.deadline = True
            self.deadline_name = deadline
        else:
            self.deadline = deadline
            self.deadline_name = "deadline"

        if (
            not self.missing_data and not self.deadline
        ) and loglik_missing_data is not None:
            raise ValueError(
                "You have specified a loglik_missing_data function, but you have not "
                "set the missing_data or deadline flag to True."
            )
        self.loglik_missing_data = loglik_missing_data

        # Update data based on missing_data and deadline
        self._handle_missing_data_and_deadline()
        # Set self.missing_data_network based on `missing_data` and `deadline`
        self.missing_data_network = self._set_missing_data_and_deadline(
            self.missing_data, self.deadline, self.data
        )

        if self.deadline and self.response is not None:  # type: ignore[attr-defined]
            if self.deadline_name not in self.response:  # type: ignore[attr-defined]
                self.response.append(self.deadline_name)  # type: ignore[attr-defined]

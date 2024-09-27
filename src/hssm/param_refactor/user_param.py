"""The Param class is a container for user-specified parameters of the HSSM model."""

from dataclasses import dataclass
from typing import Any, Union

import bambi as bmb
import numpy as np


@dataclass
class UserParam:
    """Represent the user-provided specifications for the main HSSM class.

    Also provides convenience functions that can be used by the HSSM class to parse
    arguments.

    Parameters
    ----------
    name
        The name of the parameter. This can be omitted if the Param is specified as
        kwargs in the HSSM class.
    prior
        If a formula is not specified (the non-regression case), this parameter
        expects a float value if the parameter is fixed or a dictionary that can be
        parsed by Bambi as a prior specification or a Bambi Prior object.
        If a formula is specified (the regression case), this parameter expects a
        dictionary of param:prior, where param is the name of the response variable
        specified in formula, and prior is specified as above. If left unspecified,
        default priors created by Bambi will be used.
    formula
        The regression formula if the parameter depends on other variables.
    link
        The link function for the regression. It is either a string that specifies
        a built-in link function in Bambi, or a Bambi Link object. If a regression
        is specified and link is not specified, "identity" will be used by default.
    bounds
        If provided, the prior will be created with boundary checks. If this
        parameter is specified as a regression, boundary checks will be skipped at
        this point.
    """

    name: str | None = None
    prior: int | float | np.ndarray | dict[str, Any] | bmb.Prior = None
    formula: str | None = None
    link: str | bmb.Link | None = None
    bounds: tuple[int, int] | None = None

    @staticmethod
    def from_dict(param_dict: dict[str, Any]) -> "UserParam":
        """Create a Param object from a dictionary.

        Parameters
        ----------
        param_dict
            A dictionary with the keys "name", "prior", "formula", "link", and "bounds".

        Returns
        -------
        Param
            A Param object with the specified parameters.
        """
        return UserParam(**param_dict)

    @staticmethod
    def from_kwargs(
        name: str,
        # Using Union here because "UserParam" is a forward reference
        param: Union[int, float, np.ndarray, dict[str, Any], bmb.Prior, "UserParam"],
    ) -> "UserParam":
        """Create a Param object from keyword arguments.

        Parameters
        ----------
        name
            The name of the parameter.
        param
            The prior specification for the parameter.

        Returns
        -------
        Param
            A Param object with the specified parameters.
        """
        if isinstance(param, dict):
            return UserParam(name=name, **param)
        elif isinstance(param, UserParam):
            param.name = name
            return param

        return UserParam(name=name, prior=param)

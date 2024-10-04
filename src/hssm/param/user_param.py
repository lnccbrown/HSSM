"""The Param class is a container for user-specified parameters of the HSSM model."""

from dataclasses import dataclass, fields
from typing import Any, Union

import bambi as bmb
import numpy as np

from .utils import validate_bounds


@dataclass
class UserParam:
    """Represent the user-provided specifications for the main HSSM class.

    Also provides convenience functions that can be used by the HSSM class to parse
    arguments.

    Parameters
    ----------
    name : optional
        The name of the parameter. This can be omitted if the Param is specified as
        kwargs in the HSSM class.
    prior : optional
        If a formula is not specified (the non-regression case), this parameter
        expects a float value if the parameter is fixed or a dictionary that can be
        parsed by Bambi as a prior specification or a Bambi Prior object.
        If a formula is specified (the regression case), this parameter expects a
        dictionary of param:prior, where param is the name of the response variable
        specified in formula, and prior is specified as above. If left unspecified,
        default priors created by Bambi will be used.
    formula : optional
        The regression formula if the parameter depends on other variables.
    link : optional
        The link function for the regression. It is either a string that specifies
        a built-in link function in Bambi, or a Bambi Link object. If a regression
        is specified and link is not specified, "identity" will be used by default.
    bounds : optional
        If provided, the prior will be created with boundary checks. If this
        parameter is specified as a regression, boundary checks will be skipped at
        this point.

    Attributes
    ----------
    name
    prior
    formula
    link
    bounds
    is_regression
        Whether the parameter is a regression.

    Methods
    -------
    from_dict
        Create a Param object from a dictionary.
    from_kwargs
        Create a Param object from keyword arguments.
    to_dict
        Convert the UserParam object to a dictionary with shallow copy.
    """

    name: str | None = None
    prior: float | np.ndarray | dict[str, Any] | bmb.Prior | None = None
    formula: str | None = None
    link: str | bmb.Link | None = None
    bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Validate the user-provided parameters."""
        if self.bounds is not None:
            validate_bounds(self.bounds)

    @property
    def is_regression(self) -> bool:
        """Check if the parameter is a regression parameter."""
        return self.formula is not None

    @property
    def is_simple(self) -> bool:
        """Check if the parameter is a simple parameter."""
        if self.is_regression:
            return False
        if isinstance(self.prior, (int, float, np.ndarray, bmb.Prior)):
            return True
        return isinstance(self.prior, dict) and "name" in self.prior

    @classmethod
    def from_dict(cls, param_dict: dict[str, Any]) -> "UserParam":
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
        return cls(**param_dict)

    @classmethod
    def from_kwargs(
        cls,
        name: str,
        # Using Union here because "UserParam" is a forward reference
        param: Union[float, np.ndarray, dict[str, Any], bmb.Prior, "UserParam"],
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
            if "name" in param:
                if param["name"] == name:
                    return cls(**param)
                return cls(name=name, prior=param)
            return cls(name=name, **param)
        elif isinstance(param, UserParam):
            param.name = name
            return param

        return cls(name=name, prior=param)

    def to_dict(
        self,
    ) -> dict:
        """Convert the UserParam object to a dictionary with shallow copy."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

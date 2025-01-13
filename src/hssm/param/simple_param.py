"""Class for simple (non-regression) parameterization of HSSM models."""

from typing import Any

import bambi as bmb
import numpy as np

from ..prior import Prior
from .param import Param
from .user_param import UserParam
from .utils import _make_default_prior


class SimpleParam(Param):
    """A simple parameter.

    A simple parameter is a parameter that does not have a regression.

    Parameters
    ----------
    name
        The name of the parameter.
    prior : optional
        The prior specification for the parameter.
    bounds : optional
        The bounds for the parameter.
    user_param : optional
        The parameter as specified by the user.

    Attributes
    ----------
    name
    prior
    formula
    link
    bounds
    user_param
    is_regression
        Whether the parameter is a regression.
    is_fixed
        Whether the parameter is fixed.
    is_vector
        Whether the parameter is a vector parameter.
    is_parent
        Whether the parameter is a parent parameter.

    Methods
    -------
    from_user_param
        Create a SimpleParam or DefaultParam object from a UserParam object.
    process_prior
        Process the prior specification.
    parse_bambi
        Parse the parameter for Bambi.
    validate
        Validate the parameter.
    """

    def __init__(
        self,
        name: str,
        prior: float | np.ndarray | dict[str, Any] | bmb.Prior | None = None,
        bounds: tuple[float, float] | None = None,
        user_param: UserParam | None = None,
    ) -> None:
        super().__init__(name=name, prior=prior, bounds=bounds, user_param=user_param)

    @classmethod
    def from_user_param(cls, user_param: UserParam) -> "SimpleParam":
        """Create a SimpleParam or DefaultParam object from a UserParam object.

        Parameters
        ----------
        user_param
            The UserParam object to convert.

        Returns
        -------
        Param
            A SimpleParam object with the specified parameters.

        NOTE
        ----
        We need to handle the case where the user has not specified a prior for the
        parameter. This is equivalent to simply using the default prior.
        """
        if user_param.name is None:
            raise ValueError("Name not specified for parameter")
        if user_param.is_regression:
            raise ValueError(
                f"Regression specified for simple parameter {user_param.name}"
            )
        if user_param.link is not None:
            raise ValueError(f"Link specified for simple parameter {user_param.name}")
        if user_param.prior is None:
            if user_param.bounds is None:
                raise ValueError(
                    f"Bounds not specified for parameter {user_param.name}"
                )
            return DefaultParam(
                name=user_param.name,
                prior=None,
                bounds=user_param.bounds,
            )
        return cls(
            name=user_param.name,
            prior=user_param.prior,
            bounds=user_param.bounds,
            user_param=user_param,
        )

    def fill_defaults(
        self,
        prior: dict[str, Any] | None = None,
        bounds: tuple[float, float] | None = None,
        **kwargs,
    ) -> None:
        """Fill in the default values for the parameter."""
        if "formula" in kwargs:
            raise ValueError(f"Formula specified for simple parameter {self.name}")
        if "link" in kwargs:
            raise ValueError(f"Link specified for simple parameter {self.name}")
        return super().fill_defaults(prior, bounds, **kwargs)

    def validate(self) -> None:
        """Validate the parameter."""
        if self.prior is None:
            raise ValueError(f"Prior not specified for parameter {self.name}")
        if self.bounds is not None and self.is_fixed:
            lower, upper = self.bounds
            if isinstance(self.prior, (int, float)):
                if not lower <= self.prior <= upper:
                    raise ValueError(
                        f"Fixed Value {self.prior} not in bounds {self.bounds} for"
                        f" parameter {self.name}"
                    )
            elif isinstance(self.prior, np.ndarray):
                if np.any(self.prior < lower) or np.any(self.prior > upper):
                    raise ValueError(
                        f"Fixed Value {self.prior} not in bounds {self.bounds} for"
                        f" parameter {self.name}"
                    )

    def process_prior(self) -> None:
        """Process the prior specification."""
        self.validate()
        if isinstance(self.prior, dict):
            if self.bounds is not None:
                self.prior = Prior(bounds=self.bounds, **self.prior)
            else:
                self.prior = bmb.Prior(**self.prior)

    def __repr__(self) -> str:
        """Return the representation of the class.

        Returns
        -------
        str
            A string representing the class.
        """
        if isinstance(self.prior, (int, float)):
            prior = f"Value: {self.prior}"
        elif isinstance(self.prior, np.ndarray):
            n_elements = len(self.prior)
            if n_elements > 5:
                prior = f"Value: {self.prior[:5]}..."
            else:
                prior = f"Value: {self.prior}"
        elif isinstance(self.prior, (dict, bmb.Prior)):
            prior = f"Prior: {self.prior}"
        else:
            raise ValueError(f"Invalid prior type {type(self.prior)}")

        if self.bounds is None:
            return f"{self.name}:\n    {prior}"
        return f"{self.name}:\n    {prior}\n    Explicit bounds: {self.bounds}"


class DefaultParam(SimpleParam):
    """A default parameter.

    A default parameter is a simple parameter that has no user specification.

    Parameters
    ----------
    name
        The name of the parameter.
    prior
        The prior specification for the parameter.
    bounds
        The bounds for the parameter.

    Methods
    -------
    from_defaults
        Create a DefaultParam object from default values.
    make_default_prior
        Make a default prior from bounds.
    """

    def __init__(
        self,
        name: str,
        prior: float | np.ndarray | dict[str, Any] | bmb.Prior,
        bounds: tuple[float, float],
    ) -> None:
        super().__init__(name, prior=prior, bounds=bounds)

    @classmethod
    def from_defaults(
        cls, name: str, prior: dict[str, Any], bounds: tuple[int, int]
    ) -> "DefaultParam":
        """Create a DefaultParam object from default values.

        Parameters
        ----------
        name
            The name of the parameter.
        prior
            The prior specification for the parameter.
        bounds
            The bounds for the parameter.

        Returns
        -------
        DefaultParam
            A DefaultParam object with the specified parameters.
        """
        param = cls(name, prior, bounds)

        if param.prior is None:
            param.make_default_prior()

        return param

    def process_prior(self) -> None:
        """Process the prior specification.

        NOTE
        ----
        This method is different from the one in SimpleParam, in that if we are applying
        a default parameter, we do not need to truncate the prior. This is because the
        the default parameters are already bounded.
        """
        self.validate()
        if isinstance(self.prior, dict):
            self.prior = bmb.Prior(**self.prior)

    def fill_defaults(
        self,
        prior: dict[str, Any] | None = None,
        bounds: tuple[float, float] | None = None,
        **kwargs,
    ) -> None:
        """Fill in the default values for the parameter.

        If the prior is still not specified, then a default prior is made.

        Parameters
        ----------
        prior
            The prior specification for the parameter.
        bounds
            The bounds for the parameter.
        """
        super().fill_defaults(prior, bounds, **kwargs)

        if self.prior is None:
            self.make_default_prior()

    def make_default_prior(self):
        """Make a default prior from bounds.

        Parameters
        ----------
        bounds
            The (lower, upper) bounds for the default prior.

        Returns
        -------
            A bmb.Prior object representing the default prior for the provided bounds.
        """
        self.prior = _make_default_prior(self.bounds)

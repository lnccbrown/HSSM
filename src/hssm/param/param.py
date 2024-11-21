"""The parent class for all parameters in the model."""

from typing import Any

import bambi as bmb
import numpy as np

from .user_param import UserParam
from .utils import validate_bounds


class Param:
    """The parent class for all parameters.

    Parameters
    ----------
    name
        The name of the parameter.
    prior
        The prior specification for the parameter.
    formula
        The formula for the parameter.
    link
        The link function for the parameter.
    bounds
        The bounds for the parameter.
    user_param
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
        Create a Param object from a UserParam object.
    fill_defaults
        Fill in the default values for the parameter.
    validate
        Validate the parameter.
    process_prior
        Process the prior specification
    parse_bambi
        Parse the parameter for Bambi.
    """

    name: str
    prior: float | np.ndarray | dict[str, Any] | bmb.Prior | None
    formula: str | None
    link: str | bmb.Link | None
    bounds: tuple[float, float] | None
    user_param: UserParam | None

    def __init__(
        self,
        name: str,
        prior: float | np.ndarray | dict[str, Any] | bmb.Prior | None = None,
        formula: str | None = None,
        link: str | bmb.Link | None = None,
        bounds: tuple[float, float] | None = None,
        user_param: UserParam | None = None,
    ) -> None:
        self.name = name
        self.prior = prior
        self.formula = formula
        self.link = link
        self.bounds = bounds
        self.user_param = user_param
        self._parent = False

        if self.bounds is not None:
            validate_bounds(self.bounds)

    @classmethod
    def from_user_param(cls, user_param: UserParam) -> "Param":
        """Create a Param object from a UserParam object.

        Parameters
        ----------
        user_param
            A UserParam object.

        Returns
        -------
        Param
            A Param object with the specified parameters.
        """
        return cls(
            **user_param.to_dict(),
            user_param=user_param,
        )

    @property
    def is_regression(self) -> bool:
        """Whether the parameter is a regression."""
        return self.formula is not None

    @property
    def is_fixed(self) -> bool:
        """Whether the parameter is fixed as a scalar or a vector."""
        return isinstance(self.prior, (int, float, np.ndarray))

    @property
    def is_parent(self) -> bool:
        """Whether the parameter is a parent parameter."""
        return self._parent

    @is_parent.setter
    def is_parent(self, value: bool) -> None:
        self._parent = value

    @property
    def is_vector(self) -> bool:
        """Whether the parameter is a vector parameter."""
        return (
            self.is_parent or self.is_regression or isinstance(self.prior, np.ndarray)
        )

    def fill_defaults(
        self,
        prior: dict[str, Any] | None = None,
        bounds: tuple[float, float] | None = None,
        **kwargs,
    ) -> None:
        """Fill in the default values for the parameter.

        Parameters
        ----------
        prior
            The prior specification for the parameter.
        bounds
            The bounds for the parameter.
        """
        if self.bounds is None:
            self.bounds = bounds
        if self.prior is None:
            self.prior = prior
        for key, value in kwargs.items():
            if getattr(self, key) is None:
                setattr(self, key, value)

    def validate(self) -> None:
        """Validate the parameter."""
        raise NotImplementedError("This method is to be implemented in subclasses.")

    def process_prior(self) -> None:
        """Process the prior specification."""
        raise NotImplementedError("This method is to be implemented in subclasses.")

    def parse_bambi(
        self,
    ) -> tuple[str | None, dict[str, Any] | bmb.Prior | None, str | bmb.Link | None]:
        """Parse the parameter for Bambi.

        Returns
        -------
        tuple
            1. A bmb.Formula object, or None if the parameter is simple.
            2. A bmb.Prior object, in the simple case, or a dictionary of priors in
            the regression case, if any is specified, or None if the priors
            should be determined by Bambi.
            3. A string or bmb.Link object representing the link function, or None
            in the simple case.
        """
        return self.formula, self.prior, self.link

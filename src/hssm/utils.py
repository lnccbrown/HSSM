"""
Contains classes and functions that helps the main HSSM class parse arguments.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import bambi as bmb

PriorSpec = Dict[str, Any]

PARAM_DEFAULTS = {
    "a": bmb.Prior("Uniform", lower=0.1, upper=1.0),
    "z": 0.0,
    "t": 0.0,
}


class Param:
    """
    Represents the specifications for the main HSSM class.

    Also provides convenience functions that can be used by the HSSM class to
    parse arguments.
    """

    def __init__(
        self,
        name: str,
        prior: float | PriorSpec | bmb.Prior | None = None,
        formula: str | None = None,
        dep_priors: Dict[str, PriorSpec] | Dict[str, bmb.Prior] | None = None,
        link: str | bmb.Link | None = "identity",
    ):
        """Parses the parameters to class properties.

        Parameters
        ----------
        name
            The name of the parameter
        prior, optional
            A float value if the parameter is fixed. Otherwise, provide a dictionary
            that can be parsed by Bambi as a prior specification or a Bambi Prior,
            by default None.
        formula, optional
            The regression formula if the parameter depends on other variables. The
            response variable can be omitted, by default None.
        dep_priors, optional
            A dictionary of param:prior, where param is the name of the dependent
            variable specified in formula, and prior is either a dictionary that can be
            parsed by Bambi as a prior specification, or a Bambi Prior object,
            by default None
        link, optional
            The link function for the regression. It is either a string that specifies
            a built-in link function in Bambi, or a Bambi Link object, by default
            "identity".
        """

        self.name = name

        # Check if the user has specified a formula
        self._regression = formula is not None

        if self._regression:
            if prior is not None:
                raise ValueError(
                    f"A regression model is specified for parameter {self.name}."
                    + " `prior` parameter should not be specified."
                )

            ## NOTE: for now, we only handle the case where no priors are provided at
            ## all. We delegate cases there are some variables whose priors are not
            ## provided to Bambi at the moment. This could be improved once we know more
            ## about how the formulae package works
            if dep_priors is None:
                raise ValueError(
                    f"Priors for the variables that {self.name} is regressed on "
                    + "are not specified."
                )

            if "~" not in formula:  # type: ignore
                formula = f"{self.name} ~ {formula}"

            self._formula = formula

            self.dep_priors = {
                # Convert dict to bmb.Prior if a dict is passed
                param: (prior if isinstance(prior, bmb.Prior) else bmb.Prior(**prior))
                for param, prior in dep_priors.items()
            }
            self._link = link
        else:
            if prior is None:
                raise ValueError(
                    f"Please specify a value or a prior for parameter {self.name}."
                )

            if dep_priors is not None:
                raise ValueError(
                    f"dep_priors should not be specified for {self.name} "
                    + "if a formula is not specified."
                )

            self.prior = bmb.Prior(**prior) if isinstance(prior, dict) else prior

    def is_regression(self) -> bool:
        """Determines if a regression is specified or not.

        Returns
        -------
            A boolean that indicates if a regression is specified.
        """

        return self._regression

    @property
    def link(self) -> str | bmb.Link | None:
        return self._link if self.is_regression() else None

    @property
    def formula(self) -> str | None:
        return self._formula if self.is_regression() else None

    def _parse_bambi(
        self,
    ) -> Tuple[str | None, Dict[str, Any], Dict[str, str | bmb.Link] | None]:
        """Returns a 3-tuple that helps with constructing the Bambi model.

        Returns
        -------
            A 3-tuple of formula, priors, and link functions that can be used to
            construct the Bambi model.
        """
        if self.is_regression():

            priors = {self.name: self.dep_priors}
            link = {self.name: self.link}

            return self.formula, priors, link  # type: ignore

        prior = {self.name: self.prior}

        return None, prior, None

    def __repr__(self) -> str:
        """Returns the representation of the class.

        Returns
        -------
            A string whose construction depends on whether the specification
            contains a regression or not.
        """

        if not self.is_regression():
            if isinstance(self.prior, bmb.Prior):
                return f"{self.name} ~ {self.prior}"
            return f"{self.name} = {self.prior}"

        link = (
            self.link if isinstance(self.link, str) else self.link.name  # type: ignore
        )
        priors = "\r\n".join(
            [f"{param} ~ {prior}" for param, prior in self.dep_priors.items()]
        )

        return "\r\n".join([self.formula, f"Link: {link}", priors])  # type: ignore

    def __str__(self) -> str:
        """Returns the string representation of the class.

        Returns
        -------
            A string whose construction depends on whether the specification
            contains a regression or not.
        """
        return self.__repr__()

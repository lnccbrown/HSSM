"""
HSSM has to reconcile with two representations: it's own representation as an HSSM and
the representation acceptable to Bambi. The two are not equivalent. This file contains
the Param class that reconcile these differences.

The Param class is an abstraction that stores the parameter specifications and turns
these representations in Bambi-compatible formats through convenience function
_parse_bambi().
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import bambi as bmb


class Param:
    """
    Represents the specifications for the main HSSM class.

    Also provides convenience functions that can be used by the HSSM class to
    parse arguments.
    """

    def __init__(
        self,
        name: str,
        prior: float
        | Dict[str, Any]
        | bmb.Prior
        | Dict[str, Dict[str, Any]]
        | Dict[str, bmb.Prior]
        | None = None,
        formula: str | None = None,
        link: str | bmb.Link | None = None,
        is_parent: bool = False,
    ):
        """Parses the parameters to class properties.

        Parameters
        ----------
        name
            The name of the parameter
        prior
            If a formula is not specified, this parameter expects a float value if the
            parameter is fixed or a dictionary that can be parsed by Bambi as a prior
            specification or a Bambi Prior. if a formula is specified, this parameter
            expects a dictionary of param:prior, where param is the name of the
            response variable specified in formula, and prior is either a dictionary
            that can be parsed by Bambi as a prior specification, or a Bambi Prior.
            By default None.
        formula, optional
            The regression formula if the parameter depends on other variables. The
            response variable can be omitted, by default None.
        link, optional
            The link function for the regression. It is either a string that specifies
            a built-in link function in Bambi, or a Bambi Link object. If a regression
            is speicified and link is not specified, "identity" will be used.
        is_parent:
            Determines if this parameter is a "parent" parameter. If so, the response
            term for the formula will be "c(rt, response)".
        """

        self.name = name
        self.formula = formula
        self.link = None
        self._parent = is_parent

        # Check if the user has specified a formula
        self._regression = formula is not None

        if self._regression:

            self.formula = (
                formula if "~" in formula else f"{name} ~ {formula}"  # type: ignore
            )

            self.prior = (
                {
                    # Convert dict to bmb.Prior if a dict is passed
                    param: (
                        prior if isinstance(prior, bmb.Prior) else bmb.Prior(**prior)
                    )
                    for param, prior in prior.items()  # type: ignore
                }
                if prior is not None
                else None
            )

            self.link = "identity" if link is None else link
        else:
            if prior is None:
                raise ValueError(f"Please specify a value or prior for {self.name}.")

            self.prior = (
                bmb.Prior(**prior) if isinstance(prior, dict) else prior  # type: ignore
            )

            if link is not None:
                raise ValueError("`link` should be None if no regression is specified.")

    def is_regression(self) -> bool:
        """Determines if a regression is specified or not.

        Returns
        -------
            A boolean that indicates if a regression is specified.
        """

        return self._regression

    def is_parent(self) -> bool:
        """Determines if a parameter is a parent parameter for Bambi.

        Returns
        -------
            A boolean that indicates if the parameter is a parent or not.
        """

        return self._parent

    def _parse_bambi(
        self,
    ) -> Tuple:
        """Returns a 3-tuple that helps with constructing the Bambi model.

        Returns
        -------
            A 3-tuple of formula, priors, and link functions that can be used to
            construct the Bambi model.
        """

        formula = None
        prior = None
        link = None

        if self._regression:
            left_side = "c(rt, response)" if self._parent else self.name

            right_side = self.formula.split("~")[1]  # type: ignore
            right_side = right_side.strip()
            formula = f"{left_side} ~ {right_side}"

            if self.prior is not None:
                prior = {left_side: self.prior}
            link = {self.name: self.link}

            return formula, prior, link

        formula = "c(rt, response) ~ 1" if self._parent else None

        if self._parent:
            prior = {"c(rt, response)": {"Intercept": self.prior}}
            link = {self.name: "identity"}
        else:
            prior = {self.name: self.prior}  # type: ignore

        return formula, prior, link

    def __repr__(self) -> str:
        """Returns the representation of the class.

        Returns
        -------
            A string whose construction depends on whether the specification
            contains a regression or not.
        """

        if not self._regression:
            if isinstance(self.prior, bmb.Prior):
                return f"{self.name} ~ {self.prior}"
            return f"{self.name} = {self.prior}"

        link = (
            self.link if isinstance(self.link, str) else self.link.name  # type: ignore
        )
        priors = (
            "\r\n".join([f"{param} ~ {prior}" for param, prior in self.prior.items()])
            if self.prior is not None
            else "Unspecified, using defaults"
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


def _parse_bambi(
    params: List[Param],
) -> Tuple[bmb.Formula, Dict | None, Dict[str, str | bmb.Link] | str | None]:
    """From a list of Params, retrieve three items that helps with bambi model building

    Parameters
    ----------
    params
        A list of Param objects.

    Returns
    -------
        A 3-tuple of
            1. A bmb.Formula object.
            2. A dictionary of priors, if any is specified.
            3. A dictionary of link functions, if any is specified.
    """

    # Handle the edge case where list_params is empty:
    if not params:
        return bmb.Formula("c(rt, response) ~ 1"), None, "identity"

    # Then, we check how many parameters in the specified list of params are parent.
    num_parents = sum(param.is_parent() for param in params)

    # In the case where there is more than one parent:
    assert num_parents <= 1, "More than one parent is specified!"

    formulas = []
    priors: Dict[str, Any] = {}
    links: Dict[str, str | bmb.Link] = {}
    params_copy = params.copy()

    if num_parents == 1:
        for idx, param in enumerate(params):
            if param.is_parent():
                parent_param = params_copy.pop(idx)
                break

        params_copy.insert(0, parent_param)

    for param in params_copy:
        formula, prior, link = param._parse_bambi()

        if formula is not None:
            formulas.append(formula)
        if priors is not None:
            priors |= prior
        if link is not None:
            links |= link

    result_formula: bmb.Formula = (
        bmb.Formula("c(rt, response) ~ 1", *formulas)
        if num_parents == 0
        else bmb.Formula(formulas[0], *formulas[1:])
    )
    result_priors = None if not priors else priors

    result_links: Dict | str | None = "identity" if not links else links

    return result_formula, result_priors, result_links

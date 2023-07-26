"""The Param utility class."""

from __future__ import annotations

from typing import Any, Callable, Union, cast

import bambi as bmb
import numpy as np
import pymc as pm
from bambi.backend.utils import get_distribution

# PEP604 union operator "|" not supported by pylint
# Fall back to old syntax

# Explicitly define types so they are more expressive
# and reusable
ParamSpec = Union[float, dict[str, Any], bmb.Prior]
BoundsSpec = tuple[float, float]


class Param:
    """Represents the specifications for the main HSSM class.

    Also provides convenience functions that can be used by the HSSM class to parse
    arguments.

    Parameters
    ----------
    name
        The name of the parameter.
    prior
        If a formula is not specified (the non-regression case), this parameter
        expects a float value if the parameter is fixed or a dictionary that can be
        parsed by Bambi as a prior specification or a Bambi Prior object. If not
        specified, then a default uninformative uniform prior with `bound` as
        boundaries will be constructed. An error will be thrown if `bound` is also
        not specified.
        If a formula is specified (the regression case), this parameter expects a
        dictionary of param:prior, where param is the name of the response variable
        specified in formula, and prior is specified as above. If left unspecified,
        default priors created by Bambi will be used.
    formula
        The regression formula if the parameter depends on other variables. The
        response variable can be omitted.
    link
        The link function for the regression. It is either a string that specifies
        a built-in link function in Bambi, or a Bambi Link object. If a regression
        is specified and link is not specified, "identity" will be used by default.
    bounds
        If provided, the prior will be created with boundary checks. If this
        parameter is specified as a regression, boundary checks will be skipped at
        this point.
    is_parent
        Determines if this parameter is a "parent" parameter. If so, the response
        term for the formula will be "c(rt, response)". Default is False.
    """

    def __init__(
        self,
        name: str,
        prior: ParamSpec | dict[str, ParamSpec] | None = None,
        formula: str | None = None,
        link: str | bmb.Link | None = None,
        bounds: BoundsSpec | None = None,
        is_parent: bool = False,
    ):
        self.name = name
        self.formula = formula
        self._parent = is_parent
        self.bounds = tuple(float(x) for x in bounds) if bounds is not None else None
        self._is_truncated = False

        if self.bounds is not None:
            self.bounds = cast(BoundsSpec, self.bounds)
            if any(not np.isscalar(bound) for bound in self.bounds):
                raise ValueError(f"The bounds of {self.name} should both be scalar.")
            lower, upper = self.bounds
            assert lower < upper, (
                f"The lower bound of {self.name} should be less than "
                + "its upper bound."
            )

        if isinstance(prior, int):
            prior = float(prior)

        if formula is not None:
            # The regression case

            self.formula = formula if "~" in formula else f"{name} ~ {formula}"

            if isinstance(prior, (float, bmb.Prior)):
                raise ValueError(
                    "Please specify priors for each individual parameter in the "
                    + "regression."
                )

            self.prior: float | bmb.Prior = (
                _make_prior_dict(prior) if prior is not None else prior
            )

            self.link = "identity" if link is None else link

        else:
            # The non-regression case

            if prior is None:
                if self.bounds is None:
                    raise ValueError(
                        f"Please specify the prior or bounds for {self.name}."
                    )
                self.prior = _make_default_prior(self.bounds)
            else:
                # Explicitly cast the type of prior, no runtime performance penalty
                prior = cast(ParamSpec, prior)

                if self.bounds is None:
                    if isinstance(prior, (float, bmb.Prior)):
                        self.prior = prior
                    else:
                        self.prior = bmb.Prior(**prior)
                else:
                    if isinstance(prior, float):
                        self.prior = prior
                    else:
                        self.prior = make_bounded_prior(prior, self.bounds)
                        # self._prior is internally used for informative output
                        # Not used in inference
                        self._prior = (
                            bmb.Prior(**prior) if isinstance(prior, dict) else prior
                        )
                        self._is_truncated = True

            if link is not None:
                raise ValueError("`link` should be None if no regression is specified.")

            self.link = None

    @property
    def is_regression(self) -> bool:
        """Determines if a regression is specified or not.

        Returns
        -------
        bool
            A boolean that indicates if a regression is specified.
        """
        return self.formula is not None

    @property
    def is_parent(self) -> bool:
        """Determines if a parameter is a parent parameter for Bambi.

        Returns
        -------
        bool
            A boolean that indicates if the parameter is a parent or not.
        """
        return self._parent

    @property
    def is_fixed(self) -> bool:
        """Determine if a parameter is a fixed value.

        Returns
        -------
        bool
            A boolean that indicates if the parameter is a fixed value.
        """
        return isinstance(self.prior, float)

    @property
    def is_truncated(self) -> bool:
        """Determines if a parameter is truncated.

        A parameter is truncated when it is not a regression, is not fixed and has
        bounds.

        Returns
        -------
            A boolean that indicates if a parameter is truncated.
        """
        return self._is_truncated

    def _parse_bambi(
        self,
    ) -> tuple:
        """
        Return a 3-tuple that helps with constructing the Bambi model.

        Returns
        -------
        tuple
            A 3-tuple of formula, priors, and link functions that can be used to
            construct the Bambi model.
        """
        formula = None
        prior = None
        link = None

        # Again, to satisfy type checker
        # Equivalent to `if self.is_regression`
        if self.formula is not None:
            left_side = "c(rt, response)" if self._parent else self.name

            right_side = self.formula.split("~")[1]
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
        """Return the representation of the class.

        Returns
        -------
        str
            A string whose construction depends on whether the specification contains a
            regression or not.
        """
        output = []
        output.append(self.name + ":")

        # Simplest case: float
        # Output Value: 0.2
        if isinstance(self.prior, float):
            output.append(f"    Value: {self.prior}")
            return "\r\n".join(output)

        # Regression case:
        # Output formula, priors, and link functions
        if self.is_regression:
            assert self.formula is not None
            output.append(f"    Formula: {self.formula}")
            output.append("    Priors:")

            if self.prior is not None:
                assert isinstance(self.prior, dict)

                for param, prior in self.prior.items():
                    output.append(f"        {param} ~ {prior}")
            else:
                output.append("        Unspecified. Using defaults")

            link = self.link if isinstance(self.link, str) else self.link.name
            output.append(f"    Link: {link}")

        # None regression case:
        # Output prior and bounds
        else:
            assert isinstance(self.prior, bmb.Prior)
            prior_output = self._prior if self.is_truncated else self.prior
            output.append(f"    Prior: {prior_output}")

        output.append(f"    Explicit bounds: {self.bounds}")
        return "\r\n".join(output)

    def __str__(self) -> str:
        """Return the string representation of the class.

        Returns
        -------
        str
            A string whose construction depends on whether the specification contains a
            regression or not.
        """
        return self.__repr__()


def _make_prior_dict(prior: dict[str, ParamSpec]) -> dict[str, float | bmb.Prior]:
    """Make bambi priors from a ``dict`` of priors for the regression case.

    Parameters
    ----------
    prior
        A dictionary where each key is the name of a parameter in a regression
        and each value is the prior specification for that parameter.

    Returns
    -------
    dict[str, float | bmb.Prior]
        A dictionary where each key is the name of a parameter in a regression and each
        value is either a float or a bmb.Prior object.
    """
    priors = {
        # Convert dict to bmb.Prior if a dict is passed
        param: _make_priors_recursive(prior) if isinstance(prior, dict) else prior
        for param, prior in prior.items()
    }

    return priors


def _make_priors_recursive(prior: dict[str, Any]) -> bmb.Prior:
    """Make `bmb.Prior` objects from ``dict``s.

    Helper function that recursively converts a dict that might have some fields that
    have a parameter definitions as dicts to bmb.Prior objects.

    Parameters
    ----------
    prior
        A dictionary that contains parameter specifications.

    Returns
    -------
    bmb.Prior
        A bmb.Prior object with fields that can be converted to bmb.Prior objects also
        converted.
    """
    for k, v in prior.items():
        if isinstance(v, dict) and "name" in v:
            prior[k] = _make_priors_recursive(v)

    return bmb.Prior(**prior)


def _parse_bambi(
    params: list[Param],
) -> tuple[bmb.Formula, dict | None, dict[str, str | bmb.Link] | str]:
    """From a list of Params, retrieve three items that helps with bambi model building.

    Parameters
    ----------
    params
        A list of Param objects.

    Returns
    -------
    tuple
        A tuple containing:
            1. A bmb.Formula object.
            2. A dictionary of priors, if any is specified.
            3. A dictionary of link functions, if any is specified.
    """
    # Handle the edge case where list_params is empty:
    if not params:
        return bmb.Formula("c(rt, response) ~ 1"), None, "identity"

    # Then, we check how many parameters in the specified list of params are parent.
    num_parents = sum(param.is_parent for param in params)

    # In the case where there is more than one parent:
    assert num_parents <= 1, "More than one parent is specified!"

    formulas = []
    priors: dict[str, Any] = {}
    links: dict[str, str | bmb.Link] = {}
    params_copy = params.copy()

    parent_param = None

    if num_parents == 1:
        for idx, param in enumerate(params):
            if param.is_parent:
                parent_param = params_copy.pop(idx)
                break

        assert parent_param is not None
        params_copy.insert(0, parent_param)

    for param in params_copy:
        formula, prior, link = param._parse_bambi()

        if formula is not None:
            formulas.append(formula)
        if prior is not None:
            priors |= prior
        if link is not None:
            links |= link

    result_formula: bmb.Formula = (
        bmb.Formula("c(rt, response) ~ 1", *formulas)
        if num_parents == 0
        else bmb.Formula(formulas[0], *formulas[1:])
    )
    result_priors = None if not priors else priors

    result_links: dict | str = "identity" if not links else links

    return result_formula, result_priors, result_links


def make_bounded_prior(prior: ParamSpec, bounds: BoundsSpec) -> float | bmb.Prior:
    """Create prior within specific bounds.

    Helper function that creates a prior within specified bounds. Works in the
    following cases:

    1. If a prior passed is a fixed value, then check if the value is specified within
    the bounds. Raises a ValueError if not.
    2. If a prior passed is a dictionary, we create a bmb.Prior with a truncated
    distribution.
    3. If a prior is passed as a bmb.Prior object, do the same thing above.

    The above boundary checks do not happen when bounds is None.

    Parameters
    ----------
    prior
        A prior definition. Could be a float, a dict that can be passed to a bmb.Prior
        to create a prior distribution, or a bmb.Prior.
    bounds : optional
        If provided, needs to be a tuple of floats that indicates the lower and upper
        bounds of the parameter.

    Returns
    -------
    float | bmb.Prior
        A float if `prior` is a float, otherwise a bmb.Prior object.
    """
    lower, upper = bounds

    if isinstance(prior, float):
        if not lower <= prior <= upper:
            raise ValueError(
                f"The fixed prior should be between {lower:.4f} and {upper:.4f}, "
                + f"got {prior:.4f}."
            )

        return prior

    if isinstance(prior, dict):
        if np.isinf(lower) and np.isinf(upper):
            return bmb.Prior(**prior)
        dist = make_truncated_dist(lower, upper, **prior)
        return bmb.Prior(name=prior["name"], dist=dist)

    # After handling the constant and dict case, now handle the bmb.Prior case
    if np.isinf(lower) and np.isinf(upper):
        return prior
    if prior.dist is not None:
        return prior

    name = prior.name
    args = prior.args

    dist = make_truncated_dist(lower, upper, name=name, **args)

    return bmb.Prior(name=name, dist=dist)


def make_truncated_dist(lower_bound: float, upper_bound: float, **kwargs) -> Callable:
    """Create custom functions with truncated priors.

    Helper function that creates a custom function with truncated priors.

    Parameters
    ----------
    lower_bound
        The lower bound for the distribution.
    upper_bound
        The upper bound for the distribution.
    kwargs
        Typically a dictionary with a name for the name of the Prior distribution
        and other arguments passed to bmb.Prior object.

    Returns
    -------
    Callable
        A distribution (TensorVariable) created with pm.Truncated().
    """
    dist_name = kwargs["name"]
    dist_kwargs = {k: v for k, v in kwargs.items() if k != "name"}

    initval = dist_kwargs.pop("initval") if "initval" in dist_kwargs else None

    def TruncatedDist(name):
        dist = get_distribution(dist_name).dist(**dist_kwargs)
        return pm.Truncated(
            name=name,
            dist=dist,
            lower=lower_bound if np.isfinite(lower_bound) else None,
            upper=upper_bound if np.isfinite(upper_bound) else None,
            initval=initval,
        )

    return TruncatedDist


def _make_default_prior(bounds: tuple[float, float]) -> bmb.Prior:
    """Make a default prior from bounds.

    Parameters
    ----------
    bounds
        The (lower, upper) bounds for the default prior.

    Returns
    -------
        A bmb.Prior object representing the default prior for the provided bounds.
    """
    lower, upper = bounds
    if np.isinf(lower) and np.isinf(upper):
        return bmb.Prior("Normal", mu=0.0, sigma=2.0)
    elif np.isinf(lower) and not np.isinf(upper):
        return bmb.Prior("TruncatedNormal", mu=upper, upper=upper, sigma=2.0)
    elif not np.isinf(lower) and np.isinf(upper):
        if lower == 0:
            return bmb.Prior("HalfNormal", sigma=2.0)
        return bmb.Prior("TruncatedNormal", mu=lower, lower=lower, sigma=2.0)
    else:
        return bmb.Prior(name="Uniform", lower=lower, upper=upper)

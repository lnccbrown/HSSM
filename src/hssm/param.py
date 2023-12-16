"""The Param utility class."""

import logging
from copy import deepcopy
from typing import Any, Literal, Union, cast

import bambi as bmb
import numpy as np
import pandas as pd
from formulae import design_matrices

from .link import Link
from .prior import Prior, get_default_prior, get_hddm_default_prior

# PEP604 union operator "|" not supported by pylint
# Fall back to old syntax

# Explicitly define types so they are more expressive
# and reusable
ParamSpec = Union[float, dict[str, Any], bmb.Prior]

_logger = logging.getLogger("hssm")


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
    """

    def __init__(
        self,
        name: str | None = None,
        prior: ParamSpec | dict[str, ParamSpec] | None = None,
        formula: str | None = None,
        link: str | bmb.Link | None = None,
        bounds: tuple[float, float] | None = None,
    ):
        self.name = name
        self.prior = prior
        self.formula = formula
        self.link = link
        self.bounds = bounds
        self._is_truncated = False
        self._is_parent = False
        self._is_converted = False
        self._do_not_truncate = False
        self._link_specified = link is not None

        # Provides a convenient way to specify the link function
        if self.link == "gen_logit":
            if self.bounds is None:
                raise ValueError(
                    "Bounds must be specified for generalized log link function."
                )
            self.link = Link("gen_logit", bounds=self.bounds)

        # The initializer does not do anything immediately after the object is initiated
        # because things could still change.

    def update(self, **kwargs):
        """Update the initial information stored in the class."""
        if self._is_converted:
            raise ValueError("Cannot update the object. It has already been processed.")
        for attr, value in kwargs.items():
            if not hasattr(attr):
                raise ValueError(f"{attr} does not exist.")
            setattr(self, attr, value)

    def override_default_link(self):
        """Override the default link function.

        This is most likely because both default prior and default bounds are supplied.
        """
        self._ensure_not_converted(context="link")

        if not self.is_regression or self._link_specified:
            return  # do nothing

        if self.bounds is None:
            raise ValueError(
                (
                    "Cannot override the default link function. Bounds are not"
                    + " specified for parameter %s."
                )
                % self.name,
            )

        lower, upper = self.bounds

        if np.isneginf(lower) and np.isposinf(upper):
            return
        elif lower == 0.0 and np.isposinf(upper):
            self.link = "log"
        elif not np.isneginf(lower) and not np.isposinf(upper):
            self.link = Link("gen_logit", bounds=self.bounds)
        else:
            _logger.warning(
                "The bounds for parameter %s (%f, %f) seems strange. Nothing is done to"
                + " the link function. Please check if they are correct.",
                self.name,
                lower,
                upper,
            )

    def override_default_priors(self, data: pd.DataFrame, eval_env: dict[str, Any]):
        """Override the default priors - the general case.

        By supplying priors for all parameters in the regression, we can override the
        defaults that Bambi uses.

        Parameters
        ----------
        data
            The data used to fit the model.
        eval_env
            The environment used to evaluate the formula.
        """
        self._ensure_not_converted(context="prior")

        if not self.is_regression:
            return

        override_priors = {}
        dm = self._get_design_matrices(data, eval_env)

        has_common_intercept = False
        if dm.common is not None:
            for name, term in dm.common.terms.items():
                if term.kind == "intercept":
                    has_common_intercept = True
                    override_priors[name] = get_default_prior(
                        "common_intercept", self.bounds
                    )
                else:
                    override_priors[name] = get_default_prior("common", bounds=None)

        if dm.group is not None:
            for name, term in dm.group.terms.items():
                if term.kind == "intercept":
                    if has_common_intercept:
                        override_priors[name] = get_default_prior(
                            "group_intercept_with_common", bounds=None
                        )
                    else:
                        # treat the term as any other group-specific term
                        _logger.warning(
                            f"No common intercept. Bounds for parameter {self.name} is"
                            + " not applied due to a current limitation of Bambi."
                            + " This will change in the future."
                        )
                        override_priors[name] = get_default_prior(
                            "group_intercept", bounds=None
                        )
                else:
                    override_priors[name] = get_default_prior(
                        "group_specific", bounds=None
                    )

        if not self.prior:
            self.prior = override_priors
        else:
            prior = cast(dict[str, ParamSpec], self.prior)
            self.prior = override_priors | prior

    def override_default_priors_ddm(self, data: pd.DataFrame, eval_env: dict[str, Any]):
        """Override the default priors - the ddm case.

        By supplying priors for all parameters in the regression, we can override the
        defaults that Bambi uses.

        Parameters
        ----------
        data
            The data used to fit the model.
        eval_env
            The environment used to evaluate the formula.
        """
        self._ensure_not_converted(context="prior")
        assert self.name is not None

        if not self.is_regression:
            return

        override_priors = {}
        dm = self._get_design_matrices(data, eval_env)

        has_common_intercept = False
        if dm.common is not None:
            for name, term in dm.common.terms.items():
                if term.kind == "intercept":
                    has_common_intercept = True
                    override_priors[name] = get_hddm_default_prior(
                        "common_intercept", self.name, self.bounds
                    )
                else:
                    override_priors[name] = get_hddm_default_prior(
                        "common", self.name, bounds=None
                    )

        if dm.group is not None:
            for name, term in dm.group.terms.items():
                if term.kind == "intercept":
                    if has_common_intercept:
                        override_priors[name] = get_default_prior(
                            "group_intercept_with_common", bounds=None
                        )
                    else:
                        # treat the term as any other group-specific term
                        _logger.warning(
                            f"No common intercept. Bounds for parameter {self.name} is"
                            + " not applied due to a current limitation of Bambi."
                            + " This will change in the future."
                        )
                        override_priors[name] = get_hddm_default_prior(
                            "group_intercept", self.name, bounds=None
                        )
                else:
                    override_priors[name] = get_hddm_default_prior(
                        "group_specific", self.name, bounds=None
                    )

        if not self.prior:
            self.prior = override_priors
        else:
            prior = cast(dict[str, ParamSpec], self.prior)
            self.prior = override_priors | prior

    def _get_design_matrices(self, data: pd.DataFrame, extra_namespace: dict[str, Any]):
        """Get the design matrices for the regression.

        Parameters
        ----------
        data
            A pandas DataFrame
        eval_env
            The evaluation environment
        """
        formula = cast(str, self.formula)
        rhs = formula.split("~")[1]
        formula = "rt ~ " + rhs
        dm = design_matrices(formula, data=data, extra_namespace=extra_namespace)

        return dm

    def _ensure_not_converted(self, context=Literal["link", "prior"]):
        """Ensure that the object has not been converted."""
        if self._is_converted:
            context = "link function" if context == "link" else "priors"
            raise ValueError(
                f"Cannot override the default {context} for parameter {self.name}."
                + " The object has already been processed."
            )

    def set_parent(self):
        """Set the Param as parent."""
        self._is_parent = True

    def do_not_truncate(self):
        """Flag that prior should not be truncated.

        This is most likely because both default prior and default bounds are supplied.
        """
        self._do_not_truncate = True

    def convert(self):
        """Process the information passed to the class."""
        if self._is_converted:
            raise ValueError(
                "Cannot process the object. It has already been processed."
            )

        if self.name is None:
            raise ValueError(
                "One or more parameters do not have a name. "
                + "Please ensure that names are specified to all of them."
            )

        if self.bounds is not None:
            if any(not np.isscalar(bound) for bound in self.bounds):
                raise ValueError(f"The bounds of {self.name} should both be scalar.")
            lower, upper = self.bounds
            assert lower < upper, (
                f"The lower bound of {self.name} should be less than "
                + "its upper bound."
            )

        if isinstance(self.prior, int):
            self.prior = float(self.prior)

        if self.formula is not None:
            # The regression case
            if isinstance(self.prior, (float, bmb.Prior)):
                raise ValueError(
                    "Please specify priors for each individual parameter in the "
                    + "regression."
                )

            self.prior = (
                _make_prior_dict(self.prior) if self.prior is not None else self.prior
            )

            self.link = "identity" if self.link is None else self.link

        else:
            # The non-regression case

            if self.prior is None:
                if self.bounds is None:
                    raise ValueError(
                        f"Please specify the prior or bounds for {self.name}."
                    )
                self.prior = _make_default_prior(self.bounds)
            else:
                # Explicitly cast the type of prior, no runtime performance penalty
                self.prior = cast(ParamSpec, self.prior)

                if self.bounds is None or self._do_not_truncate:
                    if isinstance(self.prior, dict):
                        self.prior = Prior(**self.prior)
                else:
                    if isinstance(self.prior, (dict, bmb.Prior)):
                        self.prior = _make_bounded_prior(
                            self.name, self.prior, self.bounds
                        )
                        self._is_truncated = True

            if self.link is not None:
                raise ValueError("`link` should be None if no regression is specified.")

            self.link = None

        self._is_converted = True

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
        return self._is_parent

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

    def parse_bambi(
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
            left_side = "c(rt, response)" if self._is_parent else self.name

            right_side = self.formula.split("~")[1]
            right_side = right_side.strip()
            formula = f"{left_side} ~ {right_side}"

            if self.prior is not None:
                prior = {left_side: self.prior}
            link = {self.name: self.link}

            return formula, prior, link

        formula = "c(rt, response) ~ 1" if self._is_parent else None

        if self._is_parent:
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
        assert self.name is not None
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

            assert self.link is not None
            link = self.link if isinstance(self.link, str) else self.link.name
            output.append(f"    Link: {link}")

        # None regression case:
        # Output prior and bounds
        else:
            assert isinstance(self.prior, bmb.Prior)
            output.append(f"    Prior: {self.prior}")

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

    def __getitem__(self, attr):
        """Return the value of an attribute.

        Mainly a convenience function to mimic the behavior of a dict.
        """
        return getattr(self, attr)

    def __setitem__(self, attr, value):
        """Set the value of an attribute.

        Mainly a convenience function to mimic the behavior of a dict.
        """
        setattr(attr, value)


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


def _make_priors_recursive(prior: dict[str, Any]) -> Prior:
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


def _make_bounded_prior(
    param_name: str, prior: ParamSpec, bounds: tuple[float, float]
) -> float | Prior:
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
    float | Prior
        A float if `prior` is a float, otherwise a hssm.Prior object.
    """
    lower, upper = bounds

    if isinstance(prior, float):
        if not lower <= prior <= upper:
            raise ValueError(
                f"The fixed prior for {param_name} should be between {lower:.4f} and "
                + f"{upper:.4f}, got {prior:.4f}."
            )

        return prior

    if isinstance(prior, dict):
        return Prior(bounds=bounds, **prior)

    if isinstance(prior, Prior) and prior.is_truncated:
        raise ValueError(
            f"The prior that you provided for {param_name} is already truncated."
        )

    if isinstance(prior, bmb.Prior) and prior.dist is not None:
        _logger.warning(
            "The prior you have provided for %s is specified with the `dist`"
            + " argument. We assume that it's already bounded and will not apply bounds"
            + " to it.",
            param_name,
        )
        return prior

    # Handles the case where prior is bmb.Prior or prior is hssm.Prior but not
    # truncated
    name = prior.name
    args = prior.args

    return Prior(name=name, bounds=bounds, **args)


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


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Recursively merge two dictionaries."""
    merged = deepcopy(dict1)
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

"""A class for regression parameterization of HSSM models."""

import logging
import re
from typing import Any, Literal, cast

import bambi as bmb
import numpy as np
import pandas as pd
from formulae import design_matrices

from ..link import Link
from ..prior import get_default_prior, get_hddm_default_prior
from .param import Param
from .user_param import UserParam

_logger = logging.getLogger("hssm")


class RegressionParam(Param):
    """A regression parameter.

    A regression parameter is a parameter that has a formula.

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
    from_defaults
        Create a RegressionParam object with default values.
    fill_defaults
        Fill in the default values for the parameter.
    validate
        Validate the parameter.
    process_prior
        Process the prior specification.
    set_loglogit_link
        Override the default link function.
    make_safe_priors
        Override the default priors.
    reformat_formula
        Reformat the formula.
    """

    terms: list[str]

    def __init__(
        self,
        name: str,
        formula: str | None = None,
        prior: dict[str, Any] | None = None,
        link: str | Link | None = None,
        bounds: tuple[float, float] | None = None,
        user_param: UserParam | None = None,
    ) -> None:
        super().__init__(
            name, prior, formula, link, bounds=bounds, user_param=user_param
        )
        self.terms = []

    @classmethod
    def from_defaults(
        cls,
        name: str,
        formula: str,
        bounds: tuple[float, float] | None = None,
        link_settings: Literal["log_logit"] | None = None,
    ) -> "RegressionParam":
        """Create a RegressionParam object with default values."""
        param = cls(
            name,
            formula=formula,
            prior=None,
            link=None,
            bounds=bounds,
            user_param=None,
        )
        if link_settings == "log_logit":
            param.set_loglogit_link()
        return param

    def fill_defaults(
        self,
        prior: dict[str, Any] | None = None,
        bounds: tuple[float, float] | None = None,
        **kwargs,
    ) -> None:
        """Fill in the default values for the parameter.

        Parameters
        ----------
        formula
            The formula for the parameter.
        bounds
            The bounds for the parameter.
        prior_settings
            The prior settings for the parameter.
        link_settings
            The link settings for the parameter.
        """
        if prior is not None:
            raise ValueError(
                f"{self.name} is a regression parameter. "
                + "It should not have a default prior."
            )
        super().fill_defaults(prior=None, bounds=bounds)
        if self.formula is None:
            if "formula" not in kwargs:
                raise ValueError(f"Formula not specified for parameter {self.name}.")
            self.formula = kwargs["formula"]
        if kwargs.get("link_settings") == "log_logit":
            self.set_loglogit_link()

    def validate(self) -> None:
        """Validate the parameter."""
        if self.formula is None:
            raise ValueError(f"Formula not specified for parameter {self.name}.")
        self.reformat_formula()
        if isinstance(self.prior, bmb.Prior):
            raise ValueError(
                "Please specify priors for each individual parameter in the "
                f"regression for {self.name}."
            )
        if self.link is None:
            self.link = "identity"
        else:
            if self.link == "log_logit":
                self.link = Link("gen_logit", bounds=self.bounds)

    def process_prior(self) -> None:
        """Process the prior specification."""
        self.validate()
        if self.prior is None:
            return
        self.prior = cast("dict[str, Any]", self.prior)
        self.prior = _make_prior_dict(self.prior)

    def set_loglogit_link(self):
        """Override the default link function.

        This is most likely because both default prior and default bounds are supplied.
        """
        if self.link is not None:
            return

        if self.bounds is None:
            raise ValueError(
                "Cannot override the default link function. "
                f"Bounds are not specified for parameter {self.name}."
            )

        lower, upper = self.bounds

        if np.isneginf(lower) and np.isposinf(upper):
            self.link = "identity"
        elif lower == 0.0 and np.isposinf(upper):
            self.link = "log"
        elif not np.isneginf(lower) and not np.isposinf(upper):
            self.link = Link("gen_logit", bounds=self.bounds)
        else:
            _logger.warning(
                "The bounds for parameter %s (%f, %f) seem strange. Nothing is done to"
                + " the link function. Please check if they are correct.",
                self.name,
                lower,
                upper,
            )

    def make_safe_priors(
        self, data: pd.DataFrame, eval_env: dict[str, Any], is_ddm: bool
    ):
        """Override the default priors.

        By supplying priors for all parameters in the regression, we can override the
        defaults that Bambi uses.

        Parameters
        ----------
        data
            The data used to fit the model.
        eval_env
            The environment used to evaluate the formula.
        use_hddm
            Whether to use HDDM default priors.
        """
        safe_priors = {}
        dm = self._get_design_matrices(data, eval_env)

        get_prior = get_hddm_default_prior if is_ddm else get_default_prior
        specified_priors = (
            set(self.prior.keys()) if isinstance(self.prior, dict) else set()
        )

        # For each term in the design matrix, if the prior is not already specified,
        # add the default prior for that term.
        # We do this separately for common and group terms.
        # We also handle intercept and non-intercept terms separately.
        has_common_intercept = False
        if dm.common is not None:
            for name, term in dm.common.terms.items():
                self.terms.append(name)
                if name not in specified_priors:
                    if term.kind == "intercept":
                        has_common_intercept = True
                        safe_priors[name] = get_prior(
                            "common_intercept", self.name, self.bounds, self.link
                        )
                    else:
                        safe_priors[name] = get_prior(
                            "common", self.name, bounds=None, link=self.link
                        )
                else:
                    if term.kind == "intercept":
                        has_common_intercept = True

        if dm.group is not None:
            for name, term in dm.group.terms.items():
                if name not in specified_priors:
                    if term.kind == "intercept":
                        self.terms.append(name)
                        if has_common_intercept:
                            safe_priors[name] = get_prior(
                                "group_intercept_with_common",
                                self.name,
                                bounds=None,
                                link=self.link,
                            )
                        else:
                            # treat the term as any other group-specific term
                            _logger.warning(
                                f"No common intercept. Bounds for parameter {self.name}"
                                " is not applied due to a current limitation of Bambi."
                                " This will change in the future."
                            )
                            safe_priors[name] = get_prior(
                                "group_intercept",
                                self.name,
                                bounds=None,
                                link=self.link,
                            )
                    else:
                        safe_priors[name] = get_prior(
                            "group_specific", self.name, bounds=None, link=self.link
                        )
        if self.prior is not None:
            self.prior = cast("dict[str, Any]", self.prior)
            safe_priors.update(self.prior)
        self.prior = safe_priors

    def _get_design_matrices(self, data: pd.DataFrame, extra_namespace: dict[str, Any]):
        """Get the design matrices for the regression.

        Parameters
        ----------
        data
            A pandas DataFrame
        eval_env
            The evaluation environment
        """
        formula = cast("str", self.formula)
        rhs = formula.split("~")[1]
        formula = "rt ~ " + rhs
        dm = design_matrices(formula, data=data, extra_namespace=extra_namespace)
        return dm

    def reformat_formula(self):
        """Reformat the formula.

        Parameters
        ----------
        formula
            The formula to reformat.

        Returns
        -------
        str
            The reformatted formula.
        """
        formula = self.formula
        if "~" in formula:
            formula_splits = re.split(r"\s?~\s?", self.formula)
            if len(formula_splits) != 2:
                raise ValueError(
                    "The formula {formula} should contain at most one '~' character."
                )
            _, rhs = formula_splits
        else:
            rhs = self.formula
        self.formula = f"{self.name} ~ {rhs}"

    def __repr__(self) -> str:
        """Return the representation of the class.

        Returns
        -------
        str
            A string representation of the class.
        """
        if self.formula is None:
            raise ValueError(
                "Formula must be specified for regression,"
                "only exception is the parent parameter for which formula"
                "can be left undefined."
            )

        if self.prior is not None:
            if not isinstance(self.prior, dict):
                raise TypeError("The prior for a regression must be a dict.")

            priors_list: list[str] = []
            for param, prior in self.prior.items():
                between = " ~ " if isinstance(prior, (bmb.Prior, dict)) else ": "
                priors_list.append(f"        {param}{between}{prior}")
            priors = "\n".join(priors_list)
        else:
            priors = "        Unspecified. Using defaults"

        link = "identity" if self.link is None else self.link

        return (
            f"{self.name}:\n"
            f"    Formula: {self.formula}\n"
            f"    Priors:\n{priors}\n"
            f"    Link: {link}"
        )


def _make_prior_dict(
    prior: dict[str, float | dict[dict, Any] | bmb.Prior],
) -> dict[str, float | bmb.Prior]:
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
        param: (
            _make_priors_recursive(cast("dict[str, Any]", prior))
            if isinstance(prior, dict)
            else prior
        )
        for param, prior in prior.items()
    }

    return priors


def _make_priors_recursive(
    prior: dict[str, float | dict[str, Any] | bmb.Prior],
) -> bmb.Prior:
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

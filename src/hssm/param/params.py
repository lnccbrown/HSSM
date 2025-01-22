"""Params class for HSSM.

The Params class is a container for the parameters of the HSSM model. It has dict-like
behavior and stores the parameters in its `params` attribute. It is also responsible for
processing the parameters and applying user specifications.
"""

from __future__ import annotations

from collections import UserDict
from typing import TYPE_CHECKING, Any

import bambi as bmb

from .param import Param
from .regression_param import RegressionParam
from .simple_param import DefaultParam, SimpleParam
from .user_param import UserParam

if TYPE_CHECKING:
    from ..hssm import HSSM


class Params(UserDict[str, Param]):
    """Container for the parameters of the HSSM model.

    Parameters
    ----------
    params
        The parameters of the model.

    Attributes
    ----------
    parent
        The name of the parent parameter.
    parent_param
        The parent parameter object.

    Methods
    -------
    from_user_specs
        Create Params from user specifications.
    """

    def __init__(
        self,
        params: dict[str, Param],
    ) -> None:
        super().__init__(params)
        self.parent = ""
        for name, param in self.items():
            if isinstance(param, RegressionParam):
                self.parent = name
                param.is_parent = True
                break

        if self.parent == "":
            for name, param in self.items():
                if not param.is_fixed:
                    self.parent = name
                    param.is_parent = True
                    break

        if self.parent == "":
            raise ValueError("No parent parameter found.")

        self.parent_param = self[self.parent]

    @classmethod
    def from_user_specs(
        cls,
        model: HSSM,
        include: list[dict[str, Any] | UserParam],
        kwargs: dict[str, Any],
        p_outlier: float | dict | bmb.Prior | None,
    ) -> "Params":
        """Create Params from user specifications.

        Parameters
        ----------
        model
            The HSSM model.
        include
            A list of dictionaries or UserParam objects specifying the parameters.
        kwargs
            Keyword arguments specifying the parameters.
        p_outlier
            The prior specification for the outlier probability.

        Returns
        -------
        Params
            The Params object with the specified parameters.
        """
        user_params = collect_user_params(model, include, kwargs, p_outlier)
        params = make_params(model, user_params)
        return cls(params)

    def parse_bambi(
        self,
        model: HSSM,
    ) -> tuple[bmb.Formula, dict | None, dict[str, str | bmb.Link] | str]:
        """Retrieve formulas, priors, and links for bambi model building.

        Returns
        -------
        tuple
            1. A bmb.Formula object.
            2. A dictionary of priors.
            3. A dictionary of string or bmb.Link object representing the link
            function.
        """
        # Handle the edge case where list_params is empty:
        if not self.data:
            return bmb.Formula(f"{model.response_c} ~ 1"), None, "identity"

        parent_formula = None
        other_formulas = []
        priors: dict[str, Any] = {}
        links: dict[str, str | bmb.Link] = {}

        for param_name, param in self.items():
            formula, prior, link = param.parse_bambi()

            if param.is_parent:
                # parent is not a regression
                if formula is None:
                    parent_formula = f"{model.response_c} ~ 1"
                    if prior is not None:
                        priors |= {param.name: {"Intercept": prior}}
                    links[param_name] = "identity"
                # parent is a regression
                else:
                    rhs = formula.split(" ~ ")[1]
                    parent_formula = f"{model.response_c} ~ {rhs}"
                    if prior is not None:
                        priors[param_name] = prior
                    if link is not None:
                        links[param_name] = link
            else:
                if formula is not None:
                    other_formulas.append(formula)
                if prior is not None:
                    priors[param_name] = prior
                if link is not None:
                    links[param_name] = link

        if parent_formula is None:
            raise ValueError("Parent parameter not found.")
        result_formula: bmb.Formula = bmb.Formula(parent_formula, *other_formulas)
        result_priors = None if not priors else priors
        result_links: dict | str = "identity" if not links else links

        return result_formula, result_priors, result_links

    def __str__(self) -> str:
        """Return the representation of the class.

        Returns
        -------
        str
            A string representation of the class.
        """
        return self.__repr__()

    def __repr__(self):
        """Return the representation of the class.

        Returns
        -------
        str
            A string representation of the class.
        """
        params = "\n".join(repr(param) for _, param in self.items())
        return f"Parameters:\n{params}"


def collect_user_params(
    model: HSSM,
    include: list[dict[str, Any] | UserParam],
    kwargs: dict[str, Any],
    p_outlier: float | dict | bmb.Prior | None,
) -> dict[str, UserParam]:
    """Collect and convert parameters to UserParam objects.

    Parameters
    ----------
    model
        The HSSM model.
    include
        A list of dictionaries or UserParam objects specifying the parameters.
    kwargs
        Keyword arguments specifying the parameters.
    p_outlier
        The prior specification for the outlier probability.

    Returns
    -------
    dict[str, UserParam]
        A dictionary with the parameter names as keys and UserParam objects as values.
    """
    user_params: dict[str, UserParam] = {}

    # Process parameters that appear as part of the `include` kwarg in HSSM class
    for param in include:
        user_param = UserParam.from_dict(param) if isinstance(param, dict) else param
        if user_param.name is None:
            raise ValueError("Parameter name must be specified.")
        if user_param.name not in model.list_params:
            raise ValueError(
                f"Parameter {user_param.name} not found in list_params."
                " This implies that the parameter is not valid for the chosen model."
            )
        if user_param.name == "p_outlier":
            raise ValueError(
                "Please do not specify `p_outlier` in `include`. "
                "Please specify it via the `p_outlier` kwarg instead."
            )
        user_params[user_param.name] = user_param

    # Process parameters that are passed directly as kwargs to the HSSM class
    # If any of the keys is found in `list_params` it is a parameter specification.
    # We add the parameter specification to `user_params` and remove it from
    # `kwargs`
    for param_name in model.list_params:
        # Update user_params only if param_name is in kwargs
        # and not already in user_params
        if param_name in kwargs:
            if param_name in user_params:
                raise ValueError(
                    f"Parameter `{param_name}` specified in both "
                    "`include` and `kwargs`."
                )
            else:
                user_params[param_name] = UserParam.from_kwargs(
                    param_name,
                    kwargs.pop(param_name),
                )

    # Process p_outliers the same way.
    if model.has_lapse:
        user_params["p_outlier"] = UserParam.from_kwargs("p_outlier", p_outlier)

    return user_params


def make_params(model: HSSM, user_params: dict[str, UserParam]) -> dict[str, Param]:
    """Make parameters from a dict of UserParams.

    Parameters
    ----------
    model
        The HSSM model.
    user_params
        A dictionary with the parameter names as keys and UserParam objects as values.

    Returns
    -------
    dict[str, Param]
        A dictionary with the parameter names as keys and Param objects as values.
    """
    params = {}
    is_ddm = (
        model.model_name in ["ddm", "ddm_sdv", "full_ddm"]
        and model.loglik_kind != "approx_differentiable"
    )

    for name in model.list_params:
        if name in user_params:
            param = make_param_from_user_param(model, name, user_params[name])
        else:
            param = make_param_from_defaults(model, name)

        if model.prior_settings == "safe":
            if isinstance(param, RegressionParam):
                param.make_safe_priors(model.data, model.additional_namespace, is_ddm)

        param.process_prior()
        params[name] = param
    return params


def make_param_from_user_param(model: HSSM, name: str, user_param: UserParam) -> Param:
    """Create a Param object from a UserParam object.

    Parameters
    ----------
    model
        The HSSM model.
    name
        The name of the parameter.
    user_param
        A UserParam object.

    Returns
    -------
    Param
        A Param object with the specified parameters.
    """
    default_prior, default_bounds = model.model_config.get_defaults(name)

    # Use information in the UserParam object to determine if the parameter is a
    # regression parameter or a simple parameter
    if user_param.is_regression:
        param: Param = RegressionParam.from_user_param(user_param)
    elif user_param.is_simple:
        param = SimpleParam.from_user_param(user_param)
    else:
        # If we cannot tell definitively that the parameter is a regression parameter
        # or a simple parameter from the UserParam object,
        # we need to determine the type based on whether the model has a global formula
        if model.global_formula is not None:
            param = RegressionParam.from_user_param(user_param)
        else:
            param = SimpleParam.from_user_param(user_param)

    # Fill in the defaults
    if isinstance(param, RegressionParam):
        param.fill_defaults(
            formula=model.global_formula,
            bounds=default_bounds,
            link_settings=model.link_settings,
        )
    else:
        param.fill_defaults(prior=default_prior, bounds=default_bounds)

    return param


def make_param_from_defaults(model: HSSM, name: str) -> Param:
    """Create a Param object from the default values.

    Parameters
    ----------
    model
        The HSSM model.
    name
        The name of the parameter.

    Returns
    -------
    Param
        A Param object with the specified parameters.
    """
    default_prior, default_bounds = model.model_config.get_defaults(name)

    if model.global_formula is not None:
        param: Param = RegressionParam.from_defaults(
            name,
            formula=model.global_formula,
            bounds=default_bounds,
            link_settings=model.link_settings,
        )
    else:
        param = DefaultParam.from_defaults(name, default_prior, default_bounds)

    return param

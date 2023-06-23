"""Utility functions and classes.

HSSM has to reconcile with two representations: it's own representation as an HSSM and
the representation acceptable to Bambi. The two are not equivalent. This file contains
the Param class that reconcile these differences.

The Param class is an abstraction that stores the parameter specifications and turns
these representations in Bambi-compatible formats through convenience function
_parse_bambi().
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, NewType, Union, cast

import bambi as bmb
import pymc as pm
from bambi.backend.utils import get_distribution
from bambi.terms import CommonTerm, GroupSpecificTerm
from pymc.model_graph import ModelGraph
from pytensor import function
from huggingface_hub import hf_hub_download

# PEP604 union operator "|" not supported by pylint
# Fall back to old syntax

# Explicitly define types so they are more expressive
# and reusable
ParamSpec = Union[float, dict[str, Any], bmb.Prior]
BoundsSpec = tuple[float, float]
REPO_ID = "franklab/HSSM"


def download_hf(path: str):
    """
    Download a file from a HuggingFace repository.

    Parameters
    ----------
    path : str
        The path of the file to download in the repository.

    Returns
    -------
    str
        The local path where the file is downloaded.

    Notes
    -----
    The repository is specified by the REPO_ID constant,
    which should be a valid HuggingFace.co repository ID.
    The file is downloaded using the HuggingFace Hub's
     hf_hub_download function.
    """
    return hf_hub_download(repo_id=REPO_ID, filename=path)


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Recursively merge two dictionaries."""
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


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
        self.bounds = bounds

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
                if bounds is None:
                    raise ValueError(
                        f"Please specify the prior or bounds for {self.name}."
                    )
                self.prior = bmb.Prior(name="Uniform", lower=bounds[0], upper=bounds[1])
            else:
                # Explicitly cast the type of prior, no runtime performance penalty
                prior = cast(ParamSpec, prior)
                self.prior = make_bounded_prior(prior, bounds)

            if link is not None:
                raise ValueError("`link` should be None if no regression is specified.")

            self.link = None

    @property
    def is_regression(self) -> bool:
        """Determines if a regression is specified or not.

        Returns
        -------
            A boolean that indicates if a regression is specified.
        """
        return self.formula is not None

    @property
    def is_parent(self) -> bool:
        """Determines if a parameter is a parent parameter for Bambi.

        Returns
        -------
            A boolean that indicates if the parameter is a parent or not.
        """
        return self._parent

    def _parse_bambi(
        self,
    ) -> tuple:
        """
        Return a 3-tuple that helps with constructing the Bambi model.

        Returns
        -------
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
            A string whose construction depends on whether the specification contains a
            regression or not.
        """
        if not self.is_regression:
            if isinstance(self.prior, bmb.Prior):
                if self.bounds is None:
                    return f"{self.name} ~ {self.prior}"
                return f"{self.name} ~ {self.prior}\tbounds: {self.bounds}"
            return f"{self.name} = {self.prior}"

        link = self.link if isinstance(self.link, str) else self.link.name

        assert not isinstance(self.prior, float)
        assert self.formula is not None

        priors = (
            "\r\n".join([f"\t{param} ~ {prior}" for param, prior in self.prior.items()])
            if self.prior is not None
            else "Unspecified, using defaults"
        )

        if self.bounds is not None:
            return "\r\n".join(
                [self.formula, f"\tLink: {link}", f"\tbounds: {self.bounds}", priors]
            )
        return "\r\n".join([self.formula, f"\tLink: {link}", priors])

    def __str__(self) -> str:
        """Return the string representation of the class.

        Returns
        -------
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

    result_links: dict | str = "identity" if not links else links

    return result_formula, result_priors, result_links


def make_alias_dict_from_parent(parent: Param) -> dict[str, str]:
    """Make aliases from the parent parameter.

    From a Param object that represents a parent parameter in Bambi,
    returns a dict that represents how Bambi should alias its parameters to
    make it more HSSM-friendly.

    Parameters
    ----------
        parent: A Param object that represents a parent parameter.

    Returns
    -------
        A dict that indicates how Bambi should alias its parameters.
    """
    assert parent.is_parent, "This Param object should be a parent!"

    result_dict = {"c(rt, response)": "rt,response"}

    # The easy case. We will just alias "Intercept" as the actual name of the
    # parameter
    if not parent.is_regression:
        result_dict |= {"Intercept": parent.name}

        return result_dict

    # The regression case:
    # In this case, the name of the response variable should actually be
    # the name of the parent parameter
    result_dict["c(rt, response)"] = parent.name

    return result_dict


def get_alias_dict(model: bmb.Model, parent: Param) -> dict[str, str | dict]:
    """Make a list of aliases.

    Iterates through a list of Param objects, and aliases a Bambi model's parameters
    to make it more HSSM-friendly.

    Parameters
    ----------
    model
        A Bambi model.
    parent
        The Param representation of the parent parameter.

    Returns
    -------
        A dict that indicates how Bambi should alias its parameters.
    """
    parent_name = parent.name

    if len(model.distributional_components) == 1:
        alias_dict: dict[str, Any] = {"c(rt, response)": "rt,response"}
        if not parent.is_regression:
            alias_dict |= {"Intercept": parent_name}
        else:
            for name, term in model.response_component.terms.items():
                if isinstance(term, (CommonTerm, GroupSpecificTerm)):
                    alias_dict |= {name: f"{parent_name}_{name}"}
    else:
        alias_dict = {"c(rt, response)": {"c(rt, response)": "rt,response"}}
        for component_name, component in model.distributional_components.items():
            if component.response_kind == "data":
                if not parent.is_regression:
                    alias_dict["c(rt, response)"] |= {"Intercept": parent_name}
                else:
                    for name, term in model.response_component.terms.items():
                        if isinstance(term, CommonTerm):
                            alias_dict["c(rt, response)"] |= {
                                name: f"{parent_name}_{name}"
                            }
            else:
                name = f"rt,response_{component_name}"
                alias_dict[component_name] = {name: component_name}

    for name in model.constant_components.keys():
        alias_dict |= {name: name}

    return alias_dict


def fast_eval(var):
    """Fast evaluation of a variable.

    Notes
    -----
    This is a helper function required for one of the functions below.
    """
    return function([], var, mode="FAST_COMPILE")()


VarName = NewType("VarName", str)


class HSSMModelGraph(ModelGraph):
    """Customize PyMC's ModelGraph class to inject the missing parent parameter.

    Notes
    -----
    This is really a hack. There might be better ways to get around the
    parent parameter issue.
    """

    def __init__(self, model, parent):
        self.parent = parent
        super().__init__(model)

    def make_graph(
        self, var_names: Iterable[VarName] | None = None, formatting: str = "plain"
    ):
        """Make graphviz Digraph of PyMC model.

        Returns
        -------
            graphviz.Digraph

        Notes
        -----
            This is a slightly modified version of the code in:
            https://github.com/pymc-devs/pymc/blob/main/pymc/model_graph.py

            Credit for this code goes to PyMC developers.
        """
        try:
            import graphviz  # pylint: disable=C0415
        except ImportError as e:
            e.msg = (
                "This function requires the python library graphviz, "
                + "along with binaries. "
                + "The easiest way to install all of this is by running\n\n"
                + "\tconda install -c conda-forge python-graphviz"
            )
            raise e
        graph = graphviz.Digraph(self.model.name)
        for plate_label, all_var_names in self.get_plates(var_names).items():
            if plate_label:
                # must be preceded by 'cluster' to get a box around it
                with graph.subgraph(name="cluster" + plate_label) as sub:
                    for var_name in all_var_names:
                        self._make_node(var_name, sub, formatting=formatting)
                    # plate label goes bottom right
                    sub.attr(
                        label=plate_label,
                        labeljust="r",
                        labelloc="b",
                        style="rounded",
                    )

            else:
                for var_name in all_var_names:
                    self._make_node(var_name, graph, formatting=formatting)

        if self.parent.is_regression:
            # Insert the parent parameter that's not included in the graph
            with graph.subgraph(name="cluster" + self.parent.name) as sub:
                sub.node(
                    self.parent.name,
                    label=f"{self.parent.name}\n~\nDeterministic",
                    shape="box",
                )
                shape = fast_eval(self.model["rt,response"].shape)
                plate_label = f"rt,response_obs({shape[0]})"

                sub.attr(
                    label=plate_label,
                    labeljust="r",
                    labelloc="b",
                    style="rounded",
                )

        for child, parents in self.make_compute_graph(var_names=var_names).items():
            # parents is a set of rv names that precede child rv nodes
            for parent in parents:
                if (
                    self.parent.is_regression
                    and parent.startswith(f"{self.parent.name}_")
                    and child == "rt,response"
                ):
                    # Modify the edges so that they point to the
                    # parent parameter
                    graph.edge(parent.replace(":", "&"), self.parent.name)
                else:
                    graph.edge(parent.replace(":", "&"), child.replace(":", "&"))

        if self.parent.is_regression:
            graph.edge(self.parent.name, "rt,response")

        return graph


def make_bounded_prior(
    prior: ParamSpec, bounds: BoundsSpec | None
) -> float | bmb.Prior:
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
    bounds, optional
        If provided, needs to be a tuple of floats that indicates the lower and upper
        bounds of the parameter.

    Returns
    -------
        A float if `prior` is a float, otherwise a bmb.Prior object.
    """
    if bounds is None:
        return bmb.Prior(**prior) if isinstance(prior, dict) else prior

    lower, upper = bounds

    if isinstance(prior, float):
        if not lower <= prior <= upper:
            raise ValueError(
                f"The fixed prior should be between {lower:.4f} and {upper:.4f}, "
                + f"got {prior:.4f}."
            )

        return prior

    if isinstance(prior, dict):
        dist = make_truncated_dist(lower, upper, **prior)
        return bmb.Prior(name=prior["name"], dist=dist)

    # After handling the constant and dict case, now handle the bmb.Prior case
    if prior.dist is not None:
        return prior

    name = prior.name
    args = prior.args

    dist = make_truncated_dist(lower, upper, name=name, **args)
    prior.update(dist=dist)

    return prior


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
        A distribution (TensorVariable) created with pm.Truncated().
    """
    dist_name = kwargs["name"]
    dist_kwargs = {k: v for k, v in kwargs.items() if k != "name"}

    def TruncatedDist(name):
        dist = get_distribution(dist_name).dist(**dist_kwargs)
        return pm.Truncated(
            name="Trucated_" + name,
            dist=dist,
            lower=lower_bound,
            upper=upper_bound,
        )

    return TruncatedDist

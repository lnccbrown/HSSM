"""
HSSM has to reconcile with two representations: it's own representation as an HSSM and
the representation acceptable to Bambi. The two are not equivalent. This file contains
the Param class that reconcile these differences.

The Param class is an abstraction that stores the parameter specifications and turns
these representations in Bambi-compatible formats through convenience function
_parse_bambi().
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, NewType, Tuple

import bambi as bmb
from bambi.terms import CommonTerm, GroupSpecificTerm
from pymc.model_graph import ModelGraph
from pytensor import function

VarName = NewType("VarName", str)


def fast_eval(var):
    return function([], var, mode="FAST_COMPILE")()


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
            "\r\n".join([f"\t{param} ~ {prior}" for param, prior in self.prior.items()])
            if self.prior is not None
            else "Unspecified, using defaults"
        )

        return "\r\n".join([self.formula, f"\tLink: {link}", priors])  # type: ignore

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
            2. A dict of priors, if any is specified.
            3. A dict of link functions, if any is specified.
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


def make_alias_dict_from_parent(parent: Param) -> Dict[str, str]:
    """From a Param object that represents a parent parameter in Bambi,
    returns a dict that represents how Bambi should alias its parameters to
    make it more HSSM-friendly.

    Parameters
    ----------
    parent
        A param object that represents a parent parameter.

    Returns
    -------
        A dict that indicates how Bambi should alias its parameters.
    """

    assert parent.is_parent(), "This Param object should be a parent!"

    result_dict = {"c(rt, response)": "rt, response"}

    # The easy case. We will just alias "Intercept" as the actual name of the
    # parameter
    if not parent.is_regression():
        result_dict |= {"Intercept": parent.name}

        return result_dict

    # The regression case:
    # In this case, the name of the response variable should actually be
    # the name of the parent parameter
    result_dict["c(rt, response)"] = parent.name

    return result_dict


def get_alias_dict(model: bmb.Model, parent: Param) -> Dict[str, str | Dict]:
    """Iterates through a list of Param objects, and aliases a Bambi model's parameters
    to make it more HSSM-friendly.

    Parameters
    ----------
    model
        A Bambi model
    params
        The Param representation of the parent parameter
    """

    parent_name = parent.name

    if len(model.distributional_components) == 1:
        alias_dict: Dict[str, str | Dict] = {
            "c(rt, response)": "rt, response"
        }  # type: ignore
        if not parent.is_regression():
            alias_dict |= {"Intercept": parent_name}
        else:
            for name, term in model.response_component.terms.items():
                if isinstance(term, (CommonTerm, GroupSpecificTerm)):
                    alias_dict |= {name: f"{parent_name}_{name}"}
    else:
        alias_dict = {
            "c(rt, response)": {"c(rt, response)": "rt, response"}
        }  # type: ignore
        for component_name, component in model.distributional_components.items():
            if component.response_kind == "data":
                if not parent.is_regression():
                    alias_dict["c(rt, response)"] |= {  # type: ignore
                        "Intercept": parent_name
                    }
                else:
                    for name, term in model.response_component.terms.items():
                        if isinstance(term, CommonTerm):
                            alias_dict["c(rt, response)"] |= {  # type: ignore
                                name: f"{parent_name}_{name}"
                            }
            else:
                name = f"rt, response_{component_name}"
                alias_dict[component_name] = {name: component_name}  # type: ignore

    for name in model.constant_components.keys():
        alias_dict |= {name: name}

    return alias_dict


class HSSMModelGraph(ModelGraph):
    """Customize PyMC's ModelGraph class to inject the missing parent parameter
    into the graph.

    NOTE: this is really a hack. There might be better ways to get around the
    parent parameter issue.
    """

    def __init__(self, model, parent):
        self.parent = parent
        super().__init__(model)

    def make_graph(
        self, var_names: Iterable[VarName] | None = None, formatting: str = "plain"
    ):
        """Make graphviz Digraph of PyMC model

        NOTE: Directly ganked and modified from
        https://github.com/pymc-devs/pymc/blob/main/pymc/model_graph.py

        Credit for this code goes to PyMC developers.

        Returns
        -------
        graphviz.Digraph
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

        if self.parent.is_regression():
            # Insert the parent parameter that's not included in the graph
            with graph.subgraph(name="cluster" + self.parent.name) as sub:
                sub.node(
                    self.parent.name,
                    label=f"{self.parent.name}\n~\nDeterministic",
                    shape="box",
                )
                shape = fast_eval(self.model["rt, response"].shape)
                plate_label = f"rt, response_obs({shape[0]})"

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
                    and child == "rt, response"
                ):
                    # Modify the edges so that they point to the
                    # parent parameter
                    graph.edge(parent.replace(":", "&"), self.parent.name)
                else:
                    graph.edge(parent.replace(":", "&"), child.replace(":", "&"))

        if self.parent.is_regression():
            graph.edge(self.parent.name, "rt, response")

        return graph

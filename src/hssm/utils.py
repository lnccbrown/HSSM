"""Utility functions and classes.

HSSM has to reconcile with two representations: it's own representation as an HSSM and
the representation acceptable to Bambi. The two are not equivalent. This file contains
the Param class that reconcile these differences.

The Param class is an abstraction that stores the parameter specifications and turns
these representations in Bambi-compatible formats through convenience function
_parse_bambi().
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterable, Literal, NewType

import bambi as bmb
import pytensor
from bambi.terms import CommonTerm, GroupSpecificTerm, HSGPTerm, OffsetTerm
from huggingface_hub import hf_hub_download
from jax.config import config
from pymc.model_graph import ModelGraph
from pytensor import function

if TYPE_CHECKING:
    from .param import Param

_logger = logging.getLogger("hssm")

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
    merged = deepcopy(dict1)
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


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
    dict[str, str | dict]
        A dict that indicates how Bambi should alias its parameters.
    """
    parent_name = parent.name

    if len(model.distributional_components) == 1:
        alias_dict: dict[str, Any] = {"c(rt, response)": "rt,response"}
        if not parent.is_regression:
            alias_dict |= {"Intercept": parent_name}
        else:
            for name, term in model.response_component.terms.items():
                if isinstance(
                    term, (CommonTerm, GroupSpecificTerm, HSGPTerm, OffsetTerm)
                ):
                    alias_dict |= {name: f"{parent_name}_{name}"}
    else:
        alias_dict = {"c(rt, response)": {"c(rt, response)": "rt,response"}}
        for component_name, component in model.distributional_components.items():
            if component.response_kind == "data":
                if not parent.is_regression:
                    alias_dict["c(rt, response)"] |= {"Intercept": parent_name}
                else:
                    for name, term in model.response_component.terms.items():
                        if isinstance(
                            term, (CommonTerm, GroupSpecificTerm, HSGPTerm, OffsetTerm)
                        ):
                            alias_dict["c(rt, response)"] |= {
                                name: f"{parent_name}_{name}"
                            }
            else:
                alias_dict[component_name] = {component_name: component_name}

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


def set_floatX(dtype: Literal["float32", "float64"], jax: bool = True):
    """Set float types for pytensor and Jax.

    Often we wish to work with a specific type of float in both PyTensor and JAX.
    This function helps set float types in both packages.

    Parameters
    ----------
    dtype
        Either `float32` or `float64`. Float type for pytensor (and jax if `jax=True`).
    jax : optional
        Whether this function also sets float type for JAX by changing the
        `jax_enable_x64` setting in JAX config. Defaults to True.
    """
    if dtype not in ["float32", "float64"]:
        raise ValueError('`dtype` must be either "float32" or "float64".')

    pytensor.config.floatX = dtype
    _logger.info(f"Setting PyTensor floatX type to {dtype}.")

    if jax:
        mapping = dict(float32=False, float64=True)
        config.update("jax_enable_x64", mapping[dtype])

        _logger.info(
            f'Setting "jax_enable_x64" to {mapping[dtype]}. '
            + "If this is not intended, please set `jax` to False."
        )


def _print_prior(term: CommonTerm | GroupSpecificTerm) -> str:
    """Make the output string of a term.

    If prior is a float, print x: prior. Otherwise, print x ~ prior.

    Parameters
    ----------
    term
        A BaseTerm in Bambi

    Returns
    -------
        A string representing the term_name ~ prior pair
    """
    term_name = term.alias or term.name
    prior = term._prior

    if isinstance(prior, float):
        return f"        {term_name}: {prior}"

    return f"        {term_name} ~ {prior}"


def _process_param_in_kwargs(name, prior: float | dict | bmb.Prior) -> dict:
    """Process parameters specified in kwargs.

    Parameters
    ----------
    name
        The name of the parameters.
    prior
        The prior specified.

    Returns
    -------
    dict
        A `dict` that complies with ways to specify parameters in `include`.

    Raises
    ------
    ValueError
        When `prior` is not a `float`, a `dict`, or a `b`mb.Prior` object.
    """
    if isinstance(prior, (int, float, bmb.Prior)):
        return {"name": name, "prior": prior}
    elif isinstance(prior, dict):
        if ("prior" in prior) or ("bounds" in prior):
            return prior | {"name": name}
        else:
            return {"name": name, "prior": prior}
    else:
        raise ValueError(
            f"Parameter {name} must be a float, a dict, or a bmb.Prior object."
        )

from __future__ import annotations

from typing import Callable, Literal

import bambi as bmb
import pandas as pd
import pymc as pm
import pytensor
from numpy.typing import ArrayLike

from hssm import wfpt
from hssm.utils import HSSMModelGraph, Param, _parse_bambi, get_alias_dict, merge_dicts
from hssm.wfpt.config import default_model_config

LogLikeFunc = Callable[..., ArrayLike]

# add custom link function
class HSSM:  # pylint: disable=R0902
    """
    The Hierarchical Sequential Sampling Model (HSSM) class.

    Args:
    data (pandas.DataFrame): A pandas DataFrame with the minimum requirements of
        containing the data with the columns 'rt' and 'response'.
    model_name (str): The name of the model to use. Default is "ddm".
    Current default implementations are "ddm" | "lan" | "custom".
        ddm - Computes the log-likelihood of the drift diffusion model f(t|v,a,z) using
        the method and implementation of Navarro & Fuss, 2009.
        angle - Likelihood Approximation Network (LAN) extension for the Wiener
         First-Passage Time (WFPT) distribution.
    include (List[dict], optional): A list of dictionaries specifying additional
     parameters to include in the model. Defaults to None. Priors specification range:
        v: Mean drift rate. (-inf, inf).
        sv: Standard deviation of the drift rate [0, inf).
        a: Value of decision upper bound. (0, inf).
        z: Normalized decision starting point. (0, 1).
        t: Non-decision time [0, inf).
        theta: [0, inf).
    model_config (dict, optional): A dictionary containing the model
     configuration information. Defaults to None.

    Attributes:
        list_params (list): The list of parameter names.
        model (str): The name of the model.
        model_distribution: A SSM model object.
        likelihood: A Bambi likelihood object.
        family: A Bambi family object.
        priors (dict): A dictionary containing the prior distribution
         of parameters.
        formula (str): A string representing the model formula.
        params (list): A list of Param objects representing model parameters.

    Methods:
        _transform_params: A method to transform priors, link and formula
         into Bambi's format.
        sample: A method to sample posterior distributions.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model: str = "ddm",
        include: list[dict] | None = None,
        model_config: dict | None = None,
        loglik: LogLikeFunc | pytensor.graph.Op | None = None,
    ):
        self.data = data
        self._inference_obj = None

        if model == "custom":
            if model_config is None:
                raise ValueError(
                    "For custom models, please provide a correct model config."
                )
            self.model_config = model_config
        else:
            if model not in default_model_config:
                supported_models = list(default_model_config.keys())
                raise ValueError(
                    f"`model` must be one of {supported_models} or 'custom'."
                )
            self.model_config = (
                default_model_config[model]
                if model_config is None
                else merge_dicts(default_model_config[model], model_config)
            )

        self.model_name = model

        self.list_params = self.model_config["list_params"]
        self.parent = self.list_params[0]

        self.params, self.formula, self.priors, self.link = self._transform_params(
            include
        )
        params_is_reg = [param.is_regression() for param in self.params]

        self.is_onnx = self.model_config["loglik_kind"] == "approx_differentiable"
        boundaries = self.model_config["default_boundaries"]
        if self.model_config["loglik_kind"] == "analytical":
            self.model_distribution = wfpt.WFPT
        elif self.is_onnx:
            self.model_distribution = wfpt.make_lan_distribution(
                model=self.model_config["loglik_path"],
                list_params=self.list_params,
                backend=self.model_config["backend"],
                params_is_reg=params_is_reg,
                boundaries=boundaries,
            )
        elif self.model_name == "custom":
            self.model_distribution = wfpt.make_distribution(
                loglik=loglik,  # type: ignore
                list_params=self.list_params,  # type: ignore
                boundaries=boundaries,
            )

        self.likelihood = bmb.Likelihood(
            self.model_config["loglik_kind"],
            params=self.list_params,
            parent=self.model_config["list_params"][0],
            dist=self.model_distribution,
        )

        self.family = bmb.Family(
            self.model_config["loglik_kind"], likelihood=self.likelihood, link=self.link
        )

        self.model = bmb.Model(
            self.formula, data, family=self.family, priors=self.priors
        )

        self._aliases = get_alias_dict(self.model, self.parent_param)
        self.set_alias(self._aliases)

    def _transform_params(
        self, include: list[dict] | None
    ) -> tuple[list[Param], bmb.Formula, dict | None, dict[str, str | bmb.Link] | str]:
        """
        This function transforms a list of dictionaries containing parameter
        information into a list of Param objects. It also creates a formula, priors,
        and a link for the Bambi package based on the parameters.

        Parameters
        ----------
        include:
            A list of dictionaries containing information about the parameters.

        Returns
        -------
            A tuple of 4 items, the latter 3 are for creating the bambi model.
            - A list of the same length as self.list_params containing Param objects.
            - A bmb.formula object.
            - An optional dict containing prior information for Bambi.
            - An optional dict of link functions for Bambi.
        """
        processed = []
        params = []
        if include:
            for param_dict in include:
                processed.append(param_dict["name"])
                is_parent = param_dict["name"] == self.parent
                param = Param(is_parent=is_parent, **param_dict)
                params.append(param)
                if is_parent:
                    self.parent_param = param

        for param_str in self.list_params:
            if param_str not in processed:
                is_parent = param_str == self.parent
                param = Param(
                    name=param_str,
                    prior=self.model_config["default_prior"][param_str],
                    is_parent=is_parent,
                )
                params.append(param)
                if is_parent:
                    self.parent_param = param

        if len(params) != len(self.list_params):
            raise ValueError("Please provide a correct set of priors")

        return params, *_parse_bambi(params)

    def sample(
        self,
        sampler: Literal[
            "mcmc", "nuts_numpyro", "nuts_blackjax", "laplace", "vi"
        ] = "mcmc",
        **kwargs,
    ):
        """Performs sampling using the `fit` method via bambi.Model.

        Parameters
        ----------

        sampler
            The sampler to use. Can be either "mcmc" (default), "nuts_numpyro",
            "nuts_blackjax", "laplace", or "vi".
        kwargs
            Other arguments passed to bmb.Model.fit()

        Returns
        -------
            An ArviZ ``InferenceData`` instance if inference_method is  ``"mcmc"``
            (default), "nuts_numpyro", "nuts_blackjax" or "laplace".
            An ``Approximation`` object if  ``"vi"``.
        """

        self._inference_obj = self.model.fit(inference_method=sampler, **kwargs)

        return self.trace

    @property
    def pymc_model(self) -> pm.Model:
        """A convenience funciton that returns the PyMC model build by bambi,
        largely to avoid stuff like self.model.backend.model...

        Returns
        -------
            The PyMC model built by bambi
        """

        return self.model.backend.model

    def set_alias(self, aliases: dict[str, str | dict]):
        """Sets the aliases according to the dictionary passed to it and rebuild the
        model.

        Parameters
        ----------
        alias
            A dict specifying the paramter names being aliased and the aliases.
        """

        self.model.set_alias(aliases)
        self.model.build()

    # NOTE: can't annotate return type because the graphviz dependency is optional
    def graph(self, formatting="plain", name=None, figsize=None, dpi=300, fmt="png"):
        """Produce a graphviz Digraph from a built HSSM model.
        Requires graphviz, which may be installed most easily with
            ``conda install -c conda-forge python-graphviz``
        Alternatively, you may install the ``graphviz`` binaries yourself, and then
        ``pip install graphviz`` to get the python bindings.
        See http://graphviz.readthedocs.io/en/stable/manual.html for more information.

        The code is largely copied from
        https://github.com/bambinos/bambi/blob/main/bambi/models.py
        Credit for the code goes to Bambi developers.

        Parameters
        ----------
        formatting
            One of ``"plain"`` or ``"plain_with_params"``. Defaults to ``"plain"``.
        name
            Name of the figure to save. Defaults to ``None``, no figure is saved.
        figsize
            Maximum width and height of figure in inches. Defaults to ``None``, the
            figure size is set automatically. If defined and the drawing is larger than
            the given size, the drawing is uniformly scaled down so that it fits within
            the given size.  Only works if ``name`` is not ``None``.
        dpi
            Point per inch of the figure to save.
            Defaults to 300. Only works if ``name`` is not ``None``.
        fmt
            Format of the figure to save.
            Defaults to ``"png"``. Only works if ``name`` is not ``None``.

        Returns
            The graph
        """

        self.model._check_built()

        graphviz = HSSMModelGraph(
            model=self.pymc_model, parent=self.parent_param
        ).make_graph(formatting=formatting)

        width, height = (None, None) if figsize is None else figsize

        if name is not None:
            graphviz_ = graphviz.copy()
            graphviz_.graph_attr.update(size=f"{width},{height}!")
            graphviz_.graph_attr.update(dpi=str(dpi))
            graphviz_.render(filename=name, format=fmt, cleanup=True)

            return graphviz_

        return graphviz

    def __repr__(self) -> str:
        """Creates a representation of the model."""

        output = []

        output.append("Hierarchical Sequential Sampling Model")
        output.append(f"Model: {self.model_name}")
        output.append("")

        output.append("Response variable: rt, response")
        output.append(f"Observations: {len(self.data)}")
        output.append("")

        output.append("Parameters:")
        output.append("")

        for param in self.params:
            output.append(str(param))

        return "\r\n".join(output)

    def __str__(self) -> str:
        """Creates a string representation of the model."""

        return self.__repr__()

    @property
    def trace(self):

        if not self._inference_obj:
            raise ValueError("Please sample the model first.")

        return self._inference_obj

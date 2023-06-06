from __future__ import annotations

from typing import Any, Callable, Literal

import arviz as az
import bambi as bmb
import pandas as pd
import pymc as pm
import pytensor
from numpy.typing import ArrayLike

from hssm import wfpt
from hssm.utils import HSSMModelGraph, Param, _parse_bambi, get_alias_dict, merge_dicts
from hssm.wfpt.config import Config, SupportedModels, default_model_config, download_hf

LogLikeFunc = Callable[..., ArrayLike]


class HSSM:
    """
    The Hierarchical Sequential Sampling Model (HSSM) class.

    Parameters
    ----------

    data:
        A pandas DataFrame with the minimum requirements of containing the data with the
        columns 'rt' and 'response'.
    model:
        The name of the model to use. Currently supported models are "ddm", "angle",
        "levy", "ornstein", "weibull", "race_no_bias_angle_4", "ddm_seq2_no_bias". If
        using a custom model, please pass "custom". Defaults to "ddm".
    include, optional:
        A list of dictionaries specifying parameter specifications to include in the
        model. If left unspecified, defaults will be used for all parameter
        specifications. Defaults to None.
    model_config, optional:
        A dictionary containing the model configuration information. If None is
        provided, defaults will be used. Defaults to None.
    **kwargs:
        Additional arguments passed to the bmb.Model object.

    Attributes
    ----------
    data:
        A pandas DataFrame with at least two columns of "rt" and "response" indicating
        the response time and responses.
    list_params:
        The list of strs of parameter names.
    model_name:
        The name of the model.
    model_config:
        A dictionary representing the model configuration.
    model_distribution:
        The likelihood function of the model in the form of a pm.Distribution subclass.
    family:
        A Bambi family object.
    priors:
        A dictionary containing the prior distribution of parameters.
    formula:
        A string representing the model formula.
    link:
        A string or a dictionary representing the link functions for all parameters.
    params:
        A list of Param objects representing model parameters.

    Methods:
        sample: A method to sample posterior distributions.
        set_alias: Sets the alias for a paramter.
        graph: Plot the model with PyMC's built-in graph function.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model: SupportedModels = "ddm",
        include: list[dict] | None = None,
        likelihood_type: str = None,
        model_config: Config | None = None,
        loglik_path: str | None = None,
        loglik: LogLikeFunc | pytensor.graph.Op | None = None,
        **kwargs,
    ):
        self.data = data
        self._inference_obj = None

        if model == "custom":
            if model_config:
                self.model_config = model_config
            else:
                if likelihood_type is None and loglik is None:
                    raise ValueError(
                        "For custom models, both `likelihood_type` and `loglik` must be provided."
                    )
                if likelihood_type == "analytical":
                    self.model_config = default_model_config["custom_analytical"]
                    self.model_config["loglik"] = loglik
                elif likelihood_type == "approx_differentiable":
                    self.model_config = default_model_config["custom_angle"]
                    self.model_config["loglik_path"] = loglik
                if not self.model_config:
                    raise ValueError("Invalid custom model configuration.")
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

        if loglik_path:
            self.model_config["loglik_path"] = download_hf(loglik_path)
        self.model_name = model
        self.list_params = self.model_config["list_params"]
        self._parent = self.list_params[0]

        if include is None:
            include = []
        params_in_include = [param["name"] for param in include]

        other_kwargs: dict[Any, Any] = {}
        for k, v in kwargs.items():
            if k in self.list_params:
                if k in params_in_include:
                    raise ValueError(
                        f'Parameter "{k}" is already specified in `include`.'
                    )

                if isinstance(v, (int, float, bmb.Prior)):
                    include.append({"name": k, "prior": v})
                elif isinstance(v, dict):
                    include.append(v | {"name": k})
                else:
                    raise ValueError(
                        f"Parameter {k} must be a float, a dict, or a bmb.Prior object."
                    )
            else:
                other_kwargs |= {k: v}

        self.params, self.formula, self.priors, self.link = self._transform_params(
            include, self.model_name, self.model_config
        )

        for param in self.params:
            if param.name == self._parent:
                self._parent_param = param
                break

        assert self._parent_param is not None

        params_is_reg = [param.is_regression for param in self.params]

        if "loglik_kind" not in self.model_config or self.model_config[
            "loglik_kind"
        ] not in [
            "analytical",
            "approx_differentiable",
            "blackbox",
        ]:
            raise ValueError(
                "'loglike_kind' field of model_config must be one of "
                + '"analytical", "approx_differentiable", "blackbox".'
            )

        if "loglik" in self.model_config:
            # If a user has already provided a log-likelihood function
            if issubclass(self.model_config["loglik"], pm.Distribution):
                # Test if the it is a distribution
                self.model_distribution = self.model_config["loglik"]
            else:
                # If not, create a distribution
                self.model_distribution = wfpt.make_distribution(
                    loglik=loglik, list_params=self.list_params  # type: ignore
                )
        else:
            # If not, in the case of "approx_differentiable"
            if self.model_config["loglik_kind"] == "approx_differentiable":
                # Check if a loglik_path is provided.
                if (
                    "loglik_path" not in self.model_config
                    or self.model_config["loglik_path"] is None
                ):
                    raise ValueError(
                        "Please provide either a path to an onnx file for the log "
                        + "likelihood or a log-likelihood function."
                    )
                self.model_distribution = wfpt.make_lan_distribution(
                    model=self.model_config["loglik_path"],
                    list_params=self.list_params,
                    backend=self.model_config["backend"],
                    params_is_reg=params_is_reg,
                )
            else:
                raise ValueError(
                    "Please provide a likelihood function or a pm.Distribution "
                    + "in the `loglik` field of model_config!"
                )

        assert self.model_distribution is not None

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
            self.formula, data, family=self.family, priors=self.priors, **other_kwargs
        )

        self._aliases = get_alias_dict(self.model, self._parent_param)
        self.set_alias(self._aliases)

    def _transform_params(
        self, include: list[dict] | None, model: str, model_config: Config
    ) -> tuple[list[Param], bmb.Formula, dict | None, dict[str, str | bmb.Link] | str]:
        """
        This function transforms a list of dictionaries containing parameter
        information into a list of Param objects. It also creates a formula, priors,
        and a link for the Bambi package based on the parameters.

        Parameters
        ----------
        include:
            A list of dictionaries containing information about the parameters.
        model:
            A string that indicates the type of the model.
        model_config:
            A dict for the configuration for the model.

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
                is_parent = param_dict["name"] == self._parent
                param = Param(
                    bounds=model_config["bounds"][param_dict["name"]],
                    is_parent=is_parent,
                    **param_dict,
                )
                params.append(param)

        for param_str in self.list_params:
            if param_str not in processed:
                is_parent = param_str == self._parent
                bounds = model_config["bounds"][param_str]
                prior = 0.0 if model == "ddm" and param_str == "sv" else None
                param = Param(
                    name=param_str,
                    prior=prior,
                    bounds=bounds,
                    is_parent=is_parent,
                )
                params.append(param)

        if len(params) != len(self.list_params):
            raise ValueError("Please provide a correct set of priors")

        return params, *_parse_bambi(params)

    def sample(
        self,
        sampler: Literal[
            "mcmc", "nuts_numpyro", "nuts_blackjax", "laplace", "vi"
        ] = "mcmc",
        **kwargs,
    ) -> az.InferenceData | pm.Approximation:
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
        (default), "nuts_numpyro", "nuts_blackjax" or "laplace". An ``Approximation``
        object if  ``"vi"``.
        """

        supported_samplers = ["mcmc", "nuts_numpyro", "nuts_blackjax", "laplace", "vi"]

        if sampler not in supported_samplers:
            raise ValueError(
                f"Unsupported sampler '{sampler}', must be one of {supported_samplers}"
            )

        self._inference_obj = self.model.fit(inference_method=sampler, **kwargs)

        return self.traces

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
            model=self.pymc_model, parent=self._parent_param
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

        output.append("Response variable: rt,response")
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
    def traces(self) -> az.InferenceData | pm.Approximation:
        """
        Returns the trace of the model after sampling.

        Raises
        ------
        ValueError
            If the model has not been sampled yet.

        Returns
        -------
            The trace of the model after sampling.
        """

        if not self._inference_obj:
            raise ValueError("Please sample the model first.")

        return self._inference_obj

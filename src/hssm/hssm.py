"""HSSM: Hierarchical Sequential Sampling Models.

A package based on pymc and bambi to perform Bayesian inference for hierarchical
sequential sampling models.

This file defines the entry class HSSM.
"""


from __future__ import annotations

from inspect import isclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import bambi as bmb
import numpy as np
import pymc as pm
from numpy.typing import ArrayLike
from pytensor.graph.op import Op

from hssm import wfpt
from hssm.utils import (
    HSSMModelGraph,
    Param,
    _parse_bambi,
    download_hf,
    get_alias_dict,
    merge_dicts,
)
from hssm.wfpt.config import (
    Config,
    LoglikKind,
    SupportedModels,
    default_model_config,
    default_params,
)

if TYPE_CHECKING:
    import arviz as az
    import pandas as pd
    import pytensor

LogLikeFunc = Callable[..., ArrayLike]


class HSSM:
    """The Hierarchical Sequential Sampling Model (HSSM) class.

    Parameters
    ----------
    data
        A pandas DataFrame with the minimum requirements of containing the data with the
        columns 'rt' and 'response'.
    model
        The name of the model to use. Currently supported models are "ddm", "angle",
        "levy", "ornstein", "weibull", "race_no_bias_angle_4", "ddm_seq2_no_bias". If
        using a custom model, please pass "custom". Defaults to "ddm".
    include, optional
        A list of dictionaries specifying parameter specifications to include in the
        model. If left unspecified, defaults will be used for all parameter
        specifications. Defaults to None.
    model_config, optional
        A dictionary containing the model configuration information. If None is
        provided, defaults will be used. Defaults to None.
    loglik, optional
        A likelihood function. Defaults to None. Requirements are:
        1. if `loglik_kind` is `"analytical"` or `"blackbox"`, a pm.Distribution, a
           pytensor Op, or a Python callable can be used. Signatures are:
            - `pm.Distribution`: needs to have parameters specified exactly as listed in
            `list_params`
            - `pytensor.graph.Op` and `Callable`: needs to accept the parameters
            specified exactly as listed in `list_params`
        2. If `loglik_kind` is `"approx_differentiable"`, then in addition to the
            specifications above, a `str` or `Pathlike` can also be used to specify a
            path to an `onnx` file. If a `str` is provided, HSSM will first look locally
            for an `onnx` file. If that is not successful, HSSM will try to download
            that `onnx` file from Hugging Face hub.
        3. It can also be `None`, in which case a default likelihood function will be
            used
    loglik_kind
        A string that specifies the kind of log-likelihood function specified with
        `loglik`. Defaults to "analytical". Can be one of the following:
        - `"analytical"`: an analytical (approximation) likelihood function. It is
            differentiable and can be used with samplers that requires differentiation.
        - `"approx_differentiable"`: a likelihood approximation network (LAN) likelihood
            function. It is differentiable and can be used with samplers that requires
            differentiation.
        - `"blackbox"`: a black box likelihood function. It is typically NOT
            differentiable.
    **kwargs
        Additional arguments passed to the bmb.Model object.

    Attributes
    ----------
    data
        A pandas DataFrame with at least two columns of "rt" and "response" indicating
        the response time and responses.
    list_params
        The list of strs of parameter names.
    model_name
        The name of the model.
    loglik:
        The likelihood function or a path to an onnx file.
    loglik_kind:
    model_config
        A dictionary representing the model configuration.
    model_distribution
        The likelihood function of the model in the form of a pm.Distribution subclass.
    family
        A Bambi family object.
    priors
        A dictionary containing the prior distribution of parameters.
    formula
        A string representing the model formula.
    link
        A string or a dictionary representing the link functions for all parameters.
    params
        A list of Param objects representing model parameters.

    Methods
    -------
    sample
        A method to sample posterior distributions.
    set_alias
        Sets the alias for a paramter.
    graph
        Plot the model with PyMC's built-in graph function.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model: SupportedModels = "ddm",
        include: list[dict] | None = None,
        model_config: Config | None = None,
        loglik: str | PathLike | LogLikeFunc | pytensor.graph.Op | None = None,
        loglik_kind: Literal[
            "analytical", "approx_differentiable", "blackbox"
        ] = "analytical",
        **kwargs,
    ):
        self.data = data
        self._inference_obj = None

        # Check if model has default config
        if _model_has_default(model, loglik_kind):
            default_config = default_model_config[model][loglik_kind]

            self.model_config = (
                default_config
                if model_config is None
                else merge_dicts(default_config, model_config)
            )
            self.loglik = self.model_config["loglik"] if loglik is None else loglik
            self.list_params = default_params[model]
        else:
            # If there is no default, we require a log-likelihood and a model_config
            if loglik is None:
                raise ValueError("Please provide a valid `loglik`.")
            self.loglik = loglik

            if model == "custom":
                if model_config is None:
                    raise ValueError(
                        "For custom models, please provide a valid `model_config`."
                    )
                if "list_params" not in model_config:
                    raise ValueError(
                        "For custom models, please provide `list_params` in "
                        + "`model_config`."
                    )
                self.model_config = model_config
                self.list_params = model_config["list_params"]
            else:
                self.model_config = {} if model_config is None else model_config
                self.list_params = default_params[model]

        self.loglik_kind = loglik_kind
        self.model_name = model
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

        ### Logic for different types of likelihoods:
        # -`analytical`` and `blackbox`:
        #     loglik should be a `pm.Distribution`` or a Python callable (any arbirary
        #     function).
        # - `approx_differentiable`:
        #     In addition to `pm.Distribution` and any arbitrary function, it can also
        #     be an str (which we will download from hugging face) or a Pathlike
        #     which we will download and make a distribution.

        # If user has already provided a log-likelihood function as a distribution
        # Use it directly as the distribution
        if isclass(self.loglik) and issubclass(self.loglik, pm.Distribution):
            self.model_distribution = self.loglik
        # If the user has provided a callable or an Op
        # Wrap it around with a distribution
        elif isinstance(self.loglik, Op) or callable(self.loglik):
            self.model_distribution = wfpt.make_distribution(
                loglik=loglik, list_params=self.list_params  # type: ignore
            )
        # All other situations
        else:
            if self.loglik_kind != "approx_differentiable":
                raise ValueError(
                    "loglik_kind is not `approx_differentiable, "
                    + "Please provide a pm.Distribution, an Op, or a callable "
                    + "as `loglik`"
                )
            if isinstance(self.loglik, str):
                if not Path(self.loglik).exists():
                    self.loglik = download_hf(self.loglik)

            self.model_distribution = wfpt.make_lan_distribution(
                model=self.loglik,
                list_params=self.list_params,
                backend=self.model_config["backend"],
                params_is_reg=params_is_reg,
            )

        assert self.model_distribution is not None

        self.likelihood = bmb.Likelihood(
            self.loglik_kind,
            params=self.list_params,
            parent=self._parent,
            dist=self.model_distribution,
        )

        self.family = SSMFamily(
            self.loglik_kind, likelihood=self.likelihood, link=self.link
        )

        self.model = bmb.Model(
            self.formula, data, family=self.family, priors=self.priors, **other_kwargs
        )

        self._aliases = get_alias_dict(self.model, self._parent_param)
        self.set_alias(self._aliases)

    def _transform_params(
        self, include: list[dict] | None, model: str, model_config: Config
    ) -> tuple[list[Param], bmb.Formula, dict | None, dict[str, str | bmb.Link] | str]:
        """Transform parameters.

        Transforms a list of dictionaries containing parameter information into a
        list of Param objects. This function creates a formula, priors,and a link for
        the Bambi package based on the parameters.

        Parameters
        ----------
        include
            A list of dictionaries containing information about the parameters.
        model
            A string that indicates the type of the model.
        model_config
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
                name = param_dict["name"]
                processed.append(name)
                is_parent = name == self._parent
                bounds = (
                    model_config["bounds"].get(name, None)
                    if "bounds" in model_config
                    else None
                )
                if "prior" not in param_dict or param_dict["prior"] is None:
                    if (
                        "default_priors" in model_config
                        and name in model_config["default_priors"]
                    ):
                        param_dict["prior"] = model_config["default_priors"][name]
                param = Param(
                    bounds=bounds,
                    is_parent=is_parent,
                    **param_dict,
                )
                params.append(param)

        for param_str in self.list_params:
            if param_str not in processed:
                is_parent = param_str == self._parent
                bounds = (
                    model_config["bounds"].get(param_str, None)
                    if "bounds" in model_config
                    else None
                )
                prior = (
                    model_config["default_priors"].get(param_str, None)
                    if "default_priors" in model_config
                    else None
                )
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
        """Perform sampling using the `fit` method via bambi.Model.

        Parameters
        ----------
        sampler
            The sampler to use. Can be either "mcmc" (default), "nuts_numpyro",
            "nuts_blackjax", "laplace", or "vi".
        kwargs
            Other arguments passed to bmb.Model.fit()

        Returns
        -------
            An ArviZ `InferenceData` instance if inference_method is `"mcmc"`
            (default), "nuts_numpyro", "nuts_blackjax" or "laplace". An `Approximation`
            object if `"vi"`.
        """
        supported_samplers = ["mcmc", "nuts_numpyro", "nuts_blackjax", "laplace", "vi"]

        if sampler not in supported_samplers:
            raise ValueError(
                f"Unsupported sampler '{sampler}', must be one of {supported_samplers}"
            )

        self._inference_obj = self.model.fit(inference_method=sampler, **kwargs)

        return self.traces

    def sample_posterior_predictive(
        self,
        idata: az.InferenceData | None = None,
        data: pd.DataFrame | None = None,
        inplace: bool = True,
        include_group_specific: bool = True,
    ) -> az.InferenceData | None:
        """Perform posterior predictive sampling from the HSSM model.

        Parameters
        ----------
        idata, optional
            The `InferenceData` object returned by `HSSM.sample()`. If not provided,
            the `InferenceData` from the last time `sample()` is called will be used.
        data, optional
            An optional data frame with values for the predictors that are used to
            obtain out-of-sample predictions. If omitted, the original dataset is used.
        inplace, optional
            If `True` will modify idata in-place and append a `posterior_predictive`
            group to `idata`. Otherwise, it will return a copy of idata with the
            predictions added, by default True.
        include_group_specific, optional
            If `True` will make predictions including the group specific effects.
            Otherwise, predictions are made with common effects only (i.e. group-
            specific are set to zero), by default True.

        Raises
        ------
        ValueError
            If the model has not been sampled yet and idata is not provided.

        Returns
        -------
            InferenceData or None
        """
        if idata is None:
            if self._inference_obj is None:
                raise ValueError(
                    "The model has not been sampled yet. "
                    + "Please either provide an idata object or sample the model first."
                )
            idata = self._inference_obj
        return self.model.predict(idata, "pps", data, inplace, include_group_specific)

    @property
    def pymc_model(self) -> pm.Model:
        """Provide access to the PyMC model.

        Returns
        -------
            The PyMC model built by bambi
        """
        return self.model.backend.model

    def set_alias(self, aliases: dict[str, str | dict]):
        """Set parameter aliases.

        Sets the aliases according to the dictionary passed to it and rebuild the
        model.

        Parameters
        ----------
        alias
            A dict specifying the paramter names being aliased and the aliases.
        """
        self.model.set_alias(aliases)
        self.model.build()

    # NOTE: can't annotate return type because the graphviz dependency is
    # optional
    def graph(self, formatting="plain", name=None, figsize=None, dpi=300, fmt="png"):
        """Produce a graphviz Digraph from a built HSSM model.

        Requires graphviz, which may be installed most easily with `conda install -c
        conda-forge python-graphviz`. Alternatively, you may install the `graphviz`
        binaries yourself, and then `pip install graphviz` to get the python bindings.
        See http://graphviz.readthedocs.io/en/stable/manual.html for more information.

        Parameters
        ----------
        formatting
            One of `"plain"` or `"plain_with_params"`. Defaults to `"plain"`.
        name
            Name of the figure to save. Defaults to `None`, no figure is saved.
        figsize
            Maximum width and height of figure in inches. Defaults to `None`, the
            figure size is set automatically. If defined and the drawing is larger than
            the given size, the drawing is uniformly scaled down so that it fits within
            the given size.  Only works if `name` is not `None`.
        dpi
            Point per inch of the figure to save.
            Defaults to 300. Only works if `name` is not `None`.
        fmt
            Format of the figure to save.
            Defaults to `"png"`. Only works if `name` is not `None`.

        Returns
        -------
            The graph

        Note
        ----
            The code is largely copied from
            https://github.com/bambinos/bambi/blob/main/bambi/models.py
            Credit for the code goes to Bambi developers.
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
        """Create a representation of the model."""
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
        """Create a string representation of the model."""
        return self.__repr__()

    @property
    def traces(self) -> az.InferenceData | pm.Approximation:
        """Return the trace of the model after sampling.

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


class SSMFamily(bmb.Family):
    """Extends bmb.Family to get around the dimensionality mismatch."""

    def create_extra_pps_coord(self):
        """Create an extra dimension.

        Returns
        -------
            _description_
        """
        return np.arange(2)


def _model_has_default(model: SupportedModels, loglik_kind: LoglikKind) -> bool:
    """Determine if the specified model has default configs.

    Also checks if `model` and `loglik_kind` are valid.

    Parameters
    ----------
    model
        User-specified model type.
    loglik_kind
        User-specified likelihood kind.

    Returns
    -------
        Whether the model is supported.
    """
    if loglik_kind not in [
        "analytical",
        "approx_differentiable",
        "blackbox",
    ]:
        raise ValueError(
            "'loglike_kind' must be one of "
            + '"analytical", "approx_differentiable", "blackbox".'
        )

    if model == "custom":
        return False

    if model not in default_model_config:
        supported_models = list(default_model_config.keys())
        raise ValueError(f"`model` must be one of {supported_models} or 'custom'.")

    return loglik_kind in default_model_config[model]

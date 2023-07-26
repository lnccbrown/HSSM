"""HSSM: Hierarchical Sequential Sampling Models.

A package based on pymc and bambi to perform Bayesian inference for hierarchical
sequential sampling models.

This file defines the entry class HSSM.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from inspect import isclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import bambi as bmb
import pymc as pm
from bambi.model_components import DistributionalComponent
from numpy.typing import ArrayLike
from pytensor.graph.op import Op

from hssm.config import (
    Config,
    LoglikKind,
    SupportedModels,
    default_model_config,
    default_params,
)
from hssm.distribution_utils import (
    make_blackbox_op,
    make_distribution,
    make_distribution_from_onnx,
    make_family,
)
from hssm.param import (
    Param,
    _parse_bambi,
)
from hssm.utils import (
    HSSMModelGraph,
    _print_prior,
    _process_param_in_kwargs,
    download_hf,
    get_alias_dict,
    merge_dicts,
)

if TYPE_CHECKING:
    from os import PathLike

    import arviz as az
    import numpy as np
    import pandas as pd
    import pytensor

LogLikeFunc = Callable[..., ArrayLike]

_logger = logging.getLogger("hssm")


class HSSM:
    """The Hierarchical Sequential Sampling Model (HSSM) class.

    Parameters
    ----------
    data
        A pandas DataFrame with the minimum requirements of containing the data with the
        columns "rt" and "response".
    model
        The name of the model to use. Currently supported models are "ddm", "ddm_sdv",
        "angle", "levy", "ornstein", "weibull", "race_no_bias_angle_4",
        "ddm_seq2_no_bias". If any other string is passed, the model will be considered
        custom, in which case all `model_config`, `loglik`, and `loglik_kind` have to be
        provided by the user.
    include : optional
        A list of dictionaries specifying parameter specifications to include in the
        model. If left unspecified, defaults will be used for all parameter
        specifications. Defaults to None.
    model_config : optional
        A dictionary containing the model configuration information. If None is
        provided, defaults will be used if there are any. Defaults to None.
        Fields for this `dict` are usually:

        - `"list_params"`: a list of parameters indicating the parameters of the model.
            The order in which the parameters are specified in this list is important.
            Values for each parameter will be passed to the likelihood function in this
            order.
        - `"backend"`: Only used when `loglik_kind` is `approx_differentiable` and
            an onnx file is supplied for the likelihood approximation network (LAN).
            Valid values are `"jax"` or `"pytensor"`. It determines whether the LAN in
            ONNX should be converted to `"jax"` or `"pytensor"`. If not provided,
            `jax` will be used for maximum performance.
        - `"default_priors"`: A `dict` indicating the default priors for each parameter.
        - `"bounds"`: A `dict` indicating the boundaries for each parameter. In the case
            of LAN, these bounds are training boundaries.
        - `"rv"`: Optional. Can be a `RandomVariable` class containing the user's own
            `rng_fn` function for sampling from the distribution that the user is
            supplying. If not supplied, HSSM will automatically generate a
            `RandomVariable` using the simulator identified by `model` from the
            `ssm_simulators` package. If `model` is not supported in `ssm_simulators`,
            a warning will be raised letting the user know that sampling from the
            `RandomVariable` will result in errors.
    loglik : optional
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
    loglik_kind : optional
        A string that specifies the kind of log-likelihood function specified with
        `loglik`. Defaults to `None`. Can be one of the following:

        - `"analytical"`: an analytical (approximation) likelihood function. It is
            differentiable and can be used with samplers that requires differentiation.
        - `"approx_differentiable"`: a likelihood approximation network (LAN) likelihood
            function. It is differentiable and can be used with samplers that requires
            differentiation.
        - `"blackbox"`: a black box likelihood function. It is typically NOT
            differentiable.
        - `None`, in which a default will be used. For `ddm` type of models, the default
            will be `analytical`. For other models supported, it will be
            `approx_differentiable`. If the model is a custom one, a ValueError
            will be raised.
    p_outlier : optional
        The fixed lapse probability or the prior distribution of the lapse probability.
        Defaults to a fixed value of 0.05. When `None`, the lapse probability will not
        be included in estimation.
    lapse : optional
        The lapse distribution. This argument is required only if `p_outlier` is not
        `None`. Defaults to Uniform(0.0, 10.0).
    hierarchical : optional
        If True, and if there is a `participant_id` field in `data`, will by default
        turn any unspecified parameter theta into a regression with
        "theta ~ 1 + (1|participant_id)" and default priors set by `bambi`.
    **kwargs
        Additional arguments passed to the `bmb.Model` object.

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
        The kind of likelihood used.
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
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model: SupportedModels | str = "ddm",
        include: list[dict] | None = None,
        model_config: Config | None = None,
        loglik: str | PathLike | LogLikeFunc | pytensor.graph.Op | None = None,
        loglik_kind: LoglikKind | None = None,
        p_outlier: float | dict | bmb.Prior | None = 0.05,
        lapse: dict | bmb.Prior | None = bmb.Prior("Uniform", lower=0.0, upper=10.0),
        hierarchical: bool = True,
        **kwargs,
    ):
        self.data = data
        self._inference_obj = None
        self.hierarchical = hierarchical and "participant_id" in data.columns
        self.has_lapse = p_outlier is not None and p_outlier != 0

        if loglik_kind is None:
            if model not in default_model_config:
                raise ValueError(
                    "When using a custom model, please provide a `loglik_kind.`"
                )
            # Setting loglik_kind to be the first of analytical or
            # approx_differentiable
            for kind in ["analytical", "approx_differentiable", "blackbox"]:
                model = cast(SupportedModels, model)
                if kind in default_model_config[model]:
                    kind = cast(LoglikKind, kind)
                    loglik_kind = kind
                    break
            if loglik_kind is None:
                raise ValueError(
                    "No default model_config is found. Please provide a `loglik_kind."
                )
        else:
            if loglik_kind not in [
                "analytical",
                "approx_differentiable",
                "blackbox",
            ]:
                raise ValueError(
                    "'loglike_kind', when provided, must be one of "
                    + '"analytical", "approx_differentiable", "blackbox".'
                )

        self.loglik_kind = loglik_kind
        self.model_name = model

        # Check if model has default config
        if _model_has_default(self.model_name, self.loglik_kind):
            model = cast(SupportedModels, model)
            default_config = default_model_config[model][loglik_kind]

            self.model_config = (
                deepcopy(default_config)
                if model_config is None
                else merge_dicts(default_config, model_config)
            )
            self.loglik = self.model_config["loglik"] if loglik is None else loglik
            self.list_params = default_params[model][:]
        else:
            # If there is no default, we require a log-likelihood
            if loglik is None:
                raise ValueError("Please provide a valid `loglik`.")
            self.loglik = loglik

            if model not in default_model_config:
                # For custom models, require model_config
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
                # For supported models without configs,
                # We don't require a model_config (because list_params is known)
                model = cast(SupportedModels, model)
                self.model_config = {} if model_config is None else model_config
                self.list_params = default_params[model][:]

        if (
            loglik_kind == "approx_differentiable"
            and "backend" not in self.model_config
        ):
            self.model_config["backend"] = "jax"

        # Logic for determining if p_outlier and lapse is specified correctly.
        # Basically, avoid situations where only one of them is specified.
        self._parent = self.list_params[0]
        if self.has_lapse and lapse is None:
            raise ValueError(
                "You have specified `p_outlier`. Please also specify `lapse`."
            )
        if lapse is not None and not self.has_lapse:
            _logger.warning(
                "You have specified the `lapse` argument to include a lapse "
                + "distribution, but `p_outlier` is set to either 0 or None. "
                + "Your lapse distribution will be ignored."
            )
        if "p_outlier" in self.list_params and self.list_params[-1] != "p_outlier":
            raise ValueError(
                "Please do not include 'p_outlier' in `list_params`. "
                + "We automatically append it to `list_params` when `p_outlier` "
                + "parameter is not None"
            )

        if self.has_lapse and self.list_params[-1] != "p_outlier":
            self.list_params.append("p_outlier")

        if include is None:
            include = []
        params_in_include = [param["name"] for param in include]

        # Process kwargs
        # If any of the keys is found in `list_params` it is a parameter specification
        # We add the parameter specification to `include`, which will be processed later
        # together with other parameter specifications in `include`.
        # Otherwise we create another dict and pass it to `bmb.Model`.
        other_kwargs: dict[Any, Any] = {}
        for k, v in kwargs.items():
            if k in self.list_params:
                if k in params_in_include:
                    raise ValueError(
                        f'Parameter "{k}" is already specified in `include`.'
                    )
                include.append(_process_param_in_kwargs(k, v))
            else:
                other_kwargs |= {k: v}

        # Process p_outliers the same way.
        if self.has_lapse:
            if "p_outlier" in params_in_include:
                raise ValueError(
                    "Please do not specify `p_outlier` in `include`. "
                    + "Please specify it with `p_outlier` instead."
                )
            include.append(_process_param_in_kwargs("p_outlier", p_outlier))

        self.params, self.formula, self.priors, self.link = self._transform_params(
            include, self.model_config
        )

        self._parent_param = self.params[self.list_params[0]]
        assert self._parent_param is not None

        params_is_reg = [
            param.is_regression
            for param_name, param in self.params.items()
            if param_name != "p_outlier"
        ]

        # For parameters that are regression, apply bounds at the likelihood level to
        # ensure that the samples that are out of bounds are discarded (replaced with
        # a large negative value).
        self.bounds = {
            name: param.bounds
            for name, param in self.params.items()
            if param.is_regression and param.bounds is not None
        }

        self.p_outlier = self.params.get("p_outlier")
        self.lapse = lapse if self.has_lapse else None

        ### Logic for different types of likelihoods:
        # -`analytical` and `blackbox`:
        #     loglik should be a `pm.Distribution`` or a Python callable (any arbitrary
        #     function).
        # - `approx_differentiable`:
        #     In addition to `pm.Distribution` and any arbitrary function, it can also
        #     be an str (which we will download from hugging face) or a Pathlike
        #     which we will download and make a distribution.

        # If user has already provided a log-likelihood function as a distribution
        # Use it directly as the distribution
        if isclass(self.loglik) and issubclass(self.loglik, pm.Distribution):
            self.model_distribution = self.loglik
        # If the user has provided an Op
        # Wrap it around with a distribution
        elif isinstance(self.loglik, Op):
            self.model_distribution = make_distribution(
                self.model_config.get("rv", self.model_name),
                loglik=self.loglik,
                list_params=self.list_params,
                bounds=self.bounds,
                lapse=self.lapse,
            )  # type: ignore
        # If the user has provided a callable (an arbitrary likelihood function)
        # If `loglik_kind` is `blackbox`, wrap it in an op and then a distribution
        # Otherwise, we assume that this function is differentiable with `pytensor`
        # and wrap it directly in a distribution.
        elif callable(self.loglik):
            if self.loglik_kind == "blackbox":
                self.loglik = make_blackbox_op(self.loglik)
            self.model_distribution = make_distribution(
                self.model_config.get("rv", self.model_name),
                loglik=self.loglik,
                list_params=self.list_params,
                bounds=self.bounds,
                lapse=self.lapse,
            )  # type: ignore
        # All other situations
        else:
            if self.loglik_kind != "approx_differentiable":
                raise ValueError(
                    "You set `loglik_kind` to `approx_differentiable "
                    + "but did not provide a pm.Distribution, an Op, or a callable "
                    + "as `loglik`."
                )
            if isinstance(self.loglik, str):
                if not Path(self.loglik).exists():
                    self.loglik = download_hf(self.loglik)

            self.model_distribution = make_distribution_from_onnx(
                rv=self.model_config.get("rv", self.model_name),
                onnx_model=self.loglik,
                list_params=self.list_params,
                backend=self.model_config.get("backend", "jax"),
                params_is_reg=params_is_reg,
                bounds=self.bounds,
                lapse=self.lapse,
            )

        self.family = make_family(
            self.model_distribution,
            self.list_params,
            self.link,
            self._parent,
        )

        self.model = bmb.Model(
            self.formula, data, family=self.family, priors=self.priors, **other_kwargs
        )

        self._aliases = get_alias_dict(self.model, self._parent_param)
        self.set_alias(self._aliases)

    def _transform_params(
        self, include: list[dict] | None, model_config: Config
    ) -> tuple[
        dict[str, Param], bmb.Formula, dict | None, dict[str, str | bmb.Link] | str
    ]:
        """Transform parameters.

        Transforms a list of dictionaries containing parameter information into a
        list of Param objects. This function creates a formula, priors,and a link for
        the Bambi package based on the parameters.

        Parameters
        ----------
        include
            A list of dictionaries containing information about the parameters.
        model_config
            A dict for the configuration for the model.

        Returns
        -------
        list[Param], bmb.Formula, dict | None, dict | str
            A tuple of 4 items, the latter 3 are for creating the bambi model.
            - A list of the same length as self.list_params containing Param objects.
            - A bmb.formula object.
            - An optional dict containing prior information for Bambi.
            - An optional dict of link functions for Bambi.
        """
        processed = []
        params: dict[str, Param] = {}
        if include:
            for param_dict in include:
                name = param_dict["name"]
                processed.append(name)
                for k in param_dict.keys():
                    if k not in ["name", "formula", "prior", "link", "bounds"]:
                        raise ValueError(
                            f"Invalid key {k} for the specification of {name}!"
                        )
                param = _create_param(
                    param_dict, model_config, is_parent=name == self._parent
                )
                params[name] = param

        for param_str in self.list_params:
            if param_str not in processed:
                is_parent = param_str == self._parent
                if self.hierarchical:
                    bounds = (
                        model_config["bounds"].get(param_str)
                        if "bounds" in model_config
                        else None
                    )
                    param = Param(
                        param_str,
                        formula="1 + (1|participant_id)",
                        link="identity",
                        bounds=bounds,
                        is_parent=is_parent,
                    )
                else:
                    param = _create_param(param_str, model_config, is_parent=is_parent)
                params[param_str] = param

        sorted_params = {k: params[k] for k in self.list_params}

        return sorted_params, *_parse_bambi(list(sorted_params.values()))

    def sample(
        self,
        sampler: Literal["mcmc", "nuts_numpyro", "nuts_blackjax", "laplace", "vi"]
        | None = None,
        **kwargs,
    ) -> az.InferenceData | pm.Approximation:
        """Perform sampling using the `fit` method via bambi.Model.

        Parameters
        ----------
        sampler
            The sampler to use. Can be one of "mcmc", "nuts_numpyro",
            "nuts_blackjax", "laplace", or "vi". If using `blackbox` likelihoods,
            this cannot be "nuts_numpyro" or "nuts_blackjax". By default it is None, and
            sampler will automatically be chosen: when the model uses the
            `approx_differentiable` likelihood, and `jax` backend, "nuts_numpyro" will
            be used. Otherwise, "mcmc" (the default PyMC NUTS sampler) will be used.
        kwargs
            Other arguments passed to bmb.Model.fit(). Please see [here]
            (https://bambinos.github.io/bambi/api_reference.html#bambi.models.Model.fit)
            for full documentation.

        Returns
        -------
        az.InferenceData | pm.Approximation
            An ArviZ `InferenceData` instance if inference_method is `"mcmc"`
            (default), "nuts_numpyro", "nuts_blackjax" or "laplace". An `Approximation`
            object if `"vi"`.
        """
        if sampler is None:
            if (
                self.loglik_kind == "approx_differentiable"
                and self.model_config.get("backend") == "jax"
            ):
                sampler = "nuts_numpyro"
            else:
                sampler = "mcmc"

        supported_samplers = ["mcmc", "nuts_numpyro", "nuts_blackjax", "laplace", "vi"]

        if sampler not in supported_samplers:
            raise ValueError(
                f"Unsupported sampler '{sampler}', must be one of {supported_samplers}"
            )

        if self.loglik_kind == "blackbox":
            if sampler in ["nuts_blackjax", "nuts_numpyro"]:
                raise ValueError(
                    f"{sampler} sampler does not work with blackbox likelihoods."
                )

            if "step" not in kwargs:
                kwargs["step"] = pm.Slice(model=self.pymc_model)

        if (
            self.loglik_kind == "approx_differentiable"
            and self.model_config.get("backend") == "jax"
            and sampler == "mcmc"
            and kwargs.get("cores", None) != 1
        ):
            _logger.warning(
                "Parallel sampling might not work with `jax` backend and the PyMC NUTS "
                + "sampler on some platforms. Please consider using `nuts_numpyro` or "
                + "`nuts_blackjax` sampler if that is a problem."
            )

        self._inference_obj = self.model.fit(inference_method=sampler, **kwargs)

        return self.traces

    def sample_posterior_predictive(
        self,
        idata: az.InferenceData | None = None,
        data: pd.DataFrame | None = None,
        inplace: bool = True,
        include_group_specific: bool = True,
        kind: Literal["pps", "mean"] = "pps",
    ) -> az.InferenceData | None:
        """Perform posterior predictive sampling from the HSSM model.

        Parameters
        ----------
        idata : optional
            The `InferenceData` object returned by `HSSM.sample()`. If not provided,
            the `InferenceData` from the last time `sample()` is called will be used.
        data : optional
            An optional data frame with values for the predictors that are used to
            obtain out-of-sample predictions. If omitted, the original dataset is used.
        inplace : optional
            If `True` will modify idata in-place and append a `posterior_predictive`
            group to `idata`. Otherwise, it will return a copy of idata with the
            predictions added, by default True.
        include_group_specific : optional
            If `True` will make predictions including the group specific effects.
            Otherwise, predictions are made with common effects only (i.e. group-
            specific are set to zero), by default True.
        kind
            Indicates the type of prediction required. Can be `"mean"` or `"pps"`. The
            first returns draws from the posterior distribution of the mean, while the
            latter returns the draws from the posterior predictive distribution
            (i.e. the posterior probability distribution for a new observation).
            Defaults to `"pps"`.

        Raises
        ------
        ValueError
            If the model has not been sampled yet and idata is not provided.

        Returns
        -------
        az.InferenceData | None
            InferenceData or None
        """
        if idata is None:
            if self._inference_obj is None:
                raise ValueError(
                    "The model has not been sampled yet. "
                    + "Please either provide an idata object or sample the model first."
                )
            idata = self._inference_obj
        return self.model.predict(idata, kind, data, inplace, include_group_specific)

    def sample_prior_predictive(
        self,
        draws: int = 500,
        var_names: str | list[str] | None = None,
        omit_offsets: bool = True,
        random_seed: np.random.Generator | None = None,
    ) -> az.InferenceData:
        """Generate samples from the prior predictive distribution.

        Parameters
        ----------
        draws
            Number of draws to sample from the prior predictive distribution. Defaults
            to 500.
        var_names
            A list of names of variables for which to compute the prior predictive
            distribution. Defaults to ``None`` which means both observed and unobserved
            RVs.
        omit_offsets
            Whether to omit offset terms. Defaults to ``True``.
        random_seed
            Seed for the random number generator.

        Returns
        -------
        az.InferenceData
            ``InferenceData`` object with the groups ``prior``, ``prior_predictive`` and
            ``observed_data``.
        """
        return self.model.prior_predictive(draws, var_names, omit_offsets, random_seed)

    @property
    def pymc_model(self) -> pm.Model:
        """Provide access to the PyMC model.

        Returns
        -------
        pm.Model
            The PyMC model built by bambi
        """
        return self.model.backend.model

    def set_alias(self, aliases: dict[str, str | dict]):
        """Set parameter aliases.

        Sets the aliases according to the dictionary passed to it and rebuild the
        model.

        Parameters
        ----------
        aliases
            A dict specifying the parameter names being aliased and the aliases.
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
        graphviz.Graph
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
        output.append(f"Likelihood: {self.loglik_kind}")
        output.append(f"Observations: {len(self.data)}")
        output.append("")

        output.append("Parameters:")
        output.append("")

        for param in self.params.values():
            if param.name == "p_outlier":
                continue
            name = "c(rt, response)" if param.is_parent else param.name
            output.append(f"{param.name}:")

            component = self.model.components[name]

            # Regression case:
            if param.is_regression:
                assert isinstance(component, DistributionalComponent)
                output.append(f"    Formula: {param.formula}")
                output.append("    Priors:")
                intercept_term = component.intercept_term
                if intercept_term is not None:
                    output.append(_print_prior(intercept_term))
                for _, common_term in component.common_terms.items():
                    output.append(_print_prior(common_term))
                for _, group_specific_term in component.group_specific_terms.items():
                    output.append(_print_prior(group_specific_term))
                output.append(f"    Link: {param.link}")
            # None regression case
            else:
                if param.prior is None:
                    prior = (
                        component.intercept_term.prior
                        if param.is_parent
                        else component.prior
                    )
                else:
                    prior = param._prior if param.is_truncated else param.prior
                output.append(f"    Prior: {prior}")
            output.append(f"    Explicit bounds: {param.bounds}")

        if self.p_outlier is not None:
            # TODO: Allow regression for self.p_outlier
            # Need to determine what the output should look like
            # and whether p should be hierarchical when self.hierarchical is True.
            assert not self.p_outlier.is_regression
            output.append("")
            output.append(f"Lapse probability: {self.p_outlier.prior}")
            output.append(f"Lapse distribution: {self.lapse}")

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
        az.InferenceData | pm.Approximation
            The trace of the model after sampling.
        """
        if not self._inference_obj:
            raise ValueError("Please sample the model first.")

        return self._inference_obj


def _model_has_default(model: SupportedModels | str, loglik_kind: LoglikKind) -> bool:
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
    bool
        Whether the model is supported.
    """
    if model not in default_model_config:
        return False

    model = cast(SupportedModels, model)
    return loglik_kind in default_model_config[model]


def _create_param(param: str | dict, model_config: dict, is_parent: bool) -> Param:
    """Create a Param object.

    Parameters
    ----------
    param
        A dict or str containing parameter settings.
    model_config
        A dict containing the config for the model.
    is_parent
        Indicates whether this current param is a parent in bambi.

    Returns
    -------
    Param
        A Param object with info form param and model_config injected.
    """
    if isinstance(param, dict):
        name = param["name"]
        if "bounds" not in param:
            bounds = (
                model_config["bounds"].get(name, None)
                if "bounds" in model_config
                else None
            )
        else:
            bounds = param["bounds"]
        if "prior" not in param or param["prior"] is None:
            if (
                "default_priors" in model_config
                and name in model_config["default_priors"]
                and "formula" not in model_config
            ):
                prior = model_config["default_priors"][name]
            else:
                prior = None
        else:
            prior = param["prior"]
        return Param(
            name=name,
            prior=prior,
            formula=param.get("formula"),
            link=param.get("link"),
            bounds=bounds,
            is_parent=is_parent,
        )

    bounds = (
        model_config["bounds"].get(param, None) if "bounds" in model_config else None
    )
    prior = (
        model_config["default_priors"].get(param, None)
        if "default_priors" in model_config
        else None
    )
    return Param(
        name=param,
        prior=prior,
        bounds=bounds,
        is_parent=is_parent,
    )

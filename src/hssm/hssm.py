"""HSSM: Hierarchical Sequential Sampling Models.

A package based on pymc and bambi to perform Bayesian inference for hierarchical
sequential sampling models.

This file defines the entry class HSSM.
"""

import logging
from copy import deepcopy
from inspect import isclass
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Literal

import arviz as az
import bambi as bmb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import seaborn as sns
import xarray as xr
from bambi.model_components import DistributionalComponent
from bambi.transformations import transformations_namespace

from hssm.defaults import (
    LoglikKind,
    SupportedModels,
)
from hssm.distribution_utils import (
    make_blackbox_op,
    make_distribution,
    make_distribution_from_onnx,
    make_family,
)
from hssm.param import (
    Param,
    _make_default_prior,
)
from hssm.utils import (
    HSSMModelGraph,
    _print_prior,
    _process_param_in_kwargs,
    _random_sample,
    download_hf,
    get_alias_dict,
)

from . import plotting
from .config import Config, ModelConfig

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
        "full_ddm", "angle", "levy", "ornstein", "weibull", "race_no_bias_angle_4",
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
        - `"extra_fields"`: Optional. A list of strings indicating the additional
            columns in `data` that will be passed to the likelihood function for
            calculation. This is helpful if the likelihood function depends on data
            other than the observed data and the parameter values.
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
        "theta ~ 1 + (1|participant_id)" and default priors set by `bambi`. Also changes
        default values of `link_settings` and `prior_settings`. Defaults to False.
    link_settings : optional
        An optional string literal that indicates the link functions to use for each
        parameter. Helpful for hierarchical models where sampling might get stuck/
        very slow. Can be one of the following:

        - `"log_logit"`: applies log link functions to positive parameters and
        generalized logit link functions to parameters that have explicit bounds.
        - `None`: unless otherwise specified, the `"identity"` link functions will be
        used.
        The default value is `None`.
    prior_settings : optional
        An optional string literal that indicates the prior distributions to use for
        each parameter. Helpful for hierarchical models where sampling might get stuck/
        very slow. Can be one of the following:

        - `"safe"`: HSSM will scan all parameters in the model and apply safe priors to
        all parameters that do not have explicit bounds.
        - `None`: HSSM will use bambi to provide default priors for all parameters. Not
        recommended when you are using hierarchical models.
        The default value is `None` when `hierarchical` is `False` and `"safe"` when
        `hierarchical` is `True`.
    extra_namespace : optional
        Additional user supplied variables with transformations or data to include in
        the environment where the formula is evaluated. Defaults to `None`.
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
        include: list[dict | Param] | None = None,
        model_config: ModelConfig | dict | None = None,
        loglik: str
        | PathLike
        | Callable
        | pytensor.graph.Op
        | type[pm.Distribution]
        | None = None,
        loglik_kind: LoglikKind | None = None,
        p_outlier: float | dict | bmb.Prior | None = 0.05,
        lapse: dict | bmb.Prior | None = bmb.Prior("Uniform", lower=0.0, upper=10.0),
        hierarchical: bool = False,
        link_settings: Literal["log_logit"] | None = None,
        prior_settings: Literal["safe"] | None = None,
        extra_namespace: dict[str, Any] | None = None,
        **kwargs,
    ):
        self.data = data
        self._inference_obj = None
        self.hierarchical = hierarchical

        if self.hierarchical and "participant_id" not in self.data.columns:
            raise ValueError(
                "You have specified a hierarchical model, but there is no "
                + "`participant_id` field in the DataFrame that you have passed."
            )

        if self.hierarchical and prior_settings is None:
            prior_settings = "safe"

        self.link_settings = link_settings
        self.prior_settings = prior_settings

        additional_namespace = transformations_namespace.copy()
        if extra_namespace is not None:
            additional_namespace.update(extra_namespace)
        self.additional_namespace = additional_namespace

        responses = self.data["response"].unique().astype(int)
        self.n_responses = len(responses)
        if self.n_responses == 2:
            if -1 not in responses or 1 not in responses:
                raise ValueError(
                    "The response column must contain only -1 and 1 when there are "
                    + "two responses."
                )

        # Construct a model_config from defaults
        self.model_config = Config.from_defaults(model, loglik_kind)
        # Update defaults with user-provided config, if any
        if model_config is not None:
            self.model_config.update_config(
                model_config
                if isinstance(model_config, ModelConfig)
                else ModelConfig(**model_config)  # also serves as dict validation
            )
        # Update loglik with user-provided value
        self.model_config.update_loglik(loglik)
        # Ensure that all required fields are valid
        self.model_config.validate()

        # Set up shortcuts so old code will work
        self.list_params = self.model_config.list_params
        self.model_name = self.model_config.model_name
        self.loglik = self.model_config.loglik
        self.loglik_kind = self.model_config.loglik_kind
        self.extra_fields = self.model_config.extra_fields

        self._check_extra_fields()

        # Process lapse distribution
        self.has_lapse = p_outlier is not None and p_outlier != 0
        self._check_lapse(lapse)
        if self.has_lapse and self.list_params[-1] != "p_outlier":
            self.list_params.append("p_outlier")

        # Process kwargs and p_outlier and add them to include
        include, other_kwargs = self._add_kwargs_and_p_outlier_to_include(
            include, kwargs, p_outlier
        )

        # Process parameter specifications include
        processed = self._preprocess_include(include)
        # Process parameter specifications not in include
        self.params = self._preprocess_rest(processed)
        # Find the parent parameter
        self._parent, self._parent_param = self._find_parent()
        assert self._parent_param is not None

        self._override_defaults()
        self._process_all()

        # Get the bambi formula, priors, and link
        self.formula, self.priors, self.link = self._parse_bambi()

        # For parameters that are regression, apply bounds at the likelihood level to
        # ensure that the samples that are out of bounds are discarded (replaced with
        # a large negative value).
        self.bounds = {
            name: param.bounds
            for name, param in self.params.items()
            if param.is_regression and param.bounds is not None
        }

        # Set p_outlier and lapse
        self.p_outlier = self.params.get("p_outlier")
        self.lapse = lapse if self.has_lapse else None

        self.model_distribution = self._make_model_distribution()

        self.family = make_family(
            self.model_distribution,
            self.list_params,
            self.link,
            self._parent,
        )

        self.model = bmb.Model(
            self.formula,
            data=data,
            family=self.family,
            priors=self.priors,
            extra_namespace=extra_namespace,
            **other_kwargs,
        )

        self._aliases = get_alias_dict(self.model, self._parent_param)
        self.set_alias(self._aliases)

    def sample(
        self,
        sampler: Literal["mcmc", "nuts_numpyro", "nuts_blackjax", "laplace", "vi"]
        | None = None,
        init: str | None = None,
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
        init: optional
            Initialization method to use for the sampler. If any of the NUTS samplers
            is used, defaults to `"adapt_diag"`. Otherwise, defaults to `"auto"`.
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
                and self.model_config.backend == "jax"
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
                kwargs |= {"step": pm.Slice(model=self.pymc_model)}

        if (
            self.loglik_kind == "approx_differentiable"
            and self.model_config.backend == "jax"
            and sampler == "mcmc"
            and kwargs.get("cores", None) != 1
        ):
            _logger.warning(
                "Parallel sampling might not work with `jax` backend and the PyMC NUTS "
                + "sampler on some platforms. Please consider using `nuts_numpyro` or "
                + "`nuts_blackjax` sampler if that is a problem."
            )

        if self._check_extra_fields():
            self._update_extra_fields()

        if init is None:
            if sampler in ["mcmc", "nuts_numpyro", "nuts_blackjax"]:
                init = "adapt_diag"
            else:
                init = "auto"

        self._inference_obj = self.model.fit(
            inference_method=sampler, init=init, **kwargs
        )

        return self.traces

    def sample_posterior_predictive(
        self,
        idata: az.InferenceData | None = None,
        data: pd.DataFrame | None = None,
        inplace: bool = True,
        include_group_specific: bool = True,
        kind: Literal["pps", "mean"] = "pps",
        n_samples: int | float | None = None,
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
        n_samples
            The number of samples to draw from the posterior predictive distribution
            from each chain.
            When it's an integer >= 1, the number of samples to be extracted from the
            `draw` dimension. If this integer is larger than the number of posterior
            samples in each chain, all posterior samples will be used
            in posterior predictive sampling. When a float between 0 and 1, the
            proportion of samples from the draw dimension from each chain to be used in
            posterior predictive sampling.. If this proportion is very
            small, at least one sample will be used. When None, all posterior samples
            will be used. Defaults to None.

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

        if self._check_extra_fields(data):
            self._update_extra_fields(data)

        if n_samples is not None:
            # Make a copy of idata, set the `posterior` group to be a random sub-sample
            # of the original (draw dimension gets sub-sampled)
            idata_copy = idata.copy()
            idata_random_sample = _random_sample(
                idata_copy["posterior"], n_samples=n_samples
            )
            delattr(idata_copy, "posterior")
            idata_copy.add_groups(posterior=idata_random_sample)

            # If the user specifies an inplace operation, we need to modify the original
            if inplace:
                self.model.predict(idata_copy, kind, data, True, include_group_specific)
                idata.add_groups(
                    posterior_predictive=idata_copy["posterior_predictive"]
                )

                return None

            return self.model.predict(
                idata_copy, kind, data, False, include_group_specific
            )

        return self.model.predict(idata, kind, data, inplace, include_group_specific)

    def plot_posterior_predictive(self, **kwargs) -> mpl.axes.Axes | sns.FacetGrid:
        """Produce a posterior predictive plot.

        Equivalent to calling `hssm.plotting.plot_posterior_predictive()` with the
        model. Please see that function for
        [full documentation][hssm.plotting.plot_posterior_predictive].

        Returns
        -------
        mpl.axes.Axes | sns.FacetGrid
            The matplotlib axis or seaborn FacetGrid object containing the plot.
        """
        return plotting.plot_posterior_predictive(self, **kwargs)

    def plot_quantile_probability(self, **kwargs) -> mpl.axes.Axes | sns.FacetGrid:
        """Produce a quantile probability plot.

        Equivalent to calling `hssm.plotting.plot_quantile_probability()` with the
        model. Please see that function for
        [full documentation][hssm.plotting.plot_quantile_probability].

        Returns
        -------
        mpl.axes.Axes | sns.FacetGrid
            The matplotlib axis or seaborn FacetGrid object containing the plot.
        """
        return plotting.plot_quantile_probability(self, **kwargs)

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

    def plot_trace(
        self,
        data: az.InferenceData | None = None,
        include_deterministic: bool = False,
        tight_layout: bool = True,
        **kwargs,
    ) -> None:
        """Generate trace plot with ArviZ but with additional convenience features.

        This is a simple wrapper for the az.plot_trace() function. By default, it
        filters out the deterministic values from the plot. Please see the
        [arviz documentation]
        (https://arviz-devs.github.io/arviz/api/generated/arviz.plot_trace.html)
        for additional parameters that can be specified.

        Parameters
        ----------
        data : optional
            An ArviZ InferenceData object. If None, the traces stored in the model will
            be used.
        include_deterministic : optional
            Whether to include deterministic variables in the plot. Defaults to False.
            Note that if include deterministic is set to False and and `var_names` is
            provided, the `var_names` provided will be modified to also exclude the
            deterministic values. If this is not desirable, set
            `include deterministic` to True.
        tight_layout : optional
            Whether to call plt.tight_layout() after plotting. Defaults to True.
        """
        data = data or self.traces

        if not include_deterministic:
            var_names = self._get_deterministic_var_names(data)
            if var_names:
                if "var_names" in kwargs:
                    if isinstance(kwargs["var_names"], str):
                        if kwargs["var_names"] not in var_names:
                            var_names.append(kwargs["var_names"])
                        kwargs["var_names"] = var_names
                    elif isinstance(kwargs["var_names"], list):
                        kwargs["var_names"] = list(
                            set(var_names) | set(kwargs["var_names"])
                        )
                    elif kwargs["var_names"] is None:
                        kwargs["var_names"] = var_names
                    else:
                        raise ValueError(
                            "`var_names` must be a string, a list of strings, or None."
                        )
                else:
                    kwargs["var_names"] = var_names

        az.plot_trace(data, **kwargs)

        if tight_layout:
            plt.tight_layout()

    def summary(
        self,
        data: az.InferenceData | None = None,
        include_deterministic: bool = False,
        **kwargs,
    ) -> pd.DataFrame | xr.Dataset:
        """Produce a summary table with ArviZ but with additional convenience features.

        This is a simple wrapper for the az.summary() function. By default, it
        filters out the deterministic values from the plot. Please see the
        [arviz documentation]
        (https://arviz-devs.github.io/arviz/api/generated/arviz.summary.html)
        for additional parameters that can be specified.

        Parameters
        ----------
        data
            An ArviZ InferenceData object. If None, the traces stored in the model will
            be used.
        include_deterministic : optional
            Whether to include deterministic variables in the plot. Defaults to False.
            Note that if include_deterministic is set to False and and `var_names` is
            provided, the `var_names` provided will be modified to also exclude the
            deterministic values. If this is not desirable, set
            `include_deterministic` to True.

        Returns
        -------
        pd.DataFrame | xr.Dataset
            A pandas DataFrame or xarray Dataset containing the summary statistics.
        """
        data = data or self.traces

        if not include_deterministic:
            var_names = self._get_deterministic_var_names(data)
            if var_names:
                kwargs["var_names"] = list(set(var_names + kwargs.get("var_names", [])))

        return az.summary(data, **kwargs)

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
                    prior = param.prior
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

    def _check_lapse(self, lapse):
        """Determine if p_outlier and lapse is specified correctly."""
        # Basically, avoid situations where only one of them is specified.
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

    def _fill_default(self, p: dict | Param, param_name: str) -> dict | Param:
        """Fill parameter specification in include with defaults from config."""
        default_prior, default_bounds = self.model_config.get_defaults(param_name)
        filled_default_bounds = False
        if isinstance(p, dict):
            if p.get("bounds") is None:
                p["bounds"] = default_bounds
                filled_default_bounds = True

            if "formula" not in p and p.get("prior") is None:
                if default_prior is not None:
                    p["prior"] = default_prior
                    if filled_default_bounds:
                        p_param = Param(**p)
                        p_param.do_not_truncate()
                        return p_param
                else:
                    if p["bounds"] is not None:
                        p["prior"] = _make_default_prior(p["bounds"])

        else:
            if not p.bounds:
                p.bounds = default_bounds
                filled_default_bounds = True

            if not p.formula and not p.prior:
                if default_prior is not None:
                    p.prior = default_prior
                    if filled_default_bounds:
                        p.do_not_truncate()
                else:
                    if p.bounds is not None:
                        p.prior = _make_default_prior(p.bounds)

        return p

    def _add_kwargs_and_p_outlier_to_include(
        self, include, kwargs, p_outlier
    ) -> tuple[list, dict]:
        """Process kwargs and p_outlier and add them to include."""
        if include is None:
            include = []
        else:
            include = include.copy()
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

        return include, other_kwargs

    def _preprocess_include(self, include: list[dict | Param]) -> dict[str, Param]:
        """Turn parameter specs in include into Params."""
        result: dict[str, Param] = {}

        for param in include:
            name = param["name"]
            if name is None:
                raise ValueError(
                    "One or more parameters do not have a name. "
                    + "Please ensure that names are specified to all of them."
                )
            if name not in self.list_params:
                raise ValueError(f"{name} is not included in the list of parameters.")
            param_with_default = self._fill_default(param, name)
            result[name] = (
                Param(**param_with_default)
                if isinstance(param_with_default, dict)
                else param_with_default
            )

        return result

    def _preprocess_rest(self, processed: dict[str, Param]) -> dict[str, Param]:
        """Turn parameter specs not in include into Params."""
        not_in_include = {}

        for param_str in self.list_params:
            if param_str not in processed:
                if self.hierarchical:
                    bounds = self.model_config.bounds.get(param_str)
                    param = Param(
                        param_str,
                        formula=f"{param_str} ~ 1 + (1|participant_id)",
                        bounds=bounds,
                    )
                else:
                    prior, bounds = self.model_config.get_defaults(param_str)
                    param = Param(param_str, prior=prior, bounds=bounds)
                    param.do_not_truncate()
                not_in_include[param_str] = param

        processed |= not_in_include
        sorted_params = {}

        for param_name in self.list_params:
            sorted_params[param_name] = processed[param_name]

        return sorted_params

    def _find_parent(self) -> tuple[str, Param]:
        """Find the parent param for the model.

        The first param that has a regression will be set as parent. If none of the
        params is a regression, then the first param will be set as parent.

        Returns
        -------
        str
            The name of the param as string
        Param
            The parent Param object
        """
        for param_str in self.list_params:
            param = self.params[param_str]
            if param.is_regression:
                param.set_parent()
                return param_str, param

        param_str = self.list_params[0]
        param = self.params[param_str]
        param.set_parent()
        return param_str, param

    def _override_defaults(self):
        """Override the default priors or links."""
        is_ddm = (
            self.model_name in ["ddm", "ddm_sdv", "ddm_full"]
            and self.loglik_kind != "approx_differentiable"
        )
        for param in self.list_params:
            param_obj = self.params[param]
            if self.prior_settings == "safe":
                if is_ddm:
                    param_obj.override_default_priors_ddm(
                        self.data, self.additional_namespace
                    )
                else:
                    param_obj.override_default_priors(
                        self.data, self.additional_namespace
                    )
            if self.link_settings == "log_logit":
                param_obj.override_default_link()

    def _process_all(self):
        """Process all params."""
        assert self.list_params is not None
        for param in self.list_params:
            self.params[param].convert()

    def _parse_bambi(
        self,
    ) -> tuple[bmb.Formula, dict | None, dict[str, str | bmb.Link] | str]:
        """Retrieve three items that helps with bambi model building.

        Returns
        -------
        tuple
            A tuple containing:
                1. A bmb.Formula object.
                2. A dictionary of priors, if any is specified.
                3. A dictionary of link functions, if any is specified.
        """
        # Handle the edge case where list_params is empty:
        if not self.params:
            return bmb.Formula("c(rt, response) ~ 1"), None, "identity"

        parent_formula = None
        other_formulas = []
        priors: dict[str, Any] = {}
        links: dict[str, str | bmb.Link] = {}

        for _, param in self.params.items():
            formula, prior, link = param.parse_bambi()

            if param.is_parent:
                parent_formula = formula
            else:
                if formula is not None:
                    other_formulas.append(formula)
            if prior is not None:
                priors |= prior
            if link is not None:
                links |= link

        assert parent_formula is not None
        result_formula: bmb.Formula = bmb.Formula(parent_formula, *other_formulas)
        result_priors = None if not priors else priors
        result_links: dict | str = "identity" if not links else links

        return result_formula, result_priors, result_links

    def _make_model_distribution(self) -> type[pm.Distribution]:
        """Make a pm.Distribution for the model."""
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
            return self.loglik
        # If the user has provided an Op
        # Wrap it around with a distribution
        if isinstance(self.loglik, pytensor.graph.Op):
            return make_distribution(
                rv=self.model_config.rv or self.model_name,
                loglik=self.loglik,
                list_params=self.list_params,
                bounds=self.bounds,
                lapse=self.lapse,
                extra_fields=None
                if not self.extra_fields
                else [deepcopy(self.data[field].values) for field in self.extra_fields],
            )  # type: ignore
        # If the user has provided a callable (an arbitrary likelihood function)
        # If `loglik_kind` is `blackbox`, wrap it in an op and then a distribution
        # Otherwise, we assume that this function is differentiable with `pytensor`
        # and wrap it directly in a distribution.
        if callable(self.loglik):
            if self.loglik_kind == "blackbox":
                self.loglik = make_blackbox_op(self.loglik)
            return make_distribution(
                rv=self.model_config.rv or self.model_name,
                loglik=self.loglik,
                list_params=self.list_params,
                bounds=self.bounds,
                lapse=self.lapse,
                extra_fields=None
                if not self.extra_fields
                else [deepcopy(self.data[field].values) for field in self.extra_fields],
            )  # type: ignore
        # All other situations
        if self.loglik_kind != "approx_differentiable":
            raise ValueError(
                "You set `loglik_kind` to `approx_differentiable "
                + "but did not provide a pm.Distribution, an Op, or a callable "
                + "as `loglik`."
            )
        if isinstance(self.loglik, str):
            if not Path(self.loglik).exists():
                self.loglik = download_hf(self.loglik)

        params_is_reg = [
            param.is_regression
            for param_name, param in self.params.items()
            if param_name != "p_outlier"
        ]

        return make_distribution_from_onnx(
            rv=self.model_config.rv or self.model_name,
            onnx_model=self.loglik,
            list_params=self.list_params,
            backend=self.model_config.backend or "jax",
            params_is_reg=params_is_reg,
            bounds=self.bounds,
            lapse=self.lapse,
            extra_fields=None
            if not self.extra_fields
            else [deepcopy(self.data[field].values) for field in self.extra_fields],
        )

    def _check_extra_fields(self, data: pd.DataFrame | None = None) -> bool:
        """Check if every field in self.extra_fields exists in data."""
        if not self.extra_fields:
            return False

        if not data:
            data = self.data

        for field in self.extra_fields:
            if field not in data.columns:
                raise ValueError(f"Field {field} not found in data.")

        return True

    def _update_extra_fields(self, new_data: pd.DataFrame | None = None):
        """Update the extra fields data in self.model_distribution.

        Parameters
        ----------
        new_data
            A DataFrame containing new data for update.
        """
        if not new_data:
            new_data = self.data

        self.model_distribution.extra_fields = [
            new_data[field].values for field in self.extra_fields
        ]

    def _get_deterministic_var_names(self, idata) -> list[str]:
        """Filter out the deterministic variables in var_names."""
        var_names = [
            f"~{param_name}"
            for param_name, param in self.params.items()
            if param.is_regression and not param.is_parent
        ]

        if "rt,response_mean" in idata["posterior"].data_vars:
            var_names.append("~rt,response_mean")
        return var_names

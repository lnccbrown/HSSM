"""HSSM: Hierarchical Sequential Sampling Models.

A package based on pymc and bambi to perform Bayesian inference for hierarchical
sequential sampling models.

This file defines the entry class HSSM.
"""

import datetime
import logging
import typing
from copy import deepcopy
from inspect import isclass, signature
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union, cast, get_args

import arviz as az
import bambi as bmb
import cloudpickle as cpickle
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
from pymc.model.transform.conditioning import do
from ssms.config import model_config as ssms_model_config

from hssm._types import LoglikKind, SupportedModels
from hssm.data_validator import DataValidator
from hssm.defaults import (
    INITVAL_JITTER_SETTINGS,
    INITVAL_SETTINGS,
    MissingDataNetwork,
    missing_data_networks_suffix,
)
from hssm.distribution_utils import (
    assemble_callables,
    make_distribution,
    make_family,
    make_likelihood_callable,
    make_missing_data_callable,
)
from hssm.utils import (
    _compute_log_likelihood,
    _get_alias_dict,
    _print_prior,
    _rearrange_data,
    _split_array,
)

from . import plotting
from .config import Config, ModelConfig
from .param import Params
from .param import UserParam as Param

_logger = logging.getLogger("hssm")


class classproperty:
    """A decorator that combines the behavior of @property and @classmethod.

    This decorator allows you to define a property that can be accessed on the class
    itself, rather than on instances of the class. It is useful for defining class-level
    properties that need to perform some computation or access class-level data.

    This implementation is provided for compatibility with Python versions 3.10 through
    3.12, as one cannot combine the @property and @classmethod decorators is across all
    these versions.

    Example
    -------
    class MyClass:
        @classproperty
        def my_class_property(cls):
            return "This is a class property"

    print(MyClass.my_class_property)  # Output: This is a class property
    """

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, instance, owner):  # noqa: D105
        return self.fget(owner)


class HSSM(DataValidator):
    """The basic Hierarchical Sequential Sampling Model (HSSM) class.

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
    choices : optional
        When an `int`, the number of choices that the participants can make. If `2`, the
        choices are [-1, 1] by default. If anything greater than `2`, the choices are
        [0, 1, ..., n_choices - 1] by default. If a `list` is provided, it should be the
        list of choices that the participants can make. Defaults to `2`. If any value
        other than the choices provided is found in the "response" column of the data,
        an error will be raised.
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
    global_formula : optional
        A string that specifies a regressions formula which will be used for all model
        parameters. If you specify parameter-wise regressions in addition, these will
        override the global regression for the respective parameter.
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
        - None: HSSM will use bambi to provide default priors for all parameters. Not
        recommended when you are using hierarchical models.
        The default value is `"safe"`.
    extra_namespace : optional
        Additional user supplied variables with transformations or data to include in
        the environment where the formula is evaluated. Defaults to `None`.
    missing_data : optional
        Specifies whether the model should handle missing data. Can be a `bool` or a
        `float`. If `False`, and if the `rt` column contains in the data -999.0,
        the model will drop these rows and produce a warning. If `True`, the model will
        treat code -999.0 as missing data. If a `float` is provided, the model will
        treat this value as the missing data value. Defaults to `False`.
    deadline : optional
        Specifies whether the model should handle deadline data. Can be a `bool` or a
        `str`. If `False`, the model will not do nothing even if a deadline column is
        provided. If `True`, the model will treat the `deadline` column as deadline
        data. If a `str` is provided, the model will treat this value as the name of the
        deadline column. Defaults to `False`.
    loglik_missing_data : optional
        A likelihood function for missing data. Please see the `loglik` parameter to see
        how to specify the likelihood function this parameter. If nothing is provided,
        a default likelihood function will be used. This parameter is required only if
        either `missing_data` or `deadline` is not `False`. Defaults to `None`.
    process_initvals : optional
        If `True`, the model will process the initial values. Defaults to `True`.
    initval_jitter : optional
        The jitter value for the initial values. Defaults to `0.01`.
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
    initval_jitter
        The jitter value for the initial values.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model: SupportedModels | str = "ddm",
        choices: list[int] | None = None,
        include: list[dict[str, Any] | Param] | None = None,
        model_config: ModelConfig | dict | None = None,
        loglik: (
            str | PathLike | Callable | pytensor.graph.Op | type[pm.Distribution] | None
        ) = None,
        loglik_kind: LoglikKind | None = None,
        p_outlier: float | dict | bmb.Prior | None = 0.05,
        lapse: dict | bmb.Prior | None = bmb.Prior("Uniform", lower=0.0, upper=20.0),
        global_formula: str | None = None,
        link_settings: Literal["log_logit"] | None = None,
        prior_settings: Literal["safe"] | None = "safe",
        extra_namespace: dict[str, Any] | None = None,
        missing_data: bool | float = False,
        deadline: bool | str = False,
        loglik_missing_data: (
            str | PathLike | Callable | pytensor.graph.Op | None
        ) = None,
        process_initvals: bool = True,
        initval_jitter: float = INITVAL_JITTER_SETTINGS["jitter_epsilon"],
        **kwargs,
    ):
        # Attach arguments to the instance
        # so that we can easily define some
        # methods that need to access these
        # arguments (context: pickling / save - load).

        # Define a dict with all call arguments:
        self._init_args = {
            k: v for k, v in locals().items() if k not in ["self", "kwargs"]
        }
        if kwargs:
            self._init_args.update(kwargs)

        self.data = data.copy()
        self._inference_obj: az.InferenceData | None = None
        self._initvals: dict[str, Any] = {}
        self.initval_jitter = initval_jitter
        self._inference_obj_vi: pm.Approximation | None = None
        self._vi_approx = None
        self._map_dict = None
        self.global_formula = global_formula

        self.link_settings = link_settings
        self.prior_settings = prior_settings

        self.missing_data_value = -999.0

        additional_namespace = transformations_namespace.copy()
        if extra_namespace is not None:
            additional_namespace.update(extra_namespace)
        self.additional_namespace = additional_namespace

        # Construct a model_config from defaults
        self.model_config = Config.from_defaults(model, loglik_kind)
        # Update defaults with user-provided config, if any
        if model_config is not None:
            if isinstance(model_config, dict):
                if "choices" not in model_config:
                    if choices is not None:
                        model_config["choices"] = choices
                else:
                    if choices is not None:
                        _logger.info(
                            "choices list provided in both model_config and "
                            "as an argument directly."
                            " Using the one provided in model_config. \n"
                            "We recommend providing choices in model_config."
                        )
            elif isinstance(model_config, ModelConfig):
                if model_config.choices is None:
                    if choices is not None:
                        model_config.choices = choices
                else:
                    if choices is not None:
                        _logger.info(
                            "choices list provided in both model_config and "
                            "as an argument directly."
                            " Using the one provided in model_config. \n"
                            "We recommend providing choices in model_config."
                        )

            self.model_config.update_config(
                model_config
                if isinstance(model_config, ModelConfig)
                else ModelConfig(**model_config)  # also serves as dict validation
            )
        else:
            # Model config is not provided, but at this point was constructed from
            # defaults.
            if model not in typing.get_args(SupportedModels):
                # TODO: ideally use self.supported_models above but mypy doesn't like it
                if choices is not None:
                    self.model_config.update_choices(choices)
                elif model in ssms_model_config:
                    self.model_config.update_choices(
                        ssms_model_config[model]["choices"]
                    )
                    _logger.info(
                        "choices argument passed as None, "
                        "but found %s in ssms-simulators. "
                        "Using choices, from ssm-simulators configs: %s",
                        model,
                        ssms_model_config[model]["choices"],
                    )
            else:
                # Model config already constructed from defaults, and model string is
                # in SupportedModels. So we are guaranteed that choices are in
                # self.model_config already.

                if choices is not None:
                    _logger.info(
                        "Model string is in SupportedModels."
                        " Ignoring choices arguments."
                    )

        # Update loglik with user-provided value
        self.model_config.update_loglik(loglik)
        # Ensure that all required fields are valid
        self.model_config.validate()

        # Set up shortcuts so old code will work
        self.response = self.model_config.response
        self.list_params = self.model_config.list_params
        self.choices = self.model_config.choices
        self.model_name = self.model_config.model_name
        self.loglik = self.model_config.loglik
        self.loglik_kind = self.model_config.loglik_kind
        self.extra_fields = self.model_config.extra_fields

        if self.choices is None:
            raise ValueError(
                "`choices` must be provided either in `model_config` or as an argument."
            )

        self.n_choices = len(self.choices)
        self._pre_check_data_sanity()

        # Process missing data setting
        # AF-TODO: Could be a function in data validator?
        if isinstance(missing_data, float):
            if not ((self.data.rt == missing_data).any()):
                raise ValueError(
                    f"missing_data argument is provided as a float {missing_data}, "
                    f"However, you have no RTs of {missing_data} in your dataset!"
                )
            else:
                self.missing_data = True
                self.missing_data_value = missing_data
        elif isinstance(missing_data, bool):
            if missing_data and (not (self.data.rt == -999.0).any()):
                raise ValueError(
                    "missing_data argument is provided as True, "
                    " so RTs of -999.0 are treated as missing. \n"
                    "However, you have no RTs of -999.0 in your dataset!"
                )
            elif (not missing_data) and (self.data.rt == -999.0).any():
                # self.missing_data = True
                raise ValueError(
                    "Missing data provided as False. \n"
                    "However, you have RTs of -999.0 in your dataset!"
                )
            else:
                self.missing_data = missing_data
        else:
            raise ValueError(
                "missing_data argument must be a bool or a float! \n"
                f"You provided: {type(missing_data)}"
            )

        if isinstance(deadline, str):
            self.deadline = True
            self.deadline_name = deadline
        else:
            self.deadline = deadline
            self.deadline_name = "deadline"

        if (
            not self.missing_data and not self.deadline
        ) and loglik_missing_data is not None:
            raise ValueError(
                "You have specified a loglik_missing_data function, but you have not "
                + "set the missing_data or deadline flag to True."
            )
        self.loglik_missing_data = loglik_missing_data

        # Update data based on missing_data and deadline
        self._handle_missing_data_and_deadline()
        # Set self.missing_data_network based on `missing_data` and `deadline`
        self.missing_data_network = self._set_missing_data_and_deadline(
            self.missing_data, self.deadline, self.data
        )

        if self.deadline:
            self.response.append(self.deadline_name)

        # Process lapse distribution
        self.has_lapse = p_outlier is not None and p_outlier != 0
        self._check_lapse(lapse)
        if self.has_lapse and self.list_params[-1] != "p_outlier":
            self.list_params.append("p_outlier")

        # Process all parameters
        self.params = Params.from_user_specs(
            model=self,
            include=[] if include is None else include,
            kwargs=kwargs,
            p_outlier=p_outlier,
        )

        self._parent = self.params.parent
        self._parent_param = self.params.parent_param

        self.formula, self.priors, self.link = self.params.parse_bambi(model=self)

        # For parameters that have a regression backend, apply bounds at the likelihood
        # level to ensure that the samples that are out of bounds
        # are discarded (replaced with a large negative value).
        self.bounds = {
            name: param.bounds
            for name, param in self.params.items()
            if param.is_regression and param.bounds is not None
        }

        # Set p_outlier and lapse
        self.p_outlier = self.params.get("p_outlier")
        self.lapse = lapse if self.has_lapse else None

        self._post_check_data_sanity()

        self.model_distribution = self._make_model_distribution()

        self.family = make_family(
            self.model_distribution,
            self.list_params,
            self.link,
            self._parent,
        )

        self.model = bmb.Model(
            self.formula,
            data=self.data,
            family=self.family,
            priors=self.priors,  # center_predictors=False
            extra_namespace=self.additional_namespace,
            **kwargs,
        )

        self._aliases = _get_alias_dict(
            self.model, self._parent_param, self.response_c, self.response_str
        )
        self.set_alias(self._aliases)
        self.model.build()

        if process_initvals:
            self._postprocess_initvals_deterministic(initval_settings=INITVAL_SETTINGS)
        if self.initval_jitter > 0:
            self._jitter_initvals(
                jitter_epsilon=self.initval_jitter,
                vector_only=True,
            )

        # Make sure we reset rvs_to_initial_values --> Only None's
        # Otherwise PyMC barks at us when asking to compute likelihoods
        self.pymc_model.rvs_to_initial_values.update(
            {key_: None for key_ in self.pymc_model.rvs_to_initial_values.keys()}
        )
        _logger.info("Model initialized successfully.")

    @classproperty
    def supported_models(cls) -> tuple[SupportedModels, ...]:
        """Get a tuple of all supported models.

        Returns
        -------
        tuple[SupportedModels, ...]
            A tuple containing all supported model names.
        """
        return get_args(SupportedModels)

    @classmethod
    def _store_init_args(cls, *args, **kwargs):
        """Store initialization arguments using signature binding."""
        sig = signature(cls.__init__)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return {k: v for k, v in bound_args.arguments.items() if k != "self"}

    def find_MAP(self, **kwargs):
        """Perform Maximum A Posteriori estimation.

        Returns
        -------
        dict
            A dictionary containing the MAP estimates of the model parameters.
        """
        self._map_dict = pm.find_MAP(model=self.pymc_model, **kwargs)
        return self._map_dict

    def sample(
        self,
        sampler: (
            Literal["mcmc", "nuts_numpyro", "nuts_blackjax", "laplace", "vi"] | None
        ) = None,
        init: str | None = None,
        initvals: str | dict | None = None,
        include_response_params: bool = False,
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
        initvals: optional
            Pass initial values to the sampler. This can be a dictionary of initial
            values for parameters of the model, or a string "map" to use initialization
            at the MAP estimate. If "map" is used, the MAP estimate will be computed if
            not already attached to the base class from prior call to 'find_MAP`.
        include_response_params: optional
            Include parameters of the response distribution in the output. These usually
            take more space than other parameters as there's one of them per
            observation. Defaults to False.
        kwargs
            Other arguments passed to bmb.Model.fit(). Please see [here]
            (https://bambinos.github.io/bambi/api_reference.html#bambi.models.Model.fit)
            for full documentation.

        Returns
        -------
        az.InferenceData | pm.Approximation
            A reference to the `model.traces` object, which stores the traces of the
            last call to `model.sample()`. `model.traces` is an ArviZ `InferenceData`
            instance if `sampler` is `"mcmc"` (default), `"nuts_numpyro"`,
            `"nuts_blackjax"` or "`laplace"`, or an `Approximation` object if `"vi"`.
        """
        # If initvals are None (default)
        # we skip processing initvals here.
        if initvals is not None:
            if isinstance(initvals, dict):
                kwargs["initvals"] = initvals
            else:
                if isinstance(initvals, str):
                    if initvals == "map":
                        if self._map_dict is None:
                            _logger.info(
                                "initvals='map' but no map"
                                "estimate precomputed. \n"
                                "Running map estimation first..."
                            )
                            self.find_MAP()
                            kwargs["initvals"] = self._map_dict
                        else:
                            kwargs["initvals"] = self._map_dict
                else:
                    raise ValueError(
                        "initvals argument must be a dictionary or 'map'"
                        " to use the MAP estimate."
                    )
        else:
            kwargs["initvals"] = self._initvals
            _logger.info("Using default initvals. \n")

        if sampler is None:
            if (
                self.loglik_kind == "approx_differentiable"
                and self.model_config.backend == "jax"
            ):
                sampler = "nuts_numpyro"
            else:
                sampler = "mcmc"

        # supported_samplers = [
        #     "mcmc",
        #     "nuts_numpyro",
        #     "nuts_blackjax",
        #     "laplace",
        # ]  # "vi"]

        # if sampler not in supported_samplers:
        # if sampler == "vi":
        #     raise ValueError(
        #         "For variational inference, please use the `vi()` method instead."
        #     )
        # raise ValueError(
        #     f"Unsupported sampler '{sampler}', must be one of {supported_samplers}"
        # )

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

        # If sampler is finally `numpyro` make sure
        # the jitter argument is set to False
        if sampler == "nuts_numpyro":
            if "nuts_sampler_kwargs" in kwargs:
                if kwargs["nuts_sampler_kwargs"].get("jitter"):
                    _logger.warning(
                        "The jitter argument is set to True. "
                        + "This argument is not supported "
                        + "by the numpyro backend. "
                        + "The jitter argument will be set to False."
                    )
                kwargs["nuts_sampler_kwargs"]["jitter"] = False
            else:
                kwargs["nuts_sampler_kwargs"] = {"jitter": False}

        if self._inference_obj is not None:
            _logger.warning(
                "The model has already been sampled. Overwriting the previous "
                + "inference object. Any previous reference to the inference object "
                + "will still point to the old object."
            )

        if "nuts_sampler" not in kwargs:
            if sampler in ["mcmc", "nuts_numpyro", "nuts_blackjax"]:
                kwargs["nuts_sampler"] = (
                    "pymc" if sampler == "mcmc" else sampler.split("_")[1]
                )

        # Define whether likelihood should be computed
        compute_likelihood = True
        if "idata_kwargs" in kwargs:
            if "log_likelihood" in kwargs["idata_kwargs"]:
                compute_likelihood = kwargs["idata_kwargs"].pop("log_likelihood", True)

        omit_offsets = kwargs.pop("omit_offsets", False)
        self._inference_obj = self.model.fit(
            inference_method=(
                "mcmc"
                if sampler in ["mcmc", "nuts_numpyro", "nuts_blackjax"]
                else sampler
            ),
            init=init,
            include_response_params=include_response_params,
            omit_offsets=omit_offsets,
            **kwargs,
        )

        # Separate out log likelihood computation
        if compute_likelihood:
            self.log_likelihood(self._inference_obj, inplace=True)

        # Subset data vars in posterior
        self._clean_posterior_group(idata=self._inference_obj)
        return self.traces

    def vi(
        self,
        method: str = "advi",
        niter: int = 10000,
        draws: int = 1000,
        return_idata: bool = True,
        ignore_mcmc_start_point_defaults=False,
        **vi_kwargs,
    ) -> pm.Approximation | az.InferenceData:
        """Perform Variational Inference.

        Parameters
        ----------
        niter : int
            The number of iterations to run the VI algorithm. Defaults to 3000.
        method : str
            The method to use for VI. Can be one of "advi" or "fullrank_advi", "svgd",
            "asvgd".Defaults to "advi".
        draws : int
            The number of samples to draw from the posterior distribution.
            Defaults to 1000.
        return_idata : bool
            If True, returns an InferenceData object. Otherwise, returns the
            approximation object directly. Defaults to True.

        Returns
        -------
            pm.Approximation or az.InferenceData: The mean field approximation object.
        """
        if self.loglik_kind == "analytical":
            _logger.warning(
                "VI is not recommended for the analytical likelihood,"
                " since gradients can be brittle."
            )
        elif self.loglik_kind == "blackbox":
            raise ValueError(
                "VI is not supported for blackbox likelihoods, "
                " since likelihood gradients are needed!"
            )

        if ("start" not in vi_kwargs) and not ignore_mcmc_start_point_defaults:
            _logger.info("Using MCMC starting point defaults.")
            vi_kwargs["start"] = self._initvals

        # Run variational inference directly from pymc model
        with self.pymc_model:
            self._vi_approx = pm.fit(n=niter, method=method, **vi_kwargs)

        # Sample from the approximate posterior
        if self._vi_approx is not None:
            self._inference_obj_vi = self._vi_approx.sample(draws)

        # Post-processing
        self._clean_posterior_group(idata=self._inference_obj_vi)

        # Return the InferenceData object if return_idata is True
        if return_idata:
            return self._inference_obj_vi
        # Otherwise return the appromation object directly
        return self.vi_approx

    def _clean_posterior_group(self, idata: az.InferenceData | None = None):
        """Clean up the posterior group of the InferenceData object.

        Parameters
        ----------
        idata : az.InferenceData
            The InferenceData object to clean up. If None, the last InferenceData object
            will be used.
        """
        # # Logic behind which variables to keep:
        # # We essentially want to get rid of
        # # all the trial-wise variables.

        # # We drop all distributional components, IF they are deterministics
        # # (in which case they will be trial wise systematically)
        # # and we keep distributional components, IF they are
        # # basic random-variabels (in which case they should never
        # # appear trial-wise).
        if idata is None:
            raise ValueError(
                "The InferenceData object is None. Cannot clean up the posterior group."
            )
        elif not hasattr(idata, "posterior"):
            raise ValueError(
                "The InferenceData object does not have a posterior group. "
                + "Cannot clean up the posterior group."
            )

        vars_to_keep = set(idata["posterior"].data_vars.keys()).difference(
            set(
                key_
                for key_ in self.model.distributional_components.keys()
                if key_ in [var_.name for var_ in self.pymc_model.deterministics]
            )
        )
        vars_to_keep_clean = [var_ for var_ in vars_to_keep if "_mean" not in var_]

        setattr(
            idata,
            "posterior",
            idata["posterior"][vars_to_keep_clean],
        )

    def log_likelihood(
        self,
        idata: az.InferenceData | None = None,
        data: pd.DataFrame | None = None,
        inplace: bool = True,
        keep_likelihood_params: bool = False,
    ) -> az.InferenceData | None:
        """Compute the log likelihood of the model.

        Parameters
        ----------
        idata : optional
            The `InferenceData` object returned by `HSSM.sample()`. If not provided,
        data : optional
            A pandas DataFrame with values for the predictors that are used to obtain
            out-of-sample predictions. If omitted, the original dataset is used.
        inplace : optional
            If `True` will modify idata in-place and append a `log_likelihood` group to
            `idata`. Otherwise, it will return a copy of idata with the predictions
            added, by default True.
        keep_likelihood_params : optional
            If `True`, the trial wise likelihood parameters that are computed
            on route to getting the log likelihood are kept in the `idata` object.
            Defaults to False. See also the method `add_likelihood_parameters_to_idata`.

        Returns
        -------
        az.InferenceData | None
            InferenceData or None
        """
        if self._inference_obj is None and idata is None:
            raise ValueError(
                "Neither has the model been sampled yet nor"
                + " an idata object has been provided."
            )

        if idata is None:
            if self._inference_obj is None:
                raise ValueError(
                    "The model has not been sampled yet. "
                    + "Please provide an idata object."
                )
            else:
                idata = self._inference_obj

        # Actual likelihood computation
        idata = _compute_log_likelihood(self.model, idata, data, inplace)

        # clean up posterior:
        if not keep_likelihood_params:
            self._clean_posterior_group(idata=idata)

        if inplace:
            return None
        else:
            return idata

    def add_likelihood_parameters_to_idata(
        self,
        idata: az.InferenceData | None = None,
        inplace: bool = False,
    ) -> az.InferenceData | None:
        """Add likelihood parameters to the InferenceData object.

        Parameters
        ----------
        idata : az.InferenceData
            The InferenceData object returned by HSSM.sample().
        inplace : bool
            If True, the likelihood parameters are added to idata in-place. Otherwise,
            a copy of idata with the likelihood parameters added is returned.
            Defaults to False.

        Returns
        -------
        az.InferenceData | None
            InferenceData or None
        """
        if idata is None:
            if self._inference_obj is None:
                raise ValueError("No idata provided and model not yet sampled!")
            else:
                idata = self.model._compute_likelihood_params(  # pylint: disable=protected-access
                    deepcopy(self._inference_obj)
                    if not inplace
                    else self._inference_obj
                )
        else:
            idata = self.model._compute_likelihood_params(  # pylint: disable=protected-access
                deepcopy(idata) if not inplace else idata
            )
        return idata

    def sample_posterior_predictive(
        self,
        idata: az.InferenceData | None = None,
        data: pd.DataFrame | None = None,
        inplace: bool = True,
        include_group_specific: bool = True,
        kind: Literal["response", "response_params"] = "response",
        draws: int | float | list[int] | np.ndarray | None = None,
        safe_mode: bool = True,
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
        kind: optional
            Indicates the type of prediction required. Can be `"response_params"` or
            `"response"`. The first returns draws from the posterior distribution of the
            likelihood parameters, while the latter returns the draws from the posterior
            predictive distribution (i.e. the posterior probability distribution for a
            new observation) in addition to the posterior distribution. Defaults to
            "response_params".
        draws: optional
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
        safe_mode: bool
            If True, the function will split the draws into chunks of 10 to avoid memory
            issues. Defaults to True.

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
            _logger.info(
                "idata=None, we use the traces assigned to the HSSM object as idata."
            )

        if idata is not None:
            if "posterior_predictive" in idata.groups():
                del idata["posterior_predictive"]
                _logger.warning(
                    "pre-existing posterior_predictive group deleted from idata. \n"
                )

        if self._check_extra_fields(data):
            self._update_extra_fields(data)

        if isinstance(draws, np.ndarray):
            draws = draws.astype(int)
        elif isinstance(draws, list):
            draws = np.array(draws).astype(int)
        elif isinstance(draws, int | float):
            draws = np.arange(int(draws))
        elif draws is None:
            draws = idata["posterior"].draw.values
        else:
            raise ValueError(
                "draws must be an integer, " + "a list of integers, or a numpy array."
            )

        assert isinstance(draws, np.ndarray)

        # Make a copy of idata, set the `posterior` group to be a random sub-sample
        # of the original (draw dimension gets sub-sampled)

        idata_copy = idata.copy()

        if (draws.shape != idata["posterior"].draw.values.shape) or (
            (draws.shape == idata["posterior"].draw.values.shape)
            and not np.allclose(draws, idata["posterior"].draw.values)
        ):
            # Reassign posterior to sub-sampled version
            setattr(idata_copy, "posterior", idata["posterior"].isel(draw=draws))

        if kind == "response":
            # If we run kind == 'response' we actually run the observation RV
            if safe_mode:
                # safe mode splits the draws into chunks of 10 to avoid
                # memory issues (TODO: Figure out the source of memory issues)
                split_draws = _split_array(
                    idata_copy["posterior"].draw.values, divisor=10
                )

                posterior_predictive_list = []
                for samples_tmp in split_draws:
                    tmp_posterior = idata["posterior"].sel(draw=samples_tmp)
                    setattr(idata_copy, "posterior", tmp_posterior)
                    self.model.predict(
                        idata_copy, kind, data, True, include_group_specific
                    )
                    posterior_predictive_list.append(idata_copy["posterior_predictive"])

                if inplace:
                    idata.add_groups(
                        posterior_predictive=xr.concat(
                            posterior_predictive_list, dim="draw"
                        )
                    )
                    # for inplace, we don't return anything
                    return None
                else:
                    # Reassign original posterior to idata_copy
                    setattr(idata_copy, "posterior", idata["posterior"])
                    # Add new posterior predictive group to idata_copy
                    del idata_copy["posterior_predictive"]
                    idata_copy.add_groups(
                        posterior_predictive=xr.concat(
                            posterior_predictive_list, dim="draw"
                        )
                    )
                    return idata_copy
            else:
                if inplace:
                    # If not safe-mode
                    # We call .predict() directly without any
                    # chunking of data.

                    # .predict() is called on the copy of idata
                    # since we still subsampled (or assigned) the draws
                    self.model.predict(
                        idata_copy, kind, data, True, include_group_specific
                    )

                    # posterior predictive group added to idata
                    idata.add_groups(
                        posterior_predictive=idata_copy["posterior_predictive"]
                    )
                    # don't return anything if inplace
                    return None
                else:
                    # Not safe mode and not inplace
                    # Function acts as very thin wrapper around
                    # .predict(). It just operates on the
                    # idata_copy object
                    return self.model.predict(
                        idata_copy, kind, data, False, include_group_specific
                    )
        elif kind == "response_params":
            # If kind == 'response_params', we don't need to run the RV directly,
            # there shouldn't really be any significant memory issues here,
            # we can simply ignore settings, since the computational overhead
            # should be very small --> nudges user towards good outputs.
            _logger.warning(
                "The kind argument is set to 'mean', but 'draws' argument "
                + "is not None: The draws argument will be ignored!"
            )
            return self.model.predict(
                idata, kind, data, inplace, include_group_specific
            )
        else:
            raise ValueError("`kind` must be either 'response' or 'response_params'.")

    def plot_predictive(self, **kwargs) -> mpl.axes.Axes | sns.FacetGrid:
        """Produce a posterior predictive plot.

        Equivalent to calling `hssm.plotting.plot_predictive()` with the
        model. Please see that function for
        [full documentation][hssm.plotting.plot_predictive].

        Returns
        -------
        mpl.axes.Axes | sns.FacetGrid
            The matplotlib axis or seaborn FacetGrid object containing the plot.
        """
        return plotting.plot_predictive(self, **kwargs)

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

    def predict(self, **kwargs) -> az.InferenceData:
        """Generate samples from the predictive distribution."""
        return self.model.predict(**kwargs)

    def sample_do(
        self, params: dict[str, Any], draws: int = 100, return_model=False, **kwargs
    ) -> az.InferenceData | tuple[az.InferenceData, pm.Model]:
        """Generate samples from the predictive distribution using the `do-operator`."""
        do_model = do(self.pymc_model, params)
        do_idata = pm.sample_prior_predictive(model=do_model, draws=draws, **kwargs)

        # clean up `rt,response_mean` to `v`
        do_idata = self._drop_parent_str_from_idata(idata=do_idata)

        # rename otherwise inconsistentdims and coords
        if "rt,response_extra_dim_0" in do_idata["prior_predictive"].dims:
            setattr(
                do_idata,
                "prior_predictive",
                do_idata["prior_predictive"].rename_dims(
                    {"rt,response_extra_dim_0": "rt,response_dim"}
                ),
            )
        if "rt,response_extra_dim_0" in do_idata["prior_predictive"].coords:
            setattr(
                do_idata,
                "prior_predictive",
                do_idata["prior_predictive"].rename_vars(
                    name_dict={"rt,response_extra_dim_0": "rt,response_dim"}
                ),
            )

        if return_model:
            return do_idata, do_model
        return do_idata

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
        prior_predictive = self.model.prior_predictive(
            draws, var_names, omit_offsets, random_seed
        )

        # AF-COMMENT: Not sure if necessary to include the
        # mean prior here (which adds deterministics that
        # could be recomputed elsewhere)
        prior_predictive.add_groups(posterior=prior_predictive.prior)
        self.model.predict(prior_predictive, kind="mean", inplace=True)

        # clean
        setattr(prior_predictive, "prior", prior_predictive["posterior"])
        del prior_predictive["posterior"]

        if self._inference_obj is None:
            self._inference_obj = prior_predictive
        else:
            self._inference_obj.extend(prior_predictive)

        # clean up `rt,response_mean` to `v`
        idata = self._drop_parent_str_from_idata(idata=self._inference_obj)

        # rename otherwise inconsistentdims and coords
        if "rt,response_extra_dim_0" in idata["prior_predictive"].dims:
            setattr(
                idata,
                "prior_predictive",
                idata["prior_predictive"].rename_dims(
                    {"rt,response_extra_dim_0": "rt,response_dim"}
                ),
            )
        if "rt,response_extra_dim_0" in idata["prior_predictive"].coords:
            setattr(
                idata,
                "prior_predictive",
                idata["prior_predictive"].rename_vars(
                    name_dict={"rt,response_extra_dim_0": "rt,response_dim"}
                ),
            )

        # Update self._inference_obj to match the cleaned idata
        self._inference_obj = idata
        return deepcopy(self._inference_obj)

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

    @property
    def response_c(self) -> str:
        """Return the response variable names in c() format."""
        return f"c({', '.join(self.response)})"

    @property
    def response_str(self) -> str:
        """Return the response variable names in string format."""
        return ",".join(self.response)

    # NOTE: can't annotate return type because the graphviz dependency is optional
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
        """
        graph = self.model.graph(formatting, name, figsize, dpi, fmt)

        parent_param = self._parent_param
        if parent_param.is_regression:
            return graph

        # Modify the graph
        # 1. Remove all nodes and edges related to `{parent}_mean`:
        graph.body = [
            item for item in graph.body if f"{parent_param.name}_mean" not in item
        ]
        # 2. Add a new edge from parent to response
        graph.edge(parent_param.name, self.response_str)

        return graph

    def compile_logp(self, keep_transformed: bool = False, **kwargs):
        """Compile the log probability function for the model.

        Parameters
        ----------
        keep_transformed : bool, optional
            If True, keeps the transformed variables in the compiled function.
            If False, removes value transforms before compilation.
            Defaults to False.
        **kwargs
            Additional keyword arguments passed to PyMC's compile_logp:
            - vars: List of variables. Defaults to None (all variables).
            - jacobian: Whether to include log(|det(dP/dQ)|) term for
            transformed variables. Defaults to True.
            - sum: Whether to sum all terms instead of returning a vector.
            Defaults to True.

        Returns
        -------
        callable
            A compiled function that computes the model log probability.
        """
        if keep_transformed:
            return self.pymc_model.compile_logp(
                vars=kwargs.get("vars", None),
                jacobian=kwargs.get("jacobian", True),
                sum=kwargs.get("sum", True),
            )
        else:
            new_model = pm.model.transform.conditioning.remove_value_transforms(
                self.pymc_model
            )
            return new_model.compile_logp(
                vars=kwargs.get("vars", None),
                jacobian=kwargs.get("jacobian", True),
                sum=kwargs.get("sum", True),
            )

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
        if not isinstance(data, az.InferenceData):
            raise TypeError("data must be an InferenceData object.")

        if not include_deterministic:
            var_names = list(
                set([var.name for var in self.pymc_model.free_RVs]).intersection(
                    set(list(data["posterior"].data_vars.keys()))
                )
            )
            # var_names = self._get_deterministic_var_names(data)
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
        if not isinstance(data, az.InferenceData):
            raise TypeError("data must be an InferenceData object.")

        if not include_deterministic:
            var_names = list(
                set([var.name for var in self.pymc_model.free_RVs]).intersection(
                    set(list(data["posterior"].data_vars.keys()))
                )
            )
            # var_names = self._get_deterministic_var_names(data)
            if var_names:
                kwargs["var_names"] = list(set(var_names + kwargs.get("var_names", [])))
        return az.summary(data, **kwargs)

    def initial_point(self, transformed: bool = False) -> dict[str, np.ndarray]:
        """Compute the initial point of the model.

        This is a slightly altered version of pm.initial_point.initial_point().

        Parameters
        ----------
        transformed : bool, optional
            If True, return the initial point in transformed space.

        Returns
        -------
        dict
            A dictionary containing the initial point of the model parameters.
        """
        fn = pm.initial_point.make_initial_point_fn(
            model=self.pymc_model, return_transformed=transformed
        )
        return pm.model.Point(fn(None), model=self.pymc_model)

    def restore_traces(
        self, traces: az.InferenceData | pm.Approximation | str | PathLike
    ) -> None:
        """Restore traces from an InferenceData object or a .netcdf file.

        Parameters
        ----------
        traces
            An InferenceData object or a path to a file containing the traces.
        """
        if isinstance(traces, pm.Approximation):
            self._inference_obj_vi = traces
            return

        if isinstance(traces, (str, PathLike)):
            traces = az.from_netcdf(traces)
        self._inference_obj = cast("az.InferenceData", traces)

    def restore_vi_traces(
        self, traces: az.InferenceData | pm.Approximation | str | PathLike
    ) -> None:
        """Restore VI traces from an InferenceData object or a .netcdf file.

        Parameters
        ----------
        traces
            An InferenceData object or a path to a file containing the VI traces.
        """
        if isinstance(traces, pm.Approximation):
            self._inference_obj_vi = traces
            return

        if isinstance(traces, (str, PathLike)):
            traces = az.from_netcdf(traces)
        self._inference_obj_vi = cast("az.InferenceData", traces)

    def save_model(
        self,
        model_name: str | None = None,
        allow_absolute_base_path: bool = False,
        base_path: str | Path = "hssm_models",
        save_idata_only: bool = False,
    ) -> None:
        """Save a HSSM model instance and its inference results to disk.

        Parameters
        ----------
        model : HSSM
            The HSSM model instance to save
        model_name : str | None
            Name to use for the saved model files.
            If None, will use model.model_name with timestamp
        allow_absolute_base_path : bool
            Whether to allow absolute paths for base_path
        base_path : str | Path
            Base directory to save model files in.
            Must be relative path if allow_absolute_base_path=False
        save_idata_only: bool = False,
            Whether to save the model class instance itself

        Raises
        ------
        ValueError
            If base_path is absolute and allow_absolute_base_path=False
        """
        # check if base_path is absolute
        if not allow_absolute_base_path:
            if str(base_path).startswith("/"):
                raise ValueError(
                    "base_path must be a relative path"
                    " if allow_absolute_base_path is False"
                )

        if model_name is None:
            # Get date string format as suffix to model name
            model_name = (
                self.model_name
                + "_"
                + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            )

        # check if folder by name model_name exists
        model_name = model_name.replace(" ", "_")
        model_path = Path(base_path).joinpath(model_name)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save model to pickle file
        if not save_idata_only:
            with open(model_path.joinpath("model.pkl"), "wb") as f:
                cpickle.dump(self, f)

        # Save traces to netcdf file
        if self._inference_obj is not None:
            az.to_netcdf(self._inference_obj, model_path.joinpath("traces.nc"))

        # Save vi_traces to netcdf file
        if self._inference_obj_vi is not None:
            az.to_netcdf(self._inference_obj_vi, model_path.joinpath("vi_traces.nc"))

    @classmethod
    def load_model(
        cls, path: Union[str, Path]
    ) -> Union["HSSM", dict[str, Optional[az.InferenceData]]]:
        """Load a HSSM model instance and its inference results from disk.

        Parameters
        ----------
        path : str | Path
            Path to the model directory or model.pkl file. If a directory is provided,
            will look for model.pkl, traces.nc and vi_traces.nc files within it.

        Returns
        -------
        HSSM
            The loaded HSSM model instance with inference results attached if available.
        """
        # Convert path to Path object
        path = Path(path)

        # If path points to a file, assume it's model.pkl
        if path.is_file():
            model_dir = path.parent
            model_path = path
        else:
            # Path points to directory
            model_dir = path
            model_path = model_dir.joinpath("model.pkl")

        # check if model_dir exists
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

        # check if model.pkl exists raise logging information if not
        if not model_path.exists():
            _logger.info(
                f"model.pkl file does not exist in {model_dir}. "
                "Attempting to load traces only."
            )
            if (not model_dir.joinpath("traces.nc").exists()) and (
                not model_dir.joinpath("vi_traces.nc").exists()
            ):
                raise FileNotFoundError(f"No traces found in {model_dir}.")
            else:
                idata_dict = cls.load_model_idata(model_dir)
                return idata_dict
        else:
            # Load model from pickle file
            with open(model_path, "rb") as f:
                model = cpickle.load(f)

            # Load traces if they exist
            traces_path = model_dir.joinpath("traces.nc")
            if traces_path.exists():
                model.restore_traces(traces_path)

            # Load VI traces if they exist
            vi_traces_path = model_dir.joinpath("vi_traces.nc")
            if vi_traces_path.exists():
                model.restore_vi_traces(vi_traces_path)
        return model

    @classmethod
    def load_model_idata(cls, path: str | Path) -> dict[str, az.InferenceData | None]:
        """Load the traces from a model directory.

        Parameters
        ----------
        path : str | Path
            Path to the model directory containing traces.nc and/or vi_traces.nc files.

        Returns
        -------
        dict[str, az.InferenceData | None]
            A dictionary with keys "idata_mcmc" and "idata_vi" containing the traces
            from the model directory. If the traces do not exist, the corresponding
            value will be None.
        """
        idata_dict: dict[str, az.InferenceData | None] = {}
        model_dir = Path(path)
        # check if path exists
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

        # check if traces.nc exists
        traces_path = model_dir.joinpath("traces.nc")
        if not traces_path.exists():
            _logger.warning(f"traces.nc file does not exist in {model_dir}.")
            idata_dict["idata_mcmc"] = None
        else:
            idata_dict["idata_mcmc"] = az.from_netcdf(traces_path)

        # check if vi_traces.nc exists
        vi_traces_path = model_dir.joinpath("vi_traces.nc")
        if not vi_traces_path.exists():
            _logger.warning(f"vi_traces.nc file does not exist in {model_dir}.")
            idata_dict["idata_vi"] = None
        else:
            idata_dict["idata_vi"] = az.from_netcdf(vi_traces_path)

        return idata_dict

    def __getstate__(self):
        """Get the state of the model for pickling.

        This method is called when pickling the model.
        It returns a dictionary containing the constructor
        arguments needed to recreate the model instance.

        Returns
        -------
        dict
            A dictionary containing the constructor arguments
            under the key 'constructor_args'.
        """
        state = {"constructor_args": self._init_args}
        return state

    def __setstate__(self, state):
        """Set the state of the model when unpickling.

        This method is called when unpickling the model. It creates a new instance
        of HSSM using the constructor arguments stored in the state dictionary,
        and copies its attributes to the current instance.

        Parameters
        ----------
        state : dict
            A dictionary containing the constructor arguments under the key
            'constructor_args'.
        """
        new_instance = HSSM(**state["constructor_args"])
        self.__dict__ = new_instance.__dict__

    def __repr__(self) -> str:
        """Create a representation of the model."""
        output = [
            "Hierarchical Sequential Sampling Model",
            f"Model: {self.model_name}\n",
            f"Response variable: {self.response_str}",
            f"Likelihood: {self.loglik_kind}",
            f"Observations: {len(self.data)}\n",
            "Parameters:\n",
        ]

        for param in self.params.values():
            if param.name == "p_outlier":
                continue
            output.append(f"{param.name}:")

            component = self.model.components[param.name]

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
            output.append(
                " (ignored due to link function)"
                if self.link_settings is not None
                else ""
            )

        # TODO: Handle p_outlier regression correctly here.
        if self.p_outlier is not None:
            output.append("")
            output.append(f"Lapse probability: {self.p_outlier.prior}")
            output.append(f"Lapse distribution: {self.lapse}")

        return "\n".join(output)

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
            The trace of the model after the last call to `sample()`.
        """
        if not self._inference_obj:
            raise ValueError("Please sample the model first.")

        return self._inference_obj

    @property
    def vi_idata(self) -> az.InferenceData:
        """Return the variational inference approximation object.

        Raises
        ------
        ValueError
            If the model has not been sampled yet.

        Returns
        -------
        az.InferenceData
            The variational inference approximation object.
        """
        if not self._inference_obj_vi:
            raise ValueError(
                "Please run variational inference first, "
                "no variational posterior attached."
            )

        return self._inference_obj_vi

    @property
    def vi_approx(self) -> pm.Approximation:
        """Return the variational inference approximation object.

        Raises
        ------
        ValueError
            If the model has not been sampled yet.

        Returns
        -------
        pm.Approximation
            The variational inference approximation object.
        """
        if not self._vi_approx:
            raise ValueError(
                "Please run variational inference first, "
                "no variational approximation attached."
            )

        return self._vi_approx

    @property
    def map(self) -> dict:
        """Return the MAP estimates of the model parameters.

        Raises
        ------
        ValueError
            If the model has not been sampled yet.

        Returns
        -------
        dict
            A dictionary containing the MAP estimates of the model parameters.
        """
        if not self._map_dict:
            raise ValueError("Please compute map first.")

        return self._map_dict

    @property
    def initvals(self) -> dict:
        """Return the initial values of the model parameters for sampling.

        Returns
        -------
        dict
            A dictionary containing the initial values of the model parameters.
            This dict serves as the default for initial values, and can be passed
            directly to the `.sample()` function.
        """
        if self._initvals == {}:
            self._initvals = self.initial_point()
        return self._initvals

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

        params_is_reg = [
            param.is_vector
            for param_name, param in self.params.items()
            if param_name != "p_outlier"
        ]
        if self.extra_fields is not None:
            params_is_reg += [True for _ in self.extra_fields]

        if self.loglik_kind == "approx_differentiable":
            if self.model_config.backend == "jax":
                likelihood_callable = make_likelihood_callable(
                    loglik=self.loglik,
                    loglik_kind="approx_differentiable",
                    backend="jax",
                    params_is_reg=params_is_reg,
                )
            else:
                likelihood_callable = make_likelihood_callable(
                    loglik=self.loglik,
                    loglik_kind="approx_differentiable",
                    backend=self.model_config.backend,
                )
        else:
            likelihood_callable = make_likelihood_callable(
                loglik=self.loglik,
                loglik_kind=self.loglik_kind,
                backend=self.model_config.backend,
            )

        self.loglik = likelihood_callable

        # Make the callable for missing data
        # And assemble it with the callable for the likelihood
        if self.missing_data_network != MissingDataNetwork.NONE:
            if self.missing_data_network == MissingDataNetwork.OPN:
                params_only = False
            elif self.missing_data_network == MissingDataNetwork.CPN:
                params_only = True
            else:
                params_only = None

            if self.loglik_missing_data is None:
                self.loglik_missing_data = (
                    self.model_name
                    + missing_data_networks_suffix[self.missing_data_network]
                    + ".onnx"
                )

            backend_tmp: Literal["pytensor", "jax", "other"] | None = (
                "jax"
                if self.model_config.backend != "pytensor"
                else self.model_config.backend
            )
            missing_data_callable = make_missing_data_callable(
                self.loglik_missing_data, backend_tmp, params_is_reg, params_only
            )

            self.loglik_missing_data = missing_data_callable

            self.loglik = assemble_callables(
                self.loglik,
                self.loglik_missing_data,
                params_only,
                has_deadline=self.deadline,
            )

        if self.missing_data:
            _logger.info(
                "Re-arranging data to separate missing and observed datapoints. "
                "Missing data (rt == %s) will be on top, "
                "observed datapoints follow.",
                self.missing_data_value,
            )

        self.data = _rearrange_data(self.data)
        return make_distribution(
            rv=self.model_config.rv or self.model_name,
            loglik=self.loglik,
            list_params=self.list_params,
            bounds=self.bounds,
            lapse=self.lapse,
            extra_fields=(
                None
                if not self.extra_fields
                else [deepcopy(self.data[field].values) for field in self.extra_fields]
            ),
        )

    def _get_deterministic_var_names(self, idata) -> list[str]:
        """Filter out the deterministic variables in var_names."""
        var_names = [
            f"~{param_name}"
            for param_name, param in self.params.items()
            if (param.is_regression)
        ]

        if f"{self._parent}_mean" in idata["posterior"].data_vars:
            var_names.append(f"~{self._parent}_mean")

        # Parent parameters (always regression implicitly)
        # which don't have a formula attached
        # should be dropped from var_names, since the actual
        # parent name shows up as a regression.
        if f"{self._parent}" in idata["posterior"].data_vars:
            if self.params[self._parent].formula is None:
                # Drop from var_names
                var_names = [var for var in var_names if var != f"~{self._parent}"]

        return var_names

    def _drop_parent_str_from_idata(
        self, idata: az.InferenceData | None
    ) -> az.InferenceData:
        """Drop the parent_str variable from an InferenceData object.

        Parameters
        ----------
        idata
            The InferenceData object to be modified.

        Returns
        -------
        xr.Dataset
            The modified InferenceData object.
        """
        if idata is None:
            raise ValueError("Please provide an InferenceData object.")
        else:
            for group in idata.groups():
                if ("rt,response_mean" in idata[group].data_vars) and (
                    self._parent not in idata[group].data_vars
                ):
                    setattr(
                        idata,
                        group,
                        idata[group].rename({"rt,response_mean": self._parent}),
                    )
            return idata

    def _postprocess_initvals_deterministic(
        self, initval_settings: dict = INITVAL_SETTINGS
    ) -> None:
        """Set initial values for subset of parameters."""
        self._initvals = self.initial_point()
        # Consider case where link functions are set to 'log_logit'
        # or 'None'
        if self.link_settings not in ["log_logit", None]:
            _logger.info(
                "Not preprocessing initial values, "
                + "because none of the two standard link settings are chosen!"
            )
            return None

        # Set initial values for particular parameters
        for name_, starting_value in self.pymc_model.initial_point().items():
            # strip name of `_log__` and `_interval__` suffixes
            name_tmp = name_.replace("_log__", "").replace("_interval__", "")

            # We need to check if the parameter is actually backed by
            # a regression.

            # If not, we don't actually apply a link function to it as per default.
            # Therefore we need to apply the initial value strategy corresponding
            # to 'None' link function.

            # If the user actively supplies a link function, the user
            # should also have supplied an initial value insofar it matters.

            if self.params[self._get_prefix(name_tmp)].is_regression:
                param_link_setting = self.link_settings
            else:
                param_link_setting = None
            if name_tmp in initval_settings[param_link_setting].keys():
                if self._check_if_initval_user_supplied(name_tmp):
                    _logger.info(
                        "User supplied initial value detected for %s, \n"
                        " skipping overwrite with default value.",
                        name_tmp,
                    )
                    continue

                # Apply specific settings from initval_settings dictionary
                dtype = self._initvals[name_tmp].dtype
                self._initvals[name_tmp] = np.array(
                    initval_settings[param_link_setting][name_tmp]
                ).astype(dtype)

    def _get_prefix(self, name_str: str) -> str:
        """Get parameters wise link setting function from parameter prefix."""
        # `p_outlier` is the only basic parameter floating around that has
        # an underscore in it's name.
        # We need to handle it separately. (Renaming might be better...)
        if "_" in name_str:
            if "p_outlier" not in name_str:
                name_str_prefix = name_str.split("_")[0]
            else:
                name_str_prefix = "p_outlier"
        else:
            name_str_prefix = name_str
        return name_str_prefix

    def _check_if_initval_user_supplied(
        self,
        name_str: str,
        return_value: bool = False,
    ) -> bool | float | int | np.ndarray | dict[str, Any] | None:
        """Check if initial value is user-supplied."""
        # The function assumes that the name_str is either raw parameter name
        # or `paramname_Intercept`, because we only really provide special default
        # initial values for those types of parameters

        # `p_outlier` is the only basic parameter floating around that has
        # an underscore in it's name.
        # We need to handle it separately. (Renaming might be better...)
        if "_" in name_str:
            if "p_outlier" not in name_str:
                name_str_prefix = name_str.split("_")[0]
                # name_str_suffix = "".join(name_str.split("_")[1:])
                name_str_suffix = name_str[len(name_str_prefix + "_") :]
            else:
                name_str_prefix = "p_outlier"
                if name_str == "p_outlier":
                    name_str_suffix = ""
                else:
                    # name_str_suffix = "".join(name_str.split("_")[2:])
                    name_str_suffix = name_str[len("p_outlier_") :]
        else:
            name_str_prefix = name_str
            name_str_suffix = ""

        tmp_param = name_str_prefix
        if tmp_param == self._parent:
            # If the parameter was parent it is automatically treated as a
            # regression.
            if not name_str_suffix:
                # No suffix --> Intercept
                if isinstance(prior_tmp := self.params[tmp_param].prior, dict):
                    args_tmp = getattr(prior_tmp["Intercept"], "args")
                    if return_value:
                        return args_tmp.get("initval", None)
                    else:
                        return "initval" in args_tmp
                else:
                    if return_value:
                        return None
                    return False
            else:
                # If the parameter has a suffix --> use it
                if isinstance(prior_tmp := self.params[tmp_param].prior, dict):
                    args_tmp = getattr(prior_tmp[name_str_suffix], "args")
                    if return_value:
                        return args_tmp.get("initval", None)
                    else:
                        return "initval" in args_tmp
                else:
                    if return_value:
                        return None
                    else:
                        return False
        else:
            # If the parameter is not a parent, it is treated as a regression
            # only when actively specified as such.
            if not name_str_suffix:
                # If no suffix --> treat as basic parameter.
                if isinstance(self.params[tmp_param].prior, float) or isinstance(
                    self.params[tmp_param].prior, np.ndarray
                ):
                    if return_value:
                        return self.params[tmp_param].prior
                    else:
                        return True
                elif isinstance(self.params[tmp_param].prior, bmb.Prior):
                    args_tmp = getattr(self.params[tmp_param].prior, "args")
                    if "initval" in args_tmp:
                        if return_value:
                            return args_tmp["initval"]
                        else:
                            return True
                    else:
                        if return_value:
                            return None
                        else:
                            return False
                else:
                    if return_value:
                        return None
                    else:
                        return False
            else:
                # If suffix --> treat as regression and use suffix
                if isinstance(prior_tmp := self.params[tmp_param].prior, dict):
                    args_tmp = getattr(prior_tmp[name_str_suffix], "args")
                    if return_value:
                        return args_tmp.get("initval", None)
                    else:
                        return "initval" in args_tmp
                else:
                    if return_value:
                        return None
                    else:
                        return False

    def _jitter_initvals(
        self, jitter_epsilon: float = 0.01, vector_only: bool = False
    ) -> None:
        """Apply controlled jitter to initial values."""
        if vector_only:
            self.__jitter_initvals_vector_only(jitter_epsilon)
        else:
            self.__jitter_initvals_all(jitter_epsilon)

    def __jitter_initvals_vector_only(self, jitter_epsilon: float) -> None:
        # Note: Calling our initial point function here
        # --> operate on untransformed variables
        initial_point_dict = self.initvals
        for name_, starting_value in initial_point_dict.items():
            name_tmp = name_.replace("_log__", "").replace("_interval__", "")
            if starting_value.ndim != 0 and starting_value.shape[0] != 1:
                starting_value_tmp = starting_value + np.random.uniform(
                    -jitter_epsilon, jitter_epsilon, starting_value.shape
                ).astype(np.float32)

                # Note: self._initvals shouldn't be None when this is called
                dtype = self._initvals[name_tmp].dtype
                self._initvals[name_tmp] = np.array(starting_value_tmp).astype(dtype)

    def __jitter_initvals_all(self, jitter_epsilon: float) -> None:
        # Note: Calling our initial point function here
        # --> operate on untransformed variables
        initial_point_dict = self.initvals
        # initial_point_dict = self.pymc_model.initial_point()
        for name_, starting_value in initial_point_dict.items():
            name_tmp = name_.replace("_log__", "").replace("_interval__", "")
            starting_value_tmp = starting_value + np.random.uniform(
                -jitter_epsilon, jitter_epsilon, starting_value.shape
            ).astype(np.float32)

            dtype = self.initvals[name_tmp].dtype
            self._initvals[name_tmp] = np.array(starting_value_tmp).astype(dtype)

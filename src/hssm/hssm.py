"""HSSM: Hierarchical Sequential Sampling Models.

A package based on pymc and bambi to perform Bayesian inference for hierarchical
sequential sampling models.

This file defines the entry class HSSM.
"""

import logging
from copy import deepcopy
from inspect import isclass
from typing import Literal

import pymc as pm

from hssm.base import HSSMBase
from hssm.defaults import (
    MissingDataNetwork,
    missing_data_networks_suffix,
)
from hssm.distribution_utils import (
    assemble_callables,
    make_distribution,
    make_likelihood_callable,
    make_missing_data_callable,
)
from hssm.utils import (
    _rearrange_data,
)

_logger = logging.getLogger("hssm")

# NOTE: Temporary mapping from old sampler names to new ones in bambi 0.16.0
_new_sampler_mapping: dict[str, Literal["pymc", "numpyro", "blackjax"]] = {
    "mcmc": "pymc",
    "nuts_numpyro": "numpyro",
    "nuts_blackjax": "blackjax",
}


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


class HSSM(HSSMBase):
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

        # Type narrowing: loglik and list_params should be set by this point
        assert self.loglik is not None, "loglik should be set by model_config"
        assert self.list_params is not None, "list_params validated in __init__"

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

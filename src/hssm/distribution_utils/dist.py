"""Helper functions for creating a `pymc.Distribution`.

This module provides functions for producing for Wiener First-Passage Time (WFPT)
distributions that support arbitrary log-likelihood functions and random number
generation ops.
"""

import logging
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Literal, Type

import bambi as bmb
import numpy as np
import onnx
import pymc as pm
import pytensor
import pytensor.tensor as pt
from bambi.backend.utils import get_distribution_from_prior
from numpy.typing import ArrayLike
from pytensor.tensor.random.op import RandomVariable
from ssms.basic_simulators.simulator import simulator
from ssms.config import model_config as ssms_model_config

from ..utils import decorate_atomic_simulator, download_hf, ssms_sim_wrapper
from .blackbox import make_blackbox_op
from .onnx import (
    make_jax_logp_funcs_from_jax_callable,
    make_jax_logp_funcs_from_onnx,
    make_jax_logp_ops,
    make_pytensor_logp,
)

LogLikeFunc = Callable[..., ArrayLike]
LogLikeGrad = Callable[..., ArrayLike]

_logger = logging.getLogger("hssm")

LOGP_LB = pm.floatX(-66.1)


def apply_param_bounds_to_loglik(
    logp: Any,
    list_params: list[str],
    *dist_params: Any,
    bounds: dict[str, tuple[float, float]],
):
    """Adjust the log probability of a model based on parameter boundaries.

    Parameters
    ----------
    logp
        The log probability of the model.
    list_params
        A list of strings representing the names of the distribution parameters.
    dist_params
        The distribution parameters.
    bounds
        Boundaries for parameters in the likelihood.

    Returns
    -------
        The adjusted log likelihoods.
    """
    dist_params_dict = dict(zip(list_params, dist_params))

    bounds = {k: (pm.floatX(v[0]), pm.floatX(v[1])) for k, v in bounds.items()}
    out_of_bounds_mask = pt.zeros_like(logp, dtype=bool)

    for param_name, param in dist_params_dict.items():
        # It cannot be assumed that each parameter will have bounds.
        # Skip the paramters that do not have bounds.
        if param_name not in bounds:
            continue

        lower_bound, upper_bound = bounds[param_name]

        param_mask = pt.bitwise_or(pt.lt(param, lower_bound), pt.gt(param, upper_bound))
        out_of_bounds_mask = pt.bitwise_or(out_of_bounds_mask, param_mask)

    logp = pt.where(out_of_bounds_mask, LOGP_LB, logp)

    return logp


# AF-TODO: define clip params


def ensure_positive_ndt(data, logp, list_params, dist_params):
    """Ensure that the non-decision time is always positive.

    Replaces the log probability of the model with a lower bound if the non-decision
    time is not positive.

    Parameters
    ----------
    data
        A two-column numpy array with response time and response.
    logp
        The log-likelihoods.
    list_params
        A list of parameters that the log-likelihood accepts. The order of the
        parameters in the list will determine the order in which the parameters
        are passed to the log-likelihood function.
    dist_params
        A list of parameters used in the likelihood computation. The parameters
        can be both scalars and arrays.

    Returns
    -------
    float
        The log-likelihood of the model.
    """
    rt = data[:, 0]

    if "t" not in list_params:
        return logp

    t = dist_params[list_params.index("t")]

    return pt.where(
        # consistent with the epsilon in the analytical likelihood
        rt - t <= 1e-15,
        LOGP_LB,
        logp,
    )


def make_hssm_rv(
    simulator_fun: Callable | str,
    list_params: list[str],
    lapse: bmb.Prior | None = None,
) -> Type[RandomVariable]:
    """Build a RandomVariable Op according to the list of parameters.

    Parameters
    ----------
    simulator_fun
        A simulator function with the `model_name` and `choices` attributes.
    list_params
        A list of str of all parameters for this `RandomVariable`.
    lapse : optional
        A bmb.Prior object representing the lapse distribution.

    Returns
    -------
    Type[RandomVariable]
        A class of RandomVariable that are to be used in a `pm.Distribution`.
    """
    if isinstance(simulator_fun, str):
        # If simulator_fun is passed as a string,
        # we assume it is a valid model in the
        # ssm-simulators package.
        simulator_fun_str = simulator_fun
        if simulator_fun_str not in ssms_model_config:
            _logger.warning(
                "You supplied a model '%s', which is currently not supported in "
                + "the ssm_simulators package. An error will be thrown when sampling "
                + "from the random variable or when using any "
                + "posterior or prior predictive sampling methods.",
                simulator_fun_str,
            )
            # We still build a bogus simulator function here
            # that will raise an error when finally called.
            simulator_fun_internal = decorate_atomic_simulator(
                model_name=simulator_fun_str, choices=[0, 1, 2], obs_dim=2
            )(
                partial(
                    ssms_sim_wrapper,
                    simulator_fun=simulator,
                    model=simulator_fun_str,  # will raise error due to unkown string
                )
            )
        else:
            simulator_fun_internal = decorate_atomic_simulator(
                model_name=simulator_fun_str,
                choices=ssms_model_config[simulator_fun_str]["choices"],
                obs_dim=2,  # At least for now ssms models all fall under 2 obs dims
            )(
                partial(
                    ssms_sim_wrapper,
                    simulator_fun=simulator,  # Passing simulator from ssm-simulators
                    model=simulator_fun_str,
                )
            )
    elif callable(simulator_fun):
        simulator_fun_internal = simulator_fun
    else:
        raise ValueError(
            "The simulator argument must be a string or a callable, "
            f"but you passed {simulator_fun}."
        )

    if lapse is not None and list_params[-1] != "p_outlier":
        list_params.append("p_outlier")

    if hasattr(simulator_fun_internal, "model_name"):
        model_name = simulator_fun_internal.model_name
    else:
        raise ValueError("The simulator function must have a `model_name` attribute.")
    if hasattr(simulator_fun_internal, "choices"):
        choices = simulator_fun_internal.choices
    else:
        raise ValueError("The simulator function must have a `choices` attribute.")
    if hasattr(simulator_fun_internal, "obs_dim"):
        if isinstance(simulator_fun_internal.obs_dim, int):
            obs_dim_int = simulator_fun_internal.obs_dim
        else:
            raise ValueError("The obs_dim attribute must be an integer")
    else:
        raise ValueError("The simulator function must have a `obs_dim` attribute.")

    # pylint: disable=W0511, R0903
    class HSSMRV(RandomVariable):
        """HSSMRV random variable."""

        name: str = model_name + "_RV"
        # New in PyMC 5.16+: instead of using `ndims_supp`, we use `signature` to define
        # the signature of the random variable. The string to the left of the `->` sign
        # describes the input signature, which is `()` for each parameter, meaning each
        # parameter is a scalar. The string to the right of the
        # `->` sign describes the output signature, which is `(2)`, which means the
        # random variable is a length-2 array.
        signature: str = f"{','.join(['()']*len(list_params))}->({obs_dim_int})"
        dtype: str = "floatX"
        _print_name: tuple[str, str] = ("SSM", "\\operatorname{SSM}")
        _list_params = list_params
        _lapse = lapse

        # pylint: disable=arguments-renamed,bad-option-value,W0221
        # NOTE: `rng` now is a np.random.Generator instead of RandomState
        # since the latter is now deprecated from numpy
        @classmethod
        def rng_fn(
            cls,
            rng: np.random.Generator,
            *args,
            **kwargs,
        ) -> np.ndarray:
            """Generate random variables from this distribution.

            Parameters
            ----------
            rng
                A `np.random.Generator` object for random state.
            args
                Unnamed arguments of parameters, in the order of `_list_params`, plus
                the last one as size.
            kwargs
                Other keyword arguments passed to the ssms simulator.

            Returns
            -------
            np.ndarray
                An array of `(rt, response)` generated from the distribution.

            Note
            ----
            How size is handled in this method:

            We apply multiple tricks to get this method to work with ssm_simulators.

            First, size could be an array with one element. We squeeze the array and
            use that element as size.

            Then, size could depend on whether the parameters passed to this method.
            If all parameters passed are scalar, that is the easy case. We just
            assemble all parameters into a 1D array and pass it to the `theta`
            argument. In this case, size is number of observations.

            If one of the parameters is a vector, which happens one or more parameters
            is the target of a regression. In this case, we take the size of the
            parameter with the largest size. If size is None, we will set size to be
            this largest size. If size is not None, we check if size is a multiple of
            the largest size. If not, an error is thrown. Otherwise, we assemble the
            parameter as a matrix and pass it as the `theta` argument. The multiple then
            becomes how many samples we draw from each trial.
            """
            # First figure out what the size specified here is
            # Since the number of unnamed arguments is undetermined,
            # we are going to use this hack.

            if "size" in kwargs:
                size = kwargs.pop("size")
            else:
                size = args[-1]
                args = args[:-1]

            if size is None:
                size = 1

            # Although we got around the ndims_supp issue, the size parameter passed
            # here is still an array with one element. We need to take it out.
            if not np.isscalar(size):
                size = np.squeeze(size)

            num_params = len(cls._list_params)

            # TODO: We need to figure out what to do with extra_fields when
            # doing posterior predictive sampling. Right now nothing is done.
            if num_params < len(args):
                arg_arrays = [np.asarray(arg) for arg in args[:num_params]]
            else:
                arg_arrays = [np.asarray(arg) for arg in args]

            p_outlier = None

            if cls._list_params[-1] == "p_outlier":
                p_outlier = arg_arrays.pop(-1)

            iinfo32 = np.iinfo(np.uint32)
            seed = rng.integers(0, iinfo32.max, dtype=np.uint32)

            is_all_args_scalar = all(arg.size == 1 for arg in arg_arrays)

            if is_all_args_scalar:
                # All parameters are scalars

                theta = np.stack(arg_arrays)
                if theta.ndim > 1:
                    theta = theta.squeeze(axis=-1)

                theta = np.tile(theta, (size, 1))
                n_replicas = 1
            else:
                # Preprocess all parameters, reshape them into a matrix of dimension
                # (size, n_params) where size is the number of elements in the largest
                # of all parameters passed to *arg
                elem_max_size = np.argmax([arg.size for arg in arg_arrays])
                max_shape = arg_arrays[elem_max_size].shape

                new_data_size = max_shape[-1]

                theta = np.column_stack(
                    [np.broadcast_to(arg, max_shape).reshape(-1) for arg in arg_arrays]
                )

                # We eventually want to get rid of this part, and
                # simply make the simulators behave as would be expected
                # by pymc directly.
                if size is None or size == 1:
                    n_replicas = 1
                elif size % new_data_size != 0:
                    raise ValueError(
                        "`size` needs to be a multiple of the size of data"
                    )
                else:
                    n_replicas = size // new_data_size

            sims_out = simulator_fun_internal(
                theta=theta,
                random_state=seed,
                n_replicas=n_replicas,
                **kwargs,
            )

            # return sims_out, max_shape, size

            if not is_all_args_scalar:
                if n_replicas == 1:
                    sims_out = sims_out.reshape(
                        (*max_shape[:-1], max_shape[-1], obs_dim_int)
                    )
                else:
                    # sims_out = sims_out.reshape(
                    #     (*max_shape[:-1], max_shape[-1] * size, obs_dim_int)
                    # )
                    sims_out = sims_out.reshape(
                        (*max_shape[:-1], max_shape[-1], n_replicas, obs_dim_int)
                    )

            if p_outlier is not None:
                assert cls._lapse is not None, (
                    "You have specified `p_outlier`, the probability of the lapse "
                    + "distribution but did not specify the distribution."
                )
                out_shape = sims_out.shape[:-1]
                if not np.isscalar(p_outlier) and len(p_outlier.shape) > 0:
                    if p_outlier.shape[-1] == 1:
                        p_outlier = np.broadcast_to(p_outlier, out_shape)
                    else:
                        p_outlier = p_outlier.reshape(out_shape)

                replace = rng.binomial(n=1, p=p_outlier, size=out_shape).astype(bool)
                replace_n = int(np.sum(replace, axis=None))
                if replace_n == 0:
                    return sims_out

                replace_shape = (*out_shape[:-1], replace_n)
                replace_mask = np.stack([replace, replace], axis=-1)
                n_draws = np.prod(replace_shape)
                lapse_rt = pm.draw(
                    get_distribution_from_prior(cls._lapse).dist(**cls._lapse.args),
                    n_draws,
                    random_seed=rng,
                ).reshape(replace_shape)

                lapse_response = rng.choice(
                    choices,
                    p=1 / len(choices) * np.ones(len(choices)),
                    size=replace_shape,
                )
                lapse_output = np.stack(
                    [lapse_rt, lapse_response],
                    axis=-1,
                )
                np.putmask(sims_out, replace_mask, lapse_output)
            return sims_out

    return HSSMRV


def make_distribution(
    rv: str | Type[RandomVariable] | RandomVariable | Callable,
    loglik: LogLikeFunc | pytensor.graph.Op,
    list_params: list[str],
    bounds: dict | None = None,
    lapse: bmb.Prior | None = None,
    extra_fields: list[np.ndarray] | None = None,
) -> Type[pm.Distribution]:
    """Make a `pymc.Distribution`.

    Constructs a `pymc.Distribution` from a log-likelihood function and a
    RandomVariable op.

    Parameters
    ----------
    model_name
        The name of the model.
    choices
        A list of integers indicating the choices.
    rv
        A RandomVariable Op (a class, not an instance) or a string indicating the model.
        If a string, a RandomVariable class will be created automatically with its
        `rng_fn` class method generated using the simulator identified with this string
        from the `ssm_simulators` package. If the string is not one of the supported
        models in the `ssm_simulators` package, a warning will be raised, and any
        attempt to sample from the RandomVariable will result in an error.
    loglik
        A loglikelihood function. It can be any Callable in Python.
    list_params
        A list of parameters that the log-likelihood accepts. The order of the
        parameters in the list will determine the order in which the parameters
        are passed to the log-likelihood function.
    bounds : optional
        A dictionary with parameters as keys (a string) and its boundaries as values.
        Example: {"parameter": (lower_boundary, upper_boundary)}.
    lapse : optional
        A bmb.Prior object representing the lapse distribution.
    extra_fields : optional
        An optional list of arrays that are stored in the class created and will be
        used in likelihood calculation. Defaults to None.

    Returns
    -------
    Type[pm.Distribution]
        A pymc.Distribution that uses the log-likelihood function.
    """
    if isinstance(rv, type) and issubclass(rv, RandomVariable):
        rv_instance = rv()
    elif not isinstance(rv, type) and isinstance(rv, RandomVariable):
        rv_instance = rv
    elif callable(rv) or isinstance(rv, str):
        random_variable = make_hssm_rv(
            simulator_fun=rv,
            list_params=list_params,
            lapse=lapse,
        )
        rv_instance = random_variable()
    else:
        raise ValueError(f"rv is {rv}, which is not a valid type.")

    # random_variable = make_ssm_rv(rv, list_params, lapse)
    # if isinstance(rv, str) else rv
    extra_fields = [] if extra_fields is None else extra_fields

    if lapse is not None:
        if list_params[-1] != "p_outlier":
            list_params.append("p_outlier")

        data_vector = pt.dvector()
        lapse_logp = pm.logp(
            get_distribution_from_prior(lapse).dist(**lapse.args),
            data_vector,
        )
        lapse_func = pytensor.function(
            [data_vector],
            lapse_logp,
        )
    else:
        lapse_func = None

    class HSSMDistribution(pm.Distribution):
        """Wiener first-passage time (WFPT) log-likelihood for LANs."""

        # This is just a placeholder because pm.Distribution requires an rv_op
        # Might be updated in the future once

        # NOTE: rv_op is an INSTANCE of RandomVariable
        rv_op = rv_instance
        _params = list_params

        @classmethod
        def dist(cls, **kwargs):  # pylint: disable=arguments-renamed
            dist_params = [
                pt.as_tensor_variable(pm.floatX(kwargs[param])) for param in cls._params
            ]
            other_kwargs = {k: v for k, v in kwargs.items() if k not in cls._params}
            return super().dist(dist_params, **other_kwargs)

        def logp(data, *dist_params):  # pylint: disable=E0213
            """Calculate log probability of the data given the parameters.

            Parameters
            ----------
            data : array-like
                The observed data
            dist_params : tuple
                Distribution parameters

            Returns
            -------
            tensor
                Log probability
            """
            # AF-TODO: Apply clipping here
            if list_params[-1] == "p_outlier":
                p_outlier = dist_params[-1]
                dist_params = dist_params[:-1]

                if not callable(lapse_func):
                    raise ValueError(
                        "lapse_func is not defined. "
                        "Make sure lapse is properly initialized."
                    )
                lapse_logp = lapse_func(data[:, 0].eval())
                # AF-TODO potentially apply clipping here
                logp = loglik(data, *dist_params, *extra_fields)
                # Ensure that non-decision time is always smaller than rt.
                # Assuming that the non-decision time parameter is always named "t".
                logp = ensure_positive_ndt(data, logp, list_params, dist_params)
                logp = pt.log(
                    (1.0 - p_outlier) * pt.exp(logp)
                    + p_outlier * pt.exp(lapse_logp)
                    + 1e-29
                )
            else:
                logp = loglik(data, *dist_params, *extra_fields)
                # Ensure that non-decision time is always smaller than rt.
                # Assuming that the non-decision time parameter is always named "t".
                logp = ensure_positive_ndt(data, logp, list_params, dist_params)

            if bounds is not None:
                logp = apply_param_bounds_to_loglik(
                    logp, list_params, *dist_params, bounds=bounds
                )

            return logp

    return HSSMDistribution


def make_family(
    dist: Type[pm.Distribution],
    list_params: list[str],
    link: str | dict[str, bmb.families.Link],
    parent: str = "v",
    likelihood_name: str = "SSM Likelihood",
    family_name="SSM Family",
) -> bmb.Family:
    """Build a family in bambi.

    Parameters
    ----------
    dist
        A pm.Distribution class (not an instance).
    list_params
        A list of parameters for the likelihood function.
    link
        A link function or a dictionary of parameter: link functions.
    parent
        The parent parameter of the likelihood function. Defaults to v.
    likelihood_name
        the name of the likelihood function. Defaults to "SSM Likelihood".
    family_name
        the name of the family. Defaults to "SSM Family".

    Returns
    -------
    bmb.Family
        An instance of a bambi family.
    """
    likelihood = bmb.Likelihood(
        likelihood_name, parent=parent, params=list_params, dist=dist
    )

    family = SSMFamily(family_name, likelihood=likelihood, link=link)

    return family


class SSMFamily(bmb.Family):
    """Extends bmb.Family to get around the dimensionality mismatch."""

    def create_extra_pps_coord(self):
        """Create an extra dimension."""
        return np.arange(2)


def make_likelihood_callable(
    loglik: pytensor.graph.Op | Callable | PathLike | str,
    loglik_kind: Literal["analytical", "approx_differentiable", "blackbox"],
    backend: Literal["pytensor", "jax", "other"] | None,
    params_is_reg: list[bool] | None = None,
    params_only: bool | None = None,
) -> pytensor.graph.Op | Callable:
    """Make a callable for the likelihood function.

    This function is intended to be general to support different kinds of likelihood
    functions.

    Parameters
    ----------
    loglik
        The log-likelihood function. It can be a string, a path to an ONNX model, a
        pytensor Op, or a python function.
    loglik_kind
        The kind of the log-likelihood for the model. This parameter controls
        how the likelihood function is wrapped.
    backend : Optional
        The backend to use for the log-likelihood function.
    params_is_reg : Optional
        A list of boolean values indicating whether the parameters are regression
        parameters. Defaults to None.
    params_only : Optional
        Whether the missing data likelihood is takes its first argument as the data.
        Defaults to None.
    """
    if isinstance(loglik, pytensor.graph.Op):
        return loglik

    if callable(loglik):
        # In the analytical case, if `backend` is None or `pytensor`, we can use the
        # callable directly. Otherwise, we wrap it in a BlackBoxOp.
        if loglik_kind == "analytical":
            if backend is None or backend == "pytensor":
                return loglik
            return make_blackbox_op(loglik)
        elif loglik_kind == "blackbox":
            return make_blackbox_op(loglik)
        elif loglik_kind == "approx_differentiable":
            if backend is None or backend == "jax":
                if params_is_reg is None:
                    raise ValueError(
                        "You set `loglik_kind` to `approx_differentiable` "
                        + "and `backend` to `jax` and supplied a jax callable, "
                        + "but did not set `params_is_reg`."
                    )
                logp, logp_grad, logp_nojit = make_jax_logp_funcs_from_jax_callable(
                    loglik,
                    params_is_reg,
                    params_only=False if params_only is None else params_only,
                )
                lan_logp_jax = make_jax_logp_ops(logp, logp_grad, logp_nojit)
                return lan_logp_jax
            if backend == "pytensor":
                raise ValueError(
                    "You set `backend` to `pytensor`, `loglik_kind` to"
                    + "`approx_differentiable` and provided a callable."
                    + "Currently we support only jax callables in this case."
                )
            # In the approx_differentiable case or the blackbox case, unless the backend
            # is `pytensor`, we wrap the callable in a BlackBoxOp.
        # if backend == "pytensor":
        #     return loglik
        #     # In all other cases, we assume that the callable cannot be directly
        #     # used in the backend and thus we wrap it in a BlackBoxOp
        # return make_blackbox_op(loglik)

    # Other cases, when `loglik` is a string or a PathLike.
    if loglik_kind != "approx_differentiable":
        raise ValueError(
            "You set `loglik_kind` to `approx_differentiable "
            + "but did not provide a pm.Distribution, an Op, or a callable "
            + "as `loglik`."
        )

    if isinstance(loglik, (str, PathLike)):
        if not Path(loglik).exists():
            loglik = download_hf(str(loglik))

    onnx_model = onnx.load(str(loglik))

    if backend == "pytensor":
        lan_logp_pt = make_pytensor_logp(onnx_model)
        return lan_logp_pt

    if params_is_reg is None:
        raise ValueError(
            "You set `loglik_kind` to `approx_differentiable` "
            + "and `backend` to `jax` but did not provide `params_is_reg`."
        )

    logp, logp_grad, logp_nojit = make_jax_logp_funcs_from_onnx(
        onnx_model,
        params_is_reg,
        params_only=False if params_only is None else params_only,
    )
    lan_logp_jax = make_jax_logp_ops(logp, logp_grad, logp_nojit)

    return lan_logp_jax


def make_missing_data_callable(
    loglik: pytensor.graph.Op | Callable | PathLike | str,
    backend: Literal["pytensor", "jax", "other"] | None = "jax",
    params_is_reg: list[bool] | None = None,
    params_only: bool | None = None,
) -> pytensor.graph.Op | Callable:
    """Make a secondary network for the likelihood function.

    Please refer to the documentation of `make_likelihood_callable` for more.
    """
    if backend == "jax":
        if params_is_reg is None:
            raise ValueError(
                "You have chosen `jax` as the backend for the missing data likelihood. "
                + "However, you have not provided any values to `params_is_reg`."
            )
        if params_only is None:
            raise ValueError(
                "You have chosen `jax` as the backend for the missing data likelihood. "
                + "However, you have not provided any values to `params_only`."
            )

    # We assume that the missing data network is always approx_differentiable
    return make_likelihood_callable(
        loglik, "approx_differentiable", backend, params_is_reg, params_only
    )


def assemble_callables(
    callable: pytensor.graph.Op | Callable,
    missing_data_callable: pytensor.graph.Op | Callable,
    params_only: bool,
    has_deadline: bool,
) -> Callable:
    """Assemble the likelihood callables into a single callable.

    Parameters
    ----------
    callable
        The callable for the likelihood function.
    missing_data_callable
        The callable for the secondary network for the likelihood function.
    params_only
        Whether the missing data likelihood is takes its first argument as the data.
    has_deadline
        Whether the model has a deadline.
    """

    def likelihood_callable(data, *dist_params):
        """Compute the log-likelihoood of the model."""
        # Assuming the first column of the data is always rt
        data = pt.as_tensor_variable(data)

        # New in PyMC 5.16+: PyMC uses the signature of the RandomVariable to determine
        # the dimensions of the inputs to the likelihood function. It automatically adds
        # one additional dimension to our input variable if it is a scalar. We need to
        # squeeze this dimension out.
        dist_params = [pt.squeeze(param) for param in dist_params]

        n_missing = pt.sum(pt.eq(data[:, 0], -999.0)).astype(int)
        if n_missing == 0:
            raise ValueError("No missing data in the data.")

        observed_data = data[n_missing:, :]

        dist_params_observed = [
            param[n_missing:] if param.ndim >= 1 else param for param in dist_params
        ]

        if has_deadline:
            logp_observed = callable(observed_data[:, :-1], *dist_params_observed)
        else:
            logp_observed = callable(observed_data, *dist_params_observed)

        dist_params_missing = [
            param[:n_missing] if param.ndim >= 1 else param for param in dist_params
        ]

        if params_only:
            logp_missing = missing_data_callable(None, *dist_params_missing)
        else:
            missing_data = data[:n_missing, -1:]
            logp_missing = missing_data_callable(missing_data, *dist_params_missing)

        logp = pt.empty_like(data[:, 0], dtype=pytensor.config.floatX)
        logp = pt.set_subtensor(logp[n_missing:], logp_observed)
        logp = pt.set_subtensor(logp[:n_missing], logp_missing)

        return logp

    return likelihood_callable

"""Helper functions for creating a `pymc.Distribution`.

This module provides functions for producing for Wiener First-Passage Time (WFPT)
distributions that support arbitrary log-likelihood functions and random number
generation ops.
"""

import logging
from os import PathLike
from typing import Any, Callable, Type

import bambi as bmb
import numpy as np
import onnx
import pymc as pm
import pytensor
import pytensor.tensor as pt
from bambi.backend.utils import get_distribution_from_prior
from numpy.typing import ArrayLike
from pytensor.graph.op import Apply, Op
from pytensor.tensor.random.op import RandomVariable
from ssms.basic_simulators.simulator import simulator
from ssms.config import model_config as ssms_model_config

from .onnx import make_jax_logp_funcs_from_onnx, make_jax_logp_ops, make_pytensor_logp

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


def make_ssm_rv(
    model_name: str, list_params: list[str], lapse: bmb.Prior | None = None
) -> Type[RandomVariable]:
    """Build a RandomVariable Op according to the list of parameters.

    Parameters
    ----------
    model_name
        The name of the model. If the `model_name` is not one of the supported
        models in the `ssm_simulators` package, a warning will be raised, and any
        attempt to sample from the RandomVariable will result in an error.
    list_params
        A list of str of all parameters for this `RandomVariable`.
    lapse : optional
        A bmb.Prior object representing the lapse distribution.

    Returns
    -------
    Type[RandomVariable]
        A class of RandomVariable that are to be used in a `pm.Distribution`.
    """
    if model_name not in ssms_model_config:
        _logger.warning(
            "You supplied a model '%s', which is currently not supported in "
            + "the ssm_simulators package. An error will be thrown when sampling from "
            + "the random variable or when using any "
            + "posterior or prior predictive sampling methods.",
            model_name,
        )

    if lapse is not None and list_params[-1] != "p_outlier":
        list_params.append("p_outlier")

    # pylint: disable=W0511, R0903
    class SSMRandomVariable(RandomVariable):
        """SSM random variable."""

        name: str = "SSM_RV"
        # to get around the support checking in PyMC that would result in error
        ndim_supp: int = 1

        ndims_params: list[int] = [0 for _ in list_params]
        dtype: str = "floatX"
        _print_name: tuple[str, str] = ("SSM", "\\operatorname{SSM}")
        _list_params = list_params
        _lapse = lapse

        # PyTensor, as of version 2.12, enforces a check to ensure that
        # at least one parameter has the same ndims as the support.
        # This overrides that check and ensures that the dimension checks are correct.
        # For more information, see this issue
        # https://github.com/lnccbrown/HSSM/issues/36
        def _supp_shape_from_params(*args, **kwargs):
            return (2,)

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
                p_outlier = np.squeeze(arg_arrays.pop(-1))

            iinfo32 = np.iinfo(np.uint32)
            seed = rng.integers(0, iinfo32.max, dtype=np.uint32)

            params = (
                cls._list_params[:-1]
                if cls._list_params[-1] == "p_outlier"
                else cls._list_params
            )

            if params != ssms_model_config[model_name]["params"]:
                raise ValueError(
                    f"The list of parameters in `list_params` {params} "
                    + "is different from the model config in SSM Simulators "
                    + f"({ssms_model_config[model_name]['params']})."
                )

            is_all_scalar = all(arg.size == 1 for arg in arg_arrays)

            if is_all_scalar:
                # All parameters are scalars

                theta = np.stack(arg_arrays)
                n_samples = size
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

                if size is None or size == 1:
                    n_samples = 1
                elif size % new_data_size != 0:
                    raise ValueError(
                        "`size` needs to be a multiple of the size of data"
                    )
                else:
                    n_samples = size // new_data_size

            sim_out = simulator(
                theta=theta,
                model=model_name,
                n_samples=n_samples,
                random_state=seed,
                **kwargs,
            )

            sims_out = np.column_stack([sim_out["rts"], sim_out["choices"]])

            if not is_all_scalar:
                sims_out = sims_out.reshape(
                    (*max_shape[:-1], max_shape[-1] * n_samples, 2)
                )

            if p_outlier is not None:
                assert cls._lapse is not None, (
                    "You have specified `p_outlier`, the probability of the lapse "
                    + "distribution but did not specify the distribution."
                )
                out_shape = sims_out.shape[:-1]
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
                lapse_response = rng.binomial(n=1, p=0.5, size=replace_shape)
                lapse_response = np.where(lapse_response == 1, 1, -1)
                lapse_output = np.stack(
                    [lapse_rt, lapse_response],
                    axis=-1,
                )
                np.putmask(sims_out, replace_mask, lapse_output)

            return sims_out

    return SSMRandomVariable


def make_distribution(
    rv: str | Type[RandomVariable],
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
    random_variable = make_ssm_rv(rv, list_params, lapse) if isinstance(rv, str) else rv

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

    class SSMDistribution(pm.Distribution):
        """Wiener first-passage time (WFPT) log-likelihood for LANs."""

        # This is just a placeholder because pm.Distribution requires an rv_op
        # Might be updated in the future once

        # NOTE: rv_op is an INSTANCE of RandomVariable
        rv_op = random_variable()
        params = list_params
        _extra_fields = extra_fields

        @classmethod
        def dist(cls, **kwargs):  # pylint: disable=arguments-renamed
            dist_params = [
                pt.as_tensor_variable(pm.floatX(kwargs[param])) for param in cls.params
            ]
            if cls._extra_fields:
                dist_params += [pm.floatX(field) for field in cls._extra_fields]
            other_kwargs = {k: v for k, v in kwargs.items() if k not in cls.params}
            return super().dist(dist_params, **other_kwargs)

        def logp(data, *dist_params):  # pylint: disable=E0213
            num_params = len(list_params)
            extra_fields = []

            if num_params < len(dist_params):
                extra_fields = dist_params[num_params:]
                dist_params = dist_params[:num_params]

            if list_params[-1] == "p_outlier":
                p_outlier = dist_params[-1]
                dist_params = dist_params[:-1]
                lapse_logp = lapse_func(data[:, 0].eval())

                logp = loglik(data, *dist_params, *extra_fields)
                logp = pt.log(
                    (1.0 - p_outlier) * pt.exp(logp)
                    + p_outlier * pt.exp(lapse_logp)
                    + 1e-29
                )
            else:
                logp = loglik(data, *dist_params, *extra_fields)

            if bounds is not None:
                logp = apply_param_bounds_to_loglik(
                    logp, list_params, *dist_params, bounds=bounds
                )

            # Ensure that non-decision time is always smaller than rt.
            # Assuming that the non-decision time parameter is always named "t".
            return ensure_positive_ndt(data, logp, list_params, dist_params)

    return SSMDistribution


def make_distribution_from_onnx(
    rv: str | Type[RandomVariable],
    list_params: list[str],
    onnx_model: str | PathLike | onnx.ModelProto,
    backend: str = "jax",
    bounds: dict | None = None,
    params_is_reg: list[bool] | None = None,
    lapse: bmb.Prior | None = None,
    extra_fields: list[np.ndarray] | None = None,
) -> Type[pm.Distribution]:
    """Make a PyMC distribution from an ONNX model.

    Produces a PyMC distribution that uses the provided base or ONNX model as
    its log-likelihood function.

    Parameters
    ----------
    rv
        A RandomVariable Op (a class, not an instance) or a string indicating the model.
        If a string, a RandomVariable class will be created automatically with its
        `rng_fn` class method generated using the simulator identified with this string
        from the `ssm_simulators` package. If the string is not one of the supported
        models in the `ssm_simulators` package, a warning will be raised, and any
        attempt to sample from the RandomVariable will result in an error.
    list_params
        A list of the names of the parameters following the order of how they are fed
        to the network.
    onnx_model
        The path of the ONNX model, or one already loaded in memory.
    backend
        Whether to use "pytensor" or "jax" as the backend of the log-likelihood
        computation. If `jax`, the function will be wrapped in an pytensor Op.
    bounds : optional
        A dictionary with parameters as keys (a string) and its boundaries
        as values.Example: {"parameter": (lower_boundary, upper_boundary)}.
    params_is_reg : optional
        A list of booleans indicating whether each parameter in the
        corresponding position in `list_params` is a regression.
    lapse : optional
        A bmb.Prior object representing the lapse distribution.
    extra_fields : optional
        An optional list of arrays that are stored in the class created and will be
        used in likelihood calculation. Defaults to None.

    Returns
    -------
    Type[pm.Distribution]
        A PyMC Distribution class that uses the ONNX model as its log-likelihood
        function.
    """
    if isinstance(onnx_model, (str, PathLike)):
        onnx_model = onnx.load(str(onnx_model))
    if backend == "pytensor":
        lan_logp_pt = make_pytensor_logp(onnx_model)
        return make_distribution(
            rv,
            lan_logp_pt,
            list_params,
            bounds=bounds,
            lapse=lapse,
            extra_fields=extra_fields,
        )
    if backend == "jax":
        if params_is_reg is None:
            params_is_reg = [False for param in list_params if param != "p_outlier"]

        # Extra fields are passed to the likelihood functions as vectors
        # They do not need to be broadcast, so param_is_reg is padded with True
        if extra_fields:
            params_is_reg += [True for _ in extra_fields]

        logp, logp_grad, logp_nojit = make_jax_logp_funcs_from_onnx(
            onnx_model,
            params_is_reg,
        )
        lan_logp_jax = make_jax_logp_ops(logp, logp_grad, logp_nojit)

        return make_distribution(
            rv,
            lan_logp_jax,
            list_params,
            bounds=bounds,
            lapse=lapse,
            extra_fields=extra_fields,
        )

    raise ValueError("Currently only 'pytensor' and 'jax' backends are supported.")


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


def make_blackbox_op(logp: Callable) -> Op:
    """Wrap an arbitrary function in a pytensor Op.

    Parameters
    ----------
    logp
        A python function that represents the log-likelihood function. The function
        needs to have signature of logp(data, *dist_params) where `data` is a
        two-column numpy array and `dist_params`represents all parameters passed to the
        function.

    Returns
    -------
    Op
        An pytensor op that wraps the log-likelihood function.
    """

    class BlackBoxOp(Op):  # pylint: disable=W0223
        """Wraps an arbitrary function in a pytensor Op."""

        def make_node(self, data, *dist_params):
            """Take the inputs to the Op and puts them in a list.

            Also specifies the output types in a list, then feed them to the Apply node.

            Parameters
            ----------
            data
                A two-column numpy array with response time and response.
            dist_params
                A list of parameters used in the likelihood computation. The parameters
                can be both scalars and arrays.
            """
            inputs = [
                pt.as_tensor_variable(data),
            ] + [pt.as_tensor_variable(dist_param) for dist_param in dist_params]

            outputs = [pt.vector()]

            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, output_storage):
            """Perform the Apply node.

            Parameters
            ----------
            inputs
                This is a list of data from which the values stored in
                output_storage are to be computed using non-symbolic language.
            output_storage
                This is a list of storage cells where the output
                is to be stored. A storage cell is a one-element list. It is
                forbidden to change the length of the list(s) contained in
                output_storage. There is one storage cell for each output of
                the Op.
            """
            result = logp(*inputs)
            output_storage[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)

    blackbox_op = BlackBoxOp()
    return blackbox_op


def make_distribution_from_blackbox(
    rv: str | Type[RandomVariable],
    loglik: Callable,
    list_params: list[str],
    bounds: dict | None = None,
    extra_fields: list[np.ndarray] | None = None,
) -> Type[pm.Distribution]:
    """Make a `pymc.Distribution`.

    Constructs a `pymc.Distribution` from a blackbox likelihood function

    Parameters
    ----------
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
    extra_fields : optional
        An optional list of arrays that are stored in the class created and will be
        used in likelihood calculation. Defaults to None.

    Returns
    -------
    Type[pm.Distribution]
        A pymc.Distribution that uses the log-likelihood function.
    """
    blackbox_op = make_blackbox_op(loglik)

    return make_distribution(
        rv, blackbox_op, list_params, bounds, extra_fields=extra_fields
    )

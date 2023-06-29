"""Helper functions for creating a `pymc.Distribution`.

This module provides functions for producing for Wiener First-Passage Time (WFPT)
distributions that support arbitrary log-likelihood functions and random number
generation ops.
"""

from __future__ import annotations

import logging
from os import PathLike
from typing import Any, Callable, Type

import bambi as bmb
import numpy as np
import onnx
import pymc as pm
import pytensor
import pytensor.tensor as pt
from numpy.typing import ArrayLike
from pytensor.graph.op import Apply, Op
from pytensor.tensor.random.op import RandomVariable
from ssms.basic_simulators import simulator
from ssms.config import model_config as ssms_model_config

from .onnx import make_jax_logp_funcs_from_onnx, make_jax_logp_ops, make_pytensor_logp

LogLikeFunc = Callable[..., ArrayLike]
LogLikeGrad = Callable[..., ArrayLike]

OUT_OF_BOUNDS_VAL = pm.floatX(-66.1)


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

    bounds = {
        k: (
            pm.floatX(v[0]),
            pm.floatX(v[1]),
        )
        for k, v in bounds.items()
    }

    for param_name, param in dist_params_dict.items():
        # It cannot be assumed that each parameter will have bounds.
        # Skip the paramters that do not have bounds.
        if param_name not in bounds:
            continue

        lower_bound, upper_bound = bounds[param_name]

        out_of_bounds_mask = pt.bitwise_or(
            pt.lt(param, lower_bound), pt.gt(param, upper_bound)
        )

        broadcasted_mask = pt.broadcast_to(
            out_of_bounds_mask, logp.shape
        )  # typing: ignore

        logp = pt.where(broadcasted_mask, OUT_OF_BOUNDS_VAL, logp)

    return logp


def make_ssm_rv(model_name: str, list_params: list[str]) -> Type[RandomVariable]:
    """Build a RandomVariable Op according to the list of parameters.

    Parameters
    ----------
    model_name
        The name of the model. If the `model_name` is not one of the supported
        models in the `ssm_simulators` package, a warning will be raised, and any
        attempt to sample from the RandomVariable will result in an error.
    list_params
        A list of str of all parameters for this `RandomVariable`.

    Returns
    -------
    Type[RandomVariable]
        A class of RandomVariable that are to be used in a `pm.Distribution`.
    """
    if model_name not in ssms_model_config:
        logging.warning(
            f"You supplied a model '{model_name}', which is currently not supported in "
            + "the ssm_simulators package. An error will be thrown when sampling from "
            + "the random variable or when using any posterior sampling methods."
        )

    # pylint: disable=W0511, R0903
    class SSMRandomVariable(RandomVariable):
        """WFPT random variable."""

        name: str = "SSM_RV"

        # NOTE: This is wrong at the moment, but necessary
        # to get around the support checking in PyMC that would result in error
        ndim_supp: int = 0

        ndims_params: list[int] = [0 for _ in list_params]
        dtype: str = "floatX"
        _print_name: tuple[str, str] = ("SSM", "\\operatorname{SSM}")
        _list_params = list_params

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
            """
            # First figure out what the size specified here is
            # Since the number of unnamed arguments is undetermined,
            # we are going to use this hack.
            if "size" in kwargs:
                size = kwargs.pop("size")
            else:
                size = args[-1]
                args = args[:-1]

            arg_arrays = [np.asarray(arg) for arg in args]

            iinfo32 = np.iinfo(np.uint32)
            seed = rng.integers(0, iinfo32.max, dtype=np.uint32)

            if cls._list_params != ssms_model_config[model_name]["params"]:
                raise ValueError(
                    f"The list of parameters in `list_params` {cls._list_params} "
                    + "is different from the model config in SSM Simulators "
                    + f"({ssms_model_config[model_name]['params']})."
                )

            is_all_scalar = all(arg.size == 1 for arg in arg_arrays)

            if is_all_scalar:
                # All parameters are scalars

                theta = np.stack(arg_arrays)
                n_samples = 1 if not size else size
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

                if size is None:
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

            output = np.column_stack([sim_out["rts"], sim_out["choices"]])

            if not is_all_scalar:
                output = output.reshape((*max_shape[:-1], max_shape[-1] * n_samples, 2))

            return output

    return SSMRandomVariable


def make_distribution(
    rv: str | Type[RandomVariable],
    loglik: LogLikeFunc | pytensor.graph.Op,
    list_params: list[str],
    bounds: dict | None = None,
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

    Returns
    -------
    Type[pm.Distribution]
        A pymc.Distribution that uses the log-likelihood function.
    """
    random_variable = make_ssm_rv(rv, list_params) if isinstance(rv, str) else rv

    class SSMDistribution(pm.Distribution):
        """Wiener first-passage time (WFPT) log-likelihood for LANs."""

        # This is just a placeholder because pm.Distribution requires an rv_op
        # Might be updated in the future once

        # NOTE: rv_op is an INSTANCE of RandomVariable
        rv_op = random_variable()
        params = list_params

        @classmethod
        def dist(cls, **kwargs):  # pylint: disable=arguments-renamed
            dist_params = [
                pt.as_tensor_variable(pm.floatX(kwargs[param])) for param in cls.params
            ]
            other_kwargs = {k: v for k, v in kwargs.items() if k not in cls.params}
            return super().dist(dist_params, **other_kwargs)

        def logp(data, *dist_params):  # pylint: disable=E0213
            logp = loglik(data, *dist_params)

            if bounds is None:
                return logp

            return apply_param_bounds_to_loglik(
                logp, list_params, *dist_params, bounds=bounds
            )

    return SSMDistribution


def make_distribution_from_onnx(
    rv: str | Type[RandomVariable],
    list_params: list[str],
    onnx_model: str | PathLike | onnx.ModelProto,
    backend: str = "pytensor",
    bounds: dict | None = None,
    params_is_reg: list[bool] | None = None,
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
        )
    if backend == "jax":
        if params_is_reg is None:
            params_is_reg = [False for _ in list_params]
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

    Returns
    -------
    Type[pm.Distribution]
        A pymc.Distribution that uses the log-likelihood function.
    """
    blackbox_op = make_blackbox_op(loglik)

    return make_distribution(rv, blackbox_op, list_params, bounds)

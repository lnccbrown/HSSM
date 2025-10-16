"""Likelihood Approximation Network (LAN) utilities.

This module handles LAN-related operations, such as producing log-likelihood functions
from onnx model files and wrapping jax log-likelihood functions in pytensor Ops.
"""

from os import PathLike
from typing import Callable, Literal, overload

import jax.numpy as jnp
import numpy as np
import onnx
import pytensor.tensor as pt
from jax import grad, jit
from numpy.typing import ArrayLike

from .._types import LogLikeFunc, LogLikeGrad
from .jax import make_vmap_func
from .onnx_utils.model import load_onnx_model
from .onnx_utils.onnx2jax import make_jax_func
from .onnx_utils.onnx2pt import pt_interpret_onnx


@overload
def make_jax_logp_funcs_from_onnx(
    model: str | PathLike | onnx.ModelProto,
    params_is_reg: list[bool],
    params_only: bool = False,
    return_jit: Literal[True] = True,
) -> tuple[LogLikeFunc, LogLikeGrad, LogLikeFunc]: ...
@overload
def make_jax_logp_funcs_from_onnx(
    model: str | PathLike | onnx.ModelProto,
    params_is_reg: list[bool],
    params_only: bool = False,
    return_jit: Literal[False] = False,
) -> tuple[LogLikeFunc, LogLikeGrad]: ...
def make_jax_logp_funcs_from_onnx(
    model: str | PathLike | onnx.ModelProto,
    params_is_reg: list[bool],
    params_only: bool = False,
    return_jit: bool = True,
) -> tuple[LogLikeFunc, LogLikeGrad] | tuple[LogLikeFunc, LogLikeGrad, LogLikeFunc]:
    """Make a jax function and its Vector-Jacobian Product from an ONNX Model.

    Parameters
    ----------
    model:
        A path or url to the ONNX model, or an ONNX Model object that's
        already loaded.
    params_is_reg:
        A list of booleans indicating whether the parameters are regressions.
        Parameters that are regressions will not be vectorized in likelihood
        calculations.
    params_only:
        If True, the log-likelihood function will only take parameters as input.
    return_jit
        If `True`, the function will return a JIT-compiled version of the vectorized
        logp function, its VJP, and the non-jitted version of the logp function.
        If `False`, it will return the non-jitted version of the vectorized logp
        function and its VJP.

    Returns
    -------
    LogLikeFunc | tuple[LogLikeFunc, LogLikeFunc, LogLikeFunc]
        If `return_jit` is `True`, a triple of jax functions. The first calculates the
        forward pass, the second calculates the VJP, and the third is
        the forward-pass that's not jitted. When `params_only` is True, and all
        parameters are scalars, only a scalar function, its gradient, and the non-jitted
        version of the function are returned.
        If `return_jit` is `False`, a pair of jax functions. The first
        calculates the forward pass, and the second calculates the VJP of the logp
        function.
    """
    model_onnx = load_onnx_model(model)
    scalars_only = all(not is_reg for is_reg in params_is_reg)
    jax_func = make_jax_func(model_onnx)

    def logp(*inputs) -> np.ndarray:
        """Compute the log-likelihood.

        A function that computes the element-wise log-likelihoods given one data point
        and arbitrary numbers of parameters as scalars.

        Parameters
        ----------
        inputs
            A list of data and parameters used in the likelihood computation. Also
            supports the case where only parameters are passed.

        Returns
        -------
        jnp.ndarray
            The element-wise log-likelihoods.
        """
        # Makes a matrix to feed to the LAN model
        if params_only:
            # Constructing parameter vector
            input_vector = jnp.array(inputs)
        else:
            data = inputs[0]
            dist_params = inputs[1:]
            param_vector = jnp.array([inp.squeeze() for inp in dist_params])
            # AF-TODO: Unclear if trailing dimension of 1 is actually ever
            # happening.
            if param_vector.shape[-1] == 1:
                param_vector = param_vector.squeeze(axis=-1)
            input_vector = jnp.concatenate((param_vector, data))

        return jax_func(input_vector)

    # Special handling if parameters are scalars only
    if params_only and scalars_only:
        logp_vec = lambda *inputs: logp(*inputs).reshape((1,))
        if return_jit:
            return jit(logp_vec), jit(grad(logp)), logp_vec
        return logp_vec, grad(logp)

    in_axes: list[int | None] = [
        0 if is_regression else None for is_regression in params_is_reg
    ]
    if not params_only:
        in_axes.insert(0, 0)

    # This looks silly but is required to satisfy the type checker
    if return_jit:
        return make_vmap_func(
            logp, in_axes=in_axes, params_only=params_only, return_jit=True
        )
    return make_vmap_func(
        logp, in_axes=in_axes, params_only=params_only, return_jit=False
    )


def make_pytensor_logp_from_onnx(
    model: str | PathLike | onnx.ModelProto,
) -> Callable[..., ArrayLike]:
    """Convert onnx model file to pytensor.

    Parameters
    ----------
    model
        A path or url to the ONNX model, or an ONNX Model object that's
        already loaded.
    params_is_reg:
        A list of booleans indicating whether the parameters are regressions.
        Parameters that are regressions will not be vectorized in likelihood
        calculations.

    Returns
    -------
    Callable
        The logp function that applies the ONNX model to data and returns the element-
        wise log-likelihoods.
    """
    model_onnx = load_onnx_model(model)

    def logp(data: np.ndarray | None, *dist_params) -> ArrayLike:
        # Specify input layer of MLP
        if data is not None:
            data_dim = data.shape[1]
            inputs = pt.empty((data.shape[0], (len(dist_params) + data_dim)))
            inputs = pt.set_subtensor(inputs[:, -data_dim:], data)
        else:
            dist_params_tensors = [
                pt.as_tensor_variable(param) for param in dist_params
            ]
            n_rows = pt.max(
                [
                    1 if param.ndim == 0 else param.shape[0]
                    for param in dist_params_tensors
                ]
            )
            inputs = pt.empty((n_rows, len(dist_params)))
        for i, dist_param in enumerate(dist_params):
            inputs = pt.set_subtensor(
                inputs[:, i],
                dist_param,
            )

        # Returns elementwise log-likelihoods
        return pt.squeeze(pt_interpret_onnx(model_onnx.graph, inputs)[0])

    return logp


def make_jax_matrix_logp_funcs_from_onnx(
    model: str | PathLike | onnx.ModelProto,
) -> LogLikeFunc:
    """Make a jax function and its Vector-Jacobian Product from an ONNX Model.

    Parameters
    ----------
    model:
        A path or url to the ONNX model, or an ONNX Model object that's
        already loaded.

    Returns
    -------
    LogLikeFunc
        A jax function that is not jitted, which accepts a matrix passed
        directly to the neural network represented by the ONNX model to compute the
        log-likelihoods.
    """
    model_onnx = load_onnx_model(model)
    jax_func = make_jax_func(model_onnx)

    def logp(input_matrix) -> np.ndarray:
        """Compute the log-likelihood.

        A function that computes the element-wise log-likelihoods given one data point
        and arbitrary numbers of parameters as scalars.

        Parameters
        ----------
        inputs
            A list of data and parameters used in the likelihood computation. Also
            supports the case where only parameters are passed.

        Returns
        -------
        jnp.ndarray
            The element-wise log-likelihoods.
        """
        return jax_func(input_matrix)

    return logp

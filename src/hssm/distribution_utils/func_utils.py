"""Utility functions for creating JAX functions for log-likelihood computations."""

from typing import Callable, Literal, cast, overload

import numpy as np
from jax import jit, vjp, vmap
from jax.tree_util import Partial

from .._types import LogLikeFunc, LogLikeGrad


@overload
def make_vmap_func(
    logp: Callable,
    in_axes: list[int | None],
    params_only: bool = False,
    return_jit: Literal[True] = True,
) -> tuple[LogLikeFunc, LogLikeGrad, LogLikeFunc]: ...
@overload
def make_vmap_func(
    logp: Callable,
    in_axes: list[int | None],
    params_only: bool = False,
    return_jit: Literal[False] = False,
) -> tuple[LogLikeFunc, LogLikeGrad]: ...
def make_vmap_func(
    logp: Callable,
    in_axes: list[int | None],
    params_only: bool = False,
    return_jit: bool = True,
) -> tuple[LogLikeFunc, LogLikeGrad] | tuple[LogLikeFunc, LogLikeGrad, LogLikeFunc]:
    """Make a vectorized version of the logp function.

    Parameters
    ----------
    logp
        A JAX function that computes the log-likelihood.
    in_axes
        A list of booleans indicating which inputs to vectorize. If `True`, the
        corresponding input will be vectorized. If `False`, the input will not be
        vectorized. The length of this list should match the number of inputs to the
        `logp` function.
    params_only
        If `True`, the function will only vectorize the parameters and not the data.
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
    # The vectorization of the logp function
    vmap_logp = vmap(
        logp,
        in_axes=in_axes,
    )

    vjp_vmap_logp = make_vjp_func(
        vmap_logp,
        params_only=params_only,
    )

    if return_jit:
        return jit(vmap_logp), jit(vjp_vmap_logp), vmap_logp

    return vmap_logp, vjp_vmap_logp


def make_vjp_func(
    logp: Callable,
    params_only: bool = False,
    n_params: int | None = None,
) -> LogLikeGrad:
    """Make a non-jitted VJP of the logp function.

    Parameters
    ----------
    logp
        A JAX function that computes the log-likelihood.
    params_only
        If `True`, the function will only vectorize the parameters and not the data.

    Returns
    -------
    LogLikeGrad
        The VJP of the logp function.
    """

    def vjp_logp(
        *inputs: list[float | np.ndarray],
        gz: np.ndarray,
        params_only: bool = False,
        n_params: int | None = None,
    ) -> list[np.ndarray]:
        """Compute the VJP of the log-likelihood function.

        Parameters
        ----------
        inputs
            A list of data and parameters used in the likelihood computation. Also
            supports the case where only parameters are passed.
        gz
            The value of vmap_logp at which the VJP is evaluated, typically is just
            vmap_logp(data, *dist_params)

        Returns
        -------
        list[ArrayLike]
            The VJP of the log-likelihood function computed at gz.
        """
        _, vjp_fn = vjp(logp, *inputs)
        if params_only:
            if n_params is None:
                return vjp_fn(gz)
            return vjp_fn(gz)[:n_params]
        else:
            if n_params is None:
                return vjp_fn(gz)[1:]
            return vjp_fn(gz)[1 : n_params + 1]

    vjp_logp = Partial(vjp_logp, params_only=params_only, n_params=n_params)
    return cast("LogLikeGrad", vjp_logp)

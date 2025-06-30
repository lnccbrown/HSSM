"""Utilities for wraping JAX likelihoods in Pytensor Ops."""

from typing import Callable, Literal, cast, overload

import numpy as np
import pytensor
import pytensor.tensor as pt
from jax import jit, vjp, vmap
from numpy.typing import ArrayLike
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

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
) -> tuple[LogLikeFunc, LogLikeGrad] | tuple[LogLikeFunc, LogLikeFunc, LogLikeFunc]:
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

    def vjp_vmap_logp(
        *inputs: list[float | ArrayLike], gz: ArrayLike
    ) -> list[ArrayLike]:
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
        _, vjp_fn = vjp(vmap_logp, *inputs)
        return vjp_fn(gz) if params_only else vjp_fn(gz)[1:]

    if return_jit:
        return jit(vmap_logp), jit(vjp_vmap_logp), vmap_logp

    return vmap_logp, cast("LogLikeGrad", vjp_vmap_logp)


def make_jax_logp_ops(
    logp: LogLikeFunc,
    logp_vjp: LogLikeGrad,
    logp_nojit: LogLikeFunc,
) -> Op:
    """Wrap the JAX functions and its gradient in pytensor Ops.

    Parameters
    ----------
    logp
        A JAX function that represents the feed-forward operation of the LAN
        network.
    logp_vjp
        The Jax function that calculates the VJP of the logp function.
    logp_nojit
        The non-jit version of logp.

    Returns
    -------
    Op
        An pytensor op that wraps the feed-forward operation and can be used with
        pytensor.grad.
    """

    class LANLogpOp(Op):  # pylint: disable=W0223
        """Wraps a JAX function in an pytensor Op."""

        def make_node(self, data, *dist_params):
            """Take the inputs to the Op and puts them in a list.

            Also specifies the output types in a list, then feed them to the Apply node.

            Parameters
            ----------
            data
                A two-column numpy array with response time and response. Can be `None`
                for which case the log-likelihood is computed only from the parameters,
                which is required for choice-probability networks with binary choices.
            dist_params
                A list of parameters used in the likelihood computation. The parameters
                can be a mix of scalars and arrays.
            """
            inputs = [pt.as_tensor_variable(dist_param) for dist_param in dist_params]
            self.scalars_only = all(inp.ndim == 0 for inp in inputs)
            self.params_only = data is not None

            if self.params_only:
                inputs = [pt.as_tensor_variable(data)] + inputs

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

        def grad(self, inputs, output_gradients):
            """Perform the pytensor.grad() operation.

            Parameters
            ----------
            inputs
                The same as the inputs produced in `make_node`.
            output_gradients
                Holds the results of the perform `perform` method.

            Notes
            -----
                It should output the VJP of the Op. In other words, if this `Op`
                outputs `y`, and the gradient at `y` is grad(x), the required output
                is y*grad(x).
            """
            if self.params_only:
                results = lan_logp_vjp_op(
                    inputs[0], *inputs[1:], gz=output_gradients[0]
                )
            else:
                if self.scalars_only:
                    results = lan_logp_vjp_op(None, *inputs, gz=None)
                else:
                    results = lan_logp_vjp_op(None, *inputs, gz=output_gradients[0])

            output = results

            if self.params_only:
                output = [
                    pytensor.gradient.grad_not_implemented(self, 0, inputs[0]),
                ] + output

            return output

    class LANLogpVJPOp(Op):  # pylint: disable=W0223
        """Wraps the VJP operation of a jax function in an pytensor op."""

        def make_node(self, data, *dist_params, gz):
            """Take the inputs to the Op and puts them in a list.

            Also specifies the output types in a list, then feed them to the Apply node.

            Parameters
            ----------
            data:
                A two-column numpy array with response time and response.
            dist_params:
                A list of parameters used in the likelihood computation.
            """
            self.params_only = data is not None
            self.scalars_only = gz is None
            inputs = [pt.as_tensor_variable(dist_param) for dist_param in dist_params]
            if self.params_only:
                inputs = [pt.as_tensor_variable(data)] + inputs
            if not self.scalars_only:
                inputs += [pt.as_tensor_variable(gz)]

            if self.params_only:
                outputs = [inp.type() for inp in inputs[1:-1]]
            else:
                if self.scalars_only:
                    outputs = [inp.type() for inp in inputs]
                else:
                    outputs = [inp.type() for inp in inputs[:-1]]

            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            """Perform the Apply node.

            Parameters
            ----------
            inputs
                This is a list of data from which the values stored in
                `output_storage` are to be computed using non-symbolic language.
            output_storage
                This is a list of storage cells where the output
                is to be stored. A storage cell is a one-element list. It is
                forbidden to change the length of the list(s) contained in
                output_storage. There is one storage cell for each output of
                the Op.
            """
            if self.params_only:
                results = logp_vjp(*inputs[:-1], gz=inputs[-1])
            else:
                if self.scalars_only:
                    # NOTE: this looks like a bug, but it's not. The inputs are
                    # passed as a list so that the gradient are returned as a list.
                    # The reason why this works is that JAX can handle inputs as a
                    # list of scalars.
                    results = logp_vjp(inputs)
                else:
                    results = logp_vjp(*inputs[:-1], gz=inputs[-1])

            for i, result in enumerate(results):
                outputs[i][0] = np.asarray(result, dtype=node.outputs[i].dtype)

    lan_logp_op = LANLogpOp()
    lan_logp_vjp_op = LANLogpVJPOp()

    # Unwraps the JAX function for sampling with JAX backend.
    @jax_funcify.register(LANLogpOp)
    def logp_op_dispatch(op, **kwargs):  # pylint: disable=W0612,W0613
        return logp_nojit

    return lan_logp_op

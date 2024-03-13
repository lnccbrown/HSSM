"""Likelihood Approximation Network (LAN) utilities.

This module handles LAN-related operations, such as producing log-likelihood functions
from onnx model files and wrapping jax log-likelihood functions in pytensor Ops.
"""

from os import PathLike
from typing import Callable

import jax.numpy as jnp
import numpy as np
import onnx
import pytensor
import pytensor.tensor as pt
from jax import grad, jit, vjp, vmap
from numpy.typing import ArrayLike
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

from .onnx2pt import pt_interpret_onnx
from .onnx2xla import interpret_onnx

LogLikeFunc = Callable[..., ArrayLike]
LogLikeGrad = Callable[..., ArrayLike]


def make_jax_logp_funcs_from_onnx(
    model: str | PathLike | onnx.ModelProto,
    params_is_reg: list[bool],
    params_only: bool = False,
) -> tuple[LogLikeFunc, LogLikeGrad, LogLikeFunc]:
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

    Returns
    -------
    tuple[LogLikeFunc, LogLikeGrad, LogLikeFunc]
        A triple of jax functions. The first calculates the
        forward pass, the second calculates the VJP, and the third is
        the forward-pass that's not jitted. When `params_only` is True, and all
        parameters are scalars, only a scalar function, its gradient, and the non-jitted
        version of the function are returned.
    """
    loaded_model = (
        onnx.load(str(model)) if isinstance(model, (str, PathLike)) else model
    )

    scalars_only = all(not is_reg for is_reg in params_is_reg)

    def logp(*inputs) -> jnp.ndarray:
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
            input_vector = jnp.array(inputs)
        else:
            data = inputs[0]
            dist_params = inputs[1:]
            input_vector = jnp.concatenate((jnp.array(dist_params), data))

        return interpret_onnx(loaded_model.graph, input_vector)[0].squeeze()

    if params_only and scalars_only:
        logp_vec = lambda *inputs: logp(*inputs).reshape((1,))
        return jit(logp_vec), jit(grad(logp)), logp_vec

    in_axes: list = [0 if is_regression else None for is_regression in params_is_reg]
    if not params_only:
        in_axes = [0] + in_axes

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

    return jit(vmap_logp), jit(vjp_vmap_logp), vmap_logp


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


def make_pytensor_logp(
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
    loaded_model: onnx.ModelProto = (
        onnx.load(str(model)) if isinstance(model, (str, PathLike)) else model
    )

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
        return pt.squeeze(pt_interpret_onnx(loaded_model.graph, inputs)[0])

    return logp

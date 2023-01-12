"""
Likelihood Approximation Network (LAN) extension for the Wiener
First-Passage Time (WFPT) distribution.
Uses a neural network to approximate the likelihood function of
the Wiener First-Passage Time distribution.
"""

from __future__ import annotations

from os import PathLike
from typing import Callable, Tuple

import jax.numpy as jnp
import numpy as np
import onnx

# import aesara
# import aesara.tensor as at
import pytensor
import pytensor.tensor as pt
from jax import grad, jit
from numpy.typing import ArrayLike
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

from .onnx2aes import aes_interpret_onnx
from .onnx2xla import interpret_onnx

LogLikeFunc = Callable[..., ArrayLike]
LogLikeGrad = Callable[..., ArrayLike]


class LAN:
    """
    A factory class handling LAN-related operations, such as producing log-likelihood
    functions from onnx model files and wrapping jax log-likelihood functions in aesara
    Ops.
    """

    @classmethod
    def make_jax_logp_funcs_from_onnx(
        cls,
        model: str | PathLike | onnx.ModelProto,
        n_params: int,
    ) -> Tuple[LogLikeFunc, LogLikeGrad, LogLikeFunc,]:
        """Makes a jax function from an ONNX model.
        Args:
            model: A path or url to the ONNX model, or an ONNX Model object
            already loaded.
            compile: Whether to use jit in jax to compile the model.
        Returns: A triple of jax or Python functions. The first calculates the
            forward pass, the second calculates the gradient, and the third is
            the forward-pass that's not jitted.
        """

        loaded_model = (
            onnx.load(str(model)) if isinstance(model, (str, PathLike)) else model
        )

        def logp(data: np.ndarray, *dist_params) -> ArrayLike:
            """
            Computes the sum of the log-likelihoods given data and arbitrary
            numbers of parameters.
            Args:
                data: a two-column numpy array with response time and response
                dist_params: a list of parameters used in the likelihood computation.
            Returns:
                The sum of log-likelihoods.
            """

            # Makes a matrix to feed to the LAN model
            data = data.reshape(-1, 2)
            params_matrix = jnp.repeat(
                jnp.stack(dist_params).reshape(1, -1), axis=0, repeats=data.shape[0]
            )
            input_matrix = jnp.hstack([params_matrix, data])

            return jnp.sum(
                jnp.squeeze(interpret_onnx(loaded_model.graph, input_matrix)[0])
            )

        logp_grad = grad(logp, argnums=range(1, 1 + n_params))

        return jit(logp), jit(logp_grad), logp

    @staticmethod
    def make_jax_logp_ops(
        logp: LogLikeFunc,
        logp_grad: LogLikeGrad,
        logp_nojit: LogLikeFunc,
    ) -> Op:
        """Wraps the JAX functions and its gradient in Aesara Ops.
        Args:
            logp: A JAX function that represents the feed-forward operation of the
                LAN network.
            logp_grad: The derivative of the above function.
            logp_nojit: A Jax function
        Returns:
            An aesara op that wraps the feed-forward operation and can be used with
            aesara.grad.
        """

        class LANLogpOp(Op):  # pylint: disable=W0223
            """Wraps a JAX function in an aesara Op."""

            def make_node(self, data, *dist_params):
                inputs = [
                    pt.as_tensor_variable(data),
                ] + [pt.as_tensor_variable(dist_param) for dist_param in dist_params]

                outputs = [pt.scalar()]

                return Apply(self, inputs, outputs)

            def perform(self, node, inputs, output_storage):
                """Performs the Apply node.
                Args:
                    inputs: This is a list of data from which the values stored in
                        output_storage are to becomputed using non-symbolic language.
                    output_storage: This is a list of storage cells where the output
                        is to be stored. A storage cell is a one-element list. It is
                        forbidden to change the length of the list(s) contained in
                        output_storage. There is one storage cell for each output of
                        the Op.
                """
                result = logp(*inputs)
                output_storage[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)

            def grad(self, inputs, output_grads):
                results = lan_logp_grad_op(*inputs)
                output_gradient = output_grads[0]
                output = [
                    pytensor.gradient.grad_not_implemented(self, 0, inputs[0]),
                ] + [output_gradient * result for result in results]
                return output

        class LANLogpGradOp(Op):  # pylint: disable=W0223
            """Wraps the gradient opearation of a jax function in an aesara op."""

            def make_node(self, data, *dist_params):
                inputs = [
                    pt.as_tensor_variable(data),
                ] + [pt.as_tensor_variable(dist_param) for dist_param in dist_params]
                outputs = [inp.type() for inp in inputs[1:]]

                return Apply(self, inputs, outputs)

            def perform(self, node, inputs, outputs):
                results = logp_grad(*inputs)

                for i, result in enumerate(results):
                    outputs[i][0] = np.asarray(result, dtype=node.outputs[i].dtype)

        lan_logp_op = LANLogpOp()
        lan_logp_grad_op = LANLogpGradOp()

        # Unwraps the JAX function for sampling with JAX backend.
        @jax_funcify.register(LANLogpOp)
        def logp_op_dispatch(op, **kwargs):  # pylint: disable=W0612,W0613
            return logp_nojit

        return lan_logp_op

    @classmethod
    def make_aesara_logp(cls, model: str | PathLike | onnx.ModelProto):
        """
        Converting onnx model file to aesara
        Args:
            model (str): path to onnx model file
            data: the input dataset
        Returns:
            model applied on a data
        """
        loaded_model: onnx.ModelProto = (
            onnx.load(str(model)) if isinstance(model, (str, PathLike)) else model
        )

        def logp(data: np.ndarray, *dist_params) -> ArrayLike:

            # Specify input layer of MLP
            data = data.reshape((-1, 2))
            inputs = pt.zeros(
                (data.shape[0], len(dist_params) + 2)
            )  # (n_trials, number of parameters + 2 [for rt and choice columns])
            inputs = pt.set_subtensor(inputs[:, :-2], pt.stack(dist_params))
            inputs = pt.set_subtensor(inputs[:, -2:], data)
            return pt.sum(pt.squeeze(aes_interpret_onnx(loaded_model.graph, inputs)[0]))

        return logp

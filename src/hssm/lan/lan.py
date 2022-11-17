"""
Likelihood Approximation Network (LAN) extension for the Wiener
First-Passage Time (WFPT) distribution.

Uses a neural network to approximate the likelihood function of
the Wiener First-Passage Time distribution.
"""

from __future__ import annotations

from os import PathLike
from typing import Any, Callable, List, Tuple

import aesara
import aesara.tensor as at
import jax.numpy as jnp
import numpy as np
import onnx
import pymc as pm
from aesara.graph import Apply, Op
from jax import Array, grad, jit

from hssm.lan.onnx2xla import interpret_onnx
from hssm.wfpt import WFPTRandomVariable

aesara.config.floatX = "float32"


def _make_jax_funcs_from_onnx(
    model: str | PathLike | onnx.Model, compile_funcs: bool = True
) -> Tuple[Callable[[np.ndarray], Array], Callable[[np.ndarray], Array]]:
    """Makes a jax function from an ONNX model.

    Args:
        model: A path or url to the ONNX model, or an ONNX Model object already loaded.
        compile: Whether to use jit in jax to compile the model.

    Return: A tuple of jax or Python functions. The frist is the forward pass, and the
        other is the derivative.
    """

    loaded_model: onnx.Model = (
        onnx.load(model) if isinstance(model, (str, PathLike)) else model
    )

    def lan_logp(inputs: np.ndarray) -> Array:

        # Sum of all log-likelihoods.
        # Necessary because jax requires outputs to be scalars.
        return jnp.sum(
            # Suppresses dimensions of 1s
            jnp.squeeze(
                # interpret_onnx returns a list of outputs
                # We are only extracting the first one
                interpret_onnx(loaded_model.graph, inputs)[0]
            )
        )

    lan_logp_grad = grad(lan_logp)

    if compile_funcs:
        return jit(lan_logp), jit(lan_logp_grad)

    return lan_logp, lan_logp_grad


def _make_lan_logp_ops(
    logp: Callable[[np.ndarray], Array], logp_grad: Callable[[np.ndarray], Array]
) -> Op:
    """Wraps the JAX functions and its derivatives in Aesara Ops.

    Args:
        jax_func: A JAX function that represents the feed-forward operation of the
            LAN network
        jax_func_grad: The derivative of the above function

    Returns:
        An aesara op that wraps the feed-forward operation and can be used with
            aesara.grad.
    """

    class LogpOp(Op):
        """Wraps the jax function logp in an aesara op."""

        def make_node(self, inputs: Any) -> Apply[Any]:  # type: ignore
            """Transforms inputs for the Apply node"""

            inputs = [at.as_tensor_variable(inputs)]
            outputs = [at.dscalar()]

            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, output_storage):
            """Performs the operations.

            Args:
                node: The node of this op in the computational graph
                inputs: This is a list of data from which the values stored in
                    output_storage are to be computed using non-symbolic
                    language.
                output_storage:  This is a list of storage cells where the output
                    is to be stored. A storage cell is a one-element list. It is
                    forbidden to change the length of the list(s) contained in
                    output_storage. There is one storage cell for each
                    output of the Op.
            """
            result = logp(*inputs)
            output_storage[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)

        def grad(self, inputs, output_grads):
            results = logp_grad_op(*inputs)
            outputs = output_grads[0]
            return [results * outputs]

    class LogpGradOp(Op):
        """Wraps the gradient jax function logp in an aesara op."""

        def make_node(self, inputs):
            inputs = [at.as_tensor_variable(inputs)]
            outputs = [at.dmatrix()]

            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            results = logp_grad(*inputs)
            outputs[0][0] = np.asarray(results, dtype=node.outputs[0].dtype)

    logp_op = LogpOp()
    logp_grad_op = LogpGradOp()

    return logp_op


def make_lan_class(model: str | PathLike | onnx.Model, compile_funcs: bool = True):
    """Makes a lan distribution with the Model passed to it as the
    log-likelihood function.

    Args:
        model: the path to or the loaded ONNX model.
        compile_funcs: whether to compile the jax functions

    Returns: A LAN class with the ONNX model as its log-likelihood function.
    """
    logp, logp_grad = _make_jax_funcs_from_onnx(model, compile_funcs)
    lan_logp_op = _make_lan_logp_ops(logp, logp_grad)

    class LAN(pm.Distribution):
        """Wiener first-passage time (WFPT) log-likelihood for LANs."""

        # This is just a placeholder because pm.Distribution requires an rv_op
        # Might be updated in the future once
        rv_op = WFPTRandomVariable()

        @classmethod
        # Use a list of strs to specify both the number and order of parameters
        def dist(cls, params: List[str], **kwargs):  # pylint: disable=arguments-renamed
            dist_params = [
                at.as_tensor_variable(pm.floatX(kwargs[param])) for param in params
            ]
            other_kwargs = {k: v for k, v in kwargs.items() if k not in params}
            return super().dist(dist_params, **other_kwargs)

        def logp(data, *dist_params):

            ## Deconstruct rt into positive rts and a boolean vector of response
            rt = at.abs(data)
            response = at.where(data >= 0, 1.0, -1.0)

            # Specify input layer of MLP
            inputs = at.zeros(
                (rt.shape[0], len(dist_params) + 2)
            )  # (n_trials, number of parameters + 2 [for rt and choice columns])
            inputs = at.set_subtensor(inputs[:, :-2], at.stack(dist_params))
            inputs = at.set_subtensor(inputs[:, -2], rt)
            inputs = at.set_subtensor(inputs[:, -1], response)

            return lan_logp_op(inputs)

    return LAN

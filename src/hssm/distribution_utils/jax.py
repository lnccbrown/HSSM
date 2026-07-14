"""Utilities for wraping JAX likelihoods in Pytensor Ops."""

from typing import Callable, Literal, overload

import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt
from jax import jit
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

from .._types import LogLikeFunc, LogLikeGrad
from .func_utils import make_vjp_func, make_vmap_func


class LANLogpVJPOp(Op):  # pylint: disable=W0223
    """Wraps the VJP of a JAX log-likelihood function in a pytensor Op.

    Parameters
    ----------
    logp_vjp
        A JAX function computing the VJP of the log-likelihood, with signature
        `logp_vjp(data, *params, gz)` (or `logp_vjp(*params, gz)` when there is
        no data), where `gz` is the cotangent at the output.
    n_params : optional
        Number of parameters used in the likelihood computation. Only required
        when `extra_fields` are used, in which case the Op will not produce
        gradient outputs for the extra fields.
    """

    def __init__(self, logp_vjp: LogLikeGrad, n_params: int | None = None):
        self.logp_vjp = logp_vjp
        self.n_params = n_params

    def do_constant_folding(self, fgraph, node):
        """Keep PyTensor from trying to precompute opaque JAX-backed outputs."""
        return False

    # pyrefly: ignore[bad-override]
    def make_node(self, data, *dist_params, gz):
        """Take the inputs to the Op and puts them in a list.

        Also specifies the output types in a list, then feed them to the Apply node.

        Parameters
        ----------
        data:
            A two-column numpy array with response time and response, or `None`
            for likelihoods computed from parameters only.
        dist_params:
            A list of parameters used in the likelihood computation.
        gz:
            The cotangent (upstream gradient) at the forward Op's output.
        """
        has_data = data is not None
        inputs = [pt.as_tensor_variable(dist_param) for dist_param in dist_params]
        if has_data:
            inputs = [pt.as_tensor_variable(data)] + inputs
        inputs += [pt.as_tensor_variable(gz)]

        grad_inputs = inputs[1:-1] if has_data else inputs[:-1]
        if self.n_params is not None:
            grad_inputs = grad_inputs[: self.n_params]

        outputs = [inp.type() for inp in grad_inputs]

        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, output_storage):
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
        results = self.logp_vjp(*inputs[:-1], gz=inputs[-1])

        for i, result in enumerate(results):
            output_storage[i][0] = np.asarray(result, dtype=node.outputs[i].dtype)


class LANLogpOp(Op):  # pylint: disable=W0223
    """Wraps a JAX log-likelihood function in a pytensor Op.

    Parameters
    ----------
    logp
        A JAX function that represents the feed-forward operation of the LAN
        network.
    logp_nojit
        The non-jit version of logp, used when the whole graph is compiled
        with the JAX linker.
    vjp_op
        The `LANLogpVJPOp` emitted by `pullback` for reverse-mode gradients.
    n_params : optional
        Number of parameters used in the likelihood computation. Only required
        when `extra_fields` are used, in which case the resulting Op will not
        compute gradients with respect to the extra fields.
    """

    def __init__(
        self,
        logp: LogLikeFunc,
        logp_nojit: LogLikeFunc,
        vjp_op: LANLogpVJPOp,
        n_params: int | None = None,
    ):
        self.logp = logp
        self.logp_nojit = logp_nojit
        self.vjp_op = vjp_op
        self.n_params = n_params
        # Locked on first make_node call. The wrapped JAX callables have a
        # single calling convention (with or without data) baked in, so one
        # Op instance cannot serve both configurations; locking turns
        # conflicting reuse into an explicit error instead of silently
        # wrong gradients in `pullback`.
        self.has_data: bool | None = None

    def do_constant_folding(self, fgraph, node):
        """Keep PyTensor from trying to precompute opaque JAX-backed outputs."""
        return False

    # pyrefly: ignore[bad-override]
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
        has_data = data is not None

        if self.has_data is None:
            self.has_data = has_data
        elif self.has_data != has_data:
            raise ValueError(
                "This LANLogpOp instance was previously applied "
                f"{'with' if self.has_data else 'without'} data and is now "
                f"being applied {'with' if has_data else 'without'} data. "
                "The wrapped JAX callables support a single calling "
                "convention; create a separate Op via `make_jax_logp_ops` "
                "for each configuration."
            )

        if has_data:
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
        result = self.logp(*inputs)
        output_storage[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)

    def pullback(self, inputs, outputs, cotangents):
        """Construct the graph for the VJP (reverse-mode gradient) of the Op.

        Called by `pytensor.grad` when building the symbolic gradient graph.
        The cotangent is always passed through to the VJP so the chain rule
        holds for any downstream graph, not just unit cotangents.

        Parameters
        ----------
        inputs
            The same as the inputs produced in `make_node`.
        outputs
            The symbolic outputs of the node (unused; the VJP is computed
            from the inputs directly).
        cotangents
            The cotangents (upstream gradients) for each output.

        Notes
        -----
            It should output the VJP of the Op. In other words, if this `Op`
            outputs `y`, and the gradient at `y` is grad(x), the required output
            is y*grad(x).
        """
        if self.has_data:
            results = self.vjp_op(inputs[0], *inputs[1:], gz=cotangents[0])
        else:
            results = self.vjp_op(None, *inputs, gz=cotangents[0])

        output = list(results) if isinstance(results, (list, tuple)) else [results]

        if self.has_data:
            output = [
                pytensor.gradient.grad_not_implemented(self, 0, inputs[0]),
            ] + output

        if self.n_params is not None:
            start_idx = self.n_params + (1 if self.has_data else 0)
            output[start_idx:] = [
                pytensor.gradient.grad_undefined(self, i, inputs[i])
                for i in range(start_idx, len(inputs))
            ]

        return output


# Unwraps the JAX function for compilation with the JAX linker (e.g. sampling
# through numpyro/blackjax, or pm.fit / pm.sample with backend="jax").
@jax_funcify.register(LANLogpOp)
def lan_logp_op_dispatch(op, **kwargs):  # pylint: disable=W0613
    """Return the non-jitted forward function for the JAX linker."""
    return op.logp_nojit


# Required when PyTensor differentiates LANLogpOp symbolically (e.g. ADVI
# via `pm.fit(..., backend="jax")`) and then compiles the gradient graph
# with the JAX linker. Mirrors the input contract of LANLogpVJPOp.perform.
@jax_funcify.register(LANLogpVJPOp)
def lan_logp_vjp_op_dispatch(op, node, **kwargs):  # pylint: disable=W0613
    """Wrap the VJP function for the JAX linker.

    PyTensor's generated code tuple-unpacks multi-output nodes but assigns
    single-output nodes directly, so the wrapper returns a bare array when
    the node has exactly one output.
    """
    n_outputs = len(node.outputs)

    def lan_logp_vjp_jax(*inputs):
        results = op.logp_vjp(*inputs[:-1], gz=inputs[-1])
        return tuple(results) if n_outputs > 1 else results[0]

    return lan_logp_vjp_jax


def make_jax_logp_ops(
    logp: LogLikeFunc,
    logp_vjp: LogLikeGrad,
    logp_nojit: LogLikeFunc,
    n_params: int | None = None,
) -> Op:
    """Wrap the JAX functions and its gradient in pytensor Ops.

    Parameters
    ----------
    logp
        A JAX function that represents the feed-forward operation of the LAN
        network.
    logp_vjp
        The Jax function that calculates the VJP of the logp function. Its
        signature is `logp_vjp(data, *params, gz)` (or `logp_vjp(*params, gz)`
        for likelihoods without data), where `gz` is the cotangent.
    logp_nojit
        The non-jit version of logp.
    n_params : optional
        Number of parameters used in the likelihood computation. Only required
        when `extra_fields` are used, in which case the resulting Op will not
        compute gradients with respect to the extra fields.

    Returns
    -------
    Op
        An pytensor op that wraps the feed-forward operation and can be used with
        pytensor.grad.
    """
    return LANLogpOp(logp, logp_nojit, LANLogpVJPOp(logp_vjp, n_params), n_params)


@overload
def make_jax_logp_funcs_from_callable(
    logp: Callable,
    vmap: bool = True,
    params_is_reg: list[bool] | None = None,
    params_only: bool = False,
    return_jit: Literal[True] = True,
) -> tuple[LogLikeFunc, LogLikeGrad, LogLikeFunc]: ...
@overload
def make_jax_logp_funcs_from_callable(
    logp: Callable,
    vmap: bool = True,
    params_is_reg: list[bool] | None = None,
    params_only: bool = False,
    return_jit: Literal[False] = False,
) -> tuple[LogLikeFunc, LogLikeGrad]: ...
def make_jax_logp_funcs_from_callable(
    logp: Callable,
    vmap: bool = True,
    params_is_reg: list[bool] | None = None,
    params_only: bool = False,
    return_jit: bool = True,
) -> tuple[LogLikeFunc, LogLikeGrad] | tuple[LogLikeFunc, LogLikeGrad, LogLikeFunc]:
    """Make a jax function and its Vector-Jacobian Product from a jax callable.

    Parameters
    ----------
    logp:
        A jax callable function that computes log-likelihoods. The function should have
        this signature: `logp(data, *params, [*extra_fields]) -> jnp.ndarray` where
        extra_fields are optional additional fields that can be used in the
        likelihood computation. The `data` argument is a two-column numpy array
        with response time and response.
    vmap:
        If `True`, the function will be vectorized using JAX's vmap. If `False`, the
        function is assumed to be already vectorized.
    params_is_reg:
        A list of booleans indicating whether the parameters are regressions.
        Parameters that are regressions will not be vectorized in likelihood
        calculations.
    params_only:
        Controls the expected signature of the ``logp`` callable.
        If False (default), the callable signature is ``f(data, *params)``,
        where ``data`` is a 2-column array of [rt, choice].  This is the
        standard case for LANs and other likelihoods that condition on
        observed data.
        If True, the callable signature is ``f(*params)`` with no data
        argument.  This is used for Choice Probability Networks (CPNs)
        and Outcome Probability Networks (OPNs).
    return_jit
        If `True`, the function will return a JIT-compiled version of the vectorized
        logp function, its VJP, and the non-jitted version of the logp function.
        If `False`, it will return the non-jitted version of the vectorized logp
        function and its VJP.

    Returns
    -------
    tuple[LogLikeFunc, LogLikeGrad] | tuple[LogLikeFunc, LogLikeFunc, LogLikeFunc]
        If `return_jit` is `True`, a triple of jax functions. The first calculates the
        forward pass, the second calculates the VJP, and the third is
        the forward-pass that's not jitted. When `params_only` is True, and all
        parameters are scalars, only a scalar function, its gradient, and the non-jitted
        version of the function are returned.
        If `return_jit` is `False`, a pair of jax functions. The first
        calculates the forward pass, and the second calculates the VJP of the logp
        function.
    """
    if vmap and params_is_reg is None:
        raise ValueError(
            "If `vmap` is True, `params_is_reg` must be provided to specify which "
            "parameters are regressions."
        )

    # Looks silly but is required to please mypy.
    if vmap and params_is_reg is not None:
        in_axes: list[int | None] = [
            0 if is_regression else None for is_regression in params_is_reg
        ]
        if not params_only:
            in_axes.insert(0, 0)

        if all(axis is None for axis in in_axes):
            raise ValueError(
                "No vmap is needed in your use case, since all parameters are scalars."
            )

        if return_jit:
            return make_vmap_func(
                logp,
                in_axes=in_axes,
                params_only=params_only,
                return_jit=True,
            )
        return make_vmap_func(
            logp,
            in_axes=in_axes,
            params_only=params_only,
            return_jit=False,
        )

    vjp_logp = make_vjp_func(
        logp,
        params_only=params_only,
    )

    if return_jit:
        return jit(logp), jit(vjp_logp), logp

    return logp, vjp_logp


# AF-TODO: This needs some tests!
def make_jax_single_trial_logp_from_network_forward(
    jax_forward_fn: Callable, params_only: bool = False
) -> Callable:
    """Make a JAX log-likelihood function from a JAX forward function.

    This function creates a JAX log-likelihood function that computes the element-wise
    log-likelihoods given one data point and arbitrary numbers of parameters as scalars.

    Parameters
    ----------
    jax_forward_fn : Callable
        The JAX forward function to use for the log-likelihood computation.
    params_only : bool, optional
        Controls the expected signature of the returned callable.
        If False (default), the returned function expects
        ``(data, *params)`` where ``data`` is a 2-column array of
        [rt, choice].  This is the standard case for LANs and other
        likelihoods that condition on observed data.
        If True, the returned function expects ``(*params)`` with no
        data argument.  This is used for Choice Probability Networks
        (CPNs) and Outcome Probability Networks (OPNs).

    Returns
    -------
    Callable
        A JAX function that computes the element-wise
        log-likelihoods given one data point and arbitrary
        numbers of parameters as scalars.
    """

    def jax_single_trial_logp_from_lan_forward(*inputs) -> np.ndarray:
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
        data = inputs[0]
        dist_params = inputs[1:]

        param_vector = jnp.array([inp.squeeze() for inp in dist_params])

        if param_vector.shape[-1] == 1:
            param_vector = param_vector.squeeze(axis=-1)

        input_vector = jnp.concatenate((param_vector, data))
        return jax_forward_fn(input_vector).squeeze()

    def jax_single_trial_logp_from_opn_cpn_forward(*inputs) -> np.ndarray:
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
        input_vector = jnp.array(inputs)
        return jax_forward_fn(input_vector).squeeze()

    if params_only:
        return jax_single_trial_logp_from_opn_cpn_forward
    else:
        return jax_single_trial_logp_from_lan_forward

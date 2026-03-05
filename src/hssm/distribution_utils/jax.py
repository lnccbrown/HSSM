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
        The Jax function that calculates the VJP of the logp function.
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
            self.is_scalars_only = all(inp.ndim == 0 for inp in inputs)
            # params_only means calculate gradients only with respect to the
            # parameters, not the data.
            self.is_params_only = data is not None
            self.n_params = n_params

            if self.is_params_only:
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
            if self.is_params_only:
                results = lan_logp_vjp_op(
                    inputs[0], *inputs[1:], gz=output_gradients[0]
                )
            else:
                if self.is_scalars_only:
                    results = lan_logp_vjp_op(None, *inputs, gz=None)
                else:
                    results = lan_logp_vjp_op(None, *inputs, gz=output_gradients[0])

            output = results

            if self.is_params_only:
                output = [
                    pytensor.gradient.grad_not_implemented(self, 0, inputs[0]),
                ] + output

            if self.n_params is not None:
                start_idx = self.n_params + 1 if self.is_params_only else 0
                for i in range(start_idx, len(output)):
                    output[i] = pytensor.gradient.grad_undefined(self, i, inputs[i])

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
            self.is_params_only = data is not None
            self.is_scalars_only = gz is None
            inputs = [pt.as_tensor_variable(dist_param) for dist_param in dist_params]
            if self.is_params_only:
                inputs = [pt.as_tensor_variable(data)] + inputs
            if not self.is_scalars_only:
                inputs += [pt.as_tensor_variable(gz)]

            if self.is_params_only:
                outputs = [inp.type() for inp in inputs[1:-1]]
            else:
                if self.is_scalars_only:
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
            if self.is_params_only:
                results = logp_vjp(*inputs[:-1], gz=inputs[-1])
            else:
                if self.is_scalars_only:
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
        If True, the log-likelihood function will only take parameters as input.
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

    print("params_only: ", params_only)
    print("params_is_reg: ", params_is_reg)

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
        Whether to compute the log-likelihood for only the parameters.
        This will not assume a data part in the input.
        `params_only = True` is appropriate for CPNs and OPNs,
        where the data is not used in the log-likelihood computation.
        `params_only = False` is appropriate for LANs,
        where the data is used in the log-likelihood computation.

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

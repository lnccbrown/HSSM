from functools import partial
from typing import Any, Callable, Literal, Type
from os import PathLike

import numpy as np
from numpy.typing import ArrayLike
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor import function
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.random.op import RandomVariable
import onnx

import jax.numpy as jnp
from jax import jit, vmap

import bambi as bmb
from ssms.basic_simulators.simulator import simulator

from .utils import decorate_atomic_simulator

LOGP_LB = pm.floatX(-66.1)

LogLikeFunc = Callable[..., ArrayLike]
LogLikeGrad = Callable[..., ArrayLike]


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

        def make_node(self, subj, data, *dist_params):
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
            param_inputs = [pt.as_tensor_variable(dist_param) for dist_param in dist_params]
            self.scalars_only = all(inp.ndim == 0 for inp in param_inputs)
            self.params_only = data is not None

            # COMMENT
            # Format inputs to have subj as the first dimension (for vectorization) followed by all the parameters.
            inputs = [pt.as_tensor_variable(subj)] + param_inputs

            if self.params_only:
                inputs = [pt.as_tensor_variable(subj), pt.as_tensor_variable(data)] + param_inputs

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

            num_params = 6
            num_extra_fields = 3

            if self.params_only:
                # COMMENT
                results = lan_logp_vjp_op(
                    inputs[0], inputs[1], *inputs[2:], gz=output_gradients[0]
                )
            else:
                if self.scalars_only:
                    results = lan_logp_vjp_op(inputs[0], None, *inputs[1:], gz=None)
                else:
                    results = lan_logp_vjp_op(inputs[0], None, *inputs[1:], gz=output_gradients[0])

            # COMMENT
            output = [None] * len(inputs)
            
            # COMMENT
            # Format output to compute gradients only with respect to the params. 
            # Gradients should not be computed for inputs indexes 0 (subj) and 1 (data).
            if self.params_only:

                output[0] = pytensor.gradient.grad_not_implemented(self, 0, inputs[0])
                output[1] = pytensor.gradient.grad_not_implemented(self, 1, inputs[1])
                for i in range(num_params):
                    output[2 + i] = results[i]
                
                for i in range(num_extra_fields):
                    input_idx = 2 + num_params + i
                    output[input_idx] = pytensor.gradient.grad_undefined(
                        self, input_idx, inputs[input_idx])

            return output

    class LANLogpVJPOp(Op):  # pylint: disable=W0223
        """Wraps the VJP operation of a jax function in an pytensor op."""

        def make_node(self, subj, data, *dist_params, gz):
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
            param_inputs = [pt.as_tensor_variable(dist_param) for dist_param in dist_params]
            inputs = [pt.as_tensor_variable(subj)] + param_inputs

            # COMMENT
            if self.params_only:
                inputs = [pt.as_tensor_variable(subj), pt.as_tensor_variable(data)] + param_inputs
            if not self.scalars_only:
                inputs += [pt.as_tensor_variable(gz)]

            # COMMENT
            if self.params_only:
                # COMMENT
                outputs = [inp.type() for inp in inputs[2:-1]]
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
            # COMMENT
            if self.params_only:
                # COMMENT
                results = logp_vjp(*inputs[:-1], gz=inputs[-1])[2:]
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


def make_likelihood_callable_TEST(
    loglik: pytensor.graph.Op | Callable | PathLike | str,
    loglik_kind: Literal["analytical", "approx_differentiable", "blackbox"],
    backend: Literal["pytensor", "jax", "other"] | None,
    params_is_reg: list[bool] | None = None,
    params_only: bool | None = None,
) -> pytensor.graph.Op | Callable:

    if isinstance(loglik, pytensor.graph.Op):
        return loglik

    if callable(loglik):
        # In the analytical case, if `backend` is None or `pytensor`, we can use the
        # callable directly. Otherwise, we wrap it in a BlackBoxOp.
        if loglik_kind == "analytical":
            if backend is None or backend == "pytensor":
                return loglik
            return make_blackbox_op(loglik)
        elif loglik_kind == "blackbox":
            return make_blackbox_op(loglik)
        elif loglik_kind == "approx_differentiable":
            if backend is None or backend == "jax":
                if params_is_reg is None:
                    raise ValueError(
                        "You set `loglik_kind` to `approx_differentiable` "
                        + "and `backend` to `jax` and supplied a jax callable, "
                        + "but did not set `params_is_reg`."
                    )
                logp, logp_grad, logp_nojit = make_jax_logp_funcs_from_jax_callable_TEST(
                    loglik,
                    params_is_reg,
                    params_only=False if params_only is None else params_only,
                )
                lan_logp_jax = make_jax_logp_ops(logp, logp_grad, logp_nojit)
                return lan_logp_jax
            if backend == "pytensor":
                raise ValueError(
                    "You set `backend` to `pytensor`, `loglik_kind` to"
                    + "`approx_differentiable` and provided a callable."
                    + "Currently we support only jax callables in this case."
                )
            # In the approx_differentiable case or the blackbox case, unless the backend
            # is `pytensor`, we wrap the callable in a BlackBoxOp.
        # if backend == "pytensor":
        #     return loglik
        #     # In all other cases, we assume that the callable cannot be directly
        #     # used in the backend and thus we wrap it in a BlackBoxOp
        # return make_blackbox_op(loglik)

    # Other cases, when `loglik` is a string or a PathLike.
    if loglik_kind != "approx_differentiable":
        raise ValueError(
            "You set `loglik_kind` to `approx_differentiable "
            + "but did not provide a pm.Distribution, an Op, or a callable "
            + "as `loglik`."
        )

    if isinstance(loglik, (str, PathLike)):
        if not Path(loglik).exists():
            loglik = download_hf(str(loglik))

    onnx_model = onnx.load(str(loglik))

    if backend == "pytensor":
        lan_logp_pt = make_pytensor_logp(onnx_model)
        return lan_logp_pt

    if params_is_reg is None:
        raise ValueError(
            "You set `loglik_kind` to `approx_differentiable` "
            + "and `backend` to `jax` but did not provide `params_is_reg`."
        )

    logp, logp_grad, logp_nojit = make_jax_logp_funcs_from_onnx(
        onnx_model,
        params_is_reg,
        params_only=False if params_only is None else params_only,
    )
    lan_logp_jax = make_jax_logp_ops(logp, logp_grad, logp_nojit)

    return lan_logp_jax


def make_jax_logp_funcs_from_jax_callable_TEST(
    model: Callable,
    params_is_reg: list[bool],
    params_only: bool = False,
) -> tuple[LogLikeFunc, LogLikeGrad, LogLikeFunc]:

    return make_jax_logp_funcs_from_onnx_TEST(model, params_is_reg, params_only)

def make_jax_logp_funcs_from_onnx_TEST(
    model: str | PathLike | onnx.ModelProto | Callable,
    params_is_reg: list[bool],
    params_only: bool = False,
) -> tuple[LogLikeFunc, LogLikeGrad, LogLikeFunc]:

    if isinstance(model, (str, PathLike, onnx.ModelProto)):
        loaded_model = (
            onnx.load(str(model)) if isinstance(model, (str, PathLike)) else model
        )
    else:
        model_callable = model

    scalars_only = all(not is_reg for is_reg in params_is_reg)

    def logp(*inputs) -> jnp.ndarray:

        # COMMENT
        # Makes a matrix to feed to the LAN model
        if params_only:
            # COMMENT
            input_vector = jnp.array(inputs)
        else:
            # COMMENT
            data = jnp.array([inputs[1].squeeze()]).squeeze().T

            # COMMENT
            num_params = 6
            num_extra_fields = 3
            trials_per_subject = 200
            
            # COMMENT
            input_vector = jnp.array([inp.squeeze() for inp in inputs[2:]])
            input_vector = jnp.concatenate((data, input_vector)).T

        if isinstance(model, (str, PathLike, onnx.ModelProto)):
            return interpret_onnx(loaded_model.graph, input_vector)[0].squeeze()
        else:
            # COMMENT
            subj = jnp.array(inputs[0].squeeze())
            
            # COMMENT
            return model_callable(subj, input_vector[:, 0:2], input_vector[:, 2], input_vector[:, 3],
                                        input_vector[:, 4], input_vector[:, 5], input_vector[:, 6], input_vector[:, 7],
                                        input_vector[:, 8], input_vector[:, 9], input_vector[:, 10])

    if params_only and scalars_only:
        logp_vec = lambda *inputs: logp(*inputs).reshape((1,))
        return jit(logp_vec), jit(grad(logp)), logp_vec

    in_axes: list = [0 if is_regression else None for is_regression in params_is_reg]
    if not params_only:
        in_axes = [0] + in_axes

    # COMMENT
    # The idea here is to construct a likelihood callable where only the first dimension is vectorized. 
    # The remaining indexes are thus None. 
    in_axes = [0] + [None]*10

    _vmap_logp_internal = vmap(
        logp,
        in_axes=in_axes,
    )

    # COMMENT
    # use ravel to flatten the output
    def vmap_logp(*args):
        return _vmap_logp_internal(*args).ravel()



    def vjp_vmap_logp(
        *inputs: list[float | ArrayLike], gz: ArrayLike
    ) -> list[ArrayLike]:

        _, vjp_fn = vjp(vmap_logp, *inputs)
        return vjp_fn(gz) if params_only else vjp_fn(gz)[1:]

    return jit(vmap_logp), jit(vjp_vmap_logp), vmap_logp


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

def make_distribution(
    rv: str | Type[RandomVariable] | RandomVariable | Callable,
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
    model_name
        The name of the model.
    choices
        A list of integers indicating the choices.
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
    if isinstance(rv, type) and issubclass(rv, RandomVariable):
        rv_instance = rv()
    elif not isinstance(rv, type) and isinstance(rv, RandomVariable):
        rv_instance = rv
    elif callable(rv) or isinstance(rv, str):
        random_variable = make_hssm_rv(
            simulator_fun=rv,
            list_params=list_params,
            lapse=lapse,
        )
        rv_instance = random_variable()
    else:
        raise ValueError(f"rv is {rv}, which is not a valid type.")

    # random_variable = make_ssm_rv(rv, list_params, lapse)
    # if isinstance(rv, str) else rv
    extra_fields = [] if extra_fields is None else extra_fields

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
    else:
        lapse_func = None

    class HSSMDistribution(pm.Distribution):
        """Wiener first-passage time (WFPT) log-likelihood for LANs."""

        # This is just a placeholder because pm.Distribution requires an rv_op
        # Might be updated in the future once

        # NOTE: rv_op is an INSTANCE of RandomVariable
        rv_op = rv_instance
        _params = list_params

        @classmethod
        def dist(cls, **kwargs):  # pylint: disable=arguments-renamed
            dist_params = [
                pt.as_tensor_variable(pm.floatX(kwargs[param])) for param in cls._params
            ]
            other_kwargs = {k: v for k, v in kwargs.items() if k not in cls._params}
            return super().dist(dist_params, **other_kwargs)

        def logp(data, *dist_params):  # pylint: disable=E0213
            """Calculate log probability of the data given the parameters.

            Parameters
            ----------
            data : array-like
                The observed data
            dist_params : tuple
                Distribution parameters

            Returns
            -------
            tensor
                Log probability
            """

            # AF-TODO: Apply clipping here

            # COMMENT
            subj_np = np.arange(20)
            subj = pt.as_tensor_variable(subj_np)

            if list_params[-1] == "p_outlier":
                p_outlier = dist_params[-1]
                dist_params = dist_params[:-1]

                if not callable(lapse_func):
                    raise ValueError(
                        "lapse_func is not defined. "
                        "Make sure lapse is properly initialized."
                    )
                lapse_logp = lapse_func(data[:, 0].eval())
                # AF-TODO potentially apply clipping here
                logp = loglik(subj, data, *dist_params, *extra_fields)
                # Ensure that non-decision time is always smaller than rt.
                # Assuming that the non-decision time parameter is always named "t".
                logp = ensure_positive_ndt(data, logp, list_params, dist_params)
                logp = pt.log(
                    (1.0 - p_outlier) * pt.exp(logp)
                    + p_outlier * pt.exp(lapse_logp)
                    + 1e-29
                )
            else:

                logp = loglik(subj, data, *dist_params, *extra_fields)
                # Ensure that non-decision time is always smaller than rt.
                # Assuming that the non-decision time parameter is always named "t".
                #logp = ensure_positive_ndt(data, logp, list_params, dist_params)

            if bounds is not None:
                logp = apply_param_bounds_to_loglik(
                    logp, list_params, *dist_params, bounds=bounds
                )

            return logp

    return HSSMDistribution
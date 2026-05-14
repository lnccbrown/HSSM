"""Use jaxonnxruntime to convert ONNX models to JAX functions."""

from typing import Callable

import jax
import numpy as np
import onnx
from jaxonnxruntime import call_onnx, config

# torch.onnx.export emits some shape arguments (e.g. for Reshape inside masked
# autoregressive flows) as Constant nodes rather than as model initializers.
# jaxonnxruntime's default strict mode rejects these as static-args during
# jax.jit. The flag below relaxes that check. This is safe for our use cases:
# the shapes in question are genuinely constant, baked at export time. Setting
# it at import time means any consumer of make_jax_func (LAN MLPs, sbi-exported
# flows, etc.) benefits without per-call configuration.
config.update("jaxort_only_allow_initializers_as_static_args", False)


def make_jax_func(onnx_model: onnx.ModelProto) -> Callable:
    """Convert an ONNX model to a JAX function using jaxonnxruntime.

    Parameters
    ----------
    onnx_model : onnx.ModelProto
        The ONNX model to be converted.

    Returns
    -------
    Callable
        A JAX function that represents the ONNX model.
    """
    model_graph = onnx_model.graph

    # Get the input name and shape from the ONNX model to create a dummy input for
    # initialization.
    input_name = model_graph.input[0].name
    input_dims = tuple(
        dim.dim_value if (dim.dim_value > 0) else 1
        for dim in model_graph.input[0].type.tensor_type.shape.dim
    )
    model_func, model_weights = call_onnx.call_onnx_model(
        onnx_model, {input_name: np.ones(input_dims)}
    )

    # Create a JAX function that takes the input and applies the ONNX model.
    run_func = jax.tree_util.Partial(model_func, model_weights)
    jax_func = lambda x: run_func({input_name: x})[0].squeeze()

    return jax_func

"""Use jaxonnxruntime to convert ONNX models to JAX functions."""

import logging
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from jaxonnxruntime import call_onnx, config

_logger = logging.getLogger("hssm")

# torch.onnx.export emits some shape arguments (e.g. for Reshape inside masked
# autoregressive flows) as Constant nodes rather than as model initializers.
# jaxonnxruntime's default strict mode rejects these as static-args during
# jax.jit. The flag below relaxes that check. This is safe for our use cases:
# the shapes in question are genuinely constant, baked at export time. Setting
# it at import time means any consumer of make_jax_func (LAN MLPs, sbi-exported
# flows, etc.) benefits without per-call configuration.
config.update("jaxort_only_allow_initializers_as_static_args", False)


def _graph_has_int64_tensors(model: onnx.ModelProto) -> bool:
    """Detect int64 tensors in an ONNX graph.

    torch.onnx.export of normalizing flows (e.g. nflows MAF) emits int64
    tensors for Reshape shape arguments, Constant node values, Cast targets,
    and similar. jaxonnxruntime silently truncates int64 to int32 unless
    `jax_enable_x64` is set, producing wrong numerical outputs (~0.5 drift
    in log-prob).
    """
    int64 = onnx.TensorProto.INT64
    for init in model.graph.initializer:
        if init.data_type == int64:
            return True
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == int64:
                return True
            if attr.type == onnx.AttributeProto.TENSORS:
                for t in attr.tensors:
                    if t.data_type == int64:
                        return True
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == int64:
                    return True
    return False


def _ensure_x64_if_needed(onnx_model: onnx.ModelProto) -> None:
    """Auto-enable jax_enable_x64 when the graph requires it.

    If the graph carries int64 tensors and x64 is off, we attempt to flip the
    JAX config flag and verify the change is effective (by checking that a
    fresh `jnp.asarray([1.0])` is float64). If the flip does not take — JAX
    has already done substantive 32-bit work in this process — raise a clear
    RuntimeError directing the user to set the flag at the top of their
    script.
    """
    if not _graph_has_int64_tensors(onnx_model):
        return
    if jax.config.read("jax_enable_x64"):
        return

    jax.config.update("jax_enable_x64", True)
    # Verify the flip is effective on fresh JAX ops.
    if jnp.asarray([1.0]).dtype != jnp.float64:
        raise RuntimeError(
            "This ONNX graph carries int64 tensors (typical for torch-exported "
            "normalizing flows), which jaxonnxruntime would silently truncate "
            "to int32 — producing wrong numerical results. HSSM attempted to "
            "auto-enable `jax_enable_x64`, but JAX has already been used in "
            "32-bit mode and the flip did not take. Fix: add\n"
            "    import jax\n"
            "    jax.config.update('jax_enable_x64', True)\n"
            "at the very top of your script, before any other JAX import."
        )
    _logger.warning(
        "HSSM auto-enabled `jax_enable_x64` because the loaded ONNX graph "
        "carries int64 tensors that JAX would otherwise silently truncate. "
        "To silence this warning, set the flag yourself at the top of your "
        "script: `jax.config.update('jax_enable_x64', True)`."
    )


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
    _ensure_x64_if_needed(onnx_model)

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

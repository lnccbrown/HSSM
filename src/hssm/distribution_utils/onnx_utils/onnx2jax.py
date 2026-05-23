"""Use jaxonnxruntime to convert ONNX models to JAX functions.

This module assumes the **single-trial export contract**: ONNX graphs reaching
``make_jax_func`` are exported with a fully concrete input shape representing
*one* per-trial input vector (e.g. ``(n_params + n_data_cols,)``). HSSM batches
across trials at a layer above this, via ``jax.vmap`` in
``make_jax_logp_funcs_from_onnx`` (see ``hssm.distribution_utils.onnx``).

If a graph arrives with a symbolic / dynamic dimension, ``make_jax_func``
raises a ``ValueError`` rather than trying to make it work: jaxonnxruntime
traces against the construction-time dummy shape and bakes the resulting
shapes into the returned closure, so calling that closure with a different
shape silently produces wrong outputs for any graph that carries a
batch-dependent intermediate (e.g. a ``torch.zeros(x.shape[0])`` log-det
accumulator, or a ``Reshape`` whose ``-1`` resolves against the dynamic dim).
LANs and the sbi NRE/NLE exporters in ``LANfactory.onnx`` already follow this
contract; this guard prevents accidental violations from a future contributor.

On precision: pytensor's JAX dispatch
(``pytensor/link/jax/dispatch/basic.py``) sets ``jax_enable_x64`` from
``pytensor.config.floatX`` at import time. With HSSM's default
``floatX="float64"`` x64 is already on by the time this module loads;
under ``hssm.set_floatX("float32")`` x64 is off. The previous version of
this module also tried to flip ``jax_enable_x64`` at first call; that has
been removed (it duplicated pytensor's contract, mutated global state, and
hard-failed if JAX had already warmed up). Instead we pre-cast int64
tensors / Cast targets to int32 in the graph at load time -- lossless for
the index/shape values torch.onnx.export produces, and removes the silent
truncation that ``jax_enable_x64=False`` would otherwise apply.
"""

import logging
from typing import Callable

import jax
import numpy as np
import onnx
from jaxonnxruntime import call_onnx, config

# torch.onnx.export emits some shape arguments (e.g. for Reshape) as Constant
# nodes rather than as model initializers. jaxonnxruntime's default strict
# mode rejects these as static args during jax.jit. The flag below relaxes
# that check. This is safe for our use cases: those shapes are constant by
# construction (baked at export time).
config.update("jaxort_only_allow_initializers_as_static_args", False)

_logger = logging.getLogger("hssm")


def _recast_int64_to_int32(model: onnx.ModelProto) -> int:
    """Rewrite int64 tensors and Cast targets in the graph to int32, in place.

    torch.onnx.export carries int64 metadata (Reshape shape args, Constant
    tensors, Cast targets) whose values are indices/shapes that always fit
    losslessly in int32. With ``jax_enable_x64=False`` JAX truncates int64
    to int32 implicitly and emits a UserWarning per access. Pre-casting at
    load time:

    * is bit-identical for valid index values (twos-complement of small
      non-negative integers is preserved when the upper 32 bits are dropped),
    * silences the JAX UserWarning,
    * removes any dependency on global JAX state.

    Returns
    -------
    int
        Number of int64 sites rewritten (0 if none).
    """
    int64 = onnx.TensorProto.INT64
    int32 = onnx.TensorProto.INT32
    n_rewritten = 0

    def _convert(tensor: onnx.TensorProto) -> None:
        nonlocal n_rewritten
        if tensor.data_type == int64:
            arr = onnx.numpy_helper.to_array(tensor).astype(np.int32)
            new = onnx.numpy_helper.from_array(arr, tensor.name)
            tensor.CopyFrom(new)
            n_rewritten += 1

    for initializer in model.graph.initializer:
        _convert(initializer)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                _convert(attr.t)
            elif attr.type == onnx.AttributeProto.TENSORS:
                for t in attr.tensors:
                    _convert(t)
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == int64:
                    attr.i = int32
                    n_rewritten += 1
    return n_rewritten


def _check_single_trial_input_shape(model: onnx.ModelProto) -> None:
    """Raise if any input dimension is symbolic / dynamic.

    HSSM's ONNX-likelihood path is built around single-trial inputs that get
    vmapped over trials at a layer above this. jaxonnxruntime, however,
    traces the graph against the construction-time dummy shape and bakes
    those shapes into the returned closure -- so a graph with dynamic dims
    called at a different shape later will produce wrong-but-non-erroring
    outputs (the trace re-uses the dummy's broadcast shapes for
    batch-dependent intermediates).
    """
    bad: list[str] = []
    for inp in model.graph.input:
        for i, dim in enumerate(inp.type.tensor_type.shape.dim):
            if dim.dim_value <= 0:
                label = dim.dim_param or f"axis {i}"
                bad.append(f"{inp.name}[{label}]")
    if bad:
        raise ValueError(
            "ONNX model has dynamic (symbolic) input dimensions: "
            f"{', '.join(bad)}. HSSM uses single-trial input shapes and "
            "vmaps over trials at a layer above this conversion -- "
            "re-export the model with a concrete per-trial input shape "
            "(omit `dynamic_axes` in `torch.onnx.export`, or pass a single "
            "rank-1 dummy as LANfactory.onnx.transform_sbi_to_onnx does). "
            "Dynamic dims here would cause jaxonnxruntime to silently "
            "produce wrong outputs for graphs with batch-dependent "
            "intermediates (e.g. log-det accumulators)."
        )


def make_jax_func(onnx_model: onnx.ModelProto) -> Callable:
    """Convert an ONNX model to a JAX function using jaxonnxruntime.

    The model must have a fully concrete input shape -- see the module
    docstring for the single-trial-input + vmap contract.

    Parameters
    ----------
    onnx_model : onnx.ModelProto
        The ONNX model to be converted. Will be mutated in place to recast
        int64 tensors to int32 (lossless for index/shape values produced by
        torch.onnx.export).

    Returns
    -------
    Callable
        A JAX function ``f(x)`` that runs the ONNX graph on ``x``.

    Raises
    ------
    ValueError
        If the ONNX graph has any dynamic / symbolic input dimension.
    """
    _check_single_trial_input_shape(onnx_model)
    _recast_int64_to_int32(onnx_model)

    model_graph = onnx_model.graph
    input_name = model_graph.input[0].name
    input_dims = tuple(
        dim.dim_value for dim in model_graph.input[0].type.tensor_type.shape.dim
    )
    model_func, model_weights = call_onnx.call_onnx_model(
        onnx_model, {input_name: np.ones(input_dims)}
    )

    run_func = jax.tree_util.Partial(model_func, model_weights)
    jax_func = lambda x: run_func({input_name: x})[0].squeeze()

    return jax_func

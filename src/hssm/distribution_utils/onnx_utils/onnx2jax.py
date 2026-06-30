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
LANs and the sbi/bayesflow exporters in ``LANfactory.onnx`` already follow
this contract; this guard prevents accidental violations from a future
contributor.

On precision: HSSM's ONNX likelihoods rely on JAX x64. pytensor's JAX dispatch
enables ``jax_enable_x64`` from ``pytensor.config.floatX`` at import time, so
with HSSM's default ``floatX="float64"`` x64 is on by the time any ONNX graph
is loaded, and the int64 shape/index tensors ``torch.onnx.export`` emits are
preserved exactly. Under ``hssm.set_floatX("float32")`` x64 is off and JAX
truncates those int64 values; flow-based exports (sbi nflows, bayesflow
CouplingFlow) can carry an ``INT64_MAX`` open-ended-slice sentinel that
truncates to ``-1`` and corrupts the graph, so float32 is not currently
supported for such ONNX likelihoods. (An earlier revision rewrote int64
tensors to int32 at load time, but that non-saturating cast wrapped the
``INT64_MAX`` sentinel to ``-1`` and corrupted every flow export; it was
removed in favour of relying on x64.)
"""

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
        The ONNX model to be converted. Must have a concrete per-trial input
        shape (no symbolic / dynamic dimensions).

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

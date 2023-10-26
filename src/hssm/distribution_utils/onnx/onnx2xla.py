# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""An ONNX to XLA compiler by JAX-tracing a Numpy-backed ONNX interpreter.

Tips for extending this file to add more Ops:

First, Look up for ONNX Ops here:
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceEmpty

If there is an equivalent op in JAX, just extend the mapping below and add a
lambda function with the same number of inputs as the jax op and wrap the result
in a list.

For example, for ONNX Op Tanh, there is a jnp.tanh function. So we just create
a lambda function like this:

    lambda x: [jnp.tanh(x)]

If there is no equivalent in JAX, then we need to write a function. The `onnx_gemm`
function is an example of such a function. The number of argument is the same as
the number of inputs specified in the Opset document above. If there are any optional
inputs, also add these inputs as parameters with default values.
"""

import jax.numpy as jnp
import numpy as np
import onnx
from jax import lax
from onnx import numpy_helper


def _asarray(proto):
    return numpy_helper.to_array(proto).reshape(tuple(proto.dims))


# pylint: disable=E1101
attr_types = dict(onnx.AttributeProto.AttributeType.items())
attribute_handlers = {
    attr_types["FLOAT"]: lambda a: a.f,
    attr_types["INT"]: lambda a: a.i,
    attr_types["STRING"]: lambda a: a.s,
    attr_types["TENSOR"]: lambda a: _asarray(a.t),
    attr_types["FLOATS"]: lambda a: a.floats,
    attr_types["INTS"]: lambda a: a.ints,
    attr_types["STRINGS"]: lambda a: a.strings,
    attr_types["TENSORS"]: lambda a: [_asarray(x) for x in a.tensors],
}


def onnx_maxpool(x, kernel_shape, pads=None, strides=None):
    """Numpy-backed implementation of ONNX MaxPool op."""
    prefix = (1,) * (x.ndim - len(kernel_shape))
    dims = prefix + tuple(kernel_shape)
    pads = tuple(pads) if pads else [0] * len(kernel_shape)
    strides = (prefix + tuple(strides)) if strides else [1] * len(kernel_shape)
    return [lax.reduce_window(x, -jnp.inf, lax.max, dims, strides, "VALID")]


def onnx_conv(
    x,
    w,
    b=0,
    group=1,
    kernel_shape=None,
    pads=None,
    strides=None,
    dilations=None,
    auto_pad=None,
):
    """Numpy-backed implementation of ONNX Conv op."""
    assert group == 1
    kernel_shape = kernel_shape or w.shape
    strides = strides or [1] * (w.ndim - 2)
    if auto_pad:
        auto_pad = "SAME" if auto_pad.startswith(b"SAME") else "VALID"
        pads = lax.padtype_to_pads(x.shape[2:], w.shape[2:], strides, auto_pad)
    else:
        pads = pads or [0] * (w.ndim - 2)
    lhs_dilation = [1] * (w.ndim - 2)
    rhs_dilation = dilations or [1] * (w.ndim - 2)
    return [
        lax.conv_with_general_padding(x, w, strides, pads, lhs_dilation, rhs_dilation)
        + b
    ]


def onnx_add(a, b, axis=None, broadcast=True):
    """Numpy-backed implementation of ONNX Add op."""
    if broadcast:
        axis = (a.dim - b.ndim) if axis is None else axis % a.ndim
        assert a.shape[axis:][: b.ndim] == b.shape
        b_shape = np.ones(a.ndim, dtype="int64")
        b_shape[axis : axis + b.ndim] = b.shape
        b = jnp.reshape(b, b_shape)
    return [a + b]


# Added by HSSM Developers
def onnx_gemm(
    a, b, c=0.0, alpha=1.0, beta=1.0, transA=0, transB=0
):  # pylint: disable=C0103
    """Numpy-backed implementatio of Onnx Gemm op."""
    a = jnp.transpose(a) if transA else a
    b = jnp.transpose(b) if transB else b
    # jax.debug.print("a: {}", a.dtype)
    # jax.debug.print("b: {}", b.dtype)

    return [alpha * jnp.matmul(a, b) + beta * c]


onnx_ops = {
    "Add": onnx_add,
    "Constant": lambda value: [value],
    "Conv": onnx_conv,
    "MatMul": lambda x, y: [jnp.matmul(x, y)],
    "MaxPool": onnx_maxpool,
    "Relu": lambda x: [jnp.maximum(x, 0)],
    "Reshape": lambda x, shape: [jnp.reshape(x, shape)],
    # Added by HSSM developers
    "Tanh": lambda x: [jnp.tanh(x)],
    "Gemm": onnx_gemm,
}


def interpret_onnx(graph, *args):
    """Transform model in onnx to pytensor.

    Parameters
    ----------
    graph
        The computation graph.
    args
        Inputs to the graph.

    Returns
    -------
        The result of the computation.
    """
    vals = dict(
        {n.name: a for n, a in zip(graph.input, args)},
        **{n.name: _asarray(n) for n in graph.initializer},
    )
    for node in graph.node:
        args = (vals[name] for name in node.input)
        attrs = {a.name: attribute_handlers[a.type](a) for a in node.attribute}
        outputs = onnx_ops[node.op_type](*args, **attrs)
        for name, output in zip(node.output, outputs):
            vals[name] = output
    return [vals[n.name] for n in graph.output]

"""
An ONNX to aesara compiler.
Tips for extending this file to add more Ops:
If there is no equivalent in aesara, then we need to write a function. The `aesara_gemm`
function is an example of such a function. The number of argument is the same as
the number of inputs specified in the Opset document above. If there are any optional
inputs, also add these inputs as parameters with default values.
"""

import aesara.tensor as at

from .onnx2xla import _asarray, attribute_handlers


def aesara_gemm(
    a, b, c=0.0, alpha=1.0, beta=1.0, transA=0, transB=0
):  # pylint: disable=C0103
    """Numpy-backed implementatio of Onnx
    General Matrix Multiply (GeMM) op."""
    a = at.transpose(a) if transA else a
    b = at.transpose(b) if transB else b

    return [alpha * at.dot(a, b) + beta * c]


aes_onnx_ops = {
    "Add": at.add,
    "Constant": lambda value: [value],
    "Conv": at.nnet.conv,
    "MatMul": lambda x, y: [at.dot(x, y)],
    "Relu": lambda x: [at.math.max(x, 0)],
    "Reshape": lambda x, shape: [at.reshape(x, shape)],
    "Tanh": lambda x: [at.tanh(x)],
    "Gemm": aesara_gemm,
}


def aes_interpret_onnx(graph, *args):
    """
    This function transforms model in onnx to aesara
    """
    vals = dict(
        {n.name: a for n, a in zip(graph.input, args)},
        **{n.name: _asarray(n) for n in graph.initializer}
    )
    for node in graph.node:
        args = (vals[name] for name in node.input)
        attrs = {a.name: attribute_handlers[a.type](a) for a in node.attribute}
        outputs = aes_onnx_ops[node.op_type](*args, **attrs)
        for name, output in zip(node.output, outputs):
            vals[name] = output
    return [vals[n.name] for n in graph.output]

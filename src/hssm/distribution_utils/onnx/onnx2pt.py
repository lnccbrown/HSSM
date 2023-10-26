"""An ONNX to pytensor compiler.

Tips for extending this file to add more Ops:
If there is no equivalent in pytensor, then we need to write a function. The
`pytensor_gemm` function is an example of such a function. The number of argument is
the same as the number of inputs specified in the Opset document above. If there are
any optional inputs, also add these inputs as parameters with default values.
"""

import pytensor.tensor as pt

from .onnx2xla import _asarray, attribute_handlers


def pytensor_gemm(
    a, b, c=0.0, alpha=1.0, beta=1.0, transA=0, transB=0
):  # pylint: disable=C0103
    """Perform the GEMM op.

    Numpy-backed implementatio, of ONNX General Matrix Multiply (GeMM) op.
    """
    a = pt.transpose(a) if transA else a
    b = pt.transpose(b) if transB else b

    return [alpha * pt.dot(a, b) + beta * c]


pt_onnx_ops = {
    "Add": pt.add,
    "Constant": lambda value: [value],
    "MatMul": lambda x, y: [pt.dot(x, y)],
    "Relu": lambda x: [pt.math.max(x, 0)],
    "Reshape": lambda x, shape: [pt.reshape(x, shape)],
    "Tanh": lambda x: [pt.tanh(x)],
    "Gemm": pytensor_gemm,
}


def pt_interpret_onnx(graph, *args):
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
        outputs = pt_onnx_ops[node.op_type](*args, **attrs)
        for name, output in zip(node.output, outputs):
            vals[name] = output
    return [vals[n.name] for n in graph.output]

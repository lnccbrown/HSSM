"""Helper functions for creating blackbox ops."""

from typing import Callable

import numpy as np
import pytensor.tensor as pt
from pytensor.graph import Apply, Op


def make_blackbox_op(logp: Callable) -> Op:
    """Wrap an arbitrary function in a pytensor Op.

    Parameters
    ----------
    logp
        A python function that represents the log-likelihood function. The function
        needs to have signature of logp(data, *dist_params) where `data` is a
        two-column numpy array and `dist_params`represents all parameters passed to the
        function.

    Returns
    -------
    Op
        An pytensor op that wraps the log-likelihood function.
    """

    class BlackBoxOp(Op):  # pylint: disable=W0223
        """Wraps an arbitrary function in a pytensor Op."""

        def make_node(self, data, *dist_params):
            """Take the inputs to the Op and puts them in a list.

            Also specifies the output types in a list, then feed them to the Apply node.

            Parameters
            ----------
            data
                A two-column numpy array with response time and response.
            dist_params
                A list of parameters used in the likelihood computation. The parameters
                can be both scalars and arrays.
            """
            self.params_only = data is None
            inputs = [pt.as_tensor_variable(dist_param) for dist_param in dist_params]

            if not self.params_only:
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

    blackbox_op: Op = BlackBoxOp()
    return blackbox_op

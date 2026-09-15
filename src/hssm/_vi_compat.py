"""Scoped PyMC compatibility patches enabling ``pm.fit(backend="jax")``.

Three upstream issues block JAX-compiled variational inference (see
lnccbrown/HSSM#1056 for the diagnosis of the first two and #1328 for the
third):

1. VI approximation parameters (e.g. meanfield ``mu``/``rho``) are created
   without static shapes, producing a runtime-shape ``Alloc`` that the JAX
   backend cannot trace (``TypeError: Shapes must be 1D sequences of
   concrete values``). Upstream: pymc-devs/pymc#8359.
2. A JAX-compiled step function writes raw ``jax.Array`` objects into the
   shared-variable storage, which breaks any later default-backend (numba)
   compiled function — notably ``approx.sample()``. Upstream:
   pymc-devs/pymc#8360.
3. bambi 0.20 stores the response and the ``__obs__`` dim length as shared
   variables, and derives shapes from them: it broadcasts every response
   parameter with ``pt.broadcast_to(value, (model.dim_lengths["__obs__"],))``,
   and HSSM's missing-data logp slices on ``n_missing = sum(rt == -999)``
   computed from the (now shared) data. PyTensor's JAX linker passes shared
   variables to ``jax.jit`` as traced arguments, so those shapes are tracers
   and tracing fails with ``TypeError: Shapes must be 1D sequences of
   concrete values``. PyMC's own JAX samplers sidestep this by replacing
   every shared variable with its constant value before jaxifying
   (``pymc.sampling.jax._replace_shared_variables``); ``pm.fit`` has no such
   step, but accepts ``more_replacements`` to do the same. See
   lnccbrown/HSSM#1328.

The first two helpers are self-disabling: they detect an already-fixed PyMC
and do nothing, so they are safe to keep until the pinned PyMC includes the
upstream fixes. The third mirrors what PyMC does for its JAX samplers and
stays as long as ``pm.fit(backend="jax")`` does not do it itself.
"""

from contextlib import ExitStack, contextmanager
from functools import wraps

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Variable


def _with_static_shapes(orig_create_shared_params):
    """Wrap ``create_shared_params`` to add static shapes where missing.

    Post-processes the returned dict: any shared variable whose type has an
    unknown dimension is re-created from its value with the concrete shape.
    If PyMC already produces statically-shaped parameters (i.e. the upstream
    fix has landed), this is a no-op.
    """

    @wraps(orig_create_shared_params)
    def wrapper(self, *args, **kwargs):
        params = orig_create_shared_params(self, *args, **kwargs)
        out = {}
        for name, var in params.items():
            if None in var.type.shape:
                value = var.get_value(borrow=False)
                out[name] = pytensor.shared(value, name, shape=value.shape)
            else:
                out[name] = var
        return out

    return wrapper


@contextmanager
def static_shape_vi_params():
    """Give PyMC VI approximation parameters static shapes (scoped patch).

    Applied only around the ``pm.fit`` call when the user requests
    ``backend="jax"``; the default and C backends never see it. Not
    thread-safe (temporarily swaps a method on the PyMC group classes), which
    matches how ``HSSM.vi`` is used.

    Known ceiling: PyMC models with exactly one free parameter dimension
    (``ddim == 1``) additionally need PyMC's scan inner graphs to know the
    static shape (part of the upstream fix that cannot be applied from the
    outside) and may still fail to compile; ``backend="c"`` remains the
    fallback there. HSSM models virtually always have ``ddim > 1``.
    """
    from pymc.variational import approximations

    group_classes = [
        approximations.MeanFieldGroup,
        approximations.FullRankGroup,
        approximations.EmpiricalGroup,
    ]

    with ExitStack() as stack:
        for cls in group_classes:
            orig = cls.create_shared_params
            cls.create_shared_params = _with_static_shapes(orig)
            stack.callback(setattr, cls, "create_shared_params", orig)
        yield


def coerce_approx_params_to_numpy(approx) -> None:
    """Convert an approximation's parameter storage back to NumPy arrays.

    After a JAX-compiled fit, the shared variables hold ``jax.Array``
    objects; later default-backend compiled functions (``approx.sample``)
    cannot handle them. ``np.asarray`` is a no-op for NumPy arrays, so this
    is always safe to call regardless of the fit backend or PyMC version.
    """
    for param in approx.params:
        param.container.storage[0] = np.asarray(param.container.storage[0])


def freeze_shared_data(model: pm.Model) -> dict[Variable, Variable]:
    """Build replacements pinning a model's data and dim lengths to constants.

    Returns ``{shared: constant}`` for every ``pm.Data`` variable and every
    shared dim length of ``model``, for ``pm.fit(more_replacements=...)``.
    With these in place, any shape the graph derives from the data or the
    observation count is concrete when the JAX linker traces it. Random
    generators and the VI approximation's own parameters are not touched.

    The values are those at call time, matching what
    ``pymc.sampling.jax`` does for ``pm.sample``: a later ``pm.set_data``
    takes effect on the next ``vi()`` call, which rebuilds the replacements.
    """
    shared_vars: list[SharedVariable] = [
        var for var in model.data_vars if isinstance(var, SharedVariable)
    ]
    shared_vars += [
        var for var in model.dim_lengths.values() if isinstance(var, SharedVariable)
    ]
    return {
        var: pt.constant(var.get_value(borrow=True), name=var.name)
        for var in shared_vars
    }

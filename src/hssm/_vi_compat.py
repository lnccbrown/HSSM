"""Scoped PyMC compatibility patches enabling ``pm.fit(backend="jax")``.

Two upstream PyMC bugs block JAX-compiled variational inference for every
model (see lnccbrown/HSSM#1056 for the full diagnosis):

1. VI approximation parameters (e.g. meanfield ``mu``/``rho``) are created
   without static shapes, producing a runtime-shape ``Alloc`` that the JAX
   backend cannot trace (``TypeError: Shapes must be 1D sequences of
   concrete values``).
2. A JAX-compiled step function writes raw ``jax.Array`` objects into the
   shared-variable storage, which breaks any later default-backend (numba)
   compiled function — notably ``approx.sample()``.

Both helpers here are self-disabling: they detect an already-fixed PyMC and
do nothing, so they are safe to keep until the pinned PyMC includes the
upstream fixes, at which point this module can be deleted.
"""

from contextlib import ExitStack, contextmanager
from functools import wraps

import numpy as np
import pytensor


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

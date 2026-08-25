"""Integration tests for `HSSM.vi()` across likelihood/backend/method combos.

This file used to run a 6-row grid against each of 3 model shapes (18 tests).
Two of those rows were `blackbox`, which `vi()` rejects before it looks at
either the method or the model, so they were the same assertion repeated six
times; they are now their own fast (non-slow) tests. The remaining 12 passing
rows are replaced by a pairwise covering array over backend x method x model
shape.

Note that most of this file's runtime is the first pytensor C compile of the
`approx_differentiable` logp (~25s), not the grid itself -- whichever pytensor
row runs first pays it, so trimming rows does not remove it.
"""

from copy import deepcopy

import pymc as pm
import pytest
import xarray as xr

import hssm

from tests.integration.regression_priors import A_REG, V_REG

hssm.set_floatX("float32", update_jax=True)

# AF-TODO: Include more tests that use different link functions!

# NOTE: `analytical` is deliberately absent from the grid below. `vi()` only
# warns for analytical likelihoods (it does not raise), but the gradients are
# still broken, so these rows cannot yet be asserted either way:
#     ("analytical", "pytensor", "advi")
#     ("analytical", "pytensor", "fullrank_advi")

# Model shape -> (data fixture name, extra HSSM kwargs).
MODEL_SHAPES = {
    "simple": ("data_ddm", {}),
    "reg_v": ("data_ddm_reg", {"v": V_REG}),
    "reg_va": ("data_ddm_reg_va", {"v": V_REG, "a": A_REG}),
}

# Pairwise covering array over three factors:
#   B = backend (all `approx_differentiable`): pytensor / jax
#   S = method:                                advi / fullrank_advi
#   M = model shape:                           simple / reg_v / reg_va
#
# 6 rows cover every (B, S), (B, M) and (S, M) pair.
PARAMETER_NAMES = "backend,method,shape"
COVERING_ARRAY = [
    ("pytensor", "advi", "simple"),
    ("pytensor", "fullrank_advi", "reg_v"),
    ("pytensor", "advi", "reg_va"),
    ("jax", "fullrank_advi", "simple"),
    ("jax", "advi", "reg_v"),
    ("jax", "fullrank_advi", "reg_va"),
]

VI_METHODS = ["advi", "fullrank_advi"]


def build_model(request, shape, loglik_kind, backend):
    """Build an HSSM model of the given shape, pulling its data fixture lazily."""
    data_fixture, params = MODEL_SHAPES[shape]
    return hssm.HSSM(
        request.getfixturevalue(data_fixture),
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        # HSSM mutates the param dicts it is handed, so hand it a fresh copy.
        **deepcopy(params),
    )


def run_vi(model, method):
    """Fit the model with VI and check the approximation and idata are built."""
    model.vi(method, niter=100, progressbar=False)
    assert isinstance(model.vi_idata, xr.DataTree)
    assert isinstance(model.vi_approx, pm.variational.Approximation)


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, COVERING_ARRAY)
def test_vi_matrix(request, backend, method, shape):
    """Run VI on each covered (backend, method, model shape) cell."""
    model = build_model(request, shape, "approx_differentiable", backend)
    run_vi(model, method)


@pytest.mark.parametrize("method", VI_METHODS)
def test_vi_rejects_blackbox(data_ddm, method):
    """VI is rejected for blackbox likelihoods, which have no gradients.

    `vi()` raises before it inspects the method or the model, so one cheap model
    shape covers this and it does not need to be marked slow.
    """
    model = hssm.HSSM(
        data_ddm, loglik_kind="blackbox", model_config={"backend": "pytensor"}
    )
    with pytest.raises(ValueError):
        run_vi(model, method)


@pytest.mark.slow
@pytest.mark.parametrize("method", VI_METHODS)
@pytest.mark.parametrize("backend_arg", ["jax", None])
def test_vi_jax_compile_backend(data_ddm, method, backend_arg):
    """End-to-end regression test for #1056: VI compiled through the JAX linker.

    Exercises the LANLogpVJPOp jax_funcify registration plus the scoped
    PyMC compatibility shims in `hssm._vi_compat` (static parameter shapes;
    numpy coercion before `approx.sample`). Runs both with an explicit
    ``backend="jax"`` and with ``backend=None``, where the jax backend is
    inferred from the angle model's jax model config (#1060) — the inferred
    route must hit the same shims.
    """
    model = hssm.HSSM(data_ddm, model="angle", p_outlier=0.0)
    model.vi(method=method, niter=100, draws=10, backend=backend_arg, progressbar=False)
    assert isinstance(model.vi_idata, xr.DataTree)
    assert isinstance(model.vi_approx, pm.variational.Approximation)


@pytest.mark.slow
def test_vi_c_compile_backend(data_ddm):
    """VI with the explicit C compile backend (the documented #1056 workaround).

    The parametrized tests above exercise only pymc's default compile mode;
    this pins the `backend="c"` path, which runs the LAN Ops' perform/VJP
    through the C VM.
    """
    model = hssm.HSSM(data_ddm, model="angle", p_outlier=0.0)
    model.vi(method="advi", niter=100, backend="c", progressbar=False)
    assert isinstance(model.vi_idata, xr.DataTree)
    assert isinstance(model.vi_approx, pm.variational.Approximation)

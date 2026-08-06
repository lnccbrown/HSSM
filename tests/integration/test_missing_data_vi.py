"""Integration tests for VI on missing-data and deadline models.

Merged from `test_missing_data_vi.py` and `test_missing_data_and_deadline_vi.py`
for the same reason as the MCMC pair: `deadline` selects CPN vs OPN within one
mechanism (`missing_data_mixin.py:88-95`) rather than being a separate one, so
it becomes a `mode` column instead of a second copy of the grid. The former file
also carried a commented-out copy of the latter's deadline tests, which the
merge resolves.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pymc as pm
import pytest
import xarray as xr

import hssm
from hssm.utils import _rearrange_data

hssm.set_floatX("float32", update_jax=True)

FIXTURES = Path(__file__).parent.parent / "fixtures"

# Missing-data mode -> ONNX network. "cpn" is missing_data only; "opn" adds a
# deadline column and censors RTs past the deadline.
NETWORKS = {
    "cpn": FIXTURES / "ddm_cpn.onnx",
    "opn": FIXTURES / "ddm_opn.onnx",
}

V_REG = dict(
    formula="v ~ 1 + x + y",
    prior={
        "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
        "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
        "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
    },
)

# Model shape -> (data fixture name, extra HSSM kwargs).
MODEL_SHAPES = {
    "simple": ("data_ddm", {}),
    "reg": ("data_ddm_reg", {"v": V_REG}),
}

VI_METHODS = ["advi", "fullrank_advi"]


def prepare_data(base, mode):
    """Return a copy of `base` with missing RTs injected for the given mode."""
    rng = np.random.default_rng(seed=42)
    data = base.copy()
    if mode == "opn":
        data["deadline"] = data["rt"] + rng.normal(0, 0.1, data.shape[0])
        missing_indices = data["rt"] > data["deadline"]
    else:
        missing_indices = rng.choice(data.shape[0], 50, replace=False)
    data.iloc[missing_indices, 0] = -999.0
    return _rearrange_data(data)


def build_model(request, shape, mode, loglik_kind, backend):
    """Build a missing-data HSSM model of the given shape and mode."""
    data_fixture, params = MODEL_SHAPES[shape]
    return hssm.HSSM(
        prepare_data(request.getfixturevalue(data_fixture), mode),
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        missing_data=True,
        deadline=(mode == "opn"),
        loglik_missing_data=NETWORKS[mode],
        # HSSM mutates the param dicts it is handed, so hand it a fresh copy.
        **deepcopy(params),
    )


def run_vi(model, method):
    """Fit the model with VI and check the approximation and idata are built."""
    model.vi(method, niter=100, progressbar=False)
    assert isinstance(model.vi_idata, xr.DataTree)
    assert isinstance(model.vi_approx, pm.variational.Approximation)


# Pairwise covering array over four two-level factors:
#   B = backend (all `approx_differentiable`): pytensor / jax
#   S = method:                                advi / fullrank_advi
#   M = model shape:                           simple / reg
#   D = missing-data mode:                     cpn / opn
#
# 5 rows is the proven minimum for pairwise coverage of four binary factors
# (4 runs is impossible -- it would require a fully orthogonal OA(4,4,2,2)).
PARAMETER_NAMES = "backend,method,shape,mode"
COVERING_ARRAY = [
    ("pytensor", "advi", "simple", "cpn"),
    ("pytensor", "advi", "reg", "opn"),
    ("pytensor", "fullrank_advi", "simple", "opn"),
    ("jax", "advi", "simple", "opn"),
    ("jax", "fullrank_advi", "reg", "cpn"),
]


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, COVERING_ARRAY)
def test_missing_data_vi_matrix(request, backend, method, shape, mode):
    """Run VI on each covered (backend, method, shape, mode) cell."""
    model = build_model(request, shape, mode, "approx_differentiable", backend)
    run_vi(model, method)


@pytest.mark.parametrize("method", VI_METHODS)
def test_vi_rejects_blackbox(request, method):
    """VI is rejected for blackbox likelihoods, which have no gradients.

    `vi()` raises before it inspects the method or the model, so one cheap cell
    covers this and it does not need to be marked slow.
    """
    model = build_model(request, "simple", "cpn", "blackbox", "pytensor")
    with pytest.raises(ValueError):
        run_vi(model, method)

import pytest

import hssm
import pymc as pm
import xarray as xr

hssm.set_floatX("float32", update_jax=True)

# AF-TODO: Include more tests that use different link functions!


PARAMETER_NAMES = "loglik_kind, backend, method, expected"
PARAMETER_GRID = [
    # ("analytical", "pytensor", "advi", True),  # TODO: Gradients still broken
    # ("analytical", "pytensor", "fullrank_advi", True), # TODO Gradients still broken
    ("blackbox", "pytensor", "advi", ValueError),  # Defaults should work
    ("blackbox", "pytensor", "fullrank_advi", ValueError),
    ("approx_differentiable", "pytensor", "advi", True),  # Defaults should work
    ("approx_differentiable", "pytensor", "fullrank_advi", True),
    ("approx_differentiable", "jax", "advi", True),  # Defaults should work
    ("approx_differentiable", "jax", "fullrank_advi", True),
]


def run_vi(model, method, expected):
    if expected == True:
        model.vi(method, niter=100, progressbar=False)
        assert isinstance(model.vi_idata, xr.DataTree)
        assert isinstance(model.vi_approx, pm.variational.Approximation)
    else:
        with pytest.raises(expected):
            model.vi(method, niter=100, progressbar=False)


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_simple_models(data_ddm, loglik_kind, backend, method, expected):
    print("PYMC VERSION: ")
    print(pm.__version__)
    print("TEST INPUTS WERE: ")
    print("REPORTING FROM SIMPLE MODELS TEST")
    print(loglik_kind, backend, method, expected)

    model = hssm.HSSM(
        data_ddm, loglik_kind=loglik_kind, model_config={"backend": backend}
    )

    run_vi(model, method, expected)


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_reg_models(data_ddm_reg, loglik_kind, backend, method, expected):
    print("PYMC VERSION: ")
    print(pm.__version__)
    print("TEST INPUTS WERE: ")
    print("REPORTING FROM REG MODELS TEST")
    print(loglik_kind, backend, method, expected)

    param_reg = dict(
        formula="v ~ 1 + x + y",
        prior={
            "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
            "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
        },
    )
    model = hssm.HSSM(
        data_ddm_reg,
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        v=param_reg,
    )
    run_vi(model, method, expected)


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_reg_models_v_a(data_ddm_reg_va, loglik_kind, backend, method, expected):
    print("PYMC VERSION: ")
    print(pm.__version__)
    print("TEST INPUTS WERE: ")
    print("REPORTING FROM REG MODELS V_A TEST")
    print(loglik_kind, backend, method, expected)
    param_reg_v = dict(
        formula="v ~ 1 + x + y",
        prior={
            "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
            "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
        },
    )
    param_reg_a = dict(
        formula="a ~ 1 + m + n",
        prior={
            "Intercept": {
                "name": "Normal",
                "mu": 1.0,
                "sigma": 0.5,
            },
            "m": {"name": "Uniform", "lower": 0.0, "upper": 0.2},
            "n": {"name": "Uniform", "lower": 0.0, "upper": 0.2},
        },
        link="identity",
    )

    model = hssm.HSSM(
        data_ddm_reg_va,
        loglik_kind=loglik_kind,
        model_config={"backend": backend},
        v=param_reg_v,
        a=param_reg_a,
    )
    run_vi(model, method, expected)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["advi", "fullrank_advi"])
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

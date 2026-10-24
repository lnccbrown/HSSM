"""RLSSM with the analytical DDM as decision process (`decision_process_loglik_kind`).

Runs in float64 (the module fixture), unlike the float32-pinned test_rlssm.py; the
float32 test switches precision locally.
"""

from contextlib import contextmanager
from copy import deepcopy
from unittest import mock

import cloudpickle
import jax
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from ssms import rl

import hssm
from hssm.likelihoods.analytical import ddm_bounds, logp_ddm
from hssm.rl import registry

TRUTH = {"rl_alpha": 0.3, "scaler": 3.0, "a": 1.2, "z": 0.5, "t": 0.3}


@contextmanager
def _precision(dtype):
    """Switch PyTensor and JAX to ``dtype``, then restore the previous settings."""
    prev_floatx = pytensor.config.floatX
    prev_jax_x64 = jax.config.read("jax_enable_x64")
    hssm.set_floatX(dtype)
    try:
        yield
    finally:
        pytensor.config.floatX = prev_floatx
        jax.config.update("jax_enable_x64", prev_jax_x64)


@pytest.fixture(scope="module", autouse=True)
def _float64_precision_guard():
    """Pin float64 for this module and restore prior precision settings."""
    with _precision("float64"):
        yield


@pytest.fixture(autouse=True)
def isolated_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate global registries and start every test with an empty logp cache."""
    monkeypatch.setattr(registry, "_SSM_REGISTRY", deepcopy(registry._SSM_REGISTRY))
    monkeypatch.setattr(registry, "_RLSSM_REGISTRY", deepcopy(registry._RLSSM_REGISTRY))
    monkeypatch.setattr(registry, "_SSM_LOGP_CACHE", {})


@pytest.fixture(scope="module")
def data():
    """Two-armed bandit trials from ssms's 2AB_RW_DDM preset (responses -1/1)."""
    return rl.Simulator(rl.preset.get("2AB_RW_DDM")).simulate(
        theta=TRUTH, n_trials=60, n_participants=3, random_state=11
    )


def _reference_logp(data, rl_alpha, scaler, a, z, t):
    """Sum of logp: a NumPy Rescorla-Wagner learner plus PyTensor `logp_ddm`."""
    drifts = []
    for _, trials in data.groupby("participant_id", sort=False):
        q = np.array([0.5, 0.5])
        for response, reward in zip(trials["response"], trials["feedback"]):
            drifts.append((q[1] - q[0]) * scaler)
            arm = int(response > 0)
            q[arm] += rl_alpha * (reward - q[arm])
    n = len(drifts)
    obs = pt.matrix("obs", dtype="float64")
    params = [pt.vector(name, dtype="float64") for name in "vazt"]
    logp = pytensor.function([obs, *params], logp_ddm(obs, *params))
    full = (np.full(n, x) for x in (a, z, t))
    return logp(data[["rt", "response"]].to_numpy(), np.array(drifts), *full).sum()


def _observed_logp(model, with_grad=False):
    """Compile the observed log-likelihood as a function of the named parameters."""
    pymc_model = model.pymc_model
    inputs = [pymc_model[name] for name in TRUTH]
    obs = pymc_model.observed_RVs[0]
    logp = pm.logp(obs, pymc_model.rvs_to_values[obs]).sum()
    outputs = [logp, *pt.grad(logp, inputs)] if with_grad else logp
    fn = pm.compile(inputs=inputs, outputs=outputs, on_unused_input="ignore")
    return lambda point: fn(*(np.asarray(point[name]) for name in TRUTH))


def _analytical_model(data, **kwargs):
    return hssm.rl.RLSSM(
        data,
        model="2AB_RW_DDM",
        decision_process_loglik_kind="analytical",
        p_outlier=None,
        **kwargs,
    )


def test_analytical_kind_never_loads_onnx(data):
    """The analytical decision process builds without touching the LAN."""
    onnx = mock.patch.object(
        registry,
        "make_jax_matrix_logp_funcs_from_onnx",
        side_effect=AssertionError("the analytical path loaded an ONNX network"),
    )
    with onnx:
        _analytical_model(data)
        hssm.rl.get_rlssm_model_config(
            "2AB_RW_DDM", decision_process_loglik_kind="analytical"
        )

    assert list(registry._SSM_LOGP_CACHE) == [("ddm", "analytical")]


@pytest.mark.parametrize(
    "point",
    [TRUTH, {**TRUTH, "rl_alpha": 0.6, "a": 0.9}, {**TRUTH, "scaler": 6.0, "z": 0.3}],
)
def test_observed_logp_matches_reference(data, point):
    """The model's likelihood is the RW learner composed with `logp_ddm`."""
    logp = _observed_logp(_analytical_model(data))(point)

    np.testing.assert_allclose(logp, _reference_logp(data, **point), rtol=0, atol=1e-9)


def test_gradient_matches_finite_differences(data):
    """The model's gradient (through the RL Op's VJP) matches central differences."""
    logp_and_grad = _observed_logp(_analytical_model(data), with_grad=True)
    _, *grad = logp_and_grad(TRUTH)

    h = 1e-6
    for name, analytic in zip(TRUTH, grad):
        up = logp_and_grad({**TRUTH, name: TRUTH[name] + h})[0]
        down = logp_and_grad({**TRUTH, name: TRUTH[name] - h})[0]
        np.testing.assert_allclose(analytic, (up - down) / (2 * h), rtol=1e-4)


def test_lan_and_analytical_coexist(data):
    """Building one kind neither replaces nor changes the other, in either order."""
    point = {**TRUTH, "scaler": 2.0}
    lan_before = _observed_logp(hssm.rl.RLSSM(data, p_outlier=None))(point)
    analytical = _observed_logp(_analytical_model(data))(point)
    lan_after = _observed_logp(hssm.rl.RLSSM(data, p_outlier=None))(point)

    assert set(registry._SSM_LOGP_CACHE) == {
        ("ddm", "approx_differentiable"),
        ("ddm", "analytical"),
    }
    assert lan_after == lan_before
    assert analytical != lan_before
    np.testing.assert_allclose(analytical, _reference_logp(data, **point), atol=1e-9)


def test_kind_survives_the_model_build(data):
    """The kind reaches the model's config; the model still defaults to numpyro."""
    model = _analytical_model(data)

    assert model.model_config.decision_process_loglik_kind == "analytical"
    # base.sample() picks numpyro for an approx_differentiable model with a JAX
    # backend; the assembled RL Op is both, whatever the decision likelihood.
    assert model.loglik_kind == "approx_differentiable"
    assert model.model_config.backend == "jax"


def test_save_load_keeps_the_analytical_likelihood(data):
    """A pickled model rebuilds the analytical decision process, not the LAN."""
    model = _analytical_model(data)
    expected = _observed_logp(model)(TRUTH)
    pickled = cloudpickle.dumps(model)
    registry._SSM_LOGP_CACHE.clear()  # as in a fresh process

    restored = cloudpickle.loads(pickled)

    assert restored.model_config.decision_process_loglik_kind == "analytical"
    np.testing.assert_allclose(_observed_logp(restored)(TRUTH), expected, atol=1e-12)


@pytest.mark.parametrize(
    ("model", "kind", "match"),
    [
        ("2AB_RW_Angle", "analytical", "has no analytical likelihood"),
        ("2AB_RW_DDM", "blackbox", "has no JAX implementation"),
        ("2AB_RW_DDM", "bogus", "must be one of"),
    ],
)
def test_unusable_kinds_raise(model, kind, match):
    """A kind the decision process cannot provide in JAX is rejected up front."""
    with pytest.raises(ValueError, match=match):
        hssm.rl.get_rlssm_model_config(model, decision_process_loglik_kind=kind)


def test_kind_must_agree_with_model_config(data):
    """A conflicting kind next to model_config raises; an equal one is accepted."""
    config = hssm.rl.get_rlssm_model_config(
        "2AB_RW_DDM", decision_process_loglik_kind="analytical"
    )

    with pytest.raises(ValueError, match="conflicts with model_config"):
        hssm.rl.RLSSM(
            data,
            model_config=config,
            decision_process_loglik_kind="approx_differentiable",
        )
    hssm.rl.RLSSM(data, model_config=config, decision_process_loglik_kind="analytical")


def test_custom_model_takes_bounds_from_the_analytical_entry():
    """A registered RLSSM on the analytical ddm uses its bounds, with finite defaults."""
    hssm.rl.register_rlssm_model(
        name="rw_ddm_analytical",
        decision_process="ddm",
        decision_process_loglik_kind="analytical",
        learning_process={"v": registry._compute_v_annotated},
        learning_process_params=["rl_alpha", "scaler"],
        learning_process_bounds={"rl_alpha": (0.0, 1.0), "scaler": (0.0, 10.0)},
        learning_process_params_default=[0.2, 1.5],
        extra_fields=["feedback"],
        choices=[0, 1],
    )

    config = hssm.rl.get_rlssm_model_config("rw_ddm_analytical")

    assert config.decision_process_loglik_kind == "analytical"
    assert {p: config.bounds[p] for p in "azt"} == {p: ddm_bounds[p] for p in "azt"}
    assert np.all(np.isfinite(config.params_default))
    assert list(registry._SSM_LOGP_CACHE) == [("ddm", "analytical")]


def test_float32_gradients_are_finite(data):
    """In float32, the gradient is finite, including where t exceeds some rts."""
    rng = np.random.default_rng(12)
    with _precision("float32"):
        logp_and_grad = _observed_logp(_analytical_model(data), with_grad=True)
        for _ in range(50):
            point = {
                "rl_alpha": rng.uniform(0.05, 0.95),
                "scaler": rng.uniform(0.5, 8.0),
                "a": rng.uniform(0.5, 2.0),
                "z": rng.uniform(0.2, 0.8),
                "t": rng.uniform(0.0, 1.2 * data["rt"].min() + 0.2),
            }
            values = logp_and_grad({k: np.float32(v) for k, v in point.items()})
            assert all(np.all(np.isfinite(v)) for v in values)


@pytest.mark.slow
def test_parameter_recovery():
    """The analytical RLSSM recovers the generating parameters.

    Simulated at delta_t = 1e-4: at ssms's default 1 ms step, the Euler scheme
    widens the boundary enough to bias the exact likelihood's `a` by about 0.02.
    """
    config = rl.preset.get("2AB_RW_DDM")
    config.ssm_kwargs = {"delta_t": 1e-4, "max_t": 20.0}
    trials = rl.Simulator(config).simulate(
        theta=TRUTH, n_trials=200, n_participants=5, random_state=3
    )

    idata = _analytical_model(trials).sample(
        sampler="numpyro", chains=2, tune=500, draws=500, random_seed=3
    )

    posterior = idata["posterior"]
    for name, value in TRUTH.items():
        draws = np.asarray(posterior[name]).ravel()
        assert abs(draws.mean() - value) < 3 * draws.std(), name
    assert int(idata["sample_stats"]["diverging"].sum()) == 0

"""Phase 3 fast tests: LAN emission backend, config variants, prior harmonisation.

Non-sampling tests covering the ``approx_differentiable`` (LAN) emission path on
both the ``jax`` and ``pytensor`` backends (including a LAN-only SSM, ``angle``),
the non-default transition / initial-distribution config variants, and the
HSSM-style prior-dict shorthand resolving to the same model as the spec
dataclasses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

import hssm
from hssm import RSSSM
from hssm.hmm import (
    DirichletConcentration,
    FixedInitialDistribution,
    StickyDirichlet,
)
from hssm.hmm.likelihoods.emissions import (
    per_regime_emission_logp,
    resolve_emission_dist,
)

from .conftest import eval_at_point


def _sim(model, theta, n=80, seed=0):
    d = hssm.simulate_data(
        model=model, theta=theta, size=n, random_state=seed, output_df=False
    ).astype("float32")
    return pd.DataFrame(d, columns=["rt", "response"])


def _logp_finite(m):
    ip = m.pymc_model.initial_point()
    return bool(np.isfinite(m.pymc_model.compile_logp()(ip)))


# ---------------------------------------------------------------------------
# LAN emission backend
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ddm_df():
    return _sim("ddm", {"v": 0.8, "a": 1.0, "z": 0.5, "t": 0.3})


@pytest.fixture(scope="module")
def angle_df():
    return _sim("angle", {"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.3, "theta": 0.2})


@pytest.mark.parametrize("backend", ["jax", "pytensor"])
def test_lan_build_ddm(ddm_df, backend):
    m = RSSSM(
        data=ddm_df,
        model="ddm",
        K=2,
        switching_params=["v"],
        loglik_kind="approx_differentiable",
        backend=backend,
    )
    assert m.model_config.loglik_kind == "approx_differentiable"
    assert m.model_config.backend == backend
    assert _logp_finite(m)


@pytest.mark.parametrize("backend", ["jax", "pytensor"])
def test_lan_build_angle_lan_only(angle_df, backend):
    """`angle` has no analytical likelihood; it must build via the LAN path."""
    m = RSSSM(
        data=angle_df,
        model="angle",
        K=2,
        switching_params=["v"],
        loglik_kind="approx_differentiable",
        backend=backend,
    )
    assert m.list_params == ["v", "a", "z", "t", "theta"]
    assert _logp_finite(m)


@pytest.mark.parametrize("backend", ["jax", "pytensor"])
def test_lan_finite_gradient_at_init(ddm_df, backend):
    """The LAN start must also give finite gradients (safe `t` initval)."""
    m = RSSSM(
        data=ddm_df,
        model="ddm",
        K=2,
        switching_params=["v"],
        loglik_kind="approx_differentiable",
        backend=backend,
    )
    ip = m.pymc_model.initial_point()
    grad = m.pymc_model.compile_dlogp()(ip)
    assert np.all(np.isfinite(grad))


def test_lan_only_model_without_loglik_kind(angle_df):
    """`angle` has no analytical likelihood; omitting `loglik_kind` must still work.

    The bounds / default priors are already looked up from the model's first
    available kind, so the emission has to be built with that same kind —
    otherwise the constructor raises (`loglik must be a Callable, str, or
    PathLike`) where `hssm.HSSM(model="angle")` builds fine.
    """
    m = RSSSM(data=angle_df, model="angle", K=2, switching_params=["v"])
    assert m.model_config.loglik_kind == "approx_differentiable"
    assert m.model_config.backend == "jax"
    assert _logp_finite(m)


def test_lan_jax_default_backend(ddm_df):
    """approx_differentiable defaults to backend='jax' (HSSM default)."""
    m = RSSSM(
        data=ddm_df,
        model="ddm",
        K=2,
        switching_params=["v"],
        loglik_kind="approx_differentiable",
    )
    assert m.model_config.backend == "jax"
    assert _logp_finite(m)


def test_lan_backends_agree_at_fixed_point(ddm_df):
    """LAN jax and pytensor evaluate the same ONNX net -> identical emission."""
    data = ddm_df[["rt", "response"]].to_numpy(dtype="float32")
    M = data.shape[0]

    def emit(backend):
        dist = resolve_emission_dist(
            "ddm", "approx_differentiable", backend, list_params=["v", "a", "z", "t"]
        )
        broadcast = backend == "jax"
        with pm.Model():
            params = {}
            for name, val in [("v", 0.2), ("a", 1.0), ("z", 0.5), ("t", 0.3)]:
                tv = pt.as_tensor_variable(np.float32(val))
                params[name] = pt.broadcast_to(tv, (M,)) if broadcast else tv
            return per_regime_emission_logp(
                dist, pt.as_tensor_variable(data), [params]
            ).eval()[:, 0]

    # Exact agreement only holds at float64; under `hssm.set_floatX("float32")`
    # the two evaluations of the same net differ by float32 round-off (~1e-7).
    tol = 1e-10 if pytensor.config.floatX == "float64" else 1e-5
    assert np.max(np.abs(emit("jax") - emit("pytensor"))) < tol


# ---------------------------------------------------------------------------
# Non-default config variants
# ---------------------------------------------------------------------------


def test_dirichlet_concentration_variant(ddm_df):
    m = RSSSM(
        data=ddm_df,
        model="ddm",
        K=2,
        switching_params=["v"],
        transition_prior=DirichletConcentration(alpha=np.array([[30, 2], [2, 30]])),
    )
    # The Dirichlet concentration drives P's prior.
    assert "P" in {rv.name for rv in m.pymc_model.free_RVs}
    assert _logp_finite(m)


def test_fixed_initial_distribution_variant(ddm_df):
    m = RSSSM(
        data=ddm_df,
        model="ddm",
        K=2,
        switching_params=["v"],
        initial_distribution=FixedInitialDistribution(pi0=[0.7, 0.3]),
    )
    # Fixed pi0 -> not an estimable RV.
    assert "pi0" not in {rv.name for rv in m.pymc_model.free_RVs}
    assert _logp_finite(m)


def test_transition_prior_alpha_reaches_the_graph(ddm_df):
    """The user's concentration is what P's prior is actually built with.

    Asserting only that `P` is a free RV (or that the logp is finite) does not
    discriminate: substituting the default sticky concentration leaves both
    true.
    """
    alpha = np.array([[7.0, 3.0], [2.0, 11.0]])
    m = RSSSM(
        data=ddm_df,
        model="ddm",
        K=2,
        switching_params=["v"],
        transition_prior={"name": "Dirichlet", "alpha": alpha},
    )
    ip = m.pymc_model.initial_point()
    p_value = eval_at_point(m, m.pymc_model["P"], ip)
    got = float(
        eval_at_point(
            m, m.pymc_model.logp(vars=[m.pymc_model["P"]], jacobian=False), ip
        )
    )
    expected = float(
        pm.logp(pm.Dirichlet.dist(a=alpha, shape=(2, 2)), p_value).sum().eval()
    )
    assert abs(got - expected) < 1e-5
    # The check discriminates: the default sticky prior scores P differently.
    sticky = float(
        pm.logp(
            pm.Dirichlet.dist(a=StickyDirichlet().concentration(2), shape=(2, 2)),
            p_value,
        )
        .sum()
        .eval()
    )
    assert abs(expected - sticky) > 1.0


def test_estimable_pi0_alpha_reaches_the_graph(ddm_df):
    """The user's Dirichlet alpha for `pi0` reaches the graph, not a flat one."""
    alpha = np.array([9.0, 1.0])
    m = RSSSM(
        data=ddm_df,
        model="ddm",
        K=2,
        switching_params=["v"],
        initial_distribution={"name": "Dirichlet", "alpha": alpha},
    )
    ip = m.pymc_model.initial_point()
    pi0_value = eval_at_point(m, m.pymc_model["pi0"], ip)
    got = float(
        eval_at_point(
            m, m.pymc_model.logp(vars=[m.pymc_model["pi0"]], jacobian=False), ip
        )
    )
    expected = float(pm.logp(pm.Dirichlet.dist(a=alpha), pi0_value).eval())
    assert abs(got - expected) < 1e-5
    flat = float(pm.logp(pm.Dirichlet.dist(a=np.ones(2)), pi0_value).eval())
    assert abs(expected - flat) > 1.0


def test_fixed_pi0_reaches_the_forward_recursion(ddm_df):
    """A fixed non-uniform `pi0` is the vector the forward recursion starts from.

    Checked against an independent NumPy forward filter fed the *same* emission
    and parameter values; a uniform substitution (the previous tests' blind
    spot) gives a measurably different marginal.
    """
    from scipy.special import logsumexp

    from hssm.hmm import ffbs

    pi0 = np.array([0.85, 0.15])
    m = RSSSM(
        data=ddm_df,
        model="ddm",
        K=2,
        switching_params=["v"],
        initial_distribution=pi0,
    )
    ip = m.pymc_model.initial_point()
    potential = float(eval_at_point(m, m.pymc_model.potentials[0], ip))

    emission_fn, order = ffbs._compile_emission_fn(m)
    params = [eval_at_point(m, m.pymc_model[name], ip).astype(float) for name in order]
    log_em = np.asarray(emission_fn(*params), dtype=float)[0]  # (T, K)
    log_P = np.log(eval_at_point(m, m.pymc_model["P"], ip).astype(float))
    ref = float(logsumexp(ffbs._forward_filter(log_em, log_P, np.log(pi0))[-1]))
    assert abs(potential - ref) < 1e-2

    uniform = float(
        logsumexp(ffbs._forward_filter(log_em, log_P, np.log(np.full(2, 0.5)))[-1])
    )
    assert abs(ref - uniform) > 0.05  # the check discriminates


# ---------------------------------------------------------------------------
# Prior-input harmonisation (dict / bmb.Prior <-> spec dataclasses)
# ---------------------------------------------------------------------------


def test_transition_prior_dict_matches_dataclass(ddm_df):
    """The HSSM-style Dirichlet dict produces the same model as the dataclass."""
    alpha = np.array([[25.0, 3.0], [3.0, 25.0]])
    m_dict = RSSSM(
        data=ddm_df,
        model="ddm",
        K=2,
        switching_params=["v"],
        transition_prior={"name": "Dirichlet", "alpha": alpha},
    )
    m_dc = RSSSM(
        data=ddm_df,
        model="ddm",
        K=2,
        switching_params=["v"],
        transition_prior=DirichletConcentration(alpha=alpha),
    )
    ip = m_dict.pymc_model.initial_point()
    lp_dict = float(m_dict.pymc_model.compile_logp()(ip))
    lp_dc = float(m_dc.pymc_model.compile_logp()(ip))
    assert abs(lp_dict - lp_dc) < 1e-8


def test_initial_distribution_dict_is_estimable(ddm_df):
    """A Dirichlet dict for pi0 yields an estimable pi0 RV (dict = inferred)."""
    m = RSSSM(
        data=ddm_df,
        model="ddm",
        K=2,
        switching_params=["v"],
        initial_distribution={"name": "Dirichlet", "alpha": [1, 1]},
    )
    assert "pi0" in {rv.name for rv in m.pymc_model.free_RVs}
    assert _logp_finite(m)

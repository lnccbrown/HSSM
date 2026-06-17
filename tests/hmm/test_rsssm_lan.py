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
import pytensor.tensor as pt
import pytest

import hssm
from hssm import RSSSM
from hssm.hmm import DirichletConcentration, FixedInitialDistribution
from hssm.hmm.likelihoods.emissions import (
    per_regime_emission_logp,
    resolve_emission_dist,
)


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

    assert np.max(np.abs(emit("jax") - emit("pytensor"))) < 1e-10


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

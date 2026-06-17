"""Slow sampling tests for RSSSM (parameter recovery).

These actually run NUTS via numpyro and are marked ``slow`` (run with
``pytest --runslow``).  They exercise the overridden ``sample()`` / ``summary()``
surface and confirm parameter recovery on synthetic regime-switching data.

Note on the tutorial regression: the *structural* bit-for-bit equivalence of
RSSSM's likelihood to the hand-written tutorial is asserted deterministically in
``test_rsssm.py::test_forward_marginal_matches_tutorial`` (independent sampling
of two different parameterisations — soft Potential vs. the ``ordered``
transform — cannot match draw-for-draw).  Here we assert recovery instead.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest

from hssm import RSSSM

from .conftest import simulate_hmm_data, simulate_hmm_ddm_data

pytestmark = pytest.mark.slow


def test_sample_recovers_single_participant_k2(tutorial_data):
    """K=2, single participant: recover the (sorted) drift rates and a, z, t."""
    df, _ = tutorial_data
    model = RSSSM(
        data=df,
        model="ddm",
        K=2,
        switching_params=["v"],
        v={"name": "Normal", "mu": 0.0, "sigma": 3.0},
        a={"name": "HalfNormal", "sigma": 2.0},
        z={"name": "Beta", "alpha": 10, "beta": 10},
        t={"name": "HalfNormal", "sigma": 0.5},
    )
    idata = model.sample(
        draws=500, tune=500, chains=2, target_accept=0.9, random_seed=42
    )

    summary = model.summary()
    assert summary["r_hat"].max() < 1.05

    post = idata.posterior
    v_mean = post["v"].mean(("chain", "draw")).values  # ascending: [low, high]
    # True drifts sorted ascending: regime 1 (0.2) then regime 0 (1.5).
    assert v_mean[0] < v_mean[1]
    assert abs(v_mean[0] - 0.2) < 0.3
    # The high-drift "attentive" regime is underoccupied in the tutorial data
    # (pi0=[0.8, 0.2], sticky P), so it is informed by fewer trials -> a looser
    # but still-meaningful recovery band.
    assert abs(v_mean[1] - 1.5) < 0.5
    assert abs(float(post["a"].mean()) - 0.8) < 0.2
    assert abs(float(post["t"].mean()) - 0.3) < 0.1


def test_sample_multi_participant_k3():
    """K=3, multi-participant: chains converge in one mode (no label-switching)."""
    # N=3 participants, T=200, true drifts well separated.
    true_params = {
        0: {"v": -1.0, "a": 1.0, "z": 0.5, "t": 0.3},
        1: {"v": 0.5, "a": 1.0, "z": 0.5, "t": 0.3},
        2: {"v": 1.8, "a": 1.0, "z": 0.5, "t": 0.3},
    }
    P = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    pi0 = np.array([1 / 3, 1 / 3, 1 / 3])

    frames = []
    for p in range(3):
        data, _ = simulate_hmm_ddm_data(200, true_params, P, pi0, seed=10 + p)
        sub = pd.DataFrame(data, columns=["rt", "response"])
        sub["participant_id"] = p
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)

    model = RSSSM(
        data=df,
        model="ddm",
        K=3,
        switching_params=["v"],
        participant_col="participant_id",
        v={"name": "Normal", "mu": 0.0, "sigma": 3.0},
        a={"name": "HalfNormal", "sigma": 2.0},
        z={"name": "Beta", "alpha": 10, "beta": 10},
        t={"name": "HalfNormal", "sigma": 0.5},
    )
    idata = model.sample(
        draws=500, tune=500, chains=2, target_accept=0.9, random_seed=42
    )

    summary = model.summary()
    assert summary["r_hat"].max() < 1.1

    v_mean = idata.posterior["v"].mean(("chain", "draw")).values
    # Ascending anchor -> recovered drifts ordered and near the true sorted set.
    assert np.all(np.diff(v_mean) > 0)
    assert np.max(np.abs(v_mean - np.array([-1.0, 0.5, 1.8]))) < 0.4


def test_lan_only_angle_recovery():
    """LAN-only SSM (`angle`, no analytical comparand) samples end-to-end.

    Mirrors the §7.4 LAN-only validation: the emission has no analytical form, so
    this exercises the `approx_differentiable` jax path through NUTS on a
    well-separated two-regime drift. It asserts only robustly-recoverable facts —
    the regimes are recovered as clearly distinct and correctly signed (and the
    ordered transform keeps them ordered) with finite posteriors. Tight
    per-parameter recovery and a strict R-hat gate are *not* asserted: a
    2-regime LAN-only fit is an approximate, hard inference problem, and the
    LAN's *numerical* correctness is already covered deterministically in
    ``test_rsssm_lan.py``.
    """
    true_params = {
        0: {"v": -1.5, "a": 1.2, "z": 0.5, "t": 0.3, "theta": 0.2},
        1: {"v": 1.5, "a": 1.2, "z": 0.5, "t": 0.3, "theta": 0.2},
    }
    P = np.array([[0.92, 0.08], [0.08, 0.92]])
    pi0 = np.array([0.5, 0.5])
    data, _ = simulate_hmm_data("angle", 300, true_params, P, pi0, seed=7)
    df = pd.DataFrame(data, columns=["rt", "response"])

    model = RSSSM(
        data=df,
        model="angle",
        K=2,
        switching_params=["v"],
        loglik_kind="approx_differentiable",
        backend="jax",
    )
    idata = model.sample(
        draws=500, tune=500, chains=2, target_accept=0.9, random_seed=42
    )

    post = idata.posterior
    assert bool(np.isfinite(post["v"].values).all())
    v_mean = post["v"].mean(("chain", "draw")).values  # ascending
    # The two regimes are recovered as ordered, clearly distinct, and correctly
    # signed (true v = [-1.5, +1.5]); the ordered transform guarantees v[0]<v[1].
    assert v_mean[0] < 0.0 < v_mean[1]
    assert (v_mean[1] - v_mean[0]) > 1.0


def test_infer_regimes_and_loo_end_to_end():
    """End-to-end Phase 4: sample, recover regimes accurately, and run loo.

    Exercises the full post-hoc path on a real posterior: `infer_regimes`
    (FFBS) must recover the well-separated two-regime trajectory at high
    per-trial accuracy, and `compute_log_likelihood` must produce a
    `log_likelihood` group that `arviz.loo` consumes.
    """
    true_params = {
        0: {"v": -1.5, "a": 1.0, "z": 0.5, "t": 0.3},
        1: {"v": 1.5, "a": 1.0, "z": 0.5, "t": 0.3},
    }
    P = np.array([[0.92, 0.08], [0.08, 0.92]])
    pi0 = np.array([0.5, 0.5])
    data, regimes = simulate_hmm_ddm_data(250, true_params, P, pi0, seed=7)
    df = pd.DataFrame(data, columns=["rt", "response"])

    model = RSSSM(
        data=df,
        model="ddm",
        K=2,
        switching_params=["v"],
        v={"name": "Normal", "mu": 0.0, "sigma": 3.0},
    )
    idata = model.sample(draws=500, tune=500, chains=2, random_seed=42)

    # --- infer_regimes: marginal MAP tracks the ground-truth regimes ---
    reg_idata = model.infer_regimes(n_draws=200, seed=0)
    freq = reg_idata.posterior_regimes["regime_sample_frequency"].values[0]  # (T, K)
    assert np.allclose(freq.sum(axis=1), 1.0)
    map_regime = freq.argmax(axis=1)
    # Ascending anchor: regime 0 = low drift; matches the simulated labels.
    assert (map_regime == regimes).mean() > 0.8

    # --- compute_log_likelihood: arviz.loo runs on the attached group ---
    model.compute_log_likelihood(idata)
    assert "log_likelihood" in idata.groups()
    ll = idata.log_likelihood["obs"].values
    assert np.isfinite(ll).all()
    loo = az.loo(idata)
    assert np.isfinite(loo.elpd_loo)

"""Fast tests for post-hoc FFBS regime recovery and per-trial logp (Phase 4).

The expensive end-to-end recovery (sample -> infer_regimes -> accuracy) lives in
``test_rsssm_sampling.py``.  Here we validate the FFBS machinery deterministically:

- the NumPy forward filter / backward sampler against *exact* forward-backward
  smoothing and against the pytensor forward marginal, and
- the ``infer_regimes`` / ``compute_log_likelihood`` plumbing on a hand-built
  posterior (no sampling), including the ``arviz.loo`` hand-off.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import xarray as xr
from scipy.special import logsumexp, softmax

from hssm import RSSSM
from hssm.hmm import ffbs
from hssm.hmm.likelihoods.forward import forward_log_marginal

from .conftest import simulate_hmm_ddm_data


# ---------------------------------------------------------------------------
# NumPy FFBS primitives vs. exact references
# ---------------------------------------------------------------------------


def _random_hmm(rng, K, T):
    log_lik = rng.normal(size=(T, K))
    praw = rng.uniform(0.1, 1.0, size=(K, K))
    P = praw / praw.sum(1, keepdims=True)
    pi0raw = rng.uniform(0.1, 1.0, size=K)
    pi0 = pi0raw / pi0raw.sum()
    return log_lik, np.log(P), np.log(pi0)


def test_ffbs_converges_to_exact_smoothing():
    """FFBS sample frequencies match the exact forward-backward marginals.

    The backward sampler draws joint sequences; the per-trial marginal
    frequency must converge to the smoothing posterior ``p(s_t=k | y, theta)``
    computed exactly via forward-backward.
    """
    rng = np.random.default_rng(0)
    K, T = 3, 6
    log_lik, log_P, log_pi0 = _random_hmm(rng, K, T)

    log_alpha = ffbs._forward_filter(log_lik, log_P, log_pi0)
    # Exact backward messages: log_beta[t,k] = log p(y_{t+1:T} | s_t=k).
    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        for k in range(K):
            log_beta[t, k] = logsumexp(log_P[k] + log_lik[t + 1] + log_beta[t + 1])
    gamma = softmax(log_alpha + log_beta, axis=1)  # exact smoothing marginals

    n_samples = 20000
    counts = np.zeros((T, K))
    for _ in range(n_samples):
        seq = ffbs._backward_sample(log_alpha, log_P, rng)
        counts[np.arange(T), seq] += 1
    freq = counts / n_samples

    assert np.max(np.abs(freq - gamma)) < 0.02


def test_numpy_forward_filter_matches_pytensor_marginal():
    """The NumPy filter's evidence equals the pytensor (scaled) forward marginal."""
    rng = np.random.default_rng(1)
    K, T = 3, 50
    log_lik, log_P, log_pi0 = _random_hmm(rng, K, T)

    log_alpha = ffbs._forward_filter(log_lik, log_P, log_pi0)
    numpy_marginal = float(logsumexp(log_alpha[-1]))

    pt_marginal = float(
        forward_log_marginal(
            pt.as_tensor_variable(log_lik[None]),
            pt.as_tensor_variable(log_P),
            pt.as_tensor_variable(log_pi0),
            pt.ones((1, T)),
        ).eval()
    )
    assert abs(numpy_marginal - pt_marginal) < 1e-6


def test_per_trial_delta_sums_to_marginal():
    """sum_t delta_t equals the running log-evidence logZ_T (per-trial logp split)."""
    rng = np.random.default_rng(2)
    K, T = 2, 40
    log_lik, log_P, log_pi0 = _random_hmm(rng, K, T)

    log_alpha = ffbs._forward_filter(log_lik, log_P, log_pi0)
    logZ = logsumexp(log_alpha, axis=1)
    delta = np.empty(T)
    delta[0] = logZ[0]
    delta[1:] = logZ[1:] - logZ[:-1]
    assert abs(delta.sum() - logZ[-1]) < 1e-10


# ---------------------------------------------------------------------------
# infer_regimes / compute_log_likelihood plumbing (hand-built posterior)
# ---------------------------------------------------------------------------


def _fake_posterior(v, a, z, t, P, n_draws=3):
    """A 1-chain InferenceData posterior for a K=2, switching=['v'] DDM."""
    v, P = np.asarray(v, float), np.asarray(P, float)
    K = v.shape[0]
    ds = xr.Dataset(
        {
            "v": (("chain", "draw", "v_dim"), np.tile(v, (1, n_draws, 1))),
            "a": (("chain", "draw"), np.full((1, n_draws), a)),
            "z": (("chain", "draw"), np.full((1, n_draws), z)),
            "t": (("chain", "draw"), np.full((1, n_draws), t)),
            "P": (("chain", "draw", "P0", "P1"), np.tile(P, (1, n_draws, 1, 1))),
        },
        coords={
            "chain": [0],
            "draw": np.arange(n_draws),
            "v_dim": np.arange(K),
        },
    )
    return az.InferenceData(posterior=ds)


def _fitted_model():
    data, regimes = simulate_hmm_ddm_data(
        60,
        {
            0: {"v": 0.2, "a": 0.8, "z": 0.5, "t": 0.3},
            1: {"v": 1.5, "a": 0.8, "z": 0.5, "t": 0.3},
        },
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        np.array([0.8, 0.2]),
        seed=7,
    )
    df = pd.DataFrame(data, columns=["rt", "response"])
    model = RSSSM(data=df, model="ddm", K=2, switching_params=["v"])
    return model, regimes


def test_compute_log_likelihood_sums_to_model_marginal():
    """Per-trial delta_t (summed) equals the model's scalar forward marginal."""
    model, _ = _fitted_model()
    idata = _fake_posterior([0.2, 1.5], 0.8, 0.5, 0.3, [[0.9, 0.1], [0.1, 0.9]])

    model.compute_log_likelihood(idata)
    ll = idata.log_likelihood["obs"].values  # (chain, draw, participant, trial)

    # Independent reference: the model's own forward marginal at the same params.
    emission_fn, order = ffbs._compile_emission_fn(model)
    params = [ffbs._draw_param_value(model, nm, idata.posterior, 0, 0) for nm in order]
    log_em = np.asarray(emission_fn(*params))  # (N, T, K)
    log_P = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
    log_pi0 = np.log(np.ones(2) / 2)
    ref = float(
        forward_log_marginal(
            pt.as_tensor_variable(log_em),
            pt.as_tensor_variable(log_P),
            pt.as_tensor_variable(log_pi0),
            pt.ones(log_em.shape[:2]),
        ).eval()
    )
    assert abs(float(ll[0, 0].sum()) - ref) < 1e-4


def test_compute_log_likelihood_enables_loo():
    """The attached log_likelihood group is consumable by arviz.loo."""
    model, _ = _fitted_model()
    idata = _fake_posterior(
        [0.2, 1.5], 0.8, 0.5, 0.3, [[0.9, 0.1], [0.1, 0.9]], n_draws=8
    )
    model.compute_log_likelihood(idata)
    assert "log_likelihood" in idata.groups()
    loo = az.loo(idata)
    assert np.isfinite(loo.elpd_loo)


def test_infer_regimes_shapes_and_frequencies():
    """infer_regimes returns the documented group/shapes; frequencies are valid."""
    model, regimes = _fitted_model()
    idata = _fake_posterior(
        [0.2, 1.5], 0.8, 0.5, 0.3, [[0.9, 0.1], [0.1, 0.9]], n_draws=5
    )

    out = model.infer_regimes(idata, n_draws=5, seed=0)
    pr = out.posterior_regimes
    assert pr["regimes"].sizes == {"draw": 5, "participant": 1, "trial": 60}
    assert pr["regime_sample_frequency"].sizes == {
        "participant": 1,
        "trial": 60,
        "regime": 2,
    }
    freq = pr["regime_sample_frequency"].values[0]  # (T, K)
    assert np.allclose(freq.sum(axis=1), 1.0)
    assert freq.min() >= 0.0 and freq.max() <= 1.0
    # At well-separated true drifts the marginal MAP should track ground truth.
    map_regime = freq.argmax(axis=1)
    assert (map_regime == regimes).mean() > 0.75


def test_infer_regimes_uses_estimable_pi0(monkeypatch):
    """When pi0 is estimable, FFBS reads it from the posterior (not uniform)."""
    data, _ = simulate_hmm_ddm_data(
        40,
        {
            0: {"v": 0.2, "a": 0.8, "z": 0.5, "t": 0.3},
            1: {"v": 1.5, "a": 0.8, "z": 0.5, "t": 0.3},
        },
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        np.array([0.8, 0.2]),
        seed=1,
    )
    df = pd.DataFrame(data, columns=["rt", "response"])
    model = RSSSM(
        data=df,
        model="ddm",
        K=2,
        switching_params=["v"],
        initial_distribution={"name": "Dirichlet", "alpha": [1.0, 1.0]},
    )
    # Build a posterior that also carries an estimable pi0.
    idata = _fake_posterior(
        [0.2, 1.5], 0.8, 0.5, 0.3, [[0.9, 0.1], [0.1, 0.9]], n_draws=2
    )
    idata.posterior["pi0"] = (
        ("chain", "draw", "pi0_dim"),
        np.tile([0.7, 0.3], (1, 2, 1)),
    )

    captured = {}
    real = ffbs._log_pi0_for_draw

    def spy(m, posterior, c, d):
        out = real(m, posterior, c, d)
        captured["log_pi0"] = out
        return out

    monkeypatch.setattr(ffbs, "_log_pi0_for_draw", spy)
    model.infer_regimes(idata, n_draws=2, seed=0)
    assert np.allclose(np.exp(captured["log_pi0"]), [0.7, 0.3], atol=1e-6)


def _unbalanced_model():
    """Two participants with ragged lengths (50 and 30 real trials)."""
    a, b = [], []
    for pid, n in [(0, 50), (1, 30)]:
        data, _ = simulate_hmm_ddm_data(
            n,
            {
                0: {"v": 0.2, "a": 0.8, "z": 0.5, "t": 0.3},
                1: {"v": 1.5, "a": 0.8, "z": 0.5, "t": 0.3},
            },
            np.array([[0.9, 0.1], [0.1, 0.9]]),
            np.array([0.8, 0.2]),
            seed=pid,
        )
        sub = pd.DataFrame(data, columns=["rt", "response"])
        sub["pid"] = pid
        a.append(sub)
    df = pd.concat(a, ignore_index=True)
    return RSSSM(
        data=df, model="ddm", K=2, switching_params=["v"], participant_col="pid"
    )


def test_log_likelihood_unbalanced_excludes_padding():
    """loo counts only real trials on unbalanced panels (no padded fakes).

    Padded trials would otherwise enter as logp-0 ("perfectly predicted")
    observations and inflate ``n_data_points`` / bias elpd.
    """
    model = _unbalanced_model()
    idata = _fake_posterior(
        [0.2, 1.5], 0.8, 0.5, 0.3, [[0.9, 0.1], [0.1, 0.9]], n_draws=8
    )
    model.compute_log_likelihood(idata)
    ll = idata.log_likelihood["obs"]
    # Flat over the 50 + 30 = 80 real trials only (not the 50*2 padded grid).
    assert ll.sizes["__obs__"] == 80
    loo = az.loo(idata)
    assert int(loo.n_data_points) == 80
    assert np.isfinite(loo.elpd_loo)


def test_compute_log_likelihood_is_idempotent():
    """Re-running compute_log_likelihood overwrites rather than raising."""
    model, _ = _fitted_model()
    idata = _fake_posterior([0.2, 1.5], 0.8, 0.5, 0.3, [[0.9, 0.1], [0.1, 0.9]])
    model.compute_log_likelihood(idata)
    first = idata.log_likelihood["obs"].values.copy()
    model.compute_log_likelihood(idata)  # must not raise
    np.testing.assert_allclose(idata.log_likelihood["obs"].values, first)


def test_ffbs_and_log_likelihood_under_no_pooling():
    """FFBS + per-trial logp work with per-participant (N,K)/(N,) parameters."""
    from .conftest import make_panel

    panel = make_panel(2, 40)
    model = RSSSM(
        data=panel,
        model="ddm",
        K=2,
        switching_params=["v"],
        pooling="none",
        participant_col="participant_id",
    )
    n = model.n_participants
    ds = xr.Dataset(
        {
            "v": (
                ("chain", "draw", "p", "k"),
                np.tile([[-1.0, 1.0], [-1.0, 1.0]], (1, 4, 1, 1)),
            ),
            "a": (("chain", "draw", "p"), np.full((1, 4, n), 0.8)),
            "z": (("chain", "draw", "p"), np.full((1, 4, n), 0.5)),
            "t": (("chain", "draw", "p"), np.full((1, 4, n), 0.3)),
            "P": (
                ("chain", "draw", "i", "j"),
                np.tile([[0.9, 0.1], [0.1, 0.9]], (1, 4, 1, 1)),
            ),
        },
        coords={"chain": [0], "draw": np.arange(4)},
    )
    idata = az.InferenceData(posterior=ds)
    reg = model.infer_regimes(idata, n_draws=4, seed=0)
    assert reg.posterior_regimes["regimes"].sizes["participant"] == n
    model.compute_log_likelihood(idata)
    assert np.isfinite(idata.log_likelihood["obs"].values).all()

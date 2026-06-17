"""Fixtures and helpers for the RSSSM (regime-switching SSM) test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

import hssm
from hssm.likelihoods import DDM

# Tutorial ground truth (docs/tutorials/hmm_ddm_regime_switching.ipynb).
TUTORIAL_TRUE_PARAMS = {
    0: {"v": 1.5, "a": 0.8, "z": 0.5, "t": 0.3},  # attentive
    1: {"v": 0.2, "a": 0.8, "z": 0.5, "t": 0.3},  # distracted
}
TUTORIAL_P = np.array([[0.95, 0.05], [0.10, 0.90]])
TUTORIAL_PI0 = np.array([0.8, 0.2])


def simulate_hmm_ddm_data(n_trials, regime_params, P, pi0, seed=42):
    """Simulate one participant from a regime-switching DDM (tutorial logic)."""
    rng = np.random.default_rng(seed)
    K = len(regime_params)

    regimes = np.empty(n_trials, dtype=int)
    regimes[0] = rng.choice(K, p=pi0)
    for t in range(1, n_trials):
        regimes[t] = rng.choice(K, p=P[regimes[t - 1]])

    rts, responses = [], []
    for k in range(K):
        mask = regimes == k
        n_k = int(mask.sum())
        if n_k == 0:
            rts.append(np.array([]))
            responses.append(np.array([]))
            continue
        sims = hssm.simulate_data(
            model="ddm",
            theta=regime_params[k],
            size=n_k,
            random_state=seed + k,
            output_df=False,
        )
        rts.append(sims[:, 0])
        responses.append(sims[:, 1])

    data = np.empty((n_trials, 2))
    for k in range(K):
        mask = regimes == k
        if mask.sum():
            data[mask, 0] = rts[k]
            data[mask, 1] = responses[k]
    return data.astype("float32"), regimes


def build_tutorial_forward_marginal(data, v, a, z, t, P, K):
    """Compile the tutorial's hand-written forward marginal at a fixed point.

    Returns the scalar marginal log-likelihood (float), the structural
    reference used to assert RSSSM's emission+forward is bit-for-bit identical.
    """
    data_obs = pt.as_tensor_variable(data.astype("float32"))
    comps = [
        pm.logp(DDM.dist(v=float(v[k]), a=a, z=z, t=t), data_obs) for k in range(K)
    ]
    log_lik = pt.stack(comps, axis=1)
    log_P = pt.log(pt.as_tensor_variable(P.astype("float32")))
    log_pi0 = pt.log(pt.ones(K) / K)
    log_alpha_init = log_pi0 + log_lik[0]

    def step(ll, ap, lp):
        return pt.logsumexp(ap[:, None] + lp, axis=0) + ll

    log_alphas, _ = pytensor.scan(
        fn=step,
        sequences=[log_lik[1:]],
        outputs_info=[log_alpha_init],
        non_sequences=[log_P],
    )
    return float(pt.logsumexp(log_alphas[-1]).eval())


@pytest.fixture(scope="module")
def tutorial_data():
    """Single-participant K=2 dataset matching the tutorial (500 trials)."""
    data, regimes = simulate_hmm_ddm_data(
        500, TUTORIAL_TRUE_PARAMS, TUTORIAL_P, TUTORIAL_PI0, seed=42
    )
    df = pd.DataFrame(data, columns=["rt", "response"])
    return df, regimes


@pytest.fixture(scope="module")
def small_single_participant():
    """A small single-participant DDM dataset for fast build tests."""
    data = hssm.simulate_data(
        model="ddm",
        theta={"v": 1.0, "a": 0.8, "z": 0.5, "t": 0.3},
        size=60,
        random_state=1,
        output_df=False,
    ).astype("float32")
    return pd.DataFrame(data, columns=["rt", "response"])


def make_panel(n_participants, t_each, seed=0, theta=None):
    """Build a balanced long-format panel of DDM data."""
    theta = theta or {"v": 0.8, "a": 0.9, "z": 0.5, "t": 0.3}
    frames = []
    for p in range(n_participants):
        d = hssm.simulate_data(
            model="ddm",
            theta=theta,
            size=t_each,
            random_state=seed + p,
            output_df=False,
        )
        sub = pd.DataFrame(d, columns=["rt", "response"])
        sub["participant_id"] = p
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)

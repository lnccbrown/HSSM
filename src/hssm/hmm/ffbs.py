"""Post-hoc regime recovery (FFBS) and per-trial log-likelihood (design §5.5/§5.6).

NUTS samples only the continuous parameters; the discrete regimes ``s_{n,t}`` are
marginalised out by the forward algorithm at sampling time (§3.3).  This module
reconstructs them *after* sampling, in pure NumPy:

- ``infer_regimes`` runs Forward-Filter Backward-Sample (FFBS) for a set of
  posterior draws, drawing one plausible regime sequence per participant per
  draw — a posterior over regime trajectories.
- ``compute_log_likelihood`` reconstructs the per-trial one-step-ahead
  contributions ``delta_t = logZ_t - logZ_{t-1}`` so ``arviz.loo`` / ``waic``
  work despite the sampler graph contributing only the scalar marginal (§3.4).

Both consume the *same* emission as sampling time: ``build_log_emission`` is
recompiled to a NumPy callable via ``pytensor.function``, which works uniformly
across the analytical, LAN-pytensor, and LAN-jax backends (each is an ordinary
pytensor graph node).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytensor
import pytensor.tensor as pt
from scipy.special import logsumexp, softmax

from .likelihoods.builder import build_log_emission
from .specs import DirichletInitialDistribution, NoPooling

if TYPE_CHECKING:
    import arviz as az

    from .rsssm import RSSSM


# ----------------------------------------------------------------------------
# Emission callable (shared by FFBS and per-trial logp)
# ----------------------------------------------------------------------------


def _compile_emission_fn(model: RSSSM):
    """Compile the ``(N, T, K)`` log-emission to a NumPy callable.

    Returns ``(fn, param_order)`` where ``fn(*param_arrays)`` yields the
    ``(N, T, K)`` per-trial, per-regime emission log-density and ``param_order``
    is the list of SSM parameter names in the order ``fn`` expects them.
    """
    cfg = model.model_config
    K, N, T = model.K, model.n_participants, model.n_trials
    is_no_pooling = isinstance(cfg.pooling, NoPooling)
    pooling_mode = "none" if is_no_pooling else "full"

    inputs: list[pt.TensorVariable] = []
    param_values: dict[str, pt.TensorVariable] = {}
    for name in model.list_params or []:
        is_regime = name in model._regime_params
        if is_no_pooling:
            sym = pt.matrix(name) if is_regime else pt.vector(name)
        else:
            sym = pt.vector(name) if is_regime else pt.scalar(name)
        inputs.append(sym)
        param_values[name] = sym

    log_emission = build_log_emission(
        dist_class=model._emission_dist,
        param_values=param_values,
        data_flat_np=model._data_padded.reshape(N * T, model._data_padded.shape[-1]),
        K=K,
        n_participants=N,
        n_trials=T,
        regime_params=model._regime_params,
        pooling=pooling_mode,
        broadcast_params=model._broadcast_params,
    )
    fn = pytensor.function(inputs, log_emission, on_unused_input="ignore")
    return fn, list(model.list_params or [])


def _draw_param_value(model: RSSSM, name: str, posterior, chain: int, draw: int):
    """Return the value of SSM parameter ``name`` for one posterior draw.

    Inferred parameters are read from the posterior; fixed scalars / fixed-per-
    regime vectors are taken from the config and broadcast to the shape the
    compiled emission expects under the active pooling mode.
    """
    cfg = model.model_config
    is_no_pooling = isinstance(cfg.pooling, NoPooling)
    is_regime = name in model._regime_params
    spec = cfg.param_specs.get(name)
    K, N = model.K, model.n_participants

    # Fixed value supplied directly (scalar or length-K list).
    is_fixed_scalar = isinstance(spec, (int, float)) and not isinstance(spec, bool)
    is_fixed_vector = isinstance(spec, (list, tuple, np.ndarray))
    if is_fixed_scalar or is_fixed_vector:
        val = np.asarray(spec, dtype=float)
        if not is_no_pooling:
            return val  # full pooling: scalar or (K,)
        # no pooling: broadcast across participants.
        return np.broadcast_to(val, (N, K) if is_regime else (N,)).astype(float)

    # Inferred: read from the posterior (shape already matches the pooling mode).
    return np.asarray(posterior[name].values[chain, draw], dtype=float)


def _log_pi0_for_draw(model: RSSSM, posterior, chain: int, draw: int) -> np.ndarray:
    """Log initial-state distribution for one draw.

    Uses the posterior ``pi0`` when it is estimable, otherwise the fixed config
    vector — FFBS must use the same ``pi0`` the model was fit with (a hardcoded
    uniform would be inconsistent with an estimable or fixed-non-uniform pi0).
    """
    K = model.K
    init = model.model_config.initial_distribution
    if isinstance(init, DirichletInitialDistribution):
        pi0 = np.asarray(posterior["pi0"].values[chain, draw], dtype=float)
    else:
        pi0 = np.asarray(init.pi0_value(K), dtype=float)
    return np.log(pi0 + 1e-300)


# ----------------------------------------------------------------------------
# NumPy forward filter / backward sample
# ----------------------------------------------------------------------------


def _forward_filter(
    log_lik: np.ndarray, log_P: np.ndarray, log_pi0: np.ndarray
) -> np.ndarray:
    """Forward variables ``log_alpha[t, k] = log p(y_{1:t}, s_t=k | theta)``.

    ``log_lik`` is ``(T, K)`` for one participant's real trials.
    """
    T, K = log_lik.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi0 + log_lik[0]
    for t in range(1, T):
        # logsumexp_j (log_alpha[t-1, j] + log_P[j, k]) + log_lik[t, k]
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_P, axis=0) + log_lik[t]
    return log_alpha


def _backward_sample(
    log_alpha: np.ndarray, log_P: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw one regime sequence from ``p(s_{1:T} | y_{1:T}, theta)`` via FFBS."""
    T, K = log_alpha.shape
    s = np.empty(T, dtype=int)
    s[T - 1] = rng.choice(K, p=softmax(log_alpha[T - 1]))
    for t in range(T - 2, -1, -1):
        # p(s_t=k | s_{t+1}, y) ∝ alpha_t(k) * P(k -> s_{t+1})
        s[t] = rng.choice(K, p=softmax(log_alpha[t] + log_P[:, s[t + 1]]))
    return s


def _participant_lengths(model: RSSSM) -> np.ndarray:
    """Return the number of real (non-padded) trials per participant."""
    return model._mask.sum(axis=1).astype(int)


def _select_draws(
    posterior, n_draws: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Pick ``n_draws`` ``(chain, draw)`` index pairs without replacement."""
    n_chains, n_post = posterior.sizes["chain"], posterior.sizes["draw"]
    total = n_chains * n_post
    k = min(n_draws, total)
    flat = rng.choice(total, size=k, replace=False)
    return [divmod(int(i), n_post) for i in flat]


# ----------------------------------------------------------------------------
# Public entry points (called by RSSSM methods)
# ----------------------------------------------------------------------------


def infer_regimes(
    model: RSSSM, idata: az.InferenceData, n_draws: int, seed: int | None
) -> az.InferenceData:
    """FFBS posterior over regime sequences (design §5.5)."""
    import arviz as az
    import xarray as xr

    posterior = idata.posterior  # type: ignore[attr-defined]
    K, N, T = model.K, model.n_participants, model.n_trials
    lengths = _participant_lengths(model)
    rng = np.random.default_rng(seed)

    emission_fn, param_order = _compile_emission_fn(model)
    draw_pairs = _select_draws(posterior, n_draws, rng)
    n_eff = len(draw_pairs)

    regimes = np.full((n_eff, N, T), -1, dtype=int)
    for di, (c, d) in enumerate(draw_pairs):
        params = [_draw_param_value(model, nm, posterior, c, d) for nm in param_order]
        log_em = np.asarray(emission_fn(*params))  # (N, T, K)
        log_P = np.log(np.asarray(posterior["P"].values[c, d], dtype=float) + 1e-300)
        log_pi0 = _log_pi0_for_draw(model, posterior, c, d)
        for n in range(N):
            Tn = lengths[n]
            log_alpha = _forward_filter(log_em[n, :Tn], log_P, log_pi0)
            regimes[di, n, :Tn] = _backward_sample(log_alpha, log_P, rng)

    # Monte-Carlo marginal p(s_{n,t}=k | y, theta), NaN at padded trials.
    freq = np.full((N, T, K), np.nan)
    for n in range(N):
        Tn = lengths[n]
        seq = regimes[:, n, :Tn]  # (n_eff, Tn)
        onehot = (seq[:, :, None] == np.arange(K)[None, None, :]).mean(axis=0)
        freq[n, :Tn, :] = onehot

    ds = xr.Dataset(
        {
            "regimes": (("draw", "participant", "trial"), regimes),
            "regime_sample_frequency": (("participant", "trial", "regime"), freq),
        },
        coords={
            "draw": np.arange(n_eff),
            "participant": np.arange(N),
            "trial": np.arange(T),
            "regime": np.arange(K),
        },
    )
    return az.InferenceData(posterior_regimes=ds)


def compute_log_likelihood(model: RSSSM, idata: az.InferenceData) -> az.InferenceData:
    """Attach the post-hoc per-trial log-likelihood group (design §5.6).

    For every posterior draw, the forward filter's running log-evidence
    ``logZ_t = logsumexp_k log_alpha_t(k)`` gives the one-step-ahead per-trial
    contribution ``delta_t = logZ_t - logZ_{t-1}`` (with ``delta_0 = logZ_0``).
    By construction the per-participant sum equals that participant's marginal,
    and the grand total equals the scalar marginal the sampler used, so
    ``arviz.loo`` / ``waic`` can consume the result.

    The ``log_likelihood`` group is laid out over the **real** trials only as a
    single ``__obs__`` axis (participant-major, matching the input row order),
    shape ``(chain, draw, n_real_trials)``.  Padded trials of unbalanced panels
    are *excluded* — folding them in (as logp 0) would make ``arviz.loo`` count
    spurious, perfectly-predicted observations and bias the result.  The group
    is overwritten if it already exists (idempotent re-runs).
    """
    import xarray as xr

    posterior = idata.posterior  # type: ignore[attr-defined]
    N = model.n_participants
    n_chains, n_post = posterior.sizes["chain"], posterior.sizes["draw"]
    lengths = _participant_lengths(model)
    n_obs = int(lengths.sum())

    emission_fn, param_order = _compile_emission_fn(model)

    ll = np.zeros((n_chains, n_post, n_obs))
    for c in range(n_chains):
        for d in range(n_post):
            params = [
                _draw_param_value(model, nm, posterior, c, d) for nm in param_order
            ]
            log_em = np.asarray(emission_fn(*params))  # (N, T, K)
            log_P = np.log(np.asarray(posterior["P"].values[c, d], float) + 1e-300)
            log_pi0 = _log_pi0_for_draw(model, posterior, c, d)
            offset = 0
            for n in range(N):
                Tn = int(lengths[n])
                log_alpha = _forward_filter(log_em[n, :Tn], log_P, log_pi0)
                logZ = logsumexp(log_alpha, axis=1)  # (Tn,)
                delta = np.empty(Tn)
                delta[0] = logZ[0]
                delta[1:] = logZ[1:] - logZ[:-1]
                ll[c, d, offset : offset + Tn] = delta
                offset += Tn

    # Provenance: map each obs back to its (participant, trial) for traceability.
    obs_participant = np.concatenate(
        [np.full(int(lengths[n]), n) for n in range(N)]
    ).astype(int)
    obs_trial = np.concatenate([np.arange(int(lengths[n])) for n in range(N)])

    ds = xr.Dataset(
        {"obs": (("chain", "draw", "__obs__"), ll)},
        coords={
            "chain": posterior.coords["chain"].values,
            "draw": posterior.coords["draw"].values,
            "__obs__": np.arange(n_obs),
            "participant": ("__obs__", obs_participant),
            "trial": ("__obs__", obs_trial),
        },
    )
    if "log_likelihood" in idata.groups():
        del idata.log_likelihood  # type: ignore[attr-defined]
    idata.add_groups({"log_likelihood": ds})
    return idata

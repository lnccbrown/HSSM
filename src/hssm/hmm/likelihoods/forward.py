"""Layer 1: the batched forward recursion (``pytensor.scan``).

Marginalises the discrete regime sequence analytically so that only continuous
parameters remain for NUTS.  A *single* scan over the trial axis advances all
``N`` participants in lockstep (design doc §3.5); under numpyro it JIT-compiles
to ``jax.lax.scan``.  Unbalanced panels are handled by zeroing the emission at
padded steps via the mask, which leaves the running marginal exact.
"""

from __future__ import annotations

import pytensor
import pytensor.tensor as pt


def forward_log_marginal(
    log_emission_lik: pt.TensorVariable,
    log_P: pt.TensorVariable,
    log_pi0: pt.TensorVariable,
    mask: pt.TensorVariable | None = None,
) -> pt.TensorVariable:
    """Joint marginal log-likelihood across participants.

    Parameters
    ----------
    log_emission_lik
        ``(N, T, K)`` per-participant, per-trial, per-regime emission logp.
    log_P
        ``(K, K)`` log transition matrix (``log_P[j, k] = log P(k | j)``).
    log_pi0
        ``(K,)`` log initial-state distribution.
    mask
        Optional ``(N, T)`` mask, 1.0 for real trials and 0.0 for padded
        (unbalanced panels).  When ``None`` every trial is treated as real.

    Returns
    -------
    pt.TensorVariable
        Scalar ``sum_n log p(y_{n,1..T_n} | theta, P, pi0)``.
    """
    if mask is not None:
        # Zero the emission at padded steps; the transition still advances, and
        # because rows of P sum to 1 the marginal is left unchanged.
        log_emission_lik = pt.switch(mask[:, :, None] > 0.5, log_emission_lik, 0.0)

    # Initialise with pi0 * first emission: (N, K).
    log_alpha_init = log_pi0[None, :] + log_emission_lik[:, 0, :]

    def forward_step(log_emission_t, log_alpha_prev, log_P_):
        # log_alpha_prev[:, :, None] : (N, K_prev, 1)
        # log_P_[None, :, :]         : (1, K_prev, K_next)
        # logsumexp over the previous-regime axis (axis=1) marginalises j.
        trans = pt.logsumexp(log_alpha_prev[:, :, None] + log_P_[None, :, :], axis=1)
        return trans + log_emission_t

    # scan iterates over the leading axis -> move trial axis to the front.
    emission_seq = log_emission_lik[:, 1:, :].dimshuffle(1, 0, 2)  # (T-1, N, K)

    log_alphas = pytensor.scan(
        fn=forward_step,
        sequences=[emission_seq],
        outputs_info=[log_alpha_init],
        non_sequences=[log_P],
        return_updates=False,
    )

    # log_alphas: (T-1, N, K); the final step is the full-sequence forward var.
    # (A regime-switching model requires T >= 2; a single-trial panel is
    # degenerate and unsupported.)
    log_alpha_T = log_alphas[-1]

    # Per-participant marginal, then sum across participants.
    return pt.sum(pt.logsumexp(log_alpha_T, axis=1))

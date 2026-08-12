"""Layer 1: the batched forward recursion (``pytensor.scan``).

Marginalises the discrete regime sequence analytically so that only continuous
parameters remain for NUTS.  A *single* scan over the trial axis advances all
``N`` participants in lockstep (design doc §3.5); under numpyro it JIT-compiles
to ``jax.lax.scan``.  Unbalanced panels are handled by zeroing the emission at
padded steps via the mask, which leaves the running marginal exact.

The recursion is the **scaled** (normalised) forward algorithm: the running
log-forward vector ``log_alpha`` is renormalised to ``logsumexp = 0`` at every
trial, and the per-step log-normalisers are accumulated to recover the marginal.
This is mathematically identical to the textbook un-normalised recursion (the
marginal value matches to machine precision) but is numerically essential: the
un-normalised ``log_alpha`` drifts ~linearly with the sequence length, and for
long panels (empirically ``T >= ~400``) the reverse-mode gradient through the
scan becomes NaN — in *both* the C/PyTensor and JAX backends — even though the
forward *value* stays finite.  Normalising keeps ``log_alpha`` O(1) so the
gradient stays finite for arbitrarily long sequences.
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
        # because rows of P sum to 1 the marginal is left unchanged.  Under the
        # scaled recursion a padded step contributes a log-normaliser of exactly
        # 0 (emission == 1 for every regime keeps the normalised alpha's mass at
        # 1), so the marginal stops accumulating after the last real trial.
        log_emission_lik = pt.switch(mask[:, :, None] > 0.5, log_emission_lik, 0.0)

    # Initialise with pi0 * first emission, then split off its log-normaliser so
    # the recurrent state enters the scan already normalised (logsumexp == 0).
    log_alpha_0 = log_pi0[None, :] + log_emission_lik[:, 0, :]  # (N, K)
    log_c_0 = pt.logsumexp(log_alpha_0, axis=1)  # (N,)
    log_alpha_init = log_alpha_0 - log_c_0[:, None]  # (N, K), normalised

    def forward_step(log_emission_t, log_alpha_prev, log_P_):
        # log_alpha_prev[:, :, None] : (N, K_prev, 1)   (normalised: logsumexp 0)
        # log_P_[None, :, :]         : (1, K_prev, K_next)
        # logsumexp over the previous-regime axis (axis=1) marginalises j.
        trans = pt.logsumexp(log_alpha_prev[:, :, None] + log_P_[None, :, :], axis=1)
        log_alpha_t = trans + log_emission_t  # (N, K), un-normalised
        log_c_t = pt.logsumexp(log_alpha_t, axis=1)  # (N,) step log-normaliser
        return log_alpha_t - log_c_t[:, None], log_c_t

    # scan iterates over the leading axis -> move trial axis to the front.
    emission_seq = log_emission_lik[:, 1:, :].dimshuffle(1, 0, 2)  # (T-1, N, K)

    # (A regime-switching model requires T >= 2; a single-trial panel is
    # degenerate and unsupported.)  ``log_alpha`` is recurrent (carries its init);
    # ``log_c`` is collected each step (outputs_info ``None``).
    _, log_c_seq = pytensor.scan(
        fn=forward_step,
        sequences=[emission_seq],
        outputs_info=[log_alpha_init, None],
        non_sequences=[log_P],
        return_updates=False,
    )

    # Per-participant marginal log Z_n = log_c_0[n] + sum_t log_c_t[n].
    # log_c_seq: (T-1, N).  Sum the step normalisers, add the initial one, then
    # sum across participants.
    log_marginal_per_participant = log_c_0 + pt.sum(log_c_seq, axis=0)  # (N,)
    return pt.sum(log_marginal_per_participant)

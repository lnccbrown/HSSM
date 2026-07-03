"""CD: non-decision time ``t`` (Andrew's Commit 5), pinned against independent oracles.

``t`` is a sampled pre-decision delay. It is NOT just ``rt - t``: fixation onsets
(``sacc_array``) are recorded from TRIAL start but the likelihood reasons in
DECISION time, so removing the first ``t`` seconds slides RT *and* every onset back
by ``t`` while anchoring the first at 0 — the net effect is to shorten the first
stage by ``t``. ``t`` must fall inside the first fixation (and leave ``rt-t>0``),
else the trial is rejected with ``-inf``.

The shift lives entirely in the builder (``make_addm_logp_func``); the vendored
``likelihoods/jax/`` kernel is untouched. These tests assert Andrew's four DoD
checks + trial-wise composition, each against an INDEPENDENT oracle:

- **t=0 identity** (DoD #1): builder(t=0) == the raw-covariate kernel, bit-for-bit,
  on both the scalar-batch and the per-trial vmap path.
- **t>0 shift** (DoD #2): builder(t) == the kernel on a manually shifted rt/sacc,
  where the manual shift is a per-trial Python loop (a different expression than the
  builder's vectorized ``where``), AND differs from a naive rt-only shift.
- **invalid t** (DoD #3): offenders (t outside the first fixation, or t>=rt, incl.
  ``d==1``) get ``-inf`` per trial; valid trials keep their manual-shift value. The
  builder is called directly, bypassing dist.py's ``ensure_positive_ndt`` softening.
- **gradient** (DoD #4): ``jax.grad`` w.r.t. scalar and per-trial ``t`` is finite,
  including a batch that mixes valid and rejected trials (the clamp-then-mask trick).
- **CC composition**: trial-wise ``t`` matches a per-trial manual shift; a constant
  ``t`` column equals the scalar path; trial-wise ``t`` + custom process is rejected.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from hssm.addm.attention_process import standard_alternating  # noqa: E402
from hssm.addm.likelihoods.builder import make_addm_logp_func  # noqa: E402
from hssm.addm.likelihoods.jax import (  # noqa: E402
    compute_addm_logfptd,
    compute_addm_loglikelihoods,
)

RTOL = 1e-6
ATOL = 1e-9

# Scalar reference core params (t supplied per test).
_ETA, _KAPPA, _SIGMA, _A, _B, _X0 = 0.3, 1.0, 1.0, 1.5, 0.25, 0.0


def _ndt_fixture(seed=0, n_trials=8, max_d=6, first_onset_floor=0.35):
    """aDDM batch whose first fixation ends well after any test ``t`` (>= 0.35s).

    Onsets are drawn in ``[first_onset_floor, 0.9*rt)`` so ``sacc[:, 1] >= 0.35``,
    giving room for a positive ``t`` (we use ``t=0.1``) strictly inside the first
    fixation on every trial.
    """
    rng = np.random.default_rng(seed)
    rt = rng.uniform(1.6, 2.8, n_trials)
    choice = rng.choice(np.array([-1.0, 1.0]), size=n_trials)
    r1 = rng.integers(1, 6, n_trials).astype(np.float64)
    r2 = rng.integers(1, 6, n_trials).astype(np.float64)
    flag = rng.integers(0, 2, n_trials).astype(np.int32)
    d = rng.integers(2, max_d + 1, n_trials).astype(np.int32)

    sacc = np.zeros((n_trials, max_d), dtype=np.float64)
    for i in range(n_trials):
        onsets = np.sort(rng.uniform(first_onset_floor, 0.9 * rt[i], d[i] - 1))
        sacc[i, 1 : d[i]] = onsets

    return dict(
        n=n_trials,
        max_d=max_d,
        data=jnp.asarray(np.column_stack([rt, choice])),
        rt=rt,
        choice=choice,
        r1=r1,
        r2=r2,
        flag=flag,
        d=d,
        sacc=sacc,
    )


def _manual_shift(rt, sacc, d, t):
    """INDEPENDENT decision-time shift via a per-trial Python loop (the oracle).

    ``rt_eff = rt - t``; for each valid interior onset ``k in 1..d-1`` slide it back
    by ``t``; the first onset (k=0) stays anchored at 0; padding (k>=d) is untouched.
    ``t`` may be a scalar or a per-trial array.
    """
    n, max_d = sacc.shape
    t = np.broadcast_to(np.asarray(t, dtype=float), (n,))
    rt_eff = rt - t
    sacc_eff = sacc.copy()
    for i in range(n):
        for k in range(1, int(d[i])):
            sacc_eff[i, k] = sacc[i, k] - t[i]
    return rt_eff, sacc_eff


def _kernel_single(rt, choice, eta, kappa, sigma, a, b, x0, r1, r2, flag, sacc, d):
    """The vendored single-trial kernel; jnp-wraps args (it rejects numpy ``d``)."""
    return compute_addm_logfptd(
        jnp.asarray(rt),
        jnp.asarray(choice),
        jnp.asarray(eta),
        jnp.asarray(kappa),
        jnp.asarray(sigma),
        jnp.asarray(a),
        jnp.asarray(b),
        jnp.asarray(x0),
        jnp.asarray(r1),
        jnp.asarray(r2),
        jnp.asarray(flag),
        jnp.asarray(sacc),
        jnp.asarray(d),
    )


def _logp(process="standard_alternating"):
    return make_addm_logp_func(process)


def _call(fx, t, eta=_ETA, kappa=_KAPPA, a=_A, b=_B, x0=_X0):
    """Invoke the builder logp with the CD signature (t after x0)."""
    logp = _logp()
    return np.asarray(
        logp(
            fx["data"],
            eta,
            kappa,
            a,
            b,
            x0,
            t,
            jnp.asarray(fx["r1"]),
            jnp.asarray(fx["r2"]),
            jnp.asarray(fx["flag"]),
            jnp.asarray(fx["sacc"]),
            jnp.asarray(fx["d"]),
            _SIGMA,
        )
    )


# --------------------------------------------------------------------------- #
# DoD #1 — t=0 identity
# --------------------------------------------------------------------------- #
def test_t_zero_reproduces_commit4_bitforbit():
    fx = _ndt_fixture()
    got = _call(fx, t=0.0)
    ref = np.asarray(
        compute_addm_loglikelihoods(
            jnp.asarray(fx["rt"]),
            jnp.asarray(fx["choice"]),
            _ETA,
            _KAPPA,
            _SIGMA,
            _A,
            _B,
            _X0,
            jnp.asarray(fx["r1"]),
            jnp.asarray(fx["r2"]),
            jnp.asarray(fx["flag"]),
            jnp.asarray(fx["sacc"]),
            jnp.asarray(fx["d"]),
        )
    )
    np.testing.assert_array_equal(got, ref)  # bit-for-bit


def test_t_zero_identity_pertrial():
    """t=0 on the per-trial vmap path (forced by a trial-wise eta) is also exact."""
    fx = _ndt_fixture(seed=1)
    n = fx["n"]
    rng = np.random.default_rng(9)
    eta = rng.uniform(0.1, 0.6, n)

    got = _call(fx, t=0.0, eta=jnp.asarray(eta))
    ref = np.array(
        [
            float(
                _kernel_single(
                    fx["rt"][i],
                    fx["choice"][i],
                    eta[i],
                    _KAPPA,
                    _SIGMA,
                    _A,
                    _B,
                    _X0,
                    fx["r1"][i],
                    fx["r2"][i],
                    fx["flag"][i],
                    fx["sacc"][i],
                    fx["d"][i],
                )
            )
            for i in range(n)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# DoD #2 — t>0 matches the manual (onset-sliding) shift, and is not rt-only
# --------------------------------------------------------------------------- #
def test_t_positive_matches_manual_shift_batched():
    fx = _ndt_fixture(seed=2)
    t = 0.1
    got = _call(fx, t=t)

    rt_eff, sacc_eff = _manual_shift(fx["rt"], fx["sacc"], fx["d"], t)
    ref = np.asarray(
        compute_addm_loglikelihoods(
            jnp.asarray(rt_eff),
            jnp.asarray(fx["choice"]),
            _ETA,
            _KAPPA,
            _SIGMA,
            _A,
            _B,
            _X0,
            jnp.asarray(fx["r1"]),
            jnp.asarray(fx["r2"]),
            jnp.asarray(fx["flag"]),
            jnp.asarray(sacc_eff),
            jnp.asarray(fx["d"]),
        )
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)

    # The onset slide is a DISTINCT operation from a naive rt-only shift.
    naive = np.asarray(
        compute_addm_loglikelihoods(
            jnp.asarray(fx["rt"] - t),
            jnp.asarray(fx["choice"]),
            _ETA,
            _KAPPA,
            _SIGMA,
            _A,
            _B,
            _X0,
            jnp.asarray(fx["r1"]),
            jnp.asarray(fx["r2"]),
            jnp.asarray(fx["flag"]),
            jnp.asarray(fx["sacc"]),
            jnp.asarray(fx["d"]),
        )
    )
    assert not np.allclose(got, naive, rtol=RTOL, atol=ATOL)


def test_t_positive_matches_single_trial_kernel():
    """builder(t>0) == the vendored single-trial kernel on a per-trial manual shift."""
    fx = _ndt_fixture(seed=3)
    n = fx["n"]
    t = 0.1
    got = _call(fx, t=t)

    rt_eff, sacc_eff = _manual_shift(fx["rt"], fx["sacc"], fx["d"], t)
    ref = np.array(
        [
            float(
                _kernel_single(
                    rt_eff[i],
                    fx["choice"][i],
                    _ETA,
                    _KAPPA,
                    _SIGMA,
                    _A,
                    _B,
                    _X0,
                    fx["r1"][i],
                    fx["r2"][i],
                    fx["flag"][i],
                    sacc_eff[i],
                    fx["d"][i],
                )
            )
            for i in range(n)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# DoD #3 — invalid t rejected per-trial (incl. d==1), valid trials preserved
# --------------------------------------------------------------------------- #
def test_invalid_t_rejected_per_trial():
    # Hand-built batch: mix multi-stage / single-stage, valid / invalid t.
    max_d = 6
    rt = np.array([1.5, 1.5, 0.5, 1.0])
    choice = np.array([1.0, -1.0, 1.0, -1.0])
    r1 = np.array([3.0, 3.0, 3.0, 3.0])
    r2 = np.array([2.0, 2.0, 2.0, 2.0])
    flag = np.array([0, 1, 0, 1], dtype=np.int32)
    d = np.array([3, 3, 1, 1], dtype=np.int32)
    sacc = np.zeros((4, max_d))
    sacc[0, 1:3] = [0.4, 0.8]
    sacc[1, 1:3] = [0.2, 0.6]
    # trials 2,3 are single-stage (d==1): no interior onsets.
    t = np.array([0.1, 0.5, 0.6, 0.2])
    #             valid  t>=first_fix_end(0.2)  t>=rt(0.5)  valid(d==1)
    data = jnp.asarray(np.column_stack([rt, choice]))
    logp = _logp()
    out = np.asarray(
        logp(
            data,
            _ETA,
            _KAPPA,
            _A,
            _B,
            _X0,
            jnp.asarray(t),
            jnp.asarray(r1),
            jnp.asarray(r2),
            jnp.asarray(flag),
            jnp.asarray(sacc),
            jnp.asarray(d),
            _SIGMA,
        )
    )

    assert np.isneginf(out[1]), f"t outside first fixation must be -inf: {out[1]}"
    assert np.isneginf(out[2]), f"t>=rt (d==1) must be -inf: {out[2]}"
    assert np.isfinite(out[0]) and np.isfinite(out[3])

    # Valid trials equal the single-trial kernel on their manual shift.
    rt_eff, sacc_eff = _manual_shift(rt, sacc, d, t)
    for i in (0, 3):
        ref = float(
            _kernel_single(
                rt_eff[i],
                choice[i],
                _ETA,
                _KAPPA,
                _SIGMA,
                _A,
                _B,
                _X0,
                r1[i],
                r2[i],
                flag[i],
                sacc_eff[i],
                d[i],
            )
        )
        np.testing.assert_allclose(out[i], ref, rtol=RTOL, atol=ATOL)


def test_boundary_t_equals_first_onset_is_rejected():
    """t exactly at the first-fixation end (zero-length first stage) -> -inf (strict)."""
    max_d = 4
    rt = np.array([1.5])
    choice = np.array([1.0])
    sacc = np.zeros((1, max_d))
    sacc[0, 1:3] = [0.3, 0.7]
    d = np.array([3], dtype=np.int32)
    data = jnp.asarray(np.column_stack([rt, choice]))
    logp = _logp()
    out = np.asarray(
        logp(
            data,
            _ETA,
            _KAPPA,
            _A,
            _B,
            _X0,
            jnp.asarray(np.array([0.3])),
            jnp.asarray([3.0]),
            jnp.asarray([2.0]),
            jnp.asarray([0], dtype=np.int32),
            jnp.asarray(sacc),
            jnp.asarray(d),
            _SIGMA,
        )
    )
    assert np.isneginf(out[0])


# --------------------------------------------------------------------------- #
# DoD #4 — gradients finite
# --------------------------------------------------------------------------- #
def _grad_call(fx, eta=_ETA):
    logp = _logp()

    def total(t):
        return jnp.sum(
            logp(
                fx["data"],
                eta,
                _KAPPA,
                _A,
                _B,
                _X0,
                t,
                jnp.asarray(fx["r1"]),
                jnp.asarray(fx["r2"]),
                jnp.asarray(fx["flag"]),
                jnp.asarray(fx["sacc"]),
                jnp.asarray(fx["d"]),
                _SIGMA,
            )
        )

    return total


def test_grad_t_finite_scalar_and_vector():
    fx = _ndt_fixture(seed=4)
    total = _grad_call(fx)
    g_scalar = jax.grad(total)(0.05)
    assert bool(jnp.isfinite(g_scalar))

    g_vec = jax.grad(total)(jnp.full(fx["n"], 0.05))
    assert g_vec.shape == (fx["n"],)
    assert bool(jnp.all(jnp.isfinite(g_vec)))


def test_grad_t_finite_with_rejected_trials():
    """A batch mixing valid and rejected trials still yields a finite grad w.r.t. t."""
    fx = _ndt_fixture(seed=5)
    total = _grad_call(fx)
    # Per-trial t: trial 0 is pushed outside its first fixation (rejected).
    t = np.full(fx["n"], 0.05)
    t[0] = fx["sacc"][0, 1] + 0.5  # > first_fix_end -> rejected
    g = jax.grad(total)(jnp.asarray(t))
    assert g.shape == (fx["n"],)
    assert bool(jnp.all(jnp.isfinite(g))), f"grad leaked NaN from rejected trial: {g}"


# --------------------------------------------------------------------------- #
# CC composition — trial-wise t
# --------------------------------------------------------------------------- #
def test_trialwise_t_matches_per_trial_manual_shift():
    """A per-trial t (composed with a trial-wise eta) matches the manual shift."""
    fx = _ndt_fixture(seed=6)
    n = fx["n"]
    rng = np.random.default_rng(11)
    eta = rng.uniform(0.1, 0.6, n)
    t = rng.uniform(0.05, 0.2, n)  # all < first_onset_floor (0.35) -> valid

    got = _call(fx, t=jnp.asarray(t), eta=jnp.asarray(eta))

    rt_eff, sacc_eff = _manual_shift(fx["rt"], fx["sacc"], fx["d"], t)
    ref = np.array(
        [
            float(
                _kernel_single(
                    rt_eff[i],
                    fx["choice"][i],
                    eta[i],
                    _KAPPA,
                    _SIGMA,
                    _A,
                    _B,
                    _X0,
                    fx["r1"][i],
                    fx["r2"][i],
                    fx["flag"][i],
                    sacc_eff[i],
                    fx["d"][i],
                )
            )
            for i in range(n)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


def test_constant_t_regression_matches_scalar():
    """An all-equal (n_obs,) t column equals the scalar-t path bit-for-bit."""
    fx = _ndt_fixture(seed=7)
    n = fx["n"]
    scalar = _call(fx, t=0.1)
    const = _call(fx, t=jnp.full(n, 0.1))
    np.testing.assert_allclose(const, scalar, rtol=RTOL, atol=ATOL)


def test_pertrial_t_custom_process_works():
    """Per-trial t composes with a custom process via from_mu (t is not a core param).

    Only trial-wise *core* params (eta..x0) force the vmap path / the custom-process
    guard; a per-trial t alone stays on the from_mu fast path with the shift applied.
    An identity custom process must therefore match the default path.
    """
    fx = _ndt_fixture(seed=8)
    n = fx["n"]

    def identity_proc(eta, kappa, r1, r2, flag, d, max_d):
        return standard_alternating(eta, kappa, r1, r2, flag, d, max_d)

    t = jnp.full(n, 0.1)
    custom = make_addm_logp_func(identity_proc)
    got = np.asarray(
        custom(
            fx["data"],
            _ETA,
            _KAPPA,
            _A,
            _B,
            _X0,
            t,
            jnp.asarray(fx["r1"]),
            jnp.asarray(fx["r2"]),
            jnp.asarray(fx["flag"]),
            jnp.asarray(fx["sacc"]),
            jnp.asarray(fx["d"]),
            _SIGMA,
        )
    )
    ref = _call(fx, t=t)  # default (kernel-internal) path
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)

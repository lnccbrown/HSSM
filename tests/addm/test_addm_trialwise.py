"""CC: trial-wise / regression core aDDM params must flow through the kernel.

The headline aDDM capability: a regression or hierarchical prior on a core
parameter (``eta, kappa, a, b, x0``) makes that parameter vary *per trial*, so
it reaches the log-likelihood ``Op`` as an ``(n_obs,)`` array (bambi produces the
per-trial linear predictor; dist.py's trial-wise broadcast guarantees the shape).

These tests pin the builder's ``logp`` closure directly (no sampling) against two
trusted oracles, exactly the RED->GREEN discipline the plan calls for:

1. **Constant regression == scalar path (bit-for-bit).** An ``(n_obs,)`` column
   whose entries are all equal must give the same log-likelihood as passing the
   scalar — isolating the vmap *wiring* from the FPT math.
2. **Genuinely per-trial params == single-trial kernel, trial by trial.** With
   values that actually vary across trials, the batched ``logp`` must equal the
   vendored single-trial kernel ``compute_addm_logfptd`` evaluated one trial at a
   time with that trial's own parameter values (aDDM trials are conditionally
   independent, so this composition is a valid oracle).
3. **Gradient finite.** ``jax.grad`` of the summed logp w.r.t. a per-trial
   parameter array is finite and correctly shaped (NUTS needs it).

On the pre-CC scalar-freeze kernel these fail (the scalar batched kernel mis-
shapes an ``(n_obs,)`` core param); they pass only once the builder maps per-trial
params with ``in_axes=0``.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from hssm.addm.likelihoods.builder import make_addm_logp_func  # noqa: E402
from hssm.addm.likelihoods.jax import compute_addm_logfptd  # noqa: E402

RTOL = 1e-6
ATOL = 1e-9


def _fixture(seed=0, n_trials=7, max_d=6):
    """Small hand-built aDDM batch (rt/choice + per-trial fixation covariates)."""
    rng = np.random.default_rng(seed)
    r1 = rng.integers(1, 6, n_trials).astype(np.float64)
    r2 = rng.integers(1, 6, n_trials).astype(np.float64)
    flag = rng.integers(0, 2, n_trials).astype(np.int32)
    d = rng.integers(2, max_d + 1, n_trials).astype(np.int32)
    rt = rng.uniform(0.8, 2.0, n_trials)
    choice = rng.choice(np.array([-1.0, 1.0]), size=n_trials)

    sacc = np.zeros((n_trials, max_d), dtype=np.float64)
    for i in range(n_trials):
        onsets = np.sort(rng.uniform(0.0, rt[i], d[i] - 1))
        sacc[i, 1 : d[i]] = onsets

    data = jnp.asarray(np.column_stack([rt, choice]))
    return dict(
        n=n_trials,
        data=data,
        rt=jnp.asarray(rt),
        choice=jnp.asarray(choice),
        r1=jnp.asarray(r1),
        r2=jnp.asarray(r2),
        flag=jnp.asarray(flag),
        sacc=jnp.asarray(sacc),
        d=jnp.asarray(d),
    )


# Scalar reference parameters.
_ETA, _KAPPA, _SIGMA, _A, _B, _X0 = 0.3, 1.0, 1.0, 1.5, 0.25, 0.0


def _logp():
    return make_addm_logp_func("standard_alternating")


def _call(logp, fx, eta, kappa, a, b, x0, sigma=_SIGMA):
    return np.asarray(
        logp(
            fx["data"],
            eta,
            kappa,
            a,
            b,
            x0,
            fx["r1"],
            fx["r2"],
            fx["flag"],
            fx["sacc"],
            fx["d"],
            sigma,
        )
    )


def test_constant_regression_matches_scalar():
    """An all-equal (n_obs,) column equals the scalar path bit-for-bit."""
    fx = _fixture()
    logp = _logp()
    n = fx["n"]

    scalar_ll = _call(logp, fx, _ETA, _KAPPA, _A, _B, _X0)
    const_ll = _call(
        logp,
        fx,
        jnp.full(n, _ETA),
        jnp.full(n, _KAPPA),
        jnp.full(n, _A),
        jnp.full(n, _B),
        jnp.full(n, _X0),
    )
    np.testing.assert_allclose(const_ll, scalar_ll, rtol=RTOL, atol=ATOL)


def test_pertrial_params_match_single_trial_kernel():
    """Genuinely-varying per-trial params == the single-trial kernel per trial."""
    fx = _fixture(seed=1)
    logp = _logp()
    n = fx["n"]
    rng = np.random.default_rng(2)

    eta = jnp.asarray(rng.uniform(0.1, 0.6, n))
    kappa = jnp.asarray(rng.uniform(0.6, 1.4, n))
    a = jnp.asarray(rng.uniform(1.0, 2.0, n))
    b = jnp.asarray(rng.uniform(0.05, 0.4, n))
    x0 = jnp.asarray(rng.uniform(-0.3, 0.3, n))

    got = _call(logp, fx, eta, kappa, a, b, x0)

    ref = np.array(
        [
            float(
                compute_addm_logfptd(
                    fx["rt"][i],
                    fx["choice"][i],
                    eta[i],
                    kappa[i],
                    _SIGMA,
                    a[i],
                    b[i],
                    x0[i],
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


def test_mixed_scalar_and_trialwise():
    """Some params per-trial, some scalar — mixed in_axes must resolve correctly."""
    fx = _fixture(seed=3)
    logp = _logp()
    n = fx["n"]
    rng = np.random.default_rng(4)

    eta = jnp.asarray(rng.uniform(0.1, 0.6, n))  # trial-wise
    a = jnp.asarray(rng.uniform(1.0, 2.0, n))  # trial-wise
    # kappa, b, x0 scalar
    got = _call(logp, fx, eta, _KAPPA, a, _B, _X0)

    ref = np.array(
        [
            float(
                compute_addm_logfptd(
                    fx["rt"][i],
                    fx["choice"][i],
                    eta[i],
                    _KAPPA,
                    _SIGMA,
                    a[i],
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


def test_trialwise_grad_finite():
    """Grad of the summed logp w.r.t. a per-trial param is finite and shaped (n,)."""
    fx = _fixture(seed=5)
    logp = _logp()
    n = fx["n"]

    def total(eta_vec):
        return jnp.sum(
            logp(
                fx["data"],
                eta_vec,
                _KAPPA,
                _A,
                _B,
                _X0,
                fx["r1"],
                fx["r2"],
                fx["flag"],
                fx["sacc"],
                fx["d"],
                _SIGMA,
            )
        )

    g = jax.grad(total)(jnp.full(n, _ETA))
    assert g.shape == (n,)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_custom_process_with_trialwise_is_rejected():
    """A custom attention process + per-trial core params raises (documented gap)."""
    fx = _fixture(seed=6)
    n = fx["n"]

    def _identity_process(eta, kappa, r1, r2, flag, d, max_d):  # pragma: no cover
        raise AssertionError("should not be reached")

    logp = make_addm_logp_func(_identity_process)
    with pytest.raises(NotImplementedError):
        logp(
            fx["data"],
            jnp.full(n, _ETA),
            _KAPPA,
            _A,
            _B,
            _X0,
            fx["r1"],
            fx["r2"],
            fx["flag"],
            fx["sacc"],
            fx["d"],
            _SIGMA,
        )

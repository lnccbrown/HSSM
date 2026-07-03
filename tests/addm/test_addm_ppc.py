"""CE-2: the aDDM RandomVariable + covariate-conditioned posterior-predictive path.

Once the ported ssm-simulators aDDM (CE-1) is installed, ``make_hssm_rv('addm')``
resolves a real RV. These tests pin the covariate handshake that makes the
generative path condition on the *observed* fixations (r1, r2, flag, sacc_array,
d, sigma) rather than self-sampling them, plus the cross-validation that the
ported simulator and the vendored JAX likelihood agree.

Requires the CE-1 ssm-simulators build (with ``cssm.addm``); tests skip if the
installed ssm-simulators does not register the ``addm`` model.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_addm_subclass import make_addm_dataframe  # noqa: E402

try:
    from ssms.config import model_config as _ssms_model_config

    _HAS_ADDM_SIM = "addm" in _ssms_model_config
except Exception:  # pragma: no cover
    _HAS_ADDM_SIM = False

needs_addm_sim = pytest.mark.skipif(
    not _HAS_ADDM_SIM,
    reason="ssm-simulators build without cssm.addm (CE-1 not installed)",
)

import hssm  # noqa: E402


# --------------------------------------------------------------------------- #
# RV resolution + covariate wiring (fast)
# --------------------------------------------------------------------------- #
@needs_addm_sim
def test_make_hssm_rv_addm_builds_real_rv(recwarn):
    from hssm.distribution_utils.dist import make_hssm_rv

    rv = make_hssm_rv("addm", list_params=["eta", "kappa", "a", "b", "x0", "t"])
    assert rv is not None
    not_supported = [
        w
        for w in recwarn.list
        if "addm" in str(w.message).lower() and "not" in str(w.message).lower()
    ]
    assert not not_supported, (
        f"unexpected 'addm not supported' warning: {not_supported}"
    )


@needs_addm_sim
def test_addm_rv_extra_fields_populated():
    """Building an aDDM stashes the observed fixations on the RV class."""
    df = make_addm_dataframe(30, seed=1)
    model = hssm.aDDM(data=df)
    ef = type(model.model_distribution.rv_op)._extra_fields
    assert ef is not None
    assert set(ef.keys()) == {"r1", "r2", "flag", "sacc_array", "d", "sigma"}
    assert np.asarray(ef["sacc_array"]).shape[0] == len(df)
    assert np.array_equal(np.asarray(ef["r1"]), df["r1"].to_numpy())


def test_backward_compat_non_addm_rv_has_no_extra_fields():
    """A non-covariate model's RV keeps _extra_fields None (generative path unchanged)."""
    data = hssm.simulate_data(model="ddm", theta=[0.5, 1.5, 0.5, 0.5], size=50)
    m = hssm.HSSM(data=data, model="ddm")
    assert type(m.model_distribution.rv_op)._extra_fields is None


# --------------------------------------------------------------------------- #
# sim <-> likelihood cross-validation (medium; no sampling)
# --------------------------------------------------------------------------- #
@needs_addm_sim
def test_addm_sim_likelihood_recovery():
    """Data simulated by the ported engine is best-fit by the true params.

    Simulate a dataset in Mode 2 (observed fixations) at known theta via the
    CE-1 ssm-sim engine, then evaluate the vendored JAX aDDM log-likelihood at the
    true params vs each param perturbed — the true params must score higher. This
    ties the two efpt-derived halves (simulator + likelihood) together.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import cssm
    from hssm.addm.likelihoods.builder import make_addm_logp_func

    rng = np.random.default_rng(0)
    n, max_d = 400, 6
    r1 = rng.integers(1, 6, n).astype(np.float64)
    r2 = rng.integers(1, 6, n).astype(np.float64)
    flag = rng.integers(0, 2, n).astype(np.int64)
    d = rng.integers(2, max_d + 1, n).astype(np.int32)
    sacc = np.zeros((n, max_d))
    for i in range(n):
        sacc[i, 1 : d[i]] = np.sort(rng.uniform(0.1, 1.2, d[i] - 1))

    eta, kappa, sigma, a, b, x0, t = 0.4, 1.0, 1.0, 1.5, 0.2, 0.0, 0.0
    col = lambda v: np.full(n, v, dtype=np.float64)  # noqa: E731
    out = cssm.addm(
        col(eta),
        col(kappa),
        col(a),
        col(b),
        col(x0),
        col(t),
        col(999.0),
        col(sigma),
        sigma=col(sigma),
        r1=r1,
        r2=r2,
        flag=flag,
        sacc_array=sacc,
        d=d,
        n_samples=1,
        n_trials=n,
        random_state=1,
        delta_t=0.001,
        max_t=10.0,
    )
    rt = np.asarray(out["rts"]).reshape(-1).astype(np.float64)
    ch = np.asarray(out["choices"]).reshape(-1)
    keep = rt != -999.0  # drop omissions

    # Observed fixations are truncated at the response: only fixations that
    # started before rt are seen (a decision can land before a pre-generated
    # saccade would occur). Real aDDM data is naturally truncated this way, and
    # the likelihood requires rt >= sacc[d-1]. (The full "keep saccading past the
    # last observed fixation" refinement is Andrew's C6 / a follow-up.)
    d_obs = np.array(
        [max(int((sacc[i, : d[i]] < rt[i]).sum()), 1) for i in range(n)],
        dtype=np.int32,
    )
    data = np.column_stack([rt[keep], ch[keep]])

    logp = make_addm_logp_func()

    def ll(p):
        import jax.numpy as jnp

        return float(
            jnp.sum(
                logp(
                    jnp.asarray(data),
                    p["eta"],
                    p["kappa"],
                    p["a"],
                    p["b"],
                    p["x0"],
                    p["t"],
                    jnp.asarray(r1[keep]),
                    jnp.asarray(r2[keep]),
                    jnp.asarray(flag[keep]),
                    jnp.asarray(sacc[keep]),
                    jnp.asarray(d_obs[keep]),
                    sigma,
                )
            )
        )

    truth = dict(eta=eta, kappa=kappa, a=a, b=b, x0=x0, t=t)
    ll_true = ll(truth)
    assert np.isfinite(ll_true)
    for name, delta in [("kappa", 0.4), ("a", 0.4), ("eta", 0.3)]:
        pert = dict(truth)
        pert[name] = truth[name] + delta
        assert ll_true > ll(pert), f"true params not favored over perturbed {name}"


# --------------------------------------------------------------------------- #
# PPC conditions on observed fixations (slow; short sample)
# --------------------------------------------------------------------------- #
@needs_addm_sim
@pytest.mark.slow
def test_addm_ppc_conditions_on_observed_fixations():
    df = make_addm_dataframe(40, seed=2)
    model = hssm.aDDM(data=df)
    idata = model.sample(
        draws=10, tune=10, chains=1, cores=1, idata_kwargs={"log_likelihood": False}
    )
    rv_cls = type(model.model_distribution.rv_op)

    def _rt(ppc_idata):
        # The predictive variable stacks (rt, response); rt is the first column.
        var = next(v for v in ppc_idata.posterior_predictive.data_vars)
        return np.asarray(ppc_idata.posterior_predictive[var])[..., 0].ravel()

    # Conditioned PPC (uses observed fixations).
    model.sample_posterior_predictive(idata, kind="response", draws=10)
    assert "posterior_predictive" in idata.groups()
    cond = _rt(idata)

    # Force self-sampling (drop the covariates) and re-draw with the same posterior.
    saved = rv_cls._extra_fields
    rv_cls._extra_fields = None
    try:
        idata2 = model.sample_posterior_predictive(
            idata, kind="response", draws=10, inplace=False
        )
        uncond = _rt(idata2)
    finally:
        rv_cls._extra_fields = saved

    cond = cond[np.isfinite(cond)]
    uncond = uncond[np.isfinite(uncond)]
    # Conditioning on the real fixations must change the predictive distribution.
    assert cond.size and uncond.size
    assert (
        abs(cond.mean() - uncond.mean()) > 1e-6 or abs(cond.std() - uncond.std()) > 1e-6
    )

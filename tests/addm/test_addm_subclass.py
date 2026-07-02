"""Commit 4 — verify the aDDM(HSSMBase) subclass end to end.

Runs in the repo's uv venv (full hssm stack)::

    cd .../HSSM && source .venv/bin/activate
    JAX_PLATFORMS=cpu python src/hssm/addm/commit_tests/test_commit4_subclass.py
    # or: pytest src/hssm/addm/commit_tests/test_commit4_subclass.py -v

The synthetic data is built directly with numpy (no efficient_fpt dependency).
``rt`` is drawn in-support for the default params (with a=1.0, b=2.0 the bound
collapses near t≈a/b=0.5), so the init log-likelihood is finite.
"""

import arviz as az
import numpy as np
import pandas as pd
from pytensor.graph import Op

import hssm
from hssm.addm.addm import aDDM
from hssm.addm.config import aDDMConfig
from hssm.addm.likelihoods.builder import make_addm_logp_func
from hssm.addm.likelihoods.jax import set_jax_precision
from hssm.base import HSSMBase

set_jax_precision(True)

MAX_D = 6


def make_addm_dataframe(n_trials, seed=0, n_participants=1):
    """Synthetic aDDM dataset with a per-row ``sacc_array`` (object) column."""
    rng = np.random.default_rng(seed)
    rt = rng.uniform(0.05, 0.45, n_trials)  # in-support for default b=2.0
    response = rng.choice([-1.0, 1.0], n_trials)
    r1 = rng.integers(1, 6, n_trials).astype(float)
    r2 = rng.integers(1, 6, n_trials).astype(float)
    flag = rng.integers(0, 2, n_trials).astype(int)
    d = rng.integers(2, MAX_D + 1, n_trials).astype(int)
    sigma = np.ones(n_trials)

    sacc = []
    for i in range(n_trials):
        onsets = np.sort(rng.uniform(0.0, rt[i], d[i] - 1))
        row = np.zeros(MAX_D)
        row[1 : d[i]] = onsets  # row[0] = 0.0 (stage-0 onset)
        sacc.append(row)

    participant = (
        rng.integers(0, n_participants, n_trials)
        if n_participants > 1
        else np.zeros(n_trials, dtype=int)
    )

    df = pd.DataFrame(
        {
            "rt": rt,
            "response": response,
            "r1": r1,
            "r2": r2,
            "flag": flag,
            "d": d,
            "sigma": sigma,
            "participant": participant,
        }
    )
    df["sacc_array"] = pd.Series(sacc, index=df.index)  # object column of arrays
    return df


def _assert_raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} to be raised")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_construct_default_config():
    df = make_addm_dataframe(20)
    model = hssm.aDDM(data=df)
    assert isinstance(model, aDDM)


def test_construct_explicit_config():
    df = make_addm_dataframe(20)
    model = hssm.aDDM(data=df, model_config=aDDMConfig())
    assert isinstance(model, aDDM)


def test_is_hssmbase_subclass():
    assert issubclass(hssm.aDDM, HSSMBase)
    model = hssm.aDDM(data=make_addm_dataframe(20))
    for attr in ("sample", "bounds", "model"):
        assert hasattr(model, attr), f"missing HSSMBase API: {attr}"


def test_loglik_op_injected():
    cfg = aDDMConfig()
    assert cfg.loglik is None and cfg.backend is None
    model = hssm.aDDM(data=make_addm_dataframe(20), model_config=cfg)
    assert model.model_config.loglik is not None
    assert isinstance(model.model_config.loglik, Op)
    assert model.model_config.backend == "jax"
    # Caller's config object must be untouched by dataclasses.replace.
    assert cfg.loglik is None and cfg.backend is None


def test_bad_columns_raise():
    base = make_addm_dataframe(20)

    bad_flag = base.copy()
    bad_flag.loc[0, "flag"] = 2
    _assert_raises(lambda: hssm.aDDM(data=bad_flag))

    bad_d = base.copy()
    bad_d.loc[0, "d"] = MAX_D + 1
    _assert_raises(lambda: hssm.aDDM(data=bad_d))

    missing_col = base.drop(columns=["r1"])
    _assert_raises(lambda: hssm.aDDM(data=missing_col))


def test_smoke_sample():
    df = make_addm_dataframe(200, seed=1)

    # Localize failures: confirm the init log-likelihood is finite before NUTS.
    eta, kappa, a, b, x0 = aDDMConfig().params_default
    data_arr = np.column_stack([df["rt"].to_numpy(), df["response"].to_numpy()])
    sacc2d = aDDM._stack_sacc_array(df["sacc_array"])
    logp = make_addm_logp_func()(
        data_arr, eta, kappa, a, b, x0,
        df["r1"].to_numpy(), df["r2"].to_numpy(), df["flag"].to_numpy(),
        sacc2d, df["d"].to_numpy(), df["sigma"].to_numpy(),
    )
    assert bool(np.all(np.isfinite(np.asarray(logp)))), "init logp not finite"

    model = hssm.aDDM(data=df)
    # log_likelihood is disabled: the post-hoc pointwise log-likelihood (for
    # WAIC/LOO) re-evaluates the Op with a draw dimension on the params, which
    # the scalar-param aDDM kernel does not yet support.
    idata = model.sample(
        draws=5, tune=5, chains=1, cores=1,
        idata_kwargs={"log_likelihood": False},
    )
    assert isinstance(idata, az.InferenceData)
    assert "posterior" in idata.groups()


def test_hierarchical_regression_builds():
    df = make_addm_dataframe(60, seed=2, n_participants=3)
    model = hssm.aDDM(
        data=df,
        include=[{"name": "eta", "formula": "eta ~ 1 + (1|participant)"}],
    )
    assert model.model is not None  # bambi model built without error


if __name__ == "__main__":
    for fn in (
        test_construct_default_config,
        test_construct_explicit_config,
        test_is_hssmbase_subclass,
        test_loglik_op_injected,
        test_bad_columns_raise,
        test_smoke_sample,
        test_hierarchical_regression_builds,
    ):
        fn()
        print(f"PASSED: {fn.__name__}")
    print("\nAll Commit 4 subclass checks passed.")

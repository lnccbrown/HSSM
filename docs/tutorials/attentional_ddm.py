"""Attentional Drift Diffusion Model (aDDM) tutorial — marimo notebook.

Source of truth (marimo, per the ecosystem convention). It demonstrates the aDDM
end to end: simulate a dataset with per-trial fixations, fit a **trial-wise
regression** on the attention parameter ``eta``, recover the coefficients, and run
a posterior-predictive check conditioned on the observed fixations.

Requires the aDDM build of ssm-simulators (with ``cssm.addm``). Until that ships on
PyPI, run against a local build::

    uv pip install -e ../ssm-simulators           # the aDDM build
    uv run marimo edit docs/tutorials/attentional_ddm.py

To publish into the docs once ssm-simulators ships the aDDM model, export with
outputs and wire into mkdocs (see the marimo-notebooks skill)::

    uv run marimo export ipynb docs/tutorials/attentional_ddm.py \
        -o docs/tutorials/attentional_ddm.ipynb --include-outputs
"""

import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

    import marimo as mo
    import numpy as np
    import pandas as pd

    import hssm

    hssm.set_floatX("float64")
    return hssm, mo, np, pd


@app.cell
def _(mo):
    mo.md("""
    # Attentional DDM (aDDM)

    The aDDM extends the DDM with **gaze-dependent drift**: while a decision
    maker fixates one item, evidence for that item accumulates faster. Each
    trial carries its own fixation sequence as covariates
    (`r1, r2, flag, sacc_array, d, sigma`), and the core parameters
    `eta, kappa, a, b, x0` (plus a non-decision time `t`) can be **regressed**
    on trial-level predictors — the focus of this tutorial.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Simulate a dataset with a trial-wise `eta`
    """)
    return


@app.cell
def _(np, pd):
    import cssm  # the aDDM build of ssm-simulators

    rng = np.random.default_rng(0)
    n, max_d = 800, 8

    # A per-trial predictor `x` drives eta: eta_i = 0.2 + 0.3 * x_i (the target we
    # will recover). Everything else is shared across trials.
    x = rng.uniform(0.0, 1.0, n)
    eta_true = 0.2 + 0.3 * x
    kappa, sigma, a, b, x0, t = 1.0, 1.0, 1.5, 0.2, 0.0, 0.0

    r1 = rng.integers(1, 6, n).astype(np.float64)
    r2 = rng.integers(1, 6, n).astype(np.float64)
    flag = rng.integers(0, 2, n).astype(np.int64)
    d = rng.integers(2, max_d + 1, n).astype(np.int32)
    sacc = np.zeros((n, max_d))
    for i in range(n):
        sacc[i, 1 : d[i]] = np.sort(rng.uniform(0.1, 1.2, d[i] - 1))

    def _c(v):
        return np.full(n, v, dtype=np.float64)

    out = cssm.addm(
        eta_true, _c(kappa), _c(a), _c(b), _c(x0), _c(t), _c(999.0), _c(sigma),
        sigma=_c(sigma), r1=r1, r2=r2, flag=flag, sacc_array=sacc, d=d,
        n_samples=1, n_trials=n, random_state=1, delta_t=0.001, max_t=10.0,
    )
    rt = np.asarray(out["rts"]).reshape(-1).astype(np.float64)
    ch = np.asarray(out["choices"]).reshape(-1).astype(np.float64)
    keep = rt != -999.0
    rows = np.flatnonzero(keep)

    # Observed fixations are truncated at the response (you only see fixations that
    # started before the decision; the likelihood needs rt >= sacc[d-1]).
    d_obs = np.array([max(int((sacc[i, : d[i]] < rt[i]).sum()), 1) for i in rows])
    data = pd.DataFrame({
        "rt": rt[keep], "response": ch[keep], "x": x[keep],
        "r1": r1[keep], "r2": r2[keep], "flag": flag[keep].astype(int),
        "d": d_obs, "sigma": np.full(rows.size, sigma),
    })
    data["sacc_array"] = pd.Series([sacc[i].copy() for i in rows], index=data.index)
    data.head()
    return (data,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Fit a trial-wise regression on `eta`

    `include=[{"name": "eta", "formula": "eta ~ 1 + x"}]` makes `eta` vary per
    trial as a linear function of `x`. The remaining core parameters stay
    scalar. This trial-wise / hierarchical support is the headline aDDM
    capability.
    """)
    return


@app.cell
def _(data, hssm):
    model = hssm.aDDM(
        data=data,
        include=[{"name": "eta", "formula": "eta ~ 1 + x"}],
    )
    model
    return (model,)


@app.cell
def _(model):
    idata = model.sample(
        draws=500, tune=500, chains=2, cores=1,
        idata_kwargs={"log_likelihood": False},
    )
    return (idata,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Recover the regression coefficients

    The posterior for `eta ~ 1 + x` should center on the truth
    `Intercept ≈ 0.2`, `x ≈ 0.3`.
    """)
    return


@app.cell
def _(idata):
    import arviz as az

    az.summary(
        idata,
        var_names=["eta", "kappa", "a", "b", "x0", "t"],
        filter_vars="like",  # matches eta_Intercept / eta_x from the regression
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Posterior predictive, conditioned on the observed fixations

    `sample_posterior_predictive` draws new (rt, choice) from the posterior
    **using each trial's real fixation sequence** — the aDDM simulator is fed
    the observed `r1, r2, flag, sacc_array, d`, so the check is faithful to the
    gaze pattern that produced the data.
    """)
    return


@app.cell
def _(idata, model):
    model.sample_posterior_predictive(idata, kind="response", draws=100)
    idata.posterior_predictive
    return


if __name__ == "__main__":
    app.run()

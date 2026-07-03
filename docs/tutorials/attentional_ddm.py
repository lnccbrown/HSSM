"""Attentional Drift Diffusion Model (aDDM) tutorial — marimo notebook.

Source of truth (marimo, per the ecosystem convention). It lets you (1) poke the
aDDM **simulator** interactively, (2) *see* the covariate **handshake** that makes
posterior-predictive checks condition on the observed fixations, and (3) fit a
**trial-wise regression** on the attention parameter ``eta`` and recover it.

The simulator is driven through ssm-simulators' high-level ``Simulator`` class —
the same public path HSSM's posterior-predictive checks go through — so the
covariate handshake you see here is exactly the one used under the hood.

Requires the aDDM build of ssm-simulators (with ``cssm.addm``) and marimo. Until
that ships on PyPI, run against the local build (from the HSSM worktree)::

    uv pip install -e ../../ssm-simulators/addm-sim --no-deps --force-reinstall
    uv pip install marimo
    uv run --no-sync marimo edit docs/tutorials/attentional_ddm.py

``--no-sync`` is important: a plain ``uv run`` re-syncs HSSM's pinned
ssm-simulators and would undo the local aDDM build. Sampling is gated behind a
button, so the simulator/handshake sections are instant on load.

To publish into the docs once ssm-simulators ships the aDDM model, export with
outputs and wire into mkdocs (see the marimo-notebooks skill)::

    uv run marimo export ipynb docs/tutorials/attentional_ddm.py \
        -o docs/tutorials/attentional_ddm.ipynb --include-outputs
"""

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from ssms.basic_simulators import Simulator

    import hssm

    hssm.set_floatX("float64")
    return Simulator, hssm, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Attentional DDM (aDDM)

    The aDDM extends the DDM with **gaze-dependent drift**: while a decision maker
    fixates one item, evidence for that item accumulates faster. Each trial carries
    its own fixation sequence as covariates (`r1, r2, flag, sacc_array, d, sigma`),
    and the core parameters `eta, kappa, a, b, x0` (+ a non-decision time `t`) can
    be **regressed** on trial-level predictors.

    This notebook: **(1)** poke the simulator, **(2)** see the covariate handshake,
    **(3)** fit a trial-wise regression and recover it. The simulator is driven via
    ssm-simulators' high-level `Simulator` class — the same path HSSM's PPC uses.
    """)
    return


@app.cell
def _(Simulator, np):
    sim = Simulator("addm")  # configure the aDDM simulator once (params [eta,kappa,a,b,x0,t])

    def make_fixations(n, max_d=8, seed=0):
        """A batch of observed fixation sequences (the aDDM covariates)."""
        rng = np.random.default_rng(seed)
        r1 = rng.integers(1, 6, n).astype(np.float64)
        r2 = rng.integers(1, 6, n).astype(np.float64)
        flag = rng.integers(0, 2, n).astype(np.int64)
        d = rng.integers(2, max_d + 1, n).astype(np.int32)
        sacc = np.zeros((n, max_d))
        for i in range(n):
            sacc[i, 1 : d[i]] = np.sort(rng.uniform(0.1, 1.2, d[i] - 1))
        return dict(n=n, r1=r1, r2=r2, flag=flag, d=d, sacc=sacc)

    def extra_fields(fixations):
        """Pack fixations into the extra_fields dict the simulator conditions on."""
        n = fixations["n"]
        return {
            "r1": fixations["r1"], "r2": fixations["r2"], "flag": fixations["flag"],
            "sacc_array": fixations["sacc"], "d": fixations["d"], "sigma": np.ones(n),
        }

    fix = make_fixations(3000, seed=0)
    return extra_fields, fix, make_fixations, sim


@app.cell
def _(mo):
    mo.md("""
    ## 1. Explore the simulator

    Move the sliders to change the aDDM parameters and watch the reaction-time
    distribution (split by choice) and the choice proportion update. `eta` is the
    attentional discount (how strongly the *unattended* item is down-weighted),
    `kappa` scales the drift, `a` is the boundary height, and `b` the collapse rate.
    """)
    return


@app.cell
def _(mo):
    eta_s = mo.ui.slider(0.0, 1.0, value=0.3, step=0.05, label="eta (attention)")
    kappa_s = mo.ui.slider(0.1, 3.0, value=1.0, step=0.1, label="kappa (drift scale)")
    a_s = mo.ui.slider(0.5, 3.0, value=1.5, step=0.1, label="a (boundary)")
    b_s = mo.ui.slider(0.0, 1.0, value=0.2, step=0.05, label="b (collapse)")
    mo.vstack([eta_s, kappa_s, a_s, b_s])
    return a_s, b_s, eta_s, kappa_s


@app.cell
def _(a_s, b_s, eta_s, extra_fields, fix, kappa_s, np, plt, sim):
    def simulate(eta, kappa, a, b, fixations, seed=1):
        # theta is (n_trials, 6) = [eta, kappa, a, b, x0, t]; one row per fixation
        # trial (all sharing these params here), n_samples=1 draw each.
        n = fixations["n"]
        theta = np.tile([eta, kappa, a, b, 0.0, 0.0], (n, 1))
        out = sim.simulate(
            theta=theta, n_samples=1, random_state=seed, max_t=10.0,
            extra_fields=extra_fields(fixations),  # Mode 2: condition on fixations
        )
        rt = np.asarray(out["rts"]).reshape(-1)
        ch = np.asarray(out["choices"]).reshape(-1)
        keep = rt != -999.0
        return rt[keep], ch[keep]

    _rt, _ch = simulate(eta_s.value, kappa_s.value, a_s.value, b_s.value, fix)
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.hist(_rt[_ch == 1], bins=40, alpha=0.6, label="choice +1 (upper)")
    _ax.hist(_rt[_ch == -1], bins=40, alpha=0.6, label="choice -1 (lower)")
    _ax.set_xlabel("reaction time (s)")
    _ax.set_ylabel("count")
    _ax.set_title(f"P(+1) = {np.mean(_ch == 1):.3f}   (n = {_rt.size})")
    _ax.legend()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The covariate handshake

    An aDDM prediction depends on the **fixation sequence**, so a posterior- (or
    prior-) predictive check must reuse the *observed* gaze pattern, not invent a
    new one. The simulator supports two modes:

    * **Mode 2 (conditioned)** — pass `extra_fields` with the observed `r1, r2,
      flag, sacc_array, d`; only the trajectory is sampled.
    * **Mode 1 (self-sampled)** — omit `extra_fields`, so it samples its own.

    Below, the same `Simulator` runs both ways. The distributions differ — that
    difference is exactly what the PPC handshake preserves.
    """)
    return


@app.cell
def _(extra_fields, fix, np, plt, sim):
    _theta = np.tile([0.3, 1.5, 1.5, 0.2, 0.0, 0.0], (fix["n"], 1))

    # Mode 2: pass the observed fixations via extra_fields.
    _cond = sim.simulate(
        theta=_theta, n_samples=1, random_state=7, max_t=10.0,
        extra_fields=extra_fields(fix),
    )
    # Mode 1: same params + seed, but no extra_fields -> self-sampled fixations.
    _self = sim.simulate(theta=_theta, n_samples=1, random_state=7, max_t=10.0)

    def _valid(out):
        rt = np.asarray(out["rts"]).reshape(-1)
        return rt[rt != -999.0]

    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.hist(_valid(_cond), bins=40, alpha=0.6, density=True, label="Mode 2 (observed fixations)")
    _ax.hist(_valid(_self), bins=40, alpha=0.6, density=True, label="Mode 1 (self-sampled)")
    _ax.set_xlabel("reaction time (s)")
    _ax.set_ylabel("density")
    _ax.set_title("Conditioning on the observed fixations changes the prediction")
    _ax.legend()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ### How the handshake is wired

    The likelihood already receives the fixations (as `extra_fields`); the
    *generative* path did not, so it used to self-sample. The fix threads the
    observed covariates through, end to end — the last two steps are the very
    `Simulator`/`simulator()` API used above:

    ```
    aDDM._make_model_distribution / _update_extra_fields   (HSSM)
        └─ stashes {r1,r2,flag,sacc_array,d,sigma} on the RV class: rv_op._extra_fields
    HSSMRV.rng_fn        (HSSM)            forwards _extra_fields ->
    ssms_rng_fn          (ssm-simulators) broadcasts to (draws x trials) ->
    simulator(..., extra_fields=dict)     splats into ->   (Simulator.simulate too)
    cssm.addm(..., sacc_array=...)        Mode 2: conditions on the observed fixations
    ```

    You can see the stashed covariates on the built model in section 4.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Simulate a dataset with a trial-wise `eta`

    A per-trial predictor `x` drives `eta`: `eta_i = 0.2 + 0.3 * x_i` (the target we
    will recover). Observed fixations are truncated at the response — you only see
    fixations that started before the decision, and the likelihood requires
    `rt >= sacc[d-1]`.
    """)
    return


@app.cell
def _(extra_fields, make_fixations, np, pd, sim):
    _f = make_fixations(800, seed=42)
    _n = _f["n"]
    x = np.random.default_rng(1).uniform(0.0, 1.0, _n)
    _eta_true = 0.2 + 0.3 * x

    # theta with a per-trial eta column; the rest shared.
    _theta = np.column_stack([
        _eta_true, np.full(_n, 1.0), np.full(_n, 1.5),
        np.full(_n, 0.2), np.zeros(_n), np.zeros(_n),
    ])
    _out = sim.simulate(
        theta=_theta, n_samples=1, random_state=2, max_t=10.0, smooth_unif=False,
        extra_fields=extra_fields(_f),
    )
    _rt = np.asarray(_out["rts"]).reshape(-1)
    _ch = np.asarray(_out["choices"]).reshape(-1).astype(float)
    _keep = _rt != -999.0
    _rows = np.flatnonzero(_keep)

    _sacc, _d = _f["sacc"], _f["d"]
    _d_obs = np.array([max(int((_sacc[i, : _d[i]] < _rt[i]).sum()), 1) for i in _rows])
    data = pd.DataFrame({
        "rt": _rt[_keep], "response": _ch[_keep], "x": x[_keep],
        "r1": _f["r1"][_keep], "r2": _f["r2"][_keep], "flag": _f["flag"][_keep].astype(int),
        "d": _d_obs, "sigma": np.full(_rows.size, 1.0),
    })
    data["sacc_array"] = pd.Series([_sacc[i].copy() for i in _rows], index=data.index)
    data.head()
    return (data,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Fit a trial-wise regression on `eta`

    `include=[{"name": "eta", "formula": "eta ~ 1 + x"}]` makes `eta` vary per trial
    as a linear function of `x`. This trial-wise / hierarchical support on the core
    parameters is the headline aDDM capability.
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
def _(mo, model, np):
    # The handshake, on the built model: the observed fixations are stashed on the
    # RV class, ready for the generative (PPC) path to forward to the simulator.
    _ef = type(model.model_distribution.rv_op)._extra_fields
    _rows = "\n".join(
        f"    {k:<11} shape {tuple(np.asarray(v).shape)}  dtype {np.asarray(v).dtype}"
        for k, v in _ef.items()
    )
    mo.md(f"**`rv_op._extra_fields`** (observed fixations wired into PPC):\n\n```\n{_rows}\n```")
    return


@app.cell
def _(mo):
    run_inference = mo.ui.run_button(label="Run inference (samples, ~1 min)")
    mo.md(f"## 5. Fit and recover\n\n{run_inference}")
    return (run_inference,)


@app.cell
def _(mo, model, run_inference):
    mo.stop(not run_inference.value, mo.md("*Click the button above to sample.*"))
    idata = model.sample(
        draws=500, tune=500, chains=2, cores=1,
        idata_kwargs={"log_likelihood": False},
    )
    return (idata,)


@app.cell
def _(idata):
    import arviz as az

    # Truth: eta_Intercept ~ 0.2, eta_x ~ 0.3.
    az.summary(
        idata,
        var_names=["eta", "kappa", "a", "b", "x0", "t"],
        filter_vars="like",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Posterior predictive, conditioned on the observed fixations

    `sample_posterior_predictive` draws new (rt, choice) from the posterior **using
    each trial's real fixation sequence** — the handshake from section 2 feeds the
    observed `r1, r2, flag, sacc_array, d` to the simulator, so the check is faithful
    to the gaze pattern that produced the data.
    """)
    return


@app.cell
def _(idata, model):
    model.sample_posterior_predictive(idata, kind="response", draws=100)
    idata.posterior_predictive
    return


if __name__ == "__main__":
    app.run()

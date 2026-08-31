"""Fit and check one useful HSSM regression model.

This marimo notebook is the source of truth for the output-bearing Jupyter
artifact published by MkDocs. Routine CI executes the quick configuration;
the committed artifact is regenerated in full mode::

    FULL_RUN=1 uv run --group notebook --group docs marimo export ipynb \
        --no-sandbox docs/tutorials/main_tutorial.py \
        --output docs/tutorials/main_tutorial.ipynb \
        --include-outputs --force
"""

# ruff: noqa: B018, D401, E501, PLR1711  (generated marimo notebook)
# docs-require-full-run: true
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import contextlib
    import io
    import logging
    import os
    import warnings

    logging.getLogger("pytensor").setLevel(logging.ERROR)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
    logging.getLogger("pymc").setLevel(logging.ERROR)
    logging.getLogger("hssm").setLevel(logging.ERROR)

    import arviz as az
    import marimo as mo
    import numpy as np
    from matplotlib import pyplot as plt

    import hssm

    FULL_RUN = os.environ.get("FULL_RUN", "0") == "1"
    N_TRIALS = 500 if FULL_RUN else 200
    N_TUNE = 750 if FULL_RUN else 100
    N_DRAWS = 750 if FULL_RUN else 100
    N_CHAINS = 2 if FULL_RUN else 1
    N_PPC_DRAWS = 100 if FULL_RUN else 20
    RANDOM_SEED = 20260830

    @contextlib.contextmanager
    def quiet_console():
        """Hide progress noise while replaying unique, path-free warnings."""
        previous_disable_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    yield
            finally:
                logging.disable(previous_disable_level)
        for message in dict.fromkeys(str(item.message) for item in caught_warnings):
            print(f"Warning: {message}")

    return (
        FULL_RUN,
        N_CHAINS,
        N_DRAWS,
        N_PPC_DRAWS,
        N_TRIALS,
        N_TUNE,
        RANDOM_SEED,
        az,
        hssm,
        mo,
        np,
        plt,
        quiet_console,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The HSSM tutorial

    This guided tutorial starts where the [Quickstart](https://lnccbrown.github.io/HSSM/getting_started/getting_started/)
    ends. Instead of fitting another intercept-only DDM, you will answer one
    common research question: **does an experimental condition change drift
    rate?**

    We take one recommended route from data to interpretation:

    `simulate -> specify one regression -> sample -> diagnose -> check predictions -> interpret`

    You will fit one model and make one decision at each step. The optional
    [scenic route](https://lnccbrown.github.io/HSSM/tutorials/main_tutorial_scenic_route/)
    covers alternative model families, custom priors, hierarchical variants,
    model comparison, and low-level extensions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What you will learn

    By the end you will be able to:

    1. express a trial-level predictor with HSSM's formula interface;
    2. verify that the intended parameter and predictor entered the model;
    3. read a compact ArviZ summary and basic chain diagnostic;
    4. check whether the fitted model can reproduce the observed data; and
    5. report the condition effect with posterior uncertainty.

    For installation and the mechanics of a first fit, complete the Quickstart
    first. This page deliberately does not repeat those steps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run this tutorial

    <a href="https://colab.research.google.com/github/lnccbrown/HSSM/blob/main/docs/tutorials/main_tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open this notebook in Google Colab"></a>

    On Colab, install HSSM with `%pip install hssm`, then restart the runtime.
    The published outputs use the full configuration. Routine notebook CI uses
    a smaller configuration to exercise the same workflow quickly.
    """)
    return


@app.cell
def _(FULL_RUN, N_CHAINS, N_DRAWS, N_PPC_DRAWS, N_TRIALS, N_TUNE):
    {
        "mode": "full (published outputs)" if FULL_RUN else "quick (CI smoke check)",
        "artifact_marker": (
            "<!-- hssm-full-run-artifact: true -->"
            if FULL_RUN
            else "<!-- hssm-full-run-artifact: false -->"
        ),
        "trials": N_TRIALS,
        "chains": N_CHAINS,
        "tune": N_TUNE,
        "draws": N_DRAWS,
        "posterior_predictive_draws": N_PPC_DRAWS,
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Simulate the question, not just the model

    We create two equally likely conditions. The simulated drift rate is
    `0.3` in the reference condition and `1.0` in the treatment condition, so
    the known treatment effect is `0.7`. Boundary separation, starting point,
    and non-decision time stay constant.

    In an applied analysis, `condition` would come from your experimental data;
    only this simulation step would change.
    """)
    return


@app.cell
def _(N_TRIALS, RANDOM_SEED, hssm, np):
    rng = np.random.default_rng(RANDOM_SEED)
    condition = rng.integers(0, 2, size=N_TRIALS)

    true_values = {
        "v_Intercept": 0.3,
        "v_condition": 0.7,
        "a": 1.5,
        "z": 0.5,
        "t": 0.25,
    }
    trial_v = true_values["v_Intercept"] + true_values["v_condition"] * condition

    data = hssm.simulate_data(
        model="ddm",
        theta={
            "v": trial_v,
            "a": true_values["a"],
            "z": true_values["z"],
            "t": true_values["t"],
        },
        size=1,
        random_state=RANDOM_SEED,
    )
    data["condition"] = condition
    data.head()
    return data, true_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    HSSM requires `rt` and `response`; predictor columns sit beside them. Here
    `condition=0` is the reference level, so the intercept is its drift rate
    and the `condition` coefficient is the change from 0 to 1.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Specify one parameter regression

    We let only drift rate vary with condition. The other DDM parameters use
    HSSM's defaults. This is a good first specification when the scientific
    hypothesis concerns evidence quality rather than response caution, bias,
    or non-decision processes.
    """)
    return


@app.cell
def _(data, hssm):
    model = hssm.HSSM(
        data=data,
        model="ddm",
        initval_jitter=0,
        include=[
            {
                "name": "v",
                "formula": "v ~ 1 + condition",
                "link": "identity",
            }
        ],
    )
    print(model)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the model summary before sampling. It should show `v_Intercept` and
    `v_condition` under drift rate, while `a`, `z`, and `t` remain ordinary
    parameters. This check catches misspelled predictors and unintended model
    structure before computation begins.

    For coefficient-level priors, use [Specify priors and fix parameters](https://lnccbrown.github.io/HSSM/how_to/specify_priors/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Sample once, then inspect a compact result

    The full run uses two chains and enough draws for a tutorial-quality
    diagnostic view. The quick run is only an execution smoke test; one-chain
    diagnostics are not evidence of convergence.
    """)
    return


@app.cell
def _(N_CHAINS, N_DRAWS, N_TUNE, RANDOM_SEED, model, quiet_console):
    with quiet_console():
        idata = model.sample(
            sampler="pymc",
            chains=N_CHAINS,
            cores=1,
            draws=N_DRAWS,
            tune=N_TUNE,
            random_seed=RANDOM_SEED,
            idata_kwargs={"log_likelihood": False},
            progressbar=False,
        )
    return (idata,)


@app.cell
def _(idata):
    {
        "groups": tuple(idata.children),
        "posterior_sizes": dict(idata.posterior.ds.sizes),
    }
    return


@app.cell
def _(FULL_RUN, az, idata):
    parameter_names = ["v_Intercept", "v_condition", "a", "z", "t"]
    posterior_summary = az.summary(
        idata,
        var_names=parameter_names,
        kind="all" if FULL_RUN else "stats",
        round_to=2,
    )
    posterior_summary
    return (parameter_names,)


@app.cell
def _(FULL_RUN, az, idata, parameter_names):
    divergences = int(idata.sample_stats.diverging.values.sum())
    if FULL_RUN:
        diagnostic_summary = az.summary(
            idata,
            var_names=parameter_names,
            kind="diagnostics",
            round_to="none",
        )
        max_rhat = float(diagnostic_summary["r_hat"].max())
        min_ess = float(diagnostic_summary[["ess_bulk", "ess_tail"]].min().min())
        assert divergences == 0, f"full run has {divergences} divergences"
        assert max_rhat <= 1.01, f"full-run max R-hat is {max_rhat:.4f}"
        assert min_ess >= 200, f"full-run minimum ESS is {min_ess:.1f}"
        health = {
            "scope": "full diagnostic validation",
            "divergences": divergences,
            "max_rhat": round(max_rhat, 4),
            "min_bulk_or_tail_ess": round(min_ess, 1),
        }
    else:
        health = {
            "scope": "quick execution/specification smoke check only",
            "divergences_reported_not_gated": divergences,
            "rhat_and_ess": "not evaluated with one chain",
        }
    health
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Start with the coefficient row: its mean and interval describe the
    treatment-minus-reference change in drift. Then check `r_hat` (near 1),
    effective sample sizes, and the trace view below. In real work, increase
    draws and investigate warnings before interpretation.

    See [Plot posteriors and predictions](https://lnccbrown.github.io/HSSM/tutorials/plotting/)
    for the broader diagnostics and plotting toolkit.
    """)
    return


@app.cell
def _(az, idata, plt):
    trace_plot = az.plot_trace_dist(idata, var_names=["v_Intercept", "v_condition"])
    trace_figure = trace_plot.get_viz("figure")
    plt.close(trace_figure)
    trace_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Check predictions before interpreting the coefficient

    Diagnostics tell us whether the sampler explored this model; they do not
    tell us whether the model reproduces the observed choices and response
    times. A posterior predictive check asks exactly that question.
    """)
    return


@app.cell
def _(N_PPC_DRAWS, idata, model, quiet_console):
    with quiet_console():
        ppc_idata = model.sample_posterior_predictive(
            dt=idata,
            draws=N_PPC_DRAWS,
            inplace=False,
        )
    return (ppc_idata,)


@app.cell
def _(model, plt, ppc_idata):
    predictive_grid = model.plot_predictive(dt=ppc_idata, col="condition")
    predictive_figure = predictive_grid.figure
    # The condition facet labels carry the useful comparison; the default
    # global title overlaps them in the static docs layout.
    predictive_figure.suptitle("")
    predictive_figure.subplots_adjust(top=0.88)
    plt.close(predictive_figure)
    predictive_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Look for broad agreement in response proportions and response-time shape.
    A visible mismatch is a reason to revise the model before telling a story
    about `v_condition`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Report the condition effect

    We summarize the posterior mean, a 94% highest-density interval, and the
    posterior probability that the treatment effect is positive. These are
    direct descriptions of uncertainty, not a binary significance test.
    """)
    return


@app.cell
def _(FULL_RUN, az, idata, np, true_values):
    effect_draws = az.extract(idata, var_names=["v_condition"]).values
    effect_hdi = az.hdi(effect_draws, prob=0.94)
    probability_positive = float((effect_draws > 0).mean())
    if FULL_RUN:
        assert effect_hdi[0] <= true_values["v_condition"] <= effect_hdi[1], (
            "full-run condition-effect HDI misses the known simulated value"
        )
        assert probability_positive >= 0.95, (
            "full-run posterior does not clearly support a positive condition effect"
        )
    effect_report = {
        "known_simulated_effect": true_values["v_condition"],
        "posterior_mean": round(float(effect_draws.mean()), 3),
        "94%_HDI": tuple(np.round(effect_hdi, 3)),
        "P(v_condition > 0)": round(probability_positive, 3),
    }
    effect_report
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where to go next

    You now have one reusable HSSM workflow: put predictors in the data, attach
    a formula to the cognitive parameter named by your hypothesis, sample once,
    diagnose, check predictions, and report posterior uncertainty.

    Choose the next page for the question you actually have:

    - add participants with [Hierarchical modeling](https://lnccbrown.github.io/HSSM/getting_started/hierarchical_modeling/);
    - refine assumptions with [Specify priors and fix parameters](https://lnccbrown.github.io/HSSM/how_to/specify_priors/);
    - compare pre-specified candidates with [Compare and interpret models](https://lnccbrown.github.io/HSSM/how_to/compare_models/);
    - see an applied end-to-end analysis in [A complete scientific workflow](https://lnccbrown.github.io/HSSM/tutorials/scientific_workflow_hssm/); or
    - explore every alternative in [The HSSM tutorial: scenic route](https://lnccbrown.github.io/HSSM/tutorials/main_tutorial_scenic_route/).
    """)
    return


if __name__ == "__main__":
    app.run()

"""Take the scenic route through HSSM's full modeling surface.

This marimo notebook is the source of truth for the output-bearing Jupyter
artifact published by MkDocs. Routine CI executes the quick configuration;
the committed artifact is regenerated in full mode::

    FULL_RUN=1 uv run --group notebook --group docs marimo export ipynb \
        --no-sandbox docs/tutorials/main_tutorial_scenic_route.py \
        --output docs/tutorials/main_tutorial_scenic_route.ipynb \
        --include-outputs --force
"""

# ruff: noqa: B018, D401, E501, F841, PLR1711, Q000  (generated marimo notebook)
# docs-require-full-run: true
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The HSSM tutorial: scenic route

    <div style="text-align: center;">
      <img src="images/HSSM_logo.png" alt="HSSM logo" style="height: 120px; width: auto; max-width: 80%;">
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is the optional, comprehensive route through HSSM. It preserves the
    breadth of the original tutorial: model families, priors, regressions,
    participant hierarchies, comparison, custom likelihoods, and low-level PyMC.

    If you are fitting your first useful model, start with the concise
    [guided HSSM tutorial](https://lnccbrown.github.io/HSSM/tutorials/main_tutorial/).
    Return here when you want to understand the alternatives or work below the
    high-level interface.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What HSSM can help you model

    HSSM is built for computational neurocognitive modeling. You can:

    - fit sequential-sampling models to choices and response times;
    - estimate trial-level, condition-level, neural, or behavioral effects on model parameters;
    - pool information across participants with hierarchical models;
    - use [reinforcement-learning sequential-sampling models](https://lnccbrown.github.io/HSSM/tutorials/rlssm_basic/) and alternative decision processes; and
    - extend the model collection with custom likelihoods or low-level PyMC models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What you will learn

    1. Build and inspect a DDM from simulated data.
    2. Diagnose posterior samples with ArviZ and compare them with known values.
    3. Check posterior predictions before interpreting a model.
    4. Add priors, regressions, and participant hierarchies.
    5. Compare models, then explore advanced customization when needed.

    **Prerequisites:** complete the [Quickstart](https://lnccbrown.github.io/HSSM/getting_started/getting_started/)
    and the [guided tutorial](https://lnccbrown.github.io/HSSM/tutorials/main_tutorial/)
    first. This page is intentionally long and is not part of the shortest
    first-user path.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A quick map of the workflow

    Bayesian workflow map: `data -> model -> posterior samples -> diagnostics -> posterior predictive checks -> model comparison -> interpretation`.

    The first part of this tutorial follows that loop with simulated data, where the true parameter values are known. Later sections reuse the same loop with richer models, regressions, participant-level effects, and comparison across candidate models.

    <details>
    <summary><strong>What HSSM uses under the hood</strong></summary>

    <ul>
      <li><strong>HSSM</strong> provides the user-facing model interface for sequential-sampling models.</li>
      <li><strong>PyMC</strong> builds and samples the Bayesian model. Most calls to <code>.sample()</code> use PyMC's MCMC samplers.</li>
      <li><strong>ArviZ</strong> summarizes, diagnoses, visualizes, and compares fitted Bayesian models.</li>
      <li><strong>xarray</strong> and <strong>DataTree</strong> store labeled results such as posterior draws, sampler statistics, observed data, and posterior predictions.</li>
      <li><strong>Bambi</strong> supplies the formula syntax used when HSSM parameters depend on predictors, such as <code>v ~ 1 + x</code>.</li>
      <li><strong>PyTensor</strong> and <strong>JAX</strong> are computational backends. You usually only notice them when choosing advanced likelihoods or samplers.</li>
      <li><strong>ssm-simulators</strong> generates synthetic sequential-sampling data for examples and simulation studies.</li>
      <li><strong>ONNX</strong> is a portable format for neural-network likelihood approximators used by some advanced models.</li>
    </ul>

    You do not need to master these packages before starting. The main idea is to recognize which tool is responsible for each step of the workflow.
    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run this tutorial

    <a href="https://colab.research.google.com/github/lnccbrown/HSSM/blob/main/docs/tutorials/main_tutorial_scenic_route.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open this notebook in Google Colab"></a>

    On Colab, copy the command in the next block into a new code cell, run it once, then restart the runtime. For local setup, GPU extras, and troubleshooting see the [Installation guide](https://lnccbrown.github.io/HSSM/getting_started/installation/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```python
    %pip install hssm
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    import contextlib
    import io
    import logging
    import os
    import warnings

    logging.getLogger("pytensor").setLevel(logging.ERROR)

    import arviz as az
    import bambi as bmb
    import hddm_wfpt
    import jax
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    import hssm

    FULL_RUN = os.environ.get("FULL_RUN", "0") == "1"
    INITVAL_JITTER = 0
    SEEDS = {
        "simple": 134,
        "angle": 135,
        "only_v": 136,
        "bad_prior": 137,
        "reg": 138,
        "reg_angle": 139,
        "reg_multi": 140,
        "categorical": 141,
        "hier": 142,
        "compare_data": 143,
        "compare_1": 144,
        "compare_2": 145,
        "compare_3": 146,
        "trialwise": 147,
        "alternative": 148,
        "blackbox": 149,
        "pymc_data": 150,
        "pymc_ddm": 151,
        "pymc_angle": 152,
        "pymc_reg_data": 153,
        "pymc_reg": 154,
    }

    # Full mode reproduces the published scientific walkthrough. Quick mode
    # keeps routine notebook CI practical while exercising every branch.
    N_TUNE = 500 if FULL_RUN else 75
    N_DRAWS = 500 if FULL_RUN else 75
    N_PRIMARY_REG_TUNE = 750 if FULL_RUN else N_TUNE
    N_PRIMARY_REG_DRAWS = 750 if FULL_RUN else N_DRAWS
    N_CHAINS = 2 if FULL_RUN else 1
    N_SIMPLE_TRIALS = 500 if FULL_RUN else 200
    N_ANGLE_TRIALS = 1_000 if FULL_RUN else 250
    N_REG_TRIALS = 1_000 if FULL_RUN else 250
    N_HIER_PARTICIPANTS = 15 if FULL_RUN else 3
    N_HIER_TRIALS = 200 if FULL_RUN else 50
    N_COMPARE_TRIALS = 500 if FULL_RUN else 200
    N_ADVANCED_TRIALS = 1_000 if FULL_RUN else 250
    N_PPC_DRAWS = 100 if FULL_RUN else 20

    # Progress streams are useful interactively but make the static page huge.
    PYMC_PROGRESS = False
    EXTERNAL_PROGRESS = False

    def quiet_call(callable_, /, *args, **kwargs):
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
                    result = callable_(*args, **kwargs)
            finally:
                logging.disable(previous_disable_level)
        for message in dict.fromkeys(str(item.message) for item in caught_warnings):
            print(f"Warning: {message}")
        return result

    def add_trace_reference_lines(plot_collection, values) -> None:
        """Mark true values on distribution and MCMC-trace panels."""
        for name, value in values.items():
            if name not in plot_collection.data.data_vars:
                continue
            plot_collection.get_target(name, {"column": "dist"}).axvline(
                value, color="red", linestyle="--"
            )
            plot_collection.get_target(name, {"column": "trace"}).axhline(
                value, color="red", linestyle="--"
            )

    def compact_diagnostics(idata, var_names=None):
        """Return a small health snapshot without overstating one-chain runs."""
        divergences = int(idata.sample_stats.diverging.values.sum())
        divergence_draws = int(idata.sample_stats.diverging.size)
        divergence_rate = divergences / divergence_draws
        if idata.posterior.ds.sizes.get("chain", 0) < 2:
            return {
                "scope": "quick execution/specification smoke check only",
                "divergences_reported_not_gated": divergences,
                "divergence_rate_reported_not_gated": divergence_rate,
                "rhat_and_ess": "not evaluated with one chain",
            }
        diagnostics = az.summary(
            idata,
            var_names=var_names,
            kind="diagnostics",
            round_to="none",
        )
        return {
            "scope": "full diagnostic validation",
            "divergences": divergences,
            "divergence_rate": divergence_rate,
            "max_rhat": float(diagnostics["r_hat"].max()),
            "min_bulk_or_tail_ess": float(
                diagnostics[["ess_bulk", "ess_tail"]].min().min()
            ),
        }

    return (
        EXTERNAL_PROGRESS,
        FULL_RUN,
        INITVAL_JITTER,
        N_ADVANCED_TRIALS,
        N_ANGLE_TRIALS,
        N_CHAINS,
        N_COMPARE_TRIALS,
        N_DRAWS,
        N_HIER_PARTICIPANTS,
        N_HIER_TRIALS,
        N_PPC_DRAWS,
        N_PRIMARY_REG_DRAWS,
        N_PRIMARY_REG_TUNE,
        N_REG_TRIALS,
        N_SIMPLE_TRIALS,
        N_TUNE,
        PYMC_PROGRESS,
        SEEDS,
        add_trace_reference_lines,
        az,
        bmb,
        hddm_wfpt,
        hssm,
        jax,
        np,
        pd,
        plt,
        compact_diagnostics,
        quiet_call,
    )


@app.cell
def _(
    FULL_RUN,
    N_CHAINS,
    N_DRAWS,
    N_HIER_PARTICIPANTS,
    N_HIER_TRIALS,
    N_PPC_DRAWS,
    N_PRIMARY_REG_DRAWS,
    N_PRIMARY_REG_TUNE,
    N_TUNE,
):
    {
        "mode": "full (published outputs)" if FULL_RUN else "quick (CI smoke check)",
        "artifact_marker": (
            "<!-- hssm-full-run-artifact: true -->"
            if FULL_RUN
            else "<!-- hssm-full-run-artifact: false -->"
        ),
        "chains": N_CHAINS,
        "tune": N_TUNE,
        "draws": N_DRAWS,
        "primary_regression_tune": N_PRIMARY_REG_TUNE,
        "primary_regression_draws": N_PRIMARY_REG_DRAWS,
        "hierarchical_participants": N_HIER_PARTICIPANTS,
        "hierarchical_trials_per_participant": N_HIER_TRIALS,
        "posterior_predictive_draws": N_PPC_DRAWS,
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Build and inspect your first HSSM model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulate a simple drift-diffusion dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The drift-diffusion model (DDM) is a useful first example because it describes both the response a participant makes and how long the decision takes. Its key parameters are:

    - `v`: drift rate, the average rate of evidence accumulation;
    - `a`: boundary separation, a speed--accuracy setting;
    - `z`: starting point, an a priori response bias; and
    - `t`: non-decision time, such as encoding and motor time.

    <div style="text-align: center;">
      <img src="images/DDM_with_params_pic.png" alt="Drift diffusion model diagram with parameters v, a, t, and z" style="width: 360px; max-width: 90%; height: auto;">
    </div>

    We simulate data with known values first. This makes the later posterior checks concrete: the red reference lines will show the values used to generate the data.
    """)
    return


@app.cell
def _(N_SIMPLE_TRIALS, SEEDS, hssm):
    param_dict_init = dict(v=0.5, a=1.5, z=0.5, t=0.5)
    v_true, a_true, z_true, t_true = (
        param_dict_init["v"],
        param_dict_init["a"],
        param_dict_init["z"],
        param_dict_init["t"],
    )

    dataset = hssm.simulate_data(
        model="ddm",
        theta=param_dict_init,
        size=N_SIMPLE_TRIALS,
        random_state=SEEDS["simple"],
    )
    dataset
    return dataset, param_dict_init, t_true, v_true, z_true


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fit the model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To create the simplest HSSM model, provide a `pandas.DataFrame` with `rt` and `response` columns. HSSM supplies the default DDM configuration, including an `analytical` likelihood and default priors.

    What happens in this one line: HSSM checks the data columns, chooses the default DDM parameterization, attaches priors and bounds, and builds the corresponding PyMC model. If you have used HDDM, the workflow will feel familiar. HSSM builds the probabilistic model with PyMC and uses Bambi-style formulas when parameters depend on predictors.
    """)
    return


@app.cell
def _(INITVAL_JITTER, dataset, hssm):
    simple_ddm_model = hssm.HSSM(data=dataset, initval_jitter=INITVAL_JITTER)
    return (simple_ddm_model,)


@app.cell
def _(simple_ddm_model):
    simple_ddm_model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The printed model summary is the first specification check. It shows the observations, free parameters, priors, bounds, and likelihood, so you can confirm that HSSM is estimating the model you intended before spending time sampling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Inspect the model graph
    """)
    return


@app.cell
def _(simple_ddm_model):
    simple_ddm_model.graph()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The graph uses probabilistic-programming notation:

    - white nodes are unknown random variables to estimate;
    - the grey node is the observed choice/response-time data;
    - rounded rectangles describe dimensions; and
    - sharp-cornered rectangles denote deterministic quantities.

    For simple models the graph is compact. The goal is not to memorize every node, but to check that the observed data, parameters, and deterministic transformations match your scientific story. The graph becomes especially helpful once regressions and participant-level effects are added.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Sample from the posterior

    We now use PyMC's NUTS sampler to draw posterior samples. A posterior sample is a collection of plausible parameter values after combining the prior, the likelihood, and the observed data. The settings below are intentionally small so the tutorial remains runnable; increase chains, draws, and tuning for a real analysis.
    """)
    return


@app.cell
def _(
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    PYMC_PROGRESS,
    SEEDS,
    quiet_call,
    simple_ddm_model,
):
    infer_data_simple_ddm_model = quiet_call(
        simple_ddm_model.sample,
        sampler="pymc",
        cores=1,
        chains=N_CHAINS,
        draws=N_DRAWS,
        tune=N_TUNE,
        idata_kwargs=dict(log_likelihood=False),
        mp_ctx="spawn",
        progressbar=PYMC_PROGRESS,
        random_seed=SEEDS["simple"],
    )
    return (infer_data_simple_ddm_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sampling returns an `xarray.DataTree`: a labeled container for all the fitted-model results. Next, we will inspect its contents and use ArviZ to assess what the sampler returned.
    """)
    return


@app.cell
def _(infer_data_simple_ddm_model):
    type(infer_data_simple_ddm_model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Understand the fitted result
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    HSSM (via the ArviZ package) stores results in an `xarray.DataTree`. Each group contains a related part of the Bayesian workflow, such as posterior samples, pointwise log likelihoods, sampler statistics, or observed data.

    You do not need to manipulate every group to use HSSM, but recognizing this structure makes it easier to use ArviZ and to add your own analyses. When you see later calls such as `az.summary(...)`, `az.plot_trace(...)`, or `az.compare(...)`, ArviZ is reading these labeled groups.
    """)
    return


@app.cell
def _(infer_data_simple_ddm_model):
    {
        "groups": tuple(infer_data_simple_ddm_model.children),
        "posterior_sizes": dict(infer_data_simple_ddm_model.posterior.ds.sizes),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For this model, the most important groups are:

    - `posterior`: sampled values for model parameters such as `v`, `a`, `z`, and `t`;
    - `sample_stats`: sampler diagnostics such as divergences, tree depth, and acceptance information;
    - `observed_data`: the choices and response times that were modeled;
    - `posterior_predictive`: simulated data from the fitted model, added later when we run posterior predictive checks.

    This first fit omits `log_likelihood` to keep the tutorial fast. The three model-comparison fits later enable it because leave-one-out comparison needs trial-level likelihood contributions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Work with posterior draws
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Access groups and variables
    """)
    return


@app.cell
def _(infer_data_simple_ddm_model):
    tuple(infer_data_simple_ddm_model.posterior.ds.data_vars)
    return


@app.cell
def _(infer_data_simple_ddm_model, np):
    np.round(infer_data_simple_ddm_model.posterior.a.values[0, :5], 3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To simply access the underlying data as a `numpy.ndarray`, we can use `.values` (as e.g. when using `pandas.DataFrame` objects).
    """)
    return


@app.cell
def _(infer_data_simple_ddm_model):
    type(infer_data_simple_ddm_model.posterior.a.values)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Combine chains and draws
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Many follow-up calculations are easier when the `chain` and `draw` dimensions are combined into a single `sample` dimension. ArviZ's `extract` helper provides a convenient interface for this common operation. The following cell shows the equivalent lower-level `xarray` operation.
    """)
    return


@app.cell
def _(az, infer_data_simple_ddm_model):
    idata_extracted = az.extract(infer_data_simple_ddm_model)
    {
        "variables": tuple(idata_extracted.data_vars),
        "sizes": dict(idata_extracted.sizes),
    }
    return (idata_extracted,)


@app.cell
def _(infer_data_simple_ddm_model):
    dict(infer_data_simple_ddm_model.posterior.ds.stack(sample=("chain", "draw")).sizes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ArviZ for diagnostics and visualization

    <div style="text-align: center;">
      <img src="images/arviz.png" alt="ArviZ logo" style="height: 120px; width: auto; max-width: 80%;">
    </div>

    HSSM returns `xarray`-based inference results that work directly with [ArviZ](https://python.arviz.org/en/stable/index.html). We will use ArviZ to summarize posterior uncertainty, inspect MCMC traces, check posterior predictions, and compare models. The examples below focus on the few summaries and plots that are most useful when starting out.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Diagnose and interpret the posterior
    """)
    return


@app.cell
def _(az, infer_data_simple_ddm_model, simple_ddm_model):
    az.summary(
        infer_data_simple_ddm_model,
        var_names=[var_name.name for var_name in simple_ddm_model.pymc_model.free_RVs],
    )
    return


@app.cell
def _(FULL_RUN, compact_diagnostics, infer_data_simple_ddm_model):
    simple_health = compact_diagnostics(
        infer_data_simple_ddm_model,
        var_names=["v", "a", "z", "t"],
    )
    if FULL_RUN:
        assert simple_health["divergence_rate"] <= 0.005, (
            "core DDM divergence rate exceeds 0.5%"
        )
        assert simple_health["max_rhat"] <= 1.01, (
            f"core DDM max R-hat is {simple_health['max_rhat']:.4f}"
        )
        assert simple_health["min_bulk_or_tail_ess"] >= 200, (
            "core DDM minimum bulk/tail ESS is below 200"
        )
    simple_health
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The summary reports posterior location and uncertainty for each parameter, plus diagnostics. The `mean` and `sd` columns describe the center and spread of the posterior draws. The highest-density interval (`hdi_3%` to `hdi_97%` by default) gives a compact uncertainty interval. Start diagnostics with `r_hat`: values near 1 indicate that independent chains explored the same distribution. As a practical rule, investigate values above 1.01 and inspect trace plots before interpreting parameter estimates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Trace and distribution plots
    """)
    return


@app.cell
def _(
    add_trace_reference_lines,
    az,
    infer_data_simple_ddm_model,
    param_dict_init,
):
    _pc = az.plot_trace_dist(infer_data_simple_ddm_model)
    add_trace_reference_lines(_pc, param_dict_init)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    HSSM also stores the latest result on `.traces`. Both access patterns are equivalent; this reactive notebook keeps using the explicitly returned result so every diagnostic has a visible dependency on sampling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The distribution panel summarizes posterior uncertainty for each parameter. The MCMC trace panel shows each chain across draws; stable, overlapping chains suggest the sampler is repeatedly visiting the same high-probability region rather than getting stuck in different places. The red reference lines mark the known values used to simulate the data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Forest plots
    """)
    return


@app.cell
def _(az, infer_data_simple_ddm_model):
    _ = az.plot_forest(infer_data_simple_ddm_model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A forest plot turns posterior uncertainty into intervals, which is useful when many parameters or chains are shown at once. By default, chains are shown separately. Combining chains can make a large forest plot easier to scan once you have already checked trace diagnostics.
    """)
    return


@app.cell
def _(az, infer_data_simple_ddm_model):
    _ = az.plot_forest(infer_data_simple_ddm_model, combined=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Marginal posterior plots
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A marginal posterior plot ignores sampling order and focuses on the distribution of one parameter at a time. Because this is simulated data, we can compare the posterior with the known generating values. The standalone marginal plot below uses vertical reference lines; the paired trace/distribution plots use vertical lines on distributions and horizontal lines on traces.
    """)
    return


@app.cell
def _(az, infer_data_simple_ddm_model, param_dict_init):
    _pc = az.plot_dist(infer_data_simple_ddm_model, kind="hist")
    _ = az.add_lines(
        _pc,
        values=param_dict_init,
        orientation="vertical",
        visuals={"ref_line": dict(color="red", linestyle="--")},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Posterior pair plots
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pair plots reveal relationships between posterior parameters. Strong trade-offs can signal weak identification: when one parameter increases, another may compensate while producing similar predicted behavior. This is common in cognitive models, where several parameters can affect the same response-time or choice pattern.
    """)
    return


@app.cell
def _(az, infer_data_simple_ddm_model):
    _ = az.plot_pair(infer_data_simple_ddm_model, marginal_kind="kde")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ArviZ provides many additional diagnostics and plotting tools. The [current user guide](https://python.arviz.org/en/stable/user_guide/getting_started.html) is the best next reference when you need a specific plot or diagnostic.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Compute quantities from posterior draws
    """)
    return


@app.cell
def _(idata_extracted, np, plt):
    # Calculate the correlation matrix
    posterior_correlation_matrix = np.corrcoef(
        np.stack(
            [
                idata_extracted[var_].values
                for var_ in idata_extracted.data_vars.variables
            ]
        )
    )
    num_vars = posterior_correlation_matrix.shape[0]
    fig, ax = plt.subplots(1, 1)
    cax = ax.imshow(posterior_correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax)
    ax.set_title("Posterior Correlation Matrix")
    ax.set_xticks(range(posterior_correlation_matrix.shape[0]))
    # Make heatmap
    ax.set_xticklabels([var_ for var_ in idata_extracted.data_vars.variables])
    ax.set_yticks(range(posterior_correlation_matrix.shape[0]))
    ax.set_yticklabels([var_ for var_ in idata_extracted.data_vars.variables])
    for _i in range(num_vars):
        for j in range(num_vars):
            # Add ticks
            ax.text(
                j,
                _i,
                f"{posterior_correlation_matrix[_i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    # Annotate heatmap
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check posterior predictions

    Good MCMC diagnostics show that the sampler explored the stated model reliably; they do not show that the model captures the data. Posterior predictive checks compare data simulated from the fitted model with the observations and are an essential step before substantive interpretation. In workflow terms, this is where we ask: if the fitted model were true, would it generate choices and response times that look like the data we actually observed?
    """)
    return


@app.cell
def _(N_PPC_DRAWS, infer_data_simple_ddm_model, quiet_call, simple_ddm_model):
    ppc_idata = quiet_call(
        simple_ddm_model.sample_posterior_predictive,
        dt=infer_data_simple_ddm_model,
        draws=N_PPC_DRAWS,
        inplace=False,
    )
    return (ppc_idata,)


@app.cell
def _(plt, ppc_idata, simple_ddm_model):
    _ppc_axes = simple_ddm_model.plot_predictive(dt=ppc_idata)
    _ppc_figure = _ppc_axes.figure
    plt.close(_ppc_figure)
    _ppc_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The posterior predictive call returns a result with a `posterior_predictive` group, and `plot_predictive()` visualizes those simulated datasets against the observations. If predictions reproduce the main features of the observed response-time and choice distributions, the model is a useful approximation for those features. Systematic mismatches suggest revisiting the likelihood, parameterization, covariates, or model family.

    HSSM does not currently expose a random-seed argument for posterior predictive sampling. All data simulation and MCMC fits on this page are seeded; the exact predictive draws may vary while the same model check is performed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Choose a model and likelihood
    """)
    return


@app.cell
def _(simple_ddm_model):
    simple_ddm_model.loglik_kind
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The DDM above used HSSM’s `analytical` likelihood. Other sequential-sampling models may instead use an `approx_differentiable` likelihood, such as a likelihood approximation network (LAN), or a user-supplied `blackbox` likelihood. The model interface stays similar; HSSM chooses compatible computational machinery behind the scenes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### An angle model with collapsing boundaries

    The `angle` model extends the DDM with `theta`, which controls the rate at which decision boundaries collapse over time.

    <div style="text-align: center;">
      <img src="images/ANGLE_with_params_pic.png" alt="Angle model diagram with collapsing decision boundaries and theta parameter" style="width: 360px; max-width: 90%; height: auto;">
    </div>

    Collapsing boundaries are useful when urgency or time pressure may change a participant’s decision criterion. HSSM makes inference for these models practical through packaged `approx_differentiable` likelihoods.
    """)
    return


@app.cell
def _(N_ANGLE_TRIALS, SEEDS, hssm):
    # Simulate angle data
    v_angle_true = 0.5
    a_angle_true = 1.5
    z_angle_true = 0.5
    t_angle_true = 0.2
    theta_angle_true = 0.2

    param_dict_angle = dict(v=0.5, a=1.5, z=0.5, t=0.2, theta=0.2)

    dataset_angle = hssm.simulate_data(
        model="angle",
        theta=param_dict_angle,
        size=N_ANGLE_TRIALS,
        random_state=SEEDS["angle"],
    )
    return dataset_angle, param_dict_angle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We pass a single additional argument to our `HSSM` class and set `model='angle'`.
    """)
    return


@app.cell
def _(INITVAL_JITTER, dataset_angle, hssm):
    model_angle = hssm.HSSM(
        data=dataset_angle, model="angle", initval_jitter=INITVAL_JITTER
    )

    model_angle
    return (model_angle,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The graph now includes the additional `theta` parameter. This is a quick way to confirm that the model specification matches the scientific question.
    """)
    return


@app.cell
def _(model_angle):
    model_angle.graph()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's check the type of likelihood that is used under the hood ...
    """)
    return


@app.cell
def _(model_angle):
    model_angle.loglik_kind
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This model uses an `approx_differentiable` likelihood. In the packaged model collection, that typically means a LAN is used internally to approximate the likelihood.
    """)
    return


@app.cell
def _(
    EXTERNAL_PROGRESS,
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    SEEDS,
    jax,
    model_angle,
    quiet_call,
):
    jax.config.update("jax_enable_x64", False)
    infer_data_angle = quiet_call(
        model_angle.sample,
        sampler="numpyro",
        chains=N_CHAINS,
        cores=1,
        draws=N_DRAWS,
        tune=N_TUNE,
        idata_kwargs=dict(log_likelihood=False),  # no need to return likelihoods here
        # mp_ctx="spawn",
        progressbar=EXTERNAL_PROGRESS,
        random_seed=SEEDS["angle"],
    )
    return (infer_data_angle,)


@app.cell
def _(add_trace_reference_lines, az, infer_data_angle, param_dict_angle):
    _pc = az.plot_trace_dist(infer_data_angle)
    add_trace_reference_lines(_pc, param_dict_angle)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Customize priors and model structure
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Priors express plausible parameter ranges before observing the data. HSSM supports defaults that respect model bounds, fixed values for parameters you do not want to estimate, and explicit PyMC distributions when you need stronger domain knowledge.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fix parameters when the design justifies it
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sometimes a parameter is fixed by design or is outside the present research question. Here we estimate only the drift rate `v` while holding the other DDM parameters fixed.

    <div style="text-align: center;">
      <img src="images/DDM_only_v_pic.png" alt="Drift diffusion model diagram with only the drift-rate parameter v estimated" style="width: 360px; max-width: 90%; height: auto;">
    </div>

    Fixing parameters reduces model flexibility, so it should be justified by theory, design, or a deliberate comparison.
    """)
    return


@app.cell
def _(param_dict_init):
    param_dict_init
    return


@app.cell
def _(INITVAL_JITTER, dataset, hssm, param_dict_init):
    ddm_model_only_v = hssm.HSSM(
        data=dataset,
        model="ddm",
        a=param_dict_init["a"],
        t=param_dict_init["t"],
        z=param_dict_init["z"],
        initval_jitter=INITVAL_JITTER,
    )
    return (ddm_model_only_v,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since we fix all but one parameter, we estimate only one parameter. This is a useful pattern when a parameter is known from design, when a previous analysis justifies a fixed value, or when you want to isolate one cognitive process. The model graph should reflect this choice: we expect only one free random variable, `v`.
    """)
    return


@app.cell
def _(ddm_model_only_v):
    ddm_model_only_v.graph()
    return


@app.cell
def _(
    N_CHAINS,
    N_PRIMARY_REG_DRAWS,
    N_PRIMARY_REG_TUNE,
    PYMC_PROGRESS,
    SEEDS,
    ddm_model_only_v,
    quiet_call,
):
    infer_data_only_v = quiet_call(
        ddm_model_only_v.sample,
        sampler="pymc",
        chains=N_CHAINS,
        cores=1,
        draws=N_PRIMARY_REG_DRAWS,
        tune=N_PRIMARY_REG_TUNE,
        idata_kwargs=dict(log_likelihood=False),  # no need to return likelihoods here
        mp_ctx="spawn",
        progressbar=PYMC_PROGRESS,
        random_seed=SEEDS["only_v"],
    )
    return (infer_data_only_v,)


@app.cell
def _(add_trace_reference_lines, az, infer_data_only_v, param_dict_init):
    _pc = az.plot_trace_dist(infer_data_only_v)
    add_trace_reference_lines(_pc, {"v": param_dict_init["v"]})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A rank plot complements a trace plot by checking whether chains are sampling from the same distribution. Roughly uniform ranks across chains are consistent with good mixing; visible chain-specific patterns call for further diagnosis.
    """)
    return


@app.cell
def _(FULL_RUN, az, infer_data_only_v):
    if FULL_RUN:
        _ = az.plot_rank(infer_data_only_v)
    else:
        print("Rank diagnostics require the two-chain full run.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Specify informative priors
    """)
    return


@app.cell
def _(INITVAL_JITTER, dataset, hssm):
    model_normal = hssm.HSSM(
        data=dataset,
        include=[
            {
                "name": "v",
                "prior": {"name": "Normal", "mu": 0, "sigma": 0.01},
            }
        ],
        initval_jitter=INITVAL_JITTER,
    )
    return (model_normal,)


@app.cell
def _(model_normal):
    model_normal
    return


@app.cell
def _(
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    PYMC_PROGRESS,
    SEEDS,
    model_normal,
    quiet_call,
):
    infer_data_normal = quiet_call(
        model_normal.sample,
        sampler="pymc",
        chains=N_CHAINS,
        cores=1,
        draws=N_DRAWS,
        tune=N_TUNE,
        idata_kwargs=dict(log_likelihood=False),  # no need to return likelihoods here
        mp_ctx="spawn",
        progressbar=PYMC_PROGRESS,
        random_seed=SEEDS["bad_prior"],
    )
    return (infer_data_normal,)


@app.cell
def _(add_trace_reference_lines, az, infer_data_normal, param_dict_init):
    _pc = az.plot_trace_dist(infer_data_normal)
    add_trace_reference_lines(_pc, param_dict_init)
    return


@app.cell
def _(compact_diagnostics, infer_data_normal):
    narrow_prior_health = compact_diagnostics(
        infer_data_normal,
        var_names=["v", "a", "z", "t"],
    )
    narrow_prior_health
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The data were generated with `v=0.5`, `a=1.5`, `z=0.5`, and `t=0.5`. Compare the trace and compact diagnostic snapshot with the baseline fit. If the narrow prior pulls `v` toward zero, other parameters may compensate; any divergences or weak diagnostics are reasons not to interpret this intentionally restrictive fit. This example is reported, not gated, because it is designed to illustrate a problematic prior.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regressions on cognitive parameters

    <div style="text-align: center;">
      <img src="images/bambi.png" alt="Bambi logo" style="height: 120px; width: auto; max-width: 80%;">
    </div>

    HSSM can link individual SSM parameters to trial-level covariates with Bambi-style formulas. This lets you ask questions such as whether a neural signal, condition, or behavioral measure predicts drift rate, boundary separation, or bias.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### One parameter as a regression target
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Simulating Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We simulate data in which drift rate varies with two trial-level covariates. The known coefficients give us a clear recovery target.
    """)
    return


@app.cell
def _(N_REG_TRIALS, SEEDS, hssm, np):
    # Set up trial by trial parameters
    _rng = np.random.default_rng(SEEDS["reg"])
    v_intercept = 0.3
    x = _rng.uniform(-1, 1, size=N_REG_TRIALS)
    v_x = 0.8
    y = _rng.uniform(-1, 1, size=N_REG_TRIALS)
    v_y = 0.3
    _v_reg_v = v_intercept + v_x * x + v_y * y
    _a_reg_v = 1.5
    # rest
    _z_reg_v = 0.5
    _t_reg_v = 0.1
    param_dict_reg_v = dict(
        a=1.5,
        z=0.5,
        t=0.1,
        v=_v_reg_v,
        v_x=v_x,
        v_y=v_y,
        v_Intercept=v_intercept,
        theta=0.0,
    )
    dataset_reg_v = hssm.simulate_data(
        model="ddm",
        theta=param_dict_reg_v,
        size=1,
        random_state=SEEDS["reg"],
    )
    dataset_reg_v["x"] = x
    # base dataset
    # Adding covariates into the datsaframe
    dataset_reg_v["y"] = y
    return dataset_reg_v, param_dict_reg_v, v_intercept, v_x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Define the regression
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `include` argument contains one specification per parameter with a regression. Each specification names the parameter, supplies a formula and link, and can define priors for regression coefficients.

    Formula syntax follows Bambi and familiar R-style mixed-model notation. HSSM uses Bambi to translate these formulas into a PyMC model. Conceptually, this means the SSM parameter is no longer a single value; it can vary systematically with trial-level or participant-level predictors.
    """)
    return


@app.cell
def _(INITVAL_JITTER, dataset_reg_v, hssm):
    model_reg_v_simple = hssm.HSSM(
        data=dataset_reg_v,
        include=[{"name": "v", "formula": "v ~ 1 + x + y"}],
        initval_jitter=INITVAL_JITTER,
    )
    return (model_reg_v_simple,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### `Param` class
    As illustrated below, there is an alternative way of specifying the parameter specific data via the `Param` class.
    """)
    return


@app.cell
def _(INITVAL_JITTER, dataset_reg_v, hssm):
    model_reg_v_simple_new = hssm.HSSM(
        data=dataset_reg_v,
        include=[hssm.Param(name="v", formula="v ~ 1 + x + y")],
        initval_jitter=INITVAL_JITTER,
    )
    return (model_reg_v_simple_new,)


@app.cell
def _(model_reg_v_simple, model_reg_v_simple_new):
    {
        "dict_specification_free_RVs": tuple(
            rv.name for rv in model_reg_v_simple.pymc_model.free_RVs
        ),
        "Param_specification_free_RVs": tuple(
            rv.name for rv in model_reg_v_simple_new.pymc_model.free_RVs
        ),
    }
    return


@app.cell
def _(model_reg_v_simple):
    model_reg_v_simple.graph()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Customize parameter-specific priors
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The default regression specification is often a good starting point. When prior knowledge is available, specify coefficient-level priors explicitly and then verify the resulting model summary before sampling.
    """)
    return


@app.cell
def _(INITVAL_JITTER, dataset_reg_v, hssm):
    model_reg_v = hssm.HSSM(
        data=dataset_reg_v,
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
                    "x": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
                    "y": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
                },
                "formula": "v ~ 1 + x + y",
                "link": "identity",
            }
        ],
        initval_jitter=INITVAL_JITTER,
    )
    return (model_reg_v,)


@app.cell
def _(model_reg_v):
    model_reg_v
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model summary now shows `v` as a regression with an intercept and covariate coefficients. This is the same specification-check step as before, but now the summary should also confirm that the intended predictors entered the model.
    """)
    return


@app.cell
def _(
    N_CHAINS,
    N_PRIMARY_REG_DRAWS,
    N_PRIMARY_REG_TUNE,
    PYMC_PROGRESS,
    SEEDS,
    model_reg_v,
    quiet_call,
):
    infer_data_reg_v = quiet_call(
        model_reg_v.sample,
        sampler="pymc",
        chains=N_CHAINS,
        cores=1,
        draws=N_PRIMARY_REG_DRAWS,
        tune=N_PRIMARY_REG_TUNE,
        idata_kwargs={"log_likelihood": False},
        mp_ctx="spawn",
        progressbar=PYMC_PROGRESS,
        random_seed=SEEDS["reg"],
    )
    return (infer_data_reg_v,)


@app.cell
def _(az, infer_data_reg_v):
    az.summary(infer_data_reg_v, var_names=["~a", "~z", "~t"])
    return


@app.cell
def _(add_trace_reference_lines, az, infer_data_reg_v, param_dict_reg_v):
    _pc = az.plot_trace_dist(
        infer_data_reg_v,
        var_names=["v_Intercept", "v_x", "v_y", "a", "z", "t"],
    )
    add_trace_reference_lines(_pc, param_dict_reg_v)
    return


@app.cell
def _(
    FULL_RUN,
    az,
    compact_diagnostics,
    infer_data_reg_v,
    param_dict_reg_v,
):
    regression_health = compact_diagnostics(
        infer_data_reg_v,
        var_names=["v_Intercept", "v_x", "v_y", "a", "z", "t"],
    )
    recovery_report = {}
    raw_hdis = {}
    probabilities_positive = {}
    for _name in ("v_Intercept", "v_x", "v_y"):
        _draws = az.extract(infer_data_reg_v, var_names=[_name]).values
        _hdi = az.hdi(_draws, prob=0.94)
        raw_hdis[_name] = _hdi
        probabilities_positive[_name] = float((_draws > 0).mean())
        recovery_report[_name] = {
            "known_value": param_dict_reg_v[_name],
            "posterior_mean": round(float(_draws.mean()), 3),
            "94%_HDI": tuple(round(float(value), 3) for value in _hdi),
            "P(>0)": round(probabilities_positive[_name], 3),
        }
    if FULL_RUN:
        assert regression_health["divergence_rate"] <= 0.005, (
            "primary regression divergence rate exceeds 0.5%"
        )
        assert regression_health["max_rhat"] <= 1.01, (
            f"primary regression max R-hat is {regression_health['max_rhat']:.4f}"
        )
        assert regression_health["min_bulk_or_tail_ess"] >= 200, (
            "primary regression minimum bulk/tail ESS is below 200"
        )
        _v_x_hdi = raw_hdis["v_x"]
        assert _v_x_hdi[0] <= param_dict_reg_v["v_x"] <= _v_x_hdi[1], (
            "primary regression HDI misses the known focal v_x effect"
        )
        assert probabilities_positive["v_x"] >= 0.95, (
            "primary regression does not clearly support a positive v_x effect"
        )
    {"diagnostics": regression_health, "coefficient_recovery": recovery_report}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The full publication run requires acceptable diagnostics and checks that the 94% interval for the focal `v_x` coefficient contains its known value (`0.8`) with high posterior probability above zero. The intercept and `v_y` estimates are reported without simultaneous-coverage assertions; inspect their uncertainty rather than treating every finite simulation as exact recovery.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Regression with an angle model
    """)
    return


@app.cell
def _(INITVAL_JITTER, dataset_reg_v, hssm):
    model_reg_v_angle = hssm.HSSM(
        data=dataset_reg_v,
        model="angle",
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {
                        "name": "Uniform",
                        "lower": -3.0,
                        "upper": 3.0,
                    },
                    "x": {
                        "name": "Uniform",
                        "lower": -1.0,
                        "upper": 1.0,
                    },
                    "y": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
                },
                "formula": "v ~ 1 + x + y",
                "link": "identity",
            }
        ],
        initval_jitter=INITVAL_JITTER,
    )
    return (model_reg_v_angle,)


@app.cell
def _(model_reg_v_angle):
    model_reg_v_angle.graph()
    return


@app.cell
def _(
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    PYMC_PROGRESS,
    SEEDS,
    model_reg_v_angle,
    quiet_call,
):
    trace_reg_v_angle = quiet_call(
        model_reg_v_angle.sample,
        sampler="pymc",
        chains=N_CHAINS,
        cores=1,
        draws=N_DRAWS,
        tune=N_TUNE,
        idata_kwargs={"log_likelihood": False},
        mp_ctx="spawn",
        progressbar=PYMC_PROGRESS,
        random_seed=SEEDS["reg_angle"],
    )
    return (trace_reg_v_angle,)


@app.cell
def _(add_trace_reference_lines, az, param_dict_reg_v, trace_reg_v_angle):
    _pc = az.plot_trace_dist(
        trace_reg_v_angle,
        var_names=["v_Intercept", "v_x", "v_y", "a", "z", "t", "theta"],
    )
    add_trace_reference_lines(_pc, param_dict_reg_v)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Inspect the posterior interval for `theta`: data generated by a standard DDM support the expected story only when that interval is compatible with zero and the sampler diagnostics are acceptable. Interpret the remaining parameters with the same conditional workflow.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Regress multiple parameters
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now fit regressions for both `v` and `a`. Only `v` truly varies with the simulated covariates, so the `a` coefficients should be centered near zero.
    """)
    return


@app.cell
def _(INITVAL_JITTER, dataset_reg_v, hssm, param_dict_reg_v):
    # Instantiate our hssm model
    from copy import deepcopy

    param_dict_reg_v_a = deepcopy(param_dict_reg_v)
    param_dict_reg_v_a["a_Intercept"] = param_dict_reg_v_a["a"]
    param_dict_reg_v_a["a_x"] = 0
    param_dict_reg_v_a["a_y"] = 0

    hssm_reg_v_a_angle = hssm.HSSM(
        data=dataset_reg_v,
        model="angle",
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
                    "x": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
                    "y": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
                },
                "formula": "v ~ 1 + x + y",
            },
            {
                "name": "a",
                "prior": {
                    "Intercept": {"name": "Uniform", "lower": 0.5, "upper": 3.0},
                    "x": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
                    "y": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
                },
                "formula": "a ~ 1 + x + y",
            },
        ],
        initval_jitter=INITVAL_JITTER,
    )
    return hssm_reg_v_a_angle, param_dict_reg_v_a


@app.cell
def _(hssm_reg_v_a_angle):
    hssm_reg_v_a_angle
    return


@app.cell
def _(hssm_reg_v_a_angle):
    hssm_reg_v_a_angle.graph()
    return


@app.cell
def _(
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    PYMC_PROGRESS,
    SEEDS,
    hssm_reg_v_a_angle,
    quiet_call,
):
    infer_data_reg_v_a = quiet_call(
        hssm_reg_v_a_angle.sample,
        sampler="pymc",
        chains=N_CHAINS,
        cores=1,
        draws=N_DRAWS,
        tune=N_TUNE,
        idata_kwargs={"log_likelihood": False},
        mp_ctx="spawn",
        progressbar=PYMC_PROGRESS,
        random_seed=SEEDS["reg_multi"],
    )
    return (infer_data_reg_v_a,)


@app.cell
def _(az, infer_data_reg_v_a):
    az.summary(
        infer_data_reg_v_a,
        var_names=[
            "v_Intercept",
            "v_x",
            "v_y",
            "a_Intercept",
            "a_x",
            "a_y",
            "z",
            "t",
            "theta",
        ],
    )
    return


@app.cell
def _(add_trace_reference_lines, az, infer_data_reg_v_a, param_dict_reg_v_a):
    _pc = az.plot_trace_dist(
        infer_data_reg_v_a,
        var_names=[
            "v_Intercept",
            "v_x",
            "v_y",
            "a_Intercept",
            "a_x",
            "a_y",
            "z",
            "t",
            "theta",
        ],
    )
    add_trace_reference_lines(_pc, param_dict_reg_v_a)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use the numeric summary to assess recovery: the `v` rows should be compared with their known coefficients, while the `a_x` and `a_y` intervals should be checked for compatibility with zero because the data-generating process did not vary `a` with these covariates. Treat either pattern as evidence from this finite run, not a guaranteed outcome.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Categorical covariates
    """)
    return


@app.cell
def _(N_REG_TRIALS, SEEDS, hssm, np):
    _rng = np.random.default_rng(SEEDS["categorical"])
    x_1 = _rng.choice(4, size=N_REG_TRIALS).astype(int)
    x_offset = np.array([0, 1, -0.5, 0.75])
    y_1 = _rng.uniform(-1, 1, size=N_REG_TRIALS)
    v_y_1 = 0.3
    _v_reg_v = 0 + v_y_1 * y_1 + x_offset[x_1]
    _a_reg_v = 1.5
    _z_reg_v = 0.5
    _t_reg_v = 0.1
    dataset_reg_v_cat = hssm.simulate_data(
        model="ddm",
        theta=dict(v=_v_reg_v, a=_a_reg_v, z=_z_reg_v, t=_t_reg_v),
        size=1,
        random_state=SEEDS["categorical"],
    )
    dataset_reg_v_cat["x"] = x_1
    dataset_reg_v_cat["y"] = y_1
    return dataset_reg_v_cat, v_y_1


@app.cell
def _(INITVAL_JITTER, dataset_reg_v_cat, hssm):
    model_reg_v_cat = hssm.HSSM(
        data=dataset_reg_v_cat,
        model="angle",
        include=[
            {
                "name": "v",
                "formula": "v ~ 0 + C(x) + y",
                "link": "identity",
            }
        ],
        initval_jitter=INITVAL_JITTER,
    )
    return (model_reg_v_cat,)


@app.cell
def _(model_reg_v_cat):
    model_reg_v_cat.graph()
    return


@app.cell
def _(
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    PYMC_PROGRESS,
    SEEDS,
    model_reg_v_cat,
    quiet_call,
):
    infer_data_reg_v_cat = quiet_call(
        model_reg_v_cat.sample,
        sampler="pymc",
        chains=N_CHAINS,
        cores=1,
        draws=N_DRAWS,
        tune=N_TUNE,
        idata_kwargs={"log_likelihood": False},
        mp_ctx="spawn",
        progressbar=PYMC_PROGRESS,
        random_seed=SEEDS["categorical"],
    )
    return (infer_data_reg_v_cat,)


@app.cell
def _(az, infer_data_reg_v_cat):
    _ = az.plot_forest(infer_data_reg_v_cat)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hierarchical participant effects
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we simulate data from 15 participants, each with 200 trials. A hierarchy lets participant-level estimates share information through a group distribution while preserving individual differences.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulate Data
    """)
    return


@app.cell
def _(N_HIER_PARTICIPANTS, N_HIER_TRIALS, SEEDS, hssm, np, pd):
    _rng = np.random.default_rng(SEEDS["hier"])
    n_participants = N_HIER_PARTICIPANTS
    n_trials = N_HIER_TRIALS
    sd_v = 0.5
    mean_v = 0.5
    data_list = []
    for _i in range(n_participants):
        v_intercept_hier = _rng.normal(mean_v, sd_v, size=1)
        x_2 = _rng.uniform(-1, 1, size=n_trials)
        v_x_hier = 0.8
        y_2 = _rng.uniform(-1, 1, size=n_trials)
        v_y_hier = 0.3
        v_hier = v_intercept_hier + v_x_hier * x_2 + v_y_hier * y_2
        a_hier = 1.5
        t_hier = 0.5
        z_hier = 0.5
        data_tmp = hssm.simulate_data(
            model="ddm",
            theta=dict(v=v_hier, a=a_hier, z=z_hier, t=t_hier),
            size=1,
            random_state=SEEDS["hier"] + _i,
        )
        data_tmp["participant_id"] = _i
        data_tmp["x"] = x_2
        data_tmp["y"] = y_2
        data_list.append(data_tmp)
    dataset_reg_v_hier = pd.concat(data_list)
    dataset_reg_v_hier
    return dataset_reg_v_hier, x_2, y_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use `v ~ 1 + (1|participant_id) + x + y`. The random-intercept term `(1|participant_id)` gives each participant an offset around the group intercept; the remaining coefficients are shared across participants. The hierarchy lets participants borrow strength from the group while still allowing individual differences.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Basic Hierarchical Model

    Note the `noncentered=True` argument below: it selects the non-centered parameterization for the group-specific terms. The default is the right choice here — when and why to flip it (globally, or per parameter) is covered in [Centered vs. non-centered parameterizations](https://lnccbrown.github.io/HSSM/tutorials/centered_vs_noncentered_basic_logic/) and [Per-parameter parameterization](https://lnccbrown.github.io/HSSM/tutorials/parameterization_per_parameter/).
    """)
    return


@app.cell
def _(INITVAL_JITTER, dataset_reg_v_hier, hssm):
    model_reg_v_angle_hier = hssm.HSSM(
        data=dataset_reg_v_hier,
        model="angle",
        noncentered=True,
        initval_jitter=INITVAL_JITTER,
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {
                        "name": "Normal",
                        "mu": 0.0,
                        "sigma": 0.5,
                    },
                    "x": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
                    "y": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
                },
                "formula": "v ~ 1 + (1|participant_id) + x + y",
                "link": "identity",
            }
        ],
    )
    return (model_reg_v_angle_hier,)


@app.cell
def _(model_reg_v_angle_hier):
    model_reg_v_angle_hier.graph()
    return


@app.cell
def _(
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    PYMC_PROGRESS,
    SEEDS,
    jax,
    model_reg_v_angle_hier,
    quiet_call,
):
    jax.config.update("jax_enable_x64", False)
    infer_data_reg_v_angle_hier = quiet_call(
        model_reg_v_angle_hier.sample,
        sampler="pymc",
        chains=N_CHAINS,
        cores=1,
        draws=N_DRAWS,
        tune=N_TUNE,
        idata_kwargs={"log_likelihood": False},
        mp_ctx="spawn",
        progressbar=PYMC_PROGRESS,
        random_seed=SEEDS["hier"],
    )
    return (infer_data_reg_v_angle_hier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Inspect the posterior to distinguish group-level effects from participant-level variation and to confirm that the chains mix well. In hierarchical models, it is especially useful to separate population-level parameters from participant-specific offsets before interpreting the substantive effects.
    """)
    return


@app.cell
def _(az, infer_data_reg_v_angle_hier):
    _ = az.plot_forest(infer_data_reg_v_angle_hier, combined=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Validate and compare models
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Model comparison asks how well competing models predict the same data while accounting for fit and complexity. It comes after diagnostics and posterior predictive checks because a model with poor sampling behavior or obvious predictive failures is not a strong scientific candidate, even if a comparison table looks favorable. Here the data are generated with `a=1.5`, and we compare three DDMs that fix `a` too low (`1.3`), correctly (`1.5`), or too high (`1.7`).

    We use ArviZ's `compare()` function with expected log predictive density from leave-one-out cross-validation (`elpd_loo`). Higher expected predictive accuracy is better, but close differences should be interpreted as uncertainty rather than a hard ranking.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Simulation
    """)
    return


@app.cell
def _(N_COMPARE_TRIALS, SEEDS, hssm):
    # Parameters
    param_dict_mod_comp = dict(v=0.5, a=1.5, z=0.5, t=0.2)

    # Simulation
    dataset_model_comp = hssm.simulate_data(
        model="ddm",
        theta=param_dict_mod_comp,
        size=N_COMPARE_TRIALS,
        random_state=SEEDS["compare_data"],
    )

    {
        "shape": dataset_model_comp.shape,
        "preview": dataset_model_comp.head(),
    }
    return (dataset_model_comp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Defining the Models
    """)
    return


@app.cell
def _(INITVAL_JITTER, dataset_model_comp, hssm):
    # "under-specified" model — boundary fixed too low
    model_model_comp_1 = hssm.HSSM(
        data=dataset_model_comp,
        model="ddm",
        a=1.3,
        initval_jitter=INITVAL_JITTER,
    )
    return (model_model_comp_1,)


@app.cell
def _(INITVAL_JITTER, dataset_model_comp, hssm):
    # "correct" model — boundary fixed at the data-generating value
    model_model_comp_2 = hssm.HSSM(
        data=dataset_model_comp,
        model="ddm",
        a=1.5,
        initval_jitter=INITVAL_JITTER,
    )
    return (model_model_comp_2,)


@app.cell
def _(INITVAL_JITTER, dataset_model_comp, hssm):
    # "over-specified" model — boundary fixed too high
    model_model_comp_3 = hssm.HSSM(
        data=dataset_model_comp,
        model="ddm",
        a=1.7,
        initval_jitter=INITVAL_JITTER,
    )
    return (model_model_comp_3,)


@app.cell
def _(
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    PYMC_PROGRESS,
    SEEDS,
    model_model_comp_1,
    quiet_call,
):
    infer_data_model_comp_1 = quiet_call(
        model_model_comp_1.sample,
        sampler="pymc",
        cores=1,
        chains=N_CHAINS,
        draws=N_DRAWS,
        tune=N_TUNE,
        idata_kwargs=dict(
            log_likelihood=True
        ),  # model comparison metrics usually need this!
        mp_ctx="spawn",
        progressbar=PYMC_PROGRESS,
        random_seed=SEEDS["compare_1"],
    )
    return (infer_data_model_comp_1,)


@app.cell
def _(
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    PYMC_PROGRESS,
    SEEDS,
    model_model_comp_2,
    quiet_call,
):
    infer_data_model_comp_2 = quiet_call(
        model_model_comp_2.sample,
        sampler="pymc",
        cores=1,
        chains=N_CHAINS,
        draws=N_DRAWS,
        tune=N_TUNE,
        idata_kwargs=dict(
            log_likelihood=True
        ),  # model comparison metrics usually need this!
        mp_ctx="spawn",
        progressbar=PYMC_PROGRESS,
        random_seed=SEEDS["compare_2"],
    )
    return (infer_data_model_comp_2,)


@app.cell
def _(
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    PYMC_PROGRESS,
    SEEDS,
    model_model_comp_3,
    quiet_call,
):
    infer_data_model_comp_3 = quiet_call(
        model_model_comp_3.sample,
        sampler="pymc",
        cores=1,
        chains=N_CHAINS,
        draws=N_DRAWS,
        tune=N_TUNE,
        idata_kwargs=dict(
            log_likelihood=True
        ),  # model comparison metrics usually need this!
        mp_ctx="spawn",
        progressbar=PYMC_PROGRESS,
        random_seed=SEEDS["compare_3"],
    )
    return (infer_data_model_comp_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compare
    """)
    return


@app.cell
def _(
    az,
    infer_data_model_comp_1,
    infer_data_model_comp_2,
    infer_data_model_comp_3,
    quiet_call,
):
    compare_data = quiet_call(
        az.compare,
        {
            "a_fixed_1.3(under)": infer_data_model_comp_1,
            "a_fixed_1.5(correct)": infer_data_model_comp_2,
            "a_fixed_1.7(over)": infer_data_model_comp_3,
        },
    )

    compare_data
    return (compare_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this controlled example, inspect whether the correctly specified model has the highest expected predictive accuracy and whether its uncertainty overlaps the alternatives. If the ordering differs or estimates are close, report that uncertainty rather than forcing a winner. In applied work, use comparison as one piece of evidence alongside posterior predictive checks and domain knowledge.
    """)
    return


@app.cell
def _(az, compare_data):
    _ = az.plot_compare(compare_data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The forest plot can explain comparison differences: when `a` is fixed incorrectly, inspect whether other parameters shift to compensate. This is a useful habit after model comparison, regardless of which candidate ranks first in a finite run.
    """)
    return


@app.cell
def _(
    az,
    infer_data_model_comp_1,
    infer_data_model_comp_2,
    infer_data_model_comp_3,
):
    _ = az.plot_forest(
        {
            "a_fixed_1.3(under)": infer_data_model_comp_1,
            "a_fixed_1.5(correct)": infer_data_model_comp_2,
            "a_fixed_1.7(over)": infer_data_model_comp_3,
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulation and model configuration under the hood

    The following optional examples show trial-wise simulation, model discovery, and direct use of `ssm-simulators`. They are useful when you move beyond the introductory DDM workflow.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Trial-wise simulation with HSSM
    """)
    return


@app.cell
def _(N_ADVANCED_TRIALS, SEEDS, hssm, np, t_true, v_true, z_true):
    # a changes trial wise
    _rng = np.random.default_rng(SEEDS["trialwise"])
    a_trialwise = _rng.normal(loc=2, scale=0.3, size=N_ADVANCED_TRIALS)

    dataset_a_trialwise = hssm.simulate_data(
        model="ddm",
        theta=dict(
            v=v_true,
            a=a_trialwise,
            z=z_true,
            t=t_true,
        ),
        size=1,
        random_state=SEEDS["trialwise"],
    )

    dataset_a_trialwise
    return (a_trialwise,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    HSSM can simulate many supported models. The models available for simulation and the models with packaged likelihood functions are related but not identical; inspect the supported-model list when choosing a model for inference.
    """)
    return


@app.cell
def _(hssm):
    hssm.HSSM.supported_models
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model configuration records parameter names, default likelihoods, bounds, and prior settings. It is useful when you are adapting a built-in model or contributing a new one.
    """)
    return


@app.cell
def _(hssm):
    hssm.modelconfig.get_default_model_config("ddm")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For simulation, two configuration entries are particularly useful:

    1. `list_params` gives the parameter order and names expected by the model.
    2. `likelihoods` records the available `analytical`, `approx_differentiable`, and `blackbox` likelihood options, together with their bounds and defaults.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Direct `ssm-simulators` usage
    """)
    return


@app.cell
def _(N_ADVANCED_TRIALS, SEEDS, a_trialwise, np, pd, t_true, v_true, z_true):
    from ssms.basic_simulators.simulator import simulator

    theta_mat = np.zeros((N_ADVANCED_TRIALS, 4))
    theta_mat[:, 0] = v_true
    theta_mat[:, 1] = a_trialwise
    # a changes trial wise
    theta_mat[:, 2] = z_true
    theta_mat[:, 3] = t_true  # v
    sim_out_trialwise = simulator(
        theta=theta_mat,
        model="ddm",
        n_samples=1,
        random_state=SEEDS["trialwise"],
    )  # a
    dataset_trialwise = pd.DataFrame(
        np.column_stack(
            [sim_out_trialwise["rts"][:, 0], sim_out_trialwise["choices"][:, 0]]
        ),
        columns=["rt", "response"],
    )  # z
    # simulate data
    # Turn into nice dataset
    dataset_trialwise  # t  # parameter_matrix  # specify model (many are included in ssms)  # number of samples for each set of parameters  # (plays the role of `size` parameter in `hssm.simulate_data`)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For most HSSM workflows, `hssm.simulate_data()` is the clearest starting point. Direct `ssm-simulators` access is useful when you need the simulator’s lower-level output or a custom simulation pipeline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Optional advanced extensions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The remaining sections show how HSSM connects to the broader computational ecosystem. They are optional for a first analysis, but useful when you need custom simulators, likelihoods, or a lower-level PyMC model. If you are new to HSSM, it is reasonable to stop after model comparison and return here once you need more control.

    <div style="text-align: center;">
      <img src="images/pytensor_jax.png" alt="PyTensor and JAX logos" style="height: 120px; width: auto; max-width: 80%;">
    </div>
    """)
    return


@app.cell
def _(hssm):
    hssm.config.default_model_config["ddm"].keys()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A model configuration describes its response coding, parameter list, description, and available likelihood definitions. Inspecting it is a practical starting point for advanced customization.
    """)
    return


@app.cell
def _(hssm):
    hssm.config.default_model_config["ddm"]["likelihoods"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The DDM configuration exposes three likelihood kinds: `analytical`, `approx_differentiable`, and `blackbox`. The kind determines the representation of the likelihood and which samplers are compatible.
    """)
    return


@app.cell
def _(hssm):
    hssm.config.default_model_config["ddm"]["likelihoods"]["analytical"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The key entries are `loglik`, `backend`, `bounds`, and `default_priors`. Bounds constrain valid parameter regions, while defaults provide a usable prior specification when one is not supplied explicitly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An `approx_differentiable` likelihood can be represented by a differentiable approximation, such as a likelihood approximation network. HSSM can evaluate compatible likelihoods through PyTensor or JAX backends.
    """)
    return


@app.cell
def _(hssm):
    hssm.config.default_model_config["ddm"]["likelihoods"]["approx_differentiable"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For packaged approximate likelihoods, `loglik` may point to an ONNX model. ONNX is a portable format for neural-network likelihood approximators. The backend determines whether HSSM evaluates the likelihood through PyTensor or JAX, which in turn affects compatible MCMC samplers.

    The practical takeaway is that HSSM keeps the user-facing model specification stable while allowing different likelihood representations underneath.
    """)
    return


@app.cell
def _(INITVAL_JITTER, dataset, hssm):
    hssm_alternative_model = hssm.HSSM(
        data=dataset,
        model="ddm",
        loglik_kind="approx_differentiable",
        initval_jitter=INITVAL_JITTER,
    )
    return (hssm_alternative_model,)


@app.cell
def _(hssm_alternative_model):
    hssm_alternative_model.loglik_kind
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This model uses an `approx_differentiable` LAN likelihood rather than the `analytical` likelihood used in the first DDM example. The assumed generative model remains the DDM.
    """)
    return


@app.cell
def _(
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    PYMC_PROGRESS,
    SEEDS,
    hssm_alternative_model,
    quiet_call,
):
    infer_data_alternative = quiet_call(
        hssm_alternative_model.sample,
        sampler="pymc",
        cores=1,
        chains=N_CHAINS,
        draws=N_DRAWS,
        tune=N_TUNE,
        idata_kwargs=dict(log_likelihood=False),  # no comparison here
        mp_ctx="spawn",
        progressbar=PYMC_PROGRESS,
        random_seed=SEEDS["alternative"],
    )
    return (infer_data_alternative,)


@app.cell
def _(az, infer_data_alternative):
    _ = az.plot_forest(infer_data_alternative)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also provide a custom likelihood directly. The next section illustrates the non-differentiable `blackbox` case.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Black-box likelihoods

    <div style="text-align: center;">
      <img src="images/blackbox.png" alt="Black-box likelihood concept illustration" style="width: 360px; max-width: 90%; height: auto;">
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A black-box likelihood is a Python callable that returns trial-wise log likelihoods. It is useful when you can evaluate a model numerically but do not have a differentiable likelihood representation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because black-box likelihoods are generally non-differentiable, HSSM uses a gradient-free sampling strategy by default. This makes them flexible, but they are usually slower and require especially careful diagnostics.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulating simple dataset from the DDM
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As always, let's begin by generating some simple dataset.
    """)
    return


@app.cell
def _(N_ADVANCED_TRIALS, SEEDS, hssm):
    # Set parameters
    param_dict_blackbox = dict(v=0.5, a=1.5, z=0.5, t=0.5)

    # Simulate
    dataset_blackbox = hssm.simulate_data(
        model="ddm",
        theta=param_dict_blackbox,
        size=N_ADVANCED_TRIALS,
        random_state=SEEDS["blackbox"],
    )
    return dataset_blackbox, param_dict_blackbox


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define the likelihood
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The callable receives the observed data and model parameters, then returns trial-wise log likelihoods. In this demonstration it delegates to a DDM likelihood, but the same interface can wrap an appropriate custom computation.
    """)
    return


@app.cell
def _(hddm_wfpt, np):
    def my_blackbox_loglik(data, v, a, z, t, err=1e-08):
        """Create a custom blackbox likelihood function."""
        data = data[:, 0] * data[:, 1]
        data_nrows = data.shape[0]
        return hddm_wfpt.wfpt.wiener_logp_array(
            np.float64(data),
            (np.ones(data_nrows) * v).astype(np.float64),
            np.ones(data_nrows) * 0,
            (np.ones(data_nrows) * 2 * a).astype(np.float64),
            (np.ones(data_nrows) * z).astype(np.float64),
            np.ones(data_nrows) * 0,
            (np.ones(data_nrows) * t).astype(np.float64),
            np.ones(data_nrows) * 0,
            err,
            1,
        )  # Our function expects inputs as float64, but they are not guaranteed to  # come in as such --> we type convert

    return (my_blackbox_loglik,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define HSSM class with our Blackbox Likelihood
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Construct the HSSM model as usual, passing the callable as `loglik` and declaring `loglik_kind="blackbox"`. Bounds remain important because they define the region where the custom likelihood is valid.
    """)
    return


@app.cell
def _(INITVAL_JITTER, bmb, dataset_blackbox, hssm, my_blackbox_loglik):
    blackbox_model = hssm.HSSM(
        data=dataset_blackbox,
        model="ddm",
        loglik=my_blackbox_loglik,
        loglik_kind="blackbox",
        model_config={
            "bounds": {
                "v": (-10.0, 10.0),
                "a": (0.5, 5.0),
                "z": (0.0, 1.0),
            }
        },
        t=bmb.Prior("Uniform", lower=0.0, upper=2.0),
        initval_jitter=INITVAL_JITTER,
    )
    return (blackbox_model,)


@app.cell
def _(blackbox_model):
    blackbox_model.graph()
    return


@app.cell
def _(
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    PYMC_PROGRESS,
    SEEDS,
    blackbox_model,
    quiet_call,
):
    sample = quiet_call(
        blackbox_model.sample,
        chains=N_CHAINS,
        cores=1,
        draws=N_DRAWS,
        tune=N_TUNE,
        idata_kwargs={"log_likelihood": False},
        progressbar=PYMC_PROGRESS,
        random_seed=SEEDS["blackbox"],
    )
    return (sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Black-box likelihoods default to a gradient-free Slice sampler. You may choose another suitable PyMC sampler, but gradient-based JAX samplers are not compatible with a non-differentiable likelihood.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Results
    """)
    return


@app.cell
def _(az, sample):
    az.summary(sample)
    return


@app.cell
def _(add_trace_reference_lines, az, param_dict_blackbox, sample):
    _pc = az.plot_trace_dist(sample)
    add_trace_reference_lines(_pc, param_dict_blackbox)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Low-level PyMC integration

    HSSM can also expose sequential-sampling random variables for use inside a custom PyMC model. This is an advanced path for models that require structure beyond the high-level HSSM interface.

    See the [low-level PyMC tutorial](https://lnccbrown.github.io/HSSM/tutorials/pymc/) for a focused follow-up example.
    """)
    return


@app.cell
def _():
    # DDM models (the Wiener First-Passage Time distribution)
    from hssm.distribution_utils import make_distribution
    from hssm.likelihoods import DDM

    return DDM, make_distribution


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulate some data
    """)
    return


@app.cell
def _(N_ADVANCED_TRIALS, SEEDS, hssm):
    # Simulate
    param_dict_pymc = dict(v=0.5, a=1.5, z=0.5, t=0.5)

    dataset_pymc = hssm.simulate_data(
        model="ddm",
        theta=param_dict_pymc,
        size=N_ADVANCED_TRIALS,
        random_state=SEEDS["pymc_data"],
    )
    return dataset_pymc, param_dict_pymc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build a custom PyMC Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now use our custom random variable `DDM` directly in a PyMC model.
    """)
    return


@app.cell
def _(DDM, dataset_pymc):
    import pymc as pm

    with pm.Model() as ddm_pymc:
        _v = pm.Uniform("v", lower=-10.0, upper=10.0)
        _a = pm.HalfNormal("a", sigma=2.0)
        _z = pm.Uniform("z", lower=0.01, upper=0.99)
        _t = pm.Uniform("t", lower=0.0, upper=0.6)
        ddm = DDM(
            "DDM",
            observed=dataset_pymc[["rt", "response"]].values,
            v=_v,
            a=_a,
            z=_z,
            t=_t,
        )
    return ddm_pymc, pm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's check the model graph:
    """)
    return


@app.cell
def _(ddm_pymc, pm):
    pm.model_to_graphviz(model=ddm_pymc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The custom PyMC graph resembles the HSSM graph because HSSM builds on the same probabilistic-programming components. You can sample the PyMC model directly and use ArviZ for diagnostics.
    """)
    return


@app.cell
def _(N_CHAINS, N_DRAWS, N_TUNE, PYMC_PROGRESS, SEEDS, ddm_pymc, pm, quiet_call):
    with ddm_pymc:
        ddm_pymc_trace = quiet_call(
            pm.sample,
            chains=N_CHAINS,
            cores=1,
            draws=N_DRAWS,
            tune=N_TUNE,
            idata_kwargs={"log_likelihood": False},
            progressbar=PYMC_PROGRESS,
            random_seed=SEEDS["pymc_ddm"],
        )
    return (ddm_pymc_trace,)


@app.cell
def _(add_trace_reference_lines, az, ddm_pymc_trace, param_dict_pymc):
    _pc = az.plot_trace_dist(ddm_pymc_trace)
    add_trace_reference_lines(_pc, param_dict_pymc)
    return


@app.cell
def _(az, ddm_pymc_trace):
    _ = az.plot_forest(ddm_pymc_trace)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Alternative models with PyMC
    """)
    return


@app.cell
def _(hssm, make_distribution):
    from hssm.distribution_utils import make_likelihood_callable

    _angle_loglik = make_likelihood_callable(
        loglik="angle.onnx",
        loglik_kind="approx_differentiable",
        backend="jax",
        params_is_reg=[0, 0, 0, 0, 0],
    )
    ANGLE = make_distribution(
        "angle",
        loglik=_angle_loglik,
        list_params=hssm.defaults.default_model_config["angle"]["list_params"],
    )
    return ANGLE, make_likelihood_callable


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `params_is_reg` vector identifies which likelihood inputs vary trial by trial, as happens when a parameter is produced by a regression formula.
    """)
    return


@app.cell
def _(ANGLE, dataset_pymc, pm):
    with pm.Model() as angle_pymc:
        _v = pm.Uniform("v", lower=-10.0, upper=10.0)
        _a = pm.Uniform("a", lower=0.5, upper=2.5)
        _z = pm.Uniform("z", lower=0.01, upper=0.99)
        _t = pm.Uniform("t", lower=0.0, upper=0.6)
        _theta = pm.Uniform("theta", lower=-0.1, upper=1.0)
        _angle = ANGLE(
            "ANGLE",
            v=_v,
            a=_a,
            z=_z,
            t=_t,
            theta=_theta,
            observed=dataset_pymc[["rt", "response"]].values,
        )
    return (angle_pymc,)


@app.cell
def _(
    EXTERNAL_PROGRESS,
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    SEEDS,
    angle_pymc,
    pm,
    quiet_call,
):
    with angle_pymc:
        idata_object = quiet_call(
            pm.sample,
            nuts_sampler="numpyro",
            chains=N_CHAINS,
            cores=1,
            draws=N_DRAWS,
            tune=N_TUNE,
            idata_kwargs={"log_likelihood": False},
            progressbar=EXTERNAL_PROGRESS,
            random_seed=SEEDS["pymc_angle"],
        )
    return (idata_object,)


@app.cell
def _(add_trace_reference_lines, az, idata_object, param_dict_pymc):
    _pc = az.plot_trace_dist(idata_object)
    add_trace_reference_lines(_pc, param_dict_pymc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Regression directly in PyMC
    """)
    return


@app.cell
def _():
    from typing import Optional

    def make_params_is_reg_vec(
        reg_parameters: Optional[list] = None, parameter_names: Optional[list] = None
    ):
        """Make a list of Trues and Falses to indicate which parameters are vectors."""
        if (not isinstance(reg_parameters, list)) or (
            not isinstance(parameter_names, list)
        ):
            raise ValueError("Both reg_parameters and parameter_names should be lists")

        bool_list = [0] * len(parameter_names)
        for param in reg_parameters:
            bool_list[parameter_names.index(param)] = 1
        return bool_list

    return (make_params_is_reg_vec,)


@app.cell
def _(
    N_ADVANCED_TRIALS,
    SEEDS,
    hssm,
    make_distribution,
    make_likelihood_callable,
    make_params_is_reg_vec,
    np,
):
    v_intercept_pymc_reg = 0.3
    _rng = np.random.default_rng(SEEDS["pymc_reg_data"])
    x_pymc_reg = _rng.uniform(-1, 1, size=N_ADVANCED_TRIALS)
    v_x_pymc_reg = 0.8
    y_pymc_reg = _rng.uniform(-1, 1, size=N_ADVANCED_TRIALS)
    v_y_pymc_reg = 0.3
    v_pymc_reg = (
        v_intercept_pymc_reg + v_x_pymc_reg * x_pymc_reg + v_y_pymc_reg * y_pymc_reg
    )
    param_dict_pymc_reg = dict(
        v_Intercept=v_intercept_pymc_reg,
        v_x=v_x_pymc_reg,
        v_y=v_y_pymc_reg,
        v=v_pymc_reg,
        a=1.5,
        z=0.5,
        t=0.1,
        theta=0.0,
    )
    pymc_reg_data = hssm.simulate_data(
        model="ddm",
        theta=param_dict_pymc_reg,
        size=1,
        random_state=SEEDS["pymc_reg_data"],
    )
    pymc_reg_data["x"] = x_pymc_reg
    pymc_reg_data["y"] = y_pymc_reg
    bool_param_reg = make_params_is_reg_vec(
        reg_parameters=["v"],
        parameter_names=hssm.defaults.default_model_config["angle"]["list_params"],
    )
    _angle_loglik = make_likelihood_callable(
        loglik="angle.onnx",
        loglik_kind="approx_differentiable",
        backend="jax",
        params_is_reg=bool_param_reg,
    )
    ANGLE_1 = make_distribution(
        "angle",
        loglik=_angle_loglik,
        list_params=hssm.defaults.default_model_config["angle"]["list_params"],
    )
    return ANGLE_1, pymc_reg_data


@app.cell
def _(ANGLE_1, pm, pymc_reg_data):
    import pytensor.tensor as pt

    with pm.Model(
        coords={
            "idx": pymc_reg_data.index,
            "resp": ["rt", "response"],
            "features": ["x", "y"],
        }
    ) as pymc_model_reg:
        x_ = pm.Data("x", pymc_reg_data["x"].values, dims="idx")
        y_ = pm.Data("y", pymc_reg_data["y"].values, dims="idx")
        obs = pm.Data(
            "obs", pymc_reg_data[["rt", "response"]].values, dims=("idx", "resp")
        )
        _a = pm.Uniform("a", lower=0.5, upper=2.5)
        _z = pm.Uniform("z", lower=0.01, upper=0.99)
        _t = pm.Uniform("t", lower=0.0, upper=0.6)
        _theta = pm.Uniform("theta", lower=-0.1, upper=1.0)
        v_Intercept = pm.Uniform("v_Intercept", lower=-3, upper=3)
        v_betas = pm.Normal("v_beta", mu=[0, 0], sigma=0.5, dims="features")
        _v = pm.Deterministic(
            "v", v_Intercept + pt.stack([x_, y_], axis=1) @ v_betas, dims="idx"
        )
        _angle = ANGLE_1(
            "angle",
            v=_v.squeeze(),
            a=_a,
            z=_z,
            t=_t,
            theta=_theta,
            observed=obs,
            dims=("idx", "resp"),
        )
    return (pymc_model_reg,)


@app.cell
def _(
    EXTERNAL_PROGRESS,
    N_CHAINS,
    N_DRAWS,
    N_TUNE,
    SEEDS,
    pm,
    pymc_model_reg,
    quiet_call,
):
    with pymc_model_reg:
        idata_pymc_reg = quiet_call(
            pm.sample,
            nuts_sampler="numpyro",
            chains=N_CHAINS,
            cores=1,
            draws=N_DRAWS,
            tune=N_TUNE,
            idata_kwargs={"log_likelihood": False},
            progressbar=EXTERNAL_PROGRESS,
            random_seed=SEEDS["pymc_reg"],
        )
    return (idata_pymc_reg,)


@app.cell
def _(az, idata_pymc_reg):
    _ = az.plot_forest(idata_pymc_reg, var_names=["~v"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion and further resources

    You have now followed the core HSSM workflow: simulate or load data, define a model, sample the posterior, diagnose MCMC behavior, check predictions, and compare alternatives. From there, use the following resources according to your next question.

    ### Learn more

    - **HSSM foundations:** [documentation](https://lnccbrown.github.io/HSSM/), [Quickstart](https://lnccbrown.github.io/HSSM/getting_started/getting_started/), and [hierarchical modeling](https://lnccbrown.github.io/HSSM/getting_started/hierarchical_modeling/)
    - **Applied workflows:** [plotting](https://lnccbrown.github.io/HSSM/tutorials/plotting/) and the [Scientific Workflow tutorial](https://lnccbrown.github.io/HSSM/tutorials/scientific_workflow_hssm/)
    - **Ecosystem tools:** [PyMC](https://www.pymc.io/), [Bambi](https://bambinos.github.io/bambi/), [ArviZ](https://python.arviz.org/en/stable/user_guide/getting_started.html), and [ssm-simulators](https://github.com/lnccbrown/ssm-simulators)
    - **Community and contributions:** [GitHub](https://github.com/lnccbrown/HSSM), [Discussions](https://github.com/lnccbrown/HSSM/discussions), and the [contribution guide](https://github.com/lnccbrown/HSSM/blob/main/docs/CONTRIBUTING.md)

    As you extend an analysis, keep returning to the same cycle: state the scientific question, make the model assumptions explicit, check sampler diagnostics and predictions, and communicate uncertainty alongside point estimates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A note on tutorial scale

    The examples use modest sampling settings so the notebook is practical to run. Before drawing scientific conclusions, increase the number of chains, draws, and tuning iterations; inspect convergence diagnostics; and perform model checks tailored to your data.
    """)
    return


if __name__ == "__main__":
    app.run()

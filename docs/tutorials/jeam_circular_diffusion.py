"""Compare JEAM's blackbox and analytical circular likelihoods through HSSM.

From the HSSM repository root, install the pinned prototype and open this marimo
notebook with::

    uv sync --group docs --group jeam-prototype
    uv run --group docs --group jeam-prototype \
        marimo edit docs/tutorials/jeam_circular_diffusion.py

Choose the likelihood and sampler in the notebook. The default two-chain fit is
intentionally small enough for an interactive tutorial. For a longer local run, set
``FULL_RUN=1`` before starting marimo. For a plumbing-only run, set
``JEAM_NOTEBOOK_SMOKE=1``. The repeated-recovery benchmark in
``docs/tutorials/jeam_repeated_recovery.py`` remains the scientific gate.
"""

# ruff: noqa: B018, D401, E501, PLR1711

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import importlib.metadata
    import json
    import logging
    import os
    import warnings
    from time import perf_counter

    import arviz as az
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pymc as pm
    import pytensor
    from jeam.Models.Circular import CircularDiffusionModel

    import hssm
    from hssm.integrations.jeam import simulate_circular_diffusion

    return (
        CircularDiffusionModel,
        az,
        hssm,
        importlib,
        json,
        logging,
        mo,
        np,
        os,
        pd,
        perf_counter,
        plt,
        pm,
        pytensor,
        simulate_circular_diffusion,
        warnings,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Compare JEAM likelihood routes through HSSM

    This tutorial exercises the complete prototype handshake: simulate angular
    responses with JEAM, construct **ordinary `hssm.HSSM` objects**, compare both
    registered likelihoods against direct JEAM, compile the analytical gradient,
    fit a selected likelihood/sampler route, and perform posterior predictive checks.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > **Experimental integration.** The current handshake covers JEAM's
    > **two-dimensional, fixed-boundary circular diffusion model (CDM)** only.
    > Responses are angles in radians on the half-open interval $[-\pi, \pi)$.
    > The prototype fixes $\sigma=1$, $s_v=0$, $s_t=0$, and threshold decay to
    > zero. It does not yet register spherical, hyperspherical, projected,
    > categorical, or ordinary continuous-response JEAM models in HSSM.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Install the pinned prototype

    In an HSSM checkout, the exact install command is:

    ```bash
    uv sync --group docs --group jeam-prototype
    ```

    The dependency group resolves the HSSM fork of JEAM at commit
    `0c0ef8b834dd062ad8aea5ff8e7a09dfb55492ce`. This is a VCS-pinned prototype,
    not a released HSSM runtime dependency.

    The default run below uses 120 trials and two short chains. Start marimo with
    `FULL_RUN=1` to use 300 trials and longer chains:

    ```bash
    FULL_RUN=1 uv run --group docs --group jeam-prototype \
      marimo edit docs/tutorials/jeam_circular_diffusion.py
    ```

    Automated or plumbing-only checks can use 24 trials and 10 tune plus 10
    posterior draws per chain:

    ```bash
    JEAM_NOTEBOOK_SMOKE=1 uv run --group docs --group jeam-prototype \
      marimo export html docs/tutorials/jeam_circular_diffusion.py \
      -o /tmp/jeam-circular-smoke.html
    ```
    """)
    return


@app.cell
def _(hssm, logging, os, pytensor, warnings):
    warnings.filterwarnings(
        "ignore",
        message=r"Numba will use object mode to run .*",
        category=UserWarning,
        module=r"pytensor\.link\.numba\.dispatch\.basic",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`init='adapt_diag'` is ignored by `nuts_sampler='numpyro'`.*",
        category=UserWarning,
        module=r"bambi\.backend\.pymc",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"There are not enough devices to run parallel chains:.*",
        category=UserWarning,
        module=r"pymc\.sampling\.jax",
    )
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
    hssm.set_floatX("float64")
    FULL_RUN = os.getenv("FULL_RUN", "0") == "1"
    SMOKE_RUN = os.getenv("JEAM_NOTEBOOK_SMOKE", "0") == "1"
    if FULL_RUN and SMOKE_RUN:
        raise RuntimeError("FULL_RUN and JEAM_NOTEBOOK_SMOKE cannot both be enabled.")
    RUN_CONFIG = {
        "mode": "full" if FULL_RUN else "smoke" if SMOKE_RUN else "tutorial",
        "trials": 300 if FULL_RUN else 24 if SMOKE_RUN else 120,
        "chains": 2,
        "tune": 1_500 if FULL_RUN else 10 if SMOKE_RUN else 500,
        "draws": 1_500 if FULL_RUN else 10 if SMOKE_RUN else 500,
        "prior_draws": 20 if FULL_RUN else 2 if SMOKE_RUN else 8,
        "predictive_draws": 100 if FULL_RUN else 2 if SMOKE_RUN else 30,
        "data_seed": 1_492,
        "chain_seeds": (3_101, 3_102),
        "prior_seed": 6_101,
        "predictive_seed": 7_291,
    }
    TRUTH = {"a": 1.20, "t": 0.15, "v_x": 0.80, "v_y": -0.50}
    PARAMETER_ORDER = ("a", "t", "v_x", "v_y")
    HSSM_PARAMETER_ORDER = ("v_x", "v_y", "a", "t")
    assert pytensor.config.floatX == "float64"
    return (
        FULL_RUN,
        HSSM_PARAMETER_ORDER,
        PARAMETER_ORDER,
        RUN_CONFIG,
        SMOKE_RUN,
        TRUTH,
    )


@app.cell
def _(importlib, json, mo):
    expected_jeam_revision = "0c0ef8b834dd062ad8aea5ff8e7a09dfb55492ce"
    _distribution = importlib.metadata.distribution("jeam")
    _direct_url_text = _distribution.read_text("direct_url.json")
    if _direct_url_text is None:
        raise RuntimeError("JEAM installation has no direct-url provenance metadata.")
    installed_jeam_revision = (
        json.loads(_direct_url_text).get("vcs_info", {}).get("commit_id")
    )
    if installed_jeam_revision != expected_jeam_revision:
        raise RuntimeError(
            "This tutorial requires JEAM revision "
            f"{expected_jeam_revision}, but found {installed_jeam_revision!r}. "
            "Run `uv sync --group docs --group jeam-prototype --locked`."
        )
    mo.callout(
        f"Verified installed JEAM revision `{installed_jeam_revision}`.",
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Simulate a known circular dataset

    The data-generating values deliberately create an asymmetric drift direction.
    The seeds are fixed so rerunning this notebook is deterministic.
    """)
    return


@app.cell
def _(RUN_CONFIG, mo, pd):
    _configuration = pd.DataFrame(
        [
            {"setting": key.replace("_", " "), "value": value}
            for key, value in RUN_CONFIG.items()
        ]
    )
    mo.ui.table(_configuration, selection=None, pagination=False)
    return


@app.cell
def _(RUN_CONFIG, TRUTH, np, pd, simulate_circular_diffusion):
    simulator_theta = np.array(
        [TRUTH["v_x"], TRUTH["v_y"], TRUTH["a"], TRUTH["t"]],
        dtype=np.float64,
    )
    observations = simulate_circular_diffusion(
        theta=simulator_theta,
        random_state=RUN_CONFIG["data_seed"],
        n_replicas=RUN_CONFIG["trials"],
    )
    data = pd.DataFrame(observations, columns=["rt", "response"])
    assert np.all(data["rt"] > 0.0)
    assert np.all((-np.pi <= data["response"]) & (data["response"] < np.pi))
    return data, observations


@app.cell
def _(RUN_CONFIG, mo):
    point_count = mo.ui.slider(
        start=20,
        stop=RUN_CONFIG["trials"],
        step=10,
        value=min(120, RUN_CONFIG["trials"]),
        label="Trials shown",
    )
    point_count
    return (point_count,)


@app.cell
def _(TRUTH, data, np, plt, point_count):
    _visible = data.iloc[: point_count.value]
    _figure = plt.figure(figsize=(8.5, 4.1))
    _polar = _figure.add_subplot(1, 2, 1, projection="polar")
    _polar.scatter(
        _visible["response"],
        _visible["rt"],
        s=24,
        alpha=0.65,
        color="#4472C4",
    )
    _drift_angle = np.arctan2(TRUTH["v_y"], TRUTH["v_x"])
    _polar.axvline(
        _drift_angle,
        color="#B22222",
        linewidth=2,
        label="drift direction",
    )
    _polar.set_title("Each response is an angle and an RT")
    _polar.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.18), fontsize=8)
    _rt_axis = _figure.add_subplot(1, 2, 2)
    _rt_axis.hist(data["rt"], bins=20, color="#9DC3E6", edgecolor="white")
    _rt_axis.axvline(TRUTH["t"], color="#B22222", linestyle="--", label="true t")
    _rt_axis.set(xlabel="response time", ylabel="trials", title="RT marginal")
    _rt_axis.legend()
    _figure.suptitle("Observed fixed-CDM data")
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    HSSM expects two columns in this order:

    | column | meaning | allowed values |
    |---|---|---|
    | `rt` | ordinary response time | finite and strictly positive |
    | `response` | angular coordinate in radians | $[-\pi, \pi)$ |

    The model has no finite list of choices: an angle is a circular continuous
    response, not a categorical label.
    """)
    return


@app.cell
def _(mo, os):
    inference_routes = {
        "Blackbox + PyMC Slice": "blackbox_pymc",
        "Analytical + PyMC NUTS": "analytical_pymc",
        "Analytical + NumPyro NUTS": "analytical_numpyro",
    }
    default_route = os.getenv("JEAM_INFERENCE_ROUTE", "blackbox_pymc")
    if default_route not in inference_routes.values():
        raise RuntimeError(
            "JEAM_INFERENCE_ROUTE must be one of "
            f"{tuple(inference_routes.values())}, not {default_route!r}."
        )
    default_route_label = next(
        label for label, value in inference_routes.items() if value == default_route
    )
    inference_route = mo.ui.dropdown(
        options=inference_routes,
        value=default_route_label,
        label="Likelihood and sampler",
    )
    inference_route
    return (inference_route,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Build both ordinary HSSM models

    Use the dropdown above to choose the route that will be sampled later. The
    notebook always constructs and numerically checks **both** likelihoods on the
    same observations:

    - `blackbox` calls JEAM's NumPy implementation and requires gradient-free Slice;
    - `analytical` evaluates JEAM's truncated first-passage series with JAX and
      exposes parameter gradients for NUTS.

    There is no `HSSMCircular` subclass. HSSM discovers each likelihood, the shared
    simulator, bounds, and response semantics from the registered
    `circular_diffusion` model configuration.
    """)
    return


@app.cell
def _(data, hssm, inference_route):
    blackbox_model = hssm.HSSM(
        data=data,
        model="circular_diffusion",
        loglik_kind="blackbox",
        p_outlier=None,
    )
    analytical_model = hssm.HSSM(
        data=data,
        model="circular_diffusion",
        loglik_kind="analytical",
        p_outlier=None,
    )
    models_by_likelihood = {
        "blackbox": blackbox_model,
        "analytical": analytical_model,
    }
    selected_likelihood, selected_sampler = inference_route.value.split("_", 1)
    circular_model = models_by_likelihood[selected_likelihood]
    return (
        analytical_model,
        blackbox_model,
        circular_model,
        selected_likelihood,
        selected_sampler,
    )


@app.cell
def _(mo):
    mo.md("""
    ### Selected HSSM contract
    """)
    return


@app.cell
def _(circular_model, hssm, inference_route, mo, np, pd, selected_sampler):
    supported_samplers = circular_model.model_config.supported_samplers
    model_contract = pd.DataFrame(
        [
            {"property": "selected route", "value": inference_route.selected_key},
            {"property": "Python class", "value": type(circular_model).__name__},
            {"property": "model name", "value": circular_model.model_name},
            {"property": "likelihood", "value": circular_model.loglik_kind},
            {"property": "sampler", "value": selected_sampler},
            {
                "property": "supported samplers",
                "value": ", ".join(supported_samplers or ()),
            },
            {"property": "response kind", "value": circular_model.response_kind},
            {
                "property": "angle bounds",
                "value": str(circular_model.response_bounds["response"]),
            },
            {"property": "parameters", "value": ", ".join(circular_model.list_params)},
            {"property": "categorical choices", "value": str(circular_model.choices)},
            {"property": "p_outlier", "value": "None (required)"},
        ]
    )
    assert type(circular_model) is hssm.HSSM
    assert circular_model.response_bounds == {"response": (-np.pi, np.pi)}
    assert selected_sampler in (supported_samplers or ())
    mo.ui.table(model_contract, selection=None, pagination=False)
    return


@app.cell
def _(RUN_CONFIG, data, hssm, selected_likelihood):
    # Keep prior draws separate from the HSSM object that will hold posterior traces.
    _prior_check_model = hssm.HSSM(
        data=data,
        model="circular_diffusion",
        loglik_kind=selected_likelihood,
        p_outlier=None,
    )
    prior_predictive = _prior_check_model.sample_prior_predictive(
        draws=RUN_CONFIG["prior_draws"],
        random_seed=RUN_CONFIG["prior_seed"],
    )
    prior_values = prior_predictive["prior_predictive"]["rt,response"].values
    return (prior_values,)


@app.cell
def _(data, np, plt, prior_values):
    _figure, _axes = plt.subplots(1, 2, figsize=(8.5, 3.7))
    _rt_limit = np.quantile(
        np.concatenate([data["rt"].to_numpy(), prior_values[..., 0].ravel()]),
        0.98,
    )
    _rt_bins = np.linspace(0.0, _rt_limit, 28)
    _axes[0].hist(
        prior_values[..., 0].ravel(),
        bins=_rt_bins,
        density=True,
        alpha=0.45,
        color="#ED7D31",
        label="prior predictive",
    )
    _axes[0].hist(
        data["rt"],
        bins=_rt_bins,
        density=True,
        histtype="step",
        linewidth=2,
        color="#111111",
        label="observed",
    )
    _axes[0].set(xlabel="response time", ylabel="density", title="RT scale")
    _axes[0].legend(fontsize=8)
    _angle_bins = np.linspace(-np.pi, np.pi, 25)
    _axes[1].hist(
        prior_values[..., 1].ravel(),
        bins=_angle_bins,
        density=True,
        alpha=0.45,
        color="#ED7D31",
        label="prior predictive",
    )
    _axes[1].hist(
        data["response"],
        bins=_angle_bins,
        density=True,
        histtype="step",
        linewidth=2,
        color="#111111",
        label="observed",
    )
    _axes[1].set(
        xlabel="response angle (radians)",
        ylabel="density",
        title="Circular response",
        xlim=(-np.pi, np.pi),
    )
    _axes[1].set_xticks([-np.pi, 0.0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
    _axes[1].legend(fontsize=8)
    _figure.suptitle("Prior predictive check")
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    The prior predictive plot is the first model check. It asks whether the default
    priors and simulator produce values on plausible RT and angular scales before the
    observed likelihood is used. For an applied analysis, replace default priors with
    domain-informed priors rather than treating this picture as a one-time ritual.

    ## 3. Verify both numerical handshakes

    The next cell evaluates every observation three ways at the known parameters:

    1. directly through JEAM's `CircularDiffusionModel.joint_lpdf`;
    2. through HSSM's registered `blackbox` distribution; and
    3. through HSSM's registered `analytical` distribution.

    It also compiles the analytical model's four-parameter gradient at HSSM's initial
    point and requires every component to be finite. Agreement checks the objectives
    themselves, rather than merely whether all three APIs return finite numbers.
    """)
    return


@app.cell
def _(
    CircularDiffusionModel,
    TRUTH,
    analytical_model,
    blackbox_model,
    np,
    observations,
    pd,
    pm,
):
    direct_jeam = CircularDiffusionModel(threshold_dynamic="fixed")
    direct_logp = np.asarray(
        direct_jeam.joint_lpdf(
            rt=observations[:, 0],
            theta=observations[:, 1],
            drift_vec=np.column_stack(
                (
                    np.full(len(observations), TRUTH["v_x"]),
                    np.full(len(observations), TRUTH["v_y"]),
                )
            ),
            ndt=np.full(len(observations), TRUTH["t"]),
            threshold=TRUTH["a"],
            decay=0.0,
            threshold_function=None,
            dt_threshold_function=None,
            s_v=0.0,
            s_t=0.0,
            sigma=1.0,
        ),
        dtype=np.float64,
    )
    blackbox_logp = np.asarray(
        blackbox_model.model_distribution.logp(
            observations,
            TRUTH["v_x"],
            TRUTH["v_y"],
            TRUTH["a"],
            TRUTH["t"],
        ).eval(),
        dtype=np.float64,
    )
    analytical_logp = np.asarray(
        analytical_model.model_distribution.logp(
            observations,
            TRUTH["v_x"],
            TRUTH["v_y"],
            TRUTH["a"],
            TRUTH["t"],
        ).eval(),
        dtype=np.float64,
    )
    logp_by_likelihood = {
        "blackbox": blackbox_logp,
        "analytical": analytical_logp,
    }
    logp_absolute_errors = {
        likelihood: np.abs(values - direct_logp)
        for likelihood, values in logp_by_likelihood.items()
    }
    _transformed_initial_point = pm.initial_point.make_initial_point_fn(
        model=analytical_model.pymc_model,
        overrides=analytical_model.initvals,
        return_transformed=True,
    )(None)
    analytical_gradient = np.asarray(
        analytical_model.pymc_model.compile_dlogp()(_transformed_initial_point),
        dtype=np.float64,
    )
    assert analytical_gradient.shape == (4,)
    assert np.all(np.isfinite(analytical_gradient))
    assert np.any(analytical_gradient != 0.0)

    likelihood_comparison = pd.DataFrame(
        [
            {
                "HSSM likelihood": likelihood,
                "summed log likelihood": values.sum(),
                "maximum error vs direct JEAM": logp_absolute_errors[likelihood].max(),
                "parameter gradient": (
                    "finite, nonzero (4 components)"
                    if likelihood == "analytical"
                    else "not exposed"
                ),
            }
            for likelihood, values in logp_by_likelihood.items()
        ]
    )
    for _values in logp_by_likelihood.values():
        np.testing.assert_allclose(_values, direct_logp, rtol=1e-6, atol=5e-5)
    return analytical_gradient, likelihood_comparison, logp_absolute_errors


@app.cell
def _(likelihood_comparison, mo):
    mo.ui.table(likelihood_comparison.round(10), selection=None, pagination=False)
    return


@app.cell
def _(inference_route, mo):
    mo.md(f"""
    ## 4. Fit the selected route through HSSM

    **Selected:** `{inference_route.selected_key}`.

    The blackbox route uses one explicit PyMC `Slice` step per parameter. The
    analytical routes use NUTS through either PyMC or NumPyro. Every route uses the
    same model formulas, priors, observed dataset, chain seeds, tune count, and draw
    count; only the likelihood implementation and sampler change.

    `cores=1` keeps the notebook portable and reproducible. Runtime here is
    descriptive, not a benchmark: JAX compilation and short-run warmup costs are a
    large fraction of an interactive run. The repeated recovery and efficiency gates
    remain separate from this workflow demonstration.
    """)
    return


@app.cell
def _(
    PARAMETER_ORDER,
    RUN_CONFIG,
    circular_model,
    perf_counter,
    pm,
    selected_likelihood,
    selected_sampler,
):
    _sample_kwargs = {
        "sampler": selected_sampler,
        "chains": RUN_CONFIG["chains"],
        "cores": 1,
        "tune": RUN_CONFIG["tune"],
        "draws": RUN_CONFIG["draws"],
        "random_seed": list(RUN_CONFIG["chain_seeds"]),
        "progressbar": False,
        "idata_kwargs": {"log_likelihood": False},
    }
    if selected_likelihood == "blackbox":
        _sample_kwargs["step"] = [
            pm.Slice(
                vars=[circular_model.pymc_model[name]],
                model=circular_model.pymc_model,
            )
            for name in PARAMETER_ORDER
        ]
    sampling_started = perf_counter()
    traces = circular_model.sample(**_sample_kwargs)
    sampling_seconds = perf_counter() - sampling_started
    return sampling_seconds, traces


@app.cell
def _(
    FULL_RUN,
    PARAMETER_ORDER,
    SMOKE_RUN,
    TRUTH,
    az,
    np,
    pd,
    sampling_seconds,
    selected_likelihood,
    selected_sampler,
    traces,
):
    posterior_summary = az.summary(
        traces,
        var_names=list(PARAMETER_ORDER),
        ci_prob=0.94,
        ci_kind="hdi",
        round_to=8,
    )
    lower_column = next(
        column
        for column in posterior_summary.columns
        if column.startswith("hdi") and column.endswith("_lb")
    )
    upper_column = next(
        column
        for column in posterior_summary.columns
        if column.startswith("hdi") and column.endswith("_ub")
    )
    recovery_table = posterior_summary.reset_index(names="parameter")
    recovery_table.insert(
        1,
        "truth",
        recovery_table["parameter"].map(TRUTH),
    )
    recovery_table["mean error"] = recovery_table["mean"] - recovery_table["truth"]
    recovery_table["truth in 94% HDI"] = (
        recovery_table[lower_column] <= recovery_table["truth"]
    ) & (recovery_table["truth"] <= recovery_table[upper_column])
    sample_stat_names = tuple(
        sorted(str(name) for name in traces["sample_stats"].data_vars)
    )
    nuts_statistics = {
        "acceptance_rate",
        "diverging",
        "n_steps",
        "step_size",
        "tree_depth",
    }
    sampler_statistics_ok = (
        sample_stat_names == ("nstep_in", "nstep_out")
        if selected_likelihood == "blackbox"
        else nuts_statistics <= set(sample_stat_names)
    )
    maximum_rhat = float(posterior_summary["r_hat"].max())
    minimum_bulk_ess = float(posterior_summary["ess_bulk"].min())
    diagnostic_limits = {
        "maximum R-hat": 1.01 if FULL_RUN else 1.05,
        "minimum bulk ESS": 400.0 if FULL_RUN else 100.0,
    }
    diagnostics_ok = (
        not SMOKE_RUN
        and np.isfinite(maximum_rhat)
        and maximum_rhat <= diagnostic_limits["maximum R-hat"]
        and minimum_bulk_ess >= diagnostic_limits["minimum bulk ESS"]
    )
    sampling_report = pd.DataFrame(
        [
            {"diagnostic": "likelihood", "observed": selected_likelihood},
            {"diagnostic": "sampler", "observed": selected_sampler},
            {"diagnostic": "sampler statistics", "observed": str(sample_stat_names)},
            {
                "diagnostic": "sampler statistics match route",
                "observed": sampler_statistics_ok,
            },
            {"diagnostic": "sampling seconds", "observed": sampling_seconds},
            {"diagnostic": "maximum R-hat", "observed": maximum_rhat},
            {"diagnostic": "minimum bulk ESS", "observed": minimum_bulk_ess},
            {
                "diagnostic": "mode-specific diagnostic screen",
                "observed": diagnostics_ok,
            },
        ]
    )
    assert sampler_statistics_ok
    return (
        diagnostic_limits,
        diagnostics_ok,
        lower_column,
        maximum_rhat,
        minimum_bulk_ess,
        posterior_summary,
        recovery_table,
        sampler_statistics_ok,
        sampling_report,
        upper_column,
    )


@app.cell
def _(
    SMOKE_RUN,
    diagnostic_limits,
    diagnostics_ok,
    maximum_rhat,
    minimum_bulk_ess,
    mo,
):
    _message = (
        "Smoke mode only checks execution and sampler identity; 10 draws per chain "
        "cannot support convergence or recovery claims."
        if SMOKE_RUN
        else (
            f"The diagnostic screen passed: maximum R-hat is {maximum_rhat:.3f} "
            f"(limit {diagnostic_limits['maximum R-hat']:.2f}) and minimum bulk ESS "
            f"is {minimum_bulk_ess:.0f} "
            f"(limit {diagnostic_limits['minimum bulk ESS']:.0f})."
            if diagnostics_ok
            else f"Treat the posterior as provisional: maximum R-hat is "
            f"{maximum_rhat:.3f} and minimum bulk ESS is {minimum_bulk_ess:.0f}. "
            "Run FULL_RUN=1, inspect the chains, and revise the model before "
            "substantive interpretation."
        )
    )
    mo.callout(
        _message,
        kind="info" if SMOKE_RUN else "success" if diagnostics_ok else "warn",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Diagnose before interpreting
    """)
    return


@app.cell
def _(mo, sampling_report):
    mo.ui.table(sampling_report.round(4), selection=None, pagination=False)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Parameter recovery in this one illustrative dataset
    """)
    return


@app.cell
def _(lower_column, mo, recovery_table, upper_column):
    mo.ui.table(
        recovery_table[
            [
                "parameter",
                "truth",
                "mean",
                "mean error",
                lower_column,
                upper_column,
                "truth in 94% HDI",
                "ess_bulk",
                "r_hat",
            ]
        ].round(4),
        selection=None,
        pagination=False,
    )
    return


@app.cell
def _(
    PARAMETER_ORDER,
    TRUTH,
    lower_column,
    plt,
    posterior_summary,
    upper_column,
):
    _positions = list(range(len(PARAMETER_ORDER)))
    _means = posterior_summary.loc[list(PARAMETER_ORDER), "mean"].to_numpy()
    _lower = posterior_summary.loc[list(PARAMETER_ORDER), lower_column].to_numpy()
    _upper = posterior_summary.loc[list(PARAMETER_ORDER), upper_column].to_numpy()
    _truth = [TRUTH[name] for name in PARAMETER_ORDER]
    _figure, _axis = plt.subplots(figsize=(8, 3.8))
    _axis.errorbar(
        _means,
        _positions,
        xerr=[_means - _lower, _upper - _means],
        fmt="s",
        capsize=3,
        color="#4472C4",
        label="posterior mean and 94% HDI",
    )
    _axis.scatter(
        _truth,
        _positions,
        marker="x",
        s=80,
        linewidth=2,
        color="#B22222",
        label="generating truth",
        zorder=4,
    )
    _axis.set_yticks(_positions, PARAMETER_ORDER)
    _axis.invert_yaxis()
    _axis.set(xlabel="parameter value", title="Recovery in one simulated dataset")
    _axis.grid(axis="x", alpha=0.2)
    _axis.legend(loc="best")
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(RUN_CONFIG, circular_model, perf_counter, traces):
    predictive_started = perf_counter()
    posterior_predictive = circular_model.sample_posterior_predictive(
        dt=traces,
        inplace=False,
        kind="response",
        draws=RUN_CONFIG["predictive_draws"],
        safe_mode=True,
        random_seed=RUN_CONFIG["predictive_seed"],
    )
    predictive_seconds = perf_counter() - predictive_started
    if posterior_predictive is None:
        raise RuntimeError("Expected non-inplace posterior predictive output.")
    posterior_predictive_values = posterior_predictive["posterior_predictive"][
        "rt,response"
    ].values
    return posterior_predictive_values, predictive_seconds


@app.cell
def _(mo):
    predictive_view = mo.ui.dropdown(
        options={
            "Response time and angle": "both",
            "Response time only": "rt",
            "Angle only": "angle",
        },
        value="Response time and angle",
        label="Posterior predictive view",
    )
    predictive_view
    return (predictive_view,)


@app.cell
def _(data, np, plt, posterior_predictive_values, predictive_view):
    _show_rt = predictive_view.value in {"both", "rt"}
    _show_angle = predictive_view.value in {"both", "angle"}
    _panels = int(_show_rt) + int(_show_angle)
    _figure, _axes_array = plt.subplots(1, _panels, figsize=(4.4 * _panels, 3.7))
    _axes = np.atleast_1d(_axes_array)
    _axis_index = 0
    if _show_rt:
        _axis = _axes[_axis_index]
        _rt_limit = np.quantile(
            np.concatenate(
                [data["rt"].to_numpy(), posterior_predictive_values[..., 0].ravel()]
            ),
            0.99,
        )
        _bins = np.linspace(0.0, _rt_limit, 30)
        _axis.hist(
            posterior_predictive_values[..., 0].ravel(),
            bins=_bins,
            density=True,
            alpha=0.5,
            color="#4472C4",
            label="posterior predictive",
        )
        _axis.hist(
            data["rt"],
            bins=_bins,
            density=True,
            histtype="step",
            linewidth=2,
            color="#111111",
            label="observed",
        )
        _axis.set(xlabel="response time", ylabel="density", title="RT distribution")
        _axis.legend(fontsize=8)
        _axis_index += 1
    if _show_angle:
        _axis = _axes[_axis_index]
        _bins = np.linspace(-np.pi, np.pi, 25)
        _axis.hist(
            posterior_predictive_values[..., 1].ravel(),
            bins=_bins,
            density=True,
            alpha=0.5,
            color="#4472C4",
            label="posterior predictive",
        )
        _axis.hist(
            data["response"],
            bins=_bins,
            density=True,
            histtype="step",
            linewidth=2,
            color="#111111",
            label="observed",
        )
        _axis.set(
            xlabel="response angle (radians)",
            ylabel="density",
            title="Circular response",
            xlim=(-np.pi, np.pi),
        )
        _axis.set_xticks([-np.pi, 0.0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
        _axis.legend(fontsize=8)
    _figure.suptitle("Posterior predictive check")
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(data, np, pd, posterior_predictive_values, predictive_seconds):
    def _circular_summary(values):
        _resultant = np.mean(np.exp(1j * np.asarray(values)))
        return float(np.angle(_resultant)), float(np.abs(_resultant))

    _observed_angle, _observed_length = _circular_summary(data["response"])
    _predictive_angle, _predictive_length = _circular_summary(
        posterior_predictive_values[..., 1]
    )
    _probabilities = np.array([0.1, 0.5, 0.9])
    _observed_rt = np.quantile(data["rt"], _probabilities)
    _predictive_rt = np.quantile(posterior_predictive_values[..., 0], _probabilities)
    predictive_summary = pd.DataFrame(
        [
            *[
                {
                    "quantity": f"RT q{int(probability * 100)}",
                    "observed": observed,
                    "posterior predictive": predicted,
                    "absolute difference": abs(observed - predicted),
                }
                for probability, observed, predicted in zip(
                    _probabilities, _observed_rt, _predictive_rt, strict=True
                )
            ],
            {
                "quantity": "circular mean angle",
                "observed": _observed_angle,
                "posterior predictive": _predictive_angle,
                "absolute difference": abs(
                    np.angle(np.exp(1j * (_observed_angle - _predictive_angle)))
                ),
            },
            {
                "quantity": "mean resultant length",
                "observed": _observed_length,
                "posterior predictive": _predictive_length,
                "absolute difference": abs(_observed_length - _predictive_length),
            },
            {
                "quantity": "predictive runtime (seconds)",
                "observed": np.nan,
                "posterior predictive": predictive_seconds,
                "absolute difference": np.nan,
            },
        ]
    )
    return (predictive_summary,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Check posterior predictions

    The visual check compares both components of the joint response. The table adds
    RT quantiles and circular summaries; an arithmetic mean of angles would be wrong
    near the minus-pi/pi seam.
    """)
    return


@app.cell
def _(mo, predictive_summary):
    mo.ui.table(predictive_summary.round(4), selection=None, pagination=False)
    return


@app.cell
def _(
    HSSM_PARAMETER_ORDER,
    analytical_gradient,
    circular_model,
    hssm,
    logp_absolute_errors,
    np,
    pd,
    posterior_predictive_values,
    sampler_statistics_ok,
):
    predictive_support_ok = bool(
        np.all(np.isfinite(posterior_predictive_values))
        and np.all(posterior_predictive_values[..., 0] > 0.0)
        and np.all(posterior_predictive_values[..., 1] >= -np.pi)
        and np.all(posterior_predictive_values[..., 1] < np.pi)
    )
    both_likelihoods_match = all(
        float(errors.max()) <= 5e-5 for errors in logp_absolute_errors.values()
    )
    analytical_gradient_ok = bool(
        analytical_gradient.shape == (4,)
        and np.all(np.isfinite(analytical_gradient))
        and np.any(analytical_gradient != 0.0)
    )
    validation_table = pd.DataFrame(
        [
            {
                "workflow gate": "ordinary HSSM class",
                "result": type(circular_model) is hssm.HSSM,
            },
            {
                "workflow gate": "registered parameter order",
                "result": tuple(circular_model.list_params) == HSSM_PARAMETER_ORDER,
            },
            {
                "workflow gate": "both HSSM likelihoods vs direct JEAM",
                "result": both_likelihoods_match,
            },
            {
                "workflow gate": "finite nonzero analytical gradient",
                "result": analytical_gradient_ok,
            },
            {
                "workflow gate": "selected sampler statistics",
                "result": sampler_statistics_ok,
            },
            {
                "workflow gate": "posterior predictive support",
                "result": predictive_support_ok,
            },
        ]
    )
    assert validation_table["result"].all()
    return (validation_table,)


@app.cell
def _(mo):
    mo.md("""
    ## What this run established
    """)
    return


@app.cell
def _(mo, validation_table):
    mo.ui.table(validation_table, selection=None, pagination=False)
    return


@app.cell
def _(inference_route, mo):
    mo.md(rf"""
    The notebook has now checked both registered HSSM likelihoods against direct JEAM,
    verified that the analytical route exposes a finite nonzero parameter gradient,
    and completed inference with **{inference_route.selected_key}**. Predictive sampling
    called JEAM's simulator through the selected ordinary HSSM object.

    Before using this prototype for substantive work:

    1. run `FULL_RUN=1` for each route you intend to evaluate and require satisfactory
       diagnostics for your data;
    2. choose and justify domain-specific priors;
    3. compare posterior predictions to all scientifically important summaries;
    4. consult `docs/tutorials/jeam_repeated_recovery.py` for the predeclared
       multi-dataset recovery evidence; and
    5. retain the exact JEAM revision in the analysis environment.

    The integration remains deliberately narrow: fixed CDM, two-dimensional angular
    responses, and no lapse mixture. Successful NUTS execution here is a capability
    check, not yet a scientific recovery or efficiency recommendation.
    """)
    return


if __name__ == "__main__":
    app.run()

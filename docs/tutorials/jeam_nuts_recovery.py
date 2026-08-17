"""Inspect authenticated compact evidence from the fixed-CDM sampler study.

This marimo report calls the evidence verifier through one reporting API. It imports
neither HSSM, JEAM, nor PyMC and does not access the network, optimize, or sample. Its
tables and empirical figures present authenticated compact summaries; missing raw draws
cannot be reconstructed here.

Open it from the HSSM repository root::

    uv run --group docs marimo edit docs/tutorials/jeam_nuts_recovery.py

Export a path-clean static rendering with::

    uv run --group docs marimo export html docs/tutorials/jeam_nuts_recovery.py \
        -o /tmp/jeam-nuts-recovery.html --force --no-include-code
"""

# ruff: noqa: B018, D401, E501, PLR1711
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys as _sys
    from pathlib import Path as _Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    _source = _Path(globals().get("__file__", _Path.cwd())).resolve()
    _repo_root = next(
        _candidate
        for _candidate in (_source, *_source.parents)
        if (_candidate / "pyproject.toml").is_file()
    )
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))

    from scripts.jeam_sampler_comparison_report import (
        SAMPLER_COLORS,
        SAMPLER_LABELS,
        SAMPLER_ORDER,
        load_sampler_comparison_report,
    )

    return (
        SAMPLER_COLORS,
        SAMPLER_LABELS,
        SAMPLER_ORDER,
        load_sampler_comparison_report,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Fixed-CDM sampler evidence: an authenticated historical snapshot

    This report asks a bounded historical question: what did the preregistered,
    fixed two-dimensional circular-diffusion study record for a blackbox Slice route
    and two analytical NUTS routes?

    It is an **authenticated compact-only smoke benchmark**, not a rerun and not a
    durable posterior archive. The routes below are historical candidates that
    demonstrate capabilities under one protocol. They are not current defaults or
    recommendations.
    """)
    return


@app.cell
def _(load_sampler_comparison_report):
    _report = load_sampler_comparison_report()
    verification = _report["verification"]
    specification = _report["specification"]
    compact_result = _report["compact_result"]
    summary = _report["summary"]
    parameter_records = _report["parameter_records"]
    fit_records = _report["fit_records"]
    objective_records = _report["objective_records"]
    predictive_records = _report["predictive_records"]
    efficiency_records = _report["efficiency_records"]
    evidence_boundary = _report["evidence_boundary"]
    return (
        compact_result,
        efficiency_records,
        evidence_boundary,
        fit_records,
        objective_records,
        parameter_records,
        predictive_records,
        specification,
        summary,
        verification,
    )


@app.cell
def _(
    efficiency_records,
    fit_records,
    objective_records,
    parameter_records,
    pd,
    predictive_records,
    specification,
):
    parameter_df = pd.DataFrame(parameter_records)
    fit_df = pd.DataFrame(fit_records)
    objective_df = pd.DataFrame(objective_records)
    predictive_df = pd.DataFrame(predictive_records)
    efficiency_df = pd.DataFrame(efficiency_records)
    canonical_scenarios = [
        row["name"] for row in specification["scenarios"] if row["role"] == "canonical"
    ]
    return (
        canonical_scenarios,
        efficiency_df,
        fit_df,
        objective_df,
        parameter_df,
        predictive_df,
    )


@app.cell
def _(evidence_boundary, mo, summary, verification):
    _science = "PASS" if verification["scientific_gate"]["passed"] else "FAIL"
    _machine = (
        "PASS" if verification["recorded_machine_efficiency"]["passed"] else "FAIL"
    )
    _promotion = (
        "BLOCKED" if evidence_boundary["ecosystem_promotion_blocked"] else "OPEN"
    )
    mo.md(f"""
    ## Result at a glance

    | Authenticated compact gate | Fits | Unique generating truths | Route-specific 94% HDI checks | Reported NUTS divergences | Historical recorded-machine thresholds | Ecosystem promotion |
    |---|---:|---:|---:|---:|---|---|
    | **{_science}** | {summary["fit_count"]} | {summary["unique_truth_count"]} | {summary["route_hdi_inclusions"]}/{summary["route_hdi_checks"]} | {summary["nuts_divergences"]} | **{_machine}** | **{_promotion}** |

    Authenticated headline: 15 fits; 16 unique truths; 48/48 route HDI checks
    (48 route-specific checks); 0 reported NUTS divergences; and a historical
    recorded-machine threshold pass. Promotion remains blocked.

    “PASS” here means that the retained compact rows satisfy the frozen smoke-test
    arithmetic. It does not turn those rows into raw posterior evidence, establish
    portable performance, or validate the current JEAM revision.
    """)
    return


@app.cell
def _(compact_result, evidence_boundary, mo, pd):
    _revisions = evidence_boundary["jeam_revisions"]
    _retention = evidence_boundary["retention"]
    _retention_rows = [
        {
            "evidence": "posterior traces",
            "retained": f"{_retention['raw_trace_files_retained']} of 15",
            "consequence": "posterior diagnostics and sampler backend identity cannot be independently recomputed",
        },
        {
            "evidence": "raw prior-predictive draws",
            "retained": "no",
            "consequence": "prior-predictive compact summaries cannot be independently recomputed",
        },
        {
            "evidence": "raw posterior-predictive draws",
            "retained": "no",
            "consequence": "posterior-predictive compact summaries cannot be independently recomputed",
        },
        {
            "evidence": "sampler backend trace attributes",
            "retained": "no",
            "consequence": "the recorded backend labels cannot be checked against trace metadata",
        },
        {
            "evidence": "historical uv.lock bytes",
            "retained": "no",
            "consequence": "the recorded environment hash cannot recreate the historical lockfile",
        },
        {
            "evidence": "1,500-trial scale input",
            "retained": "post-hoc reconstruction only",
            "consequence": "the historical scale dataset bytes themselves are absent",
        },
    ]
    mo.vstack(
        [
            mo.md(f"""
            ## Provenance and evidence boundary

            The three JEAM revisions have different roles and must not be collapsed:

            - historical analytical result: `{_revisions["historical_analytical_result"]}`;
            - durable blackbox reference: `{_revisions["durable_blackbox_reference"]}`;
            - current safety revision: `{_revisions["current_safety_revision"]}` — **not rerun** by this study.

            The recorded scope was PyTensor `float64` with JAX x64 enabled on
            `{compact_result["provenance"]["platform"]}` using the JAX CPU backend. In
            plain terms: this is one macOS ARM recorded-machine result, not a portable
            timing claim. Trace hashes and sizes were recorded, but zero posterior trace
            files remain.
            """),
            mo.ui.table(
                pd.DataFrame(_retention_rows), selection=None, pagination=False
            ),
        ]
    )
    return


@app.cell
def _(SAMPLER_LABELS, compact_result, mo, pd, specification):
    _sampler_specs = {row["id"]: row for row in specification["samplers"]}
    _rows = []
    for _sampler in specification["execution"]["sampler_order"]:
        _sampler_spec = _sampler_specs[_sampler]
        _rows.append(
            {
                "historical route": SAMPLER_LABELS[_sampler],
                "likelihood": _sampler_spec["likelihood"],
                "backend": _sampler_spec["backend"],
                "step": _sampler_spec["step"],
                "chains": compact_result["execution"]["chains"],
                "tune": compact_result["execution"]["tune"],
                "draws": compact_result["execution"]["draws"],
            }
        )
    mo.vstack(
        [
            mo.md("""
            ## Frozen study design

            Four deterministic 300-trial scenarios were designated for recovery, and
            one nested 1,500-trial scenario was designated for scaling. Crossing five
            scenarios with three routes produced the 15 recorded fits. The compact
            result says each historical route used the same scenario input, priors,
            initial values, four chains, and fresh subprocess policy.

            The protocol intended to save each trace before summarization, but those
            trace files are not retained. Consequently, the report can authenticate the
            compact rows while neither recreating nor independently deriving them.
            """),
            mo.ui.table(pd.DataFrame(_rows), selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(canonical_scenarios, mo, specification):
    scenario_selector = mo.ui.dropdown(
        options={name.replace("_", " ").title(): name for name in canonical_scenarios},
        value=canonical_scenarios[0].replace("_", " ").title(),
        label="Scenario",
    )
    parameter_selector = mo.ui.dropdown(
        options={
            name: name for name in specification["scope"]["scientific_parameter_order"]
        },
        value=specification["scope"]["scientific_parameter_order"][0],
        label="Parameter",
    )
    mo.hstack([scenario_selector, parameter_selector], justify="start", gap=2)
    return parameter_selector, scenario_selector


@app.cell
def _(mo):
    mo.md(r"""
    ## Why the analytical likelihood is differentiable *almost everywhere*

    This conceptual figure explains the historical analytical route; it is not a
    benchmark recomputation. For fixed diffusion scale $\sigma=1$, define scaled
    decision time

    $$s = \frac{rt-t}{a^2}.$$

    The implementation uses a stable short-time representation for $s\leq0.002$, a
    long-time Bessel-series representation for $s\geq0.02$, and a linear density-space
    blend between them:

    $$w(s)=\operatorname{clip}\!\left(\frac{s-0.002}{0.02-0.002},0,1\right).$$

    Inside the overlap, the long series is faded in according to its signal above a
    conservative summation-error bound $E$:

    $$r=\operatorname{clip}\!\left(\frac{\log(S/E)}{\log 100},0,1\right).$$

    The likelihood is smooth within each open region and intentionally non-smooth on
    support surfaces, blend knots, and reliability clipping surfaces. The historical
    capability claim is finite, accurate gradients at ordinary interior points—not a
    derivative at every exact boundary.
    """)
    return


@app.cell
def _(np, plt):
    _scaled_time = np.linspace(0.0, 0.026, 400)
    _blend = np.clip((_scaled_time - 0.002) / (0.02 - 0.002), 0.0, 1.0)
    _signal_ratio = np.logspace(-1, 3, 400)
    _reliability = np.clip(np.log(_signal_ratio) / np.log(100.0), 0.0, 1.0)

    _figure, (_blend_axis, _reliability_axis) = plt.subplots(1, 2, figsize=(9, 3.7))
    _blend_axis.plot(_scaled_time, _blend, color="#4472C4", linewidth=2.5)
    _blend_axis.axvline(0.002, color="#B22222", linestyle="--", linewidth=1)
    _blend_axis.axvline(0.02, color="#B22222", linestyle="--", linewidth=1)
    _blend_axis.set_xlabel(r"scaled decision time $s$")
    _blend_axis.set_ylabel("long-time blend weight")
    _blend_axis.set_title("Conceptual representation blend")
    _blend_axis.grid(alpha=0.2)

    _reliability_axis.semilogx(
        _signal_ratio, _reliability, color="#E67E22", linewidth=2.5
    )
    _reliability_axis.axvline(1.0, color="#B22222", linestyle="--", linewidth=1)
    _reliability_axis.axvline(100.0, color="#B22222", linestyle="--", linewidth=1)
    _reliability_axis.set_xlabel("long-series signal / error bound")
    _reliability_axis.set_ylabel("reliability fade weight")
    _reliability_axis.set_title("Conceptual reliability fade")
    _reliability_axis.grid(alpha=0.2)
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell
def _(compact_result, mo, objective_df, plt, specification):
    _scenario_error = objective_df.groupby("scenario", sort=False)[
        "maximum_absolute_error"
    ].max()
    _threshold = specification["objective_parity"]["maximum_absolute_error"]
    _figure, _axis = plt.subplots(figsize=(8.5, 3.8))
    _axis.barh(
        [name.replace("_", " ") for name in _scenario_error.index],
        _scenario_error.values,
        color="#70AD47",
    )
    _axis.axvline(_threshold, color="#B22222", linestyle="--", label="frozen limit")
    _axis.set_xscale("log")
    _axis.set_xlabel("recorded maximum absolute objective difference")
    _axis.set_title("Authenticated compact objective summaries")
    _axis.legend()
    _figure.tight_layout()
    mo.vstack(
        [
            mo.md(f"""
            ## Objective and gradient summaries

            The figure presents authenticated compact summaries for 15 preregistered
            parameter points; it does not reevaluate JEAM, HSSM, or any objective. The
            largest recorded discrepancy is
            `{objective_df["maximum_absolute_error"].max():.2e}`, against the frozen
            `{_threshold:.0e}` limit.

            The compact fit rows also report finite, nonzero four-coordinate gradients
            for each analytical fit before sampling. Those gradients and compile times
            are historical summaries, not recomputed derivatives. Their producer is the
            historical JEAM revision
            `{compact_result["provenance"]["jeam_revision"]}`.
            """),
            _figure,
        ]
    )
    return


@app.cell
def _(SAMPLER_LABELS, compact_result, mo, np, pd, scenario_selector):
    _rows = []
    for _fit in compact_result["fits"]:
        if (
            _fit["scenario"] == scenario_selector.value
            and _fit["initial_gradient"] is not None
        ):
            _gradient = np.asarray(_fit["initial_gradient"])
            _rows.append(
                {
                    "historical route": SAMPLER_LABELS[_fit["sampler"]],
                    "reported finite coordinates": int(np.isfinite(_gradient).sum()),
                    "reported nonzero coordinates": int(np.count_nonzero(_gradient)),
                    "reported gradient L2 norm": float(np.linalg.norm(_gradient)),
                    "reported compile + first evaluation (s)": _fit["runtime_seconds"][
                        "gradient_compile_and_first_eval"
                    ],
                }
            )
    mo.ui.table(pd.DataFrame(_rows).round(4), selection=None, pagination=False)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Recorded posterior recovery summaries

    Select a parameter above. The plot subtracts each scenario's generating truth, so
    zero denotes the truth even though the four datasets use different values. Points
    and 94% HDIs come from authenticated compact posterior summaries. Because all 15
    posterior traces are missing, the notebook cannot recompute any mean, interval, or
    route-specific coverage decision from raw draws.

    There are 16 unique scenario–parameter truths. Repeating each across three routes
    produces 48 route-specific HDI checks; it does not create 48 independent truths.
    """)
    return


@app.cell
def _(
    SAMPLER_COLORS,
    SAMPLER_LABELS,
    SAMPLER_ORDER,
    canonical_scenarios,
    np,
    parameter_df,
    parameter_selector,
    plt,
):
    _selected = parameter_df.loc[parameter_df["parameter"] == parameter_selector.value]
    _positions = np.arange(len(canonical_scenarios), dtype=float)
    _offsets = {"slice": -0.22, "pymc_nuts": 0.0, "numpyro_nuts": 0.22}
    _figure, _axis = plt.subplots(figsize=(8.5, 4.5))
    for _sampler in SAMPLER_ORDER:
        _rows = (
            _selected.loc[_selected["sampler"] == _sampler]
            .set_index("scenario")
            .loc[canonical_scenarios]
        )
        _means = _rows["posterior_error"].to_numpy()
        _lower = _rows["relative_hdi_lower"].to_numpy()
        _upper = _rows["relative_hdi_upper"].to_numpy()
        _axis.errorbar(
            _means,
            _positions + _offsets[_sampler],
            xerr=np.vstack((_means - _lower, _upper - _means)),
            fmt="o",
            capsize=3,
            linewidth=1.8,
            color=SAMPLER_COLORS[_sampler],
            label=SAMPLER_LABELS[_sampler],
        )
    _axis.axvline(0.0, color="#111111", linestyle="--", linewidth=1.2)
    _axis.set_yticks(
        _positions, [name.replace("_", " ") for name in canonical_scenarios]
    )
    _axis.invert_yaxis()
    _axis.set_xlabel("recorded posterior summary minus generating truth")
    _axis.set_title(
        f"Authenticated compact recovery summary: {parameter_selector.value}"
    )
    _axis.grid(axis="x", alpha=0.2)
    _axis.legend(fontsize=8)
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell
def _(mo, parameter_df, scenario_selector):
    _columns = {
        "route": "historical route",
        "parameter": "parameter",
        "truth": "truth",
        "posterior_mean": "recorded posterior mean",
        "hdi_lower": "recorded 94% HDI lower",
        "hdi_upper": "recorded 94% HDI upper",
        "truth_in_hdi": "recorded coverage",
        "rhat": "recorded R-hat",
        "ess_bulk": "recorded bulk ESS",
        "ess_tail": "recorded tail ESS",
        "mcse_over_posterior_sd": "recorded MCSE/SD",
    }
    _selected = (
        parameter_df.loc[parameter_df["scenario"] == scenario_selector.value]
        .loc[:, list(_columns)]
        .rename(columns=_columns)
        .round(4)
    )
    mo.ui.table(_selected, selection=None, pagination=False)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Recorded diagnostic summaries

    Each bar is the worst parameter-level value stored in a selected compact fit. Red
    lines show frozen protocol boundaries. These are authenticated reported diagnostics,
    not diagnostics recomputed from chains. In particular, “0 reported divergences” is
    what the compact NUTS rows say; missing raw traces and backend trace attributes
    prevent independent chain- and backend-level verification.
    """)
    return


@app.cell
def _(SAMPLER_COLORS, SAMPLER_ORDER, fit_df, np, plt, scenario_selector, specification):
    _selected = (
        fit_df.loc[
            (fit_df["scenario"] == scenario_selector.value)
            & (fit_df["role"] == "canonical")
        ]
        .set_index("sampler")
        .loc[list(SAMPLER_ORDER)]
    )
    _labels = ["Slice", "PyMC\nNUTS", "NumPyro\nNUTS"]
    _colors = [SAMPLER_COLORS[name] for name in SAMPLER_ORDER]
    _thresholds = specification["scientific_acceptance"]
    _figure, (_rhat_axis, _ess_axis, _mcse_axis) = plt.subplots(1, 3, figsize=(10, 3.8))
    _rhat_axis.bar(_labels, _selected["maximum_rhat"], color=_colors)
    _rhat_axis.axhline(
        _thresholds["maximum_rhat_exclusive"], color="#B22222", linestyle="--"
    )
    _rhat_axis.set_ylim(0.99, 1.011)
    _rhat_axis.set_title("Recorded maximum R-hat")

    _x = np.arange(len(_labels))
    _width = 0.34
    _ess_axis.bar(
        _x - _width / 2,
        _selected["minimum_bulk_ess"],
        _width,
        label="bulk",
        color="#5B9BD5",
    )
    _ess_axis.bar(
        _x + _width / 2,
        _selected["minimum_tail_ess"],
        _width,
        label="tail",
        color="#A5A5A5",
    )
    _ess_axis.axhline(
        _thresholds["minimum_bulk_ess_exclusive"],
        color="#B22222",
        linestyle="--",
    )
    _ess_axis.set_xticks(_x, _labels)
    _ess_axis.set_title("Recorded minimum ESS")
    _ess_axis.legend(fontsize=8)

    _mcse_axis.bar(_labels, _selected["maximum_mcse_over_sd"], color=_colors)
    _mcse_axis.axhline(
        _thresholds["maximum_mcse_over_posterior_sd_exclusive"],
        color="#B22222",
        linestyle="--",
    )
    _mcse_axis.set_title("Recorded maximum MCSE / SD")
    for _axis in (_rhat_axis, _ess_axis, _mcse_axis):
        _axis.grid(axis="y", alpha=0.2)
        _axis.tick_params(axis="x", labelsize=8)
    _figure.suptitle("Authenticated compact diagnostic summaries", y=1.03)
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell
def _(mo):
    mo.md("""
    ## Recorded posterior-predictive summaries

    The compact rows store RT-quantile errors, circular mean-angle distance, and
    mean-resultant-length error. The figure divides those stored errors by their frozen
    limits. It does not regenerate posterior prediction: no raw posterior-predictive
    draws were retained, so these summaries cannot be independently recomputed.
    """)
    return


@app.cell
def _(
    SAMPLER_COLORS,
    SAMPLER_LABELS,
    SAMPLER_ORDER,
    mo,
    np,
    plt,
    predictive_df,
    scenario_selector,
):
    _selected = (
        predictive_df.loc[predictive_df["scenario"] == scenario_selector.value]
        .set_index("sampler")
        .loc[list(SAMPLER_ORDER)]
    )
    _metrics = [
        "rt_fraction_of_limit",
        "angle_fraction_of_limit",
        "resultant_fraction_of_limit",
    ]
    _labels = ["RT quantiles", "mean angle", "resultant length"]
    _x = np.arange(len(_metrics))
    _width = 0.24
    _figure, _axis = plt.subplots(figsize=(8.5, 3.9))
    for _index, _sampler in enumerate(SAMPLER_ORDER):
        _axis.bar(
            _x + (_index - 1) * _width,
            _selected.loc[_sampler, _metrics],
            _width,
            color=SAMPLER_COLORS[_sampler],
            label=SAMPLER_LABELS[_sampler],
        )
    _axis.axhline(1.0, color="#B22222", linestyle="--", label="frozen limit")
    _axis.set_xticks(_x, _labels)
    _axis.set_ylabel("recorded compact error / frozen limit")
    _axis.set_title("Authenticated compact predictive summaries")
    _axis.grid(axis="y", alpha=0.2)
    _axis.legend(fontsize=8, ncol=2)
    _figure.tight_layout()
    mo.vstack(
        [
            _figure,
            mo.ui.table(
                _selected.reset_index()[
                    [
                        "route",
                        "maximum_rt_quantile_error",
                        "mean_angle_error",
                        "resultant_length_error",
                    ]
                ].round(4),
                selection=None,
                pagination=False,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Historical recorded-machine efficiency

    The frozen protocol compared each NUTS route with Slice using minimum bulk ESS per
    sampling second, total time, and a 1,500-trial normalized-efficiency check. The
    authenticated compact summaries pass those historical thresholds on the recorded
    macOS ARM machine. They are neither portable benchmarks nor grounds for ecosystem
    promotion.
    """)
    return


@app.cell
def _(efficiency_df, mo):
    _table = efficiency_df[
        [
            "route",
            "efficiency_ratio_vs_slice",
            "maximum_total_time_ratio_vs_slice",
            "scale_normalized_efficiency_ratio",
            "recorded_machine_gate_passed",
        ]
    ].rename(
        columns={
            "route": "historical route",
            "efficiency_ratio_vs_slice": "recorded median min bulk ESS/s vs Slice",
            "maximum_total_time_ratio_vs_slice": "recorded worst total-time ratio vs Slice",
            "scale_normalized_efficiency_ratio": "recorded 1,500-trial normalized efficiency",
            "recorded_machine_gate_passed": "historical threshold pass",
        }
    )
    mo.ui.table(_table.round(3), selection=None, pagination=False)
    return


@app.cell
def _(SAMPLER_COLORS, efficiency_df, np, plt, specification):
    _samplers = efficiency_df["sampler"].tolist()
    _labels = ["PyMC NUTS", "NumPyro NUTS"]
    _colors = [SAMPLER_COLORS[name] for name in _samplers]
    _limits = specification["promotion_decision"]
    _x = np.arange(len(_samplers))
    _width = 0.34
    _figure, (_gain_axis, _time_axis) = plt.subplots(1, 2, figsize=(9, 3.8))
    _gain_axis.bar(
        _x - _width / 2,
        efficiency_df["efficiency_ratio_vs_slice"],
        _width,
        color=_colors,
        label="recorded canonical ratio",
    )
    _gain_axis.bar(
        _x + _width / 2,
        efficiency_df["scale_normalized_efficiency_ratio"],
        _width,
        color=_colors,
        alpha=0.5,
        hatch="//",
        label="recorded scaling ratio",
    )
    _gain_axis.axhline(
        _limits["canonical_minimum_median_bulk_ess_per_sampling_second_ratio_vs_slice"],
        color="#B22222",
        linestyle="--",
        label="frozen ESS/s limit",
    )
    _gain_axis.axhline(
        _limits["scale_minimum_normalized_efficiency_ratio_vs_reference"],
        color="#7030A0",
        linestyle=":",
        label="frozen scaling limit",
    )
    _gain_axis.set_xticks(_x, _labels)
    _gain_axis.set_ylabel("recorded ratio")
    _gain_axis.set_title("Historical recorded-machine efficiency")
    _gain_axis.legend(fontsize=7)

    _time_axis.bar(
        _labels,
        efficiency_df["maximum_total_time_ratio_vs_slice"],
        color=_colors,
    )
    _time_axis.axhline(
        _limits["canonical_maximum_total_seconds_ratio_vs_slice_per_scenario"],
        color="#B22222",
        linestyle="--",
        label="frozen maximum",
    )
    _time_axis.axhline(1.0, color="#111111", linestyle=":", label="equal to Slice")
    _time_axis.set_ylabel("recorded worst total-time ratio")
    _time_axis.set_title("Historical recorded-machine wall time")
    _time_axis.legend(fontsize=7)
    for _axis in (_gain_axis, _time_axis):
        _axis.grid(axis="y", alpha=0.2)
        _axis.tick_params(axis="x", labelsize=8)
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell
def _(SAMPLER_ORDER, fit_df, mo, np, plt, scenario_selector):
    _selected = (
        fit_df.loc[fit_df["scenario"] == scenario_selector.value]
        .set_index("sampler")
        .loc[list(SAMPLER_ORDER)]
    )
    _labels = ["Slice", "PyMC NUTS", "NumPyro NUTS"]
    _segments = [
        ("model_build_seconds", "model build", "#A5A5A5"),
        ("objective_compile_seconds", "objective compile + first eval", "#70AD47"),
        ("gradient_compile_seconds", "gradient compile + first eval", "#FFC000"),
        ("sampling_seconds", "sampling call", "#4472C4"),
        ("predictive_seconds", "posterior prediction", "#ED7D31"),
    ]
    _bottom = np.zeros(len(_selected))
    _figure, _axis = plt.subplots(figsize=(8.5, 4.2))
    for _field, _label, _color in _segments:
        _values = _selected[_field].to_numpy()
        _axis.bar(_labels, _values, bottom=_bottom, label=_label, color=_color)
        _bottom += _values
    _axis.set_ylabel("recorded seconds on one historical machine")
    _axis.set_title(
        f"Authenticated compact timing summary: {scenario_selector.value.replace('_', ' ')}"
    )
    _axis.grid(axis="y", alpha=0.2)
    _axis.legend(fontsize=8)
    _figure.tight_layout()
    mo.vstack(
        [
            mo.md("""
            ### Timing interpretation

            These stacked durations are values stored in the compact result. They were
            not remeasured by this report and should not be extrapolated to another OS,
            architecture, dependency stack, or JEAM revision.
            """),
            _figure,
        ]
    )
    return


@app.cell
def _(fit_df, mo, verification):
    _scale = fit_df.loc[fit_df["role"] == "scale"][
        [
            "route",
            "trials",
            "route_hdi_inclusions",
            "route_hdi_checks",
            "minimum_bulk_ess",
            "minimum_tail_ess",
            "maximum_rhat",
            "sampling_seconds",
        ]
    ].copy()
    _scale["recorded HDI coverage"] = (
        _scale["route_hdi_inclusions"].astype(str)
        + "/"
        + _scale["route_hdi_checks"].astype(str)
    )
    mo.vstack(
        [
            mo.md(f"""
            ### The scaling input is reconstructed evidence

            The 1,500-trial compact rows are shown because scaling was part of the frozen
            protocol. However, the historical scale dataset bytes were not retained.
            Only a post-hoc reconstruction is now available; its authenticated SHA256 is
            `{verification["reconstructed_scale_dataset_sha256"]}`. Treat the scaling
            table as a compact historical summary with a reconstructed input binding,
            not as a fully retained rerunnable benchmark.

            The scale scenario's frozen gate covered diagnostics and normalized
            efficiency, not recovery. Its recorded 3/4 HDI results must not be added to
            the 48 canonical route checks.
            """),
            mo.ui.table(
                _scale[
                    [
                        "route",
                        "trials",
                        "recorded HDI coverage",
                        "minimum_bulk_ess",
                        "minimum_tail_ess",
                        "maximum_rhat",
                        "sampling_seconds",
                    ]
                ].round(3),
                selection=None,
                pagination=False,
            ),
        ]
    )
    return


@app.cell
def _(evidence_boundary, mo):
    _blockers = "\n".join(
        f"- {blocker}" for blocker in evidence_boundary["ecosystem_promotion_blockers"]
    )
    mo.md(f"""
    ## Interpretation

    - The compact result records that both analytical NUTS routes met the study's
      historical smoke thresholds. That establishes bounded historical candidates and
      capabilities only.
    - Blackbox + Slice remains the durable reference and gradient-free route.
    - This report does **not** recommend NUTS, change HSSM's current sampler or
      likelihood defaults, or claim portable timing dominance.
    - This deterministic recovery smoke is not simulation-based calibration. It does
      not establish nominal long-run coverage, hierarchical-model performance, GPU
      scaling, or support beyond the fixed CDM scope.

    Ecosystem promotion is blocked for all verifier-reported reasons:

    {_blockers}
    """)
    return


@app.cell
def _(mo, specification):
    mo.md(f"""
    ## Audit without rerunning the benchmark

    From the repository root, run the standalone verifier-only audit:

    ```bash
    uv run --python 3.12 --group jeam-prototype python -m scripts.verify_jeam_sampler_comparison_evidence
    ```

    That command authenticates the frozen inputs and compact result, recomputes the
    bounded verification summary, and performs no sampling. The v1 protocol was frozen
    at `{specification["frozen_at_utc"]}`. A scientifically necessary new run requires
    a **new versioned protocol and result**; it must not overwrite or silently
    regenerate v1.
    """)
    return


if __name__ == "__main__":
    app.run()

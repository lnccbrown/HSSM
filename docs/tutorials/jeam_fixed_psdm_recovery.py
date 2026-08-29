"""Inspect authenticated compact evidence from fixed-PSDM recovery v1.

This marimo report calls one hash-before-parse reporting API. It imports neither HSSM,
JEAM, nor PyMC and does not access the network, simulate, optimize, predict, or sample.
The figures present authenticated producer-recorded compact summaries; missing raw draws
cannot be reconstructed here.

Open it from the HSSM repository root::

    uv run --group docs marimo edit docs/tutorials/jeam_fixed_psdm_recovery.py

Export a path-clean static rendering with::

    uv run --group docs marimo export html \
        docs/tutorials/jeam_fixed_psdm_recovery.py \
        -o /tmp/jeam-fixed-psdm-recovery.html --force --no-include-code
"""

# ruff: noqa: B018, D401, E501, PLR1711
import marimo

__generated_with = "0.23.16"
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

    from scripts.jeam_psdm_recovery_report import load_psdm_recovery_report

    return load_psdm_recovery_report, mo, np, pd, plt


@app.cell
def _(load_psdm_recovery_report):
    report = load_psdm_recovery_report()
    aggregate_records = report["aggregate_records"]
    evidence_boundary = report["evidence_boundary"]
    failure_records = report["failure_records"]
    objective_records = report["objective_records"]
    parameter_records = report["parameter_records"]
    predictive_records = report["predictive_records"]
    provenance = report["provenance"]
    scenario_records = report["scenario_records"]
    summary = report["summary"]
    return (
        aggregate_records,
        evidence_boundary,
        failure_records,
        objective_records,
        parameter_records,
        predictive_records,
        provenance,
        scenario_records,
        summary,
    )


@app.cell
def _(mo, summary):
    _gate = "passed" if summary["overall_pass"] else "failed"
    _promotion = "blocked" if summary["ecosystem_promotion_blocked"] else "not blocked"
    mo.md(f"""
    # Archived fixed-PSDM recovery evidence

    This report audits a seeded four-scenario, scalar/intercept-only recovery smoke from
    authenticated compact summaries. It is **not** a calibration study, a current-JEAM
    rerun, or raw-trace reanalysis.

    | Archived result | Scenarios | HDI summary | Exact failures | Support status |
    |---|---:|---:|---:|---|
    | **Overall gate {_gate}** | {summary["scenario_count"]} | **{summary["truth_in_hdi"]}/{summary["truth_total"]} truths in archived HDIs** | {summary["failure_count"]} | **Public support remains {_promotion}** |

    A successful integrity check preserves the expected negative scientific result. It
    does not turn that result into recovery evidence.
    """)
    return


@app.cell
def _(evidence_boundary, mo, pd, provenance, summary):
    _provenance = pd.DataFrame(
        [
            ("result SHA-256", provenance["result_sha256"]),
            ("result commit", provenance["result_commit"]),
            ("historical HSSM runner", provenance["hssm_revision"]),
            ("historical JEAM", provenance["historical_jeam_revision"]),
            ("current safety JEAM", provenance["current_safety_jeam_revision"]),
            ("current JEAM rerun", provenance["current_safety_revision_rerun"]),
            ("historical Python", provenance["python_version"]),
            ("evidence class", summary["evidence_class"]),
        ],
        columns=["provenance field", "authenticated value"],
    )
    _retention = evidence_boundary["retention"]
    _boundary = pd.DataFrame(
        [
            ("raw datasets retained", _retention["raw_datasets_retained_in_git"]),
            ("raw traces retained", _retention["raw_traces_retained_in_git"]),
            (
                "raw prior-predictive draws retained",
                _retention["raw_prior_predictive_draws_retained"],
            ),
            (
                "raw posterior-predictive draws retained",
                _retention["raw_posterior_predictive_draws_retained"],
            ),
            (
                "ordered Slice identity authenticated",
                evidence_boundary["ordered_slice_identity_independently_authenticated"],
            ),
            (
                "runtime HSSM checkout authenticated",
                evidence_boundary["runtime_hssm_import_bound_to_recorded_checkout"],
            ),
            (
                "independent raw re-verification",
                evidence_boundary["independent_raw_reverification"],
            ),
        ],
        columns=["evidence boundary", "status"],
    )
    mo.vstack(
        [
            mo.md("""
            ## Provenance and evidence boundary

            The verifier authenticates the frozen spec, post-hoc archive addendum, and
            compact result before parsing the same byte snapshots. Recorded hashes are
            identifiers for missing payloads, not substitutes for retained data or draws.
            """),
            mo.ui.table(_provenance, selection=None, pagination=False),
            mo.ui.table(_boundary, selection=None, pagination=False),
            mo.md(
                "**Promotion blockers:** "
                + "; ".join(evidence_boundary["promotion_blockers"])
            ),
        ]
    )
    return


@app.cell
def _(mo, parameter_records, scenario_records):
    scenario_selector = mo.ui.dropdown(
        options={
            row["scenario"].replace("_", " ").title(): row["scenario"]
            for row in scenario_records
        },
        value="High Threshold Strong Radial",
        label="Scenario",
    )
    parameter_selector = mo.ui.dropdown(
        options={row["parameter"]: row["parameter"] for row in parameter_records},
        value="v_y",
        label="Parameter",
    )
    mo.hstack([scenario_selector, parameter_selector], justify="start", gap=2)
    return parameter_selector, scenario_selector


@app.cell
def _(
    aggregate_records,
    failure_records,
    objective_records,
    parameter_records,
    pd,
    predictive_records,
    scenario_records,
):
    aggregate_df = pd.DataFrame(aggregate_records)
    failure_df = pd.DataFrame(failure_records)
    objective_df = pd.DataFrame(objective_records)
    parameter_df = pd.DataFrame(parameter_records)
    predictive_df = pd.DataFrame(predictive_records)
    scenario_df = pd.DataFrame(scenario_records)
    return (
        aggregate_df,
        failure_df,
        objective_df,
        parameter_df,
        predictive_df,
        scenario_df,
    )


@app.cell
def _(failure_df, mo):
    mo.vstack(
        [
            mo.md("""
            ## The eight recomputed compact failures

            These messages are derived from authenticated compact fields and the frozen
            thresholds. They identify failed recorded subgates; without raw traces they
            do not establish a sampler, mixing, or identifiability cause.
            """),
            mo.ui.table(
                failure_df.loc[:, ["order", "message", "category"]],
                selection=None,
                pagination=False,
            ),
        ]
    )
    return


@app.cell
def _(mo, parameter_df, scenario_selector):
    _columns = {
        "parameter": "parameter",
        "truth": "truth",
        "optimizer_endpoint": "JEAM fixed-budget endpoint",
        "optimizer_absolute_error": "endpoint |error|",
        "optimizer_recovery_passed": "endpoint gate",
        "posterior_mean": "recorded posterior mean",
        "posterior_sd": "recorded posterior SD",
        "hdi_lower": "94% HDI lower",
        "hdi_upper": "94% HDI upper",
        "truth_in_hdi": "covers truth",
        "rhat": "R-hat",
        "ess_bulk": "bulk ESS",
        "ess_tail": "tail ESS",
        "mcse_over_posterior_sd": "MCSE/SD",
        "diagnostics_passed": "diagnostic subgate",
    }
    _selected = (
        parameter_df.loc[parameter_df["scenario"] == scenario_selector.value]
        .loc[:, list(_columns)]
        .rename(columns=_columns)
        .round(4)
    )
    mo.vstack(
        [
            mo.md("""
            ## Inspect one scenario

            The optimizer column is a deterministic differential-evolution endpoint from
            a fixed budget of 20 iterations and 1,260 evaluations with `polish=false`.
            It is not a demonstrated converged optimum or maximum-likelihood estimate.
            """),
            mo.ui.table(_selected, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Recorded recovery summaries across four datasets

    Choose a parameter above. Crosses mark generating truth, open circles the JEAM
    fixed-budget endpoint, squares the recorded posterior mean, and horizontal lines the
    recorded 94% HDI. These are compact summaries, not posterior draws.
    """)
    return


@app.cell
def _(parameter_df, parameter_selector, plt):
    _selected = parameter_df.loc[
        parameter_df["parameter"] == parameter_selector.value
    ].reset_index(drop=True)
    _positions = range(len(_selected))
    _figure, _axis = plt.subplots(figsize=(9.5, 4.2))
    for _position, _row in _selected.iterrows():
        _axis.hlines(
            _position,
            _row["hdi_lower"],
            _row["hdi_upper"],
            color="#4472C4",
            linewidth=4,
            alpha=0.42,
            label="recorded 94% HDI" if _position == 0 else None,
        )
    _axis.scatter(
        _selected["truth"],
        list(_positions),
        marker="x",
        s=80,
        linewidth=2.2,
        color="#111111",
        label="generating truth",
        zorder=4,
    )
    _axis.scatter(
        _selected["optimizer_endpoint"],
        list(_positions),
        marker="o",
        s=52,
        facecolors="white",
        edgecolors="#E67E22",
        linewidth=1.8,
        label="JEAM fixed-budget endpoint",
        zorder=3,
    )
    _axis.scatter(
        _selected["posterior_mean"],
        list(_positions),
        marker="s",
        s=48,
        color="#4472C4",
        label="recorded posterior mean",
        zorder=3,
    )
    _axis.set_yticks(
        list(_positions),
        [name.replace("_", " ") for name in _selected["scenario"]],
    )
    _axis.invert_yaxis()
    _axis.set_xlabel(parameter_selector.value)
    _axis.set_title(f"Archived summaries for {parameter_selector.value}")
    _axis.grid(axis="x", alpha=0.2)
    _axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell
def _(mo, objective_df, scenario_selector):
    _selected = (
        objective_df.loc[objective_df["scenario"] == scenario_selector.value]
        .loc[
            :,
            [
                "candidate",
                "direct_jeam",
                "compiled_hssm",
                "absolute_error",
                "passed",
            ],
        ]
        .rename(
            columns={
                "direct_jeam": "direct historical JEAM NLL",
                "compiled_hssm": "compiled historical HSSM NLL",
                "absolute_error": "absolute difference",
                "passed": "compact parity gate",
            }
        )
    )
    mo.vstack(
        [
            mo.md("""
            ## Same-producer adapter fidelity

            The archive compares direct and compiled objectives from the same historical
            JEAM producer. Agreement supports adapter fidelity at the recorded candidates;
            it is not independent likelihood validation or evidence about current JEAM.
            """),
            mo.ui.table(_selected, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(objective_df, plt, scenario_df):
    _errors = objective_df.groupby("scenario")["absolute_error"].max()
    _limits = scenario_df.set_index("scenario")["objective_absolute_error_limit"]
    _labels = [name.replace("_", " ") for name in _errors.index]
    _figure, _axis = plt.subplots(figsize=(8.7, 3.8))
    _axis.barh(_labels, _errors.values, color="#70AD47")
    _axis.axvline(
        _limits.iloc[0], color="#B22222", linestyle="--", label="frozen limit"
    )
    _axis.set_xlabel("maximum recorded absolute NLL difference")
    _axis.set_title("Historical same-producer objective comparisons")
    _axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    _axis.grid(axis="x", alpha=0.2)
    _axis.legend()
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell
def _(mo):
    mo.md("""
    ## Recorded posterior diagnostic margins

    Each bar is normalized so **one or above fails** the frozen exclusive threshold.
    These values are authenticated compact summaries. Missing traces prevent independent
    recomputation and prevent attributing failures to an authenticated sampler identity.
    """)
    return


@app.cell
def _(np, parameter_df, plt, scenario_selector):
    _selected = parameter_df.loc[
        parameter_df["scenario"] == scenario_selector.value
    ].reset_index(drop=True)
    _x = np.arange(len(_selected))
    _width = 0.19
    _figure, _axis = plt.subplots(figsize=(9, 4.1))
    _axis.bar(
        _x - 1.5 * _width,
        (_selected["rhat"] - 1.0) / (_selected["rhat_limit"] - 1.0),
        _width,
        label="R-hat margin",
        color="#9DC3E6",
    )
    _axis.bar(
        _x - 0.5 * _width,
        _selected["bulk_ess_limit"] / _selected["ess_bulk"],
        _width,
        label="bulk ESS margin",
        color="#70AD47",
    )
    _axis.bar(
        _x + 0.5 * _width,
        _selected["tail_ess_limit"] / _selected["ess_tail"],
        _width,
        label="tail ESS margin",
        color="#8064A2",
    )
    _axis.bar(
        _x + 1.5 * _width,
        _selected["mcse_over_posterior_sd"] / _selected["mcse_over_posterior_sd_limit"],
        _width,
        label="MCSE/SD margin",
        color="#F4B183",
    )
    _axis.axhline(1.0, color="#B22222", linestyle="--", label="failure boundary")
    _axis.set_xticks(_x, _selected["parameter"])
    _axis.set_ylabel("recorded diagnostic / frozen boundary")
    _axis.set_title(scenario_selector.value.replace("_", " ").title())
    _axis.grid(axis="y", alpha=0.2)
    _axis.legend(ncol=2, fontsize=8)
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell
def _(mo):
    mo.md("""
    ## Producer-recorded predictive summaries

    The compact RT-quantile and polar-unit-vector errors meet their frozen thresholds.
    Raw posterior-predictive draws were not retained, so the distributions cannot be
    independently recomputed or inspected here.
    """)
    return


@app.cell
def _(np, plt, predictive_df, scenario_selector):
    _row = predictive_df.loc[predictive_df["scenario"] == scenario_selector.value].iloc[
        0
    ]
    _probabilities = np.asarray(_row["rt_probabilities"])
    _figure, (_rt_axis, _angle_axis) = plt.subplots(1, 2, figsize=(9, 3.8))
    _rt_axis.plot(
        _probabilities,
        _row["observed_rt_quantiles"],
        marker="o",
        color="#111111",
        label="observed compact summary",
    )
    _rt_axis.plot(
        _probabilities,
        _row["predictive_rt_quantiles"],
        marker="s",
        color="#4472C4",
        label="predictive compact summary",
    )
    _rt_axis.set_xlabel("RT quantile probability")
    _rt_axis.set_ylabel("seconds")
    _rt_axis.set_title("Recorded RT quantiles")
    _rt_axis.grid(alpha=0.2)
    _rt_axis.legend(fontsize=8)

    _x = np.arange(2)
    _width = 0.36
    _angle_axis.bar(
        _x - _width / 2,
        _row["observed_mean_polar_unit_vector"],
        _width,
        label="observed compact summary",
        color="#A5A5A5",
    )
    _angle_axis.bar(
        _x + _width / 2,
        _row["predictive_mean_polar_unit_vector"],
        _width,
        label="predictive compact summary",
        color="#4472C4",
    )
    _angle_axis.set_xticks(_x, ["mean sin(θ)", "mean cos(θ)"])
    _angle_axis.set_title("Recorded polar unit-vector means")
    _angle_axis.grid(axis="y", alpha=0.2)
    _angle_axis.legend(fontsize=8)
    _figure.suptitle(scenario_selector.value.replace("_", " ").title())
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell
def _(aggregate_df, evidence_boundary, mo, parameter_df, summary):
    _high = parameter_df.loc[
        parameter_df["scenario"] == "high_threshold_strong_radial"
    ].set_index("parameter")
    _low_vy = parameter_df.loc[
        (parameter_df["scenario"] == "low_threshold_balanced_drift")
        & (parameter_df["parameter"] == "v_y")
    ].iloc[0]
    _aggregate = aggregate_df.set_index("parameter")
    mo.md(f"""
    ## Interpretation and next experiments

    1. **Historical adapter fidelity passed its compact checks.** The largest recorded
       same-producer objective difference is
       `{summary["maximum_objective_absolute_error"]:.2e}`. This does not independently
       validate the likelihood or current JEAM.
    2. **Six recorded diagnostic thresholds failed for `a` and `t` in one scenario.**
       The high-threshold rows contain R-hat `{_high.loc["a", "rhat"]:.4f}` for `a`
       and `{_high.loc["t", "rhat"]:.4f}` for `t`. Missing traces and unauthenticated
       sampler identity mean v1 does not establish their cause.
    3. **The low-threshold `v_y` fixed-budget endpoint missed its limit.** Truth was
       `{_low_vy["truth"]:.2f}` and the endpoint was
       `{_low_vy["optimizer_endpoint"]:.3f}`. The recorded posterior HDI covered truth;
       that pattern motivates, but does not prove, an information hypothesis.
    4. **The result remains negative and bounded.** The archived posterior rows cover
       `{summary["truth_in_hdi"]}/{summary["truth_total"]}` truths, while
       `{_aggregate["hdi_coverage"].mean():.3f}` is only a four-dataset smoke summary,
       not calibration.

    A separately preregistered **v2a** may retain datasets and traces to test the
    budget/mixing hypothesis. A distinct **v2b** may retain new trial-design evidence to
    test the information hypothesis. Neither hypothesis is a v1 conclusion.

    The scalar `t` prior was untruncated `HalfNormal(2)`. Configured bounds `[0, 2]` were
    separate metadata, while likelihood support required `t < min(rt)`. The frozen
    optimizer endpoint used `nextafter(min(rt), -inf)`, which lies inside HSSM's `1e-15`
    numerical floor; a successor must preregister a real support margin.

    **Public support remains blocked.** {evidence_boundary["successor_interpretation"]}
    """)
    return


@app.cell
def _(mo, provenance):
    mo.md(f"""
    ## Audit without rerunning v1

    Canonical v1 execution and writes to the archived result path are disabled. Smoke
    mode is only a noncanonical wiring check and cannot create v1 evidence. Audit the
    immutable compact archive with:

    ```bash
    python -m scripts.verify_jeam_psdm_recovery_evidence
    ```

    A passing verifier must report an authentic compact archive, an expected failed
    scientific gate, and blocked promotion. The pinned result is
    `{provenance["result_sha256"]}`.
    """)
    return


if __name__ == "__main__":
    app.run()

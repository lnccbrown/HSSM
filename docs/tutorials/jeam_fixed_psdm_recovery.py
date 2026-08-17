"""Inspect the preregistered fixed-PSDM recovery evidence without rerunning MCMC.

This marimo notebook reads the committed protocol and compact v1 result. It does not
import JEAM, construct an HSSM model, or load the raw posterior traces.

Open it from the HSSM repository root::

    uv run --group docs marimo edit docs/tutorials/jeam_fixed_psdm_recovery.py
"""

# ruff: noqa: B018, D401, E501, PLR1711
import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return Path, json, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Fixed projected-spherical diffusion recovery in HSSM

    This report asks whether the fixed projected spherical diffusion model (PSDM) keeps
    JEAM's likelihood intact when used through ordinary `hssm.HSSM`, and whether one
    frozen gradient-free inference protocol recovers its parameters across four datasets.

    The answer is deliberately mixed: **the numerical handshake passes, but the complete
    recovery gate fails**. The failure is useful evidence. It prevents us from presenting
    this exact ordered-Slice configuration as generally reliable while showing precisely
    which parts of the integration already work.

    Everything below is reconstructed from committed JSON. Opening this notebook never
    reruns optimization, simulation, posterior sampling, or prediction.
    """)
    return


@app.cell
def _(Path, json):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_path = (
        repo_root / "benchmarks" / "results" / "jeam_fixed_psdm_recovery_v1.json"
    )
    spec_path = repo_root / "benchmarks" / "specs" / "jeam_fixed_psdm_recovery_v1.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return artifact, artifact_path, repo_root, spec, spec_path


@app.cell
def _(artifact, mo):
    _parameters = [
        parameter
        for scenario in artifact["scenarios"]
        for parameter in scenario["parameters"]
    ]
    _covered = sum(parameter["truth_in_hdi"] for parameter in _parameters)
    _maximum_rhat = max(parameter["rhat"] for parameter in _parameters)
    _minimum_bulk_ess = min(parameter["ess_bulk"] for parameter in _parameters)
    _maximum_objective_error = max(
        scenario["maximum_objective_absolute_error"]
        for scenario in artifact["scenarios"]
    )
    mo.md(f"""
    ## Result at a glance

    | Frozen v1 gate | Scenarios | 94% HDI coverage | Maximum R-hat | Minimum bulk ESS | Maximum objective error |
    |---|---:|---:|---:|---:|---:|
    | **{"PASS" if artifact["gate"]["passed"] else "FAIL"}** | {len(artifact["scenarios"])} | {_covered}/{len(_parameters)} | {_maximum_rhat:.4f} | {_minimum_bulk_ess:.0f} | {_maximum_objective_error:.2e} |

    The canonical run took `{artifact["total_runtime_seconds"]:.1f}` seconds with JEAM
    `{artifact["provenance"]["jeam_revision"][:12]}`. Its thresholds were committed
    before the run and were not relaxed afterward.
    """)
    return


@app.cell
def _(artifact, mo):
    scenario_selector = mo.ui.dropdown(
        options={
            scenario["scenario"].replace("_", " ").title(): scenario["scenario"]
            for scenario in artifact["scenarios"]
        },
        value="High Threshold Strong Radial",
        label="Scenario",
    )
    parameter_selector = mo.ui.dropdown(
        options={name: name for name in artifact["parameter_order"]},
        value="v_y",
        label="Parameter",
    )
    mo.hstack([scenario_selector, parameter_selector], justify="start", gap=2)
    return parameter_selector, scenario_selector


@app.cell
def _(artifact, pd, spec):
    _optimizer_limits = spec["scientific_acceptance"]["optimizer_recovery"][
        "maximum_absolute_error"
    ]
    _posterior_limits = spec["scientific_acceptance"]["posterior_recovery"]
    _parameter_rows = []
    _scenario_rows = []
    for _scenario in artifact["scenarios"]:
        _scenario_rows.append(
            {
                "scenario": _scenario["scenario"],
                "objective_error": _scenario["maximum_objective_absolute_error"],
                "optimizer_parameter_error": _scenario["optimizer"][
                    "maximum_parameter_absolute_error"
                ],
                "sampling_seconds": _scenario["runtime_seconds"]["sampling"],
                "total_seconds": _scenario["runtime_seconds"]["total"],
            }
        )
        for _parameter in _scenario["parameters"]:
            _name = _parameter["name"]
            _parameter_rows.append(
                {
                    "scenario": _scenario["scenario"],
                    **_parameter,
                    "optimizer_abs_error": abs(
                        _parameter["optimizer_estimate"] - _parameter["truth"]
                    ),
                    "posterior_abs_error": abs(
                        _parameter["posterior_mean"] - _parameter["truth"]
                    ),
                    "optimizer_gate_pass": abs(
                        _parameter["optimizer_estimate"] - _parameter["truth"]
                    )
                    <= _optimizer_limits[_name],
                    "diagnostics_pass": (
                        _parameter["rhat"] < _posterior_limits["maximum_rhat_exclusive"]
                        and _parameter["ess_bulk"]
                        > _posterior_limits["minimum_bulk_ess_exclusive"]
                        and _parameter["ess_tail"]
                        > _posterior_limits["minimum_tail_ess_exclusive"]
                        and _parameter["mcse_over_posterior_sd"]
                        < _posterior_limits["maximum_mcse_over_posterior_sd_exclusive"]
                    ),
                }
            )
    parameter_df = pd.DataFrame(_parameter_rows)
    scenario_df = pd.DataFrame(_scenario_rows)
    aggregate_df = pd.DataFrame(artifact["aggregate"])
    return aggregate_df, parameter_df, scenario_df


@app.cell
def _(artifact, mo, spec):
    _scope = spec["scope"]
    _execution = spec["execution"]
    _settings = _scope["fixed_model_settings"]
    mo.md(f"""
    ## What was tested

    - **HSSM class:** ordinary `hssm.HSSM`, not a PSDM subclass
    - **Model:** `{_scope["model"]}` with black-box JEAM likelihood
    - **Observation:** `[rt, response]` with density measure `{_scope["density_measure"]}`
    - **RT support:** finite and strictly positive
    - **Response support:** polar angle on the closed interval `[0, π]`
    - **Parameters:** `[v_x, v_y, a, t]`; `v_y` is a nonnegative radial drift
    - **Fixed settings:** `sigma={_settings["sigma"]}`, `s_v={_settings["s_v"]}`,
      `s_t={_settings["s_t"]}`, fixed threshold, and `p_outlier=None`
    - **Inference:** {_execution["chains"]} chains, {_execution["tune"]} tuning +
      {_execution["draws"]} retained draws, with one PyMC Slice step per parameter in
      `[v_x, v_y, a, t]`

    The HSSM priors were not tuned to the generating values. The runner constructed the
    public model defaults and asserted their exact distributions, bounds, and initial
    values before fitting. Every raw dataset and NetCDF trace was saved before summaries;
    the compact artifact records their hashes.
    """)
    return


@app.cell
def _(artifact, mo, pd):
    _failures = pd.DataFrame({"predeclared failure": artifact["gate"]["failures"]})
    mo.vstack(
        [
            mo.md("""
            ## The exact failed gates

            Eight messages arise from four substantive issues: convergence and efficiency
            fail for `a` and `t` in the high-threshold scenario, while `v_y` optimizer
            recovery fails in the low-threshold scenario and therefore also fails its
            aggregate RMSE gate.
            """),
            mo.ui.table(_failures, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Inspect one scenario

    Select a scenario above. The table separates three targets: generating truth, the
    direct-JEAM maximum-likelihood estimate, and the HSSM posterior. An MLE is not expected
    to equal a posterior mean. The numerical handshake is instead tested by running the
    direct JEAM and compiled HSSM optimizers with identical bounds, seeds, and budgets.
    """)
    return


@app.cell
def _(mo, parameter_df, scenario_selector):
    _columns = {
        "name": "parameter",
        "truth": "truth",
        "optimizer_estimate": "JEAM MLE",
        "optimizer_abs_error": "MLE |error|",
        "optimizer_gate_pass": "MLE gate",
        "posterior_mean": "HSSM mean",
        "posterior_abs_error": "posterior |error|",
        "hdi_lower": "94% HDI lower",
        "hdi_upper": "94% HDI upper",
        "truth_in_hdi": "covers truth",
        "rhat": "R-hat",
        "ess_bulk": "bulk ESS",
        "mcse_over_posterior_sd": "MCSE/SD",
        "diagnostics_pass": "diagnostics gate",
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
    ## Recovery across all four datasets

    Choose a parameter above. Crosses mark generating truth, open circles the direct-JEAM
    MLE, squares the HSSM posterior mean, and horizontal lines the 94% HSSM HDI. This view
    keeps a single problematic dataset visible instead of hiding it in an average.
    """)
    return


@app.cell
def _(parameter_df, parameter_selector, plt):
    _selected = parameter_df.loc[
        parameter_df["name"] == parameter_selector.value
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
            label="HSSM 94% HDI" if _position == 0 else None,
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
        _selected["optimizer_estimate"],
        list(_positions),
        marker="o",
        s=52,
        facecolors="white",
        edgecolors="#E67E22",
        linewidth=1.8,
        label="direct JEAM MLE",
        zorder=3,
    )
    _axis.scatter(
        _selected["posterior_mean"],
        list(_positions),
        marker="s",
        s=48,
        color="#4472C4",
        label="HSSM posterior mean",
        zorder=3,
    )
    _axis.set_yticks(
        list(_positions),
        [name.replace("_", " ") for name in _selected["scenario"]],
    )
    _axis.invert_yaxis()
    _axis.set_xlabel(parameter_selector.value)
    _axis.set_title(f"Recovery of {parameter_selector.value}")
    _axis.grid(axis="x", alpha=0.2)
    _axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md("""
    ## Did HSSM change JEAM's likelihood?

    No detectable mismatch appears. The plot shows the largest direct-JEAM versus
    compiled-HSSM objective difference among three points per dataset. The red line is the
    preregistered tolerance. Both optimizers also returned exactly the same parameter
    vector in every scenario.
    """)
    return


@app.cell
def _(artifact, mo, pd, scenario_selector):
    _scenario = next(
        row
        for row in artifact["scenarios"]
        if row["scenario"] == scenario_selector.value
    )
    _rows = []
    for _index, _candidate in enumerate(_scenario["objective_candidates"], start=1):
        _rows.append(
            {
                "candidate": _index,
                "direct JEAM NLL": _candidate["direct_jeam"],
                "compiled HSSM NLL": _candidate["compiled_hssm"],
                "absolute error": _candidate["absolute_error"],
            }
        )
    _objective_table = pd.DataFrame(_rows).round(12)
    mo.ui.table(_objective_table, selection=None, pagination=False)
    return


@app.cell
def _(plt, scenario_df, spec):
    _threshold = spec["scientific_acceptance"]["maximum_objective_absolute_error"]
    _figure, _axis = plt.subplots(figsize=(8.5, 3.7))
    _axis.barh(
        [name.replace("_", " ") for name in scenario_df["scenario"]],
        scenario_df["objective_error"],
        color="#70AD47",
    )
    _axis.axvline(_threshold, color="#B22222", linestyle="--", label="frozen limit")
    _axis.set_xscale("log")
    _axis.set_xlabel("maximum absolute NLL difference (log scale)")
    _axis.set_title("Direct JEAM and compiled HSSM agree to numerical precision")
    _axis.legend()
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md("""
    ## Where ordered Slice falls short

    Each bar divides a diagnostic by its frozen failure boundary; **bars above one fail**.
    For R-hat, the plotted quantity is `(R-hat - 1) / (1.01 - 1)`. For bulk ESS it is
    `400 / ESS`, so all three groups share the same direction. Select the high-threshold
    scenario to see the `a` and `t` failures directly.
    """)
    return


@app.cell
def _(np, parameter_df, plt, scenario_selector, spec):
    _selected = parameter_df.loc[
        parameter_df["scenario"] == scenario_selector.value
    ].reset_index(drop=True)
    _limits = spec["scientific_acceptance"]["posterior_recovery"]
    _rhat_limit = _limits["maximum_rhat_exclusive"]
    _ess_limit = _limits["minimum_bulk_ess_exclusive"]
    _mcse_limit = _limits["maximum_mcse_over_posterior_sd_exclusive"]
    _x = np.arange(len(_selected))
    _width = 0.24
    _figure, _axis = plt.subplots(figsize=(8.5, 4.0))
    _axis.bar(
        _x - _width,
        (_selected["rhat"] - 1.0) / (_rhat_limit - 1.0),
        _width,
        label="R-hat margin",
        color="#9DC3E6",
    )
    _axis.bar(
        _x,
        _ess_limit / _selected["ess_bulk"],
        _width,
        label="bulk ESS margin",
        color="#70AD47",
    )
    _axis.bar(
        _x + _width,
        _selected["mcse_over_posterior_sd"] / _mcse_limit,
        _width,
        label="MCSE/SD margin",
        color="#F4B183",
    )
    _axis.axhline(1.0, color="#B22222", linestyle="--", label="failure boundary")
    _axis.set_xticks(_x, _selected["name"])
    _axis.set_ylabel("observed diagnostic / frozen boundary")
    _axis.set_title(scenario_selector.value.replace("_", " ").title())
    _axis.grid(axis="y", alpha=0.2)
    _axis.legend(ncol=2, fontsize=8)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md("""
    ## Posterior prediction still passes

    The inference limitations do not produce a visible failure in the frozen predictive
    summaries. The left panel compares observed and posterior-predictive RT quantiles. The
    right compares mean polar unit-vector components, `[mean(sin(response)),
    mean(cos(response))]`. Raw angular means are inappropriate for directional data.
    """)
    return


@app.cell
def _(artifact, np, plt, scenario_selector):
    _scenario = next(
        row
        for row in artifact["scenarios"]
        if row["scenario"] == scenario_selector.value
    )
    _predictive = _scenario["posterior_predictive"]
    _probabilities = np.asarray(_predictive["rt_probabilities"])
    _figure, (_rt_axis, _angle_axis) = plt.subplots(1, 2, figsize=(9, 3.8))
    _rt_axis.plot(
        _probabilities,
        _predictive["observed_rt_quantiles"],
        marker="o",
        color="#111111",
        label="observed",
    )
    _rt_axis.plot(
        _probabilities,
        _predictive["predictive_rt_quantiles"],
        marker="s",
        color="#4472C4",
        label="posterior predictive",
    )
    _rt_axis.set_xlabel("RT quantile probability")
    _rt_axis.set_ylabel("seconds")
    _rt_axis.set_title("Response-time quantiles")
    _rt_axis.grid(alpha=0.2)
    _rt_axis.legend(fontsize=8)

    _x = np.arange(2)
    _width = 0.36
    _angle_axis.bar(
        _x - _width / 2,
        _predictive["observed_mean_polar_unit_vector"],
        _width,
        label="observed",
        color="#A5A5A5",
    )
    _angle_axis.bar(
        _x + _width / 2,
        _predictive["predictive_mean_polar_unit_vector"],
        _width,
        label="posterior predictive",
        color="#4472C4",
    )
    _angle_axis.set_xticks(_x, ["mean sin(θ)", "mean cos(θ)"])
    _angle_axis.set_title("Polar unit-vector mean")
    _angle_axis.grid(axis="y", alpha=0.2)
    _angle_axis.legend(fontsize=8)
    _figure.suptitle(scenario_selector.value.replace("_", " ").title())
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(aggregate_df, artifact, mo, parameter_df):
    _high = parameter_df.loc[
        parameter_df["scenario"] == "high_threshold_strong_radial"
    ].set_index("name")
    _low_vy = parameter_df.loc[
        (parameter_df["scenario"] == "low_threshold_balanced_drift")
        & (parameter_df["name"] == "v_y")
    ].iloc[0]
    _aggregate = aggregate_df.set_index("name")
    mo.md(f"""
    ## Scientific interpretation

    1. **The HSSM handshake is numerically intact.** Direct JEAM and compiled HSSM agree
       to `{max(s["maximum_objective_absolute_error"] for s in artifact["scenarios"]):.2e}`
       in NLL, and their optimizer vectors are identical in every scenario.
    2. **This ordered-Slice budget is not a general recommendation.** In the
       high-threshold case, `a` has R-hat `{_high.loc["a", "rhat"]:.4f}` and bulk ESS
       `{_high.loc["a", "ess_bulk"]:.0f}`; `t` has R-hat
       `{_high.loc["t", "rhat"]:.4f}` and bulk ESS
       `{_high.loc["t", "ess_bulk"]:.0f}`. The compact artifact establishes insufficient
       mixing at this budget; it does not by itself identify the full posterior geometry.
    3. **Low-threshold radial drift is weakly recovered in this dataset.** For `v_y`, truth
       is `{_low_vy["truth"]:.2f}`, the MLE is `{_low_vy["optimizer_estimate"]:.3f}`, and
       the posterior is `{_low_vy["posterior_mean"]:.3f} ± {_low_vy["posterior_sd"]:.3f}`
       with 94% HDI `[{_low_vy["hdi_lower"]:.3f}, {_low_vy["hdi_upper"]:.3f}]`. The broad
       interval covers truth, so the optimizer miss should not be misread as an HSSM/JEAM
       mismatch.
    4. **Aggregate posterior recovery is encouraging but limited.** Posterior RMSE passes
       for all four parameters and {_aggregate["hdi_coverage"].mul(4).sum():.0f}/16
       intervals cover truth. Four datasets are a regression benchmark, not a calibration
       study of nominal 94% coverage.

    The appropriate next experiment is a separately preregistered v2: first increase the
    sampling budget on the same saved datasets to isolate mixing, then vary trial count or
    design for the low-threshold `v_y` case to isolate information. V1 remains immutable.
    """)
    return


@app.cell
def _(artifact_path, mo, spec_path):
    mo.md(f"""
    ## Reproduce deliberately

    This notebook reads:

    - `{spec_path.relative_to(spec_path.parents[2])}`
    - `{artifact_path.relative_to(artifact_path.parents[2])}`

    A smoke run is the safe first check:

    ```bash
    python -m scripts.benchmark_jeam_psdm_recovery \
        --smoke --scenario baseline_asymmetric \
        --work-dir /tmp/jeam-psdm-smoke \
        --output /tmp/jeam-psdm-smoke.json
    ```

    The canonical command requires the optional JEAM dependency and several minutes. V1
    exits nonzero because its scientific gate fails, but still writes the complete result.
    Do not overwrite the committed v1 artifact; use a new versioned protocol for follow-up
    work.
    """)
    return


if __name__ == "__main__":
    app.run()

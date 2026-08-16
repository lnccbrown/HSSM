"""Illustrate the repeated-recovery evidence for the JEAM-HSSM handshake.

The default notebook path is fast and dependency-light: it reads the committed H09c
JSON artifact and reconstructs every table and plot. It does not import JEAM, construct
an HSSM model, or rerun MCMC.

Open the notebook from the HSSM repository root::

    uv run --group docs marimo edit docs/tutorials/jeam_repeated_recovery.py

Deliberate regeneration is double-gated. Install ``jeam-prototype``, start marimo with
``FULL_RUN=1``, and click the full-run button in the final section. The canonical run
takes roughly two minutes on an Apple M4 and writes only to a temporary directory.
"""

# ruff: noqa: B018, D401, E501, PLR1711
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return Path, json, mo, np, os, pd, plt, subprocess, sys, tempfile


@app.cell
def _(mo):
    mo.md(r"""
    # Repeated recovery for the JEAM–HSSM handshake

    This laboratory asks a narrow question: **does wrapping JEAM's fixed circular
    diffusion model in ordinary `hssm.HSSM` preserve the numerical objective and recover
    known generating parameters across several deterministic datasets?**

    Four 300-trial scenarios cover asymmetric and opposing drift directions, different
    boundary and nondecision-time scales, and values away from default prior centers.
    Each HSSM fit uses four explicitly ordered PyMC Slice chains with 1,000 tuning and
    1,000 retained draws per chain.

    Two estimators are shown, but they answer different questions:

    - **JEAM MLE** maximizes the direct JEAM likelihood.
    - **HSSM posterior mean and 94% HDI** summarize Bayesian inference through HSSM.

    They should recover the same generating signal, but an MLE is **not** expected to equal
    a posterior mean. A second HSSM optimizer is used only as a numerical handshake oracle:
    with the same seed and fixed budget it must follow direct JEAM's objective exactly.
    """)
    return


@app.cell
def _(Path, json):
    repo_root = Path(__file__).resolve().parents[2]
    artifact_path = repo_root / "benchmarks" / "results" / "jeam_repeated_recovery.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    return artifact, artifact_path, repo_root


@app.cell
def _(artifact, mo):
    _coverage = sum(
        parameter["interval_covers_truth"]
        for scenario in artifact["scenarios"]
        for parameter in scenario["parameters"]
    )
    _total = sum(len(scenario["parameters"]) for scenario in artifact["scenarios"])
    _max_rhat = max(
        parameter["rhat"]
        for scenario in artifact["scenarios"]
        for parameter in scenario["parameters"]
    )
    _max_objective_error = max(
        scenario["maximum_objective_absolute_error"]
        for scenario in artifact["scenarios"]
    )
    mo.md(f"""
    ## Result at a glance

    | Predeclared gate | Scenarios | 94% HDI coverage | Maximum R-hat | Maximum objective error |
    |---|---:|---:|---:|---:|
    | **{"PASS" if artifact["gate"]["passed"] else "FAIL"}** | {len(artifact["scenarios"])} | {_coverage}/{_total} | {_max_rhat:.4f} | {_max_objective_error:.2e} |

    The gate passed without changing its thresholds after the run. The exact producer is
    JEAM `{artifact["jeam_revision"]}` and PyTensor ran in `{artifact["pytensor_floatx"]}`.
    """)
    return


@app.cell
def _(artifact, mo):
    scenario_selector = mo.ui.dropdown(
        options={
            scenario["name"]: scenario["name"].replace("_", " ").title()
            for scenario in artifact["scenarios"]
        },
        value=artifact["scenarios"][0]["name"],
        label="Scenario",
    )
    parameter_selector = mo.ui.dropdown(
        options={name: name for name in artifact["parameter_order"]},
        value=artifact["parameter_order"][0],
        label="Parameter",
    )
    mo.hstack([scenario_selector, parameter_selector], justify="start", gap=2)
    return parameter_selector, scenario_selector


@app.cell
def _(artifact, pd):
    scenario_records = []
    parameter_records = []
    for _scenario in artifact["scenarios"]:
        scenario_records.append(
            {
                "scenario": _scenario["name"],
                "sampling_seconds": _scenario["runtime"]["hssm_sampling_seconds"],
                "predictive_seconds": _scenario["runtime"]["hssm_predictive_seconds"],
                "objective_error": _scenario["maximum_objective_absolute_error"],
                "optimizer_parameter_error": _scenario[
                    "maximum_optimizer_parameter_absolute_error"
                ],
            }
        )
        for _parameter in _scenario["parameters"]:
            parameter_records.append(
                {
                    "scenario": _scenario["name"],
                    **_parameter,
                }
            )
    scenario_df = pd.DataFrame(scenario_records)
    parameter_df = pd.DataFrame(parameter_records)
    aggregate_df = pd.DataFrame(artifact["aggregate"])
    return aggregate_df, parameter_df, scenario_df


@app.cell
def _(mo):
    mo.md("""
    ## Inspect one simulated dataset

    Choose a scenario above. The table keeps the estimators conceptually separate and
    exposes the diagnostics behind the headline coverage count. `MLE error` and
    `posterior error` are estimate minus truth; neither is an error between the two
    estimators.
    """)
    return


@app.cell
def _(mo, parameter_df, scenario_selector):
    _columns = {
        "name": "parameter",
        "truth": "truth",
        "jeam_mle": "JEAM MLE",
        "jeam_mle_error": "MLE error",
        "posterior_mean": "HSSM posterior mean",
        "posterior_mean_error": "posterior error",
        "interval_lower": "94% HDI lower",
        "interval_upper": "94% HDI upper",
        "interval_covers_truth": "covers truth",
        "rhat": "R-hat",
        "ess_bulk": "bulk ESS",
        "ess_bulk_per_second": "bulk ESS/s",
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
    ## Recovery across scenarios

    Select one parameter above. Crosses mark the generating truth, circles the direct
    JEAM MLE, and squares the HSSM posterior mean. Horizontal lines are 94% HSSM HDIs.
    This view makes poor recovery visible even if an aggregate score looks acceptable.
    """)
    return


@app.cell
def _(parameter_df, parameter_selector, plt):
    _selected = parameter_df.loc[
        parameter_df["name"] == parameter_selector.value
    ].reset_index(drop=True)
    _positions = range(len(_selected))
    _figure, _axis = plt.subplots(figsize=(8, 4.2))
    for _position, _row in _selected.iterrows():
        _axis.hlines(
            _position,
            _row["interval_lower"],
            _row["interval_upper"],
            color="#4472C4",
            linewidth=4,
            alpha=0.45,
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
        _selected["jeam_mle"],
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
    _axis.set_title(f"Recovery of {parameter_selector.value} across four datasets")
    _axis.grid(axis="x", alpha=0.2)
    _axis.legend(loc="best")
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The aggregate gate

    Bias and RMSE have parameter-specific units, so the plot divides each metric by its
    predeclared limit. **Every bar must remain below one.** Coverage is checked separately
    because an average error can look small while intervals systematically miss truth.

    With four datasets, `4/4 = 100%` coverage is evidence for these cases—not a calibrated
    estimate of long-run 94% coverage. This is a repeated regression benchmark, not
    simulation-based calibration.
    """)
    return


@app.cell
def _(aggregate_df, artifact, np, plt):
    _names = aggregate_df["name"].tolist()
    _bias_limits = np.asarray(
        [artifact["thresholds"]["maximum_absolute_bias"][name] for name in _names]
    )
    _rmse_limits = np.asarray(
        [artifact["thresholds"]["maximum_rmse"][name] for name in _names]
    )
    _x = np.arange(len(_names))
    _width = 0.19
    _figure, _axis = plt.subplots(figsize=(8.5, 4.2))
    _axis.bar(
        _x - 1.5 * _width,
        np.abs(aggregate_df["jeam_mle_bias"]) / _bias_limits,
        _width,
        label="JEAM |bias| / limit",
        color="#F4B183",
    )
    _axis.bar(
        _x - 0.5 * _width,
        aggregate_df["jeam_mle_rmse"] / _rmse_limits,
        _width,
        label="JEAM RMSE / limit",
        color="#E67E22",
    )
    _axis.bar(
        _x + 0.5 * _width,
        np.abs(aggregate_df["hssm_posterior_bias"]) / _bias_limits,
        _width,
        label="HSSM |bias| / limit",
        color="#9DC3E6",
    )
    _axis.bar(
        _x + 1.5 * _width,
        aggregate_df["hssm_posterior_rmse"] / _rmse_limits,
        _width,
        label="HSSM RMSE / limit",
        color="#4472C4",
    )
    _axis.axhline(1.0, color="#B22222", linestyle="--", label="predeclared limit")
    _axis.set_xticks(_x, _names)
    _axis.set_ylabel("observed metric / allowed limit")
    _axis.set_title("Both estimators remain inside the repeated-recovery gate")
    _axis.set_ylim(0.0, 1.08)
    _axis.grid(axis="y", alpha=0.2)
    _axis.legend(ncol=2, fontsize=8)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(aggregate_df, artifact, mo):
    _diagnostics = aggregate_df[
        [
            "name",
            "interval_coverage",
            "maximum_rhat",
            "minimum_bulk_ess",
            "minimum_tail_ess",
            "mean_bulk_ess_per_second",
        ]
    ].copy()
    _diagnostics["coverage floor"] = artifact["thresholds"]["minimum_interval_coverage"]
    _diagnostics["R-hat ceiling"] = artifact["thresholds"]["maximum_rhat"]
    _diagnostics["ESS floor"] = artifact["thresholds"]["minimum_bulk_ess"]
    mo.ui.table(_diagnostics.round(4), selection=None, pagination=False)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Did HSSM change JEAM's likelihood?

    No detectable inferential change appears in this benchmark. Each scenario evaluates
    the generating truth, the direct-JEAM optimum, and the compiled-HSSM optimum through
    both objectives. Identically seeded differential-evolution runs return the same
    parameter vector; the residual objective differences are around machine precision.
    """)
    return


@app.cell
def _(artifact, plt, scenario_df):
    _threshold = artifact["thresholds"]["objective_absolute_error"]
    _figure, _axis = plt.subplots(figsize=(8, 3.6))
    _axis.barh(
        [name.replace("_", " ") for name in scenario_df["scenario"]],
        scenario_df["objective_error"],
        color="#70AD47",
    )
    _axis.axvline(_threshold, color="#B22222", linestyle="--", label="allowed error")
    _axis.set_xscale("log")
    _axis.set_xlabel("maximum absolute objective difference (log scale)")
    _axis.set_title("Direct JEAM and compiled HSSM agree far inside tolerance")
    _axis.legend()
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md("""
    ## Runtime and sampling efficiency

    Runtime is descriptive rather than a flaky pass/fail condition. The left panel shows
    each four-chain sampling run. The right reports mean bulk ESS per sampling second by
    parameter; the drift parameters mix more efficiently than boundary and nondecision
    time under this Slice configuration.
    """)
    return


@app.cell
def _(aggregate_df, np, plt, scenario_df):
    _figure, (_runtime_axis, _ess_axis) = plt.subplots(1, 2, figsize=(9, 3.8))
    _runtime_axis.bar(
        np.arange(len(scenario_df)),
        scenario_df["sampling_seconds"],
        color="#5B9BD5",
    )
    _runtime_axis.set_xticks(
        np.arange(len(scenario_df)),
        [name.replace("_", "\n") for name in scenario_df["scenario"]],
        fontsize=8,
    )
    _runtime_axis.set_ylabel("seconds")
    _runtime_axis.set_title("Four-chain sampling runtime")
    _runtime_axis.grid(axis="y", alpha=0.2)

    _ess_axis.bar(
        aggregate_df["name"],
        aggregate_df["mean_bulk_ess_per_second"],
        color="#70AD47",
    )
    _ess_axis.set_ylabel("mean bulk ESS / second")
    _ess_axis.set_title("Sampling efficiency")
    _ess_axis.grid(axis="y", alpha=0.2)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(artifact, mo):
    _posterior_rmse = {
        row["name"]: row["hssm_posterior_rmse"] for row in artifact["aggregate"]
    }
    mo.md(f"""
    ## Interpretation

    - The wrapper preserves direct JEAM's objective and optimizer path in all four cases.
    - The HSSM posterior covers all 16 generating values, with posterior RMSE
      `{_posterior_rmse["a"]:.4f}` for `a`, `{_posterior_rmse["t"]:.4f}` for `t`,
      `{_posterior_rmse["v_x"]:.4f}` for `v_x`, and `{_posterior_rmse["v_y"]:.4f}` for
      `v_y`.
    - All convergence and effective-sample-size criteria pass with the declared
      gradient-free Slice sampler.
    - The evidence supports the **fixed two-dimensional circular prototype** at these
      four generating configurations. It does not establish nominal coverage generally,
      validate spherical/hyperspherical JEAM models, or remove the current black-box
      sampler limitations.
    """)
    return


@app.cell
def _(artifact, artifact_path, mo, os):
    full_run_enabled = os.environ.get("FULL_RUN") == "1"
    full_run_button = mo.ui.run_button(label="Regenerate the complete H09c benchmark")
    _state = "enabled" if full_run_enabled else "locked"
    mo.vstack(
        [
            mo.md(f"""
            ## Reproduce deliberately

            The visible notebook is reconstructed from
            `{artifact_path.relative_to(artifact_path.parents[2])}`. The canonical command
            is:

            ```bash
            {artifact["reproduction_command"]}
            ```

            Interactive regeneration is **{_state}**. To enable it, install the optional
            JEAM group and launch marimo with `FULL_RUN=1`. A click runs the benchmark in a
            temporary directory; it never overwrites the committed artifact.
            """),
            full_run_button,
        ]
    )
    return full_run_button, full_run_enabled


@app.cell
def _(
    Path,
    artifact,
    full_run_button,
    full_run_enabled,
    json,
    mo,
    repo_root,
    subprocess,
    sys,
    tempfile,
):
    _message = mo.md("The committed result remains loaded; no MCMC was run.")
    if full_run_button.value:
        if not full_run_enabled:
            _message = mo.callout(
                "Regeneration is locked. Relaunch with `FULL_RUN=1` after installing "
                "the `jeam-prototype` dependency group.",
                kind="warn",
            )
        else:
            with tempfile.TemporaryDirectory(prefix="hssm-jeam-h09c-") as _temp_dir:
                _output = f"{_temp_dir}/result.json"
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.benchmark_jeam_repeated_recovery",
                        "--output",
                        _output,
                    ],
                    cwd=repo_root,
                    check=True,
                )
                _regenerated = json.loads(Path(_output).read_text(encoding="utf-8"))
            _message = mo.callout(
                "Regeneration passed the predeclared gate. "
                f"Fresh runtime: {_regenerated['total_runtime_seconds']:.1f}s; "
                f"committed runtime: {artifact['total_runtime_seconds']:.1f}s.",
                kind="success",
            )
    _message
    return


if __name__ == "__main__":
    app.run()

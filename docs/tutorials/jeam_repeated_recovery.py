"""Inspect the durable schema-v2 JEAM repeated-recovery evidence.

The notebook authenticates the committed evidence bundle and independently
recomputes its scientific summaries from retained datasets and raw draws. It
does not import JEAM or HSSM, access the network, optimize, or run MCMC.

Open it from the HSSM repository root::

    uv run --group docs marimo edit docs/tutorials/jeam_repeated_recovery.py
"""

# ruff: noqa: B018, D401, E501, PLR1711
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    _repository = Path(__file__).resolve().parents[2]
    if str(_repository) not in sys.path:
        sys.path.insert(0, str(_repository))

    from scripts.jeam_repeated_recovery_report import load_report, summarize_science
    from scripts.verify_jeam_repeated_recovery_evidence import MANIFEST_SHA256

    return MANIFEST_SHA256, load_report, mo, np, pd, plt, summarize_science


@app.cell
def _(mo):
    mo.md(r"""
    # Repeated recovery for the JEAM–HSSM handshake

    This report asks a narrow question: **for four deterministic datasets from JEAM's
    fixed circular diffusion model, does the HSSM black-box integration preserve the
    numerical objective and recover the generating parameters?**

    Each dataset contains 300 trials. HSSM uses four explicitly seeded PyMC Slice
    chains with 1,000 tuning and 1,000 retained draws per chain. The direct JEAM and
    compiled HSSM optimizers use the same deliberately fixed differential-evolution
    budget. They are therefore **fixed-budget estimates**, not claimed converged MLEs.

    The report is built from the committed evidence bundle for the schema-v2 result.
    Before anything below is displayed, the verifier authenticates all 14 payloads and independently
    recomputes posterior, prior-predictive, posterior-predictive, convergence, and gate
    summaries from the retained datasets and raw draws. It never imports JEAM or HSSM,
    accesses the network, optimizes, or samples.
    """)
    return


@app.cell
def _(load_report, summarize_science):
    manifest, science, report_frames = load_report()
    headline = summarize_science(science)
    return headline, manifest, report_frames, science


@app.cell
def _(headline, mo, science):
    _status = "PASS" if headline["passed"] else "FAIL"
    mo.md(f"""
    ## Result at a glance

    | Independent gate | Scenarios | 94% HDIs containing truth | Maximum R-hat | Minimum bulk ESS | Minimum tail ESS | Maximum MCSE / SD |
    |---|---:|---:|---:|---:|---:|---:|
    | **{_status}** | {headline["scenarios"]} | {headline["hdi_inclusions"]}/{headline["hdi_total"]} | {headline["maximum_rhat"]:.4f} | {headline["minimum_bulk_ess"]:.1f} | {headline["minimum_tail_ess"]:.1f} | {headline["maximum_mcse_sd_ratio"]:.4f} |

    Numerical parity is also inside its frozen gate: the largest direct-versus-compiled
    objective difference is `{headline["maximum_objective_error"]:.2e}`, and the two
    fixed-budget optimizers return parameter vectors differing by at most
    `{headline["maximum_optimizer_parameter_error"]:.1e}`. PyTensor ran in
    `{science["pytensor_floatx"]}` precision.
    """)
    return


@app.cell
def _(mo, report_frames, science):
    _first_scenario = report_frames["scenarios"].iloc[0]["scenario"]
    scenario_selector = mo.ui.dropdown(
        options={
            name.replace("_", " ").title(): name
            for name in report_frames["scenarios"]["scenario"]
        },
        value=_first_scenario.replace("_", " ").title(),
        label="Scenario",
    )
    parameter_selector = mo.ui.dropdown(
        options={name: name for name in science["parameter_order"]},
        value=science["parameter_order"][0],
        label="Parameter",
    )
    mo.hstack([scenario_selector, parameter_selector], justify="start", gap=2)
    return parameter_selector, scenario_selector


@app.cell
def _(report_frames, scenario_selector, science):
    selected_science = next(
        row for row in science["scenarios"] if row["name"] == scenario_selector.value
    )
    selected_parameters = report_frames["parameters"].loc[
        report_frames["parameters"]["scenario"] == scenario_selector.value
    ]
    selected_scenario = (
        report_frames["scenarios"]
        .loc[report_frames["scenarios"]["scenario"] == scenario_selector.value]
        .iloc[0]
    )
    return selected_parameters, selected_scenario, selected_science


@app.cell
def _(mo):
    mo.md(r"""
    ## Inspect one dataset

    The direct fixed-budget estimate and HSSM posterior answer different questions;
    equality between them is not a gate. The interval and convergence columns show the
    evidence used for the Bayesian recovery claim.
    """)
    return


@app.cell
def _(mo, selected_parameters):
    _columns = {
        "name": "parameter",
        "truth": "truth",
        "jeam_fixed_budget_estimate": "JEAM fixed-budget estimate",
        "posterior_mean": "HSSM posterior mean",
        "interval_lower": "94% HDI lower",
        "interval_upper": "94% HDI upper",
        "hdi_contains_truth": "contains truth",
        "rhat": "R-hat",
        "ess_bulk": "bulk ESS",
        "ess_tail": "tail ESS",
        "mcse_sd_ratio": "MCSE / SD",
    }
    _table = selected_parameters.loc[:, list(_columns)].rename(columns=_columns)
    mo.ui.table(_table.round(4), selection=None, pagination=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Recovery across all four datasets

    Choose a parameter above. Crosses are generating truths, open circles are direct
    JEAM fixed-budget estimates, squares are HSSM posterior means, and horizontal lines
    are 94% HSSM HDIs. This exposes every recovery case instead of hiding it behind an
    aggregate score.
    """)
    return


@app.cell
def _(parameter_selector, plt, report_frames):
    _selected = (
        report_frames["parameters"]
        .loc[report_frames["parameters"]["name"] == parameter_selector.value]
        .reset_index(drop=True)
    )
    _positions = list(range(len(_selected)))
    _figure, _axis = plt.subplots(figsize=(8.2, 4.2))
    for _position, _row in _selected.iterrows():
        _axis.hlines(
            _position,
            _row["interval_lower"],
            _row["interval_upper"],
            color="#4472C4",
            linewidth=4,
            alpha=0.4,
            label="HSSM 94% HDI" if _position == 0 else None,
        )
    _axis.scatter(
        _selected["truth"],
        _positions,
        marker="x",
        s=80,
        linewidth=2.2,
        color="#111111",
        label="generating truth",
        zorder=4,
    )
    _axis.scatter(
        _selected["jeam_fixed_budget_estimate"],
        _positions,
        marker="o",
        s=54,
        facecolors="white",
        edgecolors="#E67E22",
        linewidth=1.8,
        label="JEAM fixed-budget estimate",
        zorder=3,
    )
    _axis.scatter(
        _selected["posterior_mean"],
        _positions,
        marker="s",
        s=48,
        color="#4472C4",
        label="HSSM posterior mean",
        zorder=3,
    )
    _axis.set_yticks(
        _positions,
        [name.replace("_", " ") for name in _selected["scenario"]],
    )
    _axis.invert_yaxis()
    _axis.set_xlabel(parameter_selector.value)
    _axis.set_title(f"Recovery of {parameter_selector.value}")
    _axis.grid(axis="x", alpha=0.2)
    _axis.legend(loc="best", fontsize=8)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Frozen aggregate gate

    Bias and RMSE have parameter-specific units, so each bar is divided by its
    verifier-owned limit. Every bar must remain at or below one. HDI inclusion and
    sampler diagnostics are evaluated separately in the table below.

    Four out of four inclusions per parameter are reassuring for these deterministic
    cases, but they are **not** an estimate of long-run 94% coverage. This is a
    repeated scientific regression smoke, not simulation-based calibration.
    """)
    return


@app.cell
def _(np, plt, report_frames):
    _aggregate = report_frames["aggregate"]
    _x = np.arange(len(_aggregate))
    _width = 0.19
    _figure, _axis = plt.subplots(figsize=(8.5, 4.2))
    _series = (
        ("jeam_fixed_budget_bias_ratio", "JEAM |bias| / limit", "#F4B183"),
        ("jeam_fixed_budget_rmse_ratio", "JEAM RMSE / limit", "#E67E22"),
        ("hssm_posterior_bias_ratio", "HSSM |bias| / limit", "#9DC3E6"),
        ("hssm_posterior_rmse_ratio", "HSSM RMSE / limit", "#4472C4"),
    )
    for _index, (_column, _label, _color) in enumerate(_series):
        _axis.bar(
            _x + (_index - 1.5) * _width,
            _aggregate[_column],
            _width,
            label=_label,
            color=_color,
        )
    _axis.axhline(1.0, color="#B22222", linestyle="--", label="frozen limit")
    _axis.set_xticks(_x, _aggregate["name"])
    _axis.set_ylabel("observed metric / allowed limit")
    _axis.set_title("Both estimators remain inside the recovery gate")
    _axis.set_ylim(0.0, 1.08)
    _axis.grid(axis="y", alpha=0.2)
    _axis.legend(ncol=2, fontsize=8)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo, report_frames, science):
    _thresholds = science["thresholds"]
    _diagnostics = report_frames["aggregate"][
        [
            "name",
            "hdi_inclusion_fraction",
            "maximum_rhat",
            "minimum_bulk_ess",
            "minimum_tail_ess",
            "maximum_mcse_sd_ratio",
        ]
    ].copy()
    _diagnostics["HDI floor"] = _thresholds["minimum_hdi_inclusion_fraction"]
    _diagnostics["R-hat ceiling"] = _thresholds["maximum_rhat"]
    _diagnostics["bulk ESS floor"] = _thresholds["minimum_bulk_ess"]
    _diagnostics["tail ESS floor"] = _thresholds["minimum_tail_ess"]
    _diagnostics["MCSE / SD ceiling"] = _thresholds["maximum_mcse_sd_ratio"]
    mo.ui.table(_diagnostics.round(4), selection=None, pagination=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Numerical handshake

    Each scenario evaluates three frozen parameter vectors through direct JEAM and the
    compiled HSSM objective, then runs both optimizers with the same seed and fixed
    budget. The table shows machine-precision agreement and identical returned vectors.

    These objective evaluations, optimizer outputs, and initial log densities are
    authenticated primary measurements. The verifier checks their finiteness and
    internal parity, but intentionally does not rerun JEAM, HSSM, or optimization.
    """)
    return


@app.cell
def _(mo, report_frames):
    _columns = {
        "scenario": "scenario",
        "objective_error": "maximum objective error",
        "optimizer_parameter_error": "optimizer parameter error",
        "optimizer_objective_error": "optimizer objective error",
        "initial_logp": "initial log density",
    }
    _table = report_frames["scenarios"].loc[:, list(_columns)].rename(columns=_columns)
    mo.ui.table(_table, selection=None, pagination=False)
    return


@app.cell
def _(mo, scenario_selector):
    mo.md(f"""
    ## Predictive checks: {scenario_selector.value.replace("_", " ")}

    The posterior-predictive RT quantiles are compared directly with the observed
    quantiles. Circular predictions are evaluated using angular distance and resultant
    length, avoiding an invalid ordinary mean across the `-π`/`π` seam. Prior-predictive
    RT ratios guard against a pathologically narrow or diffuse prior before fitting.
    """)
    return


@app.cell
def _(np, plt, selected_science):
    _predictive = selected_science["predictive"]
    _observed = np.asarray(_predictive["observed_rt_quantiles"])
    _simulated = np.asarray(_predictive["predictive_rt_quantiles"])
    _lower = min(_observed.min(), _simulated.min())
    _upper = max(_observed.max(), _simulated.max())
    _padding = max(0.05, 0.08 * (_upper - _lower))
    _figure, (_rt_axis, _circular_axis) = plt.subplots(1, 2, figsize=(9.0, 4.2))
    _rt_axis.plot(
        [_lower - _padding, _upper + _padding],
        [_lower - _padding, _upper + _padding],
        linestyle="--",
        color="#777777",
        label="observed = predictive",
    )
    _rt_axis.scatter(_observed, _simulated, s=65, color="#4472C4", zorder=3)
    for _probability, _x, _y in zip(
        _predictive["rt_probabilities"], _observed, _simulated, strict=True
    ):
        _rt_axis.annotate(
            f"p={_probability:.1f}", (_x, _y), xytext=(5, 5), textcoords="offset points"
        )
    _rt_axis.set_xlabel("observed RT quantile (s)")
    _rt_axis.set_ylabel("posterior-predictive RT quantile (s)")
    _rt_axis.set_title("RT quantiles")
    _rt_axis.grid(alpha=0.2)
    _rt_axis.legend(fontsize=8)

    _circle = np.linspace(-np.pi, np.pi, 300)
    _circular_axis.plot(np.cos(_circle), np.sin(_circle), color="#BBBBBB")
    for _label, _mean_key, _length_key, _color, _style in (
        (
            "posterior predictive",
            "predictive_mean_angle",
            "predictive_resultant_length",
            "#4472C4",
            "-",
        ),
        (
            "observed",
            "observed_mean_angle",
            "observed_resultant_length",
            "#111111",
            "--",
        ),
    ):
        _angle = _predictive[_mean_key]
        _length = _predictive[_length_key]
        _circular_axis.plot(
            [0.0, _length * np.cos(_angle)],
            [0.0, _length * np.sin(_angle)],
            color=_color,
            linestyle=_style,
            linewidth=2.5,
            marker="o",
            markevery=[1],
            label=_label,
        )
    _circular_axis.axhline(0.0, color="#DDDDDD", linewidth=0.8)
    _circular_axis.axvline(0.0, color="#DDDDDD", linewidth=0.8)
    _circular_axis.set_xlim(-1.08, 1.08)
    _circular_axis.set_ylim(-1.08, 1.08)
    _circular_axis.set_aspect("equal")
    _circular_axis.set_title("Circular resultant vectors")
    _circular_axis.legend(loc="lower right", fontsize=8)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo, pd, selected_scenario, selected_science, science):
    _predictive = selected_science["predictive"]
    _prior = selected_science["prior_predictive"]
    _thresholds = science["thresholds"]
    _predictive_table = pd.DataFrame(
        [
            {
                "check": "maximum RT quantile error",
                "observed": selected_scenario["maximum_rt_quantile_error"],
                "allowed": _thresholds["maximum_rt_quantile_absolute_error"],
            },
            {
                "check": "circular mean distance",
                "observed": _predictive["mean_angle_distance"],
                "allowed": _thresholds["maximum_mean_angle_distance"],
            },
            {
                "check": "resultant-length error",
                "observed": selected_scenario["resultant_length_error"],
                "allowed": _thresholds["maximum_resultant_length_absolute_error"],
            },
        ]
    )
    _prior_table = pd.DataFrame(
        {
            "RT quantile": [f"p={value:.1f}" for value in _prior["rt_probabilities"]],
            "observed (s)": _prior["observed_rt_quantiles"],
            "prior predictive (s)": _prior["prior_rt_quantiles"],
            "prior / observed": _prior["prior_to_observed_rt_ratios"],
        }
    )
    mo.vstack(
        [
            mo.md(
                f"Prior/observed ratios must stay in the frozen interval "
                f"`[{_thresholds['minimum_prior_to_observed_rt_ratio']}, "
                f"{_thresholds['maximum_prior_to_observed_rt_ratio']}]`."
            ),
            mo.ui.table(_prior_table.round(4), selection=None, pagination=False),
            mo.ui.table(_predictive_table.round(4), selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Runtime and Slice behavior

    These values describe this one run; they are hash-bound but deliberately excluded
    from the scientific pass/fail gate. Wall time is hardware dependent. Slice step
    counts help characterize the gradient-free sampler without turning performance into
    a scientific claim.
    """)
    return


@app.cell
def _(mo, report_frames):
    _columns = {
        "scenario": "scenario",
        "prior_predictive_seconds": "prior predictive (s)",
        "sampling_seconds": "sampling (s)",
        "posterior_predictive_seconds": "posterior predictive (s)",
        "mean_steps_in": "mean Slice steps in",
        "mean_steps_out": "mean Slice steps out",
    }
    _table = report_frames["scenarios"].loc[:, list(_columns)].rename(columns=_columns)
    mo.ui.table(_table.round(3), selection=None, pagination=False)
    return


@app.cell
def _(MANIFEST_SHA256, manifest, mo, science):
    _provenance = manifest["provenance"]
    _platform = _provenance["platform"]
    mo.md(f"""
    ## Integrity, provenance, and limits

    | Evidence field | Frozen value |
    |---|---|
    | Manifest schema | `{manifest["schema_version"]}` |
    | Result schema | `{science["schema_version"]}` |
    | Manifest SHA-256 | `{MANIFEST_SHA256}` |
    | Protocol SHA-256 | `{manifest["protocol_sha256"]}` |
    | Authenticated payloads | `{len(manifest["artifacts"])}` |
    | JEAM revision | `{_provenance["jeam_revision"]}` |
    | HSSM producer revision | `{_provenance["producer_revision"]}` |
    | HSSM producer tree | `{_provenance["producer_tree"]}` |
    | Python | `{_provenance["python"]["implementation"]} {_provenance["python"]["version"]}` |
    | Platform | `{_platform["system"]} {_platform["release"]} ({_platform["machine"]})` |

    The evidence supports the **fixed two-dimensional circular diffusion black-box
    prototype** for these four generating configurations with `s_v = 0` and `s_t = 0`.
    It does not establish calibrated long-run coverage, support the remaining JEAM model
    families, validate drift/nondecision variability, or make a default-sampler claim.

    Objective evaluations, optimizer execution, and the initial log density remain
    authenticated producer measurements rather than independently rerun calculations.
    All posterior, predictive, convergence, aggregate, and gate values shown here are
    recomputed from the retained raw evidence. The exact model producer is JEAM
    `{science["jeam_revision"]}`; this historical provenance must not be rewritten when
    the optional dependency pin advances later.

    To verify the bundle without opening marimo:

    ```bash
    uv run --group docs python scripts/verify_jeam_repeated_recovery_evidence.py
    ```
    """)
    return


if __name__ == "__main__":
    app.run()

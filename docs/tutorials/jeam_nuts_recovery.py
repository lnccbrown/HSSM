"""Inspect the fixed-CDM NUTS recovery and efficiency evidence.

This marimo notebook is artifact-backed. It imports neither HSSM nor JEAM and does not
run MCMC; every table and figure is reconstructed from the preregistered protocol and
committed canonical result.

Open it from the HSSM repository root::

    uv run --group docs marimo edit docs/tutorials/jeam_nuts_recovery.py

Export a path-clean static rendering with::

    uv run --group docs marimo export html docs/tutorials/jeam_nuts_recovery.py \
        --output /tmp/jeam-nuts-recovery.html
"""

# ruff: noqa: B018, D401, E501, PLR1711
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys as _sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    _repo_root = Path(__file__).resolve().parents[2]
    _sys.path.insert(0, str(_repo_root))
    from scripts.jeam_sampler_comparison_report import (
        SAMPLER_COLORS,
        SAMPLER_LABELS,
        SAMPLER_ORDER,
        build_fit_records,
        build_objective_records,
        build_parameter_records,
        build_predictive_records,
        build_promotion_records,
        summarize_artifact,
        validate_report_contract,
    )

    return (
        Path,
        SAMPLER_COLORS,
        SAMPLER_LABELS,
        SAMPLER_ORDER,
        build_fit_records,
        build_objective_records,
        build_parameter_records,
        build_predictive_records,
        build_promotion_records,
        json,
        mo,
        np,
        pd,
        plt,
        summarize_artifact,
        validate_report_contract,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # NUTS recovery and efficiency for JEAM circular diffusion

    This report answers the scientific question left open by the interactive likelihood
    laboratory: **can HSSM's analytical fixed circular-diffusion likelihood use NUTS
    without sacrificing recovery, diagnostics, or predictive behavior—and is it faster
    enough to recommend over the blackbox Slice baseline?**

    The answer for this preregistered study is **yes for both PyMC NUTS and NumPyro
    NUTS**. The evidence is intentionally narrow: four deterministic 300-trial recovery
    scenarios plus one nested 1,500-trial scaling case for the fixed two-dimensional CDM.
    """)
    return


@app.cell
def _(Path, json, validate_report_contract):
    repo_root = Path(__file__).resolve().parents[2]
    spec_path = (
        repo_root / "benchmarks" / "specs" / "jeam_fixed_cdm_sampler_comparison_v1.json"
    )
    artifact_path = (
        repo_root
        / "benchmarks"
        / "results"
        / "jeam_fixed_cdm_sampler_comparison_v1.json"
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    validate_report_contract(spec, artifact)
    return artifact, artifact_path, repo_root, spec, spec_path


@app.cell
def _(
    artifact,
    build_fit_records,
    build_objective_records,
    build_parameter_records,
    build_predictive_records,
    build_promotion_records,
    pd,
    spec,
    summarize_artifact,
):
    summary = summarize_artifact(artifact)
    parameter_df = pd.DataFrame(build_parameter_records(artifact))
    fit_df = pd.DataFrame(build_fit_records(artifact))
    objective_df = pd.DataFrame(build_objective_records(artifact))
    predictive_df = pd.DataFrame(build_predictive_records(spec, artifact))
    promotion_df = pd.DataFrame(build_promotion_records(spec, artifact))
    canonical_scenarios = [
        row["name"] for row in spec["scenarios"] if row["role"] == "canonical"
    ]
    return (
        canonical_scenarios,
        fit_df,
        objective_df,
        parameter_df,
        predictive_df,
        promotion_df,
        summary,
    )


@app.cell
def _(artifact, mo, summary):
    _scientific = "PASS" if artifact["scientific_gate"]["passed"] else "FAIL"
    _promotion = "PASS" if artifact["promotion_gate"]["passed"] else "FAIL"
    mo.md(f"""
    ## Result at a glance

    | Scientific gate | Promotion gate | Completed fits | Canonical HDI coverage | NUTS divergences | Maximum objective error |
    |---|---|---:|---:|---:|---:|
    | **{_scientific}** | **{_promotion}** | {summary["fit_count"]}/15 | {summary["canonical_truths_covered"]}/{summary["canonical_truth_count"]} | {summary["nuts_divergences"]} | {summary["maximum_objective_absolute_error"]:.2e} |

    Scientific acceptance and machine-local promotion were declared separately. NUTS
    could pass as scientifically usable without being faster enough to recommend; here,
    both NUTS backends pass both decisions.
    """)
    return


@app.cell
def _(artifact, mo, pd, spec):
    _sampler_specs = {row["id"]: row for row in spec["samplers"]}
    _rows = []
    for _sampler in spec["execution"]["sampler_order"]:
        _row = _sampler_specs[_sampler]
        _rows.append(
            {
                "route": {
                    "slice": "Blackbox + PyMC Slice",
                    "pymc_nuts": "Analytical + PyMC NUTS",
                    "numpyro_nuts": "Analytical + NumPyro NUTS",
                }[_sampler],
                "likelihood": _row["likelihood"],
                "backend": _row["backend"],
                "step": _row["step"],
                "chains": artifact["execution"]["chains"],
                "tune": artifact["execution"]["tune"],
                "draws": artifact["execution"]["draws"],
            }
        )
    mo.vstack(
        [
            mo.md("""
            ## A fair, frozen comparison

            Every route loaded the same saved `float64` dataset bytes, used the same
            priors and resolved initial values, and ran in a fresh subprocess. Each
            canonical fit used four chains with 1,000 tuning and 1,000 retained draws.
            Raw DataTrees were saved before summaries or posterior prediction.
            """),
            mo.ui.table(pd.DataFrame(_rows), selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(canonical_scenarios, mo, spec):
    scenario_selector = mo.ui.dropdown(
        options={name.replace("_", " ").title(): name for name in canonical_scenarios},
        value=canonical_scenarios[0].replace("_", " ").title(),
        label="Scenario",
    )
    parameter_selector = mo.ui.dropdown(
        options={name: name for name in spec["scope"]["scientific_parameter_order"]},
        value=spec["scope"]["scientific_parameter_order"][0],
        label="Parameter",
    )
    mo.hstack([scenario_selector, parameter_selector], justify="start", gap=2)
    return parameter_selector, scenario_selector


@app.cell
def _(mo):
    mo.md(r"""
    ## Why the likelihood is differentiable *almost everywhere*

    Let the scaled decision time be

    $$s = \frac{rt-t}{a^2}$$

    for the benchmark's fixed diffusion scale $\sigma=1$. JEAM uses the stable
    short-time representation for $s\leq0.002$, the long-time Bessel-series
    representation for $s\geq0.02$, and a linear density-space blend between them:

    $$w(s)=\operatorname{clip}\!\left(\frac{s-0.002}{0.02-0.002},0,1\right).$$

    Inside the overlap, the long series is also faded in according to its signal above a
    conservative summation-error bound $E$:

    $$r=\operatorname{clip}\!\left(\frac{\log(S/E)}{\log 100},0,1\right).$$

    This makes the likelihood smooth within each open region, but it is intentionally
    non-smooth at the support surfaces (`rt=t`, `a=0`, `t=0`, and the angular endpoints),
    the primary blend knots $s\in\{0.002,0.02\}$, and the reliability clipping surfaces
    $S/E\in\{1,100\}$. Those are lower-dimensional sets: a continuously distributed
    proposal has probability zero of landing on an exact knot. The numerical contract is
    therefore finite, accurate gradients at ordinary interior points—not a derivative at
    every exact boundary.
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
    _blend_axis.set_title("Primary representation blend")
    _blend_axis.grid(alpha=0.2)

    _reliability_axis.semilogx(
        _signal_ratio, _reliability, color="#E67E22", linewidth=2.5
    )
    _reliability_axis.axvline(1.0, color="#B22222", linestyle="--", linewidth=1)
    _reliability_axis.axvline(100.0, color="#B22222", linestyle="--", linewidth=1)
    _reliability_axis.set_xlabel("long-series signal / error bound")
    _reliability_axis.set_ylabel("reliability fade weight")
    _reliability_axis.set_title("Numerical reliability fade")
    _reliability_axis.grid(alpha=0.2)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(artifact, mo, objective_df, plt, spec):
    _scenario_error = objective_df.groupby("scenario", sort=False)[
        "maximum_absolute_error"
    ].max()
    _threshold = spec["objective_parity"]["maximum_absolute_error"]
    _figure, _axis = plt.subplots(figsize=(8.5, 3.8))
    _axis.barh(
        [name.replace("_", " ") for name in _scenario_error.index],
        _scenario_error.values,
        color="#70AD47",
    )
    _axis.axvline(_threshold, color="#B22222", linestyle="--", label="allowed error")
    _axis.set_xscale("log")
    _axis.set_xlabel("maximum absolute summed-log-likelihood difference")
    _axis.set_title("NumPy, JAX, and compiled HSSM objectives agree")
    _axis.legend()
    _figure.tight_layout()
    mo.vstack(
        [
            mo.md(f"""
            ### Value and gradient evidence

            Fifteen preregistered parameter points compare direct JEAM NumPy, direct JEAM
            JAX, and compiled HSSM objectives. The largest discrepancy is
            `{objective_df["maximum_absolute_error"].max():.2e}`, versus the frozen
            `{_threshold:.0e}` tolerance. Every analytical fit also compiled a finite,
            nonzero four-coordinate gradient before sampling. No finite likelihood floor
            was enabled in this study.

            Exact producer: JEAM `{artifact["provenance"]["jeam_revision"]}`.
            """),
            _figure,
        ]
    )
    return


@app.cell
def _(artifact, mo, np, pd, scenario_selector):
    _rows = []
    for _fit in artifact["fits"]:
        if (
            _fit["scenario"] == scenario_selector.value
            and _fit["initial_gradient"] is not None
        ):
            _gradient = np.asarray(_fit["initial_gradient"])
            _rows.append(
                {
                    "route": {
                        "pymc_nuts": "Analytical + PyMC NUTS",
                        "numpyro_nuts": "Analytical + NumPyro NUTS",
                    }[_fit["sampler"]],
                    "finite coordinates": int(np.isfinite(_gradient).sum()),
                    "nonzero coordinates": int(np.count_nonzero(_gradient)),
                    "gradient L2 norm": float(np.linalg.norm(_gradient)),
                    "compile + first evaluation (s)": _fit["runtime_seconds"][
                        "gradient_compile_and_first_eval"
                    ],
                }
            )
    mo.ui.table(pd.DataFrame(_rows).round(4), selection=None, pagination=False)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Parameter recovery

    Select a parameter above. The plot subtracts each scenario's generating truth, so
    zero always means exact recovery even though the four datasets have different true
    values. Points are posterior means and horizontal lines are 94% HDIs. The scaling
    case is excluded because its preregistered role is performance, not recovery.
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
    _axis.set_xlabel("posterior estimate and 94% HDI minus generating truth")
    _axis.set_title(
        f"Recovery of {parameter_selector.value} across canonical scenarios"
    )
    _axis.grid(axis="x", alpha=0.2)
    _axis.legend(fontsize=8)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo, parameter_df, scenario_selector):
    _columns = {
        "route": "route",
        "parameter": "parameter",
        "truth": "truth",
        "posterior_mean": "posterior mean",
        "hdi_lower": "94% HDI lower",
        "hdi_upper": "94% HDI upper",
        "truth_in_hdi": "covers truth",
        "rhat": "R-hat",
        "ess_bulk": "bulk ESS",
        "ess_tail": "tail ESS",
        "mcse_over_posterior_sd": "MCSE/SD",
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
    ## Diagnostics

    Each bar below is the worst parameter-level diagnostic in the selected fit. The red
    line is the preregistered boundary: R-hat and MCSE/SD must remain below it, while both
    ESS measures must remain above it. Slice does not define NUTS divergences; both NUTS
    routes recorded zero.
    """)
    return


@app.cell
def _(SAMPLER_COLORS, SAMPLER_ORDER, fit_df, np, plt, scenario_selector, spec):
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
    _thresholds = spec["scientific_acceptance"]
    _figure, (_rhat_axis, _ess_axis, _mcse_axis) = plt.subplots(1, 3, figsize=(10, 3.8))
    _rhat_axis.bar(_labels, _selected["maximum_rhat"], color=_colors)
    _rhat_axis.axhline(
        _thresholds["maximum_rhat_exclusive"], color="#B22222", linestyle="--"
    )
    _rhat_axis.set_ylim(0.99, 1.011)
    _rhat_axis.set_title("Maximum R-hat")

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
    _ess_axis.set_title("Minimum ESS")
    _ess_axis.legend(fontsize=8)

    _mcse_axis.bar(_labels, _selected["maximum_mcse_over_sd"], color=_colors)
    _mcse_axis.axhline(
        _thresholds["maximum_mcse_over_posterior_sd_exclusive"],
        color="#B22222",
        linestyle="--",
    )
    _mcse_axis.set_title("Maximum MCSE / SD")
    for _axis in (_rhat_axis, _ess_axis, _mcse_axis):
        _axis.grid(axis="y", alpha=0.2)
        _axis.tick_params(axis="x", labelsize=8)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md("""
    ## Posterior prediction

    Prediction is judged in the geometry of each observation: absolute errors for the
    10th, 50th, and 90th RT quantiles; circular distance between mean angles; and absolute
    error in mean resultant length. The plot divides the worst RT-quantile error and both
    circular errors by their own preregistered limits. Every bar must remain below one.
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
    _axis.axhline(1.0, color="#B22222", linestyle="--", label="allowed limit")
    _axis.set_xticks(_x, _labels)
    _axis.set_ylabel("observed error / allowed limit")
    _axis.set_title("Predictive checks remain inside every gate")
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
    ## Efficiency and wall time

    Promotion was deliberately harder than scientific acceptance. Each NUTS backend had
    to deliver at least `1.5x` Slice's median *worst-parameter* bulk ESS per sampling
    second, keep every canonical total time below `1.25x` Slice, and avoid degraded
    normalized efficiency in the nested 1,500-trial case. Timings are local evidence,
    not CI thresholds.
    """)
    return


@app.cell
def _(mo, promotion_df):
    _table = promotion_df[
        [
            "route",
            "efficiency_ratio_vs_slice",
            "maximum_total_time_ratio_vs_slice",
            "scale_normalized_efficiency_ratio",
        ]
    ].rename(
        columns={
            "efficiency_ratio_vs_slice": "median min bulk ESS/s vs Slice",
            "maximum_total_time_ratio_vs_slice": "worst total-time ratio vs Slice",
            "scale_normalized_efficiency_ratio": "1,500-trial normalized efficiency",
        }
    )
    mo.ui.table(_table.round(3), selection=None, pagination=False)
    return


@app.cell
def _(SAMPLER_COLORS, np, plt, promotion_df):
    _samplers = promotion_df["sampler"].tolist()
    _labels = ["PyMC NUTS", "NumPyro NUTS"]
    _colors = [SAMPLER_COLORS[name] for name in _samplers]
    _x = np.arange(len(_samplers))
    _width = 0.34
    _figure, (_gain_axis, _time_axis) = plt.subplots(1, 2, figsize=(9, 3.8))
    _gain_axis.bar(
        _x - _width / 2,
        promotion_df["efficiency_ratio_vs_slice"],
        _width,
        color=_colors,
        label="canonical ESS/s ratio",
    )
    _gain_axis.bar(
        _x + _width / 2,
        promotion_df["scale_normalized_efficiency_ratio"],
        _width,
        color=_colors,
        alpha=0.5,
        hatch="//",
        label="scaling ratio",
    )
    _gain_axis.axhline(1.5, color="#B22222", linestyle="--", label="ESS/s gate")
    _gain_axis.axhline(0.8, color="#7030A0", linestyle=":", label="scaling gate")
    _gain_axis.set_xticks(_x, _labels)
    _gain_axis.set_ylabel("ratio")
    _gain_axis.set_title("Efficiency gains")
    _gain_axis.legend(fontsize=7)

    _time_axis.bar(
        _labels,
        promotion_df["maximum_total_time_ratio_vs_slice"],
        color=_colors,
    )
    _time_axis.axhline(1.25, color="#B22222", linestyle="--", label="maximum allowed")
    _time_axis.axhline(1.0, color="#111111", linestyle=":", label="equal to Slice")
    _time_axis.set_ylabel("worst canonical total-time ratio")
    _time_axis.set_title("NUTS remains faster in every scenario")
    _time_axis.legend(fontsize=7)
    for _axis in (_gain_axis, _time_axis):
        _axis.grid(axis="y", alpha=0.2)
        _axis.tick_params(axis="x", labelsize=8)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(SAMPLER_COLORS, SAMPLER_ORDER, fit_df, np, plt, scenario_selector):
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
    _axis.set_ylabel("seconds")
    _axis.set_title(
        f"Compile, sampling, and prediction time: {scenario_selector.value.replace('_', ' ')}"
    )
    _axis.grid(axis="y", alpha=0.2)
    _axis.legend(fontsize=8)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(fit_df, mo):
    _scale = fit_df.loc[fit_df["role"] == "scale"][
        [
            "route",
            "trials",
            "truths_covered",
            "truth_count",
            "minimum_bulk_ess",
            "minimum_tail_ess",
            "maximum_rhat",
            "sampling_seconds",
        ]
    ].copy()
    _scale["HDI coverage"] = (
        _scale["truths_covered"].astype(str) + "/" + _scale["truth_count"].astype(str)
    )
    mo.vstack(
        [
            mo.md("""
            ### Interpret the scaling case correctly

            The 1,500-trial dataset is an exact 300-row prefix extension of the baseline
            data. Its gate covers diagnostics and normalized efficiency only. All three
            samplers place three of four truths inside their 94% HDIs here; that is shown
            transparently but was neither counted toward nor required by canonical
            recovery acceptance.
            """),
            mo.ui.table(
                _scale[
                    [
                        "route",
                        "trials",
                        "HDI coverage",
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
def _(artifact, mo, promotion_df):
    _pymc = promotion_df.set_index("sampler").loc["pymc_nuts"]
    _numpyro = promotion_df.set_index("sampler").loc["numpyro_nuts"]
    mo.md(f"""
    ## Recommendation and limits

    - **Analytical + PyMC NUTS is now the ordinary first recommendation** for this fixed
      CDM prototype: it passes every scientific gate and provides `{_pymc["efficiency_ratio_vs_slice"]:.2f}x`
      the median worst-parameter bulk ESS/s of Slice.
    - **Analytical + NumPyro NUTS is a validated alternative** and was fastest on this
      machine, at `{_numpyro["efficiency_ratio_vs_slice"]:.2f}x` Slice's efficiency.
    - **Blackbox + Slice remains the independent reference and gradient-free fallback.**
      The evidence supports changing documentation, not silently changing HSSM's global
      likelihood or sampler defaults.

    This is a deterministic regression study, not simulation-based calibration. It does
    not establish nominal long-run coverage, hierarchical-model performance, GPU scaling,
    or support for JEAM's spherical, hyperspherical, projected, variable-threshold, or
    variability-enabled models. Exact support and blend-knot derivatives also remain
    outside the differentiability contract.

    Environment: Python `{artifact["provenance"]["python_version"]}`, PyMC
    `{artifact["provenance"]["package_versions"]["pymc"]}`, JAX
    `{artifact["provenance"]["package_versions"]["jax"]}` on
    `{artifact["provenance"]["platform"]}`.
    """)
    return


@app.cell
def _(artifact, artifact_path, mo, spec, spec_path):
    mo.md(f"""
    ## Reproduce or audit

    This notebook reads only:

    - `{spec_path.relative_to(spec_path.parents[2])}` — protocol frozen at
      `{spec["frozen_at_utc"]}`;
    - `{artifact_path.relative_to(artifact_path.parents[2])}` — canonical result generated
      at `{artifact["generated_at_utc"]}`.

    The protocol's result command is:

    ```bash
    uv run --python 3.12 --group jeam-prototype python -m \
      scripts.benchmark_jeam_sampler_comparison \
      --work-dir /tmp/jeam-sampler-v1 \
      --output benchmarks/results/jeam_fixed_cdm_sampler_comparison_v1.json
    ```

    Regeneration is intentionally not a notebook button: the canonical evidence was run
    once after preregistration, and a scientifically necessary change requires a new
    versioned protocol rather than overwriting v1.
    """)
    return


if __name__ == "__main__":
    app.run()

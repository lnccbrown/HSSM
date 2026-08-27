"""Tests for the schema-v2 JEAM repeated-recovery report views."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from scripts.jeam_repeated_recovery_report import (
    build_report_frames,
    load_report,
    summarize_science,
)
from scripts.verify_jeam_repeated_recovery_evidence import (
    MANIFEST_SHA256,
    SCENARIOS,
    EvidenceMismatch,
)

BUNDLE = (
    Path(__file__).parents[1] / "benchmarks" / "evidence" / "jeam_repeated_recovery_v2"
)


@pytest.fixture(scope="module")
def report():
    """Load the authenticated canonical report once."""
    return load_report(BUNDLE)


def test_report_loads_authenticated_recomputed_science(report) -> None:
    """The report must inherit the verifier's manifest root and independent gate."""
    manifest, science, _ = report

    assert MANIFEST_SHA256 == (
        "d8a5c458d2194f1fb7031f6bc5ca5add3cd67afabd028880dc0bfed887ef9972"
    )
    assert manifest["provenance"]["jeam_revision"] == science["jeam_revision"]
    assert manifest["protocol"]["thresholds"] == science["thresholds"]
    assert science["gate"] == {"passed": True, "failures": []}


def test_report_summary_pins_the_canonical_outcome(report) -> None:
    """Headline numbers must remain derived from all scenarios and parameters."""
    _, science, _ = report
    summary = summarize_science(science)

    assert summary == {
        "passed": True,
        "scenarios": 4,
        "hdi_inclusions": 16,
        "hdi_total": 16,
        "maximum_rhat": pytest.approx(1.0076487),
        "minimum_bulk_ess": pytest.approx(786.13471011),
        "minimum_tail_ess": pytest.approx(1279.99191356),
        "maximum_mcse_sd_ratio": pytest.approx(0.03568501073939003),
        "maximum_objective_error": pytest.approx(2.2737367544323206e-13),
        "maximum_optimizer_parameter_error": 0.0,
    }


def test_report_frames_preserve_order_terminology_and_gate_math(report) -> None:
    """Presentation reshaping must retain all rows and fixed-budget terminology."""
    _, science, frames = report
    parameters = frames["parameters"]
    scenarios = frames["scenarios"]
    aggregate = frames["aggregate"]

    expected_scenarios = [name for name, *_ in SCENARIOS]
    assert scenarios["scenario"].tolist() == expected_scenarios
    assert parameters["scenario"].tolist() == [
        name for name in expected_scenarios for _ in science["parameter_order"]
    ]
    assert parameters["name"].tolist() == science["parameter_order"] * 4
    assert len(parameters) == 16
    assert "jeam_fixed_budget_estimate" in parameters
    assert not any("mle" in column.lower() for column in parameters.columns)

    ratio_columns = [
        "jeam_fixed_budget_bias_ratio",
        "jeam_fixed_budget_rmse_ratio",
        "hssm_posterior_bias_ratio",
        "hssm_posterior_rmse_ratio",
    ]
    assert np.all(aggregate[ratio_columns].to_numpy() <= 1.0)


def test_report_scenario_metrics_are_recomputed_from_predictive_rows(report) -> None:
    """Flattened predictive metrics must remain exact functions of verified science."""
    _, science, frames = report
    first = science["scenarios"][0]
    predictive = first["predictive"]
    row = frames["scenarios"].iloc[0]

    expected_rt_error = np.max(
        np.abs(
            np.asarray(predictive["observed_rt_quantiles"])
            - np.asarray(predictive["predictive_rt_quantiles"])
        )
    )
    assert row["maximum_rt_quantile_error"] == pytest.approx(expected_rt_error)
    assert row["resultant_length_error"] == pytest.approx(
        abs(
            predictive["observed_resultant_length"]
            - predictive["predictive_resultant_length"]
        )
    )
    assert row["mean_angle_distance"] == pytest.approx(
        predictive["mean_angle_distance"]
    )


def test_report_ratios_use_inclusive_frozen_limits(report) -> None:
    """Presentation ratios equal one exactly at each verifier-owned boundary."""
    _, science, _ = report
    altered = copy.deepcopy(science)
    row = altered["aggregate"][0]
    name = row["name"]
    bias_limit = science["thresholds"]["maximum_absolute_bias"][name]
    rmse_limit = science["thresholds"]["maximum_rmse"][name]
    row.update(
        jeam_fixed_budget_bias=-bias_limit,
        jeam_fixed_budget_rmse=rmse_limit,
        hssm_posterior_bias=bias_limit,
        hssm_posterior_rmse=rmse_limit,
        maximum_rhat=1.5,
    )

    aggregate = build_report_frames(altered)["aggregate"].iloc[0]
    assert aggregate[
        [
            "jeam_fixed_budget_bias_ratio",
            "jeam_fixed_budget_rmse_ratio",
            "hssm_posterior_bias_ratio",
            "hssm_posterior_rmse_ratio",
        ]
    ].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert summarize_science(altered)["maximum_rhat"] == pytest.approx(1.5)


def test_report_helper_cannot_run_inference_or_read_compact_results() -> None:
    """The helper may only present the public network-free verifier result."""
    source = (
        Path(__file__).parents[1] / "scripts" / "jeam_repeated_recovery_report.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert imports.isdisjoint({"hssm", "jeam", "pymc", "socket", "subprocess"})
    assert "result.json" not in source
    assert "FULL_RUN" not in source
    assert "run_button" not in source
    assert not any(
        prefix in source
        for prefix in ("/Users/", "/home/", "/private/", "/var/folders/", "file://")
    )


def test_report_fails_closed_with_the_verifier(tmp_path: Path) -> None:
    """The presentation entry point may not bypass bundle authentication."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(EvidenceMismatch, match="inventory mismatch"):
        load_report(tmp_path)

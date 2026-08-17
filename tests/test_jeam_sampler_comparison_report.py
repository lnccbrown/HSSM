"""Tests for the artifact-backed JEAM sampler comparison report."""

import copy
import json
from pathlib import Path

import pytest

from scripts.jeam_sampler_comparison_report import (
    SAMPLER_LABELS,
    build_fit_records,
    build_objective_records,
    build_parameter_records,
    build_predictive_records,
    build_promotion_records,
    summarize_artifact,
    validate_report_contract,
)

ROOT = Path(__file__).parents[1]
SPEC_PATH = ROOT / "benchmarks" / "specs" / "jeam_fixed_cdm_sampler_comparison_v1.json"
ARTIFACT_PATH = (
    ROOT / "benchmarks" / "results" / "jeam_fixed_cdm_sampler_comparison_v1.json"
)


@pytest.fixture(scope="module")
def spec():
    """Load the frozen protocol."""
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifact():
    """Load the compact canonical result."""
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_report_contract_rejects_incomplete_or_mismatched_evidence(spec, artifact):
    """The notebook must not silently render a partial or unrelated study."""
    validate_report_contract(spec, artifact)

    incomplete = copy.deepcopy(artifact)
    incomplete["fits"].pop()
    with pytest.raises(ValueError, match="one fit per scenario and sampler"):
        validate_report_contract(spec, incomplete)

    mismatched = copy.deepcopy(artifact)
    mismatched["study_id"] = "another-study"
    with pytest.raises(ValueError, match="does not belong"):
        validate_report_contract(spec, mismatched)


def test_recovery_records_exclude_the_performance_only_scale_case(artifact):
    """Canonical coverage must not be inflated by the separate scaling scenario."""
    parameters = build_parameter_records(artifact)
    fits = build_fit_records(artifact)

    assert len(parameters) == 4 * 3 * 4 == 48
    assert {row["scenario"] for row in parameters} == {
        "baseline_asymmetric",
        "reversed_drift",
        "high_threshold_strong_drift",
        "low_threshold_negative_drift",
    }
    assert all(row["truth_in_hdi"] for row in parameters)
    assert len(fits) == 15
    scale_rows = [row for row in fits if row["role"] == "scale"]
    assert len(scale_rows) == 3
    assert {row["truths_covered"] for row in scale_rows} == {3}


def test_headline_summary_is_recomputed_from_raw_rows(artifact):
    """Headline claims should reflect fit-level evidence, not stored gate booleans."""
    summary = summarize_artifact(artifact)

    assert summary["fit_count"] == 15
    assert summary["canonical_fit_count"] == 12
    assert summary["canonical_truths_covered"] == 48
    assert summary["canonical_truth_count"] == 48
    assert summary["nuts_divergences"] == 0
    assert summary["maximum_rhat"] < 1.01
    assert summary["minimum_bulk_ess"] > 400
    assert summary["minimum_tail_ess"] > 400
    assert summary["maximum_mcse_over_sd"] < 0.05
    assert summary["maximum_objective_absolute_error"] == pytest.approx(
        4.547473508864641e-13
    )


def test_objective_and_predictive_records_preserve_every_comparison(spec, artifact):
    """Presentation rows should retain all parity points and canonical PPC routes."""
    objective = build_objective_records(artifact)
    predictive = build_predictive_records(spec, artifact)

    assert len(objective) == 5 * 3 == 15
    assert {row["candidate"] for row in objective} == {1, 2, 3}
    assert max(row["maximum_absolute_error"] for row in objective) < 1e-8
    assert len(predictive) == 4 * 3 == 12
    assert {row["route"] for row in predictive} == set(SAMPLER_LABELS.values())
    assert max(row["rt_fraction_of_limit"] for row in predictive) <= 1.0
    assert max(row["angle_fraction_of_limit"] for row in predictive) <= 1.0
    assert max(row["resultant_fraction_of_limit"] for row in predictive) <= 1.0


def test_promotion_records_recompute_both_nuts_decisions(spec, artifact):
    """Efficiency, wall-time, and scaling ratios should match the frozen result."""
    records = build_promotion_records(spec, artifact)

    assert [row["sampler"] for row in records] == ["pymc_nuts", "numpyro_nuts"]
    for row in records:
        stored = artifact["promotion_gate"]["by_sampler"][row["sampler"]]
        assert row["efficiency_ratio_vs_slice"] == pytest.approx(
            stored["canonical_efficiency_ratio_vs_slice"]
        )
        assert row["maximum_total_time_ratio_vs_slice"] == pytest.approx(
            max(stored["canonical_total_seconds_ratios_vs_slice"].values())
        )
        assert row["total_time_ratios_vs_slice"] == pytest.approx(
            stored["canonical_total_seconds_ratios_vs_slice"]
        )
        assert row["scale_normalized_efficiency_ratio"] == pytest.approx(
            stored["scale_normalized_efficiency_ratio_vs_reference"]
        )
        assert row["efficiency_ratio_vs_slice"] >= 1.5
        assert row["maximum_total_time_ratio_vs_slice"] <= 1.25
        assert row["scale_normalized_efficiency_ratio"] >= 0.8

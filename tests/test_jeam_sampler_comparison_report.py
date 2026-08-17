"""Tests for the verifier-backed fixed-CDM sampler report data."""

import ast
import copy
import inspect

import pytest

import scripts.jeam_sampler_comparison_report as report_module
from scripts.jeam_sampler_comparison_report import (
    SAMPLER_LABELS,
    load_sampler_comparison_report,
    validate_report_contract,
)
from scripts.verify_jeam_sampler_comparison_evidence import (
    load_verified_sampler_report_evidence,
)

EXPECTED_REVISIONS = {
    "historical_analytical_result": "0c0ef8b834dd062ad8aea5ff8e7a09dfb55492ce",
    "durable_blackbox_reference": "a9f547b3630ae8ff31ccec1b904e0c02fdba6d99",
    "current_safety_revision": "ede7a4f4faf226e4dae52c84dfb01012939cccdc",
}


@pytest.fixture(scope="module")
def verified_documents():
    """Load the summary and its authenticated source documents together."""
    return load_verified_sampler_report_evidence()


@pytest.fixture(scope="module")
def report():
    """Build the canonical report through its only public loading path."""
    return load_sampler_comparison_report()


def test_canonical_report_loads_only_verified_documents(verified_documents, report):
    """The canonical API exposes the same authenticated study to every table."""
    verification, spec, artifact = verified_documents

    assert verification["study_id"] == spec["study_id"] == artifact["study_id"]
    assert report["verification"]["study_id"] == verification["study_id"]
    assert report["specification"] == spec
    assert report["compact_result"] == artifact
    assert len(report["fit_records"]) == verification["counts"]["fits"] == 15


def test_report_contract_rejects_incomplete_or_mismatched_evidence(
    verified_documents,
):
    """A partial or unrelated compact result cannot reach presentation code."""
    verification, spec, artifact = verified_documents
    validate_report_contract(verification, spec, artifact)

    incomplete = copy.deepcopy(artifact)
    incomplete["fits"].pop()
    with pytest.raises(ValueError, match="one fit per scenario and sampler"):
        validate_report_contract(verification, spec, incomplete)

    mismatched = copy.deepcopy(artifact)
    mismatched["study_id"] = "another-study"
    with pytest.raises(ValueError, match="does not belong"):
        validate_report_contract(verification, spec, mismatched)


def test_report_distinguishes_unique_truths_from_route_hdi_checks(report):
    """Four scenarios by four parameters are not 48 independent truths."""
    summary = report["summary"]
    parameters = report["parameter_records"]

    assert summary["unique_truth_count"] == 4 * 4 == 16
    assert summary["route_hdi_inclusions"] == 4 * 3 * 4 == 48
    assert summary["route_hdi_checks"] == 4 * 3 * 4 == 48
    assert len(parameters) == 48
    assert len({(row["scenario"], row["parameter"]) for row in parameters}) == 16
    assert all(row["truth_in_hdi"] for row in parameters)
    assert {
        "canonical_truth_count",
        "canonical_truths_covered",
    }.isdisjoint(summary)


def test_headline_diagnostics_are_recomputed_from_compact_rows(report):
    """Headline diagnostics retain the compact study's bounded smoke result."""
    summary = report["summary"]

    assert summary["fit_count"] == 15
    assert summary["canonical_fit_count"] == 12
    assert summary["nuts_divergences"] == 0
    assert summary["maximum_rhat"] < 1.01
    assert summary["minimum_bulk_ess"] > 400
    assert summary["minimum_tail_ess"] > 400
    assert summary["maximum_mcse_over_sd"] < 0.05
    assert summary["maximum_objective_absolute_error"] == pytest.approx(
        4.547473508864641e-13
    )


def test_report_retains_all_three_jeam_revisions_and_promotion_block(report):
    """Historical evidence is not relabeled as a current-revision rerun."""
    boundary = report["evidence_boundary"]

    assert boundary["jeam_revisions"] == EXPECTED_REVISIONS
    assert boundary["ecosystem_promotion_blocked"] is True
    assert report["summary"]["ecosystem_promotion_blocked"] is True
    assert boundary["efficiency_scope"] == "recorded-machine only"
    assert boundary["portable_performance_claim"] is False
    assert any(
        "current JEAM safety revision was not rerun" in blocker
        for blocker in boundary["ecosystem_promotion_blockers"]
    )


def test_report_keeps_missing_raw_evidence_visible(report):
    """Compact summaries cannot stand in for traces, draws, or lock bytes."""
    retention = report["evidence_boundary"]["retention"]
    blockers = report["evidence_boundary"]["ecosystem_promotion_blockers"]

    assert retention["raw_trace_files_retained"] == 0
    assert retention["raw_prior_predictive_draws_retained"] is False
    assert retention["raw_posterior_predictive_draws_retained"] is False
    assert retention["sampler_backend_trace_attributes_retained"] is False
    assert retention["historical_uv_lock_bytes_retained"] is False
    assert any("backend identity" in blocker for blocker in blockers)
    assert any("prior- and posterior-predictive" in blocker for blocker in blockers)
    assert any("uv.lock bytes" in blocker for blocker in blockers)


def test_efficiency_rows_are_recorded_machine_observations(report):
    """Passing timing thresholds does not imply portable ecosystem promotion."""
    rows = report["efficiency_records"]

    assert [row["sampler"] for row in rows] == ["pymc_nuts", "numpyro_nuts"]
    assert {row["route"] for row in rows} <= set(SAMPLER_LABELS.values())
    assert all(row["recorded_machine_gate_passed"] for row in rows)
    assert all(row["evidence_scope"] == "recorded-machine only" for row in rows)
    assert not any(row["portable_performance_claim"] for row in rows)
    assert all(row["efficiency_ratio_vs_slice"] >= 1.5 for row in rows)
    assert all(row["maximum_total_time_ratio_vs_slice"] <= 1.25 for row in rows)
    assert all(row["scale_normalized_efficiency_ratio"] >= 0.8 for row in rows)
    assert report["summary"]["ecosystem_promotion_blocked"] is True


def test_objective_and_predictive_tables_preserve_every_compact_comparison(report):
    """The report retains all preflight points and canonical predictive routes."""
    objective = report["objective_records"]
    predictive = report["predictive_records"]

    assert len(objective) == 5 * 3 == 15
    assert {row["candidate"] for row in objective} == {1, 2, 3}
    assert max(row["maximum_absolute_error"] for row in objective) < 1e-8
    assert len(predictive) == 4 * 3 == 12
    assert {row["route"] for row in predictive} == set(SAMPLER_LABELS.values())
    assert max(row["rt_fraction_of_limit"] for row in predictive) <= 1.0
    assert max(row["angle_fraction_of_limit"] for row in predictive) <= 1.0
    assert max(row["resultant_fraction_of_limit"] for row in predictive) <= 1.0


def test_report_loader_calls_the_verifier_once(monkeypatch, verified_documents):
    """Spec and result reuse the verifier's single authenticated snapshot."""
    calls = []

    def fake_verified_load(root):
        calls.append(root)
        return copy.deepcopy(verified_documents)

    monkeypatch.setattr(
        report_module, "load_verified_sampler_report_evidence", fake_verified_load
    )
    loaded = report_module.load_sampler_comparison_report("sentinel-root")

    assert calls == ["sentinel-root"]
    assert loaded["summary"]["unique_truth_count"] == 16
    assert loaded["summary"]["route_hdi_checks"] == 48


def test_report_module_has_no_direct_evidence_reads():
    """Only the hash-before-parse verifier may touch frozen evidence bytes."""
    tree = ast.parse(inspect.getsource(report_module))
    forbidden_calls = []
    verifier_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            assert "json" not in names
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            if node.func.id == "open":
                forbidden_calls.append(node.func.id)
            if node.func.id == "load_verified_sampler_report_evidence":
                verifier_calls += 1
        elif isinstance(node.func, ast.Attribute) and node.func.attr in {
            "open",
            "read_bytes",
            "read_text",
        }:
            forbidden_calls.append(node.func.attr)

    assert forbidden_calls == []
    assert verifier_calls == 1

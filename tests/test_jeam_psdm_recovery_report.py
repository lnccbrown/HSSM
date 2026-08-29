"""Tests for verifier-backed fixed-PSDM compact report data."""

import ast
import copy
import inspect
from pathlib import Path

import pytest

import scripts.jeam_psdm_recovery_report as report_module
import scripts.verify_jeam_psdm_recovery_evidence as verifier_module
from scripts.jeam_psdm_recovery_report import (
    REPORT_KEYS,
    load_psdm_recovery_report,
)
from scripts.verify_jeam_psdm_recovery_evidence import (
    CURRENT_SAFETY_JEAM_REVISION,
    EXPECTED_FAILURES,
    HISTORICAL_JEAM_REVISION,
    PARAMETER_ORDER,
    REPO_ROOT,
    SCENARIO_ORDER,
    EvidenceIntegrityError,
    load_verified_psdm_recovery_evidence,
    verify_psdm_recovery_documents,
)


@pytest.fixture(scope="module")
def verified_documents():
    """Load the authenticated documents and recomputed records once."""
    return load_verified_psdm_recovery_evidence()


@pytest.fixture(scope="module")
def report():
    """Build the canonical compact report through its public boundary."""
    return load_psdm_recovery_report()


def test_authentication_precedes_json_parsing(tmp_path, monkeypatch):
    """Malformed corrupt bytes must fail their pin without reaching JSON."""
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_bytes(b"{")

    def forbidden_parse(*_args, **_kwargs):
        raise AssertionError("corrupt unauthenticated bytes reached json.loads")

    monkeypatch.setattr(verifier_module.json, "loads", forbidden_parse)
    with pytest.raises(EvidenceIntegrityError, match="SHA256 mismatch"):
        verifier_module._load_authenticated_json(corrupt, "0" * 64)


def test_verifier_snapshots_each_frozen_file_once(monkeypatch):
    """Hashing and parsing must share one byte snapshot per source file."""
    original = Path.read_bytes
    counts = {
        verifier_module.SPEC_PATH.as_posix(): 0,
        verifier_module.ADDENDUM_PATH.as_posix(): 0,
        verifier_module.RESULT_PATH.as_posix(): 0,
    }

    def counted_read(path):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in counts:
            counts[relative] += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read)
    load_verified_psdm_recovery_evidence(REPO_ROOT)

    assert counts == {name: 1 for name in counts}


def test_report_has_stable_keys_and_exact_record_cardinalities(report):
    """The notebook receives one unambiguous row for every compact quantity."""
    assert tuple(report) == REPORT_KEYS
    assert len(report["parameter_records"]) == 16
    assert len(report["scenario_records"]) == 4
    assert len(report["aggregate_records"]) == 4
    assert len(report["objective_records"]) == 12
    assert len(report["predictive_records"]) == 4
    assert len(report["failure_records"]) == 8

    assert [row["scenario"] for row in report["scenario_records"]] == list(
        SCENARIO_ORDER
    )
    assert [row["parameter"] for row in report["aggregate_records"]] == list(
        PARAMETER_ORDER
    )
    assert [
        (row["scenario"], row["parameter"]) for row in report["parameter_records"]
    ] == [
        (scenario, parameter)
        for scenario in SCENARIO_ORDER
        for parameter in PARAMETER_ORDER
    ]


def test_summary_counts_unique_truths_and_failed_gate(report):
    """Four scenarios by four parameters means 16 unique truth checks."""
    summary = report["summary"]
    unique_truths = {
        (row["scenario"], row["parameter"]) for row in report["parameter_records"]
    }

    assert summary["unique_truth_count"] == len(unique_truths) == 16
    assert (summary["truth_in_hdi"], summary["truth_total"]) == (14, 16)
    assert summary["hdi_coverage"] == pytest.approx(14 / 16)
    assert summary["failure_count"] == 8
    assert summary["overall_pass"] is False
    assert summary["ecosystem_promotion_blocked"] is True


def test_failure_rows_are_recomputed_in_exact_historical_order(report):
    """The report must preserve the eight derivable gate failures exactly."""
    failures = report["failure_records"]

    assert tuple(row["message"] for row in failures) == EXPECTED_FAILURES
    assert [row["order"] for row in failures] == list(range(1, 9))
    assert [row["category"] for row in failures] == [
        "rhat",
        "bulk_ess",
        "mcse_over_posterior_sd",
        "rhat",
        "bulk_ess",
        "mcse_over_posterior_sd",
        "optimizer_absolute_error",
        "optimizer_rmse",
    ]


def test_parameter_and_aggregate_limits_explain_the_failures(report):
    """Individual and aggregate flags are derived from frozen thresholds."""
    parameters = {
        (row["scenario"], row["parameter"]): row for row in report["parameter_records"]
    }
    high_a = parameters[("high_threshold_strong_radial", "a")]
    high_t = parameters[("high_threshold_strong_radial", "t")]
    low_vy = parameters[("low_threshold_balanced_drift", "v_y")]

    for row in (high_a, high_t):
        assert row["rhat_passed"] is False
        assert row["bulk_ess_passed"] is False
        assert row["mcse_passed"] is False
        assert row["tail_ess_passed"] is True
        assert row["diagnostics_passed"] is False
    assert low_vy["optimizer_absolute_error"] == pytest.approx(0.631520764620942)
    assert low_vy["optimizer_absolute_error_limit"] == 0.45
    assert low_vy["optimizer_recovery_passed"] is False

    aggregate = {row["parameter"]: row for row in report["aggregate_records"]}
    assert aggregate["v_y"]["optimizer_rmse"] > aggregate["v_y"]["optimizer_rmse_limit"]
    assert aggregate["v_y"]["optimizer_rmse_passed"] is False
    assert all(row["posterior_bias_passed"] for row in aggregate.values())
    assert all(row["posterior_rmse_passed"] for row in aggregate.values())
    assert all(row["hdi_coverage_passed"] for row in aggregate.values())


def test_objective_and_predictive_records_are_bounded_compact_summaries(report):
    """Passing subgates stay distinct from the failed overall recovery gate."""
    objective = report["objective_records"]
    predictive = report["predictive_records"]

    assert {row["candidate"] for row in objective} == {1, 2, 3}
    assert all(row["passed"] for row in objective)
    assert max(row["absolute_error"] for row in objective) == pytest.approx(
        report["summary"]["maximum_objective_absolute_error"]
    )
    assert all(row["passed"] for row in predictive)
    assert all(
        row["evidence_scope"] == "authenticated producer summaries; raw draws absent"
        for row in predictive
    )


def test_evidence_boundary_keeps_every_missing_source_visible(report):
    """Compact summaries cannot masquerade as retained raw evidence."""
    boundary = report["evidence_boundary"]
    retention = boundary["retention"]

    assert retention["raw_datasets_retained_in_git"] == 0
    assert retention["raw_traces_retained_in_git"] == 0
    assert retention["raw_prior_predictive_draws_retained"] is False
    assert retention["raw_posterior_predictive_draws_retained"] is False
    assert retention["sampler_backend_trace_attributes_retained"] is False
    assert retention["historical_uv_lock_bytes_retained"] is False
    assert boundary["independent_raw_reverification"] == "blocked"
    assert boundary["ecosystem_promotion_blocked"] is True
    assert boundary["public_support_or_promotion"] == "blocked"
    assert boundary["ordered_slice_identity_independently_authenticated"] is False
    assert boundary["runtime_hssm_import_bound_to_recorded_checkout"] is False
    assert (
        "same historical JEAM producer" in boundary["objective_parity_interpretation"]
    )
    assert "v2a/v2b hypotheses" in boundary["successor_interpretation"]


def test_revision_and_optimizer_labels_prevent_historical_overclaim(report):
    """Historical evidence is not relabeled as a current rerun or an MLE."""
    provenance = report["provenance"]
    boundary = report["evidence_boundary"]

    assert provenance["historical_jeam_revision"] == HISTORICAL_JEAM_REVISION
    assert provenance["current_safety_jeam_revision"] == CURRENT_SAFETY_JEAM_REVISION
    assert provenance["current_safety_revision_rerun"] is False
    assert report["summary"]["optimizer_endpoint_label"] == (
        "fixed-budget DE endpoint (not MLE)"
    )
    assert (
        "fixed-budget differential-evolution endpoints"
        in boundary["optimizer_interpretation"]
    )
    assert (
        "not demonstrated converged optima or MLEs"
        in boundary["optimizer_interpretation"]
    )


def test_runtime_telemetry_is_descriptive_not_a_gate(report):
    """Recorded wall time remains descriptive and machine-scoped."""
    boundary = report["evidence_boundary"]
    scenarios = report["scenario_records"]

    assert boundary["telemetry_interpretation"] == (
        "descriptive recorded-machine observations; never a pass/fail criterion"
    )
    assert all(row["timing_scope"].startswith("descriptive") for row in scenarios)
    assert all(row["sampling_seconds"] > 0 for row in scenarios)
    assert all(row["total_seconds"] > row["sampling_seconds"] for row in scenarios)
    assert not any("seconds" in row["message"] for row in report["failure_records"])


def test_report_loader_calls_verifier_once_and_never_reads_files(
    monkeypatch, verified_documents
):
    """Only the verifier may own frozen-file reads and authentication."""
    calls = []

    def fake_load(root):
        calls.append(root)
        return copy.deepcopy(verified_documents)

    monkeypatch.setattr(
        report_module, "load_verified_psdm_recovery_evidence", fake_load
    )
    loaded = report_module.load_psdm_recovery_report("sentinel-root")

    assert calls == ["sentinel-root"]
    assert loaded["summary"]["unique_truth_count"] == 16

    tree = ast.parse(inspect.getsource(report_module))
    verifier_calls = 0
    forbidden_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            if node.func.id == "load_verified_psdm_recovery_evidence":
                verifier_calls += 1
            if node.func.id == "open":
                forbidden_calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute) and node.func.attr in {
            "open",
            "read_bytes",
            "read_text",
        }:
            forbidden_calls.append(node.func.attr)
    assert verifier_calls == 1
    assert forbidden_calls == []


def test_verifier_and_report_import_no_modeling_or_sampling_stack():
    """Compact verification must stay network-free and model-stack-free."""
    forbidden_imports = {"hssm", "jeam", "pymc", "bambi", "arviz"}
    for module in (verifier_module, report_module):
        tree = ast.parse(inspect.getsource(module))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        assert imports.isdisjoint(forbidden_imports)


def test_semantic_mutation_is_rejected_after_authentication(verified_documents):
    """Derived truth inclusion and stored gates cannot drift independently."""
    _verification, spec, addendum, artifact = verified_documents
    mutated = copy.deepcopy(artifact)
    mutated["scenarios"][0]["parameters"][0]["truth_in_hdi"] = False

    with pytest.raises(EvidenceIntegrityError, match="Stored HDI inclusion changed"):
        verify_psdm_recovery_documents(spec, addendum, mutated)


def test_verifier_cli_normalizes_integrity_errors(tmp_path, capsys):
    """CLI failures are concise and never expose an internal traceback."""
    assert verifier_module.main(["--root", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: Cannot snapshot")
    assert "Traceback" not in captured.err

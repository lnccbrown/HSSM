"""Qualification tests for bounded, reproducible marketing experiments."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from hssm.marketing import (
    ExperimentRequestError,
    available_templates,
    run_experiment,
    validate_request,
)
from hssm.marketing.__main__ import main

if TYPE_CHECKING:
    from pathlib import Path

ENVIRONMENT_DIGEST = "sha256:" + "0" * 64
TEMPLATE_IDS = {
    "same-accuracy-different-mechanism-v1",
    "partial-pooling-shrinkage-v1",
    "addm-attention-contrast-v1",
    "parameter-vs-model-recovery-v1",
    "prior-sensitivity-threshold-v1",
}


def request(template_id: str, *, inputs: dict | None = None) -> dict:
    """Return a valid small Tier-S request for a template."""
    return {
        "schema": "marketing-experiment-request-v1",
        "experiment_id": f"experiment-{template_id}",
        "brief_id": "brief-fixture",
        "template_id": template_id,
        "scientific_question": "What does the predeclared synthetic contrast show?",
        "expected_contrast": "The template-specific contrast is visible.",
        "data_class": "synthetic",
        "dataset_ref": "synthetic:generated",
        "hssm_ref": "test-ref",
        "environment_digest": ENVIRONMENT_DIGEST,
        "inputs": inputs or {},
        "random_seeds": [1729],
        "resource_tier": "S",
        "diagnostic_policy": "tier-s-scientific-v1",
        "expected_outputs": ["summary", "data", "png", "svg", "asset"],
        "visibility": "internal",
        "approval": {"mode": "validated_intake"},
    }


def test_registry_exposes_five_versioned_exact_schemas() -> None:
    """Expose exactly the five approved, versioned Tier-S templates."""
    definitions = available_templates()
    assert {entry["template_id"] for entry in definitions} == TEMPLATE_IDS
    assert all(entry["version"] == "1.0.0" for entry in definitions)
    assert all(entry["resource_tier"] == "S" for entry in definitions)


@pytest.mark.parametrize(
    ("template_id", "inputs"),
    [
        (
            "same-accuracy-different-mechanism-v1",
            {"n_trials": 500, "max_accuracy_gap": 0.1},
        ),
        ("partial-pooling-shrinkage-v1", {"n_groups": 40}),
        ("addm-attention-contrast-v1", {"n_trials": 300}),
        ("parameter-vs-model-recovery-v1", {"n_datasets": 300}),
        ("prior-sensitivity-threshold-v1", {}),
    ],
)
def test_each_registered_template_emits_a_passed_evidence_package(
    tmp_path: Path, template_id: str, inputs: dict
) -> None:
    """Emit the complete passing evidence package for every template."""
    manifest = run_experiment(
        request(template_id, inputs=inputs), tmp_path / template_id
    )
    assert manifest["schema"] == "marketing-experiment-run-v1"
    assert manifest["status"] == "passed"
    assert manifest["result_digest"].startswith("sha256:")
    assert all(entry["claim_use"] == "eligible" for entry in manifest["outputs"])
    output = tmp_path / template_id
    for name in (
        "request.json",
        "summary.json",
        "data.csv",
        "figure.png",
        "figure.svg",
        "asset.json",
        "run.json",
    ):
        assert (output / name).is_file()
    asset = json.loads((output / "asset.json").read_text(encoding="utf-8"))
    svg = (output / "figure.svg").read_text(encoding="utf-8")
    assert "<!DOCTYPE" not in svg
    assert asset["data_class"] == "synthetic"
    assert asset["caption"]
    assert asset["alt_text"]
    assert asset["review_state"] == "proposed"


def test_repeated_request_has_matching_result_and_figure_digests(
    tmp_path: Path,
) -> None:
    """Reproduce byte-identical result and figure digests."""
    value = request("prior-sensitivity-threshold-v1")
    first = run_experiment(value, tmp_path / "first")
    second = run_experiment(value, tmp_path / "second")
    assert first["result_digest"] == second["result_digest"]
    first_outputs = {entry["path"]: entry["digest"] for entry in first["outputs"]}
    second_outputs = {entry["path"]: entry["digest"] for entry in second["outputs"]}
    assert first_outputs == second_outputs


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"template_id": "arbitrary-code-v1"}, "unknown template_id"),
        ({"resource_tier": "F"}, "resource_tier `S` only"),
        ({"data_class": "public"}, "synthetic data only"),
        (
            {"dataset_ref": "synthetic:unregistered"},
            "generated synthetic dataset only",
        ),
        ({"random_seeds": [1, 2]}, "exactly one uint32 seed"),
        ({"approval": {"mode": "prompt_selected"}}, "validated_intake"),
    ],
)
def test_unregistered_authority_fails_closed(change: dict, message: str) -> None:
    """Reject every request that exceeds the fixed execution authority."""
    value = request("prior-sensitivity-threshold-v1")
    value.update(change)
    with pytest.raises(ExperimentRequestError, match=message):
        validate_request(value)


def test_code_command_dataset_and_quota_inputs_fail_closed() -> None:
    """Reject executable, remote-data, and over-quota inputs."""
    for inputs in (
        {"code": "print('no')"},
        {"command": "sh"},
        {"dataset_url": "https://example.test/data"},
        {"n_trials": 50_001},
    ):
        value = request("same-accuracy-different-mechanism-v1", inputs=inputs)
        with pytest.raises(ExperimentRequestError):
            validate_request(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("experiment_id", "../outside"),
        ("experiment_id", "nested/run"),
        ("experiment_id", "."),
        ("brief_id", "brief\\outside"),
        ("brief_id", "x" * 129),
    ],
)
def test_artifact_identifiers_are_portable_path_segments(
    field: str, value: str
) -> None:
    """Identifiers used downstream as storage keys cannot contain path syntax."""
    candidate = request("prior-sensitivity-threshold-v1")
    candidate[field] = value
    with pytest.raises(ExperimentRequestError, match="portable"):
        validate_request(candidate)


def test_maximum_length_experiment_id_produces_portable_derived_ids(
    tmp_path: Path,
) -> None:
    """A valid 128-character id cannot overflow derived run or asset ids."""
    candidate = request("prior-sensitivity-threshold-v1")
    candidate["experiment_id"] = "x" * 128
    manifest = run_experiment(candidate, tmp_path / "maximum-id")
    asset = json.loads((tmp_path / "maximum-id/asset.json").read_text())
    assert len(manifest["run_id"]) == 128
    assert len(asset["asset_id"]) <= 128


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("template_id", ["prior-sensitivity-threshold-v1"]),
        ("visibility", ["internal"]),
        ("environment_digest", [ENVIRONMENT_DIGEST]),
    ],
)
def test_malformed_enum_and_pin_types_raise_contract_errors(field, value) -> None:
    """Malformed input types fail through ExperimentRequestError, not TypeError."""
    candidate = request("prior-sensitivity-threshold-v1")
    candidate[field] = value
    with pytest.raises(ExperimentRequestError):
        validate_request(candidate)


def test_non_finite_numeric_inputs_are_rejected() -> None:
    """NaN, infinity and integers too large for floats are invalid inputs."""
    for value in (float("nan"), float("inf"), float("-inf"), 10**400):
        candidate = request(
            "prior-sensitivity-threshold-v1", inputs={"observed_mean": value}
        )
        with pytest.raises(ExperimentRequestError, match="finite"):
            validate_request(candidate)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("not-json", "Expecting value"),
        (
            json.dumps(
                request("prior-sensitivity-threshold-v1") | {"schema": "unsupported"}
            ),
            "schema must be",
        ),
    ],
)
def test_cli_reports_rejected_requests_without_tracebacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    contents: str,
    message: str,
) -> None:
    """Give callers a stable rejection exit distinct from runner failures."""
    request_path = tmp_path / "request.json"
    request_path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["hssm.marketing", str(request_path), str(tmp_path / "artifacts")],
    )
    with pytest.raises(SystemExit) as error:
        main()
    captured = capsys.readouterr()
    assert error.value.code == 1
    assert captured.out == ""
    assert captured.err.startswith("request rejected: ")
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_failed_diagnostics_block_claim_use_and_asset_review(tmp_path: Path) -> None:
    """Block failed evidence from claims and asset review."""
    value = request(
        "prior-sensitivity-threshold-v1", inputs={"min_probability_gap": 0.9}
    )
    manifest = run_experiment(value, tmp_path / "failed")
    assert manifest["status"] == "failed"
    assert all(entry["claim_use"] == "blocked" for entry in manifest["outputs"])
    asset = json.loads((tmp_path / "failed/asset.json").read_text(encoding="utf-8"))
    assert asset["review_state"] == "rejected"


def test_existing_artifacts_are_never_overwritten(tmp_path: Path) -> None:
    """Preserve every pre-existing output-directory artifact."""
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ExperimentRequestError, match="must be empty"):
        run_experiment(request("prior-sensitivity-threshold-v1"), output)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"

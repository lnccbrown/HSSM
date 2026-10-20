"""Compatibility checks against the exact unmerged HSSMSpine P1 contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from hssm.marketing import run_experiment, validate_request

PIN = Path(__file__).with_name("fixtures") / "spine-p1-contract.json"
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _request() -> dict[str, Any]:
    return {
        "schema": "marketing-experiment-request-v1",
        "experiment_id": "contract-pin-smoke",
        "brief_id": "brief-contract-pin",
        "template_id": "prior-sensitivity-threshold-v1",
        "scientific_question": "Does the evidence package match the P1 contract?",
        "expected_contrast": "The predeclared probability gap is visible.",
        "data_class": "synthetic",
        "dataset_ref": "synthetic:generated",
        "hssm_ref": "p2-contract-test",
        "environment_digest": "sha256:" + "0" * 64,
        "inputs": {},
        "random_seeds": [1729],
        "resource_tier": "S",
        "diagnostic_policy": "tier-s-scientific-v1",
        "expected_outputs": ["summary", "data", "png", "svg", "asset"],
        "visibility": "internal",
        "approval": {"mode": "validated_intake"},
    }


def _assert_contract_shape(
    pin: dict[str, Any], contract_id: str, artifact: dict[str, Any]
) -> None:
    contract = pin["contracts"][contract_id]
    assert artifact["schema"] == contract_id
    assert set(contract["required"]) <= set(artifact)
    assert set(artifact) <= set(contract["allowed"])
    for field, values in contract["enums"].items():
        assert artifact[field] in values


def test_evidence_package_matches_pinned_spine_p1_contract(tmp_path: Path) -> None:
    """Keep P2 requests, run receipts and assets compatible with P1."""
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    source = pin["source"]
    assert pin["schema"] == "hssm-spine-marketing-contract-pin-v1"
    assert source["repository"] == "lnccbrown/HSSMSpine"
    assert GIT_SHA.fullmatch(source["commit"])
    assert SHA256.fullmatch(source["sha256"])

    request = _request()
    validate_request(request)
    _assert_contract_shape(pin, "marketing-experiment-request-v1", request)

    output = tmp_path / "evidence"
    manifest = run_experiment(request, output)
    asset = json.loads((output / "asset.json").read_text(encoding="utf-8"))
    _assert_contract_shape(pin, "marketing-experiment-run-v1", manifest)
    _assert_contract_shape(pin, "marketing-asset-v1", asset)

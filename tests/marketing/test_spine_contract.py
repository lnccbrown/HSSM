"""Compatibility checks against the exact unmerged HSSMSpine P1 contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from hssm.marketing import run_experiment, validate_request

PIN = Path(__file__).with_name("fixtures") / "spine-p1-contract.json"
SMOKE_REQUEST = Path(__file__).with_name("fixtures") / "tier-s-smoke-request.json"
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _request() -> dict[str, Any]:
    return json.loads(SMOKE_REQUEST.read_text(encoding="utf-8"))


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

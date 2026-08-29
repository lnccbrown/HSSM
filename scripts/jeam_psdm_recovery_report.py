"""Build report data from one authenticated fixed-PSDM evidence load."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scripts.verify_jeam_psdm_recovery_evidence import (
    REPO_ROOT,
    load_verified_psdm_recovery_evidence,
)

if TYPE_CHECKING:
    from pathlib import Path

REPORT_KEYS = (
    "summary",
    "parameter_records",
    "scenario_records",
    "aggregate_records",
    "objective_records",
    "predictive_records",
    "failure_records",
    "evidence_boundary",
    "provenance",
)


def load_psdm_recovery_report(
    root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Return stable presentation records from one verifier-owned snapshot."""
    verification, _spec, _addendum, _artifact = load_verified_psdm_recovery_evidence(
        root
    )
    if tuple(key for key in REPORT_KEYS if key not in verification):
        raise ValueError("Verified fixed-PSDM report records are incomplete.")
    return {key: verification[key] for key in REPORT_KEYS}


__all__ = ["REPORT_KEYS", "load_psdm_recovery_report"]

"""Deterministic CI sharding for the hierarchical-TN qualification study.

The frozen manifest remains the only source of cells.  This module groups the
candidate/control timing comparisons into indivisible execution units, packs
those units into a bounded GitHub Actions matrix, and can execute one canonical
shard without accepting a caller-supplied cell list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Direct ``python scripts/...py`` execution otherwise exposes only ``scripts/``
# itself on ``sys.path``, not the repository root that owns the namespace package.
if not __package__:  # pragma: no cover - covered by the CLI subprocess test
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.truncated_hierarchy_qualification import (
    DEFAULT_MANIFEST,
    QualificationError,
    collect_environment,
    environment_sha256,
    expand_plan,
    load_environment_catalog,
    load_manifest,
    manifest_sha256,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MATRIX_JOB_LIMIT = 255
TIER_CELL_LIMITS = {
    "smoke": 2,
    "qualification": 4,
    "stress": 2,
}
TIER_RETENTION_DAYS = {
    "smoke": 14,
    "qualification": 90,
    "stress": 30,
}
TIER_TIMEOUT_SECONDS = {
    "smoke": 1_200,
    "qualification": 2_400,
    "stress": 2_400,
}


@dataclass(frozen=True, slots=True)
class ExecutionUnit:
    """One indivisible sequence of cells that must share a worker."""

    dependency_profile: str
    cell_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Shard:
    """One canonical GitHub Actions matrix entry."""

    shard_id: str
    tier: str
    dependency_profile: str
    cell_ids: tuple[str, ...]
    cell_count: int
    retention_days: int
    timeout_seconds: int


def _dependency_profile(plan_entry: Mapping[str, Any]) -> str:
    profile = plan_entry["scenario"].get("dependency_profile")
    return str(profile or "current-resolved")


def _requires_pair(plan_entry: Mapping[str, Any]) -> bool:
    scenario = plan_entry["scenario"]
    return bool(
        scenario["tier"] == "qualification"
        and scenario["purpose"] in {"candidate", "control"}
        and scenario.get("calibration_kind") != "sbc"
    )


def build_execution_units(
    manifest: Mapping[str, Any], tier: str
) -> tuple[ExecutionUnit, ...]:
    """Group canonical plan cells, preserving paired timing comparisons."""
    plan = expand_plan(manifest, tier)
    by_key = {
        (str(entry["scenario_id"]), int(entry["replicate"])): entry for entry in plan
    }
    consumed: set[str] = set()
    units: list[ExecutionUnit] = []

    for entry in plan:
        cell_id = str(entry["cell_id"])
        if cell_id in consumed:
            continue
        scenario = entry["scenario"]
        control_id = scenario.get("control_id")
        is_paired_candidate = (
            tier == "qualification"
            and scenario.get("calibration_kind") is None
            and scenario.get("purpose") == "candidate"
            and isinstance(control_id, str)
        )
        if not is_paired_candidate:
            units.append(
                ExecutionUnit(
                    dependency_profile=_dependency_profile(entry),
                    cell_ids=(cell_id,),
                )
            )
            consumed.add(cell_id)
            continue

        replicate = int(entry["replicate"])
        control = by_key.get((control_id, replicate))
        if control is None:
            raise QualificationError(
                f"paired candidate {cell_id} lacks control {control_id!r}"
            )
        control_cell_id = str(control["cell_id"])
        if control_cell_id in consumed:
            raise QualificationError(
                f"paired control {control_cell_id} was assigned more than once"
            )
        candidate_profile = _dependency_profile(entry)
        if _dependency_profile(control) != candidate_profile:
            raise QualificationError(
                f"paired cells {cell_id} and {control_cell_id} use different profiles"
            )
        ordered = (
            (cell_id, control_cell_id)
            if replicate % 2 == 0
            else (control_cell_id, cell_id)
        )
        units.append(
            ExecutionUnit(
                dependency_profile=candidate_profile,
                cell_ids=ordered,
            )
        )
        consumed.update(ordered)

    expected = {str(entry["cell_id"]) for entry in plan}
    if consumed != expected:
        missing = sorted(expected - consumed)
        unexpected = sorted(consumed - expected)
        raise QualificationError(
            f"execution units do not cover the canonical plan: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return tuple(units)


def build_shards(manifest: Mapping[str, Any], tier: str) -> tuple[Shard, ...]:
    """Pack canonical execution units without splitting a paired comparison."""
    try:
        cell_limit = TIER_CELL_LIMITS[tier]
        retention_days = TIER_RETENTION_DAYS[tier]
        timeout_seconds = TIER_TIMEOUT_SECONDS[tier]
    except KeyError as error:
        raise QualificationError(f"unknown tier: {tier}") from error

    units = build_execution_units(manifest, tier)
    shards: list[Shard] = []
    profiles = tuple(manifest["dependency_profiles"])
    unknown_profiles = sorted(
        {unit.dependency_profile for unit in units} - set(profiles)
    )
    if unknown_profiles:
        raise QualificationError(
            f"execution units use unknown dependency profiles: {unknown_profiles}"
        )

    for profile in profiles:
        profile_units = [unit for unit in units if unit.dependency_profile == profile]
        packed: list[list[str]] = []
        current: list[str] = []
        for unit in profile_units:
            if len(unit.cell_ids) > cell_limit:
                raise QualificationError(
                    f"execution unit exceeds the {tier} cell limit: {unit.cell_ids}"
                )
            if current and len(current) + len(unit.cell_ids) > cell_limit:
                packed.append(current)
                current = []
            current.extend(unit.cell_ids)
        if current:
            packed.append(current)
        for profile_index, cell_ids in enumerate(packed):
            shards.append(
                Shard(
                    shard_id=f"{tier}-{profile}-{profile_index:03d}",
                    tier=tier,
                    dependency_profile=profile,
                    cell_ids=tuple(cell_ids),
                    cell_count=len(cell_ids),
                    retention_days=retention_days,
                    timeout_seconds=timeout_seconds,
                )
            )

    if len(shards) > MATRIX_JOB_LIMIT:
        raise QualificationError(
            f"{tier} requires {len(shards)} matrix jobs; limit is {MATRIX_JOB_LIMIT}"
        )
    flattened = [cell_id for shard in shards for cell_id in shard.cell_ids]
    plan_ids = [str(entry["cell_id"]) for entry in expand_plan(manifest, tier)]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(plan_ids):
        raise QualificationError(f"{tier} shards duplicate or omit canonical cells")
    return tuple(shards)


def matrix_payload(manifest: Mapping[str, Any], tier: str) -> dict[str, Any]:
    """Return the compact dynamic-matrix object consumed by GitHub Actions."""
    return {
        "include": [
            {
                **asdict(shard),
                # Cell membership is included for auditability, but execution looks
                # the shard up again by ID and never trusts this workflow value.
                "cell_ids": list(shard.cell_ids),
            }
            for shard in build_shards(manifest, tier)
        ]
    }


def select_shard(manifest: Mapping[str, Any], *, tier: str, shard_id: str) -> Shard:
    """Resolve exactly one shard from its manifest-derived identifier."""
    matches = [
        shard for shard in build_shards(manifest, tier) if shard.shard_id == shard_id
    ]
    if len(matches) != 1:
        raise QualificationError(
            f"shard {shard_id!r} does not identify one canonical {tier} shard"
        )
    return matches[0]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{json.dumps(value, allow_nan=False, indent=2, sort_keys=True)}\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_environment_sidecar(
    manifest: Mapping[str, Any], dependency_profile: str, output: Path
) -> Mapping[str, Any]:
    """Collect and atomically publish one canonical dependency sidecar."""
    environment = collect_environment(manifest, dependency_profile)
    _atomic_json(output, environment)
    return environment


def execute_shard(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    tier: str,
    shard_id: str,
    artifact_root: Path,
    environment_paths: Sequence[Path],
) -> dict[str, Any]:
    """Attempt every cell in one shard and publish a deterministic shard report."""
    # Importing the sampler is intentionally deferred so matrix-only tests and the
    # prepare job never initialize PyMC/JAX model machinery.
    from scripts.truncated_hierarchy_runner import orchestrate_cell, orchestrate_pair

    shard = select_shard(manifest, tier=tier, shard_id=shard_id)
    catalog = load_environment_catalog(environment_paths, manifest)
    plan_by_id = {str(entry["cell_id"]): entry for entry in expand_plan(manifest, tier)}
    outcomes: list[dict[str, Any]] = []
    index = 0
    while index < len(shard.cell_ids):
        cell_id = shard.cell_ids[index]
        entry = plan_by_id[cell_id]
        if _requires_pair(entry):
            paired_cell_ids = shard.cell_ids[index : index + 2]
            if len(paired_cell_ids) != 2 or not all(
                _requires_pair(plan_by_id[paired_id]) for paired_id in paired_cell_ids
            ):
                raise QualificationError(
                    f"paired cell {cell_id} is not adjacent to its shard partner"
                )
            try:
                paired_results = orchestrate_pair(
                    entry,
                    manifest,
                    manifest_path,
                    environment_paths,
                    catalog,
                    artifact_root,
                    timeout_seconds=shard.timeout_seconds,
                )
                results_by_id = {
                    str(result["cell_id"]): result for result in paired_results
                }
                for paired_id in paired_cell_ids:
                    result = results_by_id[paired_id]
                    outcomes.append(
                        {
                            "cell_id": paired_id,
                            "execution_status": result["execution_status"],
                            "orchestration_error": None,
                        }
                    )
            except Exception as error:
                # The pair parent owns the opaque linked identities and attempts
                # both cells. If it cannot safely publish either marker, only the
                # later aggregate can create provenance-valid missing rows.
                for paired_id in paired_cell_ids:
                    outcomes.append(
                        {
                            "cell_id": paired_id,
                            "execution_status": "missing",
                            "orchestration_error": f"{type(error).__name__}: {error}",
                        }
                    )
            index += 2
            continue
        try:
            result = orchestrate_cell(
                entry,
                manifest,
                manifest_path,
                environment_paths,
                catalog,
                artifact_root,
                timeout_seconds=shard.timeout_seconds,
            )
            outcomes.append(
                {
                    "cell_id": cell_id,
                    "execution_status": result["execution_status"],
                    "orchestration_error": None,
                }
            )
        except Exception as error:
            # The runner normally contains crashes and timeouts. An exception here
            # means its opaque attempt identity is unavailable to this outer layer,
            # so aggregation must materialize the explicit missing row.
            outcomes.append(
                {
                    "cell_id": cell_id,
                    "execution_status": "missing",
                    "orchestration_error": f"{type(error).__name__}: {error}",
                }
            )
        index += 1

    report = {
        "schema_version": 1,
        "study_id": manifest["study_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "shard_id": shard.shard_id,
        "tier": shard.tier,
        "dependency_profile": shard.dependency_profile,
        "cell_ids": list(shard.cell_ids),
        "outcomes": outcomes,
    }
    _atomic_json(artifact_root / "shards" / f"{shard.shard_id}.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)

    matrix = commands.add_parser("matrix", help="emit one canonical CI matrix")
    matrix.add_argument("--tier", choices=tuple(TIER_CELL_LIMITS), required=True)

    environment = commands.add_parser(
        "environment", help="collect one dependency-profile environment sidecar"
    )
    environment.add_argument("--dependency-profile", required=True)
    environment.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run", help="execute one manifest-derived shard")
    run.add_argument("--tier", choices=tuple(TIER_CELL_LIMITS), required=True)
    run.add_argument("--shard-id", required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--environment", type=Path, action="append", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic sharding CLI."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "matrix":
            print(
                json.dumps(
                    matrix_payload(manifest, args.tier),
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "environment":
            environment = write_environment_sidecar(
                manifest, args.dependency_profile, args.output
            )
            print(environment_sha256(environment, manifest))
            return 0
        report = execute_shard(
            manifest,
            manifest_path=args.manifest,
            tier=args.tier,
            shard_id=args.shard_id,
            artifact_root=args.artifact_root,
            environment_paths=args.environment,
        )
        statuses = {outcome["execution_status"] for outcome in report["outcomes"]}
        print(json.dumps(report, allow_nan=False, sort_keys=True))
        return 0 if statuses == {"completed"} else 1
    except QualificationError as error:
        parser.exit(2, f"sharding contract error: {error}\n")


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())

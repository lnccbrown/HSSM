"""Contract tests for the no-sampling qualification sharder and workflow."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from scripts.truncated_hierarchy_qualification import (
    DEFAULT_MANIFEST,
    QualificationError,
    expand_plan,
    load_manifest,
)
from scripts.truncated_hierarchy_shards import (
    MATRIX_JOB_LIMIT,
    build_execution_units,
    build_shards,
    execute_shard,
    matrix_payload,
    select_shard,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github/workflows/qualify_truncated_hierarchy.yml"
)


@pytest.fixture(scope="module")
def manifest() -> Mapping[str, Any]:
    """Load the real frozen manifest once for all sharding assertions."""
    return load_manifest(DEFAULT_MANIFEST)


@pytest.fixture(scope="module")
def tier_shards(manifest: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    """Build each potentially large matrix once."""
    return {
        tier: build_shards(manifest, tier)
        for tier in ("smoke", "qualification", "stress")
    }


@pytest.mark.parametrize(
    ("tier", "expected_cells", "expected_shards"),
    (
        ("smoke", 10, 6),
        ("qualification", 720, 181),
        ("stress", 18, 9),
    ),
)
def test_shards_cover_every_cell_once_below_matrix_cap(
    manifest: Mapping[str, Any],
    tier_shards: Mapping[str, tuple[Any, ...]],
    tier: str,
    expected_cells: int,
    expected_shards: int,
) -> None:
    """Every frozen cell appears exactly once in a sub-256-job matrix."""
    shards = tier_shards[tier]
    planned = [entry["cell_id"] for entry in expand_plan(manifest, tier)]
    assigned = [cell_id for shard in shards for cell_id in shard.cell_ids]

    assert len(shards) == expected_shards
    assert len(shards) <= MATRIX_JOB_LIMIT
    assert len(shards) < 256
    assert len(assigned) == expected_cells
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == set(planned)
    assert all(shard.cell_count == len(shard.cell_ids) for shard in shards)


def test_shard_membership_is_deterministic(
    manifest: Mapping[str, Any], tier_shards: Mapping[str, tuple[Any, ...]]
) -> None:
    """Repeated planning produces byte-equivalent matrix data."""
    first = matrix_payload(manifest, "qualification")
    second = matrix_payload(manifest, "qualification")

    assert first == second
    assert first["include"] == [
        {
            "shard_id": shard.shard_id,
            "tier": shard.tier,
            "dependency_profile": shard.dependency_profile,
            "cell_ids": list(shard.cell_ids),
            "cell_count": shard.cell_count,
            "retention_days": shard.retention_days,
            "timeout_seconds": shard.timeout_seconds,
        }
        for shard in tier_shards["qualification"]
    ]


def test_shards_never_mix_dependency_profiles(
    manifest: Mapping[str, Any], tier_shards: Mapping[str, tuple[Any, ...]]
) -> None:
    """Each worker installs exactly the dependency profile of all its cells."""
    for tier, shards in tier_shards.items():
        plan_by_id = {entry["cell_id"]: entry for entry in expand_plan(manifest, tier)}
        for shard in shards:
            profiles = {
                plan_by_id[cell_id]["scenario"].get(
                    "dependency_profile", "current-resolved"
                )
                for cell_id in shard.cell_ids
            }
            assert profiles == {shard.dependency_profile}


def test_qualification_pairs_share_worker_and_alternate_order(
    manifest: Mapping[str, Any], tier_shards: Mapping[str, tuple[Any, ...]]
) -> None:
    """Each paired replicate is adjacent, with the frozen parity ordering."""
    shards = tier_shards["qualification"]
    position = {
        cell_id: (shard.shard_id, index)
        for shard in shards
        for index, cell_id in enumerate(shard.cell_ids)
    }
    paired_candidates = [
        entry
        for entry in expand_plan(manifest, "qualification")
        if entry["scenario"].get("purpose") == "candidate"
        and entry["scenario"].get("calibration_kind") is None
        and entry["scenario"].get("control_id") is not None
    ]

    assert len(paired_candidates) == 85
    for candidate in paired_candidates:
        replicate = candidate["replicate"]
        control_cell = (
            f"{candidate['scenario']['control_id']}--replicate-{replicate:02d}"
        )
        candidate_position = position[candidate["cell_id"]]
        control_position = position[control_cell]
        assert candidate_position[0] == control_position[0]
        if replicate % 2 == 0:
            assert candidate_position[1] + 1 == control_position[1]
        else:
            assert control_position[1] + 1 == candidate_position[1]


def test_execution_units_are_singletons_outside_primary_pairs(
    manifest: Mapping[str, Any],
) -> None:
    """Only primary candidate/control comparisons form multi-cell units."""
    assert all(
        len(unit.cell_ids) == 1 for unit in build_execution_units(manifest, "smoke")
    )
    assert all(
        len(unit.cell_ids) == 1 for unit in build_execution_units(manifest, "stress")
    )
    qualification_sizes = [
        len(unit.cell_ids) for unit in build_execution_units(manifest, "qualification")
    ]
    assert qualification_sizes.count(2) == 85
    assert qualification_sizes.count(1) == 550


def test_select_shard_rejects_noncanonical_identifier(
    manifest: Mapping[str, Any],
) -> None:
    """Workers cannot inject an arbitrary caller-supplied cell list."""
    with pytest.raises(QualificationError, match="does not identify one canonical"):
        select_shard(
            manifest,
            tier="smoke",
            shard_id="smoke-current-resolved-user-supplied",
        )


def test_direct_matrix_cli_emits_canonical_json() -> None:
    """The exact direct-script form used by Actions resolves the repo namespace."""
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/truncated_hierarchy_shards.py",
            "--manifest",
            str(DEFAULT_MANIFEST),
            "matrix",
            "--tier",
            "smoke",
        ],
        cwd=WORKFLOW.parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert len(payload["include"]) == 6
    assert sum(entry["cell_count"] for entry in payload["include"]) == 10


def test_execute_shard_continues_after_failure_and_writes_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    """A failed first pair member cannot prevent later members from being attempted."""
    shard = build_shards(manifest, "qualification")[0]
    attempted: list[str] = []
    pair_calls = 0

    def orchestrate_cell(
        entry: Mapping[str, Any], *_: Any, **__: Any
    ) -> dict[str, str]:
        raise AssertionError(
            f"paired shard called unpaired runner for {entry['cell_id']}"
        )

    def orchestrate_pair(
        _: Mapping[str, Any], *args: Any, **kwargs: Any
    ) -> tuple[dict[str, str], dict[str, str]]:
        del args, kwargs
        nonlocal pair_calls
        pair_ids = shard.cell_ids[pair_calls * 2 : pair_calls * 2 + 2]
        attempted.extend(pair_ids)
        statuses = (
            ("failed", "completed")
            if pair_calls == 0
            else (
                "completed",
                "completed",
            )
        )
        pair_calls += 1
        assert len(pair_ids) == 2
        return (
            {"cell_id": pair_ids[0], "execution_status": statuses[0]},
            {"cell_id": pair_ids[1], "execution_status": statuses[1]},
        )

    fake_runner = SimpleNamespace(
        orchestrate_cell=orchestrate_cell,
        orchestrate_pair=orchestrate_pair,
    )
    monkeypatch.setitem(sys.modules, "scripts.truncated_hierarchy_runner", fake_runner)
    monkeypatch.setattr(
        "scripts.truncated_hierarchy_shards.load_environment_catalog",
        lambda *_: {},
    )

    report = execute_shard(
        manifest,
        manifest_path=DEFAULT_MANIFEST,
        tier="qualification",
        shard_id=shard.shard_id,
        artifact_root=tmp_path,
        environment_paths=[tmp_path / "environment.json"],
    )

    assert attempted == list(shard.cell_ids)
    assert [outcome["execution_status"] for outcome in report["outcomes"]] == [
        "failed",
        *(["completed"] * (len(shard.cell_ids) - 1)),
    ]
    report_path = tmp_path / "shards" / f"{shard.shard_id}.json"
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_execute_shard_records_uncontained_cell_as_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    """An outer runner error is explicit and does not suppress the next cell."""
    shard = build_shards(manifest, "smoke")[0]
    attempted: list[str] = []

    def orchestrate_cell(
        entry: Mapping[str, Any], *_: Any, **__: Any
    ) -> dict[str, str]:
        cell_id = str(entry["cell_id"])
        attempted.append(cell_id)
        if cell_id == shard.cell_ids[0]:
            raise RuntimeError("lost opaque runner identity")
        return {"cell_id": cell_id, "execution_status": "completed"}

    fake_runner = SimpleNamespace(
        orchestrate_cell=orchestrate_cell,
        orchestrate_pair=lambda *_args, **_kwargs: pytest.fail(
            "smoke cells must not use pair execution"
        ),
    )
    monkeypatch.setitem(sys.modules, "scripts.truncated_hierarchy_runner", fake_runner)
    monkeypatch.setattr(
        "scripts.truncated_hierarchy_shards.load_environment_catalog",
        lambda *_: {},
    )

    report = execute_shard(
        manifest,
        manifest_path=DEFAULT_MANIFEST,
        tier="smoke",
        shard_id=shard.shard_id,
        artifact_root=tmp_path,
        environment_paths=[tmp_path / "environment.json"],
    )

    assert attempted == list(shard.cell_ids)
    assert [outcome["execution_status"] for outcome in report["outcomes"]] == [
        "missing",
        "completed",
    ]
    assert "lost opaque runner identity" in report["outcomes"][0]["orchestration_error"]


def _load_workflow() -> tuple[str, Mapping[str, Any]]:
    text = WORKFLOW.read_text(encoding="utf-8")
    loaded = yaml.load(text, Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    return text, loaded


def test_workflow_is_opt_in_same_repo_and_read_only() -> None:
    """The expensive study cannot become a normal or privileged PR check."""
    text, workflow = _load_workflow()
    triggers = workflow["on"]

    assert set(triggers) == {"pull_request", "workflow_dispatch"}
    assert triggers["pull_request"] == {"types": ["labeled"]}
    assert "pull_request_target" not in text
    assert not re.search(r"^\s*push\s*:", text, flags=re.MULTILINE)
    assert workflow["permissions"] == {"contents": "read"}
    prepare_condition = workflow["jobs"]["prepare"]["if"]
    assert "run-truncated-hierarchy-qualification" in prepare_condition
    assert "head.repo.full_name == github.repository" in prepare_condition


def test_workflow_checks_out_exact_head_and_pins_every_remote_action() -> None:
    """Every job executes immutable code and immutable third-party actions."""
    text, workflow = _load_workflow()

    assert "github.event.pull_request.head.sha" in workflow["env"]["QUALIFICATION_SHA"]
    assert text.count("ref: ${{ env.QUALIFICATION_SHA }}") == len(workflow["jobs"])
    remote_uses = re.findall(r"^\s*uses:\s*([^\s#]+)", text, flags=re.MULTILINE)
    assert remote_uses
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", value) for value in remote_uses)
    assert "persist-credentials: false" in text


def test_workflow_freezes_runner_and_matrix_contract() -> None:
    """The workflow uses the declared OS, Python, uv, and bounded matrices."""
    text, workflow = _load_workflow()

    assert set(job["runs-on"] for job in workflow["jobs"].values()) == {"ubuntu-24.04"}
    assert workflow["env"]["UV_VERSION"] == "0.11.21"
    assert text.count('python-version: "3.12"') == len(workflow["jobs"])
    for job_name in ("smoke", "qualification", "stress"):
        strategy = workflow["jobs"][job_name]["strategy"]
        assert strategy["fail-fast"] == "false"
        assert strategy["max-parallel"] == "16"
        assert "fromJSON(needs.prepare.outputs." in strategy["matrix"]
    assert workflow["jobs"]["environments"]["strategy"]["matrix"] == {
        "dependency_profile": ["current-resolved", "bambi-0.19"]
    }


def test_workflow_smoke_gates_later_tiers_but_stress_cannot_rescue() -> None:
    """Primary qualification and diagnostics depend on smoke, not on each other."""
    _, workflow = _load_workflow()
    jobs = workflow["jobs"]

    assert "needs.smoke_aggregate.result == 'success'" in jobs["qualification"]["if"]
    assert "needs.smoke_aggregate.result == 'success'" in jobs["stress"]["if"]
    assert "qualification" not in jobs["stress"]["needs"]
    assert "stress" not in jobs["qualification_aggregate"]["needs"]
    stress_steps = jobs["stress"]["steps"]
    stress_run = next(
        step for step in stress_steps if step["name"] == "Run canonical stress shard"
    )
    assert stress_run["continue-on-error"] == "true"
    stress_assess = next(
        step
        for step in jobs["stress_aggregate"]["steps"]
        if step["name"] == "Aggregate and assess stress evidence"
    )
    assert stress_assess["continue-on-error"] == "true"


@pytest.mark.parametrize(
    "job_name", ("smoke_aggregate", "qualification_aggregate", "stress_aggregate")
)
def test_workflow_aggregation_always_runs_and_uploads(
    job_name: str,
) -> None:
    """Started tiers retain aggregate or failure evidence even after shard errors."""
    text, workflow = _load_workflow()
    job = workflow["jobs"][job_name]

    assert "always()" in job["if"]
    upload_steps = [
        step for step in job["steps"] if "upload-artifact@" in step.get("uses", "")
    ]
    assert len(upload_steps) == 1
    assert upload_steps[0]["if"] == "always()"
    assert "--artifact-root" in text

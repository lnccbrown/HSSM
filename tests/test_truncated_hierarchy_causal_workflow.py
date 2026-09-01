"""No-sampling contract tests for the isolated causal-study workflow."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from scripts.truncated_hierarchy_causal_contract import (
    BACKEND_IDS,
    DEFAULT_MANIFEST,
    REPRESENTATION_IDS,
    build_plan,
    load_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github/workflows/qualify_truncated_hierarchy_causal.yml"
TIERS = ("smoke", "confirmation")
EXPECTED_PAIRS = {"smoke": 2, "confirmation": 16}
EXPECTED_CELLS = {"smoke": 20, "confirmation": 160}


@pytest.fixture(scope="module")
def manifest() -> Mapping[str, Any]:
    """Load and validate the prospective v3 manifest once."""
    return load_manifest(DEFAULT_MANIFEST)


@pytest.fixture(scope="module")
def plans(manifest: Mapping[str, Any]) -> dict[str, Sequence[Any]]:
    """Expand only the deterministic no-sampling plans."""
    return {tier: build_plan(manifest, tier) for tier in TIERS}


def _load_workflow() -> tuple[str, Mapping[str, Any]]:
    text = WORKFLOW.read_text(encoding="utf-8")
    loaded = yaml.load(text, Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    return text, loaded


def _run_for(job: Mapping[str, Any], step_name: str) -> str:
    for step in job["steps"]:
        if step.get("name") == step_name:
            return str(step["run"])
    raise AssertionError(f"missing workflow step {step_name!r}")


@pytest.mark.parametrize("tier", TIERS)
def test_contract_pairs_are_complete_counterbalanced_ten_cell_units(
    manifest: Mapping[str, Any], plans: Mapping[str, Sequence[Any]], tier: str
) -> None:
    """Each worker owns both five-form backends in the frozen paired order."""
    units_by_pair: dict[str, list[Any]] = defaultdict(list)
    for unit in plans[tier]:
        units_by_pair[unit.pair_id].append(unit)

    assert len(plans[tier]) == EXPECTED_CELLS[tier]
    assert len(units_by_pair) == EXPECTED_PAIRS[tier]
    declared_backends = tuple(backend["backend_id"] for backend in manifest["backends"])
    for pair_id, units in units_by_pair.items():
        assert len(units) == 2 * len(REPRESENTATION_IDS) == 10
        assert [unit.pair_position for unit in units] == list(range(10))
        assert {unit.pair_id for unit in units} == {pair_id}
        assert len({unit.regime_id for unit in units}) == 1
        assert len({unit.replicate for unit in units}) == 1
        assert {unit.backend_id for unit in units} == set(BACKEND_IDS)
        assert {(unit.backend_id, unit.representation_id) for unit in units} == {
            (backend_id, representation_id)
            for backend_id in BACKEND_IDS
            for representation_id in REPRESENTATION_IDS
        }
        assert len({unit.block_id for unit in units}) == 2
        assert len({unit.data_id for unit in units}) == 1
        assert len({unit.start_id for unit in units}) == 1
        for backend_id in BACKEND_IDS:
            block = [unit for unit in units if unit.backend_id == backend_id]
            assert [unit.block_position for unit in block] == list(range(5))
            assert {unit.representation_id for unit in block} == set(REPRESENTATION_IDS)
            assert len({unit.block_id for unit in block}) == 1
        regime_index = next(
            index
            for index, regime in enumerate(manifest["regimes"])
            if regime["regime_id"] == units[0].regime_id
        )
        backend_shift = (units[0].replicate + regime_index) % len(BACKEND_IDS)
        expected_order = (
            declared_backends[backend_shift:] + declared_backends[:backend_shift]
        )
        assert (units[0].backend_id, units[5].backend_id) == expected_order


@pytest.mark.parametrize(("tier", "expected"), EXPECTED_PAIRS.items())
def test_direct_matrix_cli_matches_workflow_cardinality(
    tier: str, expected: int
) -> None:
    """The direct command embedded in Actions emits the frozen pair matrix."""
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/truncated_hierarchy_causal_contract.py",
            "--manifest",
            str(DEFAULT_MANIFEST),
            "matrix",
            "--tier",
            tier,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert set(payload) == {"include"}
    assert len(payload["include"]) == expected
    assert all(
        set(pair) == {"pair_id", "tier", "regime_id", "replicate"}
        for pair in payload["include"]
    )
    assert all(pair["tier"] == tier for pair in payload["include"])


def test_workflow_is_opt_in_same_repo_and_read_only() -> None:
    """Untrusted forks and ordinary PR activity cannot start the causal study."""
    text, workflow = _load_workflow()
    triggers = workflow["on"]

    assert set(triggers) == {"pull_request", "workflow_dispatch"}
    assert triggers["pull_request"] == {"types": ["labeled"]}
    assert triggers["workflow_dispatch"]["inputs"]["mode"]["options"] == [
        "smoke",
        "confirmation",
    ]
    assert "pull_request_target" not in text
    assert not re.search(r"^\s*push\s*:", text, flags=re.MULTILINE)
    assert workflow["permissions"] == {"contents": "read"}
    prepare_condition = workflow["jobs"]["prepare"]["if"]
    assert "run-truncated-hierarchy-causal-smoke" in prepare_condition
    assert "run-truncated-hierarchy-causal-confirmation" in prepare_condition
    assert "head.repo.full_name == github.repository" in prepare_condition
    plan_step = next(
        step
        for step in workflow["jobs"]["prepare"]["steps"]
        if step.get("id") == "plan"
    )
    assert (
        "run-truncated-hierarchy-causal-confirmation"
        in plan_step["env"]["REQUESTED_MODE"]
    )


def test_workflow_checks_out_exact_head_and_pins_remote_actions() -> None:
    """Each job runs the selected immutable commit with pinned third parties."""
    text, workflow = _load_workflow()
    jobs = workflow["jobs"]

    assert "github.event.pull_request.head.sha" in workflow["env"]["CAUSAL_SHA"]
    assert text.count("ref: ${{ env.CAUSAL_SHA }}") == len(jobs)
    assert text.count("persist-credentials: false") == len(jobs)
    remote_uses = re.findall(r"^\s*uses:\s*([^\s#]+)", text, flags=re.MULTILINE)
    assert remote_uses
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", value) for value in remote_uses)
    for job in jobs.values():
        checkout = job["steps"][0]
        assert checkout["uses"].startswith("actions/checkout@")
        assert checkout["with"] == {
            "ref": "${{ env.CAUSAL_SHA }}",
            "persist-credentials": "false",
        }


def test_workflow_uses_one_frozen_profile_and_is_separate_from_v2() -> None:
    """The v3 execution path cannot drift into old harnesses or unlocked installs."""
    text, workflow = _load_workflow()

    assert workflow["env"]["CAUSAL_MANIFEST"].endswith(
        "truncated_hierarchy_causal_v3.json"
    )
    assert workflow["env"]["PROFILE_PROJECT"].endswith("current-resolved")
    assert workflow["env"]["UV_VERSION"] == "0.11.21"
    assert "truncated_hierarchy_v2" not in text
    assert "truncated_hierarchy_shards.py" not in text
    assert "truncated_hierarchy_qualification.py" not in text
    assert "bambi-0.19" not in text
    assert "uv lock" not in text
    assert "pip install" not in text
    assert text.count('python-version: "3.12"') == len(workflow["jobs"])
    assert all(job["runs-on"] == "ubuntu-24.04" for job in workflow["jobs"].values())
    assert all(
        "--frozen" in step["run"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "run" in step and "python" in step["run"]
    )


def test_workflow_topology_enforces_smoke_before_confirmation() -> None:
    """Only complete, contract-valid smoke evidence unlocks confirmation."""
    _, workflow = _load_workflow()
    jobs = workflow["jobs"]

    assert set(jobs) == {
        "prepare",
        "materialize",
        "smoke",
        "smoke_aggregate",
        "confirmation",
        "confirmation_aggregate",
    }
    assert jobs["materialize"]["needs"] == "prepare"
    assert jobs["smoke"]["needs"] == ["prepare", "materialize"]
    assert jobs["smoke_aggregate"]["needs"] == [
        "prepare",
        "materialize",
        "smoke",
    ]
    assert jobs["confirmation"]["needs"] == [
        "prepare",
        "materialize",
        "smoke_aggregate",
    ]
    assert "needs.smoke_aggregate.result == 'success'" in jobs["confirmation"]["if"]
    assert "run_confirmation == 'true'" in jobs["confirmation"]["if"]
    smoke_assessment = _run_for(
        jobs["smoke_aggregate"],
        "Merge exact evidence and assess smoke completeness",
    )
    assert 'value.get("contract_valid") is True' in smoke_assessment
    assert 'value.get("evidence_complete") is True' in smoke_assessment
    assert 'value.get("proceed_to_confirmation") is True' in smoke_assessment
    assert 'value.get("outcome")' not in smoke_assessment


def test_workflow_consumes_exact_contract_matrices() -> None:
    """Actions schedules the two/16 canonical pairs, never caller-supplied cells."""
    _, workflow = _load_workflow()
    jobs = workflow["jobs"]
    prepare = _run_for(
        jobs["prepare"], "Validate contract and build exact backend-pair matrices"
    )

    assert "matrix --tier smoke" in prepare
    assert "matrix --tier confirmation" in prepare
    assert "values == (2, 16)" in prepare
    assert jobs["smoke"]["strategy"] == {
        "fail-fast": "false",
        "max-parallel": "4",
        "matrix": "${{ fromJSON(needs.prepare.outputs.smoke_matrix) }}",
    }
    assert jobs["confirmation"]["strategy"] == {
        "fail-fast": "false",
        "max-parallel": "4",
        "matrix": "${{ fromJSON(needs.prepare.outputs.confirmation_matrix) }}",
    }


@pytest.mark.parametrize(
    ("job_name", "tier"), (("smoke", "smoke"), ("confirmation", "confirmation"))
)
def test_each_worker_runs_one_complete_pair_and_preserves_scientific_failures(
    job_name: str, tier: str
) -> None:
    """One runner call owns ten paired cells; scientific exit one remains evidence."""
    _, workflow = _load_workflow()
    job = workflow["jobs"][job_name]
    step_name = f"Run one co-located ten-cell {tier} backend pair"
    step = next(item for item in job["steps"] if item.get("name") == step_name)
    script = step["run"]

    assert script.count("run-unit") == 1
    assert f"--tier {tier}" in script
    assert '--pair-id "$PAIR_ID"' in script
    assert "--block-id" not in script
    assert '--run-dir "$run_root"' in script
    assert '--worker-identity "$WORKER_IDENTITY"' in script
    assert '--expected-git-commit "$CAUSAL_SHA"' in script
    assert "runner_status=${PIPESTATUS[0]}" in script
    assert "0|1)" in script
    assert '*) exit "$runner_status"' in script
    assert step["env"]["PAIR_ID"] == "${{ matrix.pair_id }}"
    assert step["env"]["WORKER_IDENTITY"].endswith(":${{ matrix.pair_id }}")


def test_materialization_and_aggregation_use_finalized_cli() -> None:
    """A single job creates shared inputs and both aggregators use the run tree."""
    _, workflow = _load_workflow()
    jobs = workflow["jobs"]
    materialize = _run_for(
        jobs["materialize"], "Freeze environment and materialize shared inputs once"
    )

    assert materialize.count("environment/environment.json") == 1
    assert materialize.count("materialize-inputs") == 2
    assert "materialize-inputs --tier smoke" in materialize
    assert "--tier confirmation --run-dir" in materialize
    for tier, job_name, step_name in (
        (
            "smoke",
            "smoke_aggregate",
            "Merge exact evidence and assess smoke completeness",
        ),
        (
            "confirmation",
            "confirmation_aggregate",
            "Merge exact evidence and assess confirmation",
        ),
    ):
        script = _run_for(jobs[job_name], step_name)
        assert f'aggregate --tier {tier} --run-dir "$run_root"' in script
        assert f"assess --tier {tier}" in script
        assert '--output "$aggregate/assessment.json"' in script


def test_public_runner_commands_use_repo_root_module_mode() -> None:
    """Every public runner command resolves ``scripts`` in a clean checkout."""
    text, workflow = _load_workflow()
    jobs = workflow["jobs"]
    runner_module = "-m scripts.truncated_hierarchy_causal_runner"

    assert "scripts/truncated_hierarchy_causal_runner.py" not in text
    assert text.count(runner_module) == 5
    public_runner_steps = (
        (
            "materialize",
            "Freeze environment and materialize shared inputs once",
        ),
        ("smoke", "Run one co-located ten-cell smoke backend pair"),
        (
            "smoke_aggregate",
            "Merge exact evidence and assess smoke completeness",
        ),
        ("confirmation", "Run one co-located ten-cell confirmation backend pair"),
        (
            "confirmation_aggregate",
            "Merge exact evidence and assess confirmation",
        ),
    )
    for job_name, step_name in public_runner_steps:
        assert runner_module in _run_for(jobs[job_name], step_name)


def test_artifacts_are_isolated_and_never_merge_overwrite() -> None:
    """Parallel pairs retain evidence in unique artifacts with collision rejection."""
    text, workflow = _load_workflow()
    jobs = workflow["jobs"]

    assert "merge-multiple: true" not in text
    assert text.count("merge-multiple: false") == 2
    assert text.count("if: always()") >= 6
    for job_name, tier in (("smoke", "smoke"), ("confirmation", "confirmation")):
        upload = next(
            step
            for step in jobs[job_name]["steps"]
            if step.get("name") == f"Upload {tier} backend-pair evidence"
        )
        artifact_name = upload["with"]["name"]
        assert f"tn-causal-{tier}-${{{{ matrix.pair_id }}}}" in artifact_name
        assert "${{ github.run_id }}" in artifact_name
        assert "${{ github.run_attempt }}" in artifact_name
        assert upload["if"] == "always()"
        assert "/logs/" not in upload["with"]["path"]
        assert "contexts/${{ matrix.pair_id }}.json" in upload["with"]["path"]
        log_upload = jobs[job_name]["steps"][-1]
        assert log_upload["with"]["name"].startswith(f"tn-causal-log-{tier}-")
        assert "${{ matrix.pair_id }}" in log_upload["with"]["name"]
        assert log_upload["if"] == "always()"
    assert "${{ matrix.block_id }}" not in text
    assert "--block-id" not in text
    for job_name in ("smoke_aggregate", "confirmation_aggregate"):
        assembly = next(
            step["run"]
            for step in jobs[job_name]["steps"]
            if step.get("name", "").startswith("Merge exact evidence")
        )
        assert 'merge-runs --source-dir "$pair_root"' in assembly
        assert '--run-dir "$run_root"' in assembly
        assert "cp --" not in assembly
        assert "merge-multiple" not in assembly

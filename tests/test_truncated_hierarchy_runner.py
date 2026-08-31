"""Fast contract tests for the artifact-producing #1282 cell runner."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import xarray as xr

import scripts.truncated_hierarchy_runner as runner
from scripts.truncated_hierarchy_qualification import (
    build_environment_catalog,
    expand_plan,
    load_manifest,
    manifest_sha256,
    validate_result_record,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


@pytest.fixture(scope="module")
def manifest() -> Mapping[str, Any]:
    """Load the frozen executable manifest."""
    return load_manifest()


@pytest.fixture
def activate_planned_precision():
    """Isolate direct helper tests from process-global precision mutations."""
    previous_floatx = str(runner.pytensor.config.floatX)
    previous_jax_x64 = bool(runner.jax.config.x64_enabled)

    def activate(entry: Mapping[str, Any]) -> None:
        floatx = str(entry["scenario"]["floatx"])
        runner.pytensor.config.floatX = floatx
        runner.jax.config.update("jax_enable_x64", floatx == "float64")

    try:
        yield activate
    finally:
        runner.pytensor.config.floatX = previous_floatx
        runner.jax.config.update("jax_enable_x64", previous_jax_x64)


def _entry(manifest, tier: str, scenario_id: str, replicate: int = 0):
    return next(
        entry
        for entry in expand_plan(manifest, tier)
        if entry["scenario_id"] == scenario_id and entry["replicate"] == replicate
    )


def _fake_trace(built: runner.BuiltCell, *, seed: int = 91) -> xr.DataTree:
    scenario = built.plan_entry["scenario"]
    chains = int(scenario["chains"])
    draws = int(scenario["draws"])
    groups = int(scenario["n_groups"])
    rng = np.random.default_rng(seed)
    assert built.geometry is not None
    blocks = {block.canonical_name: block for block in built.geometry.blocks}
    posterior = xr.Dataset(
        {
            blocks["group_location"].random_variable_name: (
                ("chain", "draw"),
                rng.normal(0.4, 0.05, size=(chains, draws)),
            ),
            blocks["group_scale"].random_variable_name: (
                ("chain", "draw"),
                rng.lognormal(-1.2, 0.1, size=(chains, draws)),
            ),
            blocks["group_effect"].random_variable_name: (
                ("chain", "draw", "group"),
                rng.normal(0.4, 0.1, size=(chains, draws, groups)),
            ),
        }
    )
    stats = xr.Dataset(
        {
            "diverging": (("chain", "draw"), np.zeros((chains, draws), dtype=bool)),
            # PyMC 6.3 may also expose this aggregate alias; it must be ignored.
            "divergences": (("chain",), np.zeros(chains, dtype=np.int64)),
            "energy": (("chain", "draw"), rng.normal(size=(chains, draws))),
            "tree_depth": (
                ("chain", "draw"),
                np.full((chains, draws), 3, dtype=np.int64),
            ),
            "n_steps": (
                ("chain", "draw"),
                np.full((chains, draws), 7, dtype=np.int64),
            ),
            "step_size": (("chain", "draw"), np.full((chains, draws), 0.1)),
            "acceptance_rate": (
                ("chain", "draw"),
                np.full((chains, draws), 0.91),
            ),
        }
    )
    return xr.DataTree.from_dict({"posterior": posterior, "sample_stats": stats})


def _frozen_environment_record(manifest):
    """Build a valid 3.12 sidecar without attesting the test host as canonical."""
    profile_name = "current-resolved"
    profile = manifest["dependency_profiles"][profile_name]
    return {
        "schema_version": manifest["schema_version"],
        "study_id": manifest["study_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "runner_version": runner.RUNNER_VERSION,
        "dependency_profile": profile_name,
        "git": {
            "commit": "test-commit",
            "branch": "test-branch",
            "dirty": False,
        },
        "project": {
            field: profile[field]
            for field in (
                "project_path",
                "project_sha256",
                "lock_path",
                "lock_sha256",
            )
        },
        "runtime": {
            "python": f"{profile['python']}.0",
            "implementation": "CPython",
            "platform": "test-runner-image",
            "jax_enable_x64": True,
        },
        "packages": {
            "hssm": "test-version",
            **profile["required_versions"],
        },
    }


def _environment_catalog(manifest, monkeypatch):
    supplied = _frozen_environment_record(manifest)
    monkeypatch.setattr(
        runner,
        "collect_environment",
        lambda *_args: copy.deepcopy(supplied),
    )
    return build_environment_catalog([supplied], manifest)


def _unpaired_identity(entry) -> runner.ExecutionIdentity:
    return runner.ExecutionIdentity(
        execution_attempt_id=hashlib.sha256(entry["cell_id"].encode()).hexdigest(),
        pair_execution_id=None,
        pair_position=None,
        worker_identity_sha256="f" * 64,
    )


def _resign_phase_context(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "context_sha256"}
    payload["context_sha256"] = hashlib.sha256(
        runner._canonical_json_bytes(body)
    ).hexdigest()
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _activate_phase_context(
    monkeypatch,
    entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifact_root: Path,
    phase: runner.PhaseName,
    identity: runner.ExecutionIdentity,
) -> runner.PhaseContext:
    """Mirror a parent launch so a low-level phase can be tested in-process."""
    environment = runner._child_environment(
        entry, manifest, artifact_root, phase, identity
    )
    for name in (
        "PYTENSOR_FLAGS",
        "JAX_ENABLE_X64",
        "JAX_PLATFORMS",
        "JAX_COMPILATION_CACHE_DIR",
        "MPLCONFIGDIR",
        "XDG_CACHE_HOME",
        runner._PHASE_CONTEXT_ENV,
    ):
        monkeypatch.setenv(name, environment[name])
    return runner._load_phase_context(entry, artifact_root, phase)


def _good_diagnostics(*_args):
    return {
        "compile_success": True,
        "initialization_success": True,
        "logp_finite": True,
        "gradient_finite": True,
        "finite_difference_gradient_abs_error_max": 0.0,
        "finite_difference_gradient_rel_error_max": 0.0,
        "finite_difference_gradient_normalized_error_max": 0.0,
        "pytensor_jax_gradient_abs_error_max": 0.0,
        "pytensor_jax_gradient_rel_error_max": 0.0,
        "pytensor_jax_gradient_normalized_error_max": 0.0,
    }


def test_select_plan_cell_requires_exactly_one_frozen_identity(manifest) -> None:
    """The executable API never accepts an ad hoc or ambiguous cell."""
    selected = runner.select_plan_cell(
        manifest,
        tier="smoke",
        cell_id="smoke-pymc-lower-outside--replicate-00",
    )

    assert selected["scenario_id"] == "smoke-pymc-lower-outside"
    with pytest.raises(runner.RunnerError, match="selected 0 cells"):
        runner.select_plan_cell(manifest, tier="smoke", cell_id="not-a-cell")


@pytest.mark.parametrize(
    ("replicate", "expected_purposes"),
    [(0, ["candidate", "control"]), (1, ["control", "candidate"])],
)
def test_qualification_pair_resolution_and_counterbalanced_order(
    manifest, replicate: int, expected_purposes: list[str]
) -> None:
    """Either member resolves the same pair and replicate parity fixes its order."""
    candidate = _entry(manifest, "qualification", "qual-pymc-lower-outside", replicate)
    control = _entry(
        manifest, "qualification", "qual-pymc-lower-outside-control", replicate
    )

    assert runner.resolve_qualification_pair(candidate, manifest) == (
        candidate,
        control,
    )
    assert runner.resolve_qualification_pair(control, manifest) == (
        candidate,
        control,
    )
    ordered = runner.ordered_qualification_pair(control, manifest)
    assert [entry["scenario"]["purpose"] for entry in ordered] == expected_purposes


def test_every_required_qualification_cell_resolves_one_linked_pair(manifest) -> None:
    """The generic pair resolver covers the complete frozen non-SBC pair matrix."""
    entries = list(expand_plan(manifest, "qualification"))
    required = [entry for entry in entries if runner._requires_paired_execution(entry)]

    assert required
    for entry in required:
        candidate, control = runner.resolve_qualification_pair(entry, manifest)
        assert candidate["scenario"]["purpose"] == "candidate"
        assert control["scenario"]["purpose"] == "control"
        assert candidate["replicate"] == control["replicate"] == entry["replicate"]
        assert candidate["scenario"]["control_id"] == control["scenario_id"]
        assert entry in (candidate, control)


def test_direct_run_rejects_a_required_pair_member(manifest, tmp_path: Path) -> None:
    """A qualification pair cannot be split across independent run invocations."""
    entry = _entry(manifest, "qualification", "qual-pymc-lower-outside")

    with pytest.raises(runner.RunnerError, match="run-pair"):
        runner.orchestrate_cell(
            entry,
            manifest,
            runner.DEFAULT_MANIFEST,
            [],
            {},
            tmp_path,
        )


@pytest.mark.parametrize(
    ("replicate", "expected_purposes"),
    [(0, ["candidate", "control"]), (1, ["control", "candidate"])],
)
def test_orchestrate_pair_attempts_second_after_first_failure(
    manifest,
    tmp_path: Path,
    monkeypatch,
    replicate: int,
    expected_purposes: list[str],
) -> None:
    """A failed first member cannot suppress its co-located paired control/candidate."""
    anchor = _entry(manifest, "qualification", "qual-pymc-lower-outside", replicate)
    calls = []

    def fake_orchestrate(entry, *_args, execution_identity, **_kwargs):
        calls.append((entry, execution_identity))
        return {
            "cell_id": entry["cell_id"],
            "execution_status": "failed" if len(calls) == 1 else "completed",
            "provenance": {
                "execution_attempt_id": execution_identity.execution_attempt_id,
                "pair_execution_id": execution_identity.pair_execution_id,
                "pair_position": execution_identity.pair_position,
                "worker_identity_sha256": execution_identity.worker_identity_sha256,
            },
        }

    monkeypatch.setattr(runner, "_orchestrate_cell", fake_orchestrate)
    results = runner.orchestrate_pair(
        anchor,
        manifest,
        runner.DEFAULT_MANIFEST,
        [],
        {},
        tmp_path,
    )

    assert [item["scenario"]["purpose"] for item, _ in calls] == expected_purposes
    assert [result["execution_status"] for result in results] == [
        "failed",
        "completed",
    ]
    identities = [identity for _, identity in calls]
    assert identities[0].pair_position == 0
    assert identities[1].pair_position == 1
    assert identities[0].pair_execution_id == identities[1].pair_execution_id
    assert identities[0].execution_attempt_id != identities[1].execution_attempt_id
    assert identities[0].worker_identity_sha256 == identities[1].worker_identity_sha256
    assert all(
        runner._SHA256.fullmatch(identity.execution_attempt_id)
        for identity in identities
    )


def test_orchestrate_pair_contains_first_exception_and_still_attempts_second(
    manifest, tmp_path: Path, monkeypatch
) -> None:
    """An unexpected first-child exception is contained without short-circuiting."""
    anchor = _entry(manifest, "qualification", "qual-pymc-lower-outside")
    attempts: list[str] = []

    def fake_orchestrate(entry, *_args, **_kwargs):
        attempts.append(entry["cell_id"])
        if len(attempts) == 1:
            raise RuntimeError("unexpected parent-side failure")
        return {"cell_id": entry["cell_id"], "execution_status": "completed"}

    def fake_synthesize(entry, *_args, **_kwargs):
        return {"cell_id": entry["cell_id"], "execution_status": "failed"}

    monkeypatch.setattr(runner, "_orchestrate_cell", fake_orchestrate)
    monkeypatch.setattr(runner, "synthesize_child_failure", fake_synthesize)

    results = runner.orchestrate_pair(
        anchor,
        manifest,
        runner.DEFAULT_MANIFEST,
        [],
        {},
        tmp_path,
    )

    assert attempts == [
        "qual-pymc-lower-outside--replicate-00",
        "qual-pymc-lower-outside-control--replicate-00",
    ]
    assert [result["execution_status"] for result in results] == [
        "failed",
        "completed",
    ]


def test_module_cli_imports_in_clean_subprocess_without_pythonpath() -> None:
    """Fresh children resolve the scripts namespace without ambient PYTHONPATH."""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.truncated_hierarchy_runner",
            "--help",
        ],
        cwd=runner.REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Run exactly one frozen" in completed.stdout


@pytest.mark.parametrize("phase", ["sample", "finalize"])
def test_direct_child_cli_is_rejected_without_parent_context(
    tmp_path: Path, phase: str
) -> None:
    """Internal child commands are not user-callable execution shortcuts."""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop(runner._PHASE_CONTEXT_ENV, None)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.truncated_hierarchy_runner",
            phase,
            "--manifest",
            str(runner.DEFAULT_MANIFEST),
            "--tier",
            "smoke",
            "--cell-id",
            "smoke-pymc-lower-outside--replicate-00",
            "--artifact-root",
            str(tmp_path),
            "--environment",
            str(tmp_path / "unused-environment.json"),
        ],
        cwd=runner.REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert f"{phase} is an orchestrator-only phase" in completed.stderr


def test_child_environment_refuses_stale_phase_cache(manifest, tmp_path: Path) -> None:
    """A retry cannot reuse compilation/JIT work from a prior sampling attempt."""
    entry = _entry(manifest, "smoke", "smoke-pymc-lower-outside")
    identity = _unpaired_identity(entry)

    environment = runner._child_environment(
        entry, manifest, tmp_path, "sample", identity
    )
    cache_root = (
        tmp_path.resolve()
        / ".cache"
        / identity.execution_attempt_id
        / entry["cell_id"]
        / "sample"
    )
    assert environment["PYTENSOR_FLAGS"].endswith(
        f"base_compiledir={cache_root / 'pytensor'}"
    )

    with pytest.raises(runner.RunnerError, match="refusing to reuse"):
        runner._child_environment(entry, manifest, tmp_path, "sample", identity)


def test_phase_context_binds_phase_cell_attempt_and_cache_paths(
    manifest, tmp_path: Path, monkeypatch
) -> None:
    """A valid digest cannot authorize a different phase, cell, attempt, or cache."""
    entry = _entry(manifest, "smoke", "smoke-pymc-lower-outside")
    other = _entry(manifest, "smoke", "smoke-pymc-two-sided-near")
    identity = _unpaired_identity(entry)
    environment = runner._child_environment(
        entry, manifest, tmp_path, "sample", identity
    )
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    context = runner._load_phase_context(entry, tmp_path, "sample")
    assert context.identity == identity

    payload = json.loads(environment[runner._PHASE_CONTEXT_ENV])
    payload["cell_id"] = other["cell_id"]
    monkeypatch.setenv(
        runner._PHASE_CONTEXT_ENV,
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True),
    )
    with pytest.raises(runner.RunnerError, match="digest is invalid"):
        runner._load_phase_context(entry, tmp_path, "sample")

    monkeypatch.setenv(
        runner._PHASE_CONTEXT_ENV, environment[runner._PHASE_CONTEXT_ENV]
    )
    with pytest.raises(runner.RunnerError, match="requested phase and cell"):
        runner._load_phase_context(entry, tmp_path, "finalize")
    with pytest.raises(runner.RunnerError, match="requested phase and cell"):
        runner._load_phase_context(other, tmp_path, "sample")

    payload = json.loads(environment[runner._PHASE_CONTEXT_ENV])
    payload["execution_attempt_id"] = "e" * 64
    monkeypatch.setenv(runner._PHASE_CONTEXT_ENV, _resign_phase_context(payload))
    with pytest.raises(runner.RunnerError, match="requested phase and cell"):
        runner._load_phase_context(entry, tmp_path, "sample")

    payload = json.loads(environment[runner._PHASE_CONTEXT_ENV])
    payload["cache_paths"]["jax"] = str(tmp_path / "wrong-jax-cache")
    monkeypatch.setenv(runner._PHASE_CONTEXT_ENV, _resign_phase_context(payload))
    with pytest.raises(runner.RunnerError, match="requested phase and cell"):
        runner._load_phase_context(entry, tmp_path, "sample")


def test_low_level_phase_rejects_context_for_another_phase(
    manifest, tmp_path: Path, monkeypatch
) -> None:
    """Calling a phase function cannot bypass the parent-minted phase binding."""
    entry = _entry(manifest, "smoke", "smoke-pymc-lower-outside")
    context = _activate_phase_context(
        monkeypatch,
        entry,
        manifest,
        tmp_path,
        "sample",
        _unpaired_identity(entry),
    )

    with pytest.raises(runner.RunnerError, match="requested phase and cell"):
        runner.finalize_cell(
            entry,
            manifest,
            {},
            tmp_path,
            phase_context=context,
        )


def test_runtime_rejects_hssm_imported_from_another_checkout(
    monkeypatch, tmp_path: Path
) -> None:
    """A matching version string cannot hide a stale editable HSSM source."""
    wrong_source = tmp_path / "other-checkout" / "hssm" / "__init__.py"
    monkeypatch.setattr(runner.hssm, "__file__", str(wrong_source))

    with pytest.raises(runner.RunnerError, match="not from this checkout"):
        runner.validate_hssm_checkout()


def test_worker_environment_allows_host_image_and_python_patch_drift(
    manifest, monkeypatch
) -> None:
    """Fresh hosted VMs may differ descriptively while honoring one locked profile."""
    entry = _entry(manifest, "smoke", "smoke-pymc-lower-outside")
    supplied = _frozen_environment_record(manifest)
    observed = copy.deepcopy(supplied)
    python_parts = observed["runtime"]["python"].split(".")
    observed["runtime"]["python"] = ".".join([*python_parts[:2], "999"])
    observed["runtime"]["platform"] = "Linux-new-runner-image-x86_64"
    observed["runtime"]["jax_enable_x64"] = not supplied["runtime"]["jax_enable_x64"]
    catalog = build_environment_catalog([supplied], manifest)
    monkeypatch.setattr(runner, "collect_environment", lambda *_args: observed)

    assert runner._environment_for_cell(entry, catalog, manifest) is supplied


def test_worker_environment_rejects_stable_profile_drift(manifest, monkeypatch) -> None:
    """A changed package or source contract cannot reuse another worker's sidecar."""
    entry = _entry(manifest, "smoke", "smoke-pymc-lower-outside")
    supplied = _frozen_environment_record(manifest)
    observed = copy.deepcopy(supplied)
    observed["packages"]["pymc"] = "0.0.0"
    catalog = build_environment_catalog([supplied], manifest)
    monkeypatch.setattr(runner, "collect_environment", lambda *_args: observed)

    with pytest.raises(runner.RunnerError, match="stable profile contract"):
        runner._environment_for_cell(entry, catalog, manifest)


def test_worker_environment_rejects_python_minor_drift(manifest, monkeypatch) -> None:
    """The host-independent test fixture must not relax the frozen Python minor."""
    entry = _entry(manifest, "smoke", "smoke-pymc-lower-outside")
    supplied = _frozen_environment_record(manifest)
    observed = copy.deepcopy(supplied)
    observed["runtime"]["python"] = "3.13.0"
    catalog = build_environment_catalog([supplied], manifest)
    monkeypatch.setattr(runner, "collect_environment", lambda *_args: observed)

    with pytest.raises(runner.RunnerError, match="stable profile contract"):
        runner._environment_for_cell(entry, catalog, manifest)


def test_candidate_and_control_generate_identical_shared_data_bytes(manifest) -> None:
    """Paired fits cannot silently redraw truths or observations."""
    candidate = _entry(manifest, "qualification", "qual-pymc-lower-outside")
    control = _entry(manifest, "qualification", "qual-pymc-lower-outside-control")

    candidate_payload = runner.generate_data_payload(candidate)
    control_payload = runner.generate_data_payload(control)

    assert candidate_payload == control_payload
    assert runner._canonical_json_bytes(candidate_payload) == (
        runner._canonical_json_bytes(control_payload)
    )


@pytest.mark.parametrize(
    "scenario_id",
    [
        "smoke-hssm-lba2-near",
        "smoke-hssm-ddm-z-near",
        "smoke-hssm-softmax-beta",
    ],
)
def test_hssm_data_generation_is_byte_stable_for_same_entry(
    manifest, scenario_id: str
) -> None:
    """Every HSSM DGP is a pure function of one frozen plan entry."""
    entry = _entry(manifest, "smoke", scenario_id)

    first = runner._canonical_json_bytes(runner.generate_data_payload(entry))
    second = runner._canonical_json_bytes(runner.generate_data_payload(entry))

    assert first == second


def test_every_hssm_candidate_control_pair_has_identical_data_bytes(manifest) -> None:
    """All HSSM pairs share exact bytes across models, samplers, and replicates."""
    entries = list(expand_plan(manifest, "qualification"))
    by_cell = {entry["cell_id"]: entry for entry in entries}
    candidates = [
        entry
        for entry in entries
        if entry["scenario"]["layer"] == "hssm"
        and entry["scenario"]["purpose"] == "candidate"
    ]
    assert {entry["scenario"]["model"] for entry in candidates} == {
        "lba2_b",
        "approx_ddm_z",
        "softmax_beta",
    }

    for candidate in candidates:
        control_cell = (
            f"{candidate['scenario']['control_id']}--"
            f"replicate-{candidate['replicate']:02d}"
        )
        control = by_cell[control_cell]
        candidate_bytes = runner._canonical_json_bytes(
            runner.generate_data_payload(candidate)
        )
        control_bytes = runner._canonical_json_bytes(
            runner.generate_data_payload(control)
        )
        assert candidate_bytes == control_bytes, candidate["cell_id"]


def test_lba_generation_restores_legacy_numpy_rng(manifest) -> None:
    """Scoped LBA determinism must not perturb later process-global RNG draws."""
    entry = _entry(manifest, "smoke", "smoke-hssm-lba2-near")
    original_state = np.random.get_state()
    try:
        np.random.seed(1282)
        expected = np.random.random(8)
        np.random.seed(1282)

        runner.generate_data_payload(entry)
        observed = np.random.random(8)
    finally:
        np.random.set_state(original_state)

    np.testing.assert_array_equal(observed, expected)


def test_shared_data_writer_reuses_only_identical_bytes(
    manifest, tmp_path: Path
) -> None:
    """A second data owner verifies rather than replaces canonical evidence."""
    candidate = _entry(manifest, "qualification", "qual-pymc-lower-outside")
    control = _entry(manifest, "qualification", "qual-pymc-lower-outside-control")

    first, first_path, first_digest = runner.materialize_data_artifact(
        candidate, manifest, tmp_path
    )
    second, second_path, second_digest = runner.materialize_data_artifact(
        control, manifest, tmp_path
    )

    assert first == second
    assert first_path == second_path
    assert first_digest == second_digest
    path = tmp_path / first_path
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="differs from canonical bytes"):
        runner.materialize_data_artifact(candidate, manifest, tmp_path)


@pytest.mark.parametrize(
    ("scenario_id", "observation_keys"),
    [
        ("smoke-hssm-lba2-near", {"rt", "response"}),
        ("smoke-hssm-ddm-z-near", {"rt", "response"}),
        ("smoke-hssm-softmax-beta", {"response"}),
    ],
)
def test_hssm_data_generators_follow_model_contracts(
    manifest, scenario_id: str, observation_keys: set[str]
) -> None:
    """Each HSSM DGP produces one balanced observation panel."""
    entry = _entry(manifest, "smoke", scenario_id)
    payload = runner.generate_data_payload(entry)

    assert set(payload["observations"]) == observation_keys
    assert len(payload["group_index"]) == (
        entry["scenario"]["n_groups"] * entry["scenario"]["n_per_group"]
    )
    assert len(payload["truth"]["group_effect"]) == entry["scenario"]["n_groups"]


def test_backend_default_starts_are_exact_and_finite(
    manifest, activate_planned_precision
) -> None:
    """Direct PyMC starts consume every planned per-chain start seed once."""
    entry = _entry(manifest, "smoke", "smoke-pymc-lower-outside")
    activate_planned_precision(entry)
    payload = runner.generate_data_payload(entry)
    built = runner.build_cell_model(entry, payload)

    starts, finite = runner.materialize_exact_starts(built)

    assert finite
    assert starts.start_seeds == tuple(entry["start_seeds"])
    assert len(starts.transformed_points) == entry["scenario"]["chains"]
    assert all(
        set(point) == {value.name for value in built.pymc_model.value_vars}
        for point in starts.transformed_points
    )


def test_pymc_sampler_receives_frozen_budget_seeds_and_adapt_diag(
    manifest, monkeypatch, activate_planned_precision
) -> None:
    """The PyMC path cannot add initializer jitter or substitute sampler seeds."""
    entry = _entry(manifest, "smoke", "smoke-pymc-lower-outside")
    activate_planned_precision(entry)
    built = runner.build_cell_model(entry, runner.generate_data_payload(entry))
    starts, _ = runner.materialize_exact_starts(built)
    sentinel = xr.DataTree()
    captured = {}

    def fake_sample(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(runner.pm, "sample", fake_sample)
    assert runner.sample_cell_model(built, starts) is sentinel
    assert captured["draws"] == 250
    assert captured["tune"] == 250
    assert captured["chains"] == 2
    assert captured["target_accept"] == 0.9
    assert captured["random_seed"] == entry["chain_seeds"]
    assert captured["init"] == "adapt_diag"
    assert captured["cores"] == 1
    assert captured["initvals"] == list(starts.transformed_points)


def test_numpyro_sampler_receives_scalar_seed_no_jitter_and_sequential_chains(
    manifest, monkeypatch, activate_planned_precision
) -> None:
    """NumPyro uses its single frozen root key and the exact transformed starts."""
    entry = _entry(manifest, "qualification", "qual-pymc-lower-outside-numpyro")
    activate_planned_precision(entry)
    built = runner.build_cell_model(entry, runner.generate_data_payload(entry))
    starts, _ = runner.materialize_exact_starts(built)
    sentinel = xr.DataTree()
    captured = {}

    def fake_sample(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(runner, "sample_numpyro_nuts", fake_sample)
    assert runner.sample_cell_model(built, starts) is sentinel
    assert captured["random_seed"] == entry["sampler_seed"]
    assert captured["jitter"] is False
    assert captured["chain_method"] == "sequential"
    assert captured["initvals"] == list(starts.transformed_points)


def test_standardized_chain_uses_diverging_not_aggregate_alias(manifest) -> None:
    """Only the per-draw divergence statistic enters the immutable chain."""
    entry = _entry(manifest, "smoke", "smoke-pymc-lower-outside")
    built = runner.build_cell_model(entry, runner.generate_data_payload(entry))
    raw = _fake_trace(built)
    raw["sample_stats"].ds["diverging"][0, 0] = True
    raw["sample_stats"].ds["divergences"][:] = 99

    standardized = runner.standardize_chain(built, raw)
    metrics = runner.compute_sampler_metrics(standardized, sampling_elapsed_seconds=2.0)

    posterior = standardized["posterior"].ds
    assert set(posterior.data_vars) == set(runner._CHAIN_POSTERIOR)
    assert posterior["group_effect"].dims == ("chain", "draw", "group")
    assert posterior["group_effect"].shape == (2, 250, 4)
    assert "divergences" not in standardized["sample_stats"].ds
    assert metrics["divergence_count"] == 1
    assert metrics["posterior_draw_count"] == 500
    assert metrics["divergence_rate"] == pytest.approx(1 / 500)
    assert metrics["leapfrog_step_count"] == 3500
    assert metrics["gradient_evaluation_count"] == 3500


def test_group_diagnostics_include_an_unmonitored_bad_coefficient(manifest) -> None:
    """A bad group outside first/middle/last must still determine the group gate."""
    entry = _entry(manifest, "smoke", "smoke-pymc-lower-outside")
    built = runner.build_cell_model(entry, runner.generate_data_payload(entry))
    standardized = runner.standardize_chain(built, _fake_trace(built))
    posterior = standardized["posterior"].ds
    # Frozen monitored indices are 0, 2, and 3, leaving coefficient 1 available
    # for this regression. Give it severe between-chain disagreement without
    # changing any of the three recovery-summary variables.
    rng = np.random.default_rng(18)
    posterior["group_effect"][0, :, 1] = rng.normal(-4.0, 0.1, size=250)
    posterior["group_effect"][1, :, 1] = rng.normal(4.0, 0.1, size=250)

    selected_rhat = runner._finite_values(
        runner.az.rhat(
            posterior[["group_first", "group_middle", "group_last"]],
            method="rank",
        )
    )
    metrics = runner.compute_sampler_metrics(standardized, sampling_elapsed_seconds=2.0)

    assert np.max(selected_rhat) < 1.1
    assert metrics["group_rhat_max"] > 1.1
    assert metrics["group_ess_bulk_fraction_ge_400"] < 1.0
    assert metrics["group_ess_tail_fraction_ge_400"] < 1.0


def test_chain_validation_binds_recovery_views_to_full_group_effect(manifest) -> None:
    """Selected recovery variables cannot disagree with all-group diagnostics."""
    entry = _entry(manifest, "smoke", "smoke-pymc-lower-outside")
    built = runner.build_cell_model(entry, runner.generate_data_payload(entry))
    standardized = runner.standardize_chain(built, _fake_trace(built))
    posterior = standardized["posterior"].to_dataset()
    posterior["group_middle"] = posterior["group_middle"] + 1.0
    inconsistent = xr.DataTree.from_dict(
        {
            "posterior": posterior,
            "sample_stats": standardized["sample_stats"].to_dataset(),
        }
    )

    with pytest.raises(runner.RunnerError, match="not the exact group_effect slice"):
        runner.validate_standardized_chain(inconsistent, entry["scenario"])


def test_chain_netcdf_is_reopened_and_hashed(manifest, tmp_path: Path) -> None:
    """The published NetCDF is structurally validated after serialization."""
    entry = _entry(manifest, "smoke", "smoke-pymc-lower-outside")
    built = runner.build_cell_model(entry, runner.generate_data_payload(entry))
    standardized = runner.standardize_chain(built, _fake_trace(built))

    relative, digest = runner.write_chain_artifact(
        standardized, entry, manifest, tmp_path
    )

    path = tmp_path / relative
    assert runner._file_sha256(path) == digest
    reopened = xr.open_datatree(path)
    try:
        reopened.load()
        runner.validate_standardized_chain(reopened, entry["scenario"])
    finally:
        reopened.close()


def test_failed_sampling_publishes_data_start_and_final_marker_only(
    manifest, tmp_path: Path, monkeypatch, activate_planned_precision
) -> None:
    """A sampler failure retains completed evidence without inventing a chain."""
    entry = _entry(manifest, "smoke", "smoke-pymc-lower-outside")
    activate_planned_precision(entry)
    catalog = _environment_catalog(manifest, monkeypatch)
    monkeypatch.setattr(
        runner,
        "validate_runtime_contract",
        lambda *_args: ("float64", True),
    )

    def fail_sample(*_args):
        raise RuntimeError("planned sampler failure")

    monkeypatch.setattr(runner, "sample_cell_model", fail_sample)
    phase_context = _activate_phase_context(
        monkeypatch,
        entry,
        manifest,
        tmp_path,
        "sample",
        _unpaired_identity(entry),
    )
    record = runner.run_sample_phase(
        entry,
        manifest,
        catalog,
        tmp_path,
        phase_context=phase_context,
    )

    assert record["execution_status"] == "failed"
    assert record["failure"] == {
        "stage": "sampling",
        "error_type": "RuntimeError",
        "message": "planned sampler failure",
    }
    assert (tmp_path / record["provenance"]["data_artifact"]).is_file()
    assert (tmp_path / record["provenance"]["actual_start_artifact"]).is_file()
    assert record["provenance"]["raw_chain_artifact"] is None
    assert (tmp_path / "cells" / f"{entry['cell_id']}.json").is_file()
    validate_result_record(record, entry, catalog, manifest)


def test_complete_mocked_cell_publishes_cell_after_all_artifacts(
    manifest, tmp_path: Path, monkeypatch, activate_planned_precision
) -> None:
    """The final marker binds byte-verified data, starts, chains, and raw metrics."""
    entry = _entry(manifest, "smoke", "smoke-pymc-lower-outside")
    activate_planned_precision(entry)
    catalog = _environment_catalog(manifest, monkeypatch)
    monkeypatch.setattr(
        runner,
        "validate_runtime_contract",
        lambda *_args: ("float64", True),
    )
    monkeypatch.setattr(
        runner,
        "sample_cell_model",
        lambda built, _starts: _fake_trace(built),
    )

    identity = _unpaired_identity(entry)
    sample_context = _activate_phase_context(
        monkeypatch, entry, manifest, tmp_path, "sample", identity
    )
    sampled = runner.run_sample_phase(
        entry,
        manifest,
        catalog,
        tmp_path,
        phase_context=sample_context,
    )
    assert sampled["phase_status"] == "sampled"
    finalize_context = _activate_phase_context(
        monkeypatch, entry, manifest, tmp_path, "finalize", identity
    )
    record = runner.finalize_cell(
        entry,
        manifest,
        catalog,
        tmp_path,
        phase_context=finalize_context,
        diagnostics_fn=_good_diagnostics,
    )

    assert record["execution_status"] == "completed"
    assert record["metrics"]["sampling_success"] is True
    assert record["provenance"]["execution_attempt_id"] == (
        identity.execution_attempt_id
    )
    assert record["provenance"]["pair_execution_id"] is None
    assert record["provenance"]["pair_position"] is None
    assert record["provenance"]["worker_identity_sha256"] == (
        identity.worker_identity_sha256
    )
    for field in (
        "data_artifact",
        "actual_start_artifact",
        "raw_chain_artifact",
    ):
        assert (tmp_path / record["provenance"][field]).is_file()
    marker = tmp_path / "cells" / f"{entry['cell_id']}.json"
    assert json.loads(marker.read_text(encoding="utf-8")) == record
    validate_result_record(record, entry, catalog, manifest)


def test_tiny_real_pymc_sampling_path_returns_datatree(
    manifest, activate_planned_precision
) -> None:
    """Exercise PyMC's real explicit-transformed-start API with a two-draw fit."""
    canonical = _entry(manifest, "smoke", "smoke-pymc-lower-outside")
    entry = copy.deepcopy(canonical)
    activate_planned_precision(entry)
    entry["scenario"]["chains"] = 1
    entry["scenario"]["tune"] = 2
    entry["scenario"]["draws"] = 2
    entry["start_seeds"] = entry["start_seeds"][:1]
    entry["chain_seeds"] = entry["chain_seeds"][:1]
    payload = runner.generate_data_payload(entry)
    built = runner.build_cell_model(entry, payload)
    starts, _ = runner.materialize_exact_starts(built)

    trace = runner.sample_cell_model(built, starts)
    standardized = runner.standardize_chain(built, trace)

    assert isinstance(trace, xr.DataTree)
    assert standardized["posterior"].ds.sizes == {
        "chain": 1,
        "draw": 2,
        "group": 4,
    }

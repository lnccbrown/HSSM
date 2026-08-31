"""Tests for the no-sampling bounded-hierarchy qualification contract."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

import pytest

import scripts.truncated_hierarchy_qualification as qualification
from scripts.truncated_hierarchy_qualification import (
    QualificationError,
    aggregate_results,
    assess_results,
    build_environment_catalog,
    compare_threshold,
    derive_seed,
    environment_sha256,
    expand_plan,
    load_jsonl,
    load_manifest,
    main,
    manifest_sha256,
    strict_json_loads,
    validate_environment,
    validate_environment_catalog,
    validate_manifest,
    validate_plan,
    validate_result_record,
    write_aggregate,
    write_cell_result,
    write_plan,
)
from scripts.truncated_hierarchy_statistics import derive_sbc_rank_tie_index

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def manifest():
    """Load the repository's frozen qualification manifest."""
    return load_manifest()


def _good_metrics(entry, manifest) -> dict[str, bool | int | float]:
    metrics = {
        "compile_success": True,
        "initialization_success": True,
        "logp_finite": True,
        "gradient_finite": True,
        "sampling_success": True,
        "divergence_count": 0,
        "posterior_draw_count": (
            entry["scenario"]["chains"] * entry["scenario"]["draws"]
        ),
        "divergence_rate": 0.0,
        "hyper_rhat_max": 1.009,
        "hyper_ess_bulk_min": 400,
        "hyper_ess_tail_min": 400,
        "bfmi_min": 0.3,
        "treedepth_saturation_rate": 0.0009,
        "hyper_mcse_over_sd_max": 0.05,
        "group_rhat_max": 1.009,
        "group_ess_bulk_fraction_ge_400": 0.95,
        "group_ess_tail_fraction_ge_400": 0.95,
        "hyper_ess_per_second_median": 100.0,
        "hyper_leapfrog_steps_per_effective_sample_median": 2.0,
    }
    conditions = qualification._gradient_contract_conditions(
        {"replicate": entry["replicate"]},
        entry["scenario"],
        manifest["analysis_policy"],
    )
    metrics.update(dict.fromkeys(conditions, 0.0))
    return metrics


def _parameter_summaries(entry, manifest, status: str) -> list[dict[str, object]]:
    scenario = entry["scenario"]
    if (
        status != "completed"
        or not scenario["recovery"]
        or scenario["purpose"] not in {"candidate", "control"}
    ):
        return []
    family = scenario["purpose"]
    summaries = []
    for parameter_id in manifest["analysis_policy"]["monitored_parameters"]:
        q025, q05, q50, q95, q975 = -2.0, -1.5, 0.0, 1.5, 2.0
        if scenario.get("calibration_kind") == "sbc":
            # R=275 and M=20 accept K90=250 and K95=260.  Preserve interval
            # nesting explicitly: 250 truths lie in both intervals, another
            # 10 lie only in the wider 95% interval, and 15 lie in neither.
            replicate = entry["replicate"]
            if 250 <= replicate < 260:
                q025, q05, q50, q95, q975 = -2.0, 0.5, 1.0, 1.5, 2.0
            elif replicate >= 260:
                q025, q05, q50, q95, q975 = 0.5, 0.75, 1.0, 1.5, 2.0
        summary: dict[str, object] = {
            "family": family,
            "scenario_id": entry["scenario_id"],
            "parameter_id": parameter_id,
            "replicate": entry["replicate"],
            "truth": 0.0,
            "posterior_mean": 0.0,
            "posterior_sd": 1.0,
            "posterior_mcse": 0.1,
            "q025": q025,
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "q975": q975,
        }
        if scenario.get("calibration_kind") == "sbc":
            rank = entry["replicate"] % (
                manifest["analysis_policy"]["sbc_rank_draw_count"] + 1
            )
            summary.update(
                {
                    "rank_less": rank,
                    "rank_equal": 0,
                    "rank_tie_index": 0,
                    "rank": rank,
                    "rank_draw_count": manifest["analysis_policy"][
                        "sbc_rank_draw_count"
                    ],
                }
            )
        summaries.append(summary)
    return summaries


def _result(
    entry,
    environment_catalog,
    manifest,
    *,
    status: str = "completed",
    metrics: dict[str, bool | int | float] | None = None,
    unavailable_metrics: dict[str, str] | None = None,
    parameter_summaries: list[dict[str, object]] | None = None,
):
    dependency_profile = entry["scenario"].get("dependency_profile", "current-resolved")
    environment = next(
        item
        for item in environment_catalog.values()
        if item["dependency_profile"] == dependency_profile
    )
    failure = None
    if status != "completed":
        failure = {
            "stage": "sample",
            "error_type": "RuntimeError",
            "message": "sampler failed",
        }
    return {
        "schema_version": entry["schema_version"],
        "study_id": entry["study_id"],
        "manifest_sha256": entry["manifest_sha256"],
        "cell_id": entry["cell_id"],
        "scenario_id": entry["scenario_id"],
        "replicate": entry["replicate"],
        "data_seed": entry["data_seed"],
        "sbc_tie_seed": entry["sbc_tie_seed"],
        "chain_seeds": entry["chain_seeds"],
        "execution_status": status,
        "metrics": _good_metrics(entry, manifest) if metrics is None else metrics,
        "unavailable_metrics": unavailable_metrics or {},
        "parameter_summaries": (
            _parameter_summaries(entry, manifest, status)
            if parameter_summaries is None
            else parameter_summaries
        ),
        "failure": failure,
        "provenance": {
            "runner_version": 1,
            "sampler": entry["scenario"]["sampler"],
            "device": "test-device",
            "floatx": entry["scenario"]["floatx"],
            "actual_start_artifact": (
                f"starts/{entry['cell_id']}.json" if status == "completed" else None
            ),
            "actual_start_sha256": "b" * 64 if status == "completed" else None,
            "git_commit": environment["git"]["commit"],
            "environment_sha256": environment_sha256(environment, manifest),
        },
    }


def _environment_record(
    manifest,
    dependency_profile: str,
    *,
    dirty: bool = False,
    jax_enable_x64: bool = True,
):
    profile = manifest["dependency_profiles"][dependency_profile]
    record = {
        "schema_version": 1,
        "study_id": manifest["study_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "runner_version": 1,
        "dependency_profile": dependency_profile,
        "git": {
            "commit": "test-commit",
            "branch": "test-branch",
            "dirty": dirty,
        },
        "project": {
            "project_path": profile["project_path"],
            "project_sha256": profile["project_sha256"],
            "lock_path": profile["lock_path"],
            "lock_sha256": profile["lock_sha256"],
        },
        "runtime": {
            "python": f"{profile['python']}.9",
            "implementation": "CPython",
            "platform": "test-platform",
            "jax_enable_x64": jax_enable_x64,
        },
        "packages": {
            "hssm": "0.4.0",
            **profile["required_versions"],
        },
    }
    assert validate_environment(record, manifest) == record
    return record


@pytest.fixture(scope="module")
def environment_record(manifest):
    """Build one exact current-profile environment sidecar."""
    return _environment_record(manifest, "current-resolved")


@pytest.fixture(scope="module")
def environment(manifest):
    """Build a two-profile catalogue for all synthetic cell records."""
    return build_environment_catalog(
        [
            _environment_record(manifest, "current-resolved"),
            _environment_record(manifest, "bambi-0.19"),
        ],
        manifest,
    )


@pytest.fixture
def make_result(manifest, environment):
    """Build a result bound to its profile-specific test environment."""

    def factory(entry, **kwargs):
        return _result(entry, environment, manifest, **kwargs)

    return factory


def test_manifest_is_frozen_and_has_all_predeclared_tiers(manifest) -> None:
    """Lock the reviewed manifest digest and exact tier/cell cardinalities."""
    assert manifest["status"] == "frozen-before-primary-runs"
    assert manifest_sha256(manifest) == (
        "e1d15e9ac8460c5c8e1a68c5b8055be288475ef28a115c5a86303733a1428cf0"
    )
    assert {scenario["tier"] for scenario in manifest["scenarios"]} == {
        "smoke",
        "qualification",
        "stress",
    }
    assert len(expand_plan(manifest, "smoke")) == 10
    assert len(expand_plan(manifest, "qualification")) == 720
    assert len(expand_plan(manifest, "stress")) == 18


def test_strict_json_rejects_nonstandard_numbers() -> None:
    """Do not let NaN or infinity enter digests or scientific decisions."""
    for value in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(QualificationError, match="forbidden"):
            strict_json_loads(f'{{"metric": {value}}}')

    for text in ('{"metric": 1, "metric": 2}', '{"nested": {"x": 1, "x": 2}}'):
        with pytest.raises(QualificationError, match="duplicate JSON object key"):
            strict_json_loads(text)


def test_manifest_rejects_unknown_fields_and_primary_gate_drift(manifest) -> None:
    """Make scenario additions and primary-budget changes explicit reviews."""
    unknown = copy.deepcopy(manifest)
    unknown["scenarios"][0]["surprise"] = True
    with pytest.raises(QualificationError, match="unknown"):
        validate_manifest(unknown)

    changed_budget = copy.deepcopy(manifest)
    qualification = next(
        item for item in changed_budget["scenarios"] if item["tier"] == "qualification"
    )
    qualification["target_accept"] = 0.95
    with pytest.raises(QualificationError, match="must remain 0.9"):
        validate_manifest(changed_budget)


def test_seed_derivation_is_stable_and_separates_data_from_chains() -> None:
    """Pin cross-process seeds and keep dataset and chain streams independent."""
    data_seed = derive_seed(1282, "qual-hssm-lba2-near-pymc", 0, "data")
    chain_seeds = [
        derive_seed(1282, "qual-hssm-lba2-near-pymc", 0, "chain", chain)
        for chain in range(4)
    ]

    assert data_seed == 1453647211
    assert chain_seeds == [2076627074, 532482532, 1752810389, 1206549751]
    assert len({data_seed, *chain_seeds}) == 5
    assert derive_seed(1282, "qual-hssm-lba2-near-pymc", 1, "data") != data_seed
    assert derive_seed(1282, "calib-pymc-lower-outside", 0, "sbc_tie") == 481314911


def test_plan_rejects_reordering_seed_changes_and_missing_cells(manifest) -> None:
    """Require the executed primary plan to equal the frozen expansion exactly."""
    plan = expand_plan(manifest, "qualification")

    reordered = list(plan)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(QualificationError, match="does not match frozen cell"):
        validate_plan(reordered, manifest, "qualification")

    changed_seed = copy.deepcopy(plan)
    changed_seed[0]["chain_seeds"][0] += 1
    with pytest.raises(QualificationError, match="does not match frozen cell"):
        validate_plan(changed_seed, manifest, "qualification")

    with pytest.raises(QualificationError, match="expected 720"):
        validate_plan(plan[:-1], manifest, "qualification")


def test_controls_match_candidates_and_share_only_their_data_seed(manifest) -> None:
    """Pair controls on scientific dimensions without coupling chain randomness."""
    scenarios = {item["scenario_id"]: item for item in manifest["scenarios"]}
    plan = expand_plan(manifest, "qualification")
    entries = {(entry["scenario_id"], entry["replicate"]): entry for entry in plan}

    for candidate in scenarios.values():
        if (
            candidate["tier"] != "qualification"
            or candidate["purpose"] != "candidate"
            or candidate.get("calibration_kind") is not None
        ):
            continue
        control = scenarios[candidate["control_id"]]
        assert control["sampler"] == candidate["sampler"]
        assert control["floatx"] == candidate["floatx"]
        assert control["layer"] == candidate["layer"]
        assert control["model"] == candidate["model"]
        for replicate in range(candidate["replicates"]):
            candidate_entry = entries[(candidate["scenario_id"], replicate)]
            control_entry = entries[(control["scenario_id"], replicate)]
            assert candidate_entry["data_seed"] == control_entry["data_seed"]
            assert set(candidate_entry["chain_seeds"]).isdisjoint(
                control_entry["chain_seeds"]
            )


def test_calibration_units_are_frozen_and_candidate_only(manifest) -> None:
    """Bind the power calculation to every declared SBC scenario and parameter."""
    calibration = [
        scenario
        for scenario in manifest["scenarios"]
        if scenario.get("calibration_kind") == "sbc"
    ]
    assert calibration
    assert {scenario["purpose"] for scenario in calibration} == {"candidate"}
    assert {scenario["control_id"] for scenario in calibration} == {None}
    assert (
        len(calibration) * len(manifest["analysis_policy"]["monitored_parameters"])
        == manifest["analysis_policy"]["coverage_power_design"][
            "candidate_parameter_units"
        ]
    )

    missing_unit_family = copy.deepcopy(manifest)
    missing_unit_family["scenarios"] = [
        scenario
        for scenario in missing_unit_family["scenarios"]
        if scenario["scenario_id"] != calibration[0]["scenario_id"]
    ]
    with pytest.raises(QualificationError, match="candidate_scenario_ids"):
        qualification.validate_manifest(missing_unit_family)

    wrong_family = copy.deepcopy(manifest)
    target = next(
        scenario
        for scenario in wrong_family["scenarios"]
        if scenario.get("calibration_kind") == "sbc"
    )
    target["purpose"] = "control"
    with pytest.raises(QualificationError, match="primary candidate"):
        qualification.validate_manifest(wrong_family)


def test_plan_writes_deterministic_jsonl_csv_and_valid_environment(
    manifest, environment_record, monkeypatch, tmp_path: Path
) -> None:
    """Emit reviewable plans plus a schema-checked provenance sidecar."""
    monkeypatch.setattr(
        qualification,
        "collect_environment",
        lambda _manifest, _dependency_profile: copy.deepcopy(environment_record),
    )
    plan = expand_plan(manifest, "smoke")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_paths = write_plan(plan, manifest, "smoke", first_dir)
    second_paths = write_plan(plan, manifest, "smoke", second_dir)

    assert first_paths[0].read_bytes() == second_paths[0].read_bytes()
    assert first_paths[1].read_bytes() == second_paths[1].read_bytes()
    assert load_jsonl(first_paths[0]) == plan
    environment = strict_json_loads(first_paths[2].read_text())
    assert validate_environment(environment, manifest) == environment
    assert set(environment["packages"]) == {
        "hssm",
        "arviz",
        "bambi",
        "formulae",
        "jaxlib",
        "jaxonnxruntime",
        "pymc",
        "pytensor",
        "jax",
        "numpy",
        "numpyro",
        "scipy",
        "ssm-simulators",
    }
    assert environment["dependency_profile"] == "current-resolved"
    assert environment["runtime"]["jax_enable_x64"] is True
    assert environment["project"]["project_path"].endswith("pyproject.toml")
    assert len(environment["project"]["project_sha256"]) == 64
    assert environment["project"]["lock_path"].endswith("uv.lock")
    assert len(environment["project"]["lock_sha256"]) == 64


def test_environment_schema_rejects_wrong_manifest_digest(
    manifest, environment_record
) -> None:
    """Prevent result provenance from silently referring to another study."""
    environment = copy.deepcopy(environment_record)
    environment["manifest_sha256"] = "0" * 64

    with pytest.raises(QualificationError, match="digest mismatch"):
        validate_environment(environment, manifest)


def test_collect_environment_records_selected_profile_and_runtime(
    manifest, monkeypatch
) -> None:
    """Collect every pinned package plus the selected project, lock, and JAX mode."""
    profile = manifest["dependency_profiles"]["bambi-0.19"]
    versions = {"hssm": "0.4.0", **profile["required_versions"]}

    monkeypatch.setattr(
        qualification.importlib.metadata, "version", versions.__getitem__
    )
    monkeypatch.setattr(qualification.platform, "python_version", lambda: "3.12.9")
    monkeypatch.setattr(
        qualification.platform, "python_implementation", lambda: "CPython"
    )
    monkeypatch.setattr(qualification.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(qualification, "_jax_enable_x64", lambda: True)

    def fake_git(*args):
        values = {
            ("status", "--porcelain"): "",
            ("rev-parse", "HEAD"): "test-commit",
            ("branch", "--show-current"): "test-branch",
        }
        return values[args]

    monkeypatch.setattr(qualification, "_git_value", fake_git)

    environment = qualification.collect_environment(manifest, "bambi-0.19")

    assert environment["dependency_profile"] == "bambi-0.19"
    assert environment["packages"] == versions
    assert environment["project"] == {
        field: profile[field]
        for field in (
            "project_path",
            "project_sha256",
            "lock_path",
            "lock_sha256",
        )
    }
    assert environment["runtime"]["jax_enable_x64"] is True
    assert environment["git"]["dirty"] is False


@pytest.mark.parametrize(
    ("dependency_profile", "bambi_version", "expected_profile"),
    [
        ("current-resolved", "0.19.0", "current-resolved"),
        ("bambi-0.19", "0.20.0", "bambi-0.19"),
    ],
)
def test_environment_rejects_wrong_profile_version(
    dependency_profile,
    bambi_version,
    expected_profile,
    manifest,
) -> None:
    """Require every sidecar to prove the exact selected dependency profile."""
    environment = _environment_record(manifest, dependency_profile)
    environment["packages"]["bambi"] = bambi_version

    with pytest.raises(QualificationError, match=expected_profile):
        validate_environment(environment, manifest)


def test_environment_catalog_rejects_forged_semantic_digest(
    manifest, environment_record
) -> None:
    """Do not allow a catalogue key to misidentify the sidecar it indexes."""
    with pytest.raises(QualificationError, match="forged semantic digest"):
        validate_environment_catalog({"0" * 64: environment_record}, manifest)


def test_missing_dependency_profile_sidecar_keeps_missing_rows_attributable(
    manifest, environment_record
) -> None:
    """Refuse to invent provenance for an unexecuted floor-profile cell."""
    current_only = build_environment_catalog([environment_record], manifest)
    plan = expand_plan(manifest, "qualification")

    with pytest.raises(QualificationError, match="lacks dependency profile bambi-0.19"):
        aggregate_results(plan, [], manifest, "qualification", current_only)


def test_result_rejects_environment_from_wrong_scenario_profile(
    manifest, environment, make_result
) -> None:
    """Bind floor scenarios to the floor sidecar, not merely a known digest."""
    entry = next(
        item
        for item in expand_plan(manifest, "qualification")
        if item["scenario"].get("dependency_profile") == "bambi-0.19"
    )
    record = make_result(entry)
    current = next(
        item
        for item in environment.values()
        if item["dependency_profile"] == "current-resolved"
    )
    record["provenance"]["environment_sha256"] = environment_sha256(current, manifest)

    with pytest.raises(QualificationError, match="does not match its scenario"):
        validate_result_record(record, entry, environment, manifest)


def test_qualification_execution_requires_clean_git(manifest, environment) -> None:
    """Reject qualification evidence collected from a modified checkout."""
    dirty_current = _environment_record(manifest, "current-resolved", dirty=True)
    floor = next(
        item
        for item in environment.values()
        if item["dependency_profile"] == "bambi-0.19"
    )
    dirty_catalog = build_environment_catalog([dirty_current, floor], manifest)
    entry = next(
        item
        for item in expand_plan(manifest, "qualification")
        if item["scenario"].get("dependency_profile", "current-resolved")
        == "current-resolved"
    )
    record = _result(entry, dirty_catalog, manifest)

    with pytest.raises(QualificationError, match="clean git checkout"):
        validate_result_record(record, entry, dirty_catalog, manifest)


def test_float64_numpyro_requires_jax_x64(manifest, environment) -> None:
    """Prevent nominal float64 NumPyro evidence from a float32 JAX runtime."""
    no_x64_current = _environment_record(
        manifest, "current-resolved", jax_enable_x64=False
    )
    floor = next(
        item
        for item in environment.values()
        if item["dependency_profile"] == "bambi-0.19"
    )
    no_x64_catalog = build_environment_catalog([no_x64_current, floor], manifest)
    entry = next(
        item
        for item in expand_plan(manifest, "qualification")
        if item["scenario"]["sampler"] == "numpyro"
        and item["scenario"]["floatx"] == "float64"
    )
    record = _result(entry, no_x64_catalog, manifest)

    with pytest.raises(QualificationError, match="requires jax_enable_x64"):
        validate_result_record(record, entry, no_x64_catalog, manifest)


def test_cell_results_are_atomic_and_identity_checked(
    manifest, environment, make_result, tmp_path: Path
) -> None:
    """Publish a complete cell only after validating its planned identity."""
    entry = expand_plan(manifest, "smoke")[0]
    record = make_result(entry)

    path = write_cell_result(record, entry, tmp_path, environment, manifest)

    assert json.loads(path.read_text()) == record
    assert list(tmp_path.glob("*.tmp")) == []
    with pytest.raises(QualificationError, match="refusing to overwrite"):
        write_cell_result(record, entry, tmp_path, environment, manifest)
    assert json.loads(path.read_text()) == record
    changed = copy.deepcopy(record)
    changed["data_seed"] += 1
    with pytest.raises(QualificationError, match="data_seed"):
        validate_result_record(changed, entry, environment, manifest)


def test_result_validation_rejects_nonfinite_and_inconsistent_metrics(
    manifest, environment, make_result
) -> None:
    """Reject metrics that cannot be compared or contradict sampler counts."""
    entry = expand_plan(manifest, "smoke")[0]
    nonfinite = make_result(entry)
    nonfinite["metrics"]["hyper_rhat_max"] = float("nan")
    with pytest.raises(QualificationError, match="domain"):
        validate_result_record(nonfinite, entry, environment, manifest)

    inconsistent = make_result(entry)
    inconsistent["metrics"]["divergence_count"] = 1
    with pytest.raises(QualificationError, match="disagrees"):
        validate_result_record(inconsistent, entry, environment, manifest)

    wrong_draw_count = make_result(entry)
    wrong_draw_count["metrics"]["posterior_draw_count"] += 1
    with pytest.raises(QualificationError, match=r"planned chains \* draws"):
        validate_result_record(wrong_draw_count, entry, environment, manifest)


@pytest.mark.parametrize(
    ("metric", "invalid"),
    [
        ("compile_success", 1),
        ("divergence_count", -1),
        ("posterior_draw_count", 1.5),
        ("divergence_rate", 1.1),
        ("hyper_rhat_max", 0.0),
        ("hyper_ess_bulk_min", -0.1),
        ("hyper_ess_per_second_median", 0.0),
    ],
)
def test_result_metrics_enforce_typed_domains(
    metric, invalid, manifest, environment, make_result
) -> None:
    """Reject booleans, counts, fractions, and ratios outside their domains."""
    entry = expand_plan(manifest, "smoke")[0]
    record = make_result(entry)
    record["metrics"][metric] = invalid

    with pytest.raises(QualificationError, match="domain"):
        validate_result_record(record, entry, environment, manifest)


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("mystery_score", 1.0),
        ("truth_in_95pct_hdi_fraction", 0.95),
        ("recovery_abs_standardized_bias", 0.1),
        ("sbc_rank_ecdf_max_abs", 0.1),
        ("gradient_contract_pass", True),
        ("backend_parity_pass", True),
    ],
)
def test_result_rejects_unknown_or_obsolete_trusted_metrics(
    metric, value, manifest, environment, make_result
) -> None:
    """Accept raw evidence only, never a runner's self-reported verdict."""
    entry = expand_plan(manifest, "smoke")[0]
    record = make_result(entry)
    record["metrics"][metric] = value

    with pytest.raises(QualificationError, match="registered metric"):
        validate_result_record(record, entry, environment, manifest)


def test_recovery_result_requires_each_monitored_parameter_once(
    manifest, environment, make_result
) -> None:
    """Reject partial or duplicate per-parameter recovery evidence."""
    entry = next(
        item
        for item in expand_plan(manifest, "qualification")
        if item["scenario_id"] == "qual-pymc-lower-inside"
    )
    missing = make_result(entry)
    missing["parameter_summaries"].pop()
    with pytest.raises(QualificationError, match="every monitored parameter once"):
        validate_result_record(missing, entry, environment, manifest)

    duplicate = make_result(entry)
    duplicate["parameter_summaries"][-1] = copy.deepcopy(
        duplicate["parameter_summaries"][0]
    )
    with pytest.raises(QualificationError, match="every monitored parameter once"):
        validate_result_record(duplicate, entry, environment, manifest)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("family", "control", "family"),
        ("scenario_id", "another-scenario", "scenario_id"),
        ("replicate", 99, "replicate"),
    ],
)
def test_parameter_summary_identity_is_bound_to_its_cell(
    field, replacement, message, manifest, environment, make_result
) -> None:
    """Prevent a valid summary from being relabelled into another cell."""
    entry = next(
        item
        for item in expand_plan(manifest, "qualification")
        if item["scenario_id"] == "qual-pymc-lower-inside"
    )
    record = make_result(entry)
    record["parameter_summaries"][0][field] = replacement

    with pytest.raises(QualificationError, match=message):
        validate_result_record(record, entry, environment, manifest)


def test_sbc_rank_primitives_are_required_only_for_calibration(
    manifest, environment, make_result
) -> None:
    """Require randomized ranks for SBC and forbid them for fixed recovery."""
    plan = expand_plan(manifest, "qualification")
    fixed_entry = next(
        item for item in plan if item["scenario_id"] == "qual-pymc-lower-inside"
    )
    fixed = make_result(fixed_entry)
    fixed["parameter_summaries"][0].update(
        {
            "rank_less": 0,
            "rank_equal": 0,
            "rank_tie_index": 0,
            "rank": 0,
            "rank_draw_count": manifest["analysis_policy"]["sbc_rank_draw_count"],
        }
    )
    with pytest.raises(QualificationError, match="must not contain SBC ranks"):
        validate_result_record(fixed, fixed_entry, environment, manifest)

    calibration_entry = next(
        item for item in plan if item["scenario_id"] == "calib-pymc-lower-outside"
    )
    calibration = make_result(calibration_entry)
    for field in (
        "rank_less",
        "rank_equal",
        "rank_tie_index",
        "rank",
        "rank_draw_count",
    ):
        calibration["parameter_summaries"][0].pop(field)
    with pytest.raises(QualificationError, match="requires SBC rank primitives"):
        validate_result_record(calibration, calibration_entry, environment, manifest)


def test_sbc_tie_rank_is_bound_to_the_frozen_plan_seed(
    manifest, environment, make_result
) -> None:
    """Reject a producer-selected tie rank even when its arithmetic is consistent."""
    entry = next(
        item
        for item in expand_plan(manifest, "qualification")
        if item["scenario_id"] == "calib-pymc-lower-outside"
    )
    record = make_result(entry)
    summary = record["parameter_summaries"][0]
    expected = derive_sbc_rank_tie_index(
        tie_seed=entry["sbc_tie_seed"],
        family="candidate",
        scenario_id=entry["scenario_id"],
        parameter_id=summary["parameter_id"],
        replicate=entry["replicate"],
        rank_less=2,
        rank_equal=3,
        rank_draw_count=manifest["analysis_policy"]["sbc_rank_draw_count"],
    )
    forged = (expected + 1) % 4
    summary.update(
        {
            "rank_less": 2,
            "rank_equal": 3,
            "rank_tie_index": forged,
            "rank": 2 + forged,
        }
    )

    with pytest.raises(QualificationError, match="deterministic tie index"):
        validate_result_record(record, entry, environment, manifest)


def test_nonrecovery_and_failed_cells_forbid_parameter_summaries(
    manifest, environment, make_result
) -> None:
    """Keep diagnostic and unsuccessful cells out of recovery aggregation."""
    nonrecovery_entry = expand_plan(manifest, "smoke")[0]
    nonrecovery = make_result(nonrecovery_entry)
    nonrecovery["parameter_summaries"] = [{}]
    with pytest.raises(QualificationError, match="require empty"):
        validate_result_record(nonrecovery, nonrecovery_entry, environment, manifest)

    failed_entry = next(
        item
        for item in expand_plan(manifest, "qualification")
        if item["scenario_id"] == "qual-pymc-lower-inside"
    )
    failed = make_result(failed_entry, status="failed", metrics={})
    failed["parameter_summaries"] = [{}]
    with pytest.raises(QualificationError, match="require empty"):
        validate_result_record(failed, failed_entry, environment, manifest)


def test_result_provenance_is_bound_to_environment_and_actual_start(
    manifest, environment, make_result
) -> None:
    """Reject forged checkout identity and unsafe or unpaired start references."""
    entry = expand_plan(manifest, "smoke")[0]
    forged_commit = make_result(entry)
    forged_commit["provenance"]["git_commit"] = "another-commit"
    with pytest.raises(QualificationError, match="git_commit does not match"):
        validate_result_record(forged_commit, entry, environment, manifest)

    forged_environment = make_result(entry)
    forged_environment["provenance"]["environment_sha256"] = "0" * 64
    with pytest.raises(QualificationError, match="absent from the catalog"):
        validate_result_record(forged_environment, entry, environment, manifest)

    unsafe_artifact = make_result(entry)
    unsafe_artifact["provenance"]["actual_start_artifact"] = "../start.json"
    with pytest.raises(QualificationError, match="canonical relative path"):
        validate_result_record(unsafe_artifact, entry, environment, manifest)

    missing_digest = make_result(entry)
    missing_digest["provenance"]["actual_start_sha256"] = None
    with pytest.raises(QualificationError, match="provided together"):
        validate_result_record(missing_digest, entry, environment, manifest)


def test_aggregation_preserves_failures_and_materializes_missing_rows(
    manifest, environment, make_result, tmp_path: Path
) -> None:
    """Account for every planned cell instead of dropping failed experiments."""
    plan = expand_plan(manifest, "smoke")
    complete = make_result(plan[0])
    failed = make_result(plan[1], status="failed", metrics={})

    aggregate = aggregate_results(
        plan, [failed, complete], manifest, "smoke", environment
    )

    assert [record["cell_id"] for record in aggregate] == [
        entry["cell_id"] for entry in plan
    ]
    assert [record["execution_status"] for record in aggregate[:3]] == [
        "completed",
        "failed",
        "missing",
    ]
    assert sum(record["execution_status"] == "missing" for record in aggregate) == 8
    assert aggregate[0]["provenance"]["actual_start_artifact"].startswith("starts/")
    assert len(aggregate[0]["provenance"]["actual_start_sha256"]) == 64
    assert "actual_start_values" not in aggregate[0]["provenance"]
    paths = write_aggregate(aggregate, tmp_path)
    assert load_jsonl(paths[0]) == aggregate
    csv_text = paths[1].read_text()
    assert "MissingResult" in csv_text
    assert "sampler failed" in csv_text


def test_aggregation_rejects_duplicate_or_unplanned_results(
    manifest, environment, make_result
) -> None:
    """Do not let reruns overwrite evidence or foreign cells enter a study."""
    plan = expand_plan(manifest, "smoke")
    record = make_result(plan[0])
    with pytest.raises(QualificationError, match="duplicate"):
        aggregate_results(plan, [record, record], manifest, "smoke", environment)

    foreign = copy.deepcopy(record)
    foreign["cell_id"] = "foreign--replicate-00"
    with pytest.raises(QualificationError, match="unplanned"):
        aggregate_results(plan, [foreign], manifest, "smoke", environment)


def test_unavailable_metrics_are_reasoned_and_make_evidence_incomplete(
    manifest, environment, make_result
) -> None:
    """Distinguish an explained unavailable metric from an omitted metric."""
    plan = expand_plan(manifest, "smoke")
    records = [make_result(entry) for entry in plan]
    records[0]["metrics"].pop("gradient_finite")
    records[0]["unavailable_metrics"] = {
        "gradient_finite": "gradient evaluation is unavailable on this worker"
    }

    assessment = assess_results(records, plan, manifest, "smoke", environment)

    assert assessment["outcome"] == "incomplete"
    assert assessment["unavailable_metrics"] == [
        {
            "cell_id": plan[0]["cell_id"],
            "metric": "gradient_finite",
            "reason": "gradient evaluation is unavailable on this worker",
        }
    ]
    assert not any(
        item.endswith(":gradient_finite") for item in assessment["missing_metrics"]
    )

    overlap = make_result(plan[0])
    overlap["unavailable_metrics"] = {"gradient_finite": "claimed unavailable"}
    with pytest.raises(QualificationError, match="both available and unavailable"):
        validate_result_record(overlap, plan[0], environment, manifest)

    empty_reason = make_result(plan[0])
    empty_reason["metrics"].pop("gradient_finite")
    empty_reason["unavailable_metrics"] = {"gradient_finite": " "}
    with pytest.raises(QualificationError, match="non-empty reason"):
        validate_result_record(empty_reason, plan[0], environment, manifest)


def test_efficiency_ratios_cannot_be_self_reported(
    manifest, environment, make_result
) -> None:
    """Reject candidate or control claims about their own comparative efficiency."""
    plan = expand_plan(manifest, "qualification")
    candidate_entry = next(
        entry
        for entry in plan
        if entry["scenario"]["purpose"] == "candidate"
        and entry["scenario"].get("calibration_kind") is None
    )
    control_entry = next(
        entry
        for entry in plan
        if entry["scenario_id"] == candidate_entry["scenario"]["control_id"]
    )
    for entry in (candidate_entry, control_entry):
        record = make_result(entry)
        record["metrics"]["ess_per_second_slowdown"] = 0.1
        record["metrics"]["leapfrog_cost_ratio"] = 0.1
        with pytest.raises(QualificationError, match="registered metric"):
            validate_result_record(record, entry, environment, manifest)


def test_assessor_computes_median_candidate_control_ratios(
    manifest, environment, make_result
) -> None:
    """Derive per-pair ratios and gate their scenario median from raw metrics."""
    plan = expand_plan(manifest, "qualification")
    records = [make_result(entry) for entry in plan]
    candidate = next(
        scenario
        for scenario in manifest["scenarios"]
        if scenario["tier"] == "qualification"
        and scenario["purpose"] == "candidate"
        and scenario.get("calibration_kind") is None
    )
    by_cell = {
        (record["scenario_id"], record["replicate"]): record for record in records
    }
    ratios = [2.0, 3.0, 4.0, 5.0, 6.0]
    for replicate, ratio in enumerate(ratios):
        candidate_record = by_cell[(candidate["scenario_id"], replicate)]
        control_record = by_cell[(candidate["control_id"], replicate)]
        candidate_record["metrics"]["hyper_ess_per_second_median"] = 10.0
        control_record["metrics"]["hyper_ess_per_second_median"] = 10.0 * ratio
        candidate_record["metrics"][
            "hyper_leapfrog_steps_per_effective_sample_median"
        ] = 2.0 * ratio
        control_record["metrics"][
            "hyper_leapfrog_steps_per_effective_sample_median"
        ] = 2.0

    assessment = assess_results(records, plan, manifest, "qualification", environment)
    median_checks = {
        check["metric"]: check
        for check in assessment["checks"]
        if check["scope"] == "control_paired_median"
        and check["scenario_id"] == candidate["scenario_id"]
    }

    assert median_checks["ess_per_second_slowdown"]["actual"] == 4.0
    assert median_checks["leapfrog_cost_ratio"]["actual"] == 4.0
    assert median_checks["ess_per_second_slowdown"]["paired_replicates"] == 5
    assert all(check["passed"] for check in median_checks.values())


def test_single_pair_above_ten_is_an_immediate_efficiency_blocker(
    manifest, environment, make_result
) -> None:
    """Block a severe pair even when the five-replicate median remains acceptable."""
    plan = expand_plan(manifest, "qualification")
    records = [make_result(entry) for entry in plan]
    candidate = next(
        scenario
        for scenario in manifest["scenarios"]
        if scenario["tier"] == "qualification"
        and scenario["purpose"] == "candidate"
        and scenario.get("calibration_kind") is None
    )
    by_cell = {
        (record["scenario_id"], record["replicate"]): record for record in records
    }
    candidate_record = by_cell[(candidate["scenario_id"], 0)]
    control_record = by_cell[(candidate["control_id"], 0)]
    candidate_record["metrics"]["hyper_ess_per_second_median"] = 1.0
    control_record["metrics"]["hyper_ess_per_second_median"] = 11.0

    assessment = assess_results(records, plan, manifest, "qualification", environment)
    median_check = next(
        check
        for check in assessment["checks"]
        if check["scope"] == "control_paired_median"
        and check["scenario_id"] == candidate["scenario_id"]
        and check["metric"] == "ess_per_second_slowdown"
    )

    assert median_check["actual"] == 1.0
    assert median_check["passed"] is True
    assert assessment["outcome"] == "fail"
    assert any(
        blocker.endswith("ess_per_second_slowdown:control-pair")
        for blocker in assessment["blockers"]
    )


@pytest.mark.parametrize(
    ("control_status", "expected_outcome"),
    [("missing", "incomplete"), ("failed", "fail")],
)
def test_missing_or_failed_control_prevents_qualification(
    control_status, expected_outcome, manifest, environment, make_result
) -> None:
    """Never qualify a candidate whose same-replicate control has no usable result."""
    plan = expand_plan(manifest, "qualification")
    records = [make_result(entry) for entry in plan]
    candidate_entry = next(
        entry
        for entry in plan
        if entry["scenario"]["purpose"] == "candidate"
        and entry["scenario"].get("calibration_kind") is None
    )
    control_cell_id = next(
        entry["cell_id"]
        for entry in plan
        if entry["scenario_id"] == candidate_entry["scenario"]["control_id"]
        and entry["replicate"] == candidate_entry["replicate"]
    )
    if control_status == "missing":
        records = [record for record in records if record["cell_id"] != control_cell_id]
    else:
        control_index = next(
            index
            for index, record in enumerate(records)
            if record["cell_id"] == control_cell_id
        )
        control_entry = next(
            entry for entry in plan if entry["cell_id"] == control_cell_id
        )
        records[control_index] = make_result(control_entry, status="failed", metrics={})
    aggregate = aggregate_results(plan, records, manifest, "qualification", environment)

    assessment = assess_results(aggregate, plan, manifest, "qualification", environment)

    assert assessment["outcome"] == expected_outcome
    assert assessment["qualifies_default"] is False


@pytest.mark.parametrize(
    ("actual", "condition", "expected"),
    [
        (1.01, {"comparator": "lt", "value": 1.01}, False),
        (1.009999, {"comparator": "lt", "value": 1.01}, True),
        (0.001, {"comparator": "lt", "value": 0.001}, False),
        (0.000999, {"comparator": "lt", "value": 0.001}, True),
        (0.95, {"comparator": "ge", "value": 0.95}, True),
        (True, {"comparator": "eq", "value": True}, True),
        (1, {"comparator": "eq", "value": True}, False),
    ],
)
def test_threshold_comparators_have_explicit_boundary_semantics(
    actual, condition, expected
) -> None:
    """Distinguish strict from inclusive thresholds at their exact boundaries."""
    assert compare_threshold(actual, condition) is expected


def test_complete_synthetic_evidence_waits_for_raw_chain_backend_check(
    manifest, environment, make_result
) -> None:
    """Keep the gate incomplete until combined-chain backend evidence exists."""
    plan = expand_plan(manifest, "qualification")
    records = [make_result(entry) for entry in plan]
    used_profiles = {
        environment[record["provenance"]["environment_sha256"]]["dependency_profile"]
        for record in records
    }

    assessment = assess_results(records, plan, manifest, "qualification", environment)

    assert used_profiles == {"current-resolved", "bambi-0.19"}
    assert assessment["outcome"] == "incomplete"
    assert assessment["qualifies_default"] is False
    assert assessment["counts"]["planned"] == 720
    assert assessment["counts"]["completed"] == 720
    assert assessment["counts"]["failed"] == 0
    assert assessment["counts"]["missing"] == 0
    assert assessment["counts"]["checks"] > 600
    assert assessment["counts"]["failed_checks"] == 0
    assert assessment["missing_metrics"] == []
    assert assessment["blockers"] == []
    assert assessment["pending_evidence"]
    assert {item["metric"] for item in assessment["pending_evidence"]} == {
        "backend_combined_rank_rhat_max"
    }


def test_assessor_does_not_dilute_recovery_or_backend_failures(
    manifest, environment, make_result
) -> None:
    """Evaluate each scenario/parameter/family and paired backend independently."""
    plan = expand_plan(manifest, "qualification")
    records = [make_result(entry) for entry in plan]
    for record in records:
        for summary in record["parameter_summaries"]:
            if (
                record["scenario_id"] == "qual-pymc-lower-inside"
                and summary["parameter_id"] == "group_location"
            ):
                summary["posterior_mean"] = 0.6
            if (
                record["scenario_id"] == "calib-pymc-lower-outside"
                and summary["parameter_id"] == "group_scale"
            ):
                summary.update(
                    {
                        "rank_less": 0,
                        "rank_equal": 0,
                        "rank_tie_index": 0,
                        "rank": 0,
                    }
                )
            if (
                record["scenario_id"] == "calib-pymc-lower-outside"
                and summary["parameter_id"] == "group_middle"
            ):
                summary.update(
                    {
                        "q025": 0.5,
                        "q05": 0.75,
                        "q50": 1.0,
                        "q95": 1.5,
                        "q975": 2.0,
                    }
                )
            if (
                record["scenario_id"] == "qual-hssm-lba2-near-numpyro"
                and record["replicate"] == 0
                and summary["parameter_id"] == "group_first"
            ):
                summary["posterior_mean"] = 1.0

    assessment = assess_results(records, plan, manifest, "qualification", environment)

    assert assessment["outcome"] == "fail"
    candidate_bias = next(
        check
        for check in assessment["checks"]
        if check["scope"] == "fixed_recovery_bias"
        and check["scenario_id"] == "qual-pymc-lower-inside"
        and check["parameter_id"] == "group_location"
    )
    control_bias = next(
        check
        for check in assessment["checks"]
        if check["scope"] == "fixed_recovery_bias"
        and check["scenario_id"] == "qual-pymc-lower-inside-control"
        and check["parameter_id"] == "group_location"
    )
    assert candidate_bias["family"] == "candidate"
    assert candidate_bias["passed"] is False
    assert candidate_bias["actual"] == pytest.approx(0.6)
    assert control_bias["family"] == "control"
    assert control_bias["passed"] is True

    bad_rank = next(
        check
        for check in assessment["checks"]
        if check["scope"] == "calibration_sbc_rank"
        and check["scenario_id"] == "calib-pymc-lower-outside"
        and check["parameter_id"] == "group_scale"
    )
    untouched_rank = next(
        check
        for check in assessment["checks"]
        if check["scope"] == "calibration_sbc_rank"
        and check["scenario_id"] == "calib-pymc-two-sided-near"
        and check["parameter_id"] == "group_scale"
    )
    assert bad_rank["passed"] is False
    assert untouched_rank["passed"] is True

    bad_coverage = [
        check
        for check in assessment["checks"]
        if check["scope"] == "calibration_coverage"
        and check["scenario_id"] == "calib-pymc-lower-outside"
        and check["parameter_id"] == "group_middle"
    ]
    untouched_coverage = [
        check
        for check in assessment["checks"]
        if check["scope"] == "calibration_coverage"
        and check["scenario_id"] == "calib-pymc-two-sided-near"
        and check["parameter_id"] == "group_middle"
    ]
    assert len(bad_coverage) == len(untouched_coverage) == 2
    assert all(check["passed"] is False for check in bad_coverage)
    assert all(check["passed"] is True for check in untouched_coverage)

    backend_check = next(
        check
        for check in assessment["checks"]
        if check["scope"] == "backend_pair"
        and check["metric"] == "posterior_mean_mcse_z"
        and check["posterior_pair_id"] == "hssm-lba2-near-truncated"
        and check["parameter_id"] == "group_first"
        and check["cell_id"].endswith("replicate-00")
    )
    assert backend_check["actual"] == pytest.approx(1 / (0.1**2 + 0.1**2) ** 0.5)
    assert (
        backend_check["threshold"]
        == manifest["analysis_policy"]["backend_posterior_mean_mcse_z_max"]
    )
    assert backend_check["passed"] is False


def test_raw_gradient_contract_is_assessor_derived_and_classified(
    manifest, environment, make_result
) -> None:
    """Require raw replicate-zero errors and label their failures correctly."""
    plan = expand_plan(manifest, "smoke")
    records = [make_result(entry) for entry in plan]
    metric = "finite_difference_gradient_abs_error_max"
    condition = qualification._gradient_contract_conditions(
        records[0], plan[0]["scenario"], manifest["analysis_policy"]
    )[metric]
    records[0]["metrics"][metric] = condition["value"] * 2

    failed = assess_results(records, plan, manifest, "smoke", environment)

    contract_check = next(
        check
        for check in failed["checks"]
        if check["scope"] == "gradient_contract"
        and check["cell_id"] == records[0]["cell_id"]
        and check["metric"] == metric
    )
    assert failed["outcome"] == "screening-fail"
    assert contract_check["passed"] is False
    assert contract_check["failure_class"] == "likelihood/backend-contract"

    missing_records = [make_result(entry) for entry in plan]
    missing_records[0]["metrics"].pop(metric)
    incomplete = assess_results(missing_records, plan, manifest, "smoke", environment)
    assert incomplete["outcome"] == "incomplete"
    assert f"{records[0]['cell_id']}:{metric}" in incomplete["missing_metrics"]


def test_qualification_distinguishes_incomplete_from_failed_evidence(
    manifest, environment, make_result
) -> None:
    """Report missing work as incomplete and observed sampler failure as failure."""
    plan = expand_plan(manifest, "qualification")
    records = [make_result(entry) for entry in plan]
    missing = aggregate_results(
        plan, records[:-1], manifest, "qualification", environment
    )

    incomplete = assess_results(missing, plan, manifest, "qualification", environment)

    assert incomplete["outcome"] == "incomplete"
    assert incomplete["qualifies_default"] is False
    assert incomplete["missing_cells"] == [plan[-1]["cell_id"]]

    failed_records = list(records)
    failed_records[0] = make_result(plan[0], status="failed", metrics={})
    failed = assess_results(
        failed_records, plan, manifest, "qualification", environment
    )

    assert failed["outcome"] == "fail"
    assert failed["qualifies_default"] is False
    assert failed["failed_cells"] == [plan[0]["cell_id"]]


def test_qualification_applies_immediate_and_repeated_no_go_rules(
    manifest, environment, make_result
) -> None:
    """Fail on one severe geometry event or two repeated diagnostic collapses."""
    plan = expand_plan(manifest, "qualification")
    immediate_records = [make_result(entry) for entry in plan]
    immediate_records[0]["metrics"].update(divergence_count=40, divergence_rate=0.01)

    immediate = assess_results(
        immediate_records, plan, manifest, "qualification", environment
    )

    assert immediate["outcome"] == "fail"
    assert any("divergence_rate" in blocker for blocker in immediate["blockers"])

    repeated_records = [make_result(entry) for entry in plan]
    same_scenario = [
        index
        for index, entry in enumerate(plan)
        if entry["scenario_id"] == plan[0]["scenario_id"]
    ][:2]
    for index in same_scenario:
        repeated_records[index]["metrics"]["hyper_rhat_max"] = 1.06

    repeated = assess_results(
        repeated_records, plan, manifest, "qualification", environment
    )

    assert repeated["outcome"] == "fail"
    assert f"{plan[0]['scenario_id']}:repeated:hyper_rhat_max" in repeated["blockers"]


def test_smoke_pass_never_qualifies_the_default(
    manifest, environment, make_result
) -> None:
    """Keep a successful cheap screen distinct from the primary release gate."""
    plan = expand_plan(manifest, "smoke")
    records = [make_result(entry) for entry in plan]

    assessment = assess_results(records, plan, manifest, "smoke", environment)

    assert assessment["outcome"] == "screening-pass"
    assert assessment["qualifies_default"] is False


def test_aggregate_cli_accepts_repeated_environment_sidecars(
    manifest, environment, tmp_path: Path
) -> None:
    """Materialize profile-attributed missing rows from repeated CLI sidecars."""
    environment_paths = []
    for item in environment.values():
        path = tmp_path / f"environment-{item['dependency_profile']}.json"
        path.write_text(json.dumps(item))
        environment_paths.append(path)
    results_dir = tmp_path / "cell-results"
    results_dir.mkdir()
    output_dir = tmp_path / "aggregate"

    assert (
        main(
            [
                "aggregate",
                "--tier",
                "smoke",
                "--results-dir",
                str(results_dir),
                "--output-dir",
                str(output_dir),
                "--environment",
                str(environment_paths[0]),
                "--environment",
                str(environment_paths[1]),
            ]
        )
        == 0
    )
    records = load_jsonl(output_dir / "results.jsonl")
    used_profiles = {
        environment[record["provenance"]["environment_sha256"]]["dependency_profile"]
        for record in records
    }
    assert used_profiles == {"current-resolved", "bambi-0.19"}


def test_assess_cli_returns_nonzero_for_incomplete_and_diagnostic_failure(
    manifest, environment, make_result, tmp_path: Path
) -> None:
    """Make incomplete and failed evidence visible to shell automation."""
    environment_paths = []
    for item in environment.values():
        path = tmp_path / f"environment-{item['dependency_profile']}.json"
        path.write_text(json.dumps(item))
        environment_paths.append(path)

    smoke_plan = expand_plan(manifest, "smoke")
    incomplete = aggregate_results(smoke_plan, [], manifest, "smoke", environment)
    incomplete_results = write_aggregate(incomplete, tmp_path / "incomplete")[0]
    assert (
        main(
            [
                "assess",
                "--tier",
                "smoke",
                "--results",
                str(incomplete_results),
                "--environment",
                str(environment_paths[0]),
                "--environment",
                str(environment_paths[1]),
                "--output",
                str(tmp_path / "incomplete-assessment.json"),
            ]
        )
        == 1
    )

    stress_plan = expand_plan(manifest, "stress")
    stress_records = [make_result(entry) for entry in stress_plan]
    stress_records[0] = make_result(stress_plan[0], status="failed", metrics={})
    failed = aggregate_results(
        stress_plan, stress_records, manifest, "stress", environment
    )
    failed_results = write_aggregate(failed, tmp_path / "failed")[0]
    assert (
        main(
            [
                "assess",
                "--tier",
                "stress",
                "--results",
                str(failed_results),
                "--environment",
                str(environment_paths[0]),
                "--environment",
                str(environment_paths[1]),
                "--output",
                str(tmp_path / "failed-assessment.json"),
            ]
        )
        == 1
    )

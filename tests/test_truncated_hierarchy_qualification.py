"""Tests for the no-sampling bounded-hierarchy qualification contract."""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

import scripts.truncated_hierarchy_qualification as qualification
from scripts.truncated_hierarchy_qualification import (
    QualificationError,
    aggregate_results,
    assess_results,
    build_environment_catalog,
    compare_threshold,
    derive_data_stream_seed,
    derive_numpyro_chain_keys,
    derive_seed,
    derive_start_seed,
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
    verify_result_artifacts,
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
        "sampling_elapsed_seconds": 10.0,
        "step_size_median": 0.1,
        "gradient_evaluation_count": 10_000,
        "leapfrog_step_count": 10_000,
        "hyper_ess_per_second_median": 100.0,
        "hyper_leapfrog_steps_per_effective_sample_median": 2.0,
    }
    contract_metrics = qualification._gradient_contract_required_metrics(
        {"replicate": entry["replicate"]},
        entry["scenario"],
        manifest["analysis_policy"],
    )
    metrics.update(dict.fromkeys(contract_metrics, 0.0))
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
    scenario = entry["scenario"]
    paired = bool(
        scenario["tier"] == "qualification"
        and scenario["purpose"] in {"candidate", "control"}
        and scenario.get("calibration_kind") is None
    )
    pair_execution_id = None
    pair_position = None
    if paired:
        if scenario["purpose"] == "candidate":
            candidate_id = entry["scenario_id"]
        else:
            candidate_id = next(
                item["scenario_id"]
                for item in manifest["scenarios"]
                if item.get("control_id") == entry["scenario_id"]
            )
        pair_key = f"{candidate_id}:{entry['replicate']}"
        pair_execution_id = hashlib.sha256(f"pair:{pair_key}".encode()).hexdigest()
        candidate_first = entry["replicate"] % 2 == 0
        pair_position = int(
            (scenario["purpose"] == "control" and candidate_first)
            or (scenario["purpose"] == "candidate" and not candidate_first)
        )
        worker_identity = hashlib.sha256(f"worker:{pair_key}".encode()).hexdigest()
    else:
        worker_identity = hashlib.sha256(
            f"worker:{entry['cell_id']}".encode()
        ).hexdigest()
    execution_attempt_id = hashlib.sha256(
        f"attempt:{entry['cell_id']}".encode()
    ).hexdigest()
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
        "truth_seed": entry["truth_seed"],
        "group_seed": entry["group_seed"],
        "observation_seed": entry["observation_seed"],
        "initialization_seed": entry["initialization_seed"],
        "start_seeds": list(entry["start_seeds"]),
        "sampler_seed": entry["sampler_seed"],
        "sbc_draw_seed": entry["sbc_draw_seed"],
        "sbc_tie_seed": entry["sbc_tie_seed"],
        "chain_seeds": list(entry["chain_seeds"]),
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
            "runner_version": 2,
            "sampler": entry["scenario"]["sampler"],
            "device": "cpu",
            "floatx": entry["scenario"]["floatx"],
            "pytensor_floatx": entry["scenario"]["floatx"],
            "jax_enable_x64": entry["scenario"]["floatx"] == "float64",
            "data_artifact": (
                f"data/{entry['scenario']['data_id']}-r{entry['replicate']}.json"
                if status == "completed"
                else None
            ),
            "data_sha256": "a" * 64 if status == "completed" else None,
            "effective_numpyro_chain_keys": (
                [
                    list(key)
                    for key in derive_numpyro_chain_keys(
                        entry["sampler_seed"], entry["scenario"]["chains"]
                    )
                ]
                if status == "completed" and entry["scenario"]["sampler"] == "numpyro"
                else None
            ),
            "actual_start_artifact": (
                f"starts/{entry['cell_id']}.json" if status == "completed" else None
            ),
            "actual_start_sha256": "b" * 64 if status == "completed" else None,
            "raw_chain_artifact": (
                f"chains/{entry['cell_id']}.nc" if status == "completed" else None
            ),
            "raw_chain_sha256": "c" * 64 if status == "completed" else None,
            "git_commit": environment["git"]["commit"],
            "environment_sha256": environment_sha256(environment, manifest),
            "execution_attempt_id": execution_attempt_id,
            "pair_execution_id": pair_execution_id,
            "pair_position": pair_position,
            "worker_identity_sha256": worker_identity,
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
        "schema_version": 2,
        "study_id": manifest["study_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "runner_version": 2,
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


def _materialize_result_artifacts(
    record: dict[str, object], artifact_root: Path
) -> None:
    """Write deterministic test artifacts and bind their exact byte digests."""
    provenance = record["provenance"]
    assert isinstance(provenance, dict)
    for artifact_field, digest_field in (
        ("data_artifact", "data_sha256"),
        ("actual_start_artifact", "actual_start_sha256"),
        ("raw_chain_artifact", "raw_chain_sha256"),
    ):
        relative = provenance[artifact_field]
        if relative is None:
            continue
        assert isinstance(relative, str)
        payload = f"{artifact_field}\n".encode()
        path = artifact_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        provenance[digest_field] = hashlib.sha256(payload).hexdigest()


def _write_standardized_test_chain(
    record: dict[str, object],
    entry: dict[str, object],
    artifact_root: Path,
    *,
    seed: int,
    group_middle_shift: float = 0.0,
) -> None:
    """Replace the placeholder chain with one valid auditable NetCDF artifact."""
    scenario = entry["scenario"]
    assert isinstance(scenario, dict)
    chains = int(scenario["chains"])
    draws = int(scenario["draws"])
    n_groups = int(scenario["n_groups"])
    rng = np.random.default_rng(seed)
    group_effect = rng.normal(size=(chains, draws, n_groups))
    group_indices = [int(index) for index in scenario["group_indices"]]
    group_effect[..., group_indices[1]] += group_middle_shift
    posterior = xr.Dataset(
        {
            "group_location": (("chain", "draw"), rng.normal(size=(chains, draws))),
            "group_scale": (("chain", "draw"), rng.normal(size=(chains, draws))),
            "group_effect": (("chain", "draw", "group"), group_effect),
            "group_first": (
                ("chain", "draw"),
                group_effect[..., group_indices[0]],
            ),
            "group_middle": (
                ("chain", "draw"),
                group_effect[..., group_indices[1]],
            ),
            "group_last": (
                ("chain", "draw"),
                group_effect[..., group_indices[2]],
            ),
        },
        coords={
            "chain": np.arange(chains),
            "draw": np.arange(draws),
            "group": np.arange(n_groups),
        },
    )
    sample_stats = xr.Dataset(
        {
            "diverging": (
                ("chain", "draw"),
                np.zeros((chains, draws), dtype=bool),
            ),
            "energy": (("chain", "draw"), rng.normal(size=(chains, draws))),
            "tree_depth": (
                ("chain", "draw"),
                np.ones((chains, draws), dtype=np.int64),
            ),
            "n_steps": (
                ("chain", "draw"),
                np.ones((chains, draws), dtype=np.int64),
            ),
            "step_size": (
                ("chain", "draw"),
                np.full((chains, draws), 0.1),
            ),
            "acceptance_rate": (
                ("chain", "draw"),
                np.full((chains, draws), 0.9),
            ),
        },
        coords={"chain": np.arange(chains), "draw": np.arange(draws)},
    )
    chain = xr.DataTree.from_dict(
        {"posterior": posterior, "sample_stats": sample_stats}
    )
    provenance = record["provenance"]
    assert isinstance(provenance, dict)
    relative = provenance["raw_chain_artifact"]
    assert isinstance(relative, str)
    path = artifact_root / relative
    chain.to_netcdf(path)
    provenance["raw_chain_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


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
        "05d8be96204f124abe723fb38f65080631d2cdcb7c9b776428201ee66045e15e"
    )
    assert {scenario["tier"] for scenario in manifest["scenarios"]} == {
        "smoke",
        "qualification",
        "stress",
    }
    assert len(expand_plan(manifest, "smoke")) == 10
    assert len(expand_plan(manifest, "qualification")) == 720
    assert len(expand_plan(manifest, "stress")) == 18


def test_runtime_floors_provide_native_erfcx_for_numpyro() -> None:
    """Keep TruncatedNormal JAXification off the incompatible TFP fallback."""
    pyproject = tomllib.loads(
        (qualification.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert "jax>=0.11.0" in pyproject["project"]["dependencies"]
    assert "pytensor>=3.2.4" in pyproject["project"]["dependencies"]
    assert pyproject["project"]["optional-dependencies"]["cuda12"] == [
        "jax[cuda12]>=0.11.0"
    ]
    assert pyproject["project"]["optional-dependencies"]["cuda13"] == [
        "jax[cuda13]>=0.11.0"
    ]


def test_v1_manifest_is_explicitly_superseded() -> None:
    """Never execute the ambiguous v1 design through the v2 runner."""
    v1_path = qualification.REPO_ROOT / "benchmarks/specs/truncated_hierarchy_v1.json"

    with pytest.raises(QualificationError, match="v1 is superseded"):
        load_manifest(v1_path)


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

    obsolete_anchor = copy.deepcopy(manifest)
    obsolete_anchor["scenarios"][0]["anchor"] = "outside"
    with pytest.raises(QualificationError, match="unknown.*anchor"):
        validate_manifest(obsolete_anchor)


def test_v2_scenarios_freeze_prior_truth_and_group_coordinate_contracts(
    manifest,
) -> None:
    """Make every prior base, truth mode, and monitored group index executable."""
    for scenario in manifest["scenarios"]:
        lower, upper = scenario["lower"], scenario["upper"]
        expected_location = (
            (lower + upper) / 2
            if scenario["prior"] == "truncated_normal"
            and lower is not None
            and upper is not None
            else 0.0
        )
        assert scenario["prior_hyper_location"] == expected_location
        assert scenario["group_indices"] == [
            0,
            scenario["n_groups"] // 2,
            scenario["n_groups"] - 1,
        ]
        if scenario["truth_kind"] == "fixed":
            assert scenario["truth_group_location"] is not None
            assert scenario["truth_group_scale"] > 0
            assert scenario.get("calibration_kind") is None
        else:
            assert scenario["truth_regime"] == "prior_predictive"
            assert scenario["truth_group_location"] is None
            assert scenario["truth_group_scale"] is None
            assert scenario["calibration_kind"] == "sbc"

    wrong_prior_base = copy.deepcopy(manifest)
    wrong_prior_base["scenarios"][0]["prior_hyper_location"] = 0.2
    with pytest.raises(QualificationError, match="frozen prior rule"):
        validate_manifest(wrong_prior_base)

    missing_fixed_truth = copy.deepcopy(manifest)
    missing_fixed_truth["scenarios"][0]["truth_group_location"] = None
    with pytest.raises(QualificationError, match="truth_group_location must be finite"):
        validate_manifest(missing_fixed_truth)

    wrong_group_index = copy.deepcopy(manifest)
    wrong_group_index["scenarios"][0]["group_indices"][1] = 1
    with pytest.raises(QualificationError, match="group_indices must remain"):
        validate_manifest(wrong_group_index)


def test_representative_v2_truth_anchors_are_exact(manifest) -> None:
    """Pin boundary, interior, narrow, and prior-predictive truth semantics."""
    scenarios = {item["scenario_id"]: item for item in manifest["scenarios"]}
    fields = (
        "bound_kind",
        "lower",
        "upper",
        "prior_hyper_location",
        "truth_kind",
        "truth_regime",
        "truth_boundary",
        "truth_group_location",
        "truth_group_scale",
        "group_indices",
        "data_id",
    )
    expected = {
        "smoke-pymc-lower-outside": (
            "lower",
            0.2,
            None,
            0.0,
            "fixed",
            "near_lower",
            "lower",
            0.23,
            0.3,
            [0, 2, 3],
            "smoke-toy-lower-outside",
        ),
        "smoke-pymc-lower-inside": (
            "lower",
            -0.2,
            None,
            0.0,
            "fixed",
            "interior",
            None,
            0.7,
            0.3,
            [0, 10, 19],
            "toy-lower-inside",
        ),
        "smoke-pymc-narrow-midpoint": (
            "narrow",
            0.49,
            0.51,
            0.5,
            "fixed",
            "near_lower",
            "lower",
            0.495,
            0.05,
            [0, 10, 19],
            "smoke-toy-narrow-midpoint",
        ),
        "calib-pymc-two-sided-midpoint": (
            "two_sided",
            0.1,
            0.9,
            0.5,
            "prior_predictive",
            "prior_predictive",
            None,
            None,
            None,
            [0, 10, 19],
            "calib-pymc-two-sided-midpoint",
        ),
    }
    for scenario_id, expected_values in expected.items():
        assert (
            tuple(scenarios[scenario_id][field] for field in fields) == expected_values
        )


def test_control_dgp_is_exact_but_prior_and_recovery_are_intentionally_distinct(
    manifest,
) -> None:
    """Allow only the reviewed prior-specific differences inside a control pair."""
    controls = {
        scenario["scenario_id"]: scenario
        for scenario in manifest["scenarios"]
        if scenario["purpose"] == "control"
    }
    candidates = [
        scenario
        for scenario in manifest["scenarios"]
        if scenario["control_id"] is not None
    ]
    assert candidates
    for candidate in candidates:
        control = controls[candidate["control_id"]]
        assert candidate["data_id"] == control["data_id"]
        assert candidate["prior"] == "truncated_normal"
        assert control["prior"] == "linked_normal"
        assert candidate["recovery"] is True
        assert control["recovery"] is False

    softmax_candidate = next(
        item
        for item in candidates
        if item["scenario_id"] == "qual-hssm-softmax-beta-pymc"
    )
    softmax_control = controls[softmax_candidate["control_id"]]
    assert softmax_candidate["posterior_pair_id"] == "hssm-softmax-beta-truncated"
    assert softmax_control["posterior_pair_id"] == "hssm-softmax-beta-linked"
    for field in qualification.DATA_MATCH_FIELDS:
        assert softmax_candidate.get(field) == softmax_control.get(field)

    altered_dgp = copy.deepcopy(manifest)
    candidate = next(
        item
        for item in altered_dgp["scenarios"]
        if item["control_id"] and item.get("posterior_pair_id") is None
    )
    control = next(
        item
        for item in altered_dgp["scenarios"]
        if item["scenario_id"] == candidate["control_id"]
    )
    control["n_per_group"] += 1
    with pytest.raises(QualificationError, match="data-generating fields.*n_per_group"):
        validate_manifest(altered_dgp)

    recovering_control = copy.deepcopy(manifest)
    control = next(
        item for item in recovering_control["scenarios"] if item["purpose"] == "control"
    )
    control["recovery"] = True
    with pytest.raises(QualificationError, match="recovery=false"):
        validate_manifest(recovering_control)


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        ("prior_contracts", lambda value: value["truncated_normal"].update(link="log")),
        ("data_generation", lambda value: value.update(rng="numpy-mt19937")),
        ("execution_policy", lambda value: value.update(cores=2)),
        (
            "artifact_policy",
            lambda value: value["chain_variable_scales"].update(
                group_location="linear_predictor"
            ),
        ),
    ],
)
def test_scientific_and_execution_policies_are_exact(manifest, section, mutate) -> None:
    """Require a manifest revision for any prior, DGP, runtime, or artifact drift."""
    changed = copy.deepcopy(manifest)
    mutate(changed[section])

    with pytest.raises(QualificationError, match="reviewed v2 contract"):
        validate_manifest(changed)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("posterior_summary", {"draw_set": "selected-draws"}),
        ("gradient_evaluation", {"point": "arbitrary"}),
        ("sampler_stat_mapping", {"diverging": "integer"}),
    ],
)
def test_analysis_algorithms_are_exact(manifest, field, replacement) -> None:
    """Make summary, gradient, and sample-stat algorithms executable policy."""
    changed = copy.deepcopy(manifest)
    changed["analysis_policy"][field] = replacement

    with pytest.raises(QualificationError, match=f"analysis {field} algorithm"):
        validate_manifest(changed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("combined_tolerance_rule", "abs <= atol or rel <= rtol", "combined"),
        ("abs_rel_maxima_role", "independent-gates", "maxima role"),
    ],
)
def test_gradient_combined_tolerance_policy_is_exact(
    manifest, field, value, message
) -> None:
    """Prevent regressions to independently gated absolute/relative maxima."""
    changed = copy.deepcopy(manifest)
    changed["analysis_policy"]["gradient_contract"][field] = value

    with pytest.raises(QualificationError, match=message):
        validate_manifest(changed)


def test_seed_derivation_is_stable_and_domain_separated() -> None:
    """Pin v2 seed streams and keep data, initialization, SBC, and chains apart."""
    data_seed = derive_seed(1282, "hssm-lba2-near", 0, "data")
    data_streams = {
        purpose: derive_data_stream_seed(data_seed, "hssm-lba2-near", 0, purpose)
        for purpose in ("truth", "group", "observation")
    }
    initialization_seed = derive_seed(
        1282, "qual-hssm-lba2-near-pymc", 0, "initialization"
    )
    start_seeds = [
        derive_start_seed(
            initialization_seed,
            "qual-hssm-lba2-near-pymc--replicate-00",
            0,
            chain,
        )
        for chain in range(4)
    ]
    chain_seeds = [
        derive_seed(1282, "qual-hssm-lba2-near-pymc", 0, "chain", chain)
        for chain in range(4)
    ]

    assert data_seed == 868132154
    assert data_streams == {
        "truth": 1157851697,
        "group": 2141294012,
        "observation": 426356076,
    }
    assert initialization_seed == 1688471175
    assert start_seeds == [916054994, 1062567866, 1189341988, 724771961]
    assert chain_seeds == [184918405, 299834293, 1943681082, 222358053]
    all_seeds = {
        data_seed,
        *data_streams.values(),
        initialization_seed,
        *start_seeds,
        *chain_seeds,
    }
    assert len(all_seeds) == 13
    sampler_seed = derive_seed(1282, "qual-hssm-lba2-near-numpyro", 0, "sampler")
    assert sampler_seed == 1357540899
    assert derive_numpyro_chain_keys(sampler_seed, 4) == (
        (1134006557, 687687184),
        (482232183, 3172376847),
        (671930531, 3218124145),
        (2923255406, 1106134060),
    )
    assert derive_numpyro_chain_keys(sampler_seed, 1) == ((0, sampler_seed),)
    assert derive_seed(1282, "hssm-lba2-near", 1, "data") != data_seed
    assert derive_seed(1282, "calib-pymc-lower-outside", 0, "sbc_draw") == 771382827
    assert derive_seed(1282, "calib-pymc-lower-outside", 0, "sbc_tie") == 71285581
    with pytest.raises(QualificationError, match="purpose must be one of"):
        derive_seed(1282, "hssm-lba2-near", 0, "truth")
    with pytest.raises(QualificationError, match="data stream purpose"):
        derive_data_stream_seed(data_seed, "hssm-lba2-near", 0, "chain")


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


def test_plan_separates_start_and_backend_sampler_seed_contracts(manifest) -> None:
    """Give each backend exactly the seed identity its sampler actually consumes."""
    for tier in ("smoke", "qualification", "stress"):
        for entry in expand_plan(manifest, tier):
            scenario = entry["scenario"]
            if scenario["initialization_policy"] == "backend-default":
                assert len(entry["start_seeds"]) == scenario["chains"]
                assert entry["start_seeds"] == [
                    derive_start_seed(
                        entry["initialization_seed"],
                        entry["cell_id"],
                        entry["replicate"],
                        chain,
                    )
                    for chain in range(scenario["chains"])
                ]
            else:
                assert scenario["layer"] == "hssm"
                assert entry["start_seeds"] == []
            if scenario["sampler"] == "pymc":
                assert entry["sampler_seed"] is None
                assert len(entry["chain_seeds"]) == scenario["chains"]
            else:
                assert isinstance(entry["sampler_seed"], int)
                assert entry["chain_seeds"] == []


def test_controls_share_dgp_streams_but_not_sampler_randomness(manifest) -> None:
    """Pair exact natural-scale data while separating initialization and chains."""
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
        assert control["recovery"] is False
        assert candidate["recovery"] is True
        assert control["prior"] == "linked_normal"
        assert candidate["prior"] == "truncated_normal"
        for replicate in range(candidate["replicates"]):
            candidate_entry = entries[(candidate["scenario_id"], replicate)]
            control_entry = entries[(control["scenario_id"], replicate)]
            for field in (
                "data_seed",
                "truth_seed",
                "group_seed",
                "observation_seed",
            ):
                assert candidate_entry[field] == control_entry[field]
            for field, purpose in (
                ("truth_seed", "truth"),
                ("group_seed", "group"),
                ("observation_seed", "observation"),
            ):
                assert candidate_entry[field] == derive_data_stream_seed(
                    candidate_entry["data_seed"],
                    candidate["data_id"],
                    replicate,
                    purpose,
                )
            assert (
                candidate_entry["initialization_seed"]
                != control_entry["initialization_seed"]
            )
            if candidate["initialization_policy"] == "backend-default":
                assert set(candidate_entry["start_seeds"]).isdisjoint(
                    control_entry["start_seeds"]
                )
            else:
                assert candidate_entry["start_seeds"] == []
                assert control_entry["start_seeds"] == []
            if candidate["sampler"] == "pymc":
                assert candidate_entry["sampler_seed"] is None
                assert control_entry["sampler_seed"] is None
                assert len(candidate_entry["chain_seeds"]) == candidate["chains"]
                assert set(candidate_entry["chain_seeds"]).isdisjoint(
                    control_entry["chain_seeds"]
                )
            else:
                assert candidate_entry["chain_seeds"] == []
                assert control_entry["chain_seeds"] == []
                assert candidate_entry["sampler_seed"] != control_entry["sampler_seed"]


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
    with pytest.raises(QualificationError, match="controls must use linked_normal"):
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


def test_cell_precision_is_independent_of_environment_collection(
    manifest, environment
) -> None:
    """Use observed cell precision rather than the sidecar collector's process mode."""
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

    validate_result_record(record, entry, no_x64_catalog, manifest)


def test_cell_results_are_atomic_and_identity_checked(
    manifest, environment, make_result, tmp_path: Path
) -> None:
    """Publish a complete cell only after validating its planned identity."""
    entry = expand_plan(manifest, "smoke")[0]
    record = make_result(entry)
    _materialize_result_artifacts(record, tmp_path)
    results_dir = tmp_path / "cells"

    path = write_cell_result(record, entry, results_dir, environment, manifest)

    assert json.loads(path.read_text()) == record
    assert list(results_dir.glob("*.tmp")) == []
    with pytest.raises(QualificationError, match="refusing to overwrite"):
        write_cell_result(record, entry, results_dir, environment, manifest)
    assert json.loads(path.read_text()) == record
    changed = copy.deepcopy(record)
    changed["data_seed"] += 1
    with pytest.raises(QualificationError, match="data_seed"):
        validate_result_record(changed, entry, environment, manifest)

    provenance = record["provenance"]
    assert isinstance(provenance, dict)
    start_path = tmp_path / provenance["actual_start_artifact"]
    start_path.write_bytes(b"tampered\n")
    with pytest.raises(QualificationError, match="does not match artifact bytes"):
        verify_result_artifacts(record, tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "data_seed",
        "truth_seed",
        "group_seed",
        "observation_seed",
        "initialization_seed",
        "start_seeds",
        "sampler_seed",
        "sbc_draw_seed",
        "sbc_tie_seed",
        "chain_seeds",
    ],
)
def test_result_identity_mirrors_every_planned_seed(
    field, manifest, environment, make_result
) -> None:
    """Reject relabelled evidence from every v2 random stream."""
    entry = next(
        item
        for item in expand_plan(manifest, "qualification")
        if (
            item["scenario"]["sampler"] == "numpyro"
            if field == "sampler_seed"
            else item["scenario"].get("calibration_kind") == "sbc"
        )
    )
    record = make_result(entry)
    if field in {"start_seeds", "chain_seeds"}:
        record[field][0] += 1
    else:
        record[field] += 1

    with pytest.raises(QualificationError, match=field):
        validate_result_record(record, entry, environment, manifest)


def test_hssm_result_records_initialization_root_without_per_chain_start_seeds(
    manifest, environment, make_result
) -> None:
    """Represent HSSM's one constructed-and-replicated start without fake seeds."""
    entry = next(
        item
        for item in expand_plan(manifest, "smoke")
        if item["scenario"]["layer"] == "hssm"
    )
    assert entry["start_seeds"] == []
    record = make_result(entry)
    record["start_seeds"] = [entry["initialization_seed"]]

    with pytest.raises(QualificationError, match="start_seeds"):
        validate_result_record(record, entry, environment, manifest)


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

    missing_raw_cost = make_result(entry)
    missing_raw_cost["metrics"].pop("sampling_elapsed_seconds")
    with pytest.raises(QualificationError, match="raw sampler metrics"):
        validate_result_record(missing_raw_cost, entry, environment, manifest)


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
        ("sampling_elapsed_seconds", 0.0),
        ("step_size_median", -0.1),
        ("gradient_evaluation_count", 1.5),
        ("leapfrog_step_count", 0),
        ("likelihood_pytensor_jax_value_abs_error_max", -0.1),
        ("likelihood_pytensor_jax_value_normalized_error_max", -0.1),
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


def test_result_provenance_is_bound_to_environment_and_raw_artifacts(
    manifest, environment, make_result
) -> None:
    """Reject forged checkout identity and unsafe or unpaired artifact references."""
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

    wrong_data_path = make_result(entry)
    wrong_data_path["provenance"]["data_artifact"] = "data/other-r0.json"
    with pytest.raises(QualificationError, match="does not match artifact_policy"):
        validate_result_record(wrong_data_path, entry, environment, manifest)

    missing_data_digest = make_result(entry)
    missing_data_digest["provenance"]["data_sha256"] = None
    with pytest.raises(QualificationError, match="provided together"):
        validate_result_record(missing_data_digest, entry, environment, manifest)

    wrong_device = make_result(entry)
    wrong_device["provenance"]["device"] = "gpu"
    with pytest.raises(QualificationError, match="device does not match"):
        validate_result_record(wrong_device, entry, environment, manifest)

    bad_attempt = make_result(entry)
    bad_attempt["provenance"]["execution_attempt_id"] = "not-a-digest"
    with pytest.raises(QualificationError, match="execution_attempt_id"):
        validate_result_record(bad_attempt, entry, environment, manifest)

    forged_pair = make_result(entry)
    forged_pair["provenance"]["pair_execution_id"] = "d" * 64
    forged_pair["provenance"]["pair_position"] = 0
    with pytest.raises(QualificationError, match="null for unpaired"):
        validate_result_record(forged_pair, entry, environment, manifest)

    paired_entry = next(
        item
        for item in expand_plan(manifest, "qualification")
        if item["scenario_id"] == "qual-pymc-lower-inside"
    )
    missing_pair = make_result(paired_entry)
    missing_pair["provenance"]["pair_execution_id"] = None
    missing_pair["provenance"]["pair_position"] = None
    with pytest.raises(QualificationError, match="required by paired execution"):
        validate_result_record(missing_pair, paired_entry, environment, manifest)

    wrong_pytensor_precision = make_result(entry)
    wrong_pytensor_precision["provenance"]["pytensor_floatx"] = "float32"
    with pytest.raises(QualificationError, match="pytensor_floatx does not match"):
        validate_result_record(wrong_pytensor_precision, entry, environment, manifest)

    wrong_jax_precision = make_result(entry)
    wrong_jax_precision["provenance"]["jax_enable_x64"] = False
    with pytest.raises(QualificationError, match="jax_enable_x64 does not match"):
        validate_result_record(wrong_jax_precision, entry, environment, manifest)

    missing_observed_precision = make_result(entry)
    missing_observed_precision["provenance"]["pytensor_floatx"] = None
    missing_observed_precision["provenance"]["jax_enable_x64"] = None
    with pytest.raises(QualificationError, match="requires observed precision"):
        validate_result_record(missing_observed_precision, entry, environment, manifest)

    failed_before_precision = make_result(entry, status="failed", metrics={})
    failed_before_precision["provenance"]["pytensor_floatx"] = None
    failed_before_precision["provenance"]["jax_enable_x64"] = None
    validate_result_record(failed_before_precision, entry, environment, manifest)

    pymc_with_jax_keys = make_result(entry)
    pymc_with_jax_keys["provenance"]["effective_numpyro_chain_keys"] = [[0, 1]]
    with pytest.raises(QualificationError, match="must be null for PyMC"):
        validate_result_record(pymc_with_jax_keys, entry, environment, manifest)

    wrong_chain_path = make_result(entry)
    wrong_chain_path["provenance"]["raw_chain_artifact"] = "chains/other.nc"
    with pytest.raises(QualificationError, match="does not match artifact_policy"):
        validate_result_record(wrong_chain_path, entry, environment, manifest)

    missing_chain_digest = make_result(entry)
    missing_chain_digest["provenance"]["raw_chain_sha256"] = None
    with pytest.raises(QualificationError, match="provided together"):
        validate_result_record(missing_chain_digest, entry, environment, manifest)

    numpyro_entry = next(
        item
        for item in expand_plan(manifest, "qualification")
        if item["scenario"]["sampler"] == "numpyro"
    )
    missing_effective_keys = make_result(numpyro_entry)
    missing_effective_keys["provenance"]["effective_numpyro_chain_keys"] = None
    with pytest.raises(QualificationError, match="required for completed NumPyro"):
        validate_result_record(
            missing_effective_keys, numpyro_entry, environment, manifest
        )

    wrong_effective_keys = make_result(numpyro_entry)
    wrong_effective_keys["provenance"]["effective_numpyro_chain_keys"][0][0] += 1
    with pytest.raises(QualificationError, match="does not match sampler_seed"):
        validate_result_record(
            wrong_effective_keys, numpyro_entry, environment, manifest
        )


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
    assert aggregate[0]["provenance"]["raw_chain_artifact"].startswith("chains/")
    assert len(aggregate[0]["provenance"]["raw_chain_sha256"]) == 64
    assert aggregate[0]["provenance"]["data_artifact"].startswith("data/")
    assert len(aggregate[0]["provenance"]["data_sha256"]) == 64
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


def test_backend_rank_rhat_is_recomputed_from_verified_raw_chains(
    manifest, environment, make_result, tmp_path: Path
) -> None:
    """Hash-verified paired chains clear pending evidence and expose disagreement."""
    plan = expand_plan(manifest, "qualification")
    left_entry = next(
        entry
        for entry in plan
        if entry["scenario_id"] == "qual-pymc-lower-outside" and entry["replicate"] == 0
    )
    right_entry = next(
        entry
        for entry in plan
        if entry["scenario_id"] == "qual-pymc-lower-outside-numpyro"
        and entry["replicate"] == 0
    )
    left = make_result(left_entry)
    right = make_result(right_entry)
    _materialize_result_artifacts(left, tmp_path)
    _materialize_result_artifacts(right, tmp_path)
    _write_standardized_test_chain(left, left_entry, tmp_path, seed=1282)
    _write_standardized_test_chain(right, right_entry, tmp_path, seed=1283)
    scenarios = {
        entry["scenario_id"]: {**entry["scenario"], "replicates": 1}
        for entry in (left_entry, right_entry)
    }
    pending: list[dict[str, object]] = []

    checks = qualification._evaluate_backend_pairs(
        [left, right],
        scenarios,
        manifest["analysis_policy"],
        pending,
        artifact_root=tmp_path,
        manifest=manifest,
    )

    assert pending == []
    rhat = next(
        check for check in checks if check["metric"] == "backend_combined_rank_rhat_max"
    )
    assert rhat["passed"] is True
    assert set(rhat["parameter_rhats"]) == set(
        manifest["analysis_policy"]["monitored_parameters"]
    )

    _write_standardized_test_chain(
        right,
        right_entry,
        tmp_path,
        seed=1283,
        group_middle_shift=4.0,
    )
    failed = qualification._evaluate_backend_pairs(
        [left, right],
        scenarios,
        manifest["analysis_policy"],
        [],
        artifact_root=tmp_path,
        manifest=manifest,
    )
    failed_rhat = next(
        check for check in failed if check["metric"] == "backend_combined_rank_rhat_max"
    )
    assert failed_rhat["passed"] is False
    assert failed_rhat["parameter_id"] == "group_middle"


def test_paired_execution_provenance_binds_worker_order_and_attempts(
    manifest, make_result
) -> None:
    """Reject split, reordered, or replayed candidate/control evidence."""
    plan = expand_plan(manifest, "qualification")
    scenario_ids = {
        "qual-pymc-lower-inside",
        "qual-pymc-lower-inside-control",
    }
    entries = [
        entry
        for entry in plan
        if entry["scenario_id"] in scenario_ids and entry["replicate"] < 2
    ]
    records = [make_result(entry) for entry in entries]
    scenarios = {
        entry["scenario_id"]: {**entry["scenario"], "replicates": 2}
        for entry in entries
    }

    checks = qualification._validate_paired_execution_records(records, scenarios)

    assert len(checks) == 2
    assert [check["replicate"] for check in checks] == [0, 1]
    assert all(check["passed"] for check in checks)

    split_worker = copy.deepcopy(records)
    control = next(
        record
        for record in split_worker
        if record["scenario_id"].endswith("-control") and record["replicate"] == 0
    )
    control["provenance"]["worker_identity_sha256"] = "f" * 64
    with pytest.raises(QualificationError, match="different worker identity"):
        qualification._validate_paired_execution_records(split_worker, scenarios)

    replayed = copy.deepcopy(records)
    pair_ids = {
        record["replicate"]: record["provenance"]["pair_execution_id"]
        for record in replayed
        if record["scenario_id"] == "qual-pymc-lower-inside"
    }
    for record in replayed:
        if record["replicate"] == 1:
            record["provenance"]["pair_execution_id"] = pair_ids[0]
    with pytest.raises(QualificationError, match="reused across pairs"):
        qualification._validate_paired_execution_records(replayed, scenarios)


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
    assert candidate_bias["family"] == "candidate"
    assert candidate_bias["passed"] is False
    assert candidate_bias["actual"] == pytest.approx(0.6)
    assert candidate_bias["sign_test_role"] == "descriptive-only"
    assert not any("reproducible-bias" in blocker for blocker in assessment["blockers"])
    assert not any(
        check["scope"] == "fixed_recovery_bias"
        and check["scenario_id"].endswith("-control")
        for check in assessment["checks"]
    )

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
        and check["scenario_id"] == "calib-pymc-two-sided-midpoint"
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
        and check["scenario_id"] == "calib-pymc-two-sided-midpoint"
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


def test_combined_gradient_contract_is_assessor_derived_and_classified(
    manifest, environment, make_result
) -> None:
    """Gate normalized combined errors and keep abs/rel maxima descriptive."""
    plan = expand_plan(manifest, "smoke")
    records = [make_result(entry) for entry in plan]
    targets = [
        (
            records[0],
            plan[0]["scenario"],
            "finite_difference_gradient_normalized_error_max",
            "prior-gradient-contract",
        ),
        (
            next(
                record
                for record in records
                if record["scenario_id"] == "smoke-bambi-lower-outside"
            ),
            next(
                entry["scenario"]
                for entry in plan
                if entry["scenario_id"] == "smoke-bambi-lower-outside"
            ),
            "bambi_isomorphism_normalized_error_max",
            "bambi-isomorphism-contract",
        ),
        (
            next(
                record
                for record in records
                if record["scenario_id"] == "smoke-hssm-lba2-near"
            ),
            next(
                entry["scenario"]
                for entry in plan
                if entry["scenario_id"] == "smoke-hssm-lba2-near"
            ),
            "likelihood_pytensor_jax_value_normalized_error_max",
            "likelihood/backend-contract",
        ),
    ]
    for record, scenario, metric, _failure_class in targets:
        condition = qualification._gradient_contract_conditions(
            record, scenario, manifest["analysis_policy"]
        )[metric]
        record["metrics"][metric] = condition["value"] * 2

    failed = assess_results(records, plan, manifest, "smoke", environment)

    assert failed["outcome"] == "screening-fail"
    for record, _scenario, metric, failure_class in targets:
        contract_check = next(
            check
            for check in failed["checks"]
            if check["scope"] == "gradient_contract"
            and check["cell_id"] == record["cell_id"]
            and check["metric"] == metric
        )
        assert contract_check["passed"] is False
        assert contract_check["failure_class"] == failure_class

    descriptive_records = [make_result(entry) for entry in plan]
    descriptive_records[0]["metrics"]["finite_difference_gradient_rel_error_max"] = (
        1_000.0
    )
    descriptive = assess_results(
        descriptive_records, plan, manifest, "smoke", environment
    )
    assert descriptive["outcome"] == "screening-pass"
    assert not any(
        check["metric"] == "finite_difference_gradient_rel_error_max"
        for check in descriptive["checks"]
    )

    missing_records = [make_result(entry) for entry in plan]
    metric = "finite_difference_gradient_abs_error_max"
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


def test_stress_assessment_evaluates_every_diagnostic_fit(
    manifest, environment, make_result
) -> None:
    """Do not call stress complete when its diagnostic metrics are absent or fail."""
    plan = expand_plan(manifest, "stress")
    records = [make_result(entry) for entry in plan]

    complete = assess_results(records, plan, manifest, "stress", environment)

    assert complete["outcome"] == "diagnostic-complete"
    assert complete["counts"]["checks"] == len(plan) * len(
        manifest["thresholds"]["diagnostic"]["per_fit"]
    )

    missing = copy.deepcopy(records)
    missing[0]["metrics"].pop("gradient_finite")
    incomplete = assess_results(missing, plan, manifest, "stress", environment)
    assert incomplete["outcome"] == "incomplete"
    assert f"{plan[0]['cell_id']}:gradient_finite" in incomplete["missing_metrics"]

    failing = copy.deepcopy(records)
    failing[0]["metrics"].update(divergence_count=40, divergence_rate=0.01)
    failed = assess_results(failing, plan, manifest, "stress", environment)
    assert failed["outcome"] == "diagnostic-failed"
    assert any(
        check["scope"] == "diagnostic_per_fit"
        and check["metric"] == "divergence_rate"
        and check["passed"] is False
        for check in failed["checks"]
    )


def test_aggregate_cli_accepts_repeated_environment_sidecars(
    manifest, environment, tmp_path: Path
) -> None:
    """Materialize profile-attributed missing rows from repeated CLI sidecars."""
    environment_paths = []
    for item in environment.values():
        path = tmp_path / f"environment-{item['dependency_profile']}.json"
        path.write_text(json.dumps(item))
        environment_paths.append(path)
    results_dir = tmp_path / "cells"
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

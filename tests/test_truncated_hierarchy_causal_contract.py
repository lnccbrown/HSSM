# ruff: noqa: D103
"""Contract tests for the no-sampling TruncatedNormal causal experiment."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest

import scripts.truncated_hierarchy_causal_contract as contract
from scripts.truncated_hierarchy_causal_contract import (
    BACKEND_IDS,
    REPRESENTATION_IDS,
    CausalContractError,
    RunContext,
    UnitSpec,
    aggregate_results,
    assess_results,
    block_units,
    build_plan,
    canonical_json_bytes,
    derive_seed,
    load_manifest,
    manifest_digest,
    pair_units,
    plan_unit_by_id,
    strict_json_loads,
    validate_manifest,
    validate_result_record,
    validate_run_context,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


@pytest.fixture(scope="module")
def manifest() -> Mapping[str, Any]:
    """Load the repository's frozen causal manifest."""
    return load_manifest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _metrics(unit: UnitSpec, *, healthy: bool = True) -> dict[str, Any]:
    divergence_count = 0 if healthy else unit.chains * unit.draws
    metrics: dict[str, Any] = {
        "compile_success": True,
        "initialization_success": True,
        "logp_finite": True,
        "gradient_finite": True,
        "sampling_success": True,
        "divergence_count": divergence_count,
        "posterior_draw_count": unit.chains * unit.draws,
        "divergence_rate": divergence_count / (unit.chains * unit.draws),
        "oracle_evaluation_count": contract._expected_oracle_evaluation_count(
            unit.as_dict(), unit.tier, posterior_trajectory=True
        ),
        "oracle_logp_scaled_error_max": 0.1,
        "oracle_gradient_scaled_error_max": 0.1,
        "oracle_hessian_scaled_error_max": 0.1,
        "roundtrip_absolute_error_max": 0.0,
        "icdf_tail_finite": True,
        "icdf_branch_continuous": True,
    }
    if unit.tier == "confirmation":
        metrics.update(
            {
                "hyper_rhat_max": 1.0,
                "hyper_ess_bulk_min": 500.0,
                "hyper_ess_tail_min": 500.0,
                "bfmi_min": 0.5,
                "treedepth_saturation_rate": 0.0,
                "hyper_mcse_over_sd_max": 0.01,
                "group_rhat_max": 1.0,
                "group_ess_bulk_fraction_ge_400": 1.0,
                "group_ess_tail_fraction_ge_400": 1.0,
            }
        )
    return metrics


def _artifact(path: str) -> dict[str, Any]:
    return {"path": path, "sha256": _digest(path), "size_bytes": 12}


def _context(unit_pair: tuple[UnitSpec, ...]) -> RunContext:
    environment: dict[str, Any] = {"git": {"commit": "a" * 40}}
    environment_sha256 = contract.environment_digest(environment)
    environment["environment_sha256"] = environment_sha256
    return RunContext(
        schema_version=3,
        study_id="truncated_hierarchy_causal_v3",
        manifest_sha256=unit_pair[0].manifest_sha256,
        pair_id=unit_pair[0].pair_id,
        block_ids=tuple(dict.fromkeys(unit.block_id for unit in unit_pair)),
        cell_ids=tuple(unit.cell_id for unit in unit_pair),
        execution_order=tuple(unit.cell_id for unit in unit_pair),
        environment=environment,
        environment_sha256=environment_sha256,
        git_commit="a" * 40,
        worker_identity_sha256=_digest("worker"),
        pair_execution_id=_digest("pair"),
        execution_attempt_ids=tuple(
            _digest(f"attempt-{unit.cell_id}") for unit in unit_pair
        ),
    )


def _record(
    unit: UnitSpec,
    context: RunContext,
    *,
    healthy: bool = True,
    status: str = "completed",
) -> dict[str, Any]:
    position = context.cell_ids.index(unit.cell_id)
    if unit.backend_id == "pymc":
        sampler_seed_input: int | list[int] = list(unit.chain_seeds)
        chain_rng = contract.derive_pymc_chain_rng_provenance(
            unit.chain_seeds, unit.chains
        )
        sampler = "pymc-nuts"
        compiler = "pytensor"
    else:
        assert unit.sampler_seed is not None
        sampler_seed_input = unit.sampler_seed
        chain_rng = [
            {"chain": chain, "rng": "jax-prng-key", "key": key}
            for chain, key in enumerate(
                contract.derive_numpyro_chain_keys(unit.sampler_seed, unit.chains)
            )
        ]
        sampler = "numpyro-nuts-via-pymc"
        compiler = "pytensor-to-jax"
    artifacts: dict[str, dict[str, Any] | None] = {
        "context": _artifact(f"contexts/{unit.pair_id}.json"),
        "data": _artifact(f"data/{unit.data_id}.json"),
        "natural_start": _artifact(f"starts/natural/{unit.start_id}.json"),
        "coordinate_start": _artifact(f"starts/coordinates/{unit.cell_id}.json"),
        "chain": _artifact(f"chains/{unit.cell_id}.nc"),
        "diagnostics": _artifact(f"diagnostics/{unit.cell_id}.json"),
    }
    failure = None
    metrics = _metrics(unit, healthy=healthy)
    if status == "failed":
        artifacts["chain"] = None
        artifacts["diagnostics"] = None
        failure = {
            "stage": "sample",
            "error_type": "RuntimeError",
            "message": "sampler failed",
        }
        metrics = {}
    return {
        "schema_version": 3,
        "runner_version": 3,
        "study_id": unit.study_id,
        "manifest_sha256": unit.manifest_sha256,
        "tier": unit.tier,
        "regime_id": unit.regime_id,
        "backend_id": unit.backend_id,
        "representation_id": unit.representation_id,
        "replicate": unit.replicate,
        "pair_id": unit.pair_id,
        "pair_position": unit.pair_position,
        "block_id": unit.block_id,
        "block_position": unit.block_position,
        "cell_id": unit.cell_id,
        "execution_status": status,
        "metrics": metrics,
        "parameter_summaries": [],
        "artifacts": artifacts,
        "failure": failure,
        "provenance": {
            "environment_sha256": context.environment_sha256,
            "git_commit": context.git_commit,
            "worker_identity_sha256": context.worker_identity_sha256,
            "pair_execution_id": context.pair_execution_id,
            "execution_attempt_id": context.execution_attempt_ids[position],
            "sampler": sampler,
            "compiler_path": compiler,
            "device": "cpu",
            "floatx": unit.floatx,
            "pytensor_floatx": unit.floatx,
            "jax_enable_x64": unit.floatx == "float64",
            "sampler_seed_input": sampler_seed_input,
            "chain_rng_provenance": chain_rng,
        },
    }


def _classifier_rows(
    *,
    left_bad_replicates: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backend in BACKEND_IDS:
        for replicate in range(8):
            for representation in REPRESENTATION_IDS:
                healthy = representation in {
                    "group-icdf-noncentered",
                    "full-icdf-noncentered",
                }
                if representation in {
                    "native-centered",
                    "manual-centered",
                    "location-icdf-noncentered",
                }:
                    healthy = replicate >= left_bad_replicates
                metrics = {
                    "compile_success": True,
                    "initialization_success": True,
                    "logp_finite": True,
                    "gradient_finite": True,
                    "sampling_success": True,
                    "divergence_count": 0 if healthy else 4000,
                    "posterior_draw_count": 4000,
                    "divergence_rate": 0.0 if healthy else 1.0,
                    "hyper_rhat_max": 1.0,
                    "hyper_ess_bulk_min": 500.0,
                    "hyper_ess_tail_min": 500.0,
                    "bfmi_min": 0.5,
                    "treedepth_saturation_rate": 0.0,
                    "hyper_mcse_over_sd_max": 0.01,
                    "group_rhat_max": 1.0,
                    "group_ess_bulk_fraction_ge_400": 1.0,
                    "group_ess_tail_fraction_ge_400": 1.0,
                    "oracle_evaluation_count": (
                        4
                        + int(replicate == 0)
                        + 5
                        * {
                            "native-centered": 0,
                            "manual-centered": 0,
                            "group-icdf-noncentered": 1,
                            "location-icdf-noncentered": 1,
                            "full-icdf-noncentered": 2,
                        }[representation]
                        + 4
                    ),
                    "oracle_logp_scaled_error_max": 0.1,
                    "oracle_gradient_scaled_error_max": 0.1,
                    "oracle_hessian_scaled_error_max": 0.1,
                    "roundtrip_absolute_error_max": 0.0,
                    "icdf_tail_finite": True,
                    "icdf_branch_continuous": True,
                }
                rows.append(
                    {
                        "collection_status": "present",
                        "execution_status": "completed",
                        "regime_id": "lower-outside-weak",
                        "backend_id": backend,
                        "representation_id": representation,
                        "replicate": replicate,
                        "metrics": metrics,
                    }
                )
    return rows


def test_manifest_is_frozen_to_v2_and_complete(manifest) -> None:
    """Bind the new study to exact v2 bytes and the complete five-form design."""
    assert manifest["frozen_v2"]["file_sha256"] == contract.sha256_file(
        contract.DEFAULT_FROZEN_V2
    )
    assert (
        tuple(item["representation_id"] for item in manifest["representations"])
        == REPRESENTATION_IDS
    )
    assert manifest["tiers"]["smoke"]["expected_fit_count"] == 20
    assert manifest["tiers"]["confirmation"]["expected_fit_count"] == 160
    assert (
        manifest["failure_policy"]["block_failure"]
        == "one-member-failure-does-not-suppress-the-other-nine-attempts"
    )
    assert (
        manifest["analysis_policy"]["family_health"]["maximum_failed_replicates"] == 0
    )
    assert manifest_digest(manifest) == manifest_digest(load_manifest())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["representations"].pop(),
            "exactly five",
        ),
        (
            lambda value: value["tiers"]["confirmation"].update(replicates=5),
            "budget changed",
        ),
        (
            lambda value: value["analysis_policy"]["family_health"].update(
                maximum_failed_replicates=1
            ),
            "require 8/8",
        ),
        (
            lambda value: value["execution_policy"].update(order="random"),
            "counterbalancing",
        ),
        (
            lambda value: value["natural_model"].update(observation_sigma=0.6),
            "natural_model",
        ),
        (
            lambda value: value["backends"][0].update(sampler="other"),
            "backend sampler/compiler",
        ),
        (
            lambda value: value["analysis_policy"]["per_fit_health"].update(
                divergence_rate_lt=0.02
            ),
            "semantic digest",
        ),
        (
            lambda value: value["analysis_policy"]["family_health"].update(
                aggregate_divergence_rate_lt=0.002
            ),
            "semantic digest",
        ),
        (
            lambda value: value["analysis_policy"]["oracle_gate"].update(
                evaluation_points=["fixed-grid"]
            ),
            "evaluation-point",
        ),
        (
            lambda value: value["analysis_policy"]["oracle_gate"].update(
                scaled_error_max=2.0
            ),
            "semantic digest",
        ),
        (
            lambda value: value["analysis_policy"]["classifier_precedence"].reverse(),
            "classifier precedence",
        ),
        (
            lambda value: value["tiers"]["smoke"].update(role="changed"),
            "semantic digest",
        ),
        (
            lambda value: value["artifact_policy"].update(hash="changed"),
            "semantic digest",
        ),
        (
            lambda value: value["failure_policy"].update(
                missing_result="silently-pass"
            ),
            "semantic digest",
        ),
    ],
)
def test_manifest_rejects_design_drift(manifest, mutation, message) -> None:
    changed = copy.deepcopy(manifest)
    mutation(changed)
    with pytest.raises(CausalContractError, match=message):
        validate_manifest(changed)


def test_strict_json_and_seed_domain() -> None:
    with pytest.raises(CausalContractError, match="duplicate"):
        strict_json_loads('{"a":1,"a":2}')
    with pytest.raises(CausalContractError, match="invalid constant"):
        strict_json_loads('{"a":NaN}')
    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n'
    assert derive_seed(1282, "data", "lower-outside-weak", 1) == derive_seed(
        1282, "data", "lower-outside-weak", 1
    )
    assert derive_seed(1282, "data", "lower-outside-weak", 1) != derive_seed(
        1282, "sampler", "lower-outside-weak", 1
    )
    pymc_rng = contract.derive_pymc_chain_rng_provenance(
        [943009182, 939311901, 1620816411, 1994453597], 4
    )
    assert [item["init_step_seed"] for item in pymc_rng] == [
        1019536840,
        896392474,
        417781057,
        822370369,
    ]
    assert [item["spawn_key"] for item in pymc_rng] == [[0], [1], [2], [3]]


def test_plan_counts_counterbalancing_and_shared_inputs(manifest) -> None:
    smoke = build_plan(manifest, "smoke")
    confirmation = build_plan(manifest, "confirmation")
    assert len(smoke) == 20
    assert len(confirmation) == 160
    assert [unit.representation_id for unit in smoke if unit.block_position == 0] == [
        "native-centered",
        "manual-centered",
        "location-icdf-noncentered",
        "group-icdf-noncentered",
    ]
    starts = [
        unit.representation_id
        for unit in confirmation
        if unit.regime_id == "lower-outside-weak"
        and unit.backend_id == "pymc"
        and unit.block_position == 0
    ]
    assert starts == [
        "native-centered",
        "full-icdf-noncentered",
        "location-icdf-noncentered",
        "group-icdf-noncentered",
        "manual-centered",
        "native-centered",
        "full-icdf-noncentered",
        "location-icdf-noncentered",
    ]
    first_pair = pair_units(manifest, "smoke", smoke[0].pair_id)
    assert len(first_pair) == 10
    assert len({unit.data_seed for unit in first_pair}) == 1
    assert len({unit.natural_start_chain_seeds for unit in first_pair}) == 1
    assert [unit.backend_id for unit in first_pair[:5]] == ["pymc"] * 5
    assert [unit.backend_id for unit in first_pair[5:]] == ["numpyro"] * 5
    second_pair = pair_units(manifest, "smoke", smoke[10].pair_id)
    assert [unit.backend_id for unit in second_pair[:5]] == ["numpyro"] * 5
    assert [unit.backend_id for unit in second_pair[5:]] == ["pymc"] * 5
    for chain, seed in enumerate(first_pair[0].natural_start_chain_seeds):
        assert seed == derive_seed(first_pair[0].natural_start_seed, "chain", chain)
    assert first_pair[0].data_seed == 1818466647
    assert first_pair[0].truth_seed == 746768835
    assert all(unit.sampler_seed is None for unit in first_pair[:5])
    numpyro = block_units(manifest, "smoke", smoke[5].block_id)
    assert all(
        unit.sampler_seed is not None and not unit.chain_seeds for unit in numpyro
    )


def test_plan_roundtrip_and_exact_lookup(manifest) -> None:
    unit = build_plan(manifest, "smoke")[0]
    assert UnitSpec.from_dict(unit.as_dict()) == unit
    assert plan_unit_by_id(manifest, "smoke", unit.cell_id) == unit
    with pytest.raises(CausalContractError, match="not planned"):
        plan_unit_by_id(manifest, "smoke", f"{unit.cell_id}--unknown")


def test_run_context_binds_both_backend_blocks_and_ten_attempts(manifest) -> None:
    plan = build_plan(manifest, "smoke")
    units = pair_units(manifest, "smoke", plan[0].pair_id)
    context = _context(units)
    assert RunContext.from_dict(context.as_dict()) == context
    assert validate_run_context(context, units) == context
    changed = context.as_dict()
    changed["execution_order"] = list(reversed(changed["execution_order"]))
    with pytest.raises(CausalContractError, match="execution_order"):
        RunContext.from_dict(changed)
    changed = context.as_dict()
    changed["environment"]["git"]["commit"] = "b" * 40
    with pytest.raises(CausalContractError, match="environment"):
        RunContext.from_dict(changed)


def test_result_identity_artifacts_failure_and_seed_provenance(manifest) -> None:
    plan = build_plan(manifest, "smoke")
    units = pair_units(manifest, "smoke", plan[0].pair_id)
    context = _context(units)
    good = _record(units[0], context)
    assert validate_result_record(good, units[0], context) == good
    forged = copy.deepcopy(good)
    forged["artifacts"]["data"]["path"] = "data/another.json"
    with pytest.raises(CausalContractError, match="does not match the plan"):
        validate_result_record(forged, units[0], context)
    forged = copy.deepcopy(good)
    forged["provenance"]["chain_rng_provenance"][0]["init_step_seed"] += 1
    with pytest.raises(CausalContractError, match="chain RNG provenance"):
        validate_result_record(forged, units[0], context)
    failed = _record(units[0], context, status="failed")
    assert validate_result_record(failed, units[0], context) == failed
    failed["failure"]["stage"] = "infrastructure"
    with pytest.raises(CausalContractError, match="scientific stage"):
        validate_result_record(failed, units[0], context)


def test_failed_sample_accepts_only_complete_auditable_pre_oracle(manifest) -> None:
    plan = build_plan(manifest, "smoke")
    units = pair_units(manifest, "smoke", plan[0].pair_id)
    context = _context(units)
    unit = units[0]
    failed = _record(unit, context, status="failed")
    complete = _metrics(unit)
    failed["metrics"] = {
        key: complete[key]
        for key in (
            "compile_success",
            "initialization_success",
            "logp_finite",
            "gradient_finite",
            *sorted(contract.ORACLE_METRICS),
        )
    }
    failed["metrics"]["sampling_success"] = False
    failed["metrics"]["oracle_evaluation_count"] = (
        contract._expected_oracle_evaluation_count(
            unit.as_dict(), "smoke", posterior_trajectory=False
        )
    )
    failed["artifacts"]["diagnostics"] = _artifact(f"diagnostics/{unit.cell_id}.json")
    assert validate_result_record(failed, unit, context) == failed

    no_diagnostics = copy.deepcopy(failed)
    no_diagnostics["artifacts"]["diagnostics"] = None
    with pytest.raises(CausalContractError, match="requires diagnostics"):
        validate_result_record(no_diagnostics, unit, context)
    missing_point = copy.deepcopy(failed)
    missing_point["metrics"]["oracle_evaluation_count"] -= 1
    with pytest.raises(CausalContractError, match="frozen evidence phase"):
        validate_result_record(missing_point, unit, context)


def test_completed_oracle_requires_every_static_and_trajectory_point(manifest) -> None:
    plan = build_plan(manifest, "smoke")
    units = pair_units(manifest, "smoke", plan[0].pair_id)
    context = _context(units)
    record = _record(units[0], context)
    record["metrics"]["oracle_evaluation_count"] -= 1
    with pytest.raises(CausalContractError, match="trajectory/static evidence"):
        validate_result_record(record, units[0], context)


def test_raw_diagnostics_recompute_registered_oracle_metrics(
    manifest, tmp_path: Path
) -> None:
    unit = build_plan(manifest, "smoke")[0]
    context = _context(pair_units(manifest, "smoke", unit.pair_id))
    record = _record(unit, context)
    count = record["metrics"]["oracle_evaluation_count"]
    coordinate_size = int(unit.regime["n_groups"]) + 2
    gradient = [float(index + 1) for index in range(coordinate_size)]
    hessian = [
        [float(row == column) for column in range(coordinate_size)]
        for row in range(coordinate_size)
    ]
    raw_record: dict[str, Any] = {
        "observed": {
            "value": 1.0,
            "gradient": list(gradient),
            "hessian": copy.deepcopy(hessian),
        },
        "oracle": {
            "value": 1.0,
            "gradient": list(gradient),
            "hessian": copy.deepcopy(hessian),
        },
        "errors": {
            "value": {"absolute_max": 0.0, "scaled_max": 0.0},
            "gradient": {"absolute_max": 0.0, "scaled_max": 0.0},
            "hessian": {"absolute_max": 0.0, "scaled_max": 0.0},
        },
        "roundtrip": {"absolute_error_max": 0.0},
        "component_finite": {"value": True, "gradient": True, "hessian": True},
        "finite": True,
        "passed": True,
    }
    records: list[dict[str, Any]] = []
    fixed = copy.deepcopy(raw_record)
    fixed.update(point_id="fixed-truth", kind="fixed-grid")
    records.append(fixed)
    for chain in range(unit.chains):
        start = copy.deepcopy(raw_record)
        start.update(point_id=f"start-chain-{chain:02d}", kind="shared-natural-start")
        records.append(start)
    for chain in range(unit.chains):
        draw = min(
            range(unit.draws),
            key=lambda candidate: hashlib.sha256(
                f"{unit.cell_id}:trajectory:{chain}:{candidate}".encode()
            ).digest(),
        )
        trajectory = copy.deepcopy(raw_record)
        trajectory.update(
            point_id=f"trajectory-chain-{chain:02d}-selection-00",
            kind="hash-selected-posterior-trajectory",
            chain=chain,
            draw=draw,
            selection_sha256=hashlib.sha256(
                f"{unit.cell_id}:trajectory:{chain}:{draw}".encode()
            ).hexdigest(),
        )
        records.append(trajectory)
    assert len(records) == count
    for name in (
        "oracle_logp_scaled_error_max",
        "oracle_gradient_scaled_error_max",
        "oracle_hessian_scaled_error_max",
    ):
        record["metrics"][name] = 0.0
    diagnostics: dict[str, Any] = {
        "schema_version": unit.schema_version,
        "study_id": unit.study_id,
        "manifest_sha256": unit.manifest_sha256,
        "cell_id": unit.cell_id,
        "runtime": {
            "pytensor_floatx": unit.floatx,
            "jax_enable_x64": unit.floatx == "float64",
        },
        "oracle": {
            "posterior_trajectory_evaluated": True,
            "records": records,
            "icdf_tail_finite": True,
            "icdf_branch_checks": [],
            "icdf_branch_continuous": True,
            "passed": True,
        },
    }
    path = tmp_path / "diagnostics.json"
    path.write_bytes(canonical_json_bytes(diagnostics))
    contract._audit_diagnostic_evidence(record, unit, path, manifest["analysis_policy"])
    forged = copy.deepcopy(record)
    forged["metrics"]["oracle_gradient_scaled_error_max"] = 0.2
    with pytest.raises(CausalContractError, match="hash-bound raw evidence"):
        contract._audit_diagnostic_evidence(
            forged, unit, path, manifest["analysis_policy"]
        )

    forged_diagnostics: dict[str, Any] = copy.deepcopy(diagnostics)
    forged_diagnostics["oracle"]["records"][0]["observed"]["gradient"][0] = 9.0
    forged_path = tmp_path / "forged-diagnostics.json"
    forged_path.write_bytes(canonical_json_bytes(forged_diagnostics))
    with pytest.raises(CausalContractError, match="raw observed/oracle arrays"):
        contract._audit_diagnostic_evidence(
            record, unit, forged_path, manifest["analysis_policy"]
        )

    nonminimal: dict[str, Any] = copy.deepcopy(diagnostics)
    trajectory = next(
        item
        for item in nonminimal["oracle"]["records"]
        if item["kind"] == "hash-selected-posterior-trajectory" and item["chain"] == 0
    )
    nonminimal_draw = next(
        draw for draw in range(unit.draws) if draw != trajectory["draw"]
    )
    trajectory["draw"] = nonminimal_draw
    trajectory["selection_sha256"] = hashlib.sha256(
        f"{unit.cell_id}:trajectory:0:{nonminimal_draw}".encode()
    ).hexdigest()
    nonminimal_path = tmp_path / "nonminimal-trajectory.json"
    nonminimal_path.write_bytes(canonical_json_bytes(nonminimal))
    with pytest.raises(CausalContractError, match="lowest-SHA draw"):
        contract._audit_diagnostic_evidence(
            record, unit, nonminimal_path, manifest["analysis_policy"]
        )


def test_raw_diagnostics_recompute_icdf_tail_and_branch_flags(
    manifest, tmp_path: Path
) -> None:
    import numpy as np

    unit = next(
        item
        for item in build_plan(manifest, "smoke")
        if item.representation_id == "full-icdf-noncentered"
        and item.backend_id == "pymc"
        and item.regime_id == "lower-outside-weak"
    )
    context = _context(pair_units(manifest, "smoke", unit.pair_id))
    result = _record(unit, context)
    for name in (
        "oracle_logp_scaled_error_max",
        "oracle_gradient_scaled_error_max",
        "oracle_hessian_scaled_error_max",
    ):
        result["metrics"][name] = 0.0
    gate = manifest["analysis_policy"]["oracle_gate"]
    tolerances = gate["component_tolerances"][unit.floatx]
    coordinate_size = int(unit.regime["n_groups"]) + 2
    zero_gradient = np.zeros(coordinate_size, dtype=np.float64)
    zero_hessian = np.zeros((coordinate_size, coordinate_size), dtype=np.float64)
    branch_epsilon = float(8.0 * np.sqrt(np.finfo(np.dtype(unit.floatx)).eps))

    def error_summary(observed: Any, expected: Any, component: str) -> dict[str, float]:
        if observed is None:
            return {
                "absolute_max": float(np.finfo(np.float64).max),
                "scaled_max": float(gate["scaled_error_max"]) + 1.0,
            }
        observed_array = np.asarray(observed, dtype=np.float64)
        expected_array = np.asarray(expected, dtype=np.float64)
        tolerance = tolerances[component]
        difference = np.abs(observed_array - expected_array)
        scale = float(tolerance["absolute_tolerance"]) + float(
            tolerance["relative_tolerance"]
        ) * np.maximum(np.abs(observed_array), np.abs(expected_array))
        return {
            "absolute_max": float(np.max(difference)),
            "scaled_max": float(np.max(difference / scale)),
        }

    def raw_record(
        point_id: str,
        kind: str,
        *,
        coordinate_index: int | None = None,
        observed_value: float | None = 0.0,
        observed_gradient: list[float] | None = None,
    ) -> dict[str, Any]:
        gradient = (
            zero_gradient.tolist() if observed_gradient is None else observed_gradient
        )
        observed = {
            "value": observed_value,
            "gradient": gradient,
            "hessian": zero_hessian.tolist(),
        }
        expected = {
            "value": 0.0,
            "gradient": zero_gradient.tolist(),
            "hessian": zero_hessian.tolist(),
        }
        finite = {
            component: observed[component] is not None
            for component in ("value", "gradient", "hessian")
        }
        errors = {
            "value": error_summary(observed["value"], expected["value"], "logp"),
            "gradient": error_summary(
                observed["gradient"], expected["gradient"], "gradient"
            ),
            "hessian": error_summary(
                observed["hessian"], expected["hessian"], "hessian"
            ),
        }
        passed = bool(
            all(finite.values())
            and all(
                summary["scaled_max"] <= float(gate["scaled_error_max"])
                for summary in errors.values()
            )
        )
        return {
            "point_id": point_id,
            "kind": kind,
            **(
                {
                    "coordinate_index": coordinate_index,
                    "branch_epsilon": branch_epsilon,
                }
                if coordinate_index is not None
                else {}
            ),
            "observed": observed,
            "oracle": expected,
            "errors": errors,
            "roundtrip": {"absolute_error_max": 0.0},
            "component_finite": finite,
            "finite": all(finite.values()),
            "passed": passed,
        }

    count = result["metrics"]["oracle_evaluation_count"]
    records = [raw_record("fixed-truth", "fixed-grid")]
    records.extend(
        raw_record(f"start-chain-{chain:02d}", "shared-natural-start")
        for chain in range(unit.chains)
    )
    for coordinate_index in (0, 2):
        records.extend(
            raw_record(
                f"icdf-{coordinate_index}-{label}",
                f"icdf-{label}",
                coordinate_index=coordinate_index,
            )
            for label in (
                "branch-left",
                "branch-zero",
                "branch-right",
                "tail-low",
                "tail-high",
            )
        )
    for chain in range(unit.chains):
        draw = min(
            range(unit.draws),
            key=lambda candidate: hashlib.sha256(
                f"{unit.cell_id}:trajectory:{chain}:{candidate}".encode()
            ).digest(),
        )
        trajectory = raw_record(
            f"trajectory-chain-{chain:02d}-selection-00",
            "hash-selected-posterior-trajectory",
        )
        trajectory.update(
            chain=chain,
            draw=draw,
            selection_sha256=hashlib.sha256(
                f"{unit.cell_id}:trajectory:{chain}:{draw}".encode()
            ).hexdigest(),
        )
        records.append(trajectory)
    assert len(records) == count
    diagnostics: dict[str, Any] = {
        "schema_version": unit.schema_version,
        "study_id": unit.study_id,
        "manifest_sha256": unit.manifest_sha256,
        "cell_id": unit.cell_id,
        "runtime": {
            "pytensor_floatx": unit.floatx,
            "jax_enable_x64": True,
        },
        "oracle": {
            "posterior_trajectory_evaluated": True,
            "records": records,
            "icdf_tail_finite": True,
            "icdf_branch_checks": [
                {"coordinate_index": coordinate_index, "passed": True}
                for coordinate_index in (0, 2)
            ],
            "icdf_branch_continuous": True,
            "passed": True,
        },
    }
    path = tmp_path / "valid-icdf.json"
    path.write_bytes(canonical_json_bytes(diagnostics))
    contract._audit_diagnostic_evidence(result, unit, path, manifest["analysis_policy"])

    missing_tail: dict[str, Any] = copy.deepcopy(diagnostics)
    missing_tail["oracle"]["records"] = [
        item
        for item in missing_tail["oracle"]["records"]
        if not (item["kind"] == "icdf-tail-high" and item["coordinate_index"] == 2)
    ]
    missing_tail_path = tmp_path / "missing-tail.json"
    missing_tail_path.write_bytes(canonical_json_bytes(missing_tail))
    with pytest.raises(CausalContractError, match="record count"):
        contract._audit_diagnostic_evidence(
            result,
            unit,
            missing_tail_path,
            manifest["analysis_policy"],
        )

    relabelled_branch: dict[str, Any] = copy.deepcopy(diagnostics)
    next(
        item
        for item in relabelled_branch["oracle"]["records"]
        if item["kind"] == "icdf-branch-left" and item["coordinate_index"] == 2
    )["kind"] = "fixed-grid"
    relabelled_branch_path = tmp_path / "relabelled-branch.json"
    relabelled_branch_path.write_bytes(canonical_json_bytes(relabelled_branch))
    with pytest.raises(CausalContractError, match="point kinds"):
        contract._audit_diagnostic_evidence(
            result,
            unit,
            relabelled_branch_path,
            manifest["analysis_policy"],
        )

    forged_tail: dict[str, Any] = copy.deepcopy(diagnostics)
    tail = next(
        item
        for item in forged_tail["oracle"]["records"]
        if item["kind"] == "icdf-tail-low"
    )
    tail["observed"]["gradient"] = None
    tail["errors"]["gradient"] = error_summary(None, zero_gradient, "gradient")
    tail["component_finite"]["gradient"] = False
    tail["finite"] = False
    tail["passed"] = False
    forged_tail_path = tmp_path / "forged-tail.json"
    forged_tail_path.write_bytes(canonical_json_bytes(forged_tail))
    with pytest.raises(CausalContractError, match="ICDF summary flags"):
        contract._audit_diagnostic_evidence(
            result,
            unit,
            forged_tail_path,
            manifest["analysis_policy"],
        )

    forged_branch: dict[str, Any] = copy.deepcopy(diagnostics)
    logp_absolute_tolerance = float(tolerances["logp"]["absolute_tolerance"])
    for kind, direction in (
        ("icdf-branch-left", -1.0),
        ("icdf-branch-right", 1.0),
    ):
        branch = next(
            item
            for item in forged_branch["oracle"]["records"]
            if item["kind"] == kind and item["coordinate_index"] == 0
        )
        branch["observed"]["value"] = direction * 0.75 * logp_absolute_tolerance
        branch["errors"]["value"] = error_summary(
            branch["observed"]["value"], 0.0, "logp"
        )
    forged_branch_path = tmp_path / "forged-branch.json"
    forged_branch_path.write_bytes(canonical_json_bytes(forged_branch))
    with pytest.raises(CausalContractError, match="branch-check summaries"):
        contract._audit_diagnostic_evidence(
            result,
            unit,
            forged_branch_path,
            manifest["analysis_policy"],
        )


def test_raw_chain_stats_recompute_registered_sampler_metrics(
    manifest, tmp_path: Path
) -> None:
    import numpy as np
    import xarray as xr

    unit = build_plan(manifest, "smoke")[0]
    context = _context(pair_units(manifest, "smoke", unit.pair_id))
    record = _record(unit, context)
    shape = (unit.chains, unit.draws)
    generator = np.random.default_rng(41)
    chain_rng = record["provenance"]["chain_rng_provenance"]
    dataset = xr.Dataset(
        {
            "group_location": (("chain", "draw"), generator.normal(size=shape)),
            "group_scale": (("chain", "draw"), generator.lognormal(size=shape)),
            "group_effect": (
                ("chain", "draw", "group"),
                generator.normal(size=(*shape, unit.regime["n_groups"])),
            ),
            "sample_stat__acceptance_rate": (
                ("chain", "draw"),
                np.full(shape, 0.9),
            ),
            "sample_stat__diverging": (
                ("chain", "draw"),
                np.zeros(shape, dtype=bool),
            ),
            "sample_stat__energy": (
                ("chain", "draw"),
                generator.normal(size=shape),
            ),
            "sample_stat__n_steps": (("chain", "draw"), np.full(shape, 7)),
            "sample_stat__step_size": (
                ("chain", "draw"),
                np.full(shape, 0.125),
            ),
            "sample_stat__tree_depth": (("chain", "draw"), np.full(shape, 3)),
        },
        attrs={
            "schema_version": unit.schema_version,
            "study_id": unit.study_id,
            "manifest_sha256": unit.manifest_sha256,
            "cell_id": unit.cell_id,
            "block_id": unit.block_id,
            "tier": unit.tier,
            "regime_id": unit.regime_id,
            "backend_id": unit.backend_id,
            "representation_id": unit.representation_id,
            "sampler_seed_input_json": json.dumps(
                record["provenance"]["sampler_seed_input"], separators=(",", ":")
            ),
            "chain_rng_provenance_json": json.dumps(
                chain_rng, separators=(",", ":"), sort_keys=True
            ),
        },
    )
    path = tmp_path / "chains.nc"
    dataset.to_netcdf(path, engine="scipy", format="NETCDF3_64BIT")
    contract._audit_chain_evidence(record, unit, path)
    forged = copy.deepcopy(record)
    forged["metrics"]["divergence_count"] = 1
    with pytest.raises(CausalContractError, match="hash-bound raw evidence"):
        contract._audit_chain_evidence(forged, unit, path)


def test_family_health_requires_eight_of_eight() -> None:
    rows = _classifier_rows(left_bad_replicates=0)
    selected = [
        row
        for row in rows
        if row["backend_id"] == "pymc"
        and row["representation_id"] == "group-icdf-noncentered"
    ]
    policy = {
        "aggregate_divergence_rate_lt": 0.001,
        "per_fit_pass_fraction_ge": 0.95,
        "maximum_failed_replicates": 0,
    }
    per_fit_policy = load_manifest()["analysis_policy"]["per_fit_health"]
    assert contract._family_health(selected, "confirmation", policy, per_fit_policy)
    selected[-1] = copy.deepcopy(selected[-1])
    selected[-1]["metrics"]["divergence_count"] = 4000
    selected[-1]["metrics"]["divergence_rate"] = 1.0
    assert not contract._family_health(selected, "confirmation", policy, per_fit_policy)


@pytest.mark.parametrize(
    ("bad_replicates", "expected"),
    [
        (1, "mixed-inconclusive"),
        (7, "mixed-inconclusive"),
        (8, "group-conditional-centering"),
    ],
)
def test_classifier_requires_all_eight_directional_pairs(
    manifest, bad_replicates, expected
) -> None:
    rows = _classifier_rows(left_bad_replicates=bad_replicates)
    result = contract._classify_regime(
        rows,
        manifest["regimes"][0],
        "confirmation",
        manifest["analysis_policy"],
    )
    assert result["classification"] == expected
    contrast = result["paired_health_contrasts"]["pymc/group-effect-at-location-c"]
    assert contrast["supports_direction"] is (bad_replicates == 8)
    if bad_replicates == 8:
        assert contrast["two_sided_exact_p"] == 0.0078125


def test_sampling_classification_requires_aggregate_family_health(manifest) -> None:
    rows = _classifier_rows(left_bad_replicates=8)
    for row in rows:
        if row["representation_id"] in {
            "group-icdf-noncentered",
            "full-icdf-noncentered",
        }:
            row["metrics"]["divergence_count"] = 8
            row["metrics"]["divergence_rate"] = 0.002
    result = contract._classify_regime(
        rows,
        manifest["regimes"][0],
        "confirmation",
        manifest["analysis_policy"],
    )
    assert result["classification"] == "mixed-inconclusive"
    assert result["paired_health_contrasts"]["pymc/group-effect-at-location-c"][
        "supports_direction"
    ]
    assert not result["family_health"]["pymc/group-icdf-noncentered"]


@pytest.mark.parametrize(
    ("bad_replicates", "expected"),
    [
        (1, "mixed-inconclusive"),
        (7, "mixed-inconclusive"),
        (8, "joint-centering-interaction"),
    ],
)
def test_joint_classifier_requires_the_pattern_in_every_pair(
    manifest, bad_replicates, expected
) -> None:
    rows = _classifier_rows(left_bad_replicates=0)
    for row in rows:
        if row["replicate"] < bad_replicates and row["representation_id"] in {
            "manual-centered",
            "group-icdf-noncentered",
            "location-icdf-noncentered",
        }:
            row["metrics"]["divergence_count"] = 4000
            row["metrics"]["divergence_rate"] = 1.0
    result = contract._classify_regime(
        rows,
        manifest["regimes"][0],
        "confirmation",
        manifest["analysis_policy"],
    )
    assert result["classification"] == expected


@pytest.mark.parametrize(
    ("bad_replicates", "expected"),
    [
        (1, "mixed-inconclusive"),
        (7, "mixed-inconclusive"),
        (8, "residual-tn-or-scale-geometry"),
    ],
)
def test_residual_classifier_requires_every_fit_to_be_unhealthy(
    manifest, bad_replicates, expected
) -> None:
    rows = _classifier_rows(left_bad_replicates=0)
    for row in rows:
        if row["replicate"] < bad_replicates:
            row["metrics"]["divergence_count"] = 4000
            row["metrics"]["divergence_rate"] = 1.0
    result = contract._classify_regime(
        rows,
        manifest["regimes"][0],
        "confirmation",
        manifest["analysis_policy"],
    )
    assert result["classification"] == expected


def test_sampling_classifier_fails_closed_on_non_native_oracle_failure(
    manifest,
) -> None:
    rows = _classifier_rows(left_bad_replicates=8)
    affected = next(
        row for row in rows if row["representation_id"] == "group-icdf-noncentered"
    )
    affected["metrics"]["oracle_hessian_scaled_error_max"] = 1.01
    result = contract._classify_regime(
        rows,
        manifest["regimes"][0],
        "confirmation",
        manifest["analysis_policy"],
    )
    assert result["classification"] == "mixed-inconclusive"


@pytest.mark.parametrize(
    ("bad_replicates", "expected"),
    [
        (1, "mixed-inconclusive"),
        (7, "mixed-inconclusive"),
        (8, "backend-path-specific"),
    ],
)
def test_backend_label_uses_one_block_health_count_omnibus(
    manifest, bad_replicates, expected
) -> None:
    rows = _classifier_rows(left_bad_replicates=0)
    for row in rows:
        if (
            row["backend_id"] == "pymc"
            and row["representation_id"] == "native-centered"
            and row["replicate"] < bad_replicates
        ):
            row["metrics"]["divergence_count"] = 4000
            row["metrics"]["divergence_rate"] = 1.0
    result = contract._classify_regime(
        rows,
        manifest["regimes"][0],
        "confirmation",
        manifest["analysis_policy"],
    )
    assert result["classification"] == expected
    omnibus = result["paired_health_contrasts"][
        "backend/five-form-health-count-omnibus"
    ]
    assert omnibus["supports_either_direction"] is (bad_replicates == 8)
    if bad_replicates == 8:
        assert omnibus["two_sided_exact_p"] == 0.0078125


def test_opposite_representation_backend_effects_are_descriptive_only(
    manifest,
) -> None:
    rows = _classifier_rows(left_bad_replicates=0)
    for row in rows:
        unhealthy = (
            row["backend_id"] == "pymc"
            and row["representation_id"] == "native-centered"
        ) or (
            row["backend_id"] == "numpyro"
            and row["representation_id"] == "manual-centered"
        )
        if unhealthy:
            row["metrics"]["divergence_count"] = 4000
            row["metrics"]["divergence_rate"] = 1.0
    result = contract._classify_regime(
        rows,
        manifest["regimes"][0],
        "confirmation",
        manifest["analysis_policy"],
    )
    assert result["classification"] == "mixed-inconclusive"
    assert result["paired_health_contrasts"]["backend/native-centered/pymc-to-numpyro"][
        "all_directional_at_threshold"
    ]
    assert result["paired_health_contrasts"]["backend/manual-centered/pymc-to-numpyro"][
        "all_directional_at_threshold"
    ]
    assert not result["paired_health_contrasts"][
        "backend/five-form-health-count-omnibus"
    ]["supports_either_direction"]


@pytest.mark.parametrize("failure_kind", ["sampling", "roundtrip"])
def test_native_non_density_failures_are_not_correctness_defects(
    manifest, failure_kind
) -> None:
    rows = _classifier_rows(left_bad_replicates=0)
    affected = [
        row
        for row in rows
        if row["representation_id"] == "native-centered" and row["replicate"] == 0
    ]
    assert len(affected) == 2
    for row in affected:
        if failure_kind == "sampling":
            row["execution_status"] = "failed"
        else:
            row["metrics"]["roundtrip_absolute_error_max"] = 1.0
    result = contract._classify_regime(
        rows,
        manifest["regimes"][0],
        "confirmation",
        manifest["analysis_policy"],
    )
    assert result["classification"] == "mixed-inconclusive"


def test_one_dataset_seen_on_both_backends_is_not_reproducible_native_defect(
    manifest,
) -> None:
    rows = _classifier_rows(left_bad_replicates=0)
    affected = [
        row
        for row in rows
        if row["representation_id"] == "native-centered" and row["replicate"] == 0
    ]
    assert len(affected) == 2
    for row in affected:
        row["metrics"]["oracle_gradient_scaled_error_max"] = 1.01
    result = contract._classify_regime(
        rows,
        manifest["regimes"][0],
        "confirmation",
        manifest["analysis_policy"],
    )
    assert result["classification"] == "mixed-inconclusive"
    assert result["native_mismatch_replicates"] == [0]


def test_two_dataset_native_derivative_mismatch_is_correctness_defect(
    manifest,
) -> None:
    rows = _classifier_rows(left_bad_replicates=0)
    affected = [
        row
        for row in rows
        if row["backend_id"] == "pymc"
        and row["representation_id"] == "native-centered"
        and row["replicate"] in {0, 1}
    ]
    assert len(affected) == 2
    for row in affected:
        row["metrics"]["oracle_gradient_scaled_error_max"] = 1.01
    result = contract._classify_regime(
        rows,
        manifest["regimes"][0],
        "confirmation",
        manifest["analysis_policy"],
    )
    assert result["classification"] == "native-pymc-correctness-defect"
    assert result["native_mismatch_replicates"] == [0, 1]


def test_native_correctness_can_use_pre_oracle_when_sampling_later_fails(
    manifest,
) -> None:
    rows = _classifier_rows(left_bad_replicates=0)
    for row in rows:
        if (
            row["backend_id"] == "pymc"
            and row["representation_id"] in {"native-centered", "manual-centered"}
            and row["replicate"] in {0, 1}
        ):
            row["execution_status"] = "failed"
            row["metrics"]["sampling_success"] = False
            row["metrics"]["oracle_evaluation_count"] -= 4
            if row["representation_id"] == "native-centered":
                row["metrics"]["oracle_hessian_scaled_error_max"] = 1.01
    result = contract._classify_regime(
        rows,
        manifest["regimes"][0],
        "confirmation",
        manifest["analysis_policy"],
    )
    assert result["classification"] == "native-pymc-correctness-defect"
    assert result["native_mismatch_replicates"] == [0, 1]


def test_aggregate_materializes_missing_and_assessment_allows_scientific_failures(
    manifest,
) -> None:
    plan = build_plan(manifest, "smoke")
    contexts = {
        unit.pair_id: _context(pair_units(manifest, "smoke", unit.pair_id))
        for unit in plan
    }
    records = {
        unit.cell_id: _record(
            unit,
            contexts[unit.pair_id],
            healthy=unit.representation_id != "native-centered",
        )
        for unit in plan
    }
    rows = aggregate_results(plan, records)
    assessment = assess_results(rows, plan, manifest, "smoke")
    assert assessment["outcome"] == "screening-fail"
    assert assessment["contract_valid"] is True
    assert assessment["evidence_complete"] is True
    assert assessment["proceed_to_confirmation"] is True
    assert {item["classification"] for item in assessment["regimes"]} == {
        "screening-only"
    }
    assert all(item["causal_classification"] is None for item in assessment["regimes"])
    missing_rows = aggregate_results(
        plan, {key: value for key, value in records.items() if key != plan[0].cell_id}
    )
    incomplete = assess_results(missing_rows, plan, manifest, "smoke")
    assert incomplete["outcome"] == "incomplete"
    assert incomplete["evidence_complete"] is False
    assert incomplete["proceed_to_confirmation"] is False


def test_cli_validate_plan_and_matrix(manifest, tmp_path: Path, capsys) -> None:
    assert contract.main(["validate"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["manifest_sha256"] == manifest_digest(manifest)
    assert contract.main(["matrix", "--tier", "smoke"]) == 0
    matrix = json.loads(capsys.readouterr().out)
    assert len(matrix["include"]) == 2
    assert set(matrix["include"][0]) == {"pair_id", "tier", "regime_id", "replicate"}
    output = tmp_path / "plan"
    assert contract.main(["plan", "--tier", "smoke", "--output-dir", str(output)]) == 0
    capsys.readouterr()
    assert len((output / "plan.jsonl").read_text().splitlines()) == 20
    assert len((output / "matrix.csv").read_text().splitlines()) == 21

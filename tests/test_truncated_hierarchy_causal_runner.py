"""Tests for the isolated backend-paired causal sampling runner."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest
import xarray as xr

import scripts.truncated_hierarchy_causal_runner as runner
from scripts.truncated_hierarchy_causal_artifacts import (
    ArtifactRef,
    ArtifactStore,
    CausalArtifactError,
    decode_canonical_json,
)
from scripts.truncated_hierarchy_causal_contract import (
    DEFAULT_MANIFEST,
    RUNNER_VERSION,
    CausalContractError,
    RunContext,
    aggregate_results,
    build_plan,
    canonical_json_bytes,
    environment_digest,
    load_manifest,
    manifest_digest,
    pair_units,
    sha256_bytes,
    validate_result_record,
    verify_result_artifacts,
)
from scripts.truncated_hierarchy_causal_models import build_causal_model
from scripts.truncated_hierarchy_models import generate_synthetic_data

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def manifest():
    """Load the real frozen causal contract once for focused runner tests."""
    return load_manifest(DEFAULT_MANIFEST)


def _context(units, manifest) -> RunContext:
    profile = manifest["dependency_profile"]
    environment = {
        "schema_version": 3,
        "study_id": manifest["study_id"],
        "manifest_sha256": manifest_digest(manifest),
        "runner_version": RUNNER_VERSION,
        "dependency_profile": profile["name"],
        "git": {"commit": "2" * 40, "branch": "test", "dirty": False},
        "project": {
            "path": profile["project_path"],
            "sha256": profile["project_sha256"],
            "lock_path": profile["lock_path"],
            "lock_sha256": profile["lock_sha256"],
        },
        "runtime": {
            "python": "3.12.0",
            "implementation": "CPython",
            "executable": "/test/python",
            "platform": "test",
        },
        "packages": dict(profile["required_versions"]),
    }
    environment["environment_sha256"] = environment_digest(environment)
    return RunContext(
        schema_version=3,
        study_id=units[0].study_id,
        manifest_sha256=units[0].manifest_sha256,
        pair_id=units[0].pair_id,
        block_ids=tuple(dict.fromkeys(unit.block_id for unit in units)),
        cell_ids=tuple(unit.cell_id for unit in units),
        execution_order=tuple(unit.cell_id for unit in units),
        environment=environment,
        environment_sha256=environment["environment_sha256"],
        git_commit="2" * 40,
        worker_identity_sha256="3" * 64,
        pair_execution_id="4" * 64,
        execution_attempt_ids=tuple(f"{index + 5:064x}" for index in range(10)),
    )


def _natural_dataset(unit) -> xr.Dataset:
    rng = np.random.default_rng(1282)
    shape = (unit.chains, unit.draws)
    lower = float(unit.regime["lower"])
    upper = unit.regime["upper"]
    location = lower + 0.2 + 0.01 * rng.normal(size=shape)
    if upper is not None:
        location = np.clip(location, lower + 0.01, float(upper) - 0.01)
    scale = np.exp(-1.2 + 0.02 * rng.normal(size=shape))
    effects = np.repeat(location[..., None], int(unit.regime["n_groups"]), axis=2)
    posterior = xr.Dataset(
        {
            "group_location": (("chain", "draw"), location),
            "group_scale": (("chain", "draw"), scale),
            "group_effect": (("chain", "draw", "group"), effects),
        },
        coords={
            "chain": np.arange(unit.chains),
            "draw": np.arange(unit.draws),
            "group": np.arange(int(unit.regime["n_groups"])),
        },
    )
    sequence = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    sample_stats = xr.Dataset(
        {
            "acceptance_rate": (("chain", "draw"), np.full(shape, 0.91)),
            "diverging": (("chain", "draw"), np.zeros(shape, dtype=bool)),
            "energy": (("chain", "draw"), np.sin(sequence / 11.0)),
            "n_steps": (("chain", "draw"), np.full(shape, 7)),
            "step_size": (("chain", "draw"), np.full(shape, 0.125)),
            "tree_depth": (("chain", "draw"), np.full(shape, 3)),
        }
    )
    return runner._standardize_natural_chains(
        unit, SimpleNamespace(posterior=posterior, sample_stats=sample_stats)
    )


@pytest.fixture(scope="module")
def native_oracle_artifacts(manifest):
    """Build one exact, no-sampling artifact set for provenance audits."""
    unit = next(
        item
        for item in build_plan(manifest, "smoke")
        if item.backend_id == "pymc"
        and item.representation_id == "native-centered"
        and item.floatx == "float64"
    )
    data = generate_synthetic_data(
        runner._toy_spec(unit),
        group_seed=unit.group_seed,
        observation_seed=unit.observation_seed,
    )
    prior = runner._prior(unit)
    data_payload = runner._data_payload(unit, data)
    start_payload = runner._natural_start_payload(unit, data, prior)
    model = build_causal_model(unit.builder, prior, data)
    oracle_spec = runner._oracle_spec(unit, prior, data)
    coordinate_payload, _ = runner._coordinate_start_payload(
        unit, model, start_payload, prior, oracle_spec
    )
    parameterization = runner.REPRESENTATION_TO_PARAMETERIZATION[unit.representation_id]

    def exact_evaluator(vector):
        expected = runner.hierarchical_posterior_components(
            vector, oracle_spec, parameterization
        ).total
        return expected.value, expected.gradient, expected.hessian

    pre_oracle = runner._oracle_diagnostics(
        unit,
        model,
        prior,
        data,
        start_payload,
        None,
        manifest["analysis_policy"],
        evaluator=exact_evaluator,
    )
    chains = _natural_dataset(unit)
    full_oracle = runner._oracle_diagnostics(
        unit,
        model,
        prior,
        data,
        start_payload,
        chains,
        manifest["analysis_policy"],
        evaluator=exact_evaluator,
        prior_records=pre_oracle["records"],
        include_static=False,
    )
    return {
        "unit": unit,
        "data": data,
        "prior": prior,
        "model": model,
        "oracle_spec": oracle_spec,
        "data_payload": data_payload,
        "start_payload": start_payload,
        "coordinate_payload": coordinate_payload,
        "chains": chains,
        "pre_oracle": pre_oracle,
        "full_oracle": full_oracle,
    }


def _completed_metrics(unit) -> dict[str, object]:
    layers = {
        "native-centered": 0,
        "manual-centered": 0,
        "group-icdf-noncentered": 1,
        "location-icdf-noncentered": 1,
        "full-icdf-noncentered": 2,
    }[unit.representation_id]
    oracle_count = unit.chains + int(unit.replicate == 0) + 5 * layers + unit.chains
    return {
        "compile_success": True,
        "initialization_success": True,
        "logp_finite": True,
        "gradient_finite": True,
        "sampling_success": True,
        "divergence_count": 0,
        "posterior_draw_count": unit.chains * unit.draws,
        "divergence_rate": 0.0,
        "sampling_elapsed_seconds": 1.0,
        "step_size_final_min": 0.01,
        "step_size_final_max": 0.02,
        "leapfrog_step_count": 1000.0,
        "oracle_evaluation_count": oracle_count,
        "oracle_logp_scaled_error_max": 0.0,
        "oracle_gradient_scaled_error_max": 0.0,
        "oracle_hessian_scaled_error_max": 0.0,
        "roundtrip_absolute_error_max": 0.0,
        "icdf_tail_finite": True,
        "icdf_branch_continuous": True,
    }


def _synthetic_oracle(unit, *, trajectory: bool) -> dict[str, object]:
    size = int(unit.regime["n_groups"]) + 2
    zeros = [0.0] * size
    zero_hessian = [[0.0] * size for _ in range(size)]
    zero_error = {"absolute_max": 0.0, "scaled_max": 0.0}

    def record(
        point_id,
        kind,
        *,
        coordinate_index=None,
        epsilon=None,
        chain=None,
        draw=None,
        selection_sha256=None,
    ):
        value = {
            "point_id": point_id,
            "kind": kind,
            "coordinate_vector": zeros,
            "roundtrip": {"absolute_error_max": 0.0},
            "observed": {
                "value": 0.0,
                "gradient": zeros,
                "hessian": zero_hessian,
            },
            "oracle": {
                "value": 0.0,
                "gradient": zeros,
                "hessian": zero_hessian,
            },
            "errors": {
                "value": dict(zero_error),
                "gradient": dict(zero_error),
                "hessian": dict(zero_error),
            },
            "component_finite": {
                "value": True,
                "gradient": True,
                "hessian": True,
            },
            "finite": True,
            "passed": True,
        }
        if coordinate_index is not None:
            value["coordinate_index"] = coordinate_index
        if epsilon is not None:
            value["branch_epsilon"] = epsilon
        if chain is not None:
            value["chain"] = chain
        if draw is not None:
            value["draw"] = draw
        if selection_sha256 is not None:
            value["selection_sha256"] = selection_sha256
        return value

    records = []
    if unit.replicate == 0:
        records.append(record("fixed-truth", "fixed-grid"))
    records.extend(
        record(f"start-chain-{chain:02d}", "shared-natural-start")
        for chain in range(unit.chains)
    )
    parameterization = runner.REPRESENTATION_TO_PARAMETERIZATION[unit.representation_id]
    indices = []
    if parameterization in {"location_icdf_noncentered", "full_icdf_noncentered"}:
        indices.append(0)
    if parameterization in {"group_icdf_noncentered", "full_icdf_noncentered"}:
        indices.append(2)
    epsilon = float(8.0 * math.sqrt(np.finfo(np.dtype(unit.floatx)).eps))
    branch_checks = []
    for coordinate_index in indices:
        for label in ("branch-left", "branch-zero", "branch-right"):
            records.append(
                record(
                    f"icdf-{coordinate_index}-{label}",
                    f"icdf-{label}",
                    coordinate_index=coordinate_index,
                    epsilon=epsilon,
                )
            )
        for label in ("tail-low", "tail-high"):
            records.append(
                record(
                    f"icdf-{coordinate_index}-{label}",
                    f"icdf-{label}",
                    coordinate_index=coordinate_index,
                    epsilon=epsilon,
                )
            )
        branch_checks.append(
            {
                "coordinate_index": coordinate_index,
                "epsilon": epsilon,
                "left_point_id": f"icdf-{coordinate_index}-branch-left",
                "zero_point_id": f"icdf-{coordinate_index}-branch-zero",
                "right_point_id": f"icdf-{coordinate_index}-branch-right",
                "observed_value_jump": 0.0,
                "oracle_value_jump": 0.0,
                "observed_gradient_jump": zeros,
                "oracle_gradient_jump": zeros,
                "value_jump_error": dict(zero_error),
                "gradient_jump_error": dict(zero_error),
                "passed": True,
            }
        )
    if trajectory:
        for chain in range(unit.chains):
            draw = min(
                range(unit.draws),
                key=lambda candidate: hashlib.sha256(
                    f"{unit.cell_id}:trajectory:{chain}:{candidate}".encode()
                ).digest(),
            )
            selection_sha256 = hashlib.sha256(
                f"{unit.cell_id}:trajectory:{chain}:{draw}".encode()
            ).hexdigest()
            records.append(
                record(
                    f"trajectory-chain-{chain:02d}-selection-00",
                    "hash-selected-posterior-trajectory",
                    chain=chain,
                    draw=draw,
                    selection_sha256=selection_sha256,
                )
            )
    return {
        "backend_id": unit.backend_id,
        "representation_id": unit.representation_id,
        "parameterization": parameterization,
        "point_selection": {
            "fixed_grid": "replicate-zero-truth",
            "starts": "every-shared-natural-chain-start",
            "trajectory": "lowest-sha256-per-chain",
            "trajectory_points_per_chain": 1,
        },
        "posterior_trajectory_evaluated": trajectory,
        "records": records,
        "icdf_tail_finite": True,
        "icdf_branch_checks": branch_checks,
        "icdf_branch_continuous": True,
        "passed": True,
    }


def _runtime_diagnostics(unit, *, oracle=None) -> dict[str, object]:
    result = {
        "schema_version": unit.schema_version,
        "study_id": unit.study_id,
        "manifest_sha256": unit.manifest_sha256,
        "cell_id": unit.cell_id,
        "runtime": {
            "process_id": 1282,
            "cache_identity_sha256": "5" * 64,
            "pytensor_floatx": unit.floatx,
            "jax_enable_x64": unit.floatx == "float64",
            "jax_platform": "cpu",
        },
    }
    if oracle is not None:
        result["oracle"] = oracle
    return result


def test_shared_natural_starts_roundtrip_through_all_five_forms(manifest) -> None:
    """One byte-identical start maps bijectively into the complete 2x2 panel."""
    units = pair_units(manifest, "smoke", build_plan(manifest, "smoke")[0].pair_id)
    reference = units[0]
    data = generate_synthetic_data(
        runner._toy_spec(reference),
        group_seed=reference.group_seed,
        observation_seed=reference.observation_seed,
    )
    prior = runner._prior(reference)
    starts = runner._natural_start_payload(reference, data, prior)

    for unit in units:
        model = build_causal_model(
            runner.BUILDER_TO_PARAMETERIZATION[unit.builder], prior, data
        )
        payload, initvals = runner._coordinate_start_payload(
            unit, model, starts, prior, runner._oracle_spec(unit, prior, data)
        )
        assert len(initvals) == unit.chains
        assert (
            max(chain["roundtrip"]["absolute_error_max"] for chain in payload["chains"])
            < 1e-10
        )


def test_natural_start_validation_rejects_support_valid_forgery(manifest) -> None:
    """Support checks cannot substitute for the exact seeded intervention."""
    unit = next(
        item for item in build_plan(manifest, "smoke") if item.floatx == "float64"
    )
    data = generate_synthetic_data(
        runner._toy_spec(unit),
        group_seed=unit.group_seed,
        observation_seed=unit.observation_seed,
    )
    prior = runner._prior(unit)
    payload = runner._natural_start_payload(unit, data, prior)
    forged = copy.deepcopy(payload)
    forged["chains"][0]["group_location"] += 1e-6

    with pytest.raises(runner.CausalRunnerError, match="deterministic regeneration"):
        runner._validate_natural_start_payload(forged, unit, prior, data)


def test_icdf_branch_gate_uses_symmetric_neighbors_and_detects_jump(manifest) -> None:
    """Continuity is an explicit left/zero/right comparison, not a zero probe."""
    unit = next(
        item
        for item in build_plan(manifest, "smoke")
        if item.floatx == "float64"
        and item.backend_id == "pymc"
        and item.representation_id == "group-icdf-noncentered"
    )
    data = generate_synthetic_data(
        runner._toy_spec(unit),
        group_seed=unit.group_seed,
        observation_seed=unit.observation_seed,
    )
    prior = runner._prior(unit)
    model = build_causal_model(unit.builder, prior, data)
    starts = runner._natural_start_payload(unit, data, prior)
    spec = runner._oracle_spec(unit, prior, data)
    parameterization = runner.REPRESENTATION_TO_PARAMETERIZATION[unit.representation_id]

    def discontinuous_evaluator(vector):
        expected = runner.hierarchical_posterior_components(
            vector, spec, parameterization
        ).total
        value = expected.value
        gradient = np.asarray(expected.gradient).copy()
        if vector[2] > 0:
            value += 1.0
            gradient[2] += 1.0
        return value, gradient, np.asarray(expected.hessian)

    diagnostics = runner._oracle_diagnostics(
        unit,
        model,
        prior,
        data,
        starts,
        None,
        manifest["analysis_policy"],
        evaluator=discontinuous_evaluator,
    )

    branch_records = [
        record
        for record in diagnostics["records"]
        if record["kind"].startswith("icdf-branch")
    ]
    coordinates = {
        record["kind"]: record["coordinate_vector"][2] for record in branch_records
    }
    assert set(coordinates) == {
        "icdf-branch-left",
        "icdf-branch-zero",
        "icdf-branch-right",
    }
    assert coordinates["icdf-branch-left"] == pytest.approx(
        -coordinates["icdf-branch-right"], abs=1e-15
    )
    assert coordinates["icdf-branch-zero"] == 0.0
    assert diagnostics["icdf_branch_checks"][0]["passed"] is False
    assert diagnostics["icdf_branch_continuous"] is False


def test_nonfinite_backend_oracle_is_finite_publishable_evidence(manifest) -> None:
    """NaN backend derivatives become explicit failed evidence, never JSON NaN."""
    unit = next(
        item
        for item in build_plan(manifest, "smoke")
        if item.floatx == "float64"
        and item.backend_id == "pymc"
        and item.representation_id == "native-centered"
    )
    data = generate_synthetic_data(
        runner._toy_spec(unit),
        group_seed=unit.group_seed,
        observation_seed=unit.observation_seed,
    )
    prior = runner._prior(unit)
    model = build_causal_model(unit.builder, prior, data)
    starts = runner._natural_start_payload(unit, data, prior)
    spec = runner._oracle_spec(unit, prior, data)
    parameterization = runner.REPRESENTATION_TO_PARAMETERIZATION[unit.representation_id]

    def nonfinite_evaluator(vector):
        expected = runner.hierarchical_posterior_components(
            vector, spec, parameterization
        ).total
        return (
            float("nan"),
            np.full_like(expected.gradient, np.nan),
            np.full_like(expected.hessian, np.nan),
        )

    diagnostics = runner._oracle_diagnostics(
        unit,
        model,
        prior,
        data,
        starts,
        None,
        manifest["analysis_policy"],
        evaluator=nonfinite_evaluator,
    )
    metrics = runner._registered_oracle_metrics(diagnostics)

    assert all(record["finite"] is False for record in diagnostics["records"])
    assert all(record["observed"]["value"] is None for record in diagnostics["records"])
    assert metrics["oracle_logp_scaled_error_max"] > 1
    assert metrics["oracle_gradient_scaled_error_max"] > 1
    assert metrics["oracle_hessian_scaled_error_max"] > 1
    runner.canonical_json_bytes(diagnostics)


def test_shared_input_artifacts_are_byte_exact_across_forms_and_backends(
    manifest, tmp_path: Path
) -> None:
    """Every candidate and backend consumes one immutable intervention pair."""
    cohort = [
        unit
        for unit in build_plan(manifest, "smoke")
        if unit.regime_id == "lower-outside-weak"
    ]
    store = ArtifactStore(tmp_path.resolve())

    references = [runner.materialize_inputs_for_unit(unit, store) for unit in cohort]

    assert len({data.sha256 for data, _start in references}) == 1
    assert len({start.sha256 for _data, start in references}) == 1
    assert len({data.path for data, _start in references}) == 1
    assert len({start.path for _data, start in references}) == 1


@pytest.mark.slow
@pytest.mark.parametrize("ambient_floatx", ["float64", "float32"])
def test_public_materialization_uses_each_regimes_planned_source_precision(
    manifest, tmp_path: Path, ambient_floatx: str
) -> None:
    """Mixed-precision tiers re-exec the opposite regime in a fresh process."""
    repository = DEFAULT_MANIFEST.parents[2]
    run_dir = (tmp_path / "run").resolve()
    cache = tmp_path / "cache"
    for name in ("pytensor", "jax", "matplotlib", "xdg"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "PYTENSOR_FLAGS": (
                f"base_compiledir={cache / 'pytensor'},floatX={ambient_floatx}"
            ),
            "JAX_COMPILATION_CACHE_DIR": str(cache / "jax"),
            "JAX_ENABLE_X64": "true" if ambient_floatx == "float64" else "false",
            "JAX_PLATFORMS": "cpu",
            "MPLCONFIGDIR": str(cache / "matplotlib"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/truncated_hierarchy_causal_runner.py",
            "--manifest",
            str(DEFAULT_MANIFEST),
            "materialize-inputs",
            "--tier",
            "smoke",
            "--run-dir",
            str(run_dir),
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    representatives: dict[str, runner.UnitSpec] = {}
    for unit in build_plan(manifest, "smoke"):
        representatives.setdefault(unit.data_id, unit)
    for unit in representatives.values():
        start = decode_canonical_json(
            (run_dir / "starts" / "natural" / f"{unit.start_id}.json").read_bytes()
        )
        data = decode_canonical_json(
            (run_dir / "data" / f"{unit.data_id}.json").read_bytes()
        )
        assert start["source_graph"]["pytensor_floatx"] == unit.floatx
        assert set(start["source_graph"]["value_variable_dtypes"]) == {unit.floatx}
        assert data["spec"]["floatx"] == unit.floatx


@pytest.mark.slow
def test_float32_coordinate_payload_uses_graph_representable_vector(
    tmp_path: Path,
) -> None:
    """Recorded round trips use the float32 point actually passed to the graph."""
    cache = tmp_path / "cache"
    for name in ("pytensor", "jax", "matplotlib", "xdg"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "PYTENSOR_FLAGS": f"base_compiledir={cache / 'pytensor'},floatX=float32",
            "JAX_COMPILATION_CACHE_DIR": str(cache / "jax"),
            "JAX_ENABLE_X64": "false",
            "JAX_PLATFORMS": "cpu",
            "MPLCONFIGDIR": str(cache / "matplotlib"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
        }
    )
    program = textwrap.dedent(
        """
        import numpy as np
        from scripts import truncated_hierarchy_causal_runner as runner
        from scripts.truncated_hierarchy_causal_contract import (
            DEFAULT_MANIFEST,
            build_plan,
            load_manifest,
        )

        manifest = load_manifest(DEFAULT_MANIFEST)
        unit = next(
            item for item in build_plan(manifest, "smoke")
            if item.floatx == "float32"
            and item.backend_id == "pymc"
            and item.representation_id == "full-icdf-noncentered"
        )
        data = runner.generate_synthetic_data(
            runner._toy_spec(unit),
            group_seed=unit.group_seed,
            observation_seed=unit.observation_seed,
        )
        prior = runner._prior(unit)
        model = runner.build_causal_model(unit.builder, prior, data)
        starts = runner._natural_start_payload(unit, data, prior)
        spec = runner._oracle_spec(unit, prior, data)
        payload, initvals = runner._coordinate_start_payload(
            unit, model, starts, prior, spec
        )
        parameterization = runner.REPRESENTATION_TO_PARAMETERIZATION[
            unit.representation_id
        ]
        maximum = 0.0
        for record, point in zip(payload["chains"], initvals, strict=True):
            graph_vector = runner._pack_model_point(model, point)
            if graph_vector.dtype != np.dtype("float32"):
                raise AssertionError(graph_vector.dtype)
            np.testing.assert_array_equal(
                graph_vector,
                np.asarray(record["coordinate_vector"], dtype=np.float32),
            )
            roundtrip = runner._natural_coordinate_roundtrip(
                record["natural"],
                graph_vector,
                oracle_spec=spec,
                parameterization=parameterization,
            )
            if roundtrip != record["roundtrip"]:
                raise AssertionError("stored roundtrip differs from graph point")
            maximum = max(maximum, roundtrip["absolute_error_max"])
        print(runner.canonical_json_bytes({"maximum": maximum}).decode(), end="")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=DEFAULT_MANIFEST.parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = decode_canonical_json(completed.stdout.encode())
    assert 0.0 <= result["maximum"] <= 2e-5


def test_backend_seed_shapes_and_numpyro_split_keys_are_exact(manifest) -> None:
    """Record PyMC spawned integers and NumPyro split keys, not seed inputs."""
    plan = build_plan(manifest, "confirmation")
    pymc_unit = next(unit for unit in plan if unit.backend_id == "pymc")
    numpyro_unit = next(unit for unit in plan if unit.backend_id == "numpyro")

    sampler_input, pymc_streams = runner._sampler_rng_provenance(pymc_unit)
    assert sampler_input == [943009182, 939311901, 1620816411, 1994453597]
    assert [stream["init_step_seed"] for stream in pymc_streams] == [
        1019536840,
        896392474,
        417781057,
        822370369,
    ]
    assert [stream["spawn_key"] for stream in pymc_streams] == [
        [0],
        [1],
        [2],
        [3],
    ]
    assert [stream["post_init_draw_state_sha256"] for stream in pymc_streams] == [
        "68ed6b06d0544f5ee8e81dbdaaef0fcadfb2ada704e93f21e6c662b9ab117ed5",
        "9f48f6b2865cff458c5d73a80e41a9b3bc3b47cbc26c1837adedf50911657fdb",
        "f63f284505c3e569d75bb43480a8b3a3dd2e54c469973d5dee90c338c897a826",
        "4140b920ccbddae64419e03cb21ce599b81cecfe500c1f3ab2f5a976f1f76238",
    ]
    master, numpyro_streams = runner._sampler_rng_provenance(numpyro_unit)
    expected = np.asarray(
        jax.random.split(jax.random.PRNGKey(numpyro_unit.sampler_seed), 4),
        dtype=np.uint32,
    ).astype(np.uint64)
    assert master == numpyro_unit.sampler_seed
    assert [stream["key"] for stream in numpyro_streams] == expected.tolist()


def test_sample_dispatch_preserves_backend_specific_seeds_and_starts(
    manifest, monkeypatch
) -> None:
    """Dispatch adds no second jitter and keeps the reviewed backend paths."""
    plan = build_plan(manifest, "smoke")
    calls: dict[str, dict] = {}
    sentinel = object()

    def fake_pymc(**kwargs):
        calls["pymc"] = kwargs
        return sentinel

    def fake_numpyro(**kwargs):
        calls["numpyro"] = kwargs
        return sentinel

    monkeypatch.setattr(runner.pm, "sample", fake_pymc)
    monkeypatch.setattr(runner, "sample_numpyro_nuts", fake_numpyro)
    with runner.pm.Model() as model:
        runner.pm.Normal("x")
    initvals = [{"x": np.array(0.0)}, {"x": np.array(0.1)}]
    pymc_unit = next(unit for unit in plan if unit.backend_id == "pymc")
    numpyro_unit = next(unit for unit in plan if unit.backend_id == "numpyro")

    assert runner._sample_model(pymc_unit, model, initvals)[0] is sentinel
    assert runner._sample_model(numpyro_unit, model, initvals)[0] is sentinel
    assert calls["pymc"]["random_seed"] == list(pymc_unit.chain_seeds)
    assert calls["pymc"]["init"] == "adapt_diag"
    assert calls["pymc"]["initvals"] == initvals
    assert calls["numpyro"]["random_seed"] == numpyro_unit.sampler_seed
    assert calls["numpyro"]["jitter"] is False
    assert calls["numpyro"]["chain_method"] == "sequential"
    assert calls["numpyro"]["initvals"] == initvals


def test_sampler_jaxification_incompatibility_is_a_compile_failure(
    manifest, monkeypatch
) -> None:
    """A known unsupported JAX lowering is evidence, not an incomplete block."""
    unit = next(
        item for item in build_plan(manifest, "smoke") if item.backend_id == "numpyro"
    )

    def fail_jaxification(**_kwargs):
        raise NotImplementedError("unsupported PyTensor Op")

    monkeypatch.setattr(runner, "sample_numpyro_nuts", fail_jaxification)
    with runner.pm.Model() as model:
        runner.pm.Normal("x")
    initvals = [{"x": np.array(0.0)}, {"x": np.array(0.1)}]

    with pytest.raises(runner.ScientificCellFailure) as captured:
        runner._sample_model(unit, model, initvals)

    assert captured.value.stage == "compile"
    assert captured.value.error_type == "NotImplementedError"


def test_chain_artifact_retains_stats_and_metrics_are_recomputed(
    manifest, tmp_path: Path
) -> None:
    """Hash-bound per-draw statistics, rather than scalar claims, drive metrics."""
    unit = build_plan(manifest, "smoke")[0]
    shape = (unit.chains, unit.draws)
    sequence = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    divergences = np.zeros(shape, dtype=bool)
    divergences[0, 3] = True
    divergences[1, 7] = True
    inference = SimpleNamespace(
        posterior=_natural_dataset(unit),
        sample_stats=xr.Dataset(
            {
                "acceptance_rate": (("chain", "draw"), np.full(shape, 0.91)),
                "diverging": (("chain", "draw"), divergences),
                "energy": (("chain", "draw"), np.sin(sequence / 11.0)),
                "energy_error": (("chain", "draw"), np.cos(sequence / 13.0)),
                "n_steps": (("chain", "draw"), np.full(shape, 7)),
                "step_size": (("chain", "draw"), np.full(shape, 0.125)),
                "tree_depth": (("chain", "draw"), np.full(shape, 3)),
            }
        ),
    )

    chains = runner._standardize_natural_chains(unit, inference)
    metrics = runner._sampler_metrics(unit, chains, 1.25)

    assert "sample_stat__energy_error" in chains
    assert chains["sample_stat__diverging"].shape == shape
    assert chains.attrs["pair_id"] == unit.pair_id
    assert chains.attrs["pair_position"] == unit.pair_position
    assert chains.attrs["cell_id"] == unit.cell_id
    assert chains.attrs["sampler_seed_input_json"]
    assert chains.attrs["chain_rng_provenance_json"]
    assert metrics["divergence_count"] == 2
    assert metrics["leapfrog_step_count"] == 7 * np.prod(shape)
    assert metrics["step_size_final_min"] == pytest.approx(0.125)

    tampered = chains.copy(deep=True)
    tampered["sample_stat__diverging"][0, 0] = True
    assert runner._sampler_metrics(unit, tampered, 1.25)["divergence_count"] == 3

    store = ArtifactStore(tmp_path.resolve())
    reference = store.write_bytes("chains/test.nc", runner._netcdf_bytes(chains))
    chain_path = store.verify(reference)
    changed_bytes = bytearray(chain_path.read_bytes())
    changed_bytes[-1] ^= 1
    chain_path.write_bytes(changed_bytes)
    with pytest.raises(CausalArtifactError, match="wrong SHA-256"):
        store.verify(reference)


def test_initial_evidence_uses_positional_binding_with_unnamed_inputs(
    manifest, monkeypatch
) -> None:
    """Hosted linkers may drop names without breaking start diagnostics."""
    unit = build_plan(manifest, "smoke")[0]
    data = generate_synthetic_data(
        runner._toy_spec(unit),
        group_seed=unit.group_seed,
        observation_seed=unit.observation_seed,
    )
    model = build_causal_model("native_centered", runner._prior(unit), data)
    original = model.compile_fn
    observed_point_fn: list[bool] = []

    def compile_unnamed(outputs, *, inputs=None, point_fn=True, **kwargs):
        assert inputs is not None
        names = [variable.name for variable in inputs]
        try:
            for variable in inputs:
                variable.name = None
            function = original(outputs, inputs=inputs, point_fn=point_fn, **kwargs)
        finally:
            for variable, name in zip(inputs, names, strict=True):
                variable.name = name
        observed_point_fn.append(point_fn)
        return function

    monkeypatch.setattr(model, "compile_fn", compile_unnamed)
    point = model.initial_point()
    evidence = runner._finite_initial_evidence(model, [point])

    assert observed_point_fn == [False]
    assert evidence["all_finite"] is True


def _mock_execute_cell_prefix(monkeypatch, *, roundtrip_error: float = 0.0) -> None:
    """Replace the build/start prefix while retaining execute_cell state logic."""
    sentinel = object()
    monkeypatch.setattr(runner, "_data_from_payload", lambda *_args: sentinel)
    monkeypatch.setattr(runner, "_prior", lambda *_args: sentinel)
    monkeypatch.setattr(runner, "_validate_natural_start_payload", lambda *_args: None)
    monkeypatch.setattr(runner, "build_causal_model", lambda *_args: sentinel)
    monkeypatch.setattr(runner, "_oracle_spec", lambda *_args: sentinel)
    monkeypatch.setattr(
        runner,
        "_coordinate_start_payload",
        lambda *_args: (
            {"chains": [{"roundtrip": {"absolute_error_max": roundtrip_error}}]},
            [{"x": np.array(0.0)}],
        ),
    )
    monkeypatch.setattr(runner, "_make_graph_evaluator", lambda *_args: sentinel)
    monkeypatch.setattr(
        runner,
        "_oracle_diagnostics",
        lambda *_args, **_kwargs: {
            "records": [
                {
                    "errors": {
                        "value": {"scaled_max": 0.0},
                        "gradient": {"scaled_max": 0.0},
                        "hessian": {"scaled_max": 0.0},
                    },
                    "roundtrip": {"absolute_error_max": 0.0},
                }
            ],
            "icdf_tail_finite": True,
            "icdf_branch_continuous": True,
        },
    )


def test_precompile_roundtrip_failure_does_not_claim_compile_success(
    manifest, monkeypatch
) -> None:
    """A coordinate mapping failure reports no compiler evidence."""
    unit = build_plan(manifest, "smoke")[0]
    _mock_execute_cell_prefix(monkeypatch, roundtrip_error=1.0)

    execution = runner.execute_cell(unit, {}, {}, manifest["analysis_policy"])

    assert execution.failure["stage"] == "initialize"
    assert execution.metrics == {"initialization_success": False}
    assert execution.chain_dataset is None


def test_compile_failure_is_distinct_from_nonfinite_initialization(
    manifest, monkeypatch
) -> None:
    """A classified compiler failure records compile_success=False only."""
    unit = build_plan(manifest, "smoke")[0]
    _mock_execute_cell_prefix(monkeypatch)

    def fail_compile(*_args):
        raise runner.ScientificCellFailure(
            "compile", "compiler rejected graph", error_type="CompileError"
        )

    monkeypatch.setattr(runner, "_finite_initial_evidence", fail_compile)
    execution = runner.execute_cell(unit, {}, {}, manifest["analysis_policy"])

    assert execution.failure["stage"] == "compile"
    assert execution.metrics == {"compile_success": False}
    assert execution.chain_dataset is None


@pytest.mark.parametrize("failure_stage", ["sample", "compile"])
def test_pre_sampling_oracle_evidence_obeys_failure_contract(
    manifest, tmp_path: Path, monkeypatch, failure_stage: str
) -> None:
    """Static oracle evidence survives sampling failure but not a compile claim."""
    units = pair_units(manifest, "smoke", build_plan(manifest, "smoke")[0].pair_id)
    unit = next(
        item
        for item in units
        if item.backend_id == "pymc" and item.representation_id == "native-centered"
    )
    data = generate_synthetic_data(
        runner._toy_spec(unit),
        group_seed=unit.group_seed,
        observation_seed=unit.observation_seed,
    )
    prior = runner._prior(unit)
    data_payload = runner._data_payload(unit, data)
    start_payload = runner._natural_start_payload(unit, data, prior)
    monkeypatch.setenv("HSSM_CAUSAL_CACHE_ID", "5" * 64)

    def fail_after_oracle(*_args, **_kwargs):
        raise runner.ScientificCellFailure(
            failure_stage,
            "controlled backend failure",
            error_type="ControlledBackendError",
        )

    monkeypatch.setattr(runner, "_sample_model", fail_after_oracle)
    execution = runner.execute_cell(
        unit, data_payload, start_payload, manifest["analysis_policy"]
    )

    assert execution.status == "failed"
    assert execution.failure["stage"] == failure_stage
    assert execution.chain_dataset is None
    assert execution.diagnostics["oracle"]["posterior_trajectory_evaluated"] is False
    assert execution.metrics["oracle_evaluation_count"] == unit.chains + 1
    if failure_stage == "sample":
        assert execution.metrics["sampling_success"] is False
    else:
        assert execution.metrics["compile_success"] is False

    context = _context(units, manifest)
    store = ArtifactStore(tmp_path.resolve())
    context_ref = store.write_json(
        f"contexts/{context.pair_id}.json", context.as_dict()
    )
    data_ref = store.write_json(f"data/{unit.data_id}.json", data_payload)
    start_ref = store.write_json(f"starts/natural/{unit.start_id}.json", start_payload)
    result_ref = runner.publish_cell_execution(
        unit,
        context,
        manifest,
        store,
        execution,
        context_reference=context_ref,
        data_reference=data_ref,
        natural_start_reference=start_ref,
    )
    record = store.read_json(result_ref)
    validate_result_record(record, unit, context)
    rows = aggregate_results(
        units,
        {unit.cell_id: record},
        context_directory=store.root / "contexts",
        artifact_root=store.root,
        manifest=manifest,
    )
    assert rows[unit.pair_position]["collection_status"] == "present"


def test_failed_chain_standardization_is_sample_failure_without_chain(
    manifest, monkeypatch
) -> None:
    """Malformed sampler output cannot masquerade as a late chain-bearing failure."""
    unit = build_plan(manifest, "smoke")[0]
    _mock_execute_cell_prefix(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_finite_initial_evidence",
        lambda *_args: {"chains": [], "all_finite": True},
    )
    monkeypatch.setattr(runner, "_sample_model", lambda *_args: (object(), 0.25))

    def fail_standardization(*_args):
        raise runner.ScientificCellFailure(
            "summarize", "posterior lacks natural variables"
        )

    monkeypatch.setattr(runner, "_standardize_natural_chains", fail_standardization)
    execution = runner.execute_cell(unit, {}, {}, manifest["analysis_policy"])

    assert execution.failure["stage"] == "sample"
    assert execution.metrics["sampling_success"] is False
    assert execution.chain_dataset is None


@pytest.mark.parametrize(
    ("failure_point", "expected_stage", "posterior_trajectory"),
    [
        ("diagnose", "diagnose", False),
        ("sampler_metrics", "summarize", True),
        ("parameter_summary", "summarize", True),
    ],
)
def test_late_failures_retain_auditable_stage_appropriate_oracle_evidence(
    manifest,
    native_oracle_artifacts,
    tmp_path: Path,
    monkeypatch,
    failure_point: str,
    expected_stage: str,
    posterior_trajectory: bool,
) -> None:
    """Late scientific failures keep exactly the oracle phase already completed."""
    units = pair_units(manifest, "smoke", build_plan(manifest, "smoke")[0].pair_id)
    unit = next(
        item
        for item in units
        if item.backend_id == "pymc" and item.representation_id == "native-centered"
    )
    _mock_execute_cell_prefix(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_runtime_evidence",
        lambda: _runtime_diagnostics(unit)["runtime"],
    )
    monkeypatch.setattr(
        runner,
        "_finite_initial_evidence",
        lambda *_args: {"chains": [], "all_finite": True},
    )
    monkeypatch.setattr(runner, "_sample_model", lambda *_args: (object(), 0.25))
    assert unit == native_oracle_artifacts["unit"]
    natural_chains = native_oracle_artifacts["chains"]
    monkeypatch.setattr(
        runner, "_standardize_natural_chains", lambda *_args: natural_chains
    )

    def oracle_diagnostics(*args, **_kwargs):
        has_trajectory = args[5] is not None
        if failure_point == "diagnose" and has_trajectory:
            raise runner.ScientificCellFailure(
                "diagnose", "controlled trajectory diagnostic failure"
            )
        return _synthetic_oracle(unit, trajectory=has_trajectory)

    monkeypatch.setattr(runner, "_oracle_diagnostics", oracle_diagnostics)
    if failure_point == "sampler_metrics":

        def fail_sampler_metrics(*_args):
            raise runner.ScientificCellFailure(
                "summarize", "controlled sampler metric failure"
            )

        monkeypatch.setattr(runner, "_sampler_metrics", fail_sampler_metrics)
    elif failure_point == "parameter_summary":

        def fail_parameter_summary(*_args):
            raise runner.ScientificCellFailure(
                "summarize", "controlled parameter summary failure"
            )

        monkeypatch.setattr(runner, "_parameter_summaries", fail_parameter_summary)

    execution = runner.execute_cell(unit, {}, {}, manifest["analysis_policy"])

    assert execution.status == "failed"
    assert execution.failure["stage"] == expected_stage
    assert execution.chain_dataset is natural_chains
    oracle = execution.diagnostics["oracle"]
    assert oracle["posterior_trajectory_evaluated"] is posterior_trajectory
    expected_count = unit.chains + 1 + (unit.chains if posterior_trajectory else 0)
    assert len(oracle["records"]) == expected_count
    assert execution.metrics["oracle_evaluation_count"] == expected_count
    assert runner.ORACLE_METRICS <= set(execution.metrics)
    assert execution.metrics["sampling_success"] is True

    retained_oracle = native_oracle_artifacts[
        "full_oracle" if posterior_trajectory else "pre_oracle"
    ]
    execution = replace(
        execution,
        coordinate_starts=native_oracle_artifacts["coordinate_payload"],
        diagnostics=_runtime_diagnostics(unit, oracle=retained_oracle),
        metrics={
            **execution.metrics,
            **runner._registered_oracle_metrics(retained_oracle),
        },
    )

    context = _context(units, manifest)
    store = ArtifactStore(tmp_path.resolve())
    context_ref = store.write_json(
        f"contexts/{context.pair_id}.json", context.as_dict()
    )
    data_ref = store.write_json(
        f"data/{unit.data_id}.json", native_oracle_artifacts["data_payload"]
    )
    start_ref = store.write_json(
        f"starts/natural/{unit.start_id}.json",
        native_oracle_artifacts["start_payload"],
    )
    result_ref = runner.publish_cell_execution(
        unit,
        context,
        manifest,
        store,
        execution,
        context_reference=context_ref,
        data_reference=data_ref,
        natural_start_reference=start_ref,
    )
    record = store.read_json(result_ref)
    rows = aggregate_results(
        units,
        {unit.cell_id: record},
        context_directory=store.root / "contexts",
        artifact_root=store.root,
        manifest=manifest,
    )
    assert rows[unit.pair_position]["collection_status"] == "present"

    if failure_point == "parameter_summary":
        tampered = copy.deepcopy(record)
        tampered["metrics"]["oracle_logp_scaled_error_max"] = 0.5
        with pytest.raises(CausalContractError, match="hash-bound raw evidence"):
            aggregate_results(
                units,
                {unit.cell_id: tampered},
                context_directory=store.root / "contexts",
                artifact_root=store.root,
                manifest=manifest,
            )


def test_oracle_points_are_bound_to_exact_source_artifacts(
    manifest, native_oracle_artifacts, tmp_path: Path
) -> None:
    """Every retained coordinate is tied to its planned natural-scale source."""
    unit = native_oracle_artifacts["unit"]
    units = pair_units(manifest, "smoke", unit.pair_id)
    context = _context(units, manifest)
    store = ArtifactStore(tmp_path.resolve())
    context_ref = store.write_json(
        f"contexts/{context.pair_id}.json", context.as_dict()
    )
    data_ref = store.write_json(
        f"data/{unit.data_id}.json", native_oracle_artifacts["data_payload"]
    )
    start_ref = store.write_json(
        f"starts/natural/{unit.start_id}.json",
        native_oracle_artifacts["start_payload"],
    )
    chains = native_oracle_artifacts["chains"]
    oracle = native_oracle_artifacts["full_oracle"]
    raw_metrics = runner._sampler_metrics(unit, chains, 1.0)
    execution = runner.CellExecution(
        status="completed",
        coordinate_starts=native_oracle_artifacts["coordinate_payload"],
        chain_dataset=chains,
        diagnostics=_runtime_diagnostics(unit, oracle=oracle),
        metrics=runner._registered_metrics(unit, raw_metrics, oracle),
        parameter_summaries=runner._parameter_summaries(chains),
    )
    result_ref = runner.publish_cell_execution(
        unit,
        context,
        manifest,
        store,
        execution,
        context_reference=context_ref,
        data_reference=data_ref,
        natural_start_reference=start_ref,
    )
    record = store.read_json(result_ref)
    verify_result_artifacts(record, store.root, unit, manifest)

    def assert_forged_artifact_rejected(
        name: str, payload: bytes, message: str
    ) -> None:
        reference = copy.deepcopy(record["artifacts"][name])
        path = store.root / reference["path"]
        original = path.read_bytes()
        try:
            path.write_bytes(payload)
            record["artifacts"][name] = {
                "path": reference["path"],
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
            with pytest.raises(CausalContractError, match=message):
                verify_result_artifacts(record, store.root, unit, manifest)
        finally:
            path.write_bytes(original)
            record["artifacts"][name] = reference

    diagnostics = copy.deepcopy(execution.diagnostics)
    trajectory = next(
        item
        for item in diagnostics["oracle"]["records"]
        if item["kind"] == "hash-selected-posterior-trajectory" and item["chain"] == 0
    )
    nonminimal_draw = next(
        draw for draw in range(unit.draws) if draw != trajectory["draw"]
    )
    alternative_natural = {
        "group_location": float(chains["group_location"][0, nonminimal_draw]),
        "group_scale": float(chains["group_scale"][0, nonminimal_draw]),
        "group_effect": np.asarray(
            chains["group_effect"][0, nonminimal_draw], dtype=np.float64
        ).tolist(),
    }
    candidate_vector, _ = runner.natural_to_coordinate(
        alternative_natural,
        prior=native_oracle_artifacts["prior"],
        oracle_spec=native_oracle_artifacts["oracle_spec"],
        representation_id=unit.representation_id,
    )
    vector = runner._pack_model_point(
        native_oracle_artifacts["model"],
        runner._point_from_vector(native_oracle_artifacts["model"], candidate_vector),
    )
    expected = runner.hierarchical_posterior_components(
        vector,
        native_oracle_artifacts["oracle_spec"],
        runner.REPRESENTATION_TO_PARAMETERIZATION[unit.representation_id],
    ).total
    trajectory.update(
        draw=nonminimal_draw,
        selection_sha256=hashlib.sha256(
            f"{unit.cell_id}:trajectory:0:{nonminimal_draw}".encode()
        ).hexdigest(),
        coordinate_vector=vector.tolist(),
        roundtrip=runner._natural_coordinate_roundtrip(
            alternative_natural,
            vector,
            oracle_spec=native_oracle_artifacts["oracle_spec"],
            parameterization=runner.REPRESENTATION_TO_PARAMETERIZATION[
                unit.representation_id
            ],
        ),
        observed={
            "value": expected.value,
            "gradient": expected.gradient.tolist(),
            "hessian": expected.hessian.tolist(),
        },
        oracle={
            "value": expected.value,
            "gradient": expected.gradient.tolist(),
            "hessian": expected.hessian.tolist(),
        },
        group_location=alternative_natural["group_location"],
        group_scale=alternative_natural["group_scale"],
        group_effect=alternative_natural["group_effect"],
    )
    assert_forged_artifact_rejected(
        "diagnostics",
        canonical_json_bytes(diagnostics),
        "lowest-SHA draw",
    )

    wrong_coordinate = copy.deepcopy(execution.diagnostics)
    start_record = next(
        item
        for item in wrong_coordinate["oracle"]["records"]
        if item["kind"] == "shared-natural-start"
    )
    start_record["coordinate_vector"][0] += 0.1
    assert_forged_artifact_rejected(
        "diagnostics",
        canonical_json_bytes(wrong_coordinate),
        "retained roundtrip differs",
    )

    unhealthy_roundtrip = copy.deepcopy(execution.diagnostics)
    fixed_record = next(
        item
        for item in unhealthy_roundtrip["oracle"]["records"]
        if item["kind"] == "fixed-grid"
    )
    shifted_vector = np.asarray(fixed_record["coordinate_vector"], dtype=np.float64)
    shifted_vector[0] += 0.1
    source_natural = {
        "group_location": fixed_record["group_location"],
        "group_scale": fixed_record["group_scale"],
        "group_effect": fixed_record["group_effect"],
    }
    shifted_roundtrip = runner._natural_coordinate_roundtrip(
        source_natural,
        shifted_vector,
        oracle_spec=native_oracle_artifacts["oracle_spec"],
        parameterization=runner.REPRESENTATION_TO_PARAMETERIZATION[
            unit.representation_id
        ],
    )
    shifted_expected = runner.hierarchical_posterior_components(
        shifted_vector,
        native_oracle_artifacts["oracle_spec"],
        runner.REPRESENTATION_TO_PARAMETERIZATION[unit.representation_id],
    ).total
    assert shifted_roundtrip["absolute_error_max"] > float(
        manifest["analysis_policy"]["oracle_gate"]["roundtrip_absolute_error_max"][
            unit.floatx
        ]
    )
    fixed_record.update(
        coordinate_vector=shifted_vector.tolist(),
        roundtrip=shifted_roundtrip,
        observed={
            "value": shifted_expected.value,
            "gradient": shifted_expected.gradient.tolist(),
            "hessian": shifted_expected.hessian.tolist(),
        },
        oracle={
            "value": shifted_expected.value,
            "gradient": shifted_expected.gradient.tolist(),
            "hessian": shifted_expected.hessian.tolist(),
        },
        passed=False,
    )
    unhealthy_roundtrip["oracle"]["passed"] = False
    unhealthy_bytes = canonical_json_bytes(unhealthy_roundtrip)
    diagnostics_reference = copy.deepcopy(record["artifacts"]["diagnostics"])
    diagnostics_path = store.root / diagnostics_reference["path"]
    original_diagnostics = diagnostics_path.read_bytes()
    original_roundtrip_metric = record["metrics"]["roundtrip_absolute_error_max"]
    try:
        diagnostics_path.write_bytes(unhealthy_bytes)
        record["artifacts"]["diagnostics"] = {
            "path": diagnostics_reference["path"],
            "sha256": sha256_bytes(unhealthy_bytes),
            "size_bytes": len(unhealthy_bytes),
        }
        record["metrics"]["roundtrip_absolute_error_max"] = shifted_roundtrip[
            "absolute_error_max"
        ]
        verify_result_artifacts(record, store.root, unit, manifest)
        rows = aggregate_results(
            units,
            {unit.cell_id: record},
            context_directory=store.root / "contexts",
            artifact_root=store.root,
            manifest=manifest,
        )
        assert rows[unit.pair_position]["collection_status"] == "present"
    finally:
        diagnostics_path.write_bytes(original_diagnostics)
        record["artifacts"]["diagnostics"] = diagnostics_reference
        record["metrics"]["roundtrip_absolute_error_max"] = original_roundtrip_metric

    forged_roundtrip = copy.deepcopy(execution.diagnostics)
    forged_roundtrip["oracle"]["records"][0]["roundtrip"]["absolute_error_max"] = 5e-11
    original_roundtrip_metric = record["metrics"]["roundtrip_absolute_error_max"]
    record["metrics"]["roundtrip_absolute_error_max"] = 5e-11
    try:
        assert_forged_artifact_rejected(
            "diagnostics",
            canonical_json_bytes(forged_roundtrip),
            "roundtrip error disagrees",
        )
    finally:
        record["metrics"]["roundtrip_absolute_error_max"] = original_roundtrip_metric

    forged_roundtrip_component = copy.deepcopy(execution.diagnostics)
    forged_roundtrip_component["oracle"]["records"][0]["roundtrip"][
        "group_location"
    ] += 0.01
    assert_forged_artifact_rejected(
        "diagnostics",
        canonical_json_bytes(forged_roundtrip_component),
        "retained roundtrip",
    )

    wrong_coordinate_start = copy.deepcopy(
        native_oracle_artifacts["coordinate_payload"]
    )
    wrong_coordinate_start["chains"][0]["coordinate_vector"][0] += 0.1
    assert_forged_artifact_rejected(
        "coordinate_start",
        canonical_json_bytes(wrong_coordinate_start),
        "coordinate start chain 0 retained roundtrip",
    )

    wrong_natural_start = copy.deepcopy(native_oracle_artifacts["start_payload"])
    wrong_natural_start["chains"][0]["group_location"] += 0.01
    assert_forged_artifact_rejected(
        "natural_start",
        canonical_json_bytes(wrong_natural_start),
        "coordinate start chain 0 natural point",
    )

    wrong_data = copy.deepcopy(native_oracle_artifacts["data_payload"])
    wrong_data["group_effect"][0] += 0.01
    assert_forged_artifact_rejected(
        "data", canonical_json_bytes(wrong_data), "fixed truth"
    )

    wrong_chain = chains.copy(deep=True)
    selected = next(
        item
        for item in execution.diagnostics["oracle"]["records"]
        if item["kind"] == "hash-selected-posterior-trajectory" and item["chain"] == 0
    )
    wrong_chain["group_location"][0, selected["draw"]] += 0.01
    assert_forged_artifact_rejected(
        "chain",
        runner._netcdf_bytes(wrong_chain),
        "selected natural chain draw",
    )


def test_completed_and_late_failure_records_validate_with_exact_artifacts(
    manifest, tmp_path: Path
) -> None:
    """Completed and post-sampling failures satisfy the strict result schema."""
    units = pair_units(manifest, "smoke", build_plan(manifest, "smoke")[0].pair_id)
    context = _context(units, manifest)
    store = ArtifactStore(tmp_path.resolve())
    context_ref = store.write_json(
        f"contexts/{context.pair_id}.json", context.as_dict()
    )
    data_ref = store.write_json(f"data/{units[0].data_id}.json", {"data": True})
    start_ref = store.write_json(
        f"starts/natural/{units[0].start_id}.json", {"starts": True}
    )
    completed = runner.CellExecution(
        status="completed",
        coordinate_starts={"cell": units[0].cell_id},
        chain_dataset=_natural_dataset(units[0]),
        diagnostics=_runtime_diagnostics(
            units[0], oracle=_synthetic_oracle(units[0], trajectory=True)
        ),
        metrics=_completed_metrics(units[0]),
        parameter_summaries=[
            {
                "parameter_id": "group_location",
                "index": None,
                "mean": 0.3,
                "sd": 0.1,
                "mcse_mean": 0.01,
            }
        ],
    )
    result_ref = runner.publish_cell_execution(
        units[0],
        context,
        manifest,
        store,
        completed,
        context_reference=context_ref,
        data_reference=data_ref,
        natural_start_reference=start_ref,
    )
    validate_result_record(store.read_json(result_ref), units[0], context)

    for unit, stage in zip(units[1:3], ("summarize", "diagnose"), strict=True):
        failure = runner.CellExecution(
            status="failed",
            coordinate_starts={"cell": unit.cell_id},
            chain_dataset=_natural_dataset(unit),
            diagnostics=_runtime_diagnostics(
                unit,
                oracle=_synthetic_oracle(unit, trajectory=stage == "summarize"),
            ),
            metrics={
                "compile_success": True,
                "initialization_success": True,
                "logp_finite": True,
                "gradient_finite": True,
                "sampling_success": True,
                **runner._registered_oracle_metrics(
                    _synthetic_oracle(unit, trajectory=stage == "summarize")
                ),
            },
            parameter_summaries=[],
            failure={"stage": stage, "error_type": "NumericalError", "message": "x"},
        )
        reference = runner.publish_cell_execution(
            unit,
            context,
            manifest,
            store,
            failure,
            context_reference=context_ref,
            data_reference=data_ref,
            natural_start_reference=start_ref,
        )
        validate_result_record(store.read_json(reference), unit, context)


def test_run_pair_attempts_all_ten_after_scientific_failure(
    manifest, tmp_path: Path, monkeypatch
) -> None:
    """One scientific failure cannot suppress either backend's paired controls."""
    units = pair_units(manifest, "smoke", build_plan(manifest, "smoke")[0].pair_id)
    context = _context(units, manifest)
    store = ArtifactStore(tmp_path.resolve())
    context_ref = store.write_json(
        f"contexts/{context.pair_id}.json", context.as_dict()
    )
    data_ref = store.write_json(f"data/{units[0].data_id}.json", {"data": True})
    start_ref = store.write_json(
        f"starts/natural/{units[0].start_id}.json", {"starts": True}
    )
    monkeypatch.setattr(
        runner,
        "materialize_inputs_for_unit",
        lambda _unit, _store, **_kwargs: (data_ref, start_ref),
    )
    attempted: list[str] = []
    published: list[str] = []

    def execute(unit):
        attempted.append(unit.cell_id)
        return runner.CellExecution(
            status="failed",
            metrics={},
            parameter_summaries=[],
            failure={"stage": "build", "error_type": "SamplingError", "message": "x"},
        )

    def publish(unit, *_args, **_kwargs):
        assert len(attempted) == 10
        published.append(unit.cell_id)
        return ArtifactRef(f"cells/{unit.cell_id}.json", "a" * 64, 1)

    monkeypatch.setattr(runner, "publish_cell_execution", publish)
    runner.run_pair(
        manifest,
        DEFAULT_MANIFEST,
        units,
        store,
        context,
        context_reference=context_ref,
        executor=execute,
    )

    assert attempted == [unit.cell_id for unit in units]
    assert published == attempted


def test_subprocess_builder_failure_publishes_one_marker_after_all_ten_attempts(
    manifest, tmp_path: Path
) -> None:
    """A real child build failure remains evidence and cannot suppress controls."""
    units = pair_units(manifest, "smoke", build_plan(manifest, "smoke")[0].pair_id)
    target = next(
        unit
        for unit in units
        if unit.backend_id == "pymc" and unit.representation_id == "manual-centered"
    )
    context = _context(units, manifest)
    store = ArtifactStore((tmp_path / "run").resolve())
    context_ref = store.write_json(
        f"contexts/{context.pair_id}.json", context.as_dict()
    )
    context_path = store.root / context_ref.path
    attempted: list[str] = []
    child_output = (tmp_path / "child-result").resolve()
    child_cache = (tmp_path / "child-cache").resolve()
    for name in ("pytensor", "jax", "matplotlib", "xdg"):
        (child_cache / name).mkdir(parents=True, exist_ok=True)

    def completed_control(unit):
        return runner.CellExecution(
            status="completed",
            coordinate_starts={"cell": unit.cell_id},
            chain_dataset=_natural_dataset(unit),
            diagnostics=_runtime_diagnostics(
                unit, oracle=_synthetic_oracle(unit, trajectory=True)
            ),
            metrics=_completed_metrics(unit),
            parameter_summaries=[
                {
                    "parameter_id": "group_location",
                    "index": None,
                    "mean": 0.3,
                    "sd": 0.1,
                    "mcse_mean": 0.01,
                }
            ],
        )

    def execute(unit):
        attempted.append(unit.cell_id)
        if unit.cell_id != target.cell_id:
            return completed_control(unit)
        program = textwrap.dedent(
            """
            import sys
            from pathlib import Path
            from scripts import truncated_hierarchy_causal_runner as runner

            original_builder = runner.build_causal_model

            def fail_manual_builder(parameterization, *args, **kwargs):
                if parameterization == "manual_centered":
                    raise FloatingPointError("controlled builder failure")
                return original_builder(parameterization, *args, **kwargs)

            runner.build_causal_model = fail_manual_builder
            raise SystemExit(
                runner._private_sample_cell(
                    Path(sys.argv[1]),
                    sys.argv[2],
                    sys.argv[3],
                    Path(sys.argv[4]),
                    Path(sys.argv[5]),
                    Path(sys.argv[6]),
                )
            )
            """
        )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.update(
            {
                "HSSM_CAUSAL_CACHE_ID": "6" * 64,
                "PYTENSOR_FLAGS": (
                    f"base_compiledir={child_cache / 'pytensor'},floatX={unit.floatx}"
                ),
                "JAX_COMPILATION_CACHE_DIR": str(child_cache / "jax"),
                "JAX_ENABLE_X64": "true",
                "JAX_PLATFORMS": "cpu",
                "MPLCONFIGDIR": str(child_cache / "matplotlib"),
                "XDG_CACHE_HOME": str(child_cache / "xdg"),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                program,
                str(DEFAULT_MANIFEST),
                unit.tier,
                unit.cell_id,
                str(store.root),
                str(context_path),
                str(child_output),
            ],
            cwd=DEFAULT_MANIFEST.parents[2],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 10, completed.stderr
        return runner._load_staged_execution(child_output)

    references = runner.run_pair(
        manifest,
        DEFAULT_MANIFEST,
        units,
        store,
        context,
        context_reference=context_ref,
        executor=execute,
    )

    assert attempted == [unit.cell_id for unit in units]
    records = [store.read_json(reference) for reference in references]
    failed = [record for record in records if record["execution_status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["cell_id"] == target.cell_id
    assert failed[0]["failure"] == {
        "stage": "build",
        "error_type": "FloatingPointError",
        "message": "controlled builder failure",
    }


@pytest.mark.parametrize(
    "invocation",
    [
        ("-m", "scripts.truncated_hierarchy_causal_runner"),
        ("scripts/truncated_hierarchy_causal_runner.py",),
    ],
)
def test_runner_starts_in_clean_subprocess_without_pythonpath(
    invocation: tuple[str, ...],
) -> None:
    """Module-mode children and the public file-path CLI resolve imports."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, *invocation, "--help"],
        cwd=DEFAULT_MANIFEST.parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "run-unit" in completed.stdout


@pytest.mark.slow
@pytest.mark.parametrize("backend_id", ["pymc", "numpyro"])
def test_tiny_actual_backend_sampling_uses_exact_mapped_start(
    manifest, backend_id: str
) -> None:
    """Exercise each locked sampler path with one short real chain."""
    original = next(
        unit
        for unit in build_plan(manifest, "smoke")
        if unit.backend_id == backend_id and unit.representation_id == "native-centered"
    )
    unit = replace(
        original,
        chains=1,
        tune=5,
        draws=5,
        natural_start_chain_seeds=(12821,),
        chain_seeds=(12822,) if backend_id == "pymc" else (),
        sampler_seed=12823 if backend_id == "numpyro" else None,
    )
    data = generate_synthetic_data(
        runner._toy_spec(unit),
        group_seed=unit.group_seed,
        observation_seed=unit.observation_seed,
    )
    prior = runner._prior(unit)
    model = build_causal_model("native_centered", prior, data)
    starts = runner._natural_start_payload(unit, data, prior)
    _, initvals = runner._coordinate_start_payload(
        unit, model, starts, prior, runner._oracle_spec(unit, prior, data)
    )

    inference, elapsed = runner._sample_model(unit, model, initvals)
    chains = runner._standardize_natural_chains(unit, inference)

    assert elapsed >= 0
    assert chains["group_effect"].shape == (1, 5, 4)
    assert np.all(
        np.isfinite(
            chains[["group_location", "group_scale", "group_effect"]].to_array()
        )
    )
    for name in (
        "acceptance_rate",
        "diverging",
        "energy",
        "n_steps",
        "step_size",
        "tree_depth",
    ):
        assert np.all(np.isfinite(chains[f"sample_stat__{name}"]))


@pytest.mark.slow
def test_tiny_real_execution_applies_manifest_oracle_tolerances(manifest) -> None:
    """Run sampling plus trajectory value/gradient/Hessian oracle diagnostics."""
    original = next(
        unit
        for unit in build_plan(manifest, "smoke")
        if unit.backend_id == "pymc" and unit.representation_id == "native-centered"
    )
    unit = replace(original, tune=10, draws=10)
    data = generate_synthetic_data(
        runner._toy_spec(unit),
        group_seed=unit.group_seed,
        observation_seed=unit.observation_seed,
    )
    prior = runner._prior(unit)
    data_payload = runner._data_payload(unit, data)
    start_payload = runner._natural_start_payload(unit, data, prior)

    execution = runner.execute_cell(
        unit, data_payload, start_payload, manifest["analysis_policy"]
    )

    assert execution.status == "completed"
    assert execution.metrics["oracle_evaluation_count"] >= 5
    assert execution.metrics["oracle_logp_scaled_error_max"] <= 1
    assert execution.metrics["oracle_gradient_scaled_error_max"] <= 1
    assert execution.metrics["oracle_hessian_scaled_error_max"] <= 1

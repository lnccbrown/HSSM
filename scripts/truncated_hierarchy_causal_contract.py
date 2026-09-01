"""Frozen no-sampling contract for the TruncatedNormal causal experiment.

The v2 study established a real sampling failure but deliberately could not assign
its cause.  This module owns a separate v3 experiment: two exact failing regimes,
five same-natural-model representations, and both PyMC and NumPyro.  It validates
the immutable design, expands deterministic five-member execution blocks, records
environment and parent-minted execution identity, aggregates cell markers, and
applies the predeclared classifier.  Model construction and sampling live elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

SCHEMA_VERSION = 3
RUNNER_VERSION = 3
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks/specs/truncated_hierarchy_causal_v3.json"
DEFAULT_FROZEN_V2 = REPO_ROOT / "benchmarks/specs/truncated_hierarchy_v2.json"
ALLOWED_TIERS = ("smoke", "confirmation")
REPRESENTATION_IDS = (
    "native-centered",
    "manual-centered",
    "group-icdf-noncentered",
    "location-icdf-noncentered",
    "full-icdf-noncentered",
)
BACKEND_IDS = ("pymc", "numpyro")
REGIME_IDS = ("lower-outside-weak", "two-sided-near")
SCIENTIFIC_FAILURE_STAGES = (
    "data",
    "build",
    "initialize",
    "compile",
    "sample",
    "summarize",
    "diagnose",
)
SAFE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SAFE_PARAMETER = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
SAFE_COMPOSITE_ID = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*(?:--[a-z0-9]+(?:-[a-z0-9]+)*)+$"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
SEED_MAX = 2**31 - 1
_BLAKE_PERSON = b"hssm1282cv3"
_BLAKE_DOMAIN = "hssm-truncated-hierarchy-causal-v3"
TRAJECTORY_POINTS_PER_CHAIN = 1
ICDF_DIAGNOSTIC_POINTS_PER_LAYER = 5
PAIRED_COMPARISON_ALPHA = 0.05 / 6
FROZEN_MANIFEST_SHA256 = (
    "dde2ce8a7bdce1db72c5cb343887be6817cf56177fd9981e7d19d48963d97744"
)

TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "study_id",
        "status",
        "description",
        "master_seed",
        "seed_derivation",
        "frozen_v2",
        "dependency_profile",
        "natural_model",
        "representations",
        "backends",
        "tiers",
        "regimes",
        "execution_policy",
        "artifact_policy",
        "failure_policy",
        "analysis_policy",
    }
)
REGIME_KEYS = frozenset(
    {
        "regime_id",
        "source_v2_scenario_id",
        "source_v2_data_id",
        "bound_kind",
        "lower",
        "upper",
        "prior_hyper_location",
        "truth_group_location",
        "truth_group_scale",
        "n_groups",
        "n_per_group",
        "group_indices",
        "floatx",
        "replicate_zero_v2_seeds",
    }
)
REPRESENTATION_KEYS = frozenset(
    {
        "representation_id",
        "builder",
        "location_coordinates",
        "group_coordinates",
        "density_implementation",
        "factorial_cell",
        "role",
    }
)
BACKEND_KEYS = frozenset({"backend_id", "sampler", "compiler_path", "device"})
TIER_KEYS = frozenset(
    {
        "role",
        "qualifies_causal_conclusion",
        "replicates",
        "chains",
        "tune",
        "draws",
        "target_accept",
        "max_treedepth",
        "expected_fit_count",
    }
)
PLAN_KEYS = frozenset(
    {
        "schema_version",
        "study_id",
        "manifest_sha256",
        "tier",
        "regime_id",
        "backend_id",
        "representation_id",
        "builder",
        "replicate",
        "pair_id",
        "pair_position",
        "block_id",
        "block_position",
        "canonical_position",
        "cell_id",
        "data_id",
        "start_id",
        "data_seed",
        "truth_seed",
        "group_seed",
        "observation_seed",
        "natural_start_seed",
        "natural_start_chain_seeds",
        "sampler_seed",
        "chain_seeds",
        "chains",
        "tune",
        "draws",
        "target_accept",
        "max_treedepth",
        "floatx",
        "regime",
        "natural_model",
    }
)
CONTEXT_KEYS = frozenset(
    {
        "schema_version",
        "study_id",
        "manifest_sha256",
        "pair_id",
        "block_ids",
        "cell_ids",
        "execution_order",
        "environment",
        "environment_sha256",
        "git_commit",
        "worker_identity_sha256",
        "pair_execution_id",
        "execution_attempt_ids",
    }
)
ARTIFACT_KEYS = frozenset(
    {
        "context",
        "data",
        "natural_start",
        "coordinate_start",
        "chain",
        "diagnostics",
    }
)
ARTIFACT_REF_KEYS = frozenset({"path", "sha256", "size_bytes"})
RESULT_KEYS = frozenset(
    {
        "schema_version",
        "runner_version",
        "study_id",
        "manifest_sha256",
        "tier",
        "regime_id",
        "backend_id",
        "representation_id",
        "replicate",
        "pair_id",
        "pair_position",
        "block_id",
        "block_position",
        "cell_id",
        "execution_status",
        "metrics",
        "parameter_summaries",
        "artifacts",
        "failure",
        "provenance",
    }
)
FAILURE_KEYS = frozenset({"stage", "error_type", "message"})
PROVENANCE_KEYS = frozenset(
    {
        "environment_sha256",
        "git_commit",
        "worker_identity_sha256",
        "pair_execution_id",
        "execution_attempt_id",
        "sampler",
        "compiler_path",
        "device",
        "floatx",
        "pytensor_floatx",
        "jax_enable_x64",
        "sampler_seed_input",
        "chain_rng_provenance",
    }
)
PARAMETER_SUMMARY_KEYS = frozenset({"parameter_id", "index", "mean", "sd", "mcse_mean"})
BOOL_METRICS = frozenset(
    {
        "compile_success",
        "initialization_success",
        "logp_finite",
        "gradient_finite",
        "sampling_success",
        "icdf_tail_finite",
        "icdf_branch_continuous",
    }
)
INTEGER_METRICS = frozenset(
    {"divergence_count", "posterior_draw_count", "oracle_evaluation_count"}
)
FLOAT_METRICS = frozenset(
    {
        "divergence_rate",
        "hyper_rhat_max",
        "hyper_ess_bulk_min",
        "hyper_ess_tail_min",
        "bfmi_min",
        "treedepth_saturation_rate",
        "hyper_mcse_over_sd_max",
        "group_rhat_max",
        "group_ess_bulk_fraction_ge_400",
        "group_ess_tail_fraction_ge_400",
        "sampling_elapsed_seconds",
        "step_size_final_min",
        "step_size_final_max",
        "leapfrog_step_count",
        "oracle_logp_scaled_error_max",
        "oracle_gradient_scaled_error_max",
        "oracle_hessian_scaled_error_max",
        "roundtrip_absolute_error_max",
    }
)
ALLOWED_METRICS = BOOL_METRICS | INTEGER_METRICS | FLOAT_METRICS
SCREENING_METRICS = frozenset(
    {
        "compile_success",
        "initialization_success",
        "logp_finite",
        "gradient_finite",
        "sampling_success",
        "divergence_count",
        "posterior_draw_count",
        "divergence_rate",
    }
)
ORACLE_METRICS = frozenset(
    {
        "oracle_evaluation_count",
        "oracle_logp_scaled_error_max",
        "oracle_gradient_scaled_error_max",
        "oracle_hessian_scaled_error_max",
        "roundtrip_absolute_error_max",
        "icdf_tail_finite",
        "icdf_branch_continuous",
    }
)
CONFIRMATION_METRICS = SCREENING_METRICS | frozenset(
    {
        "hyper_rhat_max",
        "hyper_ess_bulk_min",
        "hyper_ess_tail_min",
        "bfmi_min",
        "treedepth_saturation_rate",
        "hyper_mcse_over_sd_max",
        "group_rhat_max",
        "group_ess_bulk_fraction_ge_400",
        "group_ess_tail_fraction_ge_400",
    }
)


class CausalContractError(ValueError):
    """Raised when evidence does not satisfy the frozen causal contract."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], path: str
) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise CausalContractError(
            f"{path} keys differ; missing={missing}, extra={extra}"
        )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CausalContractError(f"JSON input contains duplicate key {key!r}")
        result[key] = value
    return result


def strict_json_loads(text: str, *, source: str = "JSON input") -> Any:
    """Parse strict JSON, rejecting duplicate keys and non-finite constants."""

    def reject_constant(value: str) -> None:
        raise CausalContractError(f"{source} contains invalid constant {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise CausalContractError(f"invalid {source}: {error}") from error


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize canonical strict JSON with a terminal newline."""
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise CausalContractError(f"value is not strict JSON: {error}") from error
    return f"{rendered}\n".encode()


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes with SHA-256."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash an existing regular file without following contract paths."""
    if not path.is_file():
        raise CausalContractError(f"required file does not exist: {path}")
    return sha256_bytes(path.read_bytes())


def _load_json(path: Path) -> Any:
    try:
        text = path.read_text()
    except OSError as error:
        raise CausalContractError(f"cannot read {path}: {error}") from error
    return strict_json_loads(text, source=str(path))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CausalContractError(f"refusing to overwrite existing artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _relative_contract_path(value: str, *, suffix: str | None = None) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in value
        or str(path) != value
    ):
        raise CausalContractError(f"unsafe relative artifact path: {value!r}")
    if suffix is not None and path.suffix != suffix:
        raise CausalContractError(f"artifact path {value!r} must end in {suffix}")


def _validate_slug(value: Any, path: str, *, composite: bool = False) -> str:
    pattern = SAFE_COMPOSITE_ID if composite else SAFE_ID
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise CausalContractError(f"{path} must be a canonical slug")
    return value


def _validate_seed(value: Any, path: str) -> int:
    if not _is_int(value) or not 0 < value <= SEED_MAX:
        raise CausalContractError(f"{path} must be a positive 31-bit integer")
    return value


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Return the semantic SHA-256 used by every v3 artifact."""
    return sha256_bytes(canonical_json_bytes(manifest))


def _derive_v2_seed(
    master_seed: int,
    owner: str,
    replicate: int,
    purpose: str,
    *,
    root_seed: int | None = None,
) -> int:
    first = master_seed if root_seed is None else root_seed
    payload = "\0".join(
        [
            "hssm-truncated-hierarchy-seed-v2",
            str(first),
            owner,
            str(replicate),
            purpose,
            "",
        ]
    ).encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"hssm1282v2").digest()
    return int.from_bytes(digest, "big") % SEED_MAX + 1


def derive_seed(master_seed: int, *components: str | int) -> int:
    """Derive a positive 31-bit seed in the new causal-v3 BLAKE2 domain."""
    if not _is_int(master_seed) or master_seed < 0:
        raise CausalContractError("master_seed must be a non-negative integer")
    if not components:
        raise CausalContractError("at least one seed component is required")
    rendered: list[str] = []
    for component in components:
        if isinstance(component, bool) or not isinstance(component, (str, int)):
            raise CausalContractError("seed components must be strings or integers")
        text = str(component)
        if not text or "\0" in text:
            raise CausalContractError("seed components must be non-empty and NUL-free")
        rendered.append(text)
    payload = "\0".join([_BLAKE_DOMAIN, str(master_seed), *rendered]).encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=_BLAKE_PERSON).digest()
    return int.from_bytes(digest, "big") % SEED_MAX + 1


def _validate_frozen_v2(
    manifest: Mapping[str, Any], frozen_v2_path: Path
) -> Mapping[str, Any]:
    frozen = manifest["frozen_v2"]
    _require_exact_keys(
        frozen,
        frozenset({"path", "file_sha256", "study_id", "schema_version", "status"}),
        "frozen_v2",
    )
    expected_path = "benchmarks/specs/truncated_hierarchy_v2.json"
    if frozen["path"] != expected_path:
        raise CausalContractError(f"frozen_v2.path must remain {expected_path!r}")
    actual_hash = sha256_file(frozen_v2_path)
    if frozen["file_sha256"] != actual_hash:
        raise CausalContractError(
            "frozen v2 bytes changed; causal evidence cannot be attached to this design"
        )
    v2 = _load_json(frozen_v2_path)
    if not isinstance(v2, Mapping):
        raise CausalContractError("frozen v2 manifest must contain an object")
    for key in ("study_id", "schema_version", "status"):
        if v2.get(key) != frozen[key]:
            raise CausalContractError(f"frozen_v2.{key} does not match frozen bytes")
    return v2


def _validate_regimes(manifest: Mapping[str, Any], v2: Mapping[str, Any]) -> None:
    regimes = manifest["regimes"]
    if not isinstance(regimes, list) or len(regimes) != 2:
        raise CausalContractError("regimes must contain exactly two entries")
    if tuple(item.get("regime_id") for item in regimes) != REGIME_IDS:
        raise CausalContractError(f"regime order must remain {REGIME_IDS}")
    v2_scenarios = {
        item.get("scenario_id"): item
        for item in v2.get("scenarios", [])
        if isinstance(item, Mapping)
    }
    source_fields = (
        "bound_kind",
        "lower",
        "upper",
        "prior_hyper_location",
        "truth_group_location",
        "truth_group_scale",
        "n_groups",
        "n_per_group",
        "group_indices",
        "floatx",
    )
    for index, regime in enumerate(regimes):
        if not isinstance(regime, Mapping):
            raise CausalContractError(f"regimes[{index}] must be an object")
        _require_exact_keys(regime, REGIME_KEYS, f"regimes[{index}]")
        source_id = _validate_slug(
            regime["source_v2_scenario_id"],
            f"regimes[{index}].source_v2_scenario_id",
        )
        source = v2_scenarios.get(source_id)
        if source is None:
            raise CausalContractError(f"v2 source scenario {source_id!r} is absent")
        if source.get("tier") != "smoke" or source.get("sampler") != "pymc":
            raise CausalContractError(
                f"{source_id} is not the frozen failing PyMC smoke"
            )
        if source.get("data_id") != regime["source_v2_data_id"]:
            raise CausalContractError(f"{source_id} data_id changed")
        for field in source_fields:
            if source.get(field) != regime[field]:
                raise CausalContractError(
                    f"regimes[{index}].{field} does not match {source_id}"
                )
        if regime["floatx"] not in {"float32", "float64"}:
            raise CausalContractError("regime floatx must be float32 or float64")
        if not _is_int(regime["n_groups"]) or regime["n_groups"] <= 0:
            raise CausalContractError("n_groups must be positive")
        if not _is_int(regime["n_per_group"]) or regime["n_per_group"] <= 0:
            raise CausalContractError("n_per_group must be positive")
        seeds = regime["replicate_zero_v2_seeds"]
        expected_seed_keys = frozenset(
            {"data_seed", "truth_seed", "group_seed", "observation_seed"}
        )
        if not isinstance(seeds, Mapping):
            raise CausalContractError("replicate_zero_v2_seeds must be an object")
        _require_exact_keys(seeds, expected_seed_keys, "replicate_zero_v2_seeds")
        data_seed = _derive_v2_seed(
            manifest["master_seed"], regime["source_v2_data_id"], 0, "data"
        )
        expected_seeds = {
            "data_seed": data_seed,
            "truth_seed": _derive_v2_seed(
                manifest["master_seed"],
                regime["source_v2_data_id"],
                0,
                "truth",
                root_seed=data_seed,
            ),
            "group_seed": _derive_v2_seed(
                manifest["master_seed"],
                regime["source_v2_data_id"],
                0,
                "group",
                root_seed=data_seed,
            ),
            "observation_seed": _derive_v2_seed(
                manifest["master_seed"],
                regime["source_v2_data_id"],
                0,
                "observation",
                root_seed=data_seed,
            ),
        }
        if dict(seeds) != expected_seeds:
            raise CausalContractError(
                f"{regime['regime_id']} replicate-zero seeds do not replay v2"
            )


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    frozen_v2_path: Path | None = None,
) -> Mapping[str, Any]:
    """Validate the complete, prospectively frozen v3 design."""
    if not isinstance(manifest, Mapping):
        raise CausalContractError("manifest must contain a JSON object")
    _require_exact_keys(manifest, TOP_LEVEL_KEYS, "manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise CausalContractError(f"schema_version must be {SCHEMA_VERSION}")
    if manifest["study_id"] != "truncated_hierarchy_causal_v3":
        raise CausalContractError("study_id must be truncated_hierarchy_causal_v3")
    if manifest["status"] != "frozen-before-sampling":
        raise CausalContractError("manifest must be frozen before sampling")
    if manifest["master_seed"] != 1282:
        raise CausalContractError("master_seed must remain 1282")

    seed_contract = manifest["seed_derivation"]
    expected_seed_contract = {
        "algorithm": "blake2b-64-causal-v3",
        "person": "hssm1282cv3",
        "domain": _BLAKE_DOMAIN,
        "positive_integer_range": [1, SEED_MAX],
        "replicate_zero_data": "exact-frozen-v2-streams",
        "later_data": "causal-v3-domain-separated-streams",
    }
    if seed_contract != expected_seed_contract:
        raise CausalContractError("seed_derivation changes the reviewed v3 domain")

    resolved_v2 = DEFAULT_FROZEN_V2 if frozen_v2_path is None else frozen_v2_path
    v2 = _validate_frozen_v2(manifest, resolved_v2)
    _validate_regimes(manifest, v2)

    expected_natural_model = {
        "likelihood": "y[i] ~ Normal(group_effect[group_index[i]], 0.5)",
        "location_prior": "TruncatedNormal(prior_hyper_location, 0.25, bounds)",
        "scale_prior": "Weibull(alpha=1.5, beta=0.3)",
        "group_prior": "TruncatedNormal(group_location, group_scale, bounds)",
        "location_prior_sigma": 0.25,
        "scale_prior_alpha": 1.5,
        "scale_prior_beta": 0.3,
        "observation_sigma": 0.5,
    }
    if manifest["natural_model"] != expected_natural_model:
        raise CausalContractError("natural_model changed")

    representations = manifest["representations"]
    if not isinstance(representations, list) or len(representations) != 5:
        raise CausalContractError("representations must contain exactly five entries")
    if (
        tuple(item.get("representation_id") for item in representations)
        != REPRESENTATION_IDS
    ):
        raise CausalContractError(
            f"representation order must remain {REPRESENTATION_IDS}"
        )
    expected_builders = (
        "native_centered",
        "manual_centered",
        "group_icdf_noncentered",
        "location_icdf_noncentered",
        "full_icdf_noncentered",
    )
    for index, (representation, builder) in enumerate(
        zip(representations, expected_builders, strict=True)
    ):
        if not isinstance(representation, Mapping):
            raise CausalContractError(f"representations[{index}] must be an object")
        _require_exact_keys(
            representation, REPRESENTATION_KEYS, f"representations[{index}]"
        )
        if representation["builder"] != builder:
            raise CausalContractError(f"representations[{index}].builder changed")
    factorial = [item["factorial_cell"] for item in representations]
    if factorial != ["C-C", "C-C", "C-NC", "NC-C", "NC-NC"]:
        raise CausalContractError("representations no longer form the frozen 2x2")
    expected_representations = [
        {
            "representation_id": "native-centered",
            "builder": "native_centered",
            "location_coordinates": "centered",
            "group_coordinates": "centered",
            "density_implementation": "pymc-native",
            "factorial_cell": "C-C",
            "role": "native-reference",
        },
        {
            "representation_id": "manual-centered",
            "builder": "manual_centered",
            "location_coordinates": "centered",
            "group_coordinates": "centered",
            "density_implementation": "independent-manual",
            "factorial_cell": "C-C",
            "role": "implementation-control",
        },
        {
            "representation_id": "group-icdf-noncentered",
            "builder": "group_icdf_noncentered",
            "location_coordinates": "centered",
            "group_coordinates": "icdf-noncentered",
            "density_implementation": ("pymc-native-location-independent-icdf-groups"),
            "factorial_cell": "C-NC",
            "role": "factorial",
        },
        {
            "representation_id": "location-icdf-noncentered",
            "builder": "location_icdf_noncentered",
            "location_coordinates": "icdf-noncentered",
            "group_coordinates": "centered",
            "density_implementation": ("independent-icdf-location-pymc-native-groups"),
            "factorial_cell": "NC-C",
            "role": "factorial",
        },
        {
            "representation_id": "full-icdf-noncentered",
            "builder": "full_icdf_noncentered",
            "location_coordinates": "icdf-noncentered",
            "group_coordinates": "icdf-noncentered",
            "density_implementation": "independent-icdf",
            "factorial_cell": "NC-NC",
            "role": "factorial",
        },
    ]
    if representations != expected_representations:
        raise CausalContractError("representation semantics changed")

    backends = manifest["backends"]
    if not isinstance(backends, list) or len(backends) != 2:
        raise CausalContractError("backends must contain exactly two entries")
    if tuple(item.get("backend_id") for item in backends) != BACKEND_IDS:
        raise CausalContractError(f"backend order must remain {BACKEND_IDS}")
    for index, backend in enumerate(backends):
        if not isinstance(backend, Mapping):
            raise CausalContractError(f"backends[{index}] must be an object")
        _require_exact_keys(backend, BACKEND_KEYS, f"backends[{index}]")
        if backend["device"] != "cpu":
            raise CausalContractError("causal comparison is CPU-only")
    expected_backends = [
        {
            "backend_id": "pymc",
            "sampler": "pymc-nuts",
            "compiler_path": "pytensor",
            "device": "cpu",
        },
        {
            "backend_id": "numpyro",
            "sampler": "numpyro-nuts-via-pymc",
            "compiler_path": "pytensor-to-jax",
            "device": "cpu",
        },
    ]
    if backends != expected_backends:
        raise CausalContractError("backend sampler/compiler policy changed")

    tiers = manifest["tiers"]
    if not isinstance(tiers, Mapping) or tuple(tiers) != ALLOWED_TIERS:
        raise CausalContractError(f"tiers must remain ordered as {ALLOWED_TIERS}")
    expected_budgets = {
        "smoke": (1, 2, 250, 250, False),
        "confirmation": (8, 4, 1000, 1000, True),
    }
    for tier, (replicates, chains, tune, draws, qualifies) in expected_budgets.items():
        spec = tiers[tier]
        if not isinstance(spec, Mapping):
            raise CausalContractError(f"tiers.{tier} must be an object")
        _require_exact_keys(spec, TIER_KEYS, f"tiers.{tier}")
        observed = (
            spec["replicates"],
            spec["chains"],
            spec["tune"],
            spec["draws"],
            spec["qualifies_causal_conclusion"],
        )
        if observed != (replicates, chains, tune, draws, qualifies):
            raise CausalContractError(f"tiers.{tier} budget changed")
        if spec["target_accept"] != 0.9 or spec["max_treedepth"] != 10:
            raise CausalContractError(f"tiers.{tier} sampler policy changed")
        expected_fits = 2 * 5 * 2 * replicates
        if spec["expected_fit_count"] != expected_fits:
            raise CausalContractError(f"tiers.{tier}.expected_fit_count is wrong")

    profile = manifest["dependency_profile"]
    if not isinstance(profile, Mapping) or profile.get("name") != "current-resolved":
        raise CausalContractError("dependency_profile must be current-resolved")
    for field in ("project_path", "lock_path"):
        _relative_contract_path(profile[field])
    for field, hash_field in (
        ("project_path", "project_sha256"),
        ("lock_path", "lock_sha256"),
    ):
        actual = sha256_file(REPO_ROOT / profile[field])
        if profile[hash_field] != actual:
            raise CausalContractError(f"dependency_profile.{hash_field} is stale")
    expected_profile_static = {
        "name": "current-resolved",
        "python": "3.12",
        "project_path": (
            "benchmarks/environments/truncated_hierarchy/current-resolved/pyproject.toml"
        ),
        "lock_path": (
            "benchmarks/environments/truncated_hierarchy/current-resolved/uv.lock"
        ),
        "required_versions": {
            "arviz": "1.3.0",
            "bambi": "0.20.0",
            "jax": "0.11.1",
            "jaxlib": "0.11.1",
            "numpy": "2.4.6",
            "numpyro": "0.21.0",
            "pymc": "6.3.1",
            "pytensor": "3.3.0",
            "scipy": "1.18.1",
        },
    }
    if any(profile.get(key) != value for key, value in expected_profile_static.items()):
        raise CausalContractError("dependency profile semantics changed")

    execution = manifest["execution_policy"]
    if execution.get("scheduling_unit") != "backend-paired-ten-cell-worker":
        raise CausalContractError(
            "execution must pair both backend blocks on one worker"
        )
    if execution.get("pair_identity") != "tier-regime-replicate":
        raise CausalContractError("backend-pair identity changed")
    if (
        execution.get("pair_members")
        != "both-five-representation-backend-blocks-exactly-once"
    ):
        raise CausalContractError("backend pair must contain both five-form blocks")
    expected_backend_order = (
        "left-rotation-of-manifest-backend-order-by-(replicate+regime_index)-modulo-2"
    )
    if execution.get("backend_order") != expected_backend_order:
        raise CausalContractError("backend counterbalancing order changed")
    expected_order = (
        "within-each-backend-left-rotation-of-manifest-representation-order-by-"
        "(4*replicate+2*regime_index+backend_index)-modulo-5"
    )
    if execution.get("order") != expected_order:
        raise CausalContractError("five-form counterbalancing order changed")
    if execution.get("target_accept") is not None:
        raise CausalContractError("target_accept belongs only to frozen tier budgets")
    if execution.get("threads") != 1:
        raise CausalContractError("execution_policy.threads must remain one")
    expected_start_generation = {
        "source_model": "native-centered",
        "support_points": "pymc-initial-point",
        "function": "pymc.initial_point.make_initial_point_fns_per_chain",
        "jitter_rvs": "all-free-rvs",
        "jitter_distribution": "uniform-minus-one-to-one-in-transformed-coordinates",
        "seed": "natural_start_chain_seeds-derived-from-natural_start_seed",
        "materialization": (
            "map-each-jittered-native-coordinate-point-to-natural-scale-once"
        ),
        "sampling_init": "adapt_diag-with-exact-mapped-start-and-no-second-jitter",
    }
    if execution.get("natural_start_generation") != expected_start_generation:
        raise CausalContractError("natural start generation rule changed")
    expected_sampler_seed_policy = {
        "pymc_plan": "chain_seeds-are-seedsequence-entropy-words",
        "pymc_execution": (
            "pymc-6.3.1-spawns-one-pcg64-generator-per-chain-draws-one-init-step-"
            "integer-below-2^30-then-samples-with-the-advanced-generators"
        ),
        "pymc_provenance": (
            "record-entropy-spawn-key-init-step-integer-and-post-draw-generator-"
            "state-hash"
        ),
        "numpyro_plan": "one-scalar-sampler-seed",
        "numpyro_provenance": "record-exact-uint32-jax-prngkey-or-split-keys",
    }
    if execution.get("sampler_seed_policy") != expected_sampler_seed_policy:
        raise CausalContractError("backend sampler-seed semantics changed")
    failure = manifest["failure_policy"]
    if failure.get("scientific_stages") != list(SCIENTIFIC_FAILURE_STAGES):
        raise CausalContractError("scientific failure stage vocabulary changed")
    artifact = manifest["artifact_policy"]
    expected_artifact_templates = {
        "context_path": "contexts/<pair_id>.json",
        "data_path": "data/<data_id>.json",
        "natural_start_path": "starts/natural/<start_id>.json",
        "coordinate_start_path": "starts/coordinates/<cell_id>.json",
        "chain_path": "chains/<cell_id>.nc",
        "diagnostic_path": "diagnostics/<cell_id>.json",
        "cell_path": "cells/<cell_id>.json",
        "aggregate_path": "aggregate/<tier>/results.jsonl",
        "assessment_path": "aggregate/<tier>/assessment.json",
    }
    for field, value in expected_artifact_templates.items():
        if artifact.get(field) != value:
            raise CausalContractError(f"artifact_policy.{field} changed")
    classifier = manifest["analysis_policy"]
    if classifier.get("scope") != "classify-each-regime-separately-never-pool-regimes":
        raise CausalContractError("regime-specific classification is mandatory")
    expected_paired_inference = {
        "replicates": 8,
        "familywise_alpha": 0.05,
        "bonferroni_comparisons": 6,
        "per_comparison_alpha": PAIRED_COMPARISON_ALPHA,
        "test": "two-sided-exact-sign-test-on-discordant-paired-health-outcomes",
        "minimum_all-directional_p_value": 0.0078125,
        "comparison_family": [
            "native-vs-manual-IUT-across-both-backends",
            "group-effect-at-location-centered-IUT-across-both-backends",
            "group-effect-at-location-noncentered-IUT-across-both-backends",
            "location-effect-at-groups-centered-IUT-across-both-backends",
            "location-effect-at-groups-noncentered-IUT-across-both-backends",
            "backend-five-form-health-count-omnibus",
        ],
        "backend_omnibus_statistic": (
            "per-replicate-pymc-healthy-form-count-minus-numpyro-healthy-form-count"
        ),
        "backend_omnibus_direction": (
            "same-nonzero-sign-in-all-eight-paired-replicates"
        ),
        "representation_backend_contrasts": ("descriptive-only-never-classifying"),
        "interpretation": (
            "minimum-inferentially-defensible-grid-not-a-high-power-small-effect-design"
        ),
    }
    if classifier.get("paired_inference") != expected_paired_inference:
        raise CausalContractError("paired-inference family or threshold changed")
    expected_tolerances = {
        "float64": {
            "logp": {"absolute_tolerance": 5e-8, "relative_tolerance": 2e-8},
            "gradient": {
                "absolute_tolerance": 3e-7,
                "relative_tolerance": 3e-7,
            },
            "hessian": {
                "absolute_tolerance": 3e-6,
                "relative_tolerance": 3e-6,
            },
        },
        "float32": {
            "logp": {
                "absolute_tolerance": 2e-4,
                "relative_tolerance": 2e-4,
            },
            "gradient": {
                "absolute_tolerance": 5e-4,
                "relative_tolerance": 7e-4,
            },
            "hessian": {
                "absolute_tolerance": 3e-3,
                "relative_tolerance": 3e-3,
            },
        },
    }
    oracle_gate = classifier.get("oracle_gate", {})
    expected_evaluation_points = [
        "fixed-grid",
        "every-shared-natural-start",
        "hash-selected-posterior-trajectory-points",
    ]
    if oracle_gate.get("evaluation_points") != expected_evaluation_points:
        raise CausalContractError("oracle evaluation-point family changed")
    expected_point_counts = {
        "fixed_grid_on_replicate_zero": 1,
        "shared_start_points_per_chain": 1,
        "icdf_points_per_noncentered_layer": ICDF_DIAGNOSTIC_POINTS_PER_LAYER,
        "posterior_trajectory_points_per_chain": TRAJECTORY_POINTS_PER_CHAIN,
        "failed_sample_phase": "pre-sampling-only",
        "completed_phase": "pre-sampling-plus-posterior-trajectory",
    }
    if oracle_gate.get("point_counts") != expected_point_counts:
        raise CausalContractError("oracle evaluation counts or phases changed")
    if oracle_gate.get("component_tolerances") != expected_tolerances:
        raise CausalContractError("oracle component tolerances changed")
    expected_scaled_error = (
        "abs(observed-reference)/(absolute_tolerance+relative_tolerance*"
        "max(abs(reference),abs(observed)))"
    )
    if oracle_gate.get("combined_scaled_error") != expected_scaled_error:
        raise CausalContractError("oracle combined scaled-error rule changed")
    family_health = classifier.get("family_health", {})
    if (
        family_health.get("per_fit_pass_fraction_ge") != 0.95
        or family_health.get("maximum_failed_replicates") != 0
    ):
        raise CausalContractError("eight-replicate family gate must require 8/8 fits")
    expected_precedence = [
        "native-pymc-correctness-defect",
        "native-graph-or-adaptation",
        "group-conditional-centering",
        "location-centering",
        "joint-centering-interaction",
        "backend-path-specific",
        "residual-tn-or-scale-geometry",
        "initialization-or-budget-sensitive",
        "all-representations-healthy",
        "mixed-inconclusive",
    ]
    if classifier.get("classifier_precedence") != expected_precedence:
        raise CausalContractError("classifier precedence changed")
    if manifest_digest(manifest) != FROZEN_MANIFEST_SHA256:
        raise CausalContractError("manifest semantic digest differs from frozen v3")

    # Bind validation to the selected manifest path without requiring that a
    # copied fixture live under the repository.  The bytes themselves are the
    # evidence identity; source paths inside the manifest remain repository-relative.
    if manifest_path.name != "truncated_hierarchy_causal_v3.json":
        raise CausalContractError("manifest filename must identify causal v3")
    return manifest


def load_manifest(path: Path = DEFAULT_MANIFEST) -> Mapping[str, Any]:
    """Load and validate the causal v3 manifest."""
    value = _load_json(path)
    if not isinstance(value, Mapping):
        raise CausalContractError("manifest must contain an object")
    return validate_manifest(value, manifest_path=path)


@dataclass(frozen=True, slots=True)
class UnitSpec:
    """One cell in a five-representation causal execution block."""

    schema_version: int
    study_id: str
    manifest_sha256: str
    tier: str
    regime_id: str
    backend_id: str
    representation_id: str
    builder: str
    replicate: int
    pair_id: str
    pair_position: int
    block_id: str
    block_position: int
    canonical_position: int
    cell_id: str
    data_id: str
    start_id: str
    data_seed: int
    truth_seed: int
    group_seed: int
    observation_seed: int
    natural_start_seed: int
    natural_start_chain_seeds: tuple[int, ...]
    sampler_seed: int | None
    chain_seeds: tuple[int, ...]
    chains: int
    tune: int
    draws: int
    target_accept: float
    max_treedepth: int
    floatx: str
    regime: Mapping[str, Any]
    natural_model: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a detached strict-JSON representation."""
        value = {
            field: getattr(self, field)
            for field in PLAN_KEYS
            if field not in {"natural_start_chain_seeds", "chain_seeds"}
        }
        value["natural_start_chain_seeds"] = list(self.natural_start_chain_seeds)
        value["chain_seeds"] = list(self.chain_seeds)
        return strict_json_loads(canonical_json_bytes(value).decode())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        """Validate and construct a unit from its canonical mapping."""
        _require_exact_keys(value, PLAN_KEYS, "plan unit")
        kwargs = dict(value)
        kwargs["natural_start_chain_seeds"] = tuple(value["natural_start_chain_seeds"])
        kwargs["chain_seeds"] = tuple(value["chain_seeds"])
        unit = cls(**kwargs)
        _validate_unit(unit)
        return unit


def _validate_unit(unit: UnitSpec) -> None:
    if unit.schema_version != SCHEMA_VERSION:
        raise CausalContractError("unit schema_version is wrong")
    if unit.study_id != "truncated_hierarchy_causal_v3":
        raise CausalContractError("unit study_id is wrong")
    if not SHA256.fullmatch(unit.manifest_sha256):
        raise CausalContractError("unit manifest_sha256 is invalid")
    if unit.tier not in ALLOWED_TIERS:
        raise CausalContractError("unit tier is invalid")
    if unit.regime_id not in REGIME_IDS:
        raise CausalContractError("unit regime_id is invalid")
    if unit.backend_id not in BACKEND_IDS:
        raise CausalContractError("unit backend_id is invalid")
    if unit.representation_id not in REPRESENTATION_IDS:
        raise CausalContractError("unit representation_id is invalid")
    for name in ("pair_id", "block_id", "cell_id"):
        _validate_slug(getattr(unit, name), f"unit.{name}", composite=True)
    for name in ("data_id", "start_id"):
        _validate_slug(getattr(unit, name), f"unit.{name}", composite=True)
    if not 0 <= unit.pair_position < 10:
        raise CausalContractError("unit pair_position must be in [0, 10)")
    if not 0 <= unit.block_position < 5 or not 0 <= unit.canonical_position < 5:
        raise CausalContractError("unit block/canonical positions must be in [0, 5)")
    if unit.replicate < 0:
        raise CausalContractError("unit replicate must be non-negative")
    for name in (
        "data_seed",
        "truth_seed",
        "group_seed",
        "observation_seed",
        "natural_start_seed",
    ):
        _validate_seed(getattr(unit, name), f"unit.{name}")
    if len(unit.natural_start_chain_seeds) != unit.chains:
        raise CausalContractError("natural start seed count must equal chains")
    if unit.backend_id == "pymc":
        if unit.sampler_seed is not None or len(unit.chain_seeds) != unit.chains:
            raise CausalContractError(
                "PyMC units require one entropy word per chain and no scalar "
                "sampler seed"
            )
    elif unit.backend_id == "numpyro":
        _validate_seed(unit.sampler_seed, "unit.sampler_seed")
        if unit.chain_seeds:
            raise CausalContractError(
                "NumPyro units require one scalar sampler seed and no integer "
                "chain seeds"
            )
    for seed in unit.natural_start_chain_seeds:
        _validate_seed(seed, "unit natural-start chain seed")
    for seed in unit.chain_seeds:
        _validate_seed(seed, "unit PyMC entropy word")


def _data_seeds(
    manifest: Mapping[str, Any], regime: Mapping[str, Any], replicate: int
) -> tuple[int, int, int, int]:
    if replicate == 0:
        seeds = regime["replicate_zero_v2_seeds"]
        return (
            seeds["data_seed"],
            seeds["truth_seed"],
            seeds["group_seed"],
            seeds["observation_seed"],
        )
    master = manifest["master_seed"]
    regime_id = regime["regime_id"]
    data_seed = derive_seed(master, "data", regime_id, replicate)
    return (
        data_seed,
        derive_seed(master, "truth", regime_id, replicate, data_seed),
        derive_seed(master, "group", regime_id, replicate, data_seed),
        derive_seed(master, "observation", regime_id, replicate, data_seed),
    )


def build_plan(manifest: Mapping[str, Any], tier: str) -> tuple[UnitSpec, ...]:
    """Expand backend-paired workers and their five-form blocks in exact order."""
    validate_manifest(manifest)
    if tier not in ALLOWED_TIERS:
        raise CausalContractError(f"tier must be one of {ALLOWED_TIERS}")
    tier_spec = manifest["tiers"][tier]
    digest = manifest_digest(manifest)
    representations = manifest["representations"]
    units: list[UnitSpec] = []
    for regime_index, regime in enumerate(manifest["regimes"]):
        regime_id = regime["regime_id"]
        for replicate in range(tier_spec["replicates"]):
            pair_id = f"{tier}--{regime_id}--replicate-{replicate:02d}"
            data_id = pair_id
            start_id = data_id
            data_seed, truth_seed, group_seed, observation_seed = _data_seeds(
                manifest, regime, replicate
            )
            natural_start_seed = derive_seed(
                manifest["master_seed"], "natural-start", tier, regime_id, replicate
            )
            natural_start_chain_seeds = tuple(
                derive_seed(
                    natural_start_seed,
                    "chain",
                    chain,
                )
                for chain in range(tier_spec["chains"])
            )
            backend_shift = (replicate + regime_index) % len(BACKEND_IDS)
            ordered_backends = (
                manifest["backends"][backend_shift:]
                + manifest["backends"][:backend_shift]
            )
            for pair_backend_position, backend in enumerate(ordered_backends):
                backend_id = backend["backend_id"]
                backend_index = BACKEND_IDS.index(backend_id)
                block_id = (
                    f"{tier}--{regime_id}--{backend_id}--replicate-{replicate:02d}"
                )
                shift = (4 * replicate + 2 * regime_index + backend_index) % len(
                    representations
                )
                ordered = representations[shift:] + representations[:shift]
                for block_position, representation in enumerate(ordered):
                    canonical_position = REPRESENTATION_IDS.index(
                        representation["representation_id"]
                    )
                    cell_id = f"{block_id}--{representation['representation_id']}"
                    sampler_seed = (
                        derive_seed(manifest["master_seed"], "sampler", cell_id)
                        if backend_id == "numpyro"
                        else None
                    )
                    chain_seeds = (
                        tuple(
                            derive_seed(
                                manifest["master_seed"],
                                "sampler-chain",
                                cell_id,
                                chain,
                            )
                            for chain in range(tier_spec["chains"])
                        )
                        if backend_id == "pymc"
                        else ()
                    )
                    unit = UnitSpec(
                        schema_version=SCHEMA_VERSION,
                        study_id=manifest["study_id"],
                        manifest_sha256=digest,
                        tier=tier,
                        regime_id=regime_id,
                        backend_id=backend_id,
                        representation_id=representation["representation_id"],
                        builder=representation["builder"],
                        replicate=replicate,
                        pair_id=pair_id,
                        pair_position=pair_backend_position * 5 + block_position,
                        block_id=block_id,
                        block_position=block_position,
                        canonical_position=canonical_position,
                        cell_id=cell_id,
                        data_id=data_id,
                        start_id=start_id,
                        data_seed=data_seed,
                        truth_seed=truth_seed,
                        group_seed=group_seed,
                        observation_seed=observation_seed,
                        natural_start_seed=natural_start_seed,
                        natural_start_chain_seeds=natural_start_chain_seeds,
                        sampler_seed=sampler_seed,
                        chain_seeds=chain_seeds,
                        chains=tier_spec["chains"],
                        tune=tier_spec["tune"],
                        draws=tier_spec["draws"],
                        target_accept=tier_spec["target_accept"],
                        max_treedepth=tier_spec["max_treedepth"],
                        floatx=regime["floatx"],
                        regime=dict(regime),
                        natural_model=dict(manifest["natural_model"]),
                    )
                    _validate_unit(unit)
                    units.append(unit)
    expected = manifest["tiers"][tier]["expected_fit_count"]
    if len(units) != expected:
        raise CausalContractError(
            f"expanded {len(units)} cells but manifest requires {expected}"
        )
    _validate_execution_groups(units)
    return tuple(units)


def _validate_execution_groups(units: Sequence[UnitSpec]) -> None:
    pairs: dict[str, list[UnitSpec]] = {}
    blocks: dict[str, list[UnitSpec]] = {}
    for unit in units:
        pairs.setdefault(unit.pair_id, []).append(unit)
        blocks.setdefault(unit.block_id, []).append(unit)
    for pair_id, members in pairs.items():
        if len(members) != 10:
            raise CausalContractError(
                f"{pair_id} is not a complete backend-paired ten-cell worker"
            )
        if [member.pair_position for member in members] != list(range(10)):
            raise CausalContractError(f"{pair_id} has invalid pair positions")
        if {(member.backend_id, member.representation_id) for member in members} != set(
            (backend, representation)
            for backend in BACKEND_IDS
            for representation in REPRESENTATION_IDS
        ):
            raise CausalContractError(
                f"{pair_id} does not contain both complete five-form blocks"
            )
        shared = {
            (
                member.data_id,
                member.start_id,
                member.data_seed,
                member.truth_seed,
                member.group_seed,
                member.observation_seed,
                member.natural_start_seed,
                member.natural_start_chain_seeds,
            )
            for member in members
        }
        if len(shared) != 1:
            raise CausalContractError(
                f"{pair_id} does not share byte-identical data and natural starts"
            )
    for block_id, members in blocks.items():
        if len(members) != 5:
            raise CausalContractError(
                f"{block_id} is not an indivisible five-form block"
            )
        if [member.block_position for member in members] != list(range(5)):
            raise CausalContractError(f"{block_id} has invalid block positions")
        if {member.representation_id for member in members} != set(REPRESENTATION_IDS):
            raise CausalContractError(f"{block_id} does not contain all five forms")


def plan_unit_by_id(manifest: Mapping[str, Any], tier: str, cell_id: str) -> UnitSpec:
    """Return one exact planned unit or reject an unplanned cell."""
    matches = [unit for unit in build_plan(manifest, tier) if unit.cell_id == cell_id]
    if len(matches) != 1:
        raise CausalContractError(f"cell_id is not planned exactly once: {cell_id!r}")
    return matches[0]


def block_units(
    manifest: Mapping[str, Any], tier: str, block_id: str
) -> tuple[UnitSpec, ...]:
    """Return all five members in their frozen execution order."""
    members = tuple(
        unit for unit in build_plan(manifest, tier) if unit.block_id == block_id
    )
    if len(members) != 5:
        raise CausalContractError(f"block_id is not planned exactly once: {block_id!r}")
    return members


def pair_units(
    manifest: Mapping[str, Any], tier: str, pair_id: str
) -> tuple[UnitSpec, ...]:
    """Return both five-form backend blocks in one worker's frozen order."""
    members = tuple(
        unit for unit in build_plan(manifest, tier) if unit.pair_id == pair_id
    )
    if len(members) != 10:
        raise CausalContractError(f"pair_id is not planned exactly once: {pair_id!r}")
    return members


@dataclass(frozen=True, slots=True)
class RunContext:
    """Parent-minted identity binding both backend blocks on one worker."""

    schema_version: int
    study_id: str
    manifest_sha256: str
    pair_id: str
    block_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    execution_order: tuple[str, ...]
    environment: Mapping[str, Any]
    environment_sha256: str
    git_commit: str
    worker_identity_sha256: str
    pair_execution_id: str
    execution_attempt_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return the strict canonical mapping stored by the parent."""
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "manifest_sha256": self.manifest_sha256,
            "pair_id": self.pair_id,
            "block_ids": list(self.block_ids),
            "cell_ids": list(self.cell_ids),
            "execution_order": list(self.execution_order),
            "environment": strict_json_loads(
                canonical_json_bytes(self.environment).decode()
            ),
            "environment_sha256": self.environment_sha256,
            "git_commit": self.git_commit,
            "worker_identity_sha256": self.worker_identity_sha256,
            "pair_execution_id": self.pair_execution_id,
            "execution_attempt_ids": list(self.execution_attempt_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        """Construct a strict context without accepting unbound extra fields."""
        _require_exact_keys(value, CONTEXT_KEYS, "run context")
        context = cls(
            schema_version=value["schema_version"],
            study_id=value["study_id"],
            manifest_sha256=value["manifest_sha256"],
            pair_id=value["pair_id"],
            block_ids=tuple(value["block_ids"]),
            cell_ids=tuple(value["cell_ids"]),
            execution_order=tuple(value["execution_order"]),
            environment=dict(value["environment"]),
            environment_sha256=value["environment_sha256"],
            git_commit=value["git_commit"],
            worker_identity_sha256=value["worker_identity_sha256"],
            pair_execution_id=value["pair_execution_id"],
            execution_attempt_ids=tuple(value["execution_attempt_ids"]),
        )
        validate_run_context(context)
        return context


def validate_run_context(
    context: RunContext,
    units: Sequence[UnitSpec] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> RunContext:
    """Validate context identity and optionally bind it to ten paired units."""
    if context.schema_version != SCHEMA_VERSION:
        raise CausalContractError("run context schema_version is wrong")
    if context.study_id != "truncated_hierarchy_causal_v3":
        raise CausalContractError("run context study_id is wrong")
    for field in (
        "manifest_sha256",
        "environment_sha256",
        "worker_identity_sha256",
        "pair_execution_id",
    ):
        if not SHA256.fullmatch(getattr(context, field)):
            raise CausalContractError(f"run context {field} must be SHA-256")
    if not GIT_SHA.fullmatch(context.git_commit):
        raise CausalContractError("run context git_commit must be a full commit hash")
    _validate_slug(context.pair_id, "run context pair_id", composite=True)
    if len(context.block_ids) != 2 or len(set(context.block_ids)) != 2:
        raise CausalContractError("run context must bind two distinct backend blocks")
    if any(
        _validate_slug(block_id, "run context block_id", composite=True) != block_id
        for block_id in context.block_ids
    ):
        raise AssertionError("validated block ID changed")
    if len(context.cell_ids) != 10 or len(set(context.cell_ids)) != 10:
        raise CausalContractError("run context must bind ten distinct cell IDs")
    if context.execution_order != context.cell_ids:
        raise CausalContractError(
            "execution_order must equal the frozen ordered cell_ids"
        )
    if len(context.execution_attempt_ids) != 10:
        raise CausalContractError("run context must bind ten execution attempts")
    if any(not SHA256.fullmatch(value) for value in context.execution_attempt_ids):
        raise CausalContractError("execution attempt IDs must be SHA-256 values")
    if len(set(context.execution_attempt_ids)) != 10:
        raise CausalContractError("execution attempt IDs must be distinct")
    if not isinstance(context.environment, Mapping):
        raise CausalContractError("run context environment must be an object")
    if context.environment.get("environment_sha256") != context.environment_sha256:
        raise CausalContractError(
            "embedded environment self-digest is not context-bound"
        )
    if environment_digest(context.environment) != context.environment_sha256:
        raise CausalContractError("embedded environment bytes have the wrong digest")
    environment_git = context.environment.get("git")
    if not isinstance(environment_git, Mapping) or (
        environment_git.get("commit") != context.git_commit
    ):
        raise CausalContractError("embedded environment commit is not context-bound")
    if manifest is not None:
        validate_environment(context.environment, manifest)
    if units is not None:
        if len(units) != 10 or {unit.pair_id for unit in units} != {context.pair_id}:
            raise CausalContractError("context units must be one complete backend pair")
        ordered_blocks = tuple(dict.fromkeys(unit.block_id for unit in units))
        if context.block_ids != ordered_blocks:
            raise CausalContractError("context backend block order does not match plan")
        ordered = tuple(unit.cell_id for unit in units)
        if context.cell_ids != ordered:
            raise CausalContractError("context cell order does not match the plan")
        if {unit.manifest_sha256 for unit in units} != {context.manifest_sha256}:
            raise CausalContractError("context manifest does not match the plan")
    return context


def write_run_context(path: Path, context: RunContext) -> Path:
    """Atomically publish a validated parent context without overwrite."""
    validate_run_context(context)
    _atomic_write(path, canonical_json_bytes(context.as_dict()))
    return path


def load_run_context(path: Path) -> RunContext:
    """Load a strict parent context."""
    value = _load_json(path)
    if not isinstance(value, Mapping):
        raise CausalContractError("run context must contain an object")
    return RunContext.from_dict(value)


def context_path(root: Path, pair_id: str) -> Path:
    """Return the parent-minted backend-pair context path."""
    _validate_slug(pair_id, "pair_id", composite=True)
    return root / "contexts" / f"{pair_id}.json"


def data_artifact_path(root: Path, unit: UnitSpec) -> Path:
    """Return the shared immutable data path for a unit."""
    return root / "data" / f"{unit.data_id}.json"


def natural_start_artifact_path(root: Path, unit: UnitSpec) -> Path:
    """Return the shared natural-start path for a unit."""
    return root / "starts" / "natural" / f"{unit.start_id}.json"


def coordinate_start_artifact_path(root: Path, unit: UnitSpec) -> Path:
    """Return the representation-specific coordinate-start path."""
    return root / "starts" / "coordinates" / f"{unit.cell_id}.json"


def chain_artifact_path(root: Path, unit: UnitSpec) -> Path:
    """Return the standardized natural-scale posterior path."""
    return root / "chains" / f"{unit.cell_id}.nc"


def diagnostic_artifact_path(root: Path, unit: UnitSpec) -> Path:
    """Return the raw diagnostic and oracle evidence path."""
    return root / "diagnostics" / f"{unit.cell_id}.json"


def cell_result_path(root: Path, unit: UnitSpec) -> Path:
    """Return the final atomic cell-marker path."""
    return root / "cells" / f"{unit.cell_id}.json"


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise CausalContractError(f"cannot collect git provenance: {error}") from error


def collect_environment(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Collect a strict environment attestation without importing samplers."""
    validate_manifest(manifest)
    profile = manifest["dependency_profile"]
    packages: dict[str, str | None] = {}
    for package, expected in profile["required_versions"].items():
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
        if packages[package] != expected:
            # Collection remains useful for diagnosis.  Validation below is the
            # hard execution gate and will reject the drifted record.
            continue
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": manifest["study_id"],
        "manifest_sha256": manifest_digest(manifest),
        "runner_version": RUNNER_VERSION,
        "dependency_profile": profile["name"],
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "project": {
            "path": profile["project_path"],
            "sha256": sha256_file(REPO_ROOT / profile["project_path"]),
            "lock_path": profile["lock_path"],
            "lock_sha256": sha256_file(REPO_ROOT / profile["lock_path"]),
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
        },
        "packages": packages,
    }
    record["environment_sha256"] = environment_digest(record)
    return record


def environment_digest(record: Mapping[str, Any]) -> str:
    """Hash an environment record independent of its self-digest field."""
    value = dict(record)
    value.pop("environment_sha256", None)
    return sha256_bytes(canonical_json_bytes(value))


def validate_environment(
    record: Mapping[str, Any], manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Require the exact frozen dependency and clean source environment."""
    expected_keys = frozenset(
        {
            "schema_version",
            "study_id",
            "manifest_sha256",
            "runner_version",
            "dependency_profile",
            "git",
            "project",
            "runtime",
            "packages",
            "environment_sha256",
        }
    )
    _require_exact_keys(record, expected_keys, "environment")
    if (
        record["schema_version"] != SCHEMA_VERSION
        or record["runner_version"] != RUNNER_VERSION
    ):
        raise CausalContractError("environment schema/runner version is wrong")
    if record["study_id"] != manifest["study_id"]:
        raise CausalContractError("environment study_id is wrong")
    if record["manifest_sha256"] != manifest_digest(manifest):
        raise CausalContractError("environment manifest digest is wrong")
    profile = manifest["dependency_profile"]
    if record["dependency_profile"] != profile["name"]:
        raise CausalContractError("environment dependency profile is wrong")
    if record["packages"] != profile["required_versions"]:
        raise CausalContractError("environment package versions do not match the lock")
    if not record["runtime"]["python"].startswith(f"{profile['python']}."):
        raise CausalContractError("environment Python minor version is wrong")
    if record["git"]["dirty"] is not False:
        raise CausalContractError("scientific execution requires a clean worktree")
    if record["project"]["sha256"] != profile["project_sha256"]:
        raise CausalContractError("environment project hash is wrong")
    if record["project"]["lock_sha256"] != profile["lock_sha256"]:
        raise CausalContractError("environment lock hash is wrong")
    if record["environment_sha256"] != environment_digest(record):
        raise CausalContractError("environment self-digest is wrong")
    return record


def _validate_artifact_ref(value: Any, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise CausalContractError(f"{path} must be null or an artifact reference")
    _require_exact_keys(value, ARTIFACT_REF_KEYS, path)
    _relative_contract_path(value["path"])
    if not isinstance(value["sha256"], str) or not SHA256.fullmatch(value["sha256"]):
        raise CausalContractError(f"{path}.sha256 must be lowercase SHA-256")
    if not _is_int(value["size_bytes"]) or value["size_bytes"] < 0:
        raise CausalContractError(f"{path}.size_bytes must be non-negative")


def _validate_metrics(metrics: Any, *, completed: bool, tier: str) -> None:
    if not isinstance(metrics, Mapping):
        raise CausalContractError("metrics must be an object")
    unknown = set(metrics) - ALLOWED_METRICS
    if unknown:
        raise CausalContractError(
            f"metrics contain unregistered fields: {sorted(unknown)}"
        )
    required = (
        (SCREENING_METRICS | ORACLE_METRICS)
        if tier == "smoke"
        else (CONFIRMATION_METRICS | ORACLE_METRICS)
    )
    if completed and not required <= set(metrics):
        raise CausalContractError(
            f"completed {tier} result misses metrics: {sorted(required - set(metrics))}"
        )
    observed_oracle = set(metrics) & ORACLE_METRICS
    if observed_oracle and observed_oracle != ORACLE_METRICS:
        raise CausalContractError(
            "oracle evidence must publish the complete registered metric set"
        )
    for key, value in metrics.items():
        if key in BOOL_METRICS and not isinstance(value, bool):
            raise CausalContractError(f"metrics.{key} must be Boolean")
        if key in INTEGER_METRICS and (not _is_int(value) or value < 0):
            raise CausalContractError(f"metrics.{key} must be a non-negative integer")
        if key in FLOAT_METRICS and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise CausalContractError(f"metrics.{key} must be finite and non-negative")
    trio = {"divergence_count", "posterior_draw_count", "divergence_rate"}
    if trio <= set(metrics):
        if metrics["posterior_draw_count"] <= 0:
            raise CausalContractError("posterior_draw_count must be positive")
        expected = metrics["divergence_count"] / metrics["posterior_draw_count"]
        if not math.isclose(metrics["divergence_rate"], expected, abs_tol=1e-15):
            raise CausalContractError("divergence count/rate/draw count disagree")


def _validate_parameter_summaries(value: Any, *, completed: bool) -> None:
    if not isinstance(value, list):
        raise CausalContractError("parameter_summaries must be a list")
    if not completed and value:
        raise CausalContractError("failed cells cannot publish posterior summaries")
    identities: set[tuple[str, int | None]] = set()
    for index, summary in enumerate(value):
        if not isinstance(summary, Mapping):
            raise CausalContractError(f"parameter_summaries[{index}] must be an object")
        _require_exact_keys(
            summary, PARAMETER_SUMMARY_KEYS, f"parameter_summaries[{index}]"
        )
        if not isinstance(summary["parameter_id"], str) or not SAFE_PARAMETER.fullmatch(
            summary["parameter_id"]
        ):
            raise CausalContractError("parameter_id must be a canonical parameter name")
        if summary["index"] is not None and (
            not _is_int(summary["index"]) or summary["index"] < 0
        ):
            raise CausalContractError(
                "parameter summary index must be null/non-negative"
            )
        identity = (summary["parameter_id"], summary["index"])
        if identity in identities:
            raise CausalContractError("parameter summary identities must be unique")
        identities.add(identity)
        for field in ("mean", "sd", "mcse_mean"):
            number = summary[field]
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
            ):
                raise CausalContractError(f"parameter summary {field} must be finite")
        if summary["sd"] < 0 or summary["mcse_mean"] < 0:
            raise CausalContractError("posterior SD/MCSE must be non-negative")


def derive_pymc_chain_rng_provenance(
    entropy_words: Sequence[int], chains: int
) -> list[dict[str, Any]]:
    """Derive the exact PyMC 6.3.1 spawned-generator provenance.

    PyMC does not consume a supplied integer sequence as one seed per chain.  It
    treats the sequence as ``SeedSequence`` entropy, spawns one generator per
    chain, draws one integer for initial-point/NUTS-step construction, and then
    hands each advanced generator to the chain sampler.
    """
    if not _is_int(chains) or chains <= 0:
        raise CausalContractError("PyMC chains must be a positive integer")
    if len(entropy_words) != chains:
        raise CausalContractError("PyMC entropy-word count must equal chains")
    for word in entropy_words:
        _validate_seed(word, "PyMC entropy word")
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - NumPy is a frozen dependency.
        raise CausalContractError(
            "NumPy is required to validate effective PyMC chain seeds"
        ) from error
    result: list[dict[str, Any]] = []
    generators = np.random.default_rng(list(entropy_words)).spawn(chains)
    for chain, generator in enumerate(generators):
        seed_sequence = generator.bit_generator.seed_seq
        spawn_key = list(getattr(seed_sequence, "spawn_key"))
        pool_size = int(getattr(seed_sequence, "pool_size"))
        if spawn_key != [chain] or pool_size != 4:
            raise CausalContractError("pinned NumPy SeedSequence semantics changed")
        init_step_seed = int(generator.integers(2**30))
        result.append(
            {
                "chain": chain,
                "rng": "numpy.random.Generator(PCG64)",
                "spawn_key": spawn_key,
                "seed_sequence_pool_size": pool_size,
                "init_step_seed": init_step_seed,
                "post_init_draw_state_sha256": sha256_bytes(
                    canonical_json_bytes(generator.bit_generator.state)
                ),
            }
        )
    return result


def derive_numpyro_chain_keys(seed: int, chains: int) -> list[list[int]]:
    """Derive the exact pinned JAX PRNG keys consumed by NumPyro."""
    _validate_seed(seed, "NumPyro sampler seed")
    if not _is_int(chains) or chains <= 0:
        raise CausalContractError("NumPyro chains must be a positive integer")
    try:
        import jax  # Imported only when validating NumPyro execution evidence.
        import numpy as np
    except ImportError as error:
        raise CausalContractError(
            "JAX is required to validate effective NumPyro PRNG keys"
        ) from error
    master = jax.random.PRNGKey(seed)
    keys = master[None, :] if chains == 1 else jax.random.split(master, chains)
    array = np.asarray(keys, dtype=np.uint32)
    return [[int(item) for item in row] for row in array]


def validate_result_record(
    record: Mapping[str, Any], unit: UnitSpec, context: RunContext | None = None
) -> Mapping[str, Any]:
    """Validate one final cell marker against its exact plan and parent context."""
    if not isinstance(record, Mapping):
        raise CausalContractError("result record must be an object")
    _require_exact_keys(record, RESULT_KEYS, "result record")
    expected_identity = {
        "schema_version": unit.schema_version,
        "runner_version": RUNNER_VERSION,
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
    }
    for field, expected in expected_identity.items():
        if record[field] != expected:
            raise CausalContractError(f"result {field} does not match the plan")
    status = record["execution_status"]
    if status not in {"completed", "failed"}:
        raise CausalContractError("execution_status must be completed or failed")
    completed = status == "completed"
    _validate_metrics(record["metrics"], completed=completed, tier=unit.tier)
    _validate_parameter_summaries(record["parameter_summaries"], completed=completed)

    artifacts = record["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise CausalContractError("artifacts must be an object")
    _require_exact_keys(artifacts, ARTIFACT_KEYS, "artifacts")
    for key, value in artifacts.items():
        _validate_artifact_ref(value, f"artifacts.{key}")
    expected_paths = {
        "context": f"contexts/{unit.pair_id}.json",
        "data": f"data/{unit.data_id}.json",
        "natural_start": f"starts/natural/{unit.start_id}.json",
        "coordinate_start": f"starts/coordinates/{unit.cell_id}.json",
        "chain": f"chains/{unit.cell_id}.nc",
        "diagnostics": f"diagnostics/{unit.cell_id}.json",
    }
    for key, reference in artifacts.items():
        if reference is not None and reference["path"] != expected_paths[key]:
            raise CausalContractError(f"artifacts.{key}.path does not match the plan")
    if completed and any(artifacts[key] is None for key in ARTIFACT_KEYS):
        raise CausalContractError("completed cells require all six artifact references")

    failure = record["failure"]
    if completed:
        if failure is not None:
            raise CausalContractError("completed cells cannot contain failure details")
    else:
        if not isinstance(failure, Mapping):
            raise CausalContractError("failed cells require failure details")
        _require_exact_keys(failure, FAILURE_KEYS, "failure")
        if failure["stage"] not in SCIENTIFIC_FAILURE_STAGES:
            raise CausalContractError("failure stage is not a scientific stage")
        if not all(
            isinstance(failure[field], str) and failure[field]
            for field in ("error_type", "message")
        ):
            raise CausalContractError("failure type/message must be non-empty strings")
        required_before_failure = {
            "data": (),
            "build": ("data",),
            "initialize": ("data",),
            "compile": ("data", "natural_start", "coordinate_start"),
            "sample": ("data", "natural_start", "coordinate_start"),
            "summarize": ("data", "natural_start", "coordinate_start", "chain"),
            "diagnose": ("data", "natural_start", "coordinate_start", "chain"),
        }[failure["stage"]]
        if any(artifacts[key] is None for key in required_before_failure):
            raise CausalContractError(
                f"failure at {failure['stage']} omits a completed prerequisite artifact"
            )

    has_oracle_evidence = ORACLE_METRICS <= set(record["metrics"])
    if completed:
        expected_oracle_count = _expected_oracle_evaluation_count(
            record,
            unit.tier,
            posterior_trajectory=True,
        )
        if record["metrics"]["oracle_evaluation_count"] != expected_oracle_count:
            raise CausalContractError(
                "completed cell oracle count omits frozen trajectory/static evidence"
            )
    elif has_oracle_evidence:
        failure_stage = failure["stage"]
        if failure_stage not in {"compile", "sample", "diagnose", "summarize"}:
            raise CausalContractError(
                "oracle evidence is invalid before graph evaluation"
            )
        if artifacts["diagnostics"] is None:
            raise CausalContractError("failed oracle evidence requires diagnostics")
        if failure_stage in {"compile", "sample"} and artifacts["chain"] is not None:
            raise CausalContractError(
                "compile/sample-stage pre-oracle failure cannot publish a chain"
            )
        expected_basic = {
            "initialization_success": True,
            "logp_finite": True,
            "gradient_finite": True,
        }
        expected_basic["compile_success"] = failure_stage != "compile"
        if failure_stage in {"sample", "diagnose", "summarize"}:
            expected_basic["sampling_success"] = failure_stage != "sample"
        if any(
            record["metrics"].get(key) is not value
            for key, value in expected_basic.items()
        ):
            raise CausalContractError(
                "failed oracle result has inconsistent execution metrics"
            )
        posterior_trajectory = failure_stage == "summarize"
        expected_oracle_count = _expected_oracle_evaluation_count(
            record,
            unit.tier,
            posterior_trajectory=posterior_trajectory,
        )
        if record["metrics"]["oracle_evaluation_count"] != expected_oracle_count:
            raise CausalContractError(
                "failed cell oracle count differs from its frozen evidence phase"
            )

    provenance = record["provenance"]
    if not isinstance(provenance, Mapping):
        raise CausalContractError("provenance must be an object")
    _require_exact_keys(provenance, PROVENANCE_KEYS, "provenance")
    for field in (
        "environment_sha256",
        "worker_identity_sha256",
        "pair_execution_id",
        "execution_attempt_id",
    ):
        if not isinstance(provenance[field], str) or not SHA256.fullmatch(
            provenance[field]
        ):
            raise CausalContractError(f"provenance.{field} must be SHA-256")
    if not GIT_SHA.fullmatch(provenance["git_commit"]):
        raise CausalContractError("provenance.git_commit must be a full commit hash")
    if (
        provenance["floatx"] != unit.floatx
        or provenance["pytensor_floatx"] != unit.floatx
    ):
        raise CausalContractError("result precision differs from the planned regime")
    if provenance["jax_enable_x64"] is not (unit.floatx == "float64"):
        raise CausalContractError("JAX x64 state differs from the planned precision")
    expected_backend = {
        "pymc": ("pymc-nuts", "pytensor"),
        "numpyro": ("numpyro-nuts-via-pymc", "pytensor-to-jax"),
    }[unit.backend_id]
    if (provenance["sampler"], provenance["compiler_path"]) != expected_backend:
        raise CausalContractError("sampler/compiler provenance differs from the plan")
    if provenance["device"] != "cpu":
        raise CausalContractError("causal evidence must come from CPU execution")
    expected_seed_input: int | list[int]
    expected_chain_rng: list[dict[str, Any]]
    if unit.backend_id == "pymc":
        expected_seed_input = list(unit.chain_seeds)
        expected_chain_rng = derive_pymc_chain_rng_provenance(
            unit.chain_seeds, unit.chains
        )
    else:
        if unit.sampler_seed is None:  # pragma: no cover - unit validation guards it
            raise AssertionError("NumPyro unit without a sampler seed")
        expected_seed_input = unit.sampler_seed
        expected_chain_rng = [
            {"chain": chain, "rng": "jax-prng-key", "key": key}
            for chain, key in enumerate(
                derive_numpyro_chain_keys(unit.sampler_seed, unit.chains)
            )
        ]
    if provenance["sampler_seed_input"] != expected_seed_input:
        raise CausalContractError("backend sampler seed input differs from the plan")
    if provenance["chain_rng_provenance"] != expected_chain_rng:
        raise CausalContractError("backend chain RNG provenance differs from the plan")
    if context is not None:
        validate_run_context(context)
        if context.pair_id != unit.pair_id or unit.block_id not in context.block_ids:
            raise CausalContractError("result context belongs to another backend pair")
        position = context.cell_ids.index(unit.cell_id)
        context_bindings = {
            "environment_sha256": context.environment_sha256,
            "git_commit": context.git_commit,
            "worker_identity_sha256": context.worker_identity_sha256,
            "pair_execution_id": context.pair_execution_id,
            "execution_attempt_id": context.execution_attempt_ids[position],
        }
        for field, expected in context_bindings.items():
            if provenance[field] != expected:
                raise CausalContractError(
                    f"result provenance.{field} is not parent-bound"
                )
    return record


def _require_derived_metric(
    record: Mapping[str, Any], name: str, observed: int | float | bool
) -> None:
    """Require a registered scalar to equal independently derived evidence."""
    claimed = record["metrics"].get(name)
    if isinstance(observed, bool) or isinstance(observed, int):
        matches = claimed == observed and type(claimed) is type(observed)
    else:
        matches = isinstance(claimed, (int, float)) and not isinstance(claimed, bool)
        matches = bool(
            matches
            and math.isclose(
                float(claimed),
                float(observed),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
    if not matches:
        raise CausalContractError(
            f"metrics.{name} disagrees with hash-bound raw evidence"
        )


def _audit_diagnostic_evidence(
    record: Mapping[str, Any],
    unit: UnitSpec,
    path: Path,
    analysis_policy: Mapping[str, Any],
) -> None:
    """Recompute oracle metrics and ICDF gates from hash-bound raw arrays."""
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - NumPy is a frozen dependency.
        raise CausalContractError(
            "NumPy is required to audit diagnostic evidence"
        ) from error
    diagnostics = _load_json(path)
    if not isinstance(diagnostics, Mapping):
        raise CausalContractError("diagnostics artifact must contain an object")
    identity = {
        "schema_version": unit.schema_version,
        "study_id": unit.study_id,
        "manifest_sha256": unit.manifest_sha256,
        "cell_id": unit.cell_id,
    }
    if any(diagnostics.get(key) != value for key, value in identity.items()):
        raise CausalContractError("diagnostics artifact identity differs from the plan")
    runtime = diagnostics.get("runtime")
    if not isinstance(runtime, Mapping):
        raise CausalContractError("diagnostics artifact lacks child runtime evidence")
    if (
        runtime.get("pytensor_floatx") != record["provenance"]["pytensor_floatx"]
        or runtime.get("jax_enable_x64") is not record["provenance"]["jax_enable_x64"]
    ):
        raise CausalContractError("diagnostic runtime differs from result provenance")
    oracle = diagnostics.get("oracle")
    has_registered_oracle = ORACLE_METRICS <= set(record["metrics"])
    if not has_registered_oracle:
        if oracle is not None:
            raise CausalContractError(
                "diagnostic oracle exists without registered oracle metrics"
            )
        return
    if not isinstance(oracle, Mapping) or not isinstance(oracle.get("records"), list):
        raise CausalContractError("diagnostics artifact lacks raw oracle records")
    completed = record["execution_status"] == "completed"
    failure = record.get("failure")
    failure_stage = failure.get("stage") if isinstance(failure, Mapping) else None
    expected_posterior = bool(completed or failure_stage == "summarize")
    if oracle.get("posterior_trajectory_evaluated") is not expected_posterior:
        raise CausalContractError("oracle phase does not match cell completion status")
    records = oracle["records"]
    expected_count = _expected_oracle_evaluation_count(
        record,
        unit.tier,
        posterior_trajectory=expected_posterior,
    )
    if len(records) != expected_count:
        raise CausalContractError("raw oracle record count differs from frozen design")

    try:
        gate = analysis_policy["oracle_gate"]
        tolerances = gate["component_tolerances"][unit.floatx]
        allowed_scaled_error = float(gate["scaled_error_max"])
        roundtrip_limit = float(gate["roundtrip_absolute_error_max"][unit.floatx])
    except (KeyError, TypeError, ValueError) as error:
        raise CausalContractError("oracle analysis policy is incomplete") from error
    component_tolerances = {
        "value": tolerances["logp"],
        "gradient": tolerances["gradient"],
        "hessian": tolerances["hessian"],
    }

    def contains_boolean(value: Any) -> bool:
        if isinstance(value, bool):
            return True
        if isinstance(value, list):
            return any(contains_boolean(item) for item in value)
        return False

    def as_finite_array(value: Any, source: str) -> Any:
        if value is None or contains_boolean(value):
            raise CausalContractError(f"{source} is not a finite numeric array")
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise CausalContractError(
                f"{source} is not a finite numeric array"
            ) from error
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise CausalContractError(f"{source} is not a finite numeric array")
        return array

    def scaled_error(
        observed: Any,
        expected: Any,
        tolerance: Mapping[str, Any],
        source: str,
    ) -> tuple[dict[str, float], bool, Any | None, Any]:
        expected_array = as_finite_array(expected, f"{source}.oracle")
        if observed is None:
            return (
                {
                    "absolute_max": sys.float_info.max,
                    "scaled_max": allowed_scaled_error + 1.0,
                },
                False,
                None,
                expected_array,
            )
        observed_array = as_finite_array(observed, f"{source}.observed")
        if observed_array.shape != expected_array.shape:
            raise CausalContractError(f"{source} observed/oracle shapes differ")
        absolute_tolerance = float(tolerance["absolute_tolerance"])
        relative_tolerance = float(tolerance["relative_tolerance"])
        difference = np.abs(observed_array - expected_array)
        scale = absolute_tolerance + relative_tolerance * np.maximum(
            np.abs(observed_array), np.abs(expected_array)
        )
        if absolute_tolerance <= 0 or relative_tolerance < 0 or np.any(scale <= 0):
            raise CausalContractError(f"{source} has invalid frozen tolerances")
        return (
            {
                "absolute_max": float(np.max(difference)),
                "scaled_max": float(np.max(difference / scale)),
            },
            True,
            observed_array,
            expected_array,
        )

    def require_error_summary(
        claimed: Any, observed: Mapping[str, float], source: str
    ) -> None:
        if not isinstance(claimed, Mapping):
            raise CausalContractError(f"{source} must be an error summary")
        _require_exact_keys(claimed, frozenset({"absolute_max", "scaled_max"}), source)
        for name, expected in observed.items():
            value = claimed[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not math.isclose(
                    float(value), expected, rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise CausalContractError(
                    f"{source}.{name} disagrees with raw observed/oracle arrays"
                )

    audited_records: list[dict[str, Any]] = []
    coordinate_size = int(unit.regime["n_groups"]) + 2
    component_shapes = {
        "value": (),
        "gradient": (coordinate_size,),
        "hessian": (coordinate_size, coordinate_size),
    }
    for index, item in enumerate(records):
        source = f"oracle.records[{index}]"
        if not isinstance(item, Mapping):
            raise CausalContractError(f"{source} must be an object")
        try:
            observed_components = item["observed"]
            oracle_components = item["oracle"]
            claimed_errors = item["errors"]
            claimed_finite = item["component_finite"]
            roundtrip_error = float(item["roundtrip"]["absolute_error_max"])
            kind = item["kind"]
        except (KeyError, TypeError, ValueError) as error:
            raise CausalContractError(
                "raw oracle record schema is incomplete"
            ) from error
        if not all(
            isinstance(value, Mapping)
            for value in (
                observed_components,
                oracle_components,
                claimed_errors,
                claimed_finite,
            )
        ):
            raise CausalContractError(f"{source} component mappings are malformed")
        component_names = frozenset({"value", "gradient", "hessian"})
        for mapping_name, value in (
            ("observed", observed_components),
            ("oracle", oracle_components),
            ("errors", claimed_errors),
            ("component_finite", claimed_finite),
        ):
            _require_exact_keys(value, component_names, f"{source}.{mapping_name}")
        if (
            not isinstance(kind, str)
            or not kind
            or not math.isfinite(roundtrip_error)
            or roundtrip_error < 0
        ):
            raise CausalContractError(f"{source} kind/roundtrip is malformed")
        component_errors: dict[str, dict[str, float]] = {}
        finite_components: dict[str, bool] = {}
        raw_arrays: dict[str, tuple[Any | None, Any]] = {}
        for component in ("value", "gradient", "hessian"):
            derived_error, finite, observed_array, oracle_array = scaled_error(
                observed_components[component],
                oracle_components[component],
                component_tolerances[component],
                f"{source}.{component}",
            )
            if oracle_array.shape != component_shapes[component]:
                raise CausalContractError(
                    f"{source}.{component} has the wrong coordinate shape"
                )
            require_error_summary(
                claimed_errors[component],
                derived_error,
                f"{source}.errors.{component}",
            )
            if claimed_finite[component] is not finite:
                raise CausalContractError(
                    f"{source}.component_finite.{component} disagrees with raw arrays"
                )
            component_errors[component] = derived_error
            finite_components[component] = finite
            raw_arrays[component] = (observed_array, oracle_array)
        finite = all(finite_components.values())
        passed = bool(
            finite
            and roundtrip_error <= roundtrip_limit
            and all(
                error["scaled_max"] <= allowed_scaled_error
                for error in component_errors.values()
            )
        )
        if item.get("finite") is not finite or item.get("passed") is not passed:
            raise CausalContractError(
                f"{source} finite/passed summary disagrees with raw arrays"
            )
        audited_records.append(
            {
                "item": item,
                "kind": kind,
                "errors": component_errors,
                "finite": finite,
                "passed": passed,
                "roundtrip_error": roundtrip_error,
                "arrays": raw_arrays,
            }
        )

    expected_points: dict[str, tuple[str, int | None]] = {}
    if unit.replicate == 0:
        expected_points["fixed-truth"] = ("fixed-grid", None)
    for chain in range(unit.chains):
        expected_points[f"start-chain-{chain:02d}"] = ("shared-natural-start", None)
    expected_icdf_indices = {
        "native-centered": (),
        "manual-centered": (),
        "group-icdf-noncentered": (2,),
        "location-icdf-noncentered": (0,),
        "full-icdf-noncentered": (0, 2),
    }[unit.representation_id]
    icdf_labels = (
        "branch-left",
        "branch-zero",
        "branch-right",
        "tail-low",
        "tail-high",
    )
    for coordinate_index in expected_icdf_indices:
        for label in icdf_labels:
            expected_points[f"icdf-{coordinate_index}-{label}"] = (
                f"icdf-{label}",
                coordinate_index,
            )
    if expected_posterior:
        for chain in range(unit.chains):
            expected_points[f"trajectory-chain-{chain:02d}-selection-00"] = (
                "hash-selected-posterior-trajectory",
                None,
            )
    observed_points: dict[str, Mapping[str, Any]] = {}
    for audited in audited_records:
        item = audited["item"]
        point_id = item.get("point_id")
        if not isinstance(point_id, str) or not point_id or point_id in observed_points:
            raise CausalContractError("oracle point IDs must be unique strings")
        observed_points[point_id] = item
    if set(observed_points) != set(expected_points):
        raise CausalContractError(
            "oracle point structure differs from the frozen static/ICDF/trajectory grid"
        )
    expected_branch_epsilon = float(
        8.0 * math.sqrt(np.finfo(np.dtype(unit.floatx)).eps)
    )
    for point_id, (kind, expected_coordinate_index) in expected_points.items():
        item = observed_points[point_id]
        if item.get("kind") != kind:
            raise CausalContractError(
                "oracle point structure differs from the frozen point kinds"
            )
        if expected_coordinate_index is not None:
            epsilon = item.get("branch_epsilon")
            if (
                item.get("coordinate_index") != expected_coordinate_index
                or isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not math.isclose(
                    float(epsilon),
                    expected_branch_epsilon,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            ):
                raise CausalContractError(
                    "ICDF point coordinate/epsilon differs from the frozen grid"
                )
        if kind == "hash-selected-posterior-trajectory":
            trajectory_chain = item.get("chain")
            trajectory_draw = item.get("draw")
            if (
                not isinstance(trajectory_chain, int)
                or isinstance(trajectory_chain, bool)
                or not isinstance(trajectory_draw, int)
                or isinstance(trajectory_draw, bool)
                or trajectory_chain < 0
                or trajectory_chain >= unit.chains
                or trajectory_draw < 0
                or trajectory_draw >= unit.draws
                or point_id != f"trajectory-chain-{trajectory_chain:02d}-selection-00"
            ):
                raise CausalContractError(
                    "posterior trajectory point differs from the frozen hash selection"
                )
            expected_draw = min(
                range(unit.draws),
                key=lambda draw: hashlib.sha256(
                    f"{unit.cell_id}:trajectory:{trajectory_chain}:{draw}".encode()
                ).digest(),
            )
            expected_selection_sha256 = hashlib.sha256(
                (
                    f"{unit.cell_id}:trajectory:{trajectory_chain}:{expected_draw}"
                ).encode()
            ).hexdigest()
            if (
                trajectory_draw != expected_draw
                or item.get("selection_sha256") != expected_selection_sha256
            ):
                raise CausalContractError(
                    "posterior trajectory point is not the frozen lowest-SHA draw"
                )

    tail_finite = all(
        item["finite"]
        for item in audited_records
        if item["kind"].startswith("icdf-tail")
    )
    branch_groups: dict[int, dict[str, Mapping[str, Any]]] = {}
    for item in audited_records:
        if not item["kind"].startswith("icdf-branch"):
            continue
        coordinate_index = item["item"].get("coordinate_index")
        if not _is_int(coordinate_index) or coordinate_index < 0:
            raise CausalContractError("ICDF branch record lacks a coordinate index")
        group = branch_groups.setdefault(coordinate_index, {})
        if item["kind"] in group:
            raise CausalContractError("ICDF branch diagnostic kind is duplicated")
        group[item["kind"]] = item

    branch_results: dict[int, bool] = {}
    required_branch_kinds = {
        "icdf-branch-left",
        "icdf-branch-zero",
        "icdf-branch-right",
    }
    for coordinate_index, group in branch_groups.items():
        if set(group) != required_branch_kinds:
            raise CausalContractError("ICDF branch diagnostic triplet is incomplete")
        left = group["icdf-branch-left"]
        zero = group["icdf-branch-zero"]
        right = group["icdf-branch-right"]
        branch_finite = bool(left["finite"] and zero["finite"] and right["finite"])
        if branch_finite:
            left_value, left_oracle_value = left["arrays"]["value"]
            right_value, right_oracle_value = right["arrays"]["value"]
            left_gradient, left_oracle_gradient = left["arrays"]["gradient"]
            right_gradient, right_oracle_gradient = right["arrays"]["gradient"]
            value_jump_error, _, _, _ = scaled_error(
                right_value - left_value,
                right_oracle_value - left_oracle_value,
                component_tolerances["value"],
                f"oracle.icdf_branch[{coordinate_index}].value_jump",
            )
            gradient_jump_error, _, _, _ = scaled_error(
                right_gradient - left_gradient,
                right_oracle_gradient - left_oracle_gradient,
                component_tolerances["gradient"],
                f"oracle.icdf_branch[{coordinate_index}].gradient_jump",
            )
            jump_passed = bool(
                value_jump_error["scaled_max"] <= allowed_scaled_error
                and gradient_jump_error["scaled_max"] <= allowed_scaled_error
            )
        else:
            jump_passed = False
        branch_results[coordinate_index] = bool(
            left["passed"] and zero["passed"] and right["passed"] and jump_passed
        )

    branch_continuous = all(branch_results.values())
    branch_checks = oracle.get("icdf_branch_checks")
    if not isinstance(branch_checks, list) or len(branch_checks) != len(branch_results):
        raise CausalContractError(
            "ICDF branch-check summaries differ from raw triplets"
        )
    claimed_branch_results: dict[int, bool] = {}
    for check in branch_checks:
        if not isinstance(check, Mapping):
            raise CausalContractError("ICDF branch-check summary must be an object")
        claimed_index = check.get("coordinate_index")
        claimed_passed = check.get("passed")
        if (
            not isinstance(claimed_index, int)
            or isinstance(claimed_index, bool)
            or not isinstance(claimed_passed, bool)
            or claimed_index in claimed_branch_results
        ):
            raise CausalContractError("ICDF branch-check summary is malformed")
        claimed_branch_results[claimed_index] = claimed_passed
    if claimed_branch_results != branch_results:
        raise CausalContractError(
            "ICDF branch-check summaries disagree with raw diagnostic arrays"
        )
    if (
        oracle.get("icdf_tail_finite") is not tail_finite
        or oracle.get("icdf_branch_continuous") is not branch_continuous
    ):
        raise CausalContractError(
            "ICDF summary flags disagree with raw diagnostic arrays"
        )
    oracle_passed = (
        all(item["passed"] for item in audited_records) and branch_continuous
    )
    if oracle.get("passed") is not oracle_passed:
        raise CausalContractError("oracle passed summary disagrees with raw arrays")

    try:
        derived: dict[str, int | float | bool] = {
            "oracle_evaluation_count": len(records),
            "oracle_logp_scaled_error_max": max(
                item["errors"]["value"]["scaled_max"] for item in audited_records
            ),
            "oracle_gradient_scaled_error_max": max(
                item["errors"]["gradient"]["scaled_max"] for item in audited_records
            ),
            "oracle_hessian_scaled_error_max": max(
                item["errors"]["hessian"]["scaled_max"] for item in audited_records
            ),
            "roundtrip_absolute_error_max": max(
                item["roundtrip_error"] for item in audited_records
            ),
            "icdf_tail_finite": tail_finite,
            "icdf_branch_continuous": branch_continuous,
        }
    except (KeyError, TypeError, ValueError) as error:
        raise CausalContractError("raw oracle record schema is incomplete") from error
    for name, value in derived.items():
        _require_derived_metric(record, name, value)


def _audit_oracle_point_bindings(
    record: Mapping[str, Any],
    unit: UnitSpec,
    diagnostics_path: Path,
    data_path: Path,
    natural_start_path: Path,
    coordinate_start_path: Path,
    chain_path: Path | None,
    analysis_policy: Mapping[str, Any],
) -> None:
    """Bind every oracle coordinate to its immutable natural-scale source."""
    if not ORACLE_METRICS <= set(record["metrics"]):
        return
    try:
        import numpy as np
        import xarray as xr

        from scripts.truncated_hierarchy_causal_oracle import (
            HierarchicalPosteriorSpec,
            TruncationBounds,
            hierarchical_natural_values,
        )
    except ImportError as error:  # pragma: no cover - frozen dependencies.
        raise CausalContractError(
            "NumPy, xarray, and the independent oracle are required to bind points"
        ) from error

    diagnostics = _load_json(diagnostics_path)
    data = _load_json(data_path)
    natural_starts = _load_json(natural_start_path)
    coordinate_starts = _load_json(coordinate_start_path)
    if not all(
        isinstance(value, Mapping)
        for value in (diagnostics, data, natural_starts, coordinate_starts)
    ):
        raise CausalContractError("oracle source artifacts must contain objects")
    oracle = diagnostics.get("oracle")
    if not isinstance(oracle, Mapping) or not isinstance(oracle.get("records"), list):
        raise CausalContractError("diagnostics artifact lacks raw oracle records")

    identities = (
        (
            data,
            {
                "schema_version": unit.schema_version,
                "study_id": unit.study_id,
                "manifest_sha256": unit.manifest_sha256,
                "data_id": unit.data_id,
                "tier": unit.tier,
                "regime_id": unit.regime_id,
                "replicate": unit.replicate,
            },
            "data",
        ),
        (
            natural_starts,
            {
                "schema_version": unit.schema_version,
                "study_id": unit.study_id,
                "manifest_sha256": unit.manifest_sha256,
                "start_id": unit.start_id,
                "data_id": unit.data_id,
                "tier": unit.tier,
                "regime_id": unit.regime_id,
                "replicate": unit.replicate,
                "natural_start_seed": unit.natural_start_seed,
                "natural_start_chain_seeds": list(unit.natural_start_chain_seeds),
            },
            "natural-start",
        ),
        (
            coordinate_starts,
            {
                "schema_version": unit.schema_version,
                "study_id": unit.study_id,
                "manifest_sha256": unit.manifest_sha256,
                "cell_id": unit.cell_id,
                "start_id": unit.start_id,
                "representation_id": unit.representation_id,
                "backend_id": unit.backend_id,
            },
            "coordinate-start",
        ),
    )
    for artifact, expected, source in identities:
        if any(artifact.get(name) != value for name, value in expected.items()):
            raise CausalContractError(
                f"{source} artifact identity differs from the plan"
            )

    coordinate_size = int(unit.regime["n_groups"]) + 2

    def finite_array(value: Any, shape: tuple[int, ...], source: str) -> Any:
        if value is None or isinstance(value, bool):
            raise CausalContractError(f"{source} is not a finite numeric array")
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise CausalContractError(
                f"{source} is not a finite numeric array"
            ) from error
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise CausalContractError(
                f"{source} is not a finite numeric array with shape {shape}"
            )
        return array

    def natural_vector(value: Mapping[str, Any], source: str) -> Any:
        try:
            location = float(value["group_location"])
            scale = float(value["group_scale"])
            groups = finite_array(
                value["group_effect"],
                (int(unit.regime["n_groups"]),),
                f"{source}.group_effect",
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise CausalContractError(f"{source} natural point is malformed") from error
        if (
            isinstance(value["group_location"], bool)
            or isinstance(value["group_scale"], bool)
            or not math.isfinite(location)
            or not math.isfinite(scale)
            or scale <= 0.0
        ):
            raise CausalContractError(f"{source} natural point is malformed")
        return np.concatenate(([location, scale], groups))

    def require_identity(observed: Any, expected: Any, source: str) -> None:
        if observed.shape != expected.shape or not np.array_equal(observed, expected):
            raise CausalContractError(
                f"{source} differs from its hash-bound source artifact"
            )

    def require_roundtrip_summary(
        summary: Any, restored: Any, source_natural: Any, source: str
    ) -> float:
        if not isinstance(summary, Mapping):
            raise CausalContractError(f"{source}.roundtrip is malformed")
        retained_natural = natural_vector(summary, f"{source}.roundtrip")
        if not np.array_equal(retained_natural, restored):
            raise CausalContractError(
                f"{source} retained roundtrip differs from recomputed natural values"
            )
        recomputed_error = float(np.max(np.abs(restored - source_natural)))
        claimed_error = summary.get("absolute_error_max")
        if (
            isinstance(claimed_error, bool)
            or not isinstance(claimed_error, (int, float))
            or not math.isclose(
                float(claimed_error),
                recomputed_error,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise CausalContractError(
                f"{source}.roundtrip error disagrees with retained coordinates"
            )
        return recomputed_error

    data_spec = data.get("spec")
    if not isinstance(data_spec, Mapping):
        raise CausalContractError("data artifact lacks its natural-model spec")
    expected_data_spec = {
        "lower": unit.regime["lower"],
        "upper": unit.regime["upper"],
        "truth_group_location": unit.regime["truth_group_location"],
        "truth_group_scale": unit.regime["truth_group_scale"],
        "n_groups": unit.regime["n_groups"],
        "n_per_group": unit.regime["n_per_group"],
        "floatx": unit.floatx,
        "observation_sigma": unit.natural_model["observation_sigma"],
    }
    if any(data_spec.get(name) != value for name, value in expected_data_spec.items()):
        raise CausalContractError("data artifact natural-model spec differs from plan")
    group_index = finite_array(
        data.get("group_index"),
        (int(unit.regime["n_groups"]) * int(unit.regime["n_per_group"]),),
        "data.group_index",
    )
    if (
        not np.all(group_index == np.floor(group_index))
        or np.any(group_index < 0)
        or np.any(group_index >= int(unit.regime["n_groups"]))
    ):
        raise CausalContractError("data.group_index contains an invalid group")
    observations = finite_array(
        data.get("observations"), group_index.shape, "data.observations"
    )
    truth = natural_vector(
        {
            "group_location": data_spec.get("truth_group_location"),
            "group_scale": data_spec.get("truth_group_scale"),
            "group_effect": data.get("group_effect"),
        },
        "data truth",
    )
    try:
        oracle_spec = HierarchicalPosteriorSpec(
            bounds=TruncationBounds(data_spec["lower"], data_spec["upper"]),
            location_base_mean=float(unit.regime["prior_hyper_location"]),
            location_prior_scale=float(unit.natural_model["location_prior_sigma"]),
            scale_prior_shape=float(unit.natural_model["scale_prior_alpha"]),
            scale_prior_scale=float(unit.natural_model["scale_prior_beta"]),
            n_groups=int(unit.regime["n_groups"]),
            group_index=group_index.astype(np.int64),
            observations=observations,
            observation_scale=float(unit.natural_model["observation_sigma"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CausalContractError(
            "data artifact cannot define the oracle model"
        ) from error
    parameterizations: Mapping[
        str,
        Literal[
            "centered",
            "location_icdf_noncentered",
            "group_icdf_noncentered",
            "full_icdf_noncentered",
        ],
    ] = {
        "native-centered": "centered",
        "manual-centered": "centered",
        "group-icdf-noncentered": "group_icdf_noncentered",
        "location-icdf-noncentered": "location_icdf_noncentered",
        "full-icdf-noncentered": "full_icdf_noncentered",
    }
    parameterization = parameterizations[unit.representation_id]

    natural_chain_rows = natural_starts.get("chains")
    coordinate_chain_rows = coordinate_starts.get("chains")
    if (
        not isinstance(natural_chain_rows, list)
        or not isinstance(coordinate_chain_rows, list)
        or len(natural_chain_rows) != unit.chains
        or len(coordinate_chain_rows) != unit.chains
    ):
        raise CausalContractError("start artifacts do not contain every planned chain")
    natural_by_chain: dict[int, Any] = {}
    coordinate_by_chain: dict[int, Any] = {}
    for chain in range(unit.chains):
        natural_row = natural_chain_rows[chain]
        coordinate_row = coordinate_chain_rows[chain]
        if not isinstance(natural_row, Mapping) or not isinstance(
            coordinate_row, Mapping
        ):
            raise CausalContractError("start artifact chain row is malformed")
        if (
            natural_row.get("chain") != chain
            or natural_row.get("seed") != unit.natural_start_chain_seeds[chain]
            or coordinate_row.get("chain") != chain
            or coordinate_row.get("natural_start_seed")
            != unit.natural_start_chain_seeds[chain]
        ):
            raise CausalContractError("start artifact chain identity differs from plan")
        natural = natural_vector(natural_row, f"natural start chain {chain}")
        coordinate_natural = coordinate_row.get("natural")
        if not isinstance(coordinate_natural, Mapping):
            raise CausalContractError("coordinate start lacks its natural source")
        require_identity(
            natural_vector(coordinate_natural, f"coordinate start chain {chain}"),
            natural,
            f"coordinate start chain {chain} natural point",
        )
        coordinate = finite_array(
            coordinate_row.get("coordinate_vector"),
            (coordinate_size,),
            f"coordinate start chain {chain}.coordinate_vector",
        )
        try:
            restored = hierarchical_natural_values(
                coordinate, oracle_spec, parameterization
            )
        except (TypeError, ValueError) as error:
            raise CausalContractError(
                "coordinate start cannot be mapped to the natural hierarchy"
            ) from error
        restored_vector = np.asarray(
            [
                restored.location.value,
                restored.scale.value,
                *(group.value for group in restored.group_effect),
            ],
            dtype=np.float64,
        )
        require_roundtrip_summary(
            coordinate_row.get("roundtrip"),
            restored_vector,
            natural,
            f"coordinate start chain {chain}",
        )
        natural_by_chain[chain] = natural
        coordinate_by_chain[chain] = coordinate

    chains = None
    if chain_path is not None:
        try:
            chains = xr.load_dataset(chain_path, engine="scipy")
        except Exception as error:
            raise CausalContractError(
                f"cannot decode chain artifact: {error}"
            ) from error
    try:
        for index, item in enumerate(oracle["records"]):
            source = f"oracle.records[{index}]"
            if not isinstance(item, Mapping):
                raise CausalContractError(f"{source} must be an object")
            coordinate = finite_array(
                item.get("coordinate_vector"),
                (coordinate_size,),
                f"{source}.coordinate_vector",
            )
            raw_natural = natural_vector(item, source)
            try:
                restored = hierarchical_natural_values(
                    coordinate, oracle_spec, parameterization
                )
            except (TypeError, ValueError) as error:
                raise CausalContractError(
                    f"{source}.coordinate_vector cannot map to natural scale"
                ) from error
            restored_natural = np.asarray(
                [
                    restored.location.value,
                    restored.scale.value,
                    *(group.value for group in restored.group_effect),
                ],
                dtype=np.float64,
            )
            roundtrip = item.get("roundtrip")
            require_roundtrip_summary(
                roundtrip,
                restored_natural,
                raw_natural,
                source,
            )

            kind = item.get("kind")
            if kind == "fixed-grid":
                require_identity(raw_natural, truth, f"{source} fixed truth")
            elif kind == "shared-natural-start":
                point_id = item.get("point_id")
                try:
                    chain = int(str(point_id).removeprefix("start-chain-"))
                    expected_natural = natural_by_chain[chain]
                    expected_coordinate = coordinate_by_chain[chain]
                except (KeyError, TypeError, ValueError) as error:
                    raise CausalContractError(
                        f"{source} start identity is malformed"
                    ) from error
                require_identity(
                    raw_natural, expected_natural, f"{source} natural start"
                )
                require_identity(
                    coordinate,
                    expected_coordinate,
                    f"{source} coordinate start",
                )
            elif isinstance(kind, str) and kind.startswith("icdf-"):
                coordinate_index = item.get("coordinate_index")
                if not _is_int(coordinate_index):
                    raise CausalContractError(f"{source} ICDF index is malformed")
                expected_coordinate = np.array(
                    coordinate_by_chain[0], dtype=np.dtype(unit.floatx), copy=True
                )
                label = kind.removeprefix("icdf-")
                try:
                    expected_coordinate[coordinate_index] = {
                        "branch-left": -float(item["branch_epsilon"]),
                        "branch-zero": 0.0,
                        "branch-right": float(item["branch_epsilon"]),
                        "tail-low": -6.0,
                        "tail-high": 6.0,
                    }[label]
                except (KeyError, TypeError, ValueError, IndexError) as error:
                    raise CausalContractError(
                        f"{source} ICDF construction metadata is malformed"
                    ) from error
                require_identity(
                    coordinate,
                    expected_coordinate.astype(np.float64),
                    f"{source} ICDF construction",
                )
            elif kind == "hash-selected-posterior-trajectory":
                trajectory_chain = item.get("chain")
                trajectory_draw = item.get("draw")
                if (
                    not isinstance(trajectory_chain, int)
                    or isinstance(trajectory_chain, bool)
                    or not isinstance(trajectory_draw, int)
                    or isinstance(trajectory_draw, bool)
                    or chains is None
                    or chains.sizes.get("chain") != unit.chains
                    or chains.sizes.get("draw") != unit.draws
                ):
                    raise CausalContractError(
                        f"{source} lacks its planned natural chain draw"
                    )
                try:
                    expected_natural = np.concatenate(
                        (
                            [
                                float(
                                    chains["group_location"][
                                        trajectory_chain, trajectory_draw
                                    ]
                                ),
                                float(
                                    chains["group_scale"][
                                        trajectory_chain, trajectory_draw
                                    ]
                                ),
                            ],
                            np.asarray(
                                chains["group_effect"][
                                    trajectory_chain, trajectory_draw
                                ],
                                dtype=np.float64,
                            ),
                        )
                    )
                except (KeyError, IndexError, TypeError, ValueError) as error:
                    raise CausalContractError(
                        f"{source} cannot be read from the natural chain"
                    ) from error
                if not np.all(np.isfinite(expected_natural)):
                    raise CausalContractError(
                        f"{source} selected natural chain draw is non-finite"
                    )
                require_identity(
                    raw_natural,
                    expected_natural,
                    f"{source} selected natural chain draw",
                )
            else:  # pragma: no cover - point structure audit rejects this first.
                raise CausalContractError(f"{source} has an unknown point kind")
    finally:
        if chains is not None:
            chains.close()


def _audit_chain_evidence(
    record: Mapping[str, Any], unit: UnitSpec, path: Path
) -> None:
    """Recompute sampler health metrics from natural chains and per-draw stats."""
    try:
        import arviz as az
        import numpy as np
        import xarray as xr
    except ImportError as error:  # pragma: no cover - all are frozen dependencies.
        raise CausalContractError(
            "ArviZ, NumPy, and xarray are required to audit chain evidence"
        ) from error
    try:
        chains = xr.load_dataset(path, engine="scipy")
    except Exception as error:
        raise CausalContractError(f"cannot decode chain artifact: {error}") from error
    try:
        expected_attrs = {
            "schema_version": unit.schema_version,
            "study_id": unit.study_id,
            "manifest_sha256": unit.manifest_sha256,
            "cell_id": unit.cell_id,
            "block_id": unit.block_id,
            "tier": unit.tier,
            "regime_id": unit.regime_id,
            "backend_id": unit.backend_id,
            "representation_id": unit.representation_id,
        }
        if any(chains.attrs.get(key) != value for key, value in expected_attrs.items()):
            raise CausalContractError("chain artifact identity differs from the plan")
        if (
            chains.sizes.get("chain") != unit.chains
            or chains.sizes.get("draw") != unit.draws
        ):
            raise CausalContractError("chain artifact dimensions differ from the plan")
        natural_shapes = {
            "group_location": (unit.chains, unit.draws),
            "group_scale": (unit.chains, unit.draws),
            "group_effect": (
                unit.chains,
                unit.draws,
                int(unit.regime["n_groups"]),
            ),
        }
        for name, shape in natural_shapes.items():
            if name not in chains or chains[name].shape != shape:
                raise CausalContractError(
                    f"chain artifact natural variable {name!r} has wrong shape"
                )
            if not np.all(np.isfinite(np.asarray(chains[name]))):
                raise CausalContractError(
                    f"chain artifact natural variable {name!r} is non-finite"
                )
        required_stats = {
            "acceptance_rate",
            "diverging",
            "energy",
            "n_steps",
            "step_size",
            "tree_depth",
        }

        def statistic(name: str) -> Any:
            variable = f"sample_stat__{name}"
            if variable not in chains or chains[variable].dims != ("chain", "draw"):
                raise CausalContractError(
                    f"chain artifact lacks shaped per-draw statistic {name!r}"
                )
            array = np.asarray(chains[variable])
            if array.shape != (unit.chains, unit.draws) or not np.all(
                np.isfinite(array)
            ):
                raise CausalContractError(
                    f"chain statistic {name!r} is non-finite or has wrong shape"
                )
            return chains[variable]

        for name in required_stats:
            statistic(name)
        try:
            seed_input = strict_json_loads(
                chains.attrs["sampler_seed_input_json"], source="chain seed input"
            )
            chain_rng = strict_json_loads(
                chains.attrs["chain_rng_provenance_json"],
                source="chain RNG provenance",
            )
        except (KeyError, TypeError) as error:
            raise CausalContractError("chain artifact lacks RNG provenance") from error
        if (
            seed_input != record["provenance"]["sampler_seed_input"]
            or chain_rng != record["provenance"]["chain_rng_provenance"]
        ):
            raise CausalContractError("chain RNG provenance differs from result marker")

        def flattened(dataset: Any) -> Any:
            pieces = [
                np.asarray(value).reshape(-1) for value in dataset.data_vars.values()
            ]
            return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float64)

        divergences = np.asarray(statistic("diverging"), dtype=np.int64)
        if "sample_stat__reached_max_treedepth" in chains:
            saturated = np.asarray(statistic("reached_max_treedepth"), dtype=bool)
        else:
            saturated = (
                np.asarray(statistic("tree_depth"), dtype=np.int64)
                >= unit.max_treedepth
            )
        hyper = chains[["group_location", "group_scale"]]
        group = chains[["group_effect"]]
        hyper_rhat = flattened(az.rhat(hyper, method="rank"))
        group_rhat = flattened(az.rhat(group, method="rank"))
        hyper_bulk = flattened(az.ess(hyper, method="bulk"))
        group_bulk = flattened(az.ess(group, method="bulk"))
        hyper_tail = flattened(az.ess(hyper, method="tail"))
        group_tail = flattened(az.ess(group, method="tail"))
        hyper_mcse = flattened(az.mcse(hyper, method="mean"))
        hyper_sd = np.asarray(
            [float(chains[name].std(dim=("chain", "draw"), ddof=1)) for name in hyper]
        )
        mcse_over_sd = np.divide(
            hyper_mcse,
            hyper_sd,
            out=np.full_like(hyper_mcse, np.inf),
            where=hyper_sd > 0,
        )
        bfmi = np.asarray(
            az.bfmi(np.asarray(statistic("energy"), dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        step_size = np.asarray(statistic("step_size"), dtype=np.float64)
        n_steps = np.asarray(statistic("n_steps"), dtype=np.float64)
        draws = unit.chains * unit.draws
        divergence_count = int(np.sum(divergences))
        derived = {
            "divergence_count": divergence_count,
            "posterior_draw_count": draws,
            "divergence_rate": divergence_count / draws,
            "hyper_rhat_max": float(np.max(hyper_rhat)),
            "hyper_ess_bulk_min": float(np.min(hyper_bulk)),
            "hyper_ess_tail_min": float(np.min(hyper_tail)),
            "bfmi_min": float(np.min(bfmi)),
            "treedepth_saturation_rate": float(np.mean(saturated)),
            "hyper_mcse_over_sd_max": float(np.max(mcse_over_sd)),
            "group_rhat_max": float(np.max(group_rhat)),
            "group_ess_bulk_fraction_ge_400": float(np.mean(group_bulk >= 400.0)),
            "group_ess_tail_fraction_ge_400": float(np.mean(group_tail >= 400.0)),
            "step_size_final_min": float(np.min(step_size[:, -1])),
            "step_size_final_max": float(np.max(step_size[:, -1])),
            "leapfrog_step_count": float(np.sum(n_steps)),
        }
        for name, value in derived.items():
            if name in record["metrics"]:
                _require_derived_metric(record, name, value)
    finally:
        chains.close()


def verify_result_artifacts(
    record: Mapping[str, Any],
    root: Path,
    unit: UnitSpec | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Verify bytes and, with a unit, recompute oracle/sampler evidence."""
    resolved_root = root.resolve()
    resolved_paths: dict[str, Path] = {}
    for name, reference in record["artifacts"].items():
        if reference is None:
            continue
        relative = reference["path"]
        _relative_contract_path(relative)
        path = (resolved_root / relative).resolve()
        if resolved_root not in path.parents:
            raise CausalContractError(f"artifacts.{name} escapes the artifact root")
        if sha256_file(path) != reference["sha256"]:
            raise CausalContractError(f"artifacts.{name} exact-byte hash mismatch")
        if path.stat().st_size != reference["size_bytes"]:
            raise CausalContractError(f"artifacts.{name} byte count mismatch")
        resolved_paths[name] = path
    if unit is None:
        return
    diagnostics = record["artifacts"]["diagnostics"]
    if diagnostics is not None:
        effective_manifest = load_manifest() if manifest is None else manifest
        validate_manifest(effective_manifest)
        if manifest_digest(effective_manifest) != unit.manifest_sha256:
            raise CausalContractError("artifact audit manifest differs from the plan")
        _audit_diagnostic_evidence(
            record,
            unit,
            resolved_root / diagnostics["path"],
            effective_manifest["analysis_policy"],
        )
        if ORACLE_METRICS <= set(record["metrics"]):
            missing_sources = {
                name
                for name in ("data", "natural_start", "coordinate_start")
                if name not in resolved_paths
            }
            if missing_sources:
                raise CausalContractError(
                    f"oracle evidence lacks source artifacts: {sorted(missing_sources)}"
                )
            _audit_oracle_point_bindings(
                record,
                unit,
                resolved_paths["diagnostics"],
                resolved_paths["data"],
                resolved_paths["natural_start"],
                resolved_paths["coordinate_start"],
                resolved_paths.get("chain"),
                effective_manifest["analysis_policy"],
            )
    chain = record["artifacts"]["chain"]
    if chain is not None:
        _audit_chain_evidence(record, unit, resolved_paths["chain"])


def load_result_records(directory: Path) -> dict[str, Mapping[str, Any]]:
    """Load final cell markers; filenames must equal their cell identity."""
    if not directory.is_dir():
        return {}
    records: dict[str, Mapping[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        record = _load_json(path)
        if not isinstance(record, Mapping):
            raise CausalContractError(f"{path} must contain an object")
        cell_id = record.get("cell_id")
        if not isinstance(cell_id, str) or path.name != f"{cell_id}.json":
            raise CausalContractError(f"result filename does not match cell_id: {path}")
        if cell_id in records:
            raise CausalContractError(f"duplicate result for {cell_id}")
        records[cell_id] = record
    return records


def aggregate_results(
    plan: Sequence[UnitSpec],
    records: Mapping[str, Mapping[str, Any]],
    *,
    context_directory: Path | None = None,
    artifact_root: Path | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Order results by plan and materialize missing evidence explicitly."""
    expected = {unit.cell_id for unit in plan}
    extras = sorted(set(records) - expected)
    if extras:
        raise CausalContractError(f"unplanned result records: {extras}")
    rows: list[dict[str, Any]] = []
    for unit in plan:
        record = records.get(unit.cell_id)
        if record is None:
            rows.append(
                {
                    "collection_status": "missing",
                    "cell_id": unit.cell_id,
                    "pair_id": unit.pair_id,
                    "pair_position": unit.pair_position,
                    "block_id": unit.block_id,
                    "tier": unit.tier,
                    "regime_id": unit.regime_id,
                    "backend_id": unit.backend_id,
                    "representation_id": unit.representation_id,
                    "replicate": unit.replicate,
                }
            )
            continue
        context = None
        if context_directory is not None:
            context = load_run_context(context_directory / f"{unit.pair_id}.json")
            validate_run_context(
                context,
                [candidate for candidate in plan if candidate.pair_id == unit.pair_id],
                manifest,
            )
        validate_result_record(record, unit, context)
        if artifact_root is not None:
            verify_result_artifacts(record, artifact_root, unit, manifest)
        rows.append({"collection_status": "present", **dict(record)})
    return rows


def _fit_health(row: Mapping[str, Any], tier: str, policy: Mapping[str, Any]) -> bool:
    if row.get("collection_status") != "present":
        return False
    if row["execution_status"] != "completed":
        return False
    metrics = row["metrics"]
    basic = (
        all(
            metrics[key]
            for key in (
                "compile_success",
                "initialization_success",
                "logp_finite",
                "gradient_finite",
                "sampling_success",
            )
        )
        and metrics["divergence_rate"] < policy["divergence_rate_lt"]
    )
    if not basic or tier == "smoke":
        return basic
    return bool(
        metrics["hyper_rhat_max"] < policy["hyper_rhat_max_lt"]
        and metrics["hyper_ess_bulk_min"] >= policy["hyper_ess_bulk_min_ge"]
        and metrics["hyper_ess_tail_min"] >= policy["hyper_ess_tail_min_ge"]
        and metrics["bfmi_min"] >= policy["bfmi_min_ge"]
        and metrics["treedepth_saturation_rate"]
        < policy["treedepth_saturation_rate_lt"]
        and metrics["hyper_mcse_over_sd_max"] <= policy["hyper_mcse_over_sd_max_le"]
        and metrics["group_rhat_max"] < policy["group_rhat_max_lt"]
        and metrics["group_ess_bulk_fraction_ge_400"]
        >= policy["group_ess_bulk_fraction_ge_400_ge"]
        and metrics["group_ess_tail_fraction_ge_400"]
        >= policy["group_ess_tail_fraction_ge_400_ge"]
    )


def _expected_oracle_evaluation_count(
    row: Mapping[str, Any], tier: str, *, posterior_trajectory: bool
) -> int:
    """Return the frozen number of oracle points for one cell and phase."""
    chains = 2 if tier == "smoke" else 4
    layers = {
        "native-centered": 0,
        "manual-centered": 0,
        "group-icdf-noncentered": 1,
        "location-icdf-noncentered": 1,
        "full-icdf-noncentered": 2,
    }[row["representation_id"]]
    static = (
        chains + int(row["replicate"] == 0) + ICDF_DIAGNOSTIC_POINTS_PER_LAYER * layers
    )
    if posterior_trajectory:
        return static + TRAJECTORY_POINTS_PER_CHAIN * chains
    return static


def _oracle_metrics_health(
    row: Mapping[str, Any],
    floatx: str,
    tier: str,
    *,
    require_posterior_trajectory: bool,
) -> bool:
    if row.get("collection_status") != "present":
        return False
    metrics = row.get("metrics", {})
    if not ORACLE_METRICS <= set(metrics):
        return False
    completed = row.get("execution_status") == "completed"
    if require_posterior_trajectory and not completed:
        return False
    failure = row.get("failure")
    failure_stage = failure.get("stage") if isinstance(failure, Mapping) else None
    posterior_trajectory = completed or failure_stage == "summarize"
    expected_count = _expected_oracle_evaluation_count(
        row,
        tier,
        posterior_trajectory=posterior_trajectory,
    )
    tolerance = 1e-10 if floatx == "float64" else 2e-5
    return bool(
        metrics["oracle_evaluation_count"] == expected_count
        and metrics["oracle_logp_scaled_error_max"] <= 1
        and metrics["oracle_gradient_scaled_error_max"] <= 1
        and metrics["oracle_hessian_scaled_error_max"] <= 1
        and metrics["roundtrip_absolute_error_max"] <= tolerance
        and metrics["icdf_tail_finite"]
        and metrics["icdf_branch_continuous"]
    )


def _pre_sampling_oracle_health(row: Mapping[str, Any], floatx: str, tier: str) -> bool:
    """Check fixed/start/ICDF evidence, whether or not sampling later succeeded."""
    return _oracle_metrics_health(
        row,
        floatx,
        tier,
        require_posterior_trajectory=False,
    )


def _full_oracle_health(row: Mapping[str, Any], floatx: str, tier: str) -> bool:
    """Check the complete oracle, including posterior-trajectory points."""
    return _oracle_metrics_health(
        row,
        floatx,
        tier,
        require_posterior_trajectory=True,
    )


def _native_density_derivative_mismatch(row: Mapping[str, Any], tier: str) -> bool:
    """Identify an explicit native value/gradient/Hessian oracle disagreement."""
    if row.get("collection_status") != "present":
        return False
    metrics = row.get("metrics", {})
    if not ORACLE_METRICS <= set(metrics):
        return False
    if row.get("execution_status") not in {"completed", "failed"}:
        return False
    completed = row.get("execution_status") == "completed"
    failure = row.get("failure")
    failure_stage = failure.get("stage") if isinstance(failure, Mapping) else None
    expected_count = _expected_oracle_evaluation_count(
        row,
        tier,
        posterior_trajectory=completed or failure_stage == "summarize",
    )
    return bool(
        metrics["oracle_evaluation_count"] == expected_count
        and any(
            metrics[metric] > 1.0
            for metric in (
                "oracle_logp_scaled_error_max",
                "oracle_gradient_scaled_error_max",
                "oracle_hessian_scaled_error_max",
            )
        )
    )


def _family_health(
    rows: Sequence[Mapping[str, Any]],
    tier: str,
    policy: Mapping[str, Any],
    per_fit_policy: Mapping[str, Any],
) -> bool:
    if not rows or any(row.get("collection_status") != "present" for row in rows):
        return False
    completed = [row for row in rows if row["execution_status"] == "completed"]
    if len(completed) != len(rows):
        return False
    fit_passes = sum(_fit_health(row, tier, per_fit_policy) for row in rows)
    pass_fraction = fit_passes / len(rows)
    divergences = sum(row["metrics"]["divergence_count"] for row in rows)
    draws = sum(row["metrics"]["posterior_draw_count"] for row in rows)
    return bool(
        pass_fraction >= policy["per_fit_pass_fraction_ge"]
        and len(rows) - fit_passes <= policy["maximum_failed_replicates"]
        and draws > 0
        and divergences / draws < policy["aggregate_divergence_rate_lt"]
    )


def _exact_two_sided_sign_p(left: int, right: int) -> float:
    """Return the exact two-sided sign-test p-value for discordant pairs."""
    discordant = left + right
    if discordant == 0:
        return 1.0
    tail = min(left, right)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def _paired_health_contrast(
    rows: Sequence[Mapping[str, Any]],
    tier: str,
    left_representation: str,
    right_representation: str,
    per_fit_policy: Mapping[str, Any],
    alpha: float,
) -> dict[str, Any]:
    """Compare paired fit health, where left-unhealthy/right-healthy is directional."""
    left = {
        row["replicate"]: _fit_health(row, tier, per_fit_policy)
        for row in rows
        if row["representation_id"] == left_representation
    }
    right = {
        row["replicate"]: _fit_health(row, tier, per_fit_policy)
        for row in rows
        if row["representation_id"] == right_representation
    }
    complete = bool(left) and set(left) == set(right)
    left_bad_right_good = 0
    left_good_right_bad = 0
    ties = 0
    if complete:
        for replicate in sorted(left):
            if not left[replicate] and right[replicate]:
                left_bad_right_good += 1
            elif left[replicate] and not right[replicate]:
                left_good_right_bad += 1
            else:
                ties += 1
    p_value = _exact_two_sided_sign_p(left_bad_right_good, left_good_right_bad)
    return {
        "left": left_representation,
        "right": right_representation,
        "complete": complete,
        "pairs": len(left) if complete else 0,
        "left_unhealthy_right_healthy": left_bad_right_good,
        "left_healthy_right_unhealthy": left_good_right_bad,
        "ties": ties,
        "two_sided_exact_p": p_value,
        "supports_direction": bool(
            complete
            and len(left) == 8
            and left_bad_right_good > left_good_right_bad
            and p_value <= alpha
        ),
    }


def _paired_backend_health_contrast(
    rows: Sequence[Mapping[str, Any]],
    tier: str,
    per_fit_policy: Mapping[str, Any],
    alpha: float,
) -> dict[str, Any]:
    """Describe one representation's backend contrast; never classify from it."""
    pymc = {
        row["replicate"]: _fit_health(row, tier, per_fit_policy)
        for row in rows
        if row["backend_id"] == "pymc"
    }
    numpyro = {
        row["replicate"]: _fit_health(row, tier, per_fit_policy)
        for row in rows
        if row["backend_id"] == "numpyro"
    }
    complete = bool(pymc) and set(pymc) == set(numpyro)
    pymc_bad_numpyro_good = 0
    pymc_good_numpyro_bad = 0
    ties = 0
    if complete:
        for replicate in sorted(pymc):
            if not pymc[replicate] and numpyro[replicate]:
                pymc_bad_numpyro_good += 1
            elif pymc[replicate] and not numpyro[replicate]:
                pymc_good_numpyro_bad += 1
            else:
                ties += 1
    p_value = _exact_two_sided_sign_p(pymc_bad_numpyro_good, pymc_good_numpyro_bad)
    return {
        "complete": complete,
        "pairs": len(pymc) if complete else 0,
        "pymc_unhealthy_numpyro_healthy": pymc_bad_numpyro_good,
        "pymc_healthy_numpyro_unhealthy": pymc_good_numpyro_bad,
        "ties": ties,
        "two_sided_exact_p": p_value,
        "all_directional_at_threshold": bool(
            complete
            and len(pymc) == 8
            and pymc_bad_numpyro_good != pymc_good_numpyro_bad
            and p_value <= alpha
        ),
        "descriptive_only": True,
    }


def _backend_omnibus_health_contrast(
    rows: Sequence[Mapping[str, Any]],
    tier: str,
    per_fit_policy: Mapping[str, Any],
    alpha: float,
) -> dict[str, Any]:
    """Compare the number of healthy forms in each paired backend block."""
    counts: dict[str, dict[int, int]] = {}
    complete = True
    for backend in BACKEND_IDS:
        counts[backend] = {}
        backend_rows = [row for row in rows if row["backend_id"] == backend]
        for replicate in range(8):
            selected = [row for row in backend_rows if row["replicate"] == replicate]
            identities = {row["representation_id"] for row in selected}
            if len(selected) != 5 or identities != set(REPRESENTATION_IDS):
                complete = False
                continue
            counts[backend][replicate] = sum(
                _fit_health(row, tier, per_fit_policy) for row in selected
            )
    complete = complete and all(
        set(counts[backend]) == set(range(8)) for backend in BACKEND_IDS
    )
    pymc_less_healthy = 0
    pymc_more_healthy = 0
    ties = 0
    differences: list[int] = []
    if complete:
        for replicate in range(8):
            difference = counts["pymc"][replicate] - counts["numpyro"][replicate]
            differences.append(difference)
            if difference < 0:
                pymc_less_healthy += 1
            elif difference > 0:
                pymc_more_healthy += 1
            else:
                ties += 1
    p_value = _exact_two_sided_sign_p(pymc_less_healthy, pymc_more_healthy)
    supports = bool(
        complete and pymc_less_healthy != pymc_more_healthy and p_value <= alpha
    )
    direction = None
    if supports:
        direction = "pymc-less-healthy" if pymc_less_healthy else "pymc-more-healthy"
    return {
        "complete": complete,
        "pairs": 8 if complete else 0,
        "statistic": "pymc-healthy-form-count-minus-numpyro-healthy-form-count",
        "differences": differences,
        "pymc_less_healthy": pymc_less_healthy,
        "pymc_more_healthy": pymc_more_healthy,
        "ties": ties,
        "two_sided_exact_p": p_value,
        "supports_either_direction": supports,
        "direction": direction,
    }


def _every_paired_fit_matches(
    rows: Sequence[Mapping[str, Any]],
    tier: str,
    expected_health: Mapping[str, bool],
    per_fit_policy: Mapping[str, Any],
) -> bool:
    """Require one exact health pattern for every replicate on both backends."""
    for backend in BACKEND_IDS:
        backend_rows = [row for row in rows if row["backend_id"] == backend]
        replicate_ids = {row["replicate"] for row in backend_rows}
        if not replicate_ids:
            return False
        for replicate in replicate_ids:
            replicate_rows = {
                row["representation_id"]: row
                for row in backend_rows
                if row["replicate"] == replicate
            }
            if not set(expected_health) <= set(replicate_rows):
                return False
            if any(
                _fit_health(replicate_rows[representation], tier, per_fit_policy)
                is not expected
                for representation, expected in expected_health.items()
            ):
                return False
    return True


def _family_pattern_matches(
    family: Mapping[str, bool], expected_health: Mapping[str, bool]
) -> bool:
    """Require exact aggregate health states for named forms on both backends."""
    return all(
        family[f"{backend}/{representation}"] is expected
        for backend in BACKEND_IDS
        for representation, expected in expected_health.items()
    )


def _backend_family_direction_matches(
    family: Mapping[str, bool], omnibus: Mapping[str, Any]
) -> bool:
    """Require the aggregate family counts to corroborate the omnibus direction."""
    counts = {
        backend: sum(
            family[f"{backend}/{representation}"]
            for representation in REPRESENTATION_IDS
        )
        for backend in BACKEND_IDS
    }
    if omnibus["direction"] == "pymc-less-healthy":
        return counts["pymc"] < counts["numpyro"]
    if omnibus["direction"] == "pymc-more-healthy":
        return counts["pymc"] > counts["numpyro"]
    return False


def _classify_regime(
    rows: Sequence[Mapping[str, Any]],
    regime: Mapping[str, Any],
    tier: str,
    analysis_policy: Mapping[str, Any],
) -> dict[str, Any]:
    regime_rows = [row for row in rows if row["regime_id"] == regime["regime_id"]]
    if any(row.get("collection_status") != "present" for row in regime_rows):
        return {
            "regime_id": regime["regime_id"],
            "classification": "incomplete",
            "family_health": {},
        }
    family: dict[str, bool] = {}
    oracle: dict[str, bool] = {}
    pre_oracle: dict[str, bool] = {}
    paired: dict[str, dict[str, Any]] = {}
    family_policy = analysis_policy["family_health"]
    per_fit_policy = analysis_policy["per_fit_health"]
    alpha = analysis_policy["paired_inference"]["per_comparison_alpha"]
    for backend in BACKEND_IDS:
        backend_rows = [row for row in regime_rows if row["backend_id"] == backend]
        for representation in REPRESENTATION_IDS:
            selected = [
                row
                for row in regime_rows
                if row["backend_id"] == backend
                and row["representation_id"] == representation
            ]
            key = f"{backend}/{representation}"
            family[key] = _family_health(selected, tier, family_policy, per_fit_policy)
            oracle[key] = all(
                _full_oracle_health(row, regime["floatx"], tier) for row in selected
            )
            pre_oracle[key] = all(
                _pre_sampling_oracle_health(row, regime["floatx"], tier)
                for row in selected
            )
        if tier == "smoke":
            continue
        contrast_pairs = {
            "native-to-manual": ("native-centered", "manual-centered"),
            "group-effect-at-location-c": (
                "manual-centered",
                "group-icdf-noncentered",
            ),
            "group-effect-at-location-nc": (
                "location-icdf-noncentered",
                "full-icdf-noncentered",
            ),
            "location-effect-at-groups-c": (
                "manual-centered",
                "location-icdf-noncentered",
            ),
            "location-effect-at-groups-nc": (
                "group-icdf-noncentered",
                "full-icdf-noncentered",
            ),
        }
        for contrast_id, (left, right) in contrast_pairs.items():
            paired[f"{backend}/{contrast_id}"] = _paired_health_contrast(
                backend_rows,
                tier,
                left,
                right,
                per_fit_policy,
                alpha,
            )

    if tier == "smoke":
        return {
            "regime_id": regime["regime_id"],
            "classification": "screening-only",
            "causal_classification": None,
            "family_health": family,
            "oracle_health": oracle,
            "pre_sampling_oracle_health": pre_oracle,
            "paired_health_contrasts": {},
        }

    for representation in REPRESENTATION_IDS:
        representation_rows = [
            row for row in regime_rows if row["representation_id"] == representation
        ]
        paired[f"backend/{representation}/pymc-to-numpyro"] = (
            _paired_backend_health_contrast(
                representation_rows,
                tier,
                per_fit_policy,
                alpha,
            )
        )
    backend_omnibus = _backend_omnibus_health_contrast(
        regime_rows,
        tier,
        per_fit_policy,
        alpha,
    )
    paired["backend/five-form-health-count-omnibus"] = backend_omnibus

    manual_rows = {
        (row["replicate"], row["backend_id"]): row
        for row in regime_rows
        if row["representation_id"] == "manual-centered"
    }
    native_mismatch_replicates = sorted(
        {
            row["replicate"]
            for row in regime_rows
            if row["representation_id"] == "native-centered"
            and _native_density_derivative_mismatch(row, tier)
            and (manual := manual_rows.get((row["replicate"], row["backend_id"])))
            is not None
            and _pre_sampling_oracle_health(manual, regime["floatx"], tier)
        }
    )
    all_oracle_ok = all(oracle.values())
    classification = "mixed-inconclusive"
    if len(native_mismatch_replicates) >= 2:
        classification = "native-pymc-correctness-defect"
    elif (
        all(
            paired[f"{backend}/native-to-manual"]["supports_direction"]
            for backend in BACKEND_IDS
        )
        and _family_pattern_matches(
            family,
            {"native-centered": False, "manual-centered": True},
        )
        and all_oracle_ok
    ):
        classification = "native-graph-or-adaptation"
    elif (
        all(
            paired[f"{backend}/group-effect-at-location-c"]["supports_direction"]
            and paired[f"{backend}/group-effect-at-location-nc"]["supports_direction"]
            for backend in BACKEND_IDS
        )
        and _family_pattern_matches(
            family,
            {
                "manual-centered": False,
                "group-icdf-noncentered": True,
                "location-icdf-noncentered": False,
                "full-icdf-noncentered": True,
            },
        )
        and all_oracle_ok
    ):
        classification = "group-conditional-centering"
    elif (
        all(
            paired[f"{backend}/location-effect-at-groups-c"]["supports_direction"]
            and paired[f"{backend}/location-effect-at-groups-nc"]["supports_direction"]
            for backend in BACKEND_IDS
        )
        and _family_pattern_matches(
            family,
            {
                "manual-centered": False,
                "group-icdf-noncentered": False,
                "location-icdf-noncentered": True,
                "full-icdf-noncentered": True,
            },
        )
        and all_oracle_ok
    ):
        classification = "location-centering"
    elif (
        _every_paired_fit_matches(
            regime_rows,
            tier,
            {
                "manual-centered": False,
                "group-icdf-noncentered": False,
                "location-icdf-noncentered": False,
                "full-icdf-noncentered": True,
            },
            per_fit_policy,
        )
        and _family_pattern_matches(
            family,
            {
                "manual-centered": False,
                "group-icdf-noncentered": False,
                "location-icdf-noncentered": False,
                "full-icdf-noncentered": True,
            },
        )
        and all_oracle_ok
    ):
        classification = "joint-centering-interaction"
    elif (
        backend_omnibus["supports_either_direction"]
        and _backend_family_direction_matches(family, backend_omnibus)
        and all_oracle_ok
    ):
        classification = "backend-path-specific"
    elif (
        _every_paired_fit_matches(
            regime_rows,
            tier,
            {representation: False for representation in REPRESENTATION_IDS},
            per_fit_policy,
        )
        and not any(family.values())
        and all_oracle_ok
    ):
        classification = "residual-tn-or-scale-geometry"
    elif all(family.values()) and all_oracle_ok:
        classification = "all-representations-healthy"
    return {
        "regime_id": regime["regime_id"],
        "classification": classification,
        "family_health": family,
        "oracle_health": oracle,
        "pre_sampling_oracle_health": pre_oracle,
        "native_mismatch_replicates": native_mismatch_replicates,
        "paired_health_contrasts": paired,
    }


def assess_results(
    rows: Sequence[Mapping[str, Any]],
    plan: Sequence[UnitSpec],
    manifest: Mapping[str, Any],
    tier: str,
) -> dict[str, Any]:
    """Apply the frozen health gates and classifier without sampling."""
    validate_manifest(manifest)
    if len(rows) != len(plan):
        raise CausalContractError("assessment rows do not match plan length")
    for row, unit in zip(rows, plan, strict=True):
        if row.get("cell_id") != unit.cell_id:
            raise CausalContractError("assessment rows are not in canonical plan order")
        if row.get("collection_status") == "present":
            validate_result_record(
                {
                    key: value
                    for key, value in row.items()
                    if key != "collection_status"
                },
                unit,
            )
        elif row.get("collection_status") != "missing":
            raise CausalContractError("collection_status must be present or missing")
    missing = sum(row.get("collection_status") == "missing" for row in rows)
    failed = sum(
        row.get("collection_status") == "present"
        and row.get("execution_status") == "failed"
        for row in rows
    )
    regime_results = [
        _classify_regime(
            rows,
            regime,
            tier,
            manifest["analysis_policy"],
        )
        for regime in manifest["regimes"]
    ]
    if missing:
        outcome = "incomplete"
    elif tier == "smoke":
        outcome = (
            "screening-pass"
            if all(
                _fit_health(
                    row,
                    tier,
                    manifest["analysis_policy"]["per_fit_health"],
                )
                and _full_oracle_health(
                    row,
                    next(
                        regime["floatx"]
                        for regime in manifest["regimes"]
                        if regime["regime_id"] == row["regime_id"]
                    ),
                    tier,
                )
                for row in rows
            )
            else "screening-fail"
        )
    elif any(
        item["classification"] in {"incomplete", "mixed-inconclusive"}
        for item in regime_results
    ):
        outcome = "inconclusive"
    elif any(
        item["classification"] == "native-pymc-correctness-defect"
        for item in regime_results
    ):
        outcome = "correctness-defect"
    else:
        outcome = "classified"
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": manifest["study_id"],
        "manifest_sha256": manifest_digest(manifest),
        "tier": tier,
        "planned_cells": len(plan),
        "present_cells": len(plan) - missing,
        "missing_cells": missing,
        "failed_cells": failed,
        "outcome": outcome,
        "contract_valid": True,
        "evidence_complete": missing == 0,
        "proceed_to_confirmation": bool(tier == "smoke" and missing == 0),
        "qualifies_causal_conclusion": bool(
            tier == "confirmation" and outcome in {"classified", "correctness-defect"}
        ),
        "regimes": regime_results,
    }


def _write_plan(plan: Sequence[UnitSpec], output_dir: Path) -> tuple[Path, Path]:
    jsonl_path = output_dir / "plan.jsonl"
    csv_path = output_dir / "matrix.csv"
    jsonl = b"".join(canonical_json_bytes(unit.as_dict()) for unit in plan)
    _atomic_write(jsonl_path, jsonl)
    fields = (
        "pair_id",
        "pair_position",
        "block_id",
        "block_position",
        "cell_id",
        "tier",
        "regime_id",
        "backend_id",
        "representation_id",
        "replicate",
        "chains",
        "tune",
        "draws",
        "floatx",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for unit in plan:
        value = unit.as_dict()
        writer.writerow({field: value[field] for field in fields})
    _atomic_write(csv_path, buffer.getvalue().encode())
    return jsonl_path, csv_path


def _write_aggregate(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    _atomic_write(path, b"".join(canonical_json_bytes(row) for row in rows))
    return path


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise CausalContractError(f"cannot read {path}: {error}") from error
    records: list[Mapping[str, Any]] = []
    for number, line in enumerate(lines, 1):
        value = strict_json_loads(line, source=f"{path}:{number}")
        if not isinstance(value, Mapping):
            raise CausalContractError(f"{path}:{number} must contain an object")
        records.append(value)
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate the frozen v3 and v2 byte anchor")
    plan = commands.add_parser("plan", help="write deterministic JSONL/CSV plans")
    plan.add_argument("--tier", choices=ALLOWED_TIERS, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    matrix = commands.add_parser(
        "matrix", help="print the GitHub matrix of backend-paired workers"
    )
    matrix.add_argument("--tier", choices=ALLOWED_TIERS, required=True)
    environment = commands.add_parser(
        "environment", help="collect an environment attestation"
    )
    environment.add_argument("--output", type=Path)
    aggregate = commands.add_parser("aggregate", help="aggregate final cell markers")
    aggregate.add_argument("--tier", choices=ALLOWED_TIERS, required=True)
    aggregate.add_argument("--run-dir", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    assess = commands.add_parser(
        "assess", help="apply health gates and causal classifier"
    )
    assess.add_argument("--tier", choices=ALLOWED_TIERS, required=True)
    assess.add_argument("--results", type=Path, required=True)
    assess.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the no-sampling v3 contract CLI."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        digest = manifest_digest(manifest)
        if args.command == "validate":
            summary = {
                "study_id": manifest["study_id"],
                "manifest_sha256": digest,
                "status": manifest["status"],
                "smoke_fits": manifest["tiers"]["smoke"]["expected_fit_count"],
                "confirmation_fits": manifest["tiers"]["confirmation"][
                    "expected_fit_count"
                ],
            }
            print(canonical_json_bytes(summary).decode(), end="")
            return 0
        if args.command == "environment":
            record = collect_environment(manifest)
            payload = canonical_json_bytes(record)
            if args.output is None:
                print(payload.decode(), end="")
            else:
                _atomic_write(args.output, payload)
                print(args.output)
            return 0
        plan = build_plan(manifest, args.tier)
        if args.command == "plan":
            paths = _write_plan(plan, args.output_dir)
            print("\n".join(str(path) for path in paths))
            return 0
        if args.command == "matrix":
            pairs = []
            for unit in plan:
                if unit.pair_position == 0:
                    pairs.append(
                        {
                            "pair_id": unit.pair_id,
                            "tier": unit.tier,
                            "regime_id": unit.regime_id,
                            "replicate": unit.replicate,
                        }
                    )
            print(canonical_json_bytes({"include": pairs}).decode(), end="")
            return 0
        if args.command == "aggregate":
            records = load_result_records(args.run_dir / "cells")
            rows = aggregate_results(
                plan,
                records,
                context_directory=args.run_dir / "contexts",
                artifact_root=args.run_dir,
                manifest=manifest,
            )
            _write_aggregate(rows, args.output)
            print(args.output)
            return 0
        assessment_rows = _load_jsonl(args.results)
        assessment = assess_results(assessment_rows, plan, manifest, args.tier)
        payload = (
            json.dumps(assessment, allow_nan=False, indent=2, sort_keys=True).encode()
            + b"\n"
        )
        if args.output is None:
            print(payload.decode(), end="")
        else:
            _atomic_write(args.output, payload)
            print(args.output)
        return 0 if assessment["evidence_complete"] else 1
    except CausalContractError as error:
        parser.exit(2, f"causal contract error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

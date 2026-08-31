"""Deterministic harness for the hierarchical TruncatedNormal qualification.

This module deliberately contains no model construction or sampling. It freezes the
experiment matrix, derives reproducible seeds, validates result identities, and
applies the predeclared decision rule. Sampling is added in later commits without
changing this contract.
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
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path, PurePosixPath
from statistics import median
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from scripts.truncated_hierarchy_statistics import (
        ParameterSummary,
        QualificationStatisticsError,
        evaluate_bias_family,
        evaluate_coverage_family,
        evaluate_sbc_rank_family,
        paired_backend_mean_check,
        validate_parameter_summary,
    )
elif __package__:
    from scripts.truncated_hierarchy_statistics import (
        ParameterSummary,
        QualificationStatisticsError,
        evaluate_bias_family,
        evaluate_coverage_family,
        evaluate_sbc_rank_family,
        paired_backend_mean_check,
        validate_parameter_summary,
    )
else:  # pragma: no cover - exercised by direct CLI invocation
    from truncated_hierarchy_statistics import (
        ParameterSummary,
        QualificationStatisticsError,
        evaluate_bias_family,
        evaluate_coverage_family,
        evaluate_sbc_rank_family,
        paired_backend_mean_check,
        validate_parameter_summary,
    )

SCHEMA_VERSION = 2
RUNNER_VERSION = 2
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks/specs/truncated_hierarchy_v2.json"
ALLOWED_TIERS = {"smoke", "qualification", "stress"}
ALLOWED_COMPARATORS = {"eq", "lt", "le", "gt", "ge"}
SAFE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SAFE_CELL_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*--replicate-[0-9]{2,}$")
SAFE_PARAMETER = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SBC_SUMMARY_FIELDS = frozenset(
    {"rank_less", "rank_equal", "rank_tie_index", "rank", "rank_draw_count"}
)
DEFAULT_DEPENDENCY_PROFILE = "current-resolved"
ENVIRONMENT_PACKAGES = {
    "hssm",
    "arviz",
    "bambi",
    "formulae",
    "jax",
    "jaxlib",
    "jaxonnxruntime",
    "numpy",
    "numpyro",
    "pymc",
    "pytensor",
    "scipy",
    "ssm-simulators",
}

MANIFEST_KEYS = {
    "schema_version",
    "study_id",
    "status",
    "description",
    "master_seed",
    "seed_derivation",
    "tiers",
    "dependency_profiles",
    "model_contracts",
    "prior_contracts",
    "data_generation",
    "analysis_policy",
    "execution_policy",
    "artifact_policy",
    "thresholds",
    "scenarios",
}
SCENARIO_KEYS = {
    "scenario_id",
    "tier",
    "gate",
    "layer",
    "model",
    "prior",
    "purpose",
    "bound_kind",
    "lower",
    "upper",
    "prior_hyper_location",
    "truth_kind",
    "truth_regime",
    "truth_boundary",
    "truth_group_location",
    "truth_group_scale",
    "n_groups",
    "n_per_group",
    "group_indices",
    "floatx",
    "sampler",
    "replicates",
    "chains",
    "tune",
    "draws",
    "target_accept",
    "canonical",
    "recovery",
    "initialization_policy",
    "control_id",
}
SCENARIO_OPTIONAL_KEYS = {
    "calibration_kind",
    "data_id",
    "dependency_profile",
    "posterior_pair_id",
}
SCENARIO_ALL_KEYS = SCENARIO_KEYS | SCENARIO_OPTIONAL_KEYS
PLAN_KEYS = {
    "schema_version",
    "study_id",
    "manifest_sha256",
    "scenario_sha256",
    "cell_id",
    "scenario_id",
    "replicate",
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
    "scenario",
}
RESULT_KEYS = {
    "schema_version",
    "study_id",
    "manifest_sha256",
    "cell_id",
    "scenario_id",
    "replicate",
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
    "execution_status",
    "metrics",
    "unavailable_metrics",
    "parameter_summaries",
    "failure",
    "provenance",
}
PROVENANCE_KEYS = {
    "runner_version",
    "sampler",
    "device",
    "floatx",
    "pytensor_floatx",
    "jax_enable_x64",
    "data_artifact",
    "data_sha256",
    "effective_numpyro_chain_keys",
    "actual_start_artifact",
    "actual_start_sha256",
    "raw_chain_artifact",
    "raw_chain_sha256",
    "git_commit",
    "environment_sha256",
}
FAILURE_KEYS = {"stage", "error_type", "message"}
ENVIRONMENT_KEYS = {
    "schema_version",
    "study_id",
    "manifest_sha256",
    "runner_version",
    "dependency_profile",
    "git",
    "project",
    "runtime",
    "packages",
}
CONTROL_MATCH_FIELDS = SCENARIO_ALL_KEYS - {
    "scenario_id",
    "prior",
    "prior_hyper_location",
    "purpose",
    "recovery",
    "canonical",
    "control_id",
    "posterior_pair_id",
}
DATA_MATCH_FIELDS = {
    "model",
    "bound_kind",
    "lower",
    "upper",
    "truth_kind",
    "truth_regime",
    "truth_boundary",
    "truth_group_location",
    "truth_group_scale",
    "n_groups",
    "n_per_group",
    "group_indices",
    "calibration_kind",
}
POSTERIOR_PAIR_MATCH_FIELDS = DATA_MATCH_FIELDS | {
    "tier",
    "gate",
    "layer",
    "prior",
    "prior_hyper_location",
    "purpose",
    "canonical",
    "recovery",
    "data_id",
    "dependency_profile",
    "floatx",
    "replicates",
    "chains",
    "tune",
    "draws",
    "target_accept",
    "initialization_policy",
}
METRIC_DOMAINS = {
    "compile_success": "boolean",
    "initialization_success": "boolean",
    "logp_finite": "boolean",
    "gradient_finite": "boolean",
    "finite_difference_gradient_abs_error_max": "nonnegative",
    "finite_difference_gradient_rel_error_max": "nonnegative",
    "finite_difference_gradient_normalized_error_max": "nonnegative",
    "pytensor_jax_gradient_abs_error_max": "nonnegative",
    "pytensor_jax_gradient_rel_error_max": "nonnegative",
    "pytensor_jax_gradient_normalized_error_max": "nonnegative",
    "likelihood_pytensor_jax_value_abs_error_max": "nonnegative",
    "likelihood_pytensor_jax_value_rel_error_max": "nonnegative",
    "likelihood_pytensor_jax_value_normalized_error_max": "nonnegative",
    "likelihood_pytensor_jax_gradient_abs_error_max": "nonnegative",
    "likelihood_pytensor_jax_gradient_rel_error_max": "nonnegative",
    "likelihood_pytensor_jax_gradient_normalized_error_max": "nonnegative",
    "bambi_isomorphism_abs_error_max": "nonnegative",
    "bambi_isomorphism_rel_error_max": "nonnegative",
    "bambi_isomorphism_normalized_error_max": "nonnegative",
    "sampling_success": "boolean",
    "divergence_count": "nonnegative_integer",
    "posterior_draw_count": "positive_integer",
    "divergence_rate": "unit_interval",
    "hyper_rhat_max": "positive",
    "hyper_ess_bulk_min": "nonnegative",
    "hyper_ess_tail_min": "nonnegative",
    "bfmi_min": "nonnegative",
    "treedepth_saturation_rate": "unit_interval",
    "hyper_mcse_over_sd_max": "nonnegative",
    "group_rhat_max": "positive",
    "group_ess_bulk_fraction_ge_400": "unit_interval",
    "group_ess_tail_fraction_ge_400": "unit_interval",
    "sampling_elapsed_seconds": "positive",
    "step_size_median": "positive",
    "gradient_evaluation_count": "positive_integer",
    "leapfrog_step_count": "positive_integer",
    "hyper_ess_per_second_median": "positive",
    "hyper_leapfrog_steps_per_effective_sample_median": "positive",
}
PRESAMPLING_METRICS = {
    "compile_success",
    "initialization_success",
    "logp_finite",
    "gradient_finite",
    "finite_difference_gradient_abs_error_max",
    "finite_difference_gradient_rel_error_max",
    "finite_difference_gradient_normalized_error_max",
    "pytensor_jax_gradient_abs_error_max",
    "pytensor_jax_gradient_rel_error_max",
    "pytensor_jax_gradient_normalized_error_max",
    "likelihood_pytensor_jax_value_abs_error_max",
    "likelihood_pytensor_jax_value_rel_error_max",
    "likelihood_pytensor_jax_value_normalized_error_max",
    "likelihood_pytensor_jax_gradient_abs_error_max",
    "likelihood_pytensor_jax_gradient_rel_error_max",
    "likelihood_pytensor_jax_gradient_normalized_error_max",
    "bambi_isomorphism_abs_error_max",
    "bambi_isomorphism_rel_error_max",
    "bambi_isomorphism_normalized_error_max",
}
SAMPLER_METRICS = METRIC_DOMAINS.keys() - PRESAMPLING_METRICS
REQUIRED_SAMPLER_RAW_METRICS = {
    "sampling_elapsed_seconds",
    "step_size_median",
    "gradient_evaluation_count",
    "leapfrog_step_count",
}
PAIRED_EFFICIENCY_METRICS = {
    "ess_per_second_slowdown": (
        "hyper_ess_per_second_median",
        "control_over_candidate",
    ),
    "leapfrog_cost_ratio": (
        "hyper_leapfrog_steps_per_effective_sample_median",
        "candidate_over_control",
    ),
}
GRADIENT_CONTRACT_METRICS = {
    "finite_difference": {
        "absolute_tolerance": "finite_difference_gradient_abs_error_max",
        "relative_tolerance": "finite_difference_gradient_rel_error_max",
    },
    "pytensor_jax": {
        "absolute_tolerance": "pytensor_jax_gradient_abs_error_max",
        "relative_tolerance": "pytensor_jax_gradient_rel_error_max",
    },
    "bambi_isomorphism": {
        "absolute_tolerance": "bambi_isomorphism_abs_error_max",
        "relative_tolerance": "bambi_isomorphism_rel_error_max",
    },
    "likelihood_pytensor_jax": {
        "value_absolute_tolerance": "likelihood_pytensor_jax_value_abs_error_max",
        "value_relative_tolerance": "likelihood_pytensor_jax_value_rel_error_max",
        "gradient_absolute_tolerance": (
            "likelihood_pytensor_jax_gradient_abs_error_max"
        ),
        "gradient_relative_tolerance": (
            "likelihood_pytensor_jax_gradient_rel_error_max"
        ),
    },
}
GRADIENT_CONTRACT_GATE_METRICS = {
    "finite_difference": ("finite_difference_gradient_normalized_error_max",),
    "pytensor_jax": ("pytensor_jax_gradient_normalized_error_max",),
    "bambi_isomorphism": ("bambi_isomorphism_normalized_error_max",),
    "likelihood_pytensor_jax": (
        "likelihood_pytensor_jax_value_normalized_error_max",
        "likelihood_pytensor_jax_gradient_normalized_error_max",
    ),
}
GRADIENT_CONTRACT_ALL_METRICS = {
    metric
    for mapping in GRADIENT_CONTRACT_METRICS.values()
    for metric in mapping.values()
} | {
    metric for metrics in GRADIENT_CONTRACT_GATE_METRICS.values() for metric in metrics
}
SCENARIO_LEVEL_CONTRACT_REASON = "scenario-level contract evaluated on replicate 0"


class QualificationError(ValueError):
    """Raised when qualification inputs violate the frozen contract."""


def _reject_json_constant(value: str) -> None:
    raise QualificationError(f"non-standard JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_loads(text: str, *, source: str = "JSON input") -> Any:
    """Load strict JSON, rejecting NaN and infinities."""
    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, QualificationError) as error:
        raise QualificationError(f"invalid {source}: {error}") from error
    _assert_finite_json(value, source)
    return value


def _assert_finite_json(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise QualificationError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise QualificationError(f"{path} contains a non-string object key")
            _assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_json(child, f"{path}[{index}]")


def _load_json(path: Path) -> Any:
    try:
        return strict_json_loads(path.read_text(encoding="utf-8"), source=str(path))
    except OSError as error:
        raise QualificationError(f"cannot read {path}: {error}") from error


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise QualificationError(f"value is not strict JSON: {error}") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        raise QualificationError(f"cannot hash {path}: {error}") from error


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Return the semantic SHA-256 digest embedded in every plan and result."""
    return _sha256(manifest)


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError(f"{path} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], path: str
) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise QualificationError(f"{path} has invalid fields: {', '.join(details)}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _validate_threshold(condition: Any, path: str) -> None:
    condition = _require_object(condition, path)
    _require_exact_keys(condition, {"comparator", "value"}, path)
    if condition["comparator"] not in ALLOWED_COMPARATORS:
        raise QualificationError(f"{path}.comparator is unsupported")
    value = condition["value"]
    if not isinstance(value, bool) and not _is_number(value):
        raise QualificationError(f"{path}.value must be boolean or numeric")
    if isinstance(value, bool) and condition["comparator"] != "eq":
        raise QualificationError(f"{path} can only compare booleans with 'eq'")
    _assert_finite_json(value, f"{path}.value")


def _validate_threshold_map(value: Any, path: str) -> None:
    value = _require_object(value, path)
    if not value:
        raise QualificationError(f"{path} must not be empty")
    for metric, condition in value.items():
        if not isinstance(metric, str) or not metric:
            raise QualificationError(f"{path} contains an invalid metric name")
        _validate_threshold(condition, f"{path}.{metric}")


def _validate_thresholds(value: Any) -> None:
    thresholds = _require_object(value, "manifest.thresholds")
    _require_exact_keys(
        thresholds, {"screening", "qualification", "diagnostic"}, "thresholds"
    )
    screening = _require_object(thresholds["screening"], "thresholds.screening")
    _require_exact_keys(screening, {"per_fit"}, "thresholds.screening")
    _validate_threshold_map(screening["per_fit"], "thresholds.screening.per_fit")

    diagnostic = _require_object(thresholds["diagnostic"], "thresholds.diagnostic")
    _require_exact_keys(diagnostic, {"per_fit"}, "thresholds.diagnostic")
    _validate_threshold_map(diagnostic["per_fit"], "thresholds.diagnostic.per_fit")

    qualification = _require_object(
        thresholds["qualification"], "thresholds.qualification"
    )
    expected = {
        "per_fit",
        "canonical_fit",
        "control_paired_fit",
        "primary_aggregate",
        "immediate_no_go",
        "repeated_no_go",
    }
    _require_exact_keys(qualification, expected, "thresholds.qualification")
    for name in expected - {"repeated_no_go"}:
        _validate_threshold_map(qualification[name], f"thresholds.qualification.{name}")
    repeated = _require_object(
        qualification["repeated_no_go"],
        "thresholds.qualification.repeated_no_go",
    )
    _require_exact_keys(
        repeated,
        {"minimum_failing_fits", "conditions"},
        "thresholds.qualification.repeated_no_go",
    )
    if (
        not _is_int(repeated["minimum_failing_fits"])
        or repeated["minimum_failing_fits"] < 2
    ):
        raise QualificationError("minimum_failing_fits must be an integer >= 2")
    _validate_threshold_map(
        repeated["conditions"],
        "thresholds.qualification.repeated_no_go.conditions",
    )


def _validate_tiers(value: Any) -> None:
    tiers = _require_object(value, "manifest.tiers")
    _require_exact_keys(tiers, ALLOWED_TIERS, "manifest.tiers")
    expected_gates = {
        "smoke": ("screening", False),
        "qualification": ("primary", True),
        "stress": ("diagnostic", False),
    }
    for tier, (gate, qualifies) in expected_gates.items():
        item = _require_object(tiers[tier], f"manifest.tiers.{tier}")
        _require_exact_keys(
            item,
            {"gate", "qualifies_default", "description"},
            f"manifest.tiers.{tier}",
        )
        if (item["gate"], item["qualifies_default"]) != (gate, qualifies):
            raise QualificationError(f"manifest.tiers.{tier} changes tier semantics")
        if not isinstance(item["description"], str) or not item["description"]:
            raise QualificationError(f"manifest.tiers.{tier}.description is empty")


def _validate_dependency_profiles(value: Any) -> None:
    profiles = _require_object(value, "manifest.dependency_profiles")
    expected_profiles = {"current-resolved", "bambi-0.19"}
    _require_exact_keys(profiles, expected_profiles, "manifest.dependency_profiles")
    expected_packages = {
        "arviz",
        "bambi",
        "formulae",
        "jax",
        "jaxlib",
        "jaxonnxruntime",
        "numpy",
        "numpyro",
        "pymc",
        "pytensor",
        "scipy",
        "ssm-simulators",
    }
    for profile_name, raw_profile in profiles.items():
        path = f"manifest.dependency_profiles.{profile_name}"
        profile = _require_object(raw_profile, path)
        _require_exact_keys(
            profile,
            {
                "python",
                "project_path",
                "project_sha256",
                "lock_path",
                "lock_sha256",
                "required_versions",
            },
            path,
        )
        if profile["python"] != "3.12":
            raise QualificationError(f"{path}.python must remain '3.12'")
        for artifact in ("project", "lock"):
            artifact_path = profile[f"{artifact}_path"]
            artifact_digest = profile[f"{artifact}_sha256"]
            if not isinstance(artifact_path, str) or not artifact_path:
                raise QualificationError(f"{path}.{artifact}_path must be a path")
            relative_path = PurePosixPath(artifact_path)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or "\\" in artifact_path
                or str(relative_path) != artifact_path
            ):
                raise QualificationError(
                    f"{path}.{artifact}_path must be repository-relative"
                )
            if not isinstance(artifact_digest, str) or not SHA256.fullmatch(
                artifact_digest
            ):
                raise QualificationError(f"{path}.{artifact}_sha256 is invalid")
            resolved = (REPO_ROOT / relative_path).resolve()
            if REPO_ROOT not in resolved.parents or not resolved.is_file():
                raise QualificationError(f"{path}.{artifact}_path is unavailable")
            if _file_sha256(resolved) != artifact_digest:
                raise QualificationError(
                    f"{path}.{artifact}_sha256 does not match its file"
                )
        versions = _require_object(
            profile["required_versions"], f"{path}.required_versions"
        )
        _require_exact_keys(versions, expected_packages, f"{path}.required_versions")
        if any(
            not isinstance(version, str) or not version for version in versions.values()
        ):
            raise QualificationError(
                f"{path} package versions must be non-empty strings"
            )
    if profiles["bambi-0.19"]["required_versions"]["bambi"] != "0.19.0":
        raise QualificationError("bambi-0.19 profile must pin bambi==0.19.0")


def _validate_model_contracts(value: Any) -> None:
    contracts = _require_object(value, "manifest.model_contracts")
    _require_exact_keys(
        contracts,
        {"toy_gaussian", "lba2_b", "approx_ddm_z", "softmax_beta"},
        "manifest.model_contracts",
    )
    ddm = _require_object(contracts["approx_ddm_z"], "model_contracts.approx_ddm_z")
    network_path = ddm.get("network_path")
    network_digest = ddm.get("network_sha256")
    if not isinstance(network_path, str) or PurePosixPath(network_path).is_absolute():
        raise QualificationError("approx-DDM network path must be repository-relative")
    if not isinstance(network_digest, str) or not SHA256.fullmatch(network_digest):
        raise QualificationError("approx-DDM network digest is invalid")
    resolved = (REPO_ROOT / network_path).resolve()
    if REPO_ROOT not in resolved.parents or not resolved.is_file():
        raise QualificationError("approx-DDM network path is unavailable")
    if _file_sha256(resolved) != network_digest:
        raise QualificationError("approx-DDM network digest does not match the fixture")
    _assert_finite_json(contracts, "manifest.model_contracts")


def _validate_frozen_contract(
    value: Any, expected: Mapping[str, Any], path: str
) -> None:
    contract = _require_object(value, path)
    _assert_finite_json(contract, path)
    if contract != expected:
        raise QualificationError(f"{path} changes the reviewed v2 contract")


def _validate_prior_contracts(value: Any) -> None:
    _validate_frozen_contract(
        value,
        {
            "truncated_normal": {
                "parameter_scale": "response",
                "link": "identity",
                "location_prior": {
                    "family": "TruncatedNormal",
                    "base_rule": "finite-midpoint-otherwise-zero",
                    "sigma": 0.25,
                    "bounds": "scenario",
                },
                "scale_prior": {
                    "family": "Weibull",
                    "alpha": 1.5,
                    "beta": 0.3,
                },
                "group_prior": {
                    "family": "TruncatedNormal",
                    "bounds": "scenario",
                    "noncentered": False,
                },
            },
            "linked_normal": {
                "parameter_scale": "linear_predictor",
                "link_rule": {
                    "lower": "lower-plus-exp",
                    "upper": "upper-minus-exp",
                    "two_sided": "generalized-logit",
                    "narrow": "generalized-logit",
                },
                "location_prior": {
                    "family": "Normal",
                    "mu": 0.0,
                    "sigma": 0.25,
                },
                "scale_prior": {
                    "family": "Weibull",
                    "alpha": 1.5,
                    "beta": 0.3,
                },
                "group_prior": {"family": "Normal", "noncentered": False},
            },
        },
        "manifest.prior_contracts",
    )


def _validate_data_generation(value: Any) -> None:
    _validate_frozen_contract(
        value,
        {
            "rng": "numpy-pcg64",
            "stream_derivation": (
                "data_seed=blake2b-64-v2(master_seed,data_id,replicate,data); "
                "truth_seed/group_seed/observation_seed=blake2b-64-v2("
                "data_seed,data_id,replicate,domain)"
            ),
            "fixed_group_values": {
                "distribution": "scipy.stats.truncnorm",
                "standardized_lower": (
                    "(lower-truth_group_location)/truth_group_scale; "
                    "negative-infinity when lower is null"
                ),
                "standardized_upper": (
                    "(upper-truth_group_location)/truth_group_scale; "
                    "positive-infinity when upper is null"
                ),
                "loc": "truth_group_location",
                "scale": "truth_group_scale",
                "size": "n_groups",
                "random_state": "numpy.random.Generator(PCG64(group_seed))",
            },
            "candidate_control_data": (
                "identical-natural-scale-group-values-and-observations"
            ),
            "fixed_truth_sequence": [
                "read scenario truth_group_location and truth_group_scale",
                "draw group coefficients from fixed_group_values using group_seed",
                "draw observations using observation_seed",
            ],
            "sbc_truth_sequence": [
                "draw group_location from the candidate TruncatedNormal location "
                "prior using truth_seed",
                "draw group_scale as 0.3*numpy_generator.weibull(1.5) using the "
                "same truth_seed stream after group_location",
                "draw group coefficients from "
                "TruncatedNormal(group_location,group_scale,scenario bounds) using "
                "group_seed",
                "draw observations using observation_seed",
            ],
            "toy_gaussian": (
                "y[group,trial] ~ Normal(group_value[group], 0.5) using "
                "observation_seed"
            ),
            "lba2_b": ("ssm_simulators-lba2-size-1-with-frozen-fixed-parameters"),
            "approx_ddm_z": ("ssm_simulators-ddm-size-1-with-frozen-fixed-parameters"),
            "softmax_beta": (
                "response=1 iff U<expit(group_value), else -1, with U from "
                "observation_seed"
            ),
            "zero_observation_semantics": (
                "n_per_group=0 omits the observed likelihood and retains n_groups "
                "latent group coefficients"
            ),
            "sbc_posterior_use": (
                "all retained draws feed posterior summaries; only rank computation "
                "uses the frozen selected 100 draws"
            ),
        },
        "manifest.data_generation",
    )


def _validate_execution_policy(value: Any) -> None:
    _validate_frozen_contract(
        value,
        {
            "process_isolation": "one-cell-per-fresh-process",
            "cache_policy": "unique-pytensor-and-jax-cache-per-cell",
            "timing_clock": "time-perf-counter",
            "phase_order": [
                "generate-or-verify-data-artifact-untimed",
                "build-model-untimed",
                "start-timer",
                "generate-exact-chain-starts-once",
                "sample-with-those-exact-starts-including-compile-jit-warmup-and-draws",
                "stop-timer",
                "write-start-and-chain-artifacts",
                "run-gradient-and-parity-diagnostics-in-a-separate-fresh-process-and-cache",
                "write-cell-artifact-as-final-atomic-completion-marker",
            ],
            "timing_scope": (
                "exact-start-generation-plus-sampler-call-including-compilation-jit-"
                "warmup-and-draws;excluding-data-generation-model-construction-"
                "artifact-io-and-postsampling-diagnostics"
            ),
            "required_device": "cpu",
            "precision_contract": {
                "float32": ("PYTENSOR_FLAGS=floatX=float32 and JAX_ENABLE_X64=false"),
                "float64": "PYTENSOR_FLAGS=floatX=float64 and JAX_ENABLE_X64=true",
            },
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            "cores": 1,
            "chain_method": "sequential",
            "progressbar": False,
            "compute_convergence_checks": False,
            "sbc_draw_selection": {
                "method": "sha256-score-without-replacement-v1",
                "source_order": "chain-major-retained-draws",
                "count": 100,
            },
            "initialization_policies": {
                "backend-default": (
                    "use the explicit start_seeds; call "
                    "make_initial_point_fns_per_chain once with every free RV "
                    "jittered; pass those exact transformed starts with adapt_diag"
                ),
                "hssm-default": (
                    "seed and restore legacy NumPy with initialization_seed while "
                    "HSSM performs constrained preprocessing and support-aware "
                    "jitter; transform the resulting _initvals once with no "
                    "additional jitter; replicate that identical transformed start "
                    "across chains and pass it with adapt_diag"
                ),
            },
            "seed_execution": {
                "start_seeds": (
                    "backend-default only: derive_start_seed(initialization_seed,"
                    "cell_id,replicate,chain_index) for every chain; hssm-default "
                    "records an empty list and consumes initialization_seed exactly "
                    "once while constructing _initvals"
                ),
                "pymc": (
                    "chain_seeds has length chains and is passed verbatim as "
                    "random_seed; sampler_seed is null; "
                    "effective_numpyro_chain_keys is null"
                ),
                "numpyro": (
                    "chain_seeds is empty; scalar sampler_seed is passed to "
                    "sample_numpyro_nuts; effective key is PRNGKey(sampler_seed) for "
                    "one chain or jax.random.split(PRNGKey(sampler_seed),chains) "
                    "otherwise; record every resulting two-uint32 key in "
                    "effective_numpyro_chain_keys"
                ),
                "numpyro_split_version": (
                    "jax-0.11.1-threefry-PRNGKey-and-random-split"
                ),
            },
        },
        "manifest.execution_policy",
    )


def _validate_artifact_policy(value: Any) -> None:
    _validate_frozen_contract(
        value,
        {
            "data_path": "data/<data_id>-r<replicate>.json",
            "cell_path": "cells/<cell_id>.json",
            "start_path": "starts/<cell_id>.json",
            "chain_path": "chains/<cell_id>.nc",
            "hash": "sha256-exact-bytes",
            "data_artifact": (
                "canonical strict JSON containing scenario-independent truth, group "
                "values, observations, and all DGP seeds; shared writers must verify "
                "identical bytes"
            ),
            "write_order": ["start", "chain", "cell"],
            "chain_dimensions": ["chain", "draw"],
            "chain_variables": [
                "group_location",
                "group_scale",
                "group_first",
                "group_middle",
                "group_last",
            ],
            "chain_variable_scales": {
                "group_location": "response",
                "group_scale": (
                    "prior-native (response for truncated_normal; linear_predictor "
                    "for linked_normal)"
                ),
                "group_first": "response",
                "group_middle": "response",
                "group_last": "response",
            },
            "sample_stats_variables": [
                "diverging",
                "energy",
                "tree_depth",
                "n_steps",
                "step_size",
                "acceptance_rate",
            ],
            "failure_semantics": (
                "data is retained after successful generation; start is retained only "
                "after successful initialization; chain is retained only after "
                "successful sampling and validation; cell result is always the final "
                "atomic completion marker and names absent artifacts as null; discard "
                "partial temporary files"
            ),
            "start_coordinates": "pymc-transformed-value-variables",
        },
        "manifest.artifact_policy",
    )


def _validate_analysis_policy(value: Any) -> None:
    policy = _require_object(value, "manifest.analysis_policy")
    expected = {
        "monitored_parameters",
        "fixed_recovery_abs_mean_standardized_error_max",
        "fixed_recovery_replicates",
        "fixed_recovery_bias_policy",
        "familywise_alpha",
        "sbc_rank_draw_count",
        "sbc_replicates",
        "sbc_draw_selection",
        "coverage_power_design",
        "coverage_levels",
        "backend_combined_rank_rhat_max",
        "backend_posterior_mean_mcse_z_max",
        "gradient_contract",
        "posterior_summary",
        "gradient_evaluation",
        "sampler_stat_mapping",
    }
    _require_exact_keys(policy, expected, "manifest.analysis_policy")
    parameters = policy["monitored_parameters"]
    if (
        not isinstance(parameters, list)
        or not parameters
        or len(parameters) != len(set(parameters))
        or any(
            not isinstance(item, str) or not SAFE_PARAMETER.fullmatch(item)
            for item in parameters
        )
    ):
        raise QualificationError("analysis monitored parameters must be unique slugs")
    if policy["coverage_levels"] != [0.9, 0.95]:
        raise QualificationError("analysis coverage levels must remain [0.9, 0.95]")
    if policy["sbc_rank_draw_count"] != 100 or policy["sbc_replicates"] != 275:
        raise QualificationError(
            "SBC rank draws and replicates must remain 100 and 275"
        )
    if policy["fixed_recovery_replicates"] != 5:
        raise QualificationError("fixed recovery must retain five replicates")
    if policy["fixed_recovery_bias_policy"] != (
        "absolute-mean-standardized-error-only; five-replicate sign tests are "
        "descriptive"
    ):
        raise QualificationError("fixed recovery bias policy has changed")
    expected_draw_selection = {
        "method": "sha256-score-without-replacement-v1",
        "source_order": "chain-major-retained-draws",
        "count": 100,
    }
    if policy["sbc_draw_selection"] != expected_draw_selection:
        raise QualificationError("SBC draw-selection policy has changed")
    exact_algorithms = {
        "posterior_summary": {
            "draw_set": "all-retained-post-warmup-draws",
            "quantile_method": "numpy-linear",
            "rhat_method": "arviz-rank-normalized-split",
            "ess_methods": ["arviz-bulk", "arviz-tail"],
            "mcse_method": "arviz-mean",
        },
        "gradient_evaluation": {
            "coordinate_system": ("pymc-transformed-value-variables-with-jacobian"),
            "point": "first-chain-exact-sampler-start",
            "finite_difference_stencil": "five-point-central",
            "finite_difference_step": (
                "machine-epsilon-to-the-one-fifth-times-max-one-abs-coordinate"
            ),
            "model_gradient_order": (
                "pymc-continuous-value-vars-raveled-in-model-order"
            ),
        },
        "sampler_stat_mapping": {
            "diverging": "boolean-per-draw",
            "energy": "hamiltonian-energy-per-draw",
            "tree_depth": "tree-depth-per-draw",
            "n_steps": "leapfrog-steps-per-draw",
            "step_size": "integrator-step-size-per-draw",
            "acceptance_rate": "acceptance-probability-per-draw",
        },
    }
    for field, expected_algorithm in exact_algorithms.items():
        if policy[field] != expected_algorithm:
            raise QualificationError(f"analysis {field} algorithm has changed")
    power_design = _require_object(
        policy["coverage_power_design"], "analysis_policy.coverage_power_design"
    )
    _require_exact_keys(
        power_design,
        {
            "method",
            "candidate_scenario_ids",
            "candidate_parameter_units",
            "coverage_checks_per_unit",
            "alternative_undercoverage",
            "minimum_power",
            "minimum_replicates",
            "power_at_minimum_for_90pct_interval",
            "power_at_minimum_for_95pct_interval",
        },
        "analysis_policy.coverage_power_design",
    )
    if (
        power_design["method"] != "two-sided-clopper-pearson-with-bonferroni"
        or power_design["candidate_scenario_ids"]
        != ["calib-pymc-lower-outside", "calib-pymc-two-sided-midpoint"]
        or power_design["candidate_parameter_units"] != 10
        or power_design["coverage_checks_per_unit"] != 2
        or power_design["minimum_replicates"] != policy["sbc_replicates"]
    ):
        raise QualificationError("coverage power design identity has changed")
    for field in (
        "alternative_undercoverage",
        "minimum_power",
        "power_at_minimum_for_90pct_interval",
        "power_at_minimum_for_95pct_interval",
    ):
        if not _is_number(power_design[field]) or not 0 < power_design[field] < 1:
            raise QualificationError(f"coverage power design {field} is invalid")
    if power_design["minimum_power"] < 0.9:
        raise QualificationError("coverage power design minimum_power is too low")
    for field in (
        "fixed_recovery_abs_mean_standardized_error_max",
        "backend_combined_rank_rhat_max",
        "backend_posterior_mean_mcse_z_max",
    ):
        if not _is_number(policy[field]) or policy[field] <= 0:
            raise QualificationError(f"analysis policy {field} must be positive")
    alpha = policy["familywise_alpha"]
    if not _is_number(alpha) or not 0 < alpha < 1:
        raise QualificationError("analysis familywise_alpha must lie in (0, 1)")
    gradient = _require_object(
        policy["gradient_contract"], "analysis_policy.gradient_contract"
    )
    _require_exact_keys(
        gradient,
        {
            "evaluation",
            "combined_tolerance_rule",
            "abs_rel_maxima_role",
            "finite_difference",
            "pytensor_jax",
            "bambi_isomorphism",
            "likelihood_pytensor_jax",
        },
        "analysis_policy.gradient_contract",
    )
    if gradient["combined_tolerance_rule"] != (
        "max(abs(observed-reference)/(absolute_tolerance+relative_tolerance*"
        "max(abs(reference),abs(observed)))) <= 1"
    ):
        raise QualificationError("gradient combined tolerance rule has changed")
    if gradient["abs_rel_maxima_role"] != "descriptive-only":
        raise QualificationError("gradient abs/rel maxima role has changed")
    evaluation = _require_object(
        gradient["evaluation"], "analysis_policy.gradient_contract.evaluation"
    )
    _require_exact_keys(
        evaluation,
        {
            "tiers",
            "scenario_replicate",
            "posterior_pair_owner_sampler",
            "bambi_isomorphism_layers",
        },
        "analysis_policy.gradient_contract.evaluation",
    )
    if evaluation != {
        "tiers": ["smoke", "qualification"],
        "scenario_replicate": 0,
        "posterior_pair_owner_sampler": "pymc",
        "bambi_isomorphism_layers": ["bambi"],
    }:
        raise QualificationError("gradient contract evaluation policy has changed")
    expected_precisions = {
        "finite_difference": {"float64", "float32", "float32_narrow_stress"},
        "pytensor_jax": {"float64", "float32"},
        "bambi_isomorphism": {"float64", "float32"},
        "likelihood_pytensor_jax": {"float64", "float32"},
    }
    for contract_name, precisions in expected_precisions.items():
        contract = _require_object(
            gradient[contract_name],
            f"analysis_policy.gradient_contract.{contract_name}",
        )
        _require_exact_keys(
            contract,
            precisions,
            f"analysis_policy.gradient_contract.{contract_name}",
        )
        for precision, raw_tolerances in contract.items():
            path = f"analysis_policy.gradient_contract.{contract_name}.{precision}"
            tolerances = _require_object(raw_tolerances, path)
            tolerance_fields = (
                {
                    "value_absolute_tolerance",
                    "value_relative_tolerance",
                    "gradient_absolute_tolerance",
                    "gradient_relative_tolerance",
                }
                if contract_name == "likelihood_pytensor_jax"
                else {"absolute_tolerance", "relative_tolerance"}
            )
            _require_exact_keys(tolerances, tolerance_fields, path)
            if any(
                not _is_number(tolerance) or tolerance <= 0
                for tolerance in tolerances.values()
            ):
                raise QualificationError(f"{path} tolerances must be positive")


def _validate_scenario(scenario: Any, tiers: Mapping[str, Any]) -> None:
    scenario = _require_object(scenario, "scenario")
    scenario_id = scenario.get("scenario_id", "<unknown>")
    path = f"scenario[{scenario_id}]"
    missing = SCENARIO_KEYS - scenario.keys()
    unknown = scenario.keys() - SCENARIO_ALL_KEYS
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise QualificationError(f"{path} has invalid fields: {', '.join(details)}")
    if not isinstance(scenario_id, str) or not SAFE_ID.fullmatch(scenario_id):
        raise QualificationError(f"{path}.scenario_id is not a canonical slug")
    tier = scenario["tier"]
    if tier not in ALLOWED_TIERS:
        raise QualificationError(f"{path}.tier is unsupported")
    if scenario["gate"] != tiers[tier]["gate"]:
        raise QualificationError(f"{path}.gate disagrees with its tier")
    enums = {
        "layer": {"pymc", "bambi", "hssm"},
        "purpose": {"candidate", "control", "diagnostic"},
        "bound_kind": {"lower", "upper", "two_sided", "narrow"},
        "prior": {"truncated_normal", "linked_normal"},
        "truth_kind": {"fixed", "prior_predictive"},
        "truth_regime": {"near_lower", "near_upper", "interior", "prior_predictive"},
        "floatx": {"float32", "float64"},
        "sampler": {"pymc", "numpyro"},
        "initialization_policy": {"backend-default", "hssm-default"},
    }
    for field, allowed in enums.items():
        if scenario[field] not in allowed:
            raise QualificationError(f"{path}.{field} is unsupported")
    for field in ("model",):
        if not isinstance(scenario[field], str) or not scenario[field]:
            raise QualificationError(f"{path}.{field} must be a non-empty string")
    for field in ("data_id", "posterior_pair_id"):
        value = scenario.get(field)
        if value is not None and (
            not isinstance(value, str) or not SAFE_ID.fullmatch(value)
        ):
            raise QualificationError(f"{path}.{field} must be a canonical slug")
    if scenario.get("data_id") is None:
        raise QualificationError(f"{path}.data_id is required in v2")
    dependency_profile = scenario.get("dependency_profile")
    if dependency_profile is not None and (
        not isinstance(dependency_profile, str) or not dependency_profile
    ):
        raise QualificationError(f"{path}.dependency_profile must be non-empty")
    calibration_kind = scenario.get("calibration_kind")
    if calibration_kind is not None and calibration_kind != "sbc":
        raise QualificationError(f"{path}.calibration_kind is unsupported")
    if not _is_number(scenario["prior_hyper_location"]) or not math.isfinite(
        scenario["prior_hyper_location"]
    ):
        raise QualificationError(f"{path}.prior_hyper_location must be finite")
    if not _is_number(scenario["target_accept"]) or not math.isfinite(
        scenario["target_accept"]
    ):
        raise QualificationError(f"{path}.target_accept must be finite")
    if not 0 < scenario["target_accept"] < 1:
        raise QualificationError(f"{path}.target_accept must lie in (0, 1)")
    for field in ("n_groups", "replicates", "chains", "tune", "draws"):
        if not _is_int(scenario[field]) or scenario[field] <= 0:
            raise QualificationError(f"{path}.{field} must be a positive integer")
    if not _is_int(scenario["n_per_group"]) or scenario["n_per_group"] < 0:
        raise QualificationError(f"{path}.n_per_group must be a non-negative integer")
    for field in ("canonical", "recovery"):
        if not isinstance(scenario[field], bool):
            raise QualificationError(f"{path}.{field} must be boolean")
    for field in ("lower", "upper"):
        bound = scenario[field]
        if bound is not None and (not _is_number(bound) or not math.isfinite(bound)):
            raise QualificationError(f"{path}.{field} must be finite or null")
    lower, upper = scenario["lower"], scenario["upper"]
    kind = scenario["bound_kind"]
    if kind == "lower" and (lower is None or upper is not None):
        raise QualificationError(f"{path} has inconsistent lower-bound fields")
    if kind == "upper" and (lower is not None or upper is None):
        raise QualificationError(f"{path} has inconsistent upper-bound fields")
    if kind in {"two_sided", "narrow"} and (
        lower is None or upper is None or lower >= upper
    ):
        raise QualificationError(f"{path} has inconsistent two-sided bounds")

    expected_hyper_location = (
        (lower + upper) / 2
        if scenario["prior"] == "truncated_normal"
        and lower is not None
        and upper is not None
        else 0.0
    )
    if scenario["prior_hyper_location"] != expected_hyper_location:
        raise QualificationError(
            f"{path}.prior_hyper_location does not follow its frozen prior rule"
        )

    truth_kind = scenario["truth_kind"]
    truth_location = scenario["truth_group_location"]
    truth_scale = scenario["truth_group_scale"]
    truth_regime = scenario["truth_regime"]
    truth_boundary = scenario["truth_boundary"]
    expected_boundary = {
        "near_lower": "lower",
        "near_upper": "upper",
        "interior": None,
        "prior_predictive": None,
    }[truth_regime]
    if truth_boundary != expected_boundary:
        raise QualificationError(f"{path}.truth_boundary disagrees with truth_regime")
    if truth_boundary is not None and truth_boundary not in {"lower", "upper"}:
        raise QualificationError(f"{path}.truth_boundary is unsupported")
    if truth_kind == "fixed":
        if truth_regime == "prior_predictive":
            raise QualificationError(f"{path} fixed truth cannot be prior_predictive")
        if not _is_number(truth_location) or not math.isfinite(truth_location):
            raise QualificationError(f"{path}.truth_group_location must be finite")
        if (
            not _is_number(truth_scale)
            or not math.isfinite(truth_scale)
            or truth_scale <= 0
        ):
            raise QualificationError(
                f"{path}.truth_group_scale must be finite and positive"
            )
        if lower is not None and truth_location <= lower:
            raise QualificationError(
                f"{path}.truth_group_location must lie strictly above lower"
            )
        if upper is not None and truth_location >= upper:
            raise QualificationError(
                f"{path}.truth_group_location must lie strictly below upper"
            )
        if calibration_kind is not None:
            raise QualificationError(f"{path} fixed truth cannot request calibration")
    else:
        if (
            truth_regime != "prior_predictive"
            or truth_location is not None
            or truth_scale is not None
            or calibration_kind != "sbc"
        ):
            raise QualificationError(
                f"{path} prior-predictive truth requires null fixed truths and SBC"
            )

    expected_group_indices = [0, scenario["n_groups"] // 2, scenario["n_groups"] - 1]
    if scenario["group_indices"] != expected_group_indices:
        raise QualificationError(
            f"{path}.group_indices must remain {expected_group_indices}"
        )
    if len(set(expected_group_indices)) != 3:
        raise QualificationError(
            f"{path}.n_groups cannot define three monitored groups"
        )

    expected_initialization = (
        "hssm-default" if scenario["layer"] == "hssm" else "backend-default"
    )
    if scenario["initialization_policy"] != expected_initialization:
        raise QualificationError(
            f"{path}.initialization_policy does not match its model layer"
        )
    expected_purpose = {
        "smoke": {"candidate"},
        "qualification": {"candidate", "control"},
        "stress": {"diagnostic"},
    }[tier]
    if scenario["purpose"] not in expected_purpose:
        raise QualificationError(f"{path}.purpose disagrees with its tier")
    if scenario["purpose"] == "control":
        if scenario["prior"] != "linked_normal" or scenario["recovery"]:
            raise QualificationError(
                f"{path} controls must use linked_normal and recovery=false"
            )
    elif scenario["prior"] != "truncated_normal":
        raise QualificationError(
            f"{path} candidate/diagnostic cells must use truncated_normal"
        )
    if tier != "qualification" and scenario["recovery"]:
        raise QualificationError(f"{path} only primary candidates may recover truth")
    control_id = scenario["control_id"]
    if control_id is not None and (
        not isinstance(control_id, str) or not SAFE_ID.fullmatch(control_id)
    ):
        raise QualificationError(f"{path}.control_id must be null or a canonical slug")

    if tier == "qualification" and calibration_kind != "sbc":
        frozen_values = {
            "replicates": 5,
            "chains": 4,
            "tune": 1000,
            "draws": 1000,
            "target_accept": 0.9,
        }
        for field, expected in frozen_values.items():
            if scenario[field] != expected:
                raise QualificationError(
                    f"{path}.{field} must remain {expected!r} in the primary gate"
                )
        if scenario["purpose"] == "candidate" and not scenario["recovery"]:
            raise QualificationError(f"{path} primary candidates require recovery")
    if calibration_kind == "sbc":
        if tier != "qualification" or scenario["purpose"] != "candidate":
            raise QualificationError(
                f"{path} SBC calibration must be a primary candidate cell"
            )
        if scenario["replicates"] != 275 or not scenario["recovery"]:
            raise QualificationError(
                f"{path} calibration requires 275 recovery replicates"
            )
        if scenario["control_id"] is not None:
            raise QualificationError(f"{path} calibration cannot reference a control")
    if scenario["canonical"] and (
        tier != "qualification"
        or scenario["purpose"] != "candidate"
        or scenario["layer"] != "hssm"
        or scenario["model"] not in {"lba2_b", "approx_ddm_z"}
    ):
        raise QualificationError(f"{path} is not an eligible canonical HSSM cell")
    if tier != "stress" and scenario["target_accept"] != 0.9:
        raise QualificationError(f"{path} changes target_accept outside diagnostics")


def validate_manifest(manifest: Any) -> Mapping[str, Any]:
    """Validate and return the frozen study manifest."""
    manifest = _require_object(manifest, "manifest")
    if manifest.get("schema_version") == 1:
        raise QualificationError(
            "truncated_hierarchy_v1 is superseded and cannot be executed"
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise QualificationError("unsupported manifest schema_version")
    _require_exact_keys(manifest, MANIFEST_KEYS, "manifest")
    if manifest["study_id"] != "truncated_hierarchy_v2":
        raise QualificationError("unexpected study_id")
    if manifest["status"] != "frozen-before-primary-runs":
        raise QualificationError("manifest must be frozen before primary runs")
    if not isinstance(manifest["description"], str) or not manifest["description"]:
        raise QualificationError("manifest.description must be non-empty")
    if not _is_int(manifest["master_seed"]) or manifest["master_seed"] < 0:
        raise QualificationError("manifest.master_seed must be a non-negative integer")
    if manifest["seed_derivation"] != "blake2b-64-v2":
        raise QualificationError("unsupported seed derivation")
    _validate_tiers(manifest["tiers"])
    _validate_dependency_profiles(manifest["dependency_profiles"])
    _validate_model_contracts(manifest["model_contracts"])
    _validate_prior_contracts(manifest["prior_contracts"])
    _validate_data_generation(manifest["data_generation"])
    _validate_analysis_policy(manifest["analysis_policy"])
    _validate_execution_policy(manifest["execution_policy"])
    _validate_artifact_policy(manifest["artifact_policy"])
    _validate_thresholds(manifest["thresholds"])
    if (
        manifest["analysis_policy"]["sbc_draw_selection"]
        != manifest["execution_policy"]["sbc_draw_selection"]
    ):
        raise QualificationError("analysis and execution SBC draw selection disagree")
    scenarios = manifest["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise QualificationError("manifest.scenarios must be a non-empty list")
    for scenario in scenarios:
        _validate_scenario(scenario, manifest["tiers"])
    scenario_ids = [scenario["scenario_id"] for scenario in scenarios]
    duplicates = sorted(
        scenario_id for scenario_id, count in Counter(scenario_ids).items() if count > 1
    )
    if duplicates:
        raise QualificationError(f"duplicate scenario identifiers: {duplicates}")
    by_id = {scenario["scenario_id"]: scenario for scenario in scenarios}
    calibration_scenarios = [
        scenario for scenario in scenarios if scenario.get("calibration_kind") == "sbc"
    ]
    declared_calibration_scenarios = manifest["analysis_policy"][
        "coverage_power_design"
    ]["candidate_scenario_ids"]
    if [scenario["scenario_id"] for scenario in calibration_scenarios] != list(
        declared_calibration_scenarios
    ):
        raise QualificationError(
            "SBC calibration scenarios do not match the frozen candidate_scenario_ids"
        )
    expected_calibration_units = len(calibration_scenarios) * len(
        manifest["analysis_policy"]["monitored_parameters"]
    )
    declared_calibration_units = manifest["analysis_policy"]["coverage_power_design"][
        "candidate_parameter_units"
    ]
    if expected_calibration_units != declared_calibration_units:
        raise QualificationError(
            "SBC calibration scenarios and monitored parameters do not match "
            "coverage_power_design.candidate_parameter_units"
        )
    profiles = manifest["dependency_profiles"]
    contracts = manifest["model_contracts"]
    for scenario in scenarios:
        profile = scenario.get("dependency_profile", "current-resolved")
        if profile not in profiles:
            raise QualificationError(
                f"scenario[{scenario['scenario_id']}] uses unknown dependency profile"
            )
        if profile == "bambi-0.19" and scenario["layer"] != "bambi":
            raise QualificationError(
                f"scenario[{scenario['scenario_id']}] uses the Bambi floor outside "
                "the Bambi layer"
            )
        if scenario["model"] not in contracts:
            raise QualificationError(
                f"scenario[{scenario['scenario_id']}] lacks a model contract"
            )

    by_data_id: dict[str, list[Mapping[str, Any]]] = {}
    for scenario in scenarios:
        data_id = scenario.get("data_id")
        if data_id is not None:
            by_data_id.setdefault(data_id, []).append(scenario)
    for data_id, grouped in by_data_id.items():
        reference = grouped[0]
        for scenario in grouped[1:]:
            mismatched = sorted(
                field
                for field in DATA_MATCH_FIELDS
                if scenario.get(field) != reference.get(field)
            )
            if mismatched:
                raise QualificationError(
                    f"data_id {data_id} mixes data-generating fields {mismatched}"
                )

    by_posterior_pair: dict[str, list[Mapping[str, Any]]] = {}
    for scenario in scenarios:
        pair_id = scenario.get("posterior_pair_id")
        if pair_id is not None:
            by_posterior_pair.setdefault(pair_id, []).append(scenario)
    for pair_id, grouped in by_posterior_pair.items():
        if len(grouped) != 2 or {item["sampler"] for item in grouped} != {
            "pymc",
            "numpyro",
        }:
            raise QualificationError(
                f"posterior_pair_id {pair_id} must join one PyMC and one NumPyro cell"
            )
        reference, other = grouped
        mismatched = sorted(
            field
            for field in POSTERIOR_PAIR_MATCH_FIELDS
            if reference.get(field, "current-resolved")
            != other.get(field, "current-resolved")
        )
        if mismatched:
            raise QualificationError(
                f"posterior_pair_id {pair_id} differs in {mismatched}"
            )
        if reference.get("data_id") is None:
            raise QualificationError(
                f"posterior_pair_id {pair_id} must declare a shared data_id"
            )

    mandatory_direct_numpyro = {
        "qual-pymc-lower-outside-numpyro",
        "qual-pymc-two-sided-near-numpyro",
    }
    if not mandatory_direct_numpyro <= by_id.keys():
        raise QualificationError(
            "direct PyMC NumPyro qualification coverage is incomplete"
        )
    referenced_controls: Counter[str] = Counter()
    for scenario in scenarios:
        control_id = scenario["control_id"]
        if scenario["purpose"] == "control" and control_id is not None:
            raise QualificationError(
                f"scenario[{scenario['scenario_id']}] control cannot reference "
                "a control"
            )
        if (
            scenario["tier"] == "qualification"
            and scenario["purpose"] == "candidate"
            and scenario.get("calibration_kind") is None
        ):
            if control_id is None:
                raise QualificationError(
                    f"scenario[{scenario['scenario_id']}] lacks a paired control"
                )
        if control_id is None:
            continue
        if control_id not in by_id:
            raise QualificationError(
                f"scenario[{scenario['scenario_id']}] references unknown control "
                f"{control_id}"
            )
        control = by_id[control_id]
        if control["purpose"] != "control" or control["tier"] != scenario["tier"]:
            raise QualificationError(
                f"scenario[{scenario['scenario_id']}] has an inadmissible control"
            )
        mismatched = sorted(
            field
            for field in CONTROL_MATCH_FIELDS
            if scenario.get(field) != control.get(field)
        )
        if mismatched:
            raise QualificationError(
                f"scenario[{scenario['scenario_id']}] control differs in {mismatched}"
            )
        if scenario["prior"] == control["prior"]:
            raise QualificationError(
                f"scenario[{scenario['scenario_id']}] control does not change the prior"
            )
        referenced_controls[control_id] += 1
    orphaned_controls = sorted(
        scenario["scenario_id"]
        for scenario in scenarios
        if scenario["purpose"] == "control"
        and referenced_controls[scenario["scenario_id"]] != 1
    )
    if orphaned_controls:
        raise QualificationError(
            f"controls must be referenced exactly once: {orphaned_controls}"
        )
    canonical_backends = {
        (scenario["model"], scenario["sampler"])
        for scenario in scenarios
        if scenario["canonical"]
    }
    required_backends = {
        ("lba2_b", "pymc"),
        ("lba2_b", "numpyro"),
        ("approx_ddm_z", "pymc"),
        ("approx_ddm_z", "numpyro"),
    }
    if not required_backends <= canonical_backends:
        raise QualificationError("canonical HSSM backend coverage is incomplete")
    bambi_primary_profiles = {
        scenario.get("dependency_profile", "current-resolved")
        for scenario in scenarios
        if scenario["tier"] == "qualification"
        and scenario["layer"] == "bambi"
        and scenario["purpose"] == "candidate"
    }
    if bambi_primary_profiles != {"current-resolved", "bambi-0.19"}:
        raise QualificationError("Bambi floor/resolved primary coverage is incomplete")
    return manifest


def load_manifest(path: Path = DEFAULT_MANIFEST) -> Mapping[str, Any]:
    """Load and validate a qualification manifest from disk."""
    return validate_manifest(_load_json(path))


def derive_seed(
    master_seed: int,
    scenario_id: str,
    replicate: int,
    purpose: str,
    chain_index: int | None = None,
) -> int:
    """Derive one domain-separated positive 31-bit v2 experiment seed."""
    if not _is_int(master_seed) or master_seed < 0:
        raise QualificationError("master_seed must be a non-negative integer")
    if not isinstance(scenario_id, str) or not SAFE_ID.fullmatch(scenario_id):
        raise QualificationError("scenario_id must be a canonical slug")
    if not _is_int(replicate) or replicate < 0:
        raise QualificationError("replicate must be a non-negative integer")
    allowed_purposes = {
        "data",
        "initialization",
        "sampler",
        "sbc_draw",
        "sbc_tie",
        "chain",
    }
    if purpose not in allowed_purposes:
        raise QualificationError(f"purpose must be one of {sorted(allowed_purposes)}")
    if purpose != "chain" and chain_index is not None:
        raise QualificationError(f"{purpose} seeds cannot have a chain_index")
    if purpose == "chain":
        if not isinstance(chain_index, int) or isinstance(chain_index, bool):
            raise QualificationError("chain seeds require a non-negative chain_index")
        if chain_index < 0:
            raise QualificationError("chain seeds require a non-negative chain_index")
    payload = "\0".join(
        [
            "hssm-truncated-hierarchy-seed-v2",
            str(master_seed),
            scenario_id,
            str(replicate),
            purpose,
            "" if chain_index is None else str(chain_index),
        ]
    ).encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"hssm1282v2").digest()
    return int.from_bytes(digest, "big") % (2**31 - 1) + 1


def derive_data_stream_seed(
    data_seed: int, data_id: str, replicate: int, purpose: str
) -> int:
    """Derive one DGP stream seed from its shared per-data-replicate root."""
    if not _is_int(data_seed) or not 0 < data_seed < 2**31:
        raise QualificationError("data_seed must be a positive 31-bit integer")
    if not isinstance(data_id, str) or not SAFE_ID.fullmatch(data_id):
        raise QualificationError("data_id must be a canonical slug")
    if not _is_int(replicate) or replicate < 0:
        raise QualificationError("replicate must be a non-negative integer")
    allowed_purposes = {"truth", "group", "observation"}
    if purpose not in allowed_purposes:
        raise QualificationError(
            f"data stream purpose must be one of {sorted(allowed_purposes)}"
        )
    payload = "\0".join(
        [
            "hssm-truncated-hierarchy-seed-v2",
            str(data_seed),
            data_id,
            str(replicate),
            purpose,
            "",
        ]
    ).encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"hssm1282v2").digest()
    return int.from_bytes(digest, "big") % (2**31 - 1) + 1


def derive_start_seed(
    initialization_seed: int,
    cell_id: str,
    replicate: int,
    chain_index: int,
) -> int:
    """Derive one chain-start seed from the cell initialization root."""
    if not _is_int(initialization_seed) or not 0 < initialization_seed < 2**31:
        raise QualificationError(
            "initialization_seed must be a positive 31-bit integer"
        )
    if not isinstance(cell_id, str) or not SAFE_CELL_ID.fullmatch(cell_id):
        raise QualificationError("cell_id must be canonical")
    if not _is_int(replicate) or replicate < 0:
        raise QualificationError("replicate must be a non-negative integer")
    if not _is_int(chain_index) or chain_index < 0:
        raise QualificationError("chain_index must be a non-negative integer")
    payload = "\0".join(
        [
            "hssm-truncated-hierarchy-seed-v2",
            str(initialization_seed),
            cell_id,
            str(replicate),
            f"start-{chain_index}",
            "",
        ]
    ).encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"hssm1282v2").digest()
    return int.from_bytes(digest, "big") % (2**31 - 1) + 1


def _cell_id(scenario_id: str, replicate: int) -> str:
    return f"{scenario_id}--replicate-{replicate:02d}"


def _planned_seed_fields(
    manifest: Mapping[str, Any], scenario: Mapping[str, Any], replicate: int
) -> dict[str, Any]:
    master_seed = manifest["master_seed"]
    data_owner = scenario["data_id"]
    scenario_id = scenario["scenario_id"]
    calibration = scenario.get("calibration_kind") == "sbc"
    data_seed = derive_seed(master_seed, data_owner, replicate, "data")
    initialization_seed = derive_seed(
        master_seed, scenario_id, replicate, "initialization"
    )
    cell_id = _cell_id(scenario_id, replicate)
    pymc_sampler = scenario["sampler"] == "pymc"
    explicit_start_seeds = scenario["initialization_policy"] == "backend-default"
    return {
        "data_seed": data_seed,
        "truth_seed": derive_data_stream_seed(
            data_seed, data_owner, replicate, "truth"
        ),
        "group_seed": derive_data_stream_seed(
            data_seed, data_owner, replicate, "group"
        ),
        "observation_seed": derive_data_stream_seed(
            data_seed, data_owner, replicate, "observation"
        ),
        "initialization_seed": initialization_seed,
        "start_seeds": [
            derive_start_seed(initialization_seed, cell_id, replicate, chain)
            for chain in range(scenario["chains"])
        ]
        if explicit_start_seeds
        else [],
        "sampler_seed": (
            None
            if pymc_sampler
            else derive_seed(master_seed, scenario_id, replicate, "sampler")
        ),
        "sbc_draw_seed": (
            derive_seed(master_seed, scenario_id, replicate, "sbc_draw")
            if calibration
            else None
        ),
        "sbc_tie_seed": (
            derive_seed(master_seed, scenario_id, replicate, "sbc_tie")
            if calibration
            else None
        ),
        "chain_seeds": (
            [
                derive_seed(master_seed, scenario_id, replicate, "chain", chain)
                for chain in range(scenario["chains"])
            ]
            if pymc_sampler
            else []
        ),
    }


def _expected_plan_entries(
    manifest: Mapping[str, Any], tier: str
) -> list[dict[str, Any]]:
    digest = manifest_sha256(manifest)
    entries = []
    seed_owners: dict[int, tuple[Any, ...]] = {}
    for scenario in manifest["scenarios"]:
        if scenario["tier"] != tier:
            continue
        for replicate in range(scenario["replicates"]):
            seeds = _planned_seed_fields(manifest, scenario, replicate)
            data_owner = scenario["data_id"]
            scalar_seed_owners = {
                "data_seed": ("data", data_owner, replicate),
                "truth_seed": ("truth", data_owner, replicate),
                "group_seed": ("group", data_owner, replicate),
                "observation_seed": ("observation", data_owner, replicate),
                "initialization_seed": (
                    "initialization",
                    scenario["scenario_id"],
                    replicate,
                ),
                "sampler_seed": ("sampler", scenario["scenario_id"], replicate),
                "sbc_draw_seed": ("sbc_draw", scenario["scenario_id"], replicate),
                "sbc_tie_seed": ("sbc_tie", scenario["scenario_id"], replicate),
            }
            for field, owner in scalar_seed_owners.items():
                seed = seeds[field]
                if seed is None:
                    continue
                previous = seed_owners.setdefault(seed, owner)
                if previous != owner:
                    raise QualificationError("derived seed collision in generated plan")
            for chain, seed in enumerate(seeds["start_seeds"]):
                start_owner = ("start", scenario["scenario_id"], replicate, chain)
                previous = seed_owners.setdefault(seed, start_owner)
                if previous != start_owner:
                    raise QualificationError("derived seed collision in generated plan")
            for chain, seed in enumerate(seeds["chain_seeds"]):
                chain_owner = ("chain", scenario["scenario_id"], replicate, chain)
                previous = seed_owners.setdefault(seed, chain_owner)
                if previous != chain_owner:
                    raise QualificationError("derived seed collision in generated plan")
            entries.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_id": manifest["study_id"],
                    "manifest_sha256": digest,
                    "scenario_sha256": _sha256(scenario),
                    "cell_id": _cell_id(scenario["scenario_id"], replicate),
                    "scenario_id": scenario["scenario_id"],
                    "replicate": replicate,
                    **seeds,
                    "scenario": dict(scenario),
                }
            )
    return entries


def expand_plan(manifest: Mapping[str, Any], tier: str) -> list[dict[str, Any]]:
    """Expand one manifest tier into its deterministic per-replicate plan."""
    validate_manifest(manifest)
    if tier not in ALLOWED_TIERS:
        raise QualificationError(f"unknown tier: {tier}")
    entries = _expected_plan_entries(manifest, tier)
    validate_plan(entries, manifest, tier)
    return entries


def validate_plan(
    plan: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], tier: str
) -> None:
    """Reject missing, duplicated, reordered, or altered canonical plan entries."""
    validate_manifest(manifest)
    if tier not in ALLOWED_TIERS:
        raise QualificationError(f"unknown tier: {tier}")
    if not isinstance(plan, Sequence) or isinstance(plan, str | bytes):
        raise QualificationError("plan must be a sequence")
    expected = _expected_plan_entries(manifest, tier)
    if len(plan) != len(expected):
        raise QualificationError(
            f"plan for {tier} has {len(plan)} cells; expected {len(expected)}"
        )
    for index, (raw_entry, wanted) in enumerate(zip(plan, expected, strict=True)):
        actual = _require_object(raw_entry, f"plan[{index}]")
        _require_exact_keys(actual, PLAN_KEYS, f"plan[{index}]")
        if actual != wanted:
            raise QualificationError(
                f"plan[{index}] does not match frozen cell {wanted['cell_id']}"
            )


def _atomic_write_text(path: Path, text: str, *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise QualificationError(
                    f"refusing to overwrite existing {path}"
                ) from error
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _jsonl(records: Iterable[Mapping[str, Any]]) -> str:
    return "".join(f"{_canonical_json(record)}\n" for record in records)


def _plan_csv(plan: Sequence[Mapping[str, Any]]) -> str:
    scenario_fields = sorted(SCENARIO_ALL_KEYS)
    fieldnames = [
        "cell_id",
        "scenario_id",
        "replicate",
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
        "manifest_sha256",
        "scenario_sha256",
        *scenario_fields,
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for entry in plan:
        row = {
            "cell_id": entry["cell_id"],
            "scenario_id": entry["scenario_id"],
            "replicate": entry["replicate"],
            "data_seed": entry["data_seed"],
            "truth_seed": entry["truth_seed"],
            "group_seed": entry["group_seed"],
            "observation_seed": entry["observation_seed"],
            "initialization_seed": entry["initialization_seed"],
            "start_seeds": ";".join(map(str, entry["start_seeds"])),
            "sampler_seed": entry["sampler_seed"],
            "sbc_draw_seed": entry["sbc_draw_seed"],
            "sbc_tie_seed": entry["sbc_tie_seed"],
            "chain_seeds": ";".join(map(str, entry["chain_seeds"])),
            "manifest_sha256": entry["manifest_sha256"],
            "scenario_sha256": entry["scenario_sha256"],
            **entry["scenario"],
        }
        writer.writerow(row)
    return stream.getvalue()


def _git_value(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _jax_enable_x64() -> bool:
    """Read the effective JAX precision setting from the installed runtime."""
    try:
        import jax
    except ImportError as error:
        raise QualificationError(
            "cannot collect jax_enable_x64 because JAX is unavailable"
        ) from error
    return bool(jax.config.x64_enabled)


def collect_environment(
    manifest: Mapping[str, Any],
    dependency_profile: str = DEFAULT_DEPENDENCY_PROFILE,
) -> dict[str, Any]:
    """Collect and validate one dependency-profile-specific environment."""
    validate_manifest(manifest)
    if dependency_profile not in manifest["dependency_profiles"]:
        raise QualificationError(
            f"unknown environment dependency profile: {dependency_profile}"
        )
    profile = manifest["dependency_profiles"][dependency_profile]
    packages: dict[str, str | None] = {}
    for package in sorted(ENVIRONMENT_PACKAGES):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    status = _git_value("status", "--porcelain")
    record = {
        "schema_version": SCHEMA_VERSION,
        "study_id": manifest["study_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "runner_version": RUNNER_VERSION,
        "dependency_profile": dependency_profile,
        "git": {
            "commit": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
            "dirty": status not in {"", "unknown"},
        },
        "project": {
            "project_path": profile["project_path"],
            "project_sha256": _file_sha256(REPO_ROOT / profile["project_path"]),
            "lock_path": profile["lock_path"],
            "lock_sha256": _file_sha256(REPO_ROOT / profile["lock_path"]),
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "jax_enable_x64": _jax_enable_x64(),
        },
        "packages": packages,
    }
    validate_environment(record, manifest)
    return record


def validate_environment(record: Any, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the environment/provenance sidecar schema."""
    record = _require_object(record, "environment")
    _require_exact_keys(record, ENVIRONMENT_KEYS, "environment")
    if record["schema_version"] != SCHEMA_VERSION:
        raise QualificationError("environment schema_version mismatch")
    if record["study_id"] != manifest["study_id"]:
        raise QualificationError("environment study_id mismatch")
    if record["manifest_sha256"] != manifest_sha256(manifest):
        raise QualificationError("environment manifest digest mismatch")
    if record["runner_version"] != RUNNER_VERSION:
        raise QualificationError("environment runner_version mismatch")
    dependency_profile = record["dependency_profile"]
    profiles = manifest["dependency_profiles"]
    if not isinstance(dependency_profile, str) or dependency_profile not in profiles:
        raise QualificationError("environment dependency_profile is unknown")
    profile = profiles[dependency_profile]
    git = _require_object(record["git"], "environment.git")
    _require_exact_keys(git, {"commit", "branch", "dirty"}, "environment.git")
    if not all(
        isinstance(git[key], str) for key in ("commit", "branch")
    ) or not isinstance(git["dirty"], bool):
        raise QualificationError("environment.git has invalid values")
    project = _require_object(record["project"], "environment.project")
    _require_exact_keys(
        project,
        {"project_path", "project_sha256", "lock_path", "lock_sha256"},
        "environment.project",
    )
    for field in ("project_path", "project_sha256", "lock_path", "lock_sha256"):
        if project[field] != profile[field]:
            raise QualificationError(
                f"environment.project.{field} does not match dependency profile "
                f"{dependency_profile}"
            )
    runtime = _require_object(record["runtime"], "environment.runtime")
    _require_exact_keys(
        runtime,
        {"python", "implementation", "platform", "jax_enable_x64"},
        "environment.runtime",
    )
    if not all(
        isinstance(runtime[key], str) and runtime[key]
        for key in ("python", "implementation", "platform")
    ):
        raise QualificationError("environment.runtime values must be non-empty strings")
    if not isinstance(runtime["jax_enable_x64"], bool):
        raise QualificationError("environment.runtime.jax_enable_x64 must be boolean")
    python_components = runtime["python"].split(".")
    if ".".join(python_components[:2]) != profile["python"]:
        raise QualificationError(
            "environment Python version does not match dependency profile "
            f"{dependency_profile}: expected {profile['python']}, "
            f"found {runtime['python']}"
        )
    packages = _require_object(record["packages"], "environment.packages")
    _require_exact_keys(packages, ENVIRONMENT_PACKAGES, "environment.packages")
    if not all(isinstance(value, str) and value for value in packages.values()):
        raise QualificationError(
            "environment package versions must be non-empty strings"
        )
    for package, required_version in profile["required_versions"].items():
        installed_version = packages[package]
        if installed_version != required_version:
            raise QualificationError(
                f"environment package {package} does not match dependency profile "
                f"{dependency_profile}: expected {required_version}, "
                f"found {installed_version}"
            )
    return record


def environment_sha256(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    """Validate and hash one semantic environment record."""
    validate_environment(record, manifest)
    return _sha256(record)


def load_environment(path: Path, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load and validate an environment sidecar from disk."""
    return validate_environment(_load_json(path), manifest)


def build_environment_catalog(
    environments: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    """Index validated sidecars by their semantic environment digest."""
    if not environments:
        raise QualificationError("environment catalog must not be empty")
    catalog: dict[str, Mapping[str, Any]] = {}
    for environment in environments:
        validate_environment(environment, manifest)
        digest = environment_sha256(environment, manifest)
        previous = catalog.get(digest)
        if previous is not None and previous != environment:
            raise QualificationError(f"conflicting environment digest: {digest}")
        catalog[digest] = environment
    return validate_environment_catalog(catalog, manifest)


def validate_environment_catalog(
    catalog: Any, manifest: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    """Validate catalogue keys against the semantic digest of each sidecar."""
    catalog = _require_object(catalog, "environment catalog")
    if not catalog:
        raise QualificationError("environment catalog must not be empty")
    validated: dict[str, Mapping[str, Any]] = {}
    for digest, raw_environment in catalog.items():
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise QualificationError("environment catalog key is not a SHA-256")
        environment = validate_environment(raw_environment, manifest)
        semantic_digest = environment_sha256(environment, manifest)
        if digest != semantic_digest:
            raise QualificationError(
                f"environment catalog key {digest} is a forged semantic digest"
            )
        validated[digest] = environment
    return validated


def load_environment_catalog(
    paths: Sequence[Path], manifest: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    """Load one or more dependency-profile sidecars into a validated catalogue."""
    if not paths:
        raise QualificationError("at least one environment sidecar is required")
    return build_environment_catalog(
        [load_environment(path, manifest) for path in paths], manifest
    )


def _effective_dependency_profile(plan_entry: Mapping[str, Any]) -> str:
    return plan_entry["scenario"].get("dependency_profile", DEFAULT_DEPENDENCY_PROFILE)


@lru_cache(maxsize=None)
def derive_numpyro_chain_keys(
    sampler_seed: int, chains: int
) -> tuple[tuple[int, int], ...]:
    """Mirror PyMC's exact NumPyro PRNGKey/split contract without sampling."""
    if not _is_int(sampler_seed) or not 0 < sampler_seed < 2**31:
        raise QualificationError("sampler_seed must be a positive 31-bit integer")
    if not _is_int(chains) or chains <= 0:
        raise QualificationError("chains must be a positive integer")
    try:
        import jax
    except ImportError as error:
        raise QualificationError(
            "cannot validate NumPyro effective chain keys because JAX is unavailable"
        ) from error
    key = jax.random.PRNGKey(sampler_seed)
    keys = jax.random.split(key, chains) if chains > 1 else key[None, :]
    raw_keys = jax.device_get(keys).tolist()
    if any(len(row) != 2 for row in raw_keys):  # pragma: no cover - JAX invariant
        raise QualificationError(
            "NumPyro effective chain keys must contain two uint32s"
        )
    return tuple((int(row[0]), int(row[1])) for row in raw_keys)


def _environment_for_profile(
    catalog: Mapping[str, Mapping[str, Any]],
    dependency_profile: str,
) -> Mapping[str, Any]:
    matching = [
        environment
        for environment in catalog.values()
        if environment["dependency_profile"] == dependency_profile
    ]
    if not matching:
        raise QualificationError(
            "environment catalog lacks dependency profile "
            f"{dependency_profile}; missing results cannot be attributed"
        )
    if len(matching) != 1:
        raise QualificationError(
            "environment catalog has multiple sidecars for dependency profile "
            f"{dependency_profile}; missing results cannot be attributed uniquely"
        )
    return matching[0]


def _environment_for_result(
    record: Mapping[str, Any],
    plan_entry: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    provenance = _require_object(record["provenance"], "result.provenance")
    digest = provenance.get("environment_sha256")
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise QualificationError(
            "result.provenance.environment_sha256 must be a SHA-256"
        )
    environment = catalog.get(digest)
    if environment is None:
        raise QualificationError(
            "result provenance environment digest is absent from the catalog"
        )
    expected_profile = _effective_dependency_profile(plan_entry)
    if environment["dependency_profile"] != expected_profile:
        raise QualificationError(
            "result dependency profile does not match its scenario: expected "
            f"{expected_profile}, found {environment['dependency_profile']}"
        )
    status = record["execution_status"]
    scenario = plan_entry["scenario"]
    if (
        status != "missing"
        and scenario["tier"] == "qualification"
        and environment["git"]["dirty"]
    ):
        raise QualificationError(
            "qualification execution evidence requires a clean git checkout"
        )
    return environment


def write_plan(
    plan: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    tier: str,
    output_dir: Path,
    dependency_profile: str = DEFAULT_DEPENDENCY_PROFILE,
) -> tuple[Path, Path, Path]:
    """Atomically write deterministic JSONL/CSV plans and environment metadata."""
    validate_plan(plan, manifest, tier)
    jsonl_path = output_dir / "plan.jsonl"
    csv_path = output_dir / "plan.csv"
    environment_path = output_dir / "environment.json"
    _atomic_write_text(jsonl_path, _jsonl(plan))
    _atomic_write_text(csv_path, _plan_csv(plan))
    environment = json.dumps(
        collect_environment(manifest, dependency_profile),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    _atomic_write_text(
        environment_path,
        f"{environment}\n",
    )
    return jsonl_path, csv_path, environment_path


def _validate_provenance(
    value: Any,
    path: str,
    plan_entry: Mapping[str, Any],
    status: str,
    environment: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    value = _require_object(value, path)
    _require_exact_keys(value, PROVENANCE_KEYS, path)
    if value["runner_version"] != RUNNER_VERSION:
        raise QualificationError(f"{path}.runner_version mismatch")
    for key in (
        "sampler",
        "device",
        "floatx",
        "git_commit",
    ):
        if not isinstance(value[key], str) or not value[key]:
            raise QualificationError(f"{path}.{key} must be a non-empty string")
    if value["sampler"] != plan_entry["scenario"]["sampler"]:
        raise QualificationError(f"{path}.sampler does not match the plan")
    if value["floatx"] != plan_entry["scenario"]["floatx"]:
        raise QualificationError(f"{path}.floatx does not match the plan")
    observed_pytensor_floatx = value["pytensor_floatx"]
    observed_jax_x64 = value["jax_enable_x64"]
    if status == "completed" and (
        observed_pytensor_floatx is None or observed_jax_x64 is None
    ):
        raise QualificationError(
            f"{path} requires observed precision for completed cells"
        )
    if observed_pytensor_floatx is not None and not isinstance(
        observed_pytensor_floatx, str
    ):
        raise QualificationError(f"{path}.pytensor_floatx must be a string or null")
    if observed_jax_x64 is not None and not isinstance(observed_jax_x64, bool):
        raise QualificationError(f"{path}.jax_enable_x64 must be boolean or null")
    if (
        observed_pytensor_floatx is not None
        and observed_pytensor_floatx != plan_entry["scenario"]["floatx"]
    ):
        raise QualificationError(
            f"{path}.pytensor_floatx does not match the planned precision"
        )
    expected_jax_x64 = plan_entry["scenario"]["floatx"] == "float64"
    if observed_jax_x64 is not None and observed_jax_x64 is not expected_jax_x64:
        raise QualificationError(
            f"{path}.jax_enable_x64 does not match the planned precision"
        )
    if (
        status != "missing"
        and value["device"] != manifest["execution_policy"]["required_device"]
    ):
        raise QualificationError(f"{path}.device does not match execution_policy")
    effective_chain_keys = value["effective_numpyro_chain_keys"]
    if plan_entry["scenario"]["sampler"] == "pymc":
        if effective_chain_keys is not None:
            raise QualificationError(
                f"{path}.effective_numpyro_chain_keys must be null for PyMC"
            )
    elif effective_chain_keys is None:
        if status == "completed":
            raise QualificationError(
                f"{path}.effective_numpyro_chain_keys is required for completed "
                "NumPyro cells"
            )
    else:
        expected_chain_keys = [
            list(key)
            for key in derive_numpyro_chain_keys(
                plan_entry["sampler_seed"], plan_entry["scenario"]["chains"]
            )
        ]
        if effective_chain_keys != expected_chain_keys:
            raise QualificationError(
                f"{path}.effective_numpyro_chain_keys does not match sampler_seed"
            )
    artifact_contracts = (
        (
            "data_artifact",
            "data_sha256",
            manifest["artifact_policy"]["data_path"]
            .replace("<data_id>", plan_entry["scenario"]["data_id"])
            .replace("<replicate>", str(plan_entry["replicate"])),
        ),
        (
            "actual_start_artifact",
            "actual_start_sha256",
            manifest["artifact_policy"]["start_path"].replace(
                "<cell_id>", plan_entry["cell_id"]
            ),
        ),
        (
            "raw_chain_artifact",
            "raw_chain_sha256",
            manifest["artifact_policy"]["chain_path"].replace(
                "<cell_id>", plan_entry["cell_id"]
            ),
        ),
    )
    for artifact_field, digest_field, expected_path in artifact_contracts:
        artifact = value[artifact_field]
        artifact_digest = value[digest_field]
        if artifact is not None:
            if not isinstance(artifact, str) or not artifact:
                raise QualificationError(
                    f"{path}.{artifact_field} must be null or a path"
                )
            artifact_path = PurePosixPath(artifact)
            if (
                artifact_path.is_absolute()
                or ".." in artifact_path.parts
                or "\\" in artifact
                or str(artifact_path) != artifact
            ):
                raise QualificationError(
                    f"{path}.{artifact_field} must be a canonical relative path"
                )
            if artifact != expected_path:
                raise QualificationError(
                    f"{path}.{artifact_field} does not match artifact_policy"
                )
        if artifact_digest is not None and (
            not isinstance(artifact_digest, str)
            or not SHA256.fullmatch(artifact_digest)
        ):
            raise QualificationError(f"{path}.{digest_field} must be null or a SHA-256")
        if (artifact is None) != (artifact_digest is None):
            raise QualificationError(
                f"{path} {artifact_field} and {digest_field} must be provided together"
            )
        if status == "completed" and artifact is None:
            raise QualificationError(
                f"{path}.{artifact_field} is required for completed cells"
            )
    if value["environment_sha256"] != environment_sha256(environment, manifest):
        raise QualificationError(f"{path}.environment_sha256 does not match sidecar")
    if value["git_commit"] != environment["git"]["commit"]:
        raise QualificationError(f"{path}.git_commit does not match sidecar")


def _validate_failure(value: Any, status: str, path: str) -> None:
    if status == "completed":
        if value is not None:
            raise QualificationError(f"{path} must be null for completed cells")
        return
    value = _require_object(value, path)
    _require_exact_keys(value, FAILURE_KEYS, path)
    if not all(isinstance(item, str) and item for item in value.values()):
        raise QualificationError(f"{path} values must be non-empty strings")


def _validate_metric_value(name: str, value: Any, path: str) -> None:
    if name not in METRIC_DOMAINS:
        raise QualificationError(f"{path} is not a registered metric")
    domain = METRIC_DOMAINS[name]
    if domain == "boolean":
        valid = isinstance(value, bool)
    elif domain == "nonnegative_integer":
        valid = _is_int(value) and value >= 0
    elif domain == "positive_integer":
        valid = _is_int(value) and value > 0
    elif domain == "nonnegative":
        valid = _is_number(value) and math.isfinite(value) and value >= 0
    elif domain == "positive":
        valid = _is_number(value) and math.isfinite(value) and value > 0
    else:
        valid = _is_number(value) and math.isfinite(value) and 0 <= value <= 1
    if not valid:
        raise QualificationError(f"{path} violates its {domain} domain")


def _validate_unavailable_metrics(value: Any, metrics: Mapping[str, Any]) -> None:
    value = _require_object(value, "result.unavailable_metrics")
    overlap = value.keys() & metrics.keys()
    if overlap:
        raise QualificationError(
            f"metrics cannot be both available and unavailable: {sorted(overlap)}"
        )
    for name, reason in value.items():
        if name not in METRIC_DOMAINS:
            raise QualificationError(
                f"result.unavailable_metrics.{name} is not a registered metric"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise QualificationError(
                f"result.unavailable_metrics.{name} needs a non-empty reason"
            )


def _parameter_summary_family(scenario: Mapping[str, Any]) -> str | None:
    purpose = scenario["purpose"]
    if purpose in {"candidate", "control"}:
        return purpose
    return None


def _validate_result_parameter_summaries(
    value: Any,
    *,
    status: str,
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[ParameterSummary, ...]:
    if not isinstance(value, list):
        raise QualificationError("result.parameter_summaries must be a JSON array")
    scenario = plan_entry["scenario"]
    family = _parameter_summary_family(scenario)
    required = status == "completed" and scenario["recovery"] and family is not None
    if not required:
        if value:
            raise QualificationError(
                "non-recovery, diagnostic, failed, and missing cells require empty "
                "parameter_summaries"
            )
        return ()

    expected_parameters = manifest["analysis_policy"]["monitored_parameters"]
    if len(value) != len(expected_parameters):
        raise QualificationError(
            "result.parameter_summaries must contain every monitored parameter once"
        )
    calibration = scenario.get("calibration_kind") == "sbc"
    summaries = []
    for index, raw_summary in enumerate(value):
        if (
            not calibration
            and isinstance(raw_summary, Mapping)
            and SBC_SUMMARY_FIELDS.intersection(raw_summary)
        ):
            raise QualificationError(
                "fixed-recovery parameter summaries must not contain SBC ranks"
            )
        try:
            summary = validate_parameter_summary(
                raw_summary,
                require_sbc=calibration,
                expected_sbc_tie_seed=(
                    plan_entry["sbc_tie_seed"] if calibration else None
                ),
            )
        except QualificationStatisticsError as error:
            raise QualificationError(
                f"result.parameter_summaries[{index}] is invalid: {error}"
            ) from error
        if summary.family != family:
            raise QualificationError(
                f"result.parameter_summaries[{index}].family does not match the cell"
            )
        if summary.scenario_id != plan_entry["scenario_id"]:
            raise QualificationError(
                f"result.parameter_summaries[{index}].scenario_id does not match "
                "the cell"
            )
        if summary.replicate != plan_entry["replicate"]:
            raise QualificationError(
                f"result.parameter_summaries[{index}].replicate does not match the cell"
            )
        if calibration:
            if (
                summary.rank_draw_count
                != manifest["analysis_policy"]["sbc_rank_draw_count"]
            ):
                raise QualificationError(
                    f"result.parameter_summaries[{index}] changes sbc_rank_draw_count"
                )
        elif summary.has_sbc_rank:
            raise QualificationError(
                "fixed-recovery parameter summaries must not contain SBC ranks"
            )
        summaries.append(summary)
    parameter_ids = [summary.parameter_id for summary in summaries]
    if set(parameter_ids) != set(expected_parameters) or len(parameter_ids) != len(
        set(parameter_ids)
    ):
        raise QualificationError(
            "result.parameter_summaries must contain every monitored parameter once"
        )
    return tuple(summaries)


def validate_result_record(
    record: Any,
    plan_entry: Mapping[str, Any],
    environment_catalog: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    allow_missing: bool = False,
) -> Mapping[str, Any]:
    """Validate one result against the exact planned cell identity and seeds."""
    record = _require_object(record, "result")
    _require_exact_keys(record, RESULT_KEYS, "result")
    identity_fields = (
        "schema_version",
        "study_id",
        "manifest_sha256",
        "cell_id",
        "scenario_id",
        "replicate",
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
    )
    for field in identity_fields:
        if record[field] != plan_entry[field]:
            raise QualificationError(f"result.{field} does not match its planned cell")
    statuses = {"completed", "failed"}
    if allow_missing:
        statuses.add("missing")
    status = record["execution_status"]
    if status not in statuses:
        raise QualificationError(f"unsupported result execution_status: {status}")
    catalog = validate_environment_catalog(environment_catalog, manifest)
    environment = _environment_for_result(record, plan_entry, catalog)
    metrics = _require_object(record["metrics"], "result.metrics")
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise QualificationError("result.metrics contains an invalid metric name")
        _validate_metric_value(name, value, f"result.metrics.{name}")
    _assert_finite_json(metrics, "result.metrics")
    _validate_unavailable_metrics(record["unavailable_metrics"], metrics)
    _validate_result_parameter_summaries(
        record["parameter_summaries"],
        status=status,
        plan_entry=plan_entry,
        manifest=manifest,
    )
    _validate_failure(record["failure"], status, "result.failure")
    _validate_provenance(
        record["provenance"],
        "result.provenance",
        plan_entry,
        status,
        environment,
        manifest,
    )
    if status == "completed":
        count = metrics.get("divergence_count")
        draws = metrics.get("posterior_draw_count")
        rate = metrics.get("divergence_rate")
        if SAMPLER_METRICS.intersection(metrics) and draws is None:
            raise QualificationError(
                "completed sampler metrics require posterior_draw_count"
            )
        missing_raw_metrics = REQUIRED_SAMPLER_RAW_METRICS - metrics.keys()
        if missing_raw_metrics:
            raise QualificationError(
                "completed cells require raw sampler metrics: "
                f"{sorted(missing_raw_metrics)}"
            )
        if draws is not None:
            expected_draws = (
                plan_entry["scenario"]["chains"] * plan_entry["scenario"]["draws"]
            )
            if draws != expected_draws:
                raise QualificationError(
                    "posterior_draw_count must equal planned chains * draws"
                )
        if any(value is not None for value in (count, draws, rate)):
            if count is None or draws is None or rate is None:
                raise QualificationError(
                    "divergence count, draw count, and rate must be provided together"
                )
            if count > draws:
                raise QualificationError("divergence_count cannot exceed draw count")
            if not math.isclose(rate, count / draws, rel_tol=1e-12, abs_tol=1e-15):
                raise QualificationError("divergence_rate disagrees with count/draws")
    return record


def write_cell_result(
    record: Mapping[str, Any],
    plan_entry: Mapping[str, Any],
    results_dir: Path,
    environment_catalog: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> Path:
    """Validate and atomically publish one per-cell JSON result."""
    validate_result_record(record, plan_entry, environment_catalog, manifest)
    expected_directory = PurePosixPath(manifest["artifact_policy"]["cell_path"]).parts[
        0
    ]
    if results_dir.name != expected_directory:
        raise QualificationError(
            f"results directory must be named {expected_directory!r}"
        )
    verify_result_artifacts(record, results_dir.parent)
    path = results_dir / f"{plan_entry['cell_id']}.json"
    _atomic_write_text(
        path,
        f"{json.dumps(record, allow_nan=False, indent=2, sort_keys=True)}\n",
        overwrite=False,
    )
    return path


def verify_result_artifacts(record: Mapping[str, Any], artifact_root: Path) -> None:
    """Verify every referenced artifact exists and matches its exact byte digest."""
    provenance = _require_object(record.get("provenance"), "result.provenance")
    root = artifact_root.resolve()
    contracts = (
        ("data_artifact", "data_sha256"),
        ("actual_start_artifact", "actual_start_sha256"),
        ("raw_chain_artifact", "raw_chain_sha256"),
    )
    for artifact_field, digest_field in contracts:
        relative = provenance.get(artifact_field)
        expected_digest = provenance.get(digest_field)
        if relative is None:
            continue
        if not isinstance(relative, str) or not isinstance(expected_digest, str):
            raise QualificationError(
                f"result.provenance.{artifact_field} has no valid digest pair"
            )
        resolved = (root / PurePosixPath(relative)).resolve()
        if root not in resolved.parents or not resolved.is_file():
            raise QualificationError(
                f"result.provenance.{artifact_field} is unavailable below artifact root"
            )
        actual_digest = _file_sha256(resolved)
        if actual_digest != expected_digest:
            raise QualificationError(
                f"result.provenance.{digest_field} does not match artifact bytes"
            )


def load_cell_results(results_dir: Path) -> list[Mapping[str, Any]]:
    """Load strict JSON cell records from a dedicated results directory."""
    if not results_dir.is_dir():
        raise QualificationError(f"results directory does not exist: {results_dir}")
    records = []
    for path in sorted(results_dir.glob("*.json")):
        record = _load_json(path)
        records.append(_require_object(record, str(path)))
    return records


def _missing_result(
    entry: Mapping[str, Any],
    environment_catalog: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    environment = _environment_for_profile(
        environment_catalog, _effective_dependency_profile(entry)
    )
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
        "start_seeds": entry["start_seeds"],
        "sampler_seed": entry["sampler_seed"],
        "sbc_draw_seed": entry["sbc_draw_seed"],
        "sbc_tie_seed": entry["sbc_tie_seed"],
        "chain_seeds": entry["chain_seeds"],
        "execution_status": "missing",
        "metrics": {},
        "unavailable_metrics": {},
        "parameter_summaries": [],
        "failure": {
            "stage": "collection",
            "error_type": "MissingResult",
            "message": "no result was published for this frozen plan cell",
        },
        "provenance": {
            "runner_version": RUNNER_VERSION,
            "sampler": entry["scenario"]["sampler"],
            "device": "unknown",
            "floatx": entry["scenario"]["floatx"],
            "pytensor_floatx": None,
            "jax_enable_x64": None,
            "data_artifact": None,
            "data_sha256": None,
            "effective_numpyro_chain_keys": None,
            "actual_start_artifact": None,
            "actual_start_sha256": None,
            "raw_chain_artifact": None,
            "raw_chain_sha256": None,
            "git_commit": environment["git"]["commit"],
            "environment_sha256": environment_sha256(environment, manifest),
        },
    }


def aggregate_results(
    plan: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    tier: str,
    environment_catalog: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Order cell results by plan and materialize explicit missing rows."""
    validate_plan(plan, manifest, tier)
    catalog = validate_environment_catalog(environment_catalog, manifest)
    expected = {entry["cell_id"]: entry for entry in plan}
    indexed: dict[str, Mapping[str, Any]] = {}
    for raw_record in records:
        record = _require_object(raw_record, "result")
        cell_id = record.get("cell_id")
        if not isinstance(cell_id, str) or cell_id not in expected:
            raise QualificationError(f"result references unplanned cell: {cell_id}")
        if cell_id in indexed:
            raise QualificationError(f"duplicate result for cell: {cell_id}")
        validate_result_record(record, expected[cell_id], catalog, manifest)
        indexed[cell_id] = record
    aggregate = []
    for entry in plan:
        selected_record = indexed.get(entry["cell_id"])
        if selected_record is None:
            selected_record = _missing_result(entry, catalog, manifest)
        aggregate.append(dict(selected_record))
    for entry, record in zip(plan, aggregate, strict=True):
        validate_result_record(record, entry, catalog, manifest, allow_missing=True)
    return aggregate


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    """Load non-empty strict JSON objects from a JSONL file."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise QualificationError(f"cannot read {path}: {error}") from error
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        record = strict_json_loads(line, source=f"{path}:{line_number}")
        records.append(_require_object(record, f"{path}:{line_number}"))
    return records


def _result_csv(records: Sequence[Mapping[str, Any]]) -> str:
    metric_names = sorted(
        {name for record in records for name in record["metrics"].keys()}
    )
    fieldnames = [
        "cell_id",
        "scenario_id",
        "replicate",
        "execution_status",
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
        "unavailable_metrics",
        "failure_stage",
        "failure_type",
        "failure_message",
        *metric_names,
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        failure = record["failure"] or {}
        writer.writerow(
            {
                "cell_id": record["cell_id"],
                "scenario_id": record["scenario_id"],
                "replicate": record["replicate"],
                "execution_status": record["execution_status"],
                "data_seed": record["data_seed"],
                "truth_seed": record["truth_seed"],
                "group_seed": record["group_seed"],
                "observation_seed": record["observation_seed"],
                "initialization_seed": record["initialization_seed"],
                "start_seeds": ";".join(map(str, record["start_seeds"])),
                "sampler_seed": record["sampler_seed"],
                "sbc_draw_seed": record["sbc_draw_seed"],
                "sbc_tie_seed": record["sbc_tie_seed"],
                "chain_seeds": ";".join(map(str, record["chain_seeds"])),
                "unavailable_metrics": _canonical_json(record["unavailable_metrics"]),
                "failure_stage": failure.get("stage", ""),
                "failure_type": failure.get("error_type", ""),
                "failure_message": failure.get("message", ""),
                **record["metrics"],
            }
        )
    return stream.getvalue()


def write_aggregate(
    records: Sequence[Mapping[str, Any]], output_dir: Path
) -> tuple[Path, Path]:
    """Atomically write deterministic aggregate JSONL and CSV files."""
    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    _atomic_write_text(jsonl_path, _jsonl(records))
    _atomic_write_text(csv_path, _result_csv(records))
    return jsonl_path, csv_path


def compare_threshold(actual: Any, condition: Mapping[str, Any]) -> bool:
    """Apply one predeclared comparator with exact boundary semantics."""
    _validate_threshold(condition, "condition")
    expected = condition["value"]
    comparator = condition["comparator"]
    if isinstance(expected, bool):
        return isinstance(actual, bool) and comparator == "eq" and actual is expected
    if not _is_number(actual) or not math.isfinite(actual):
        return False
    if comparator == "eq":
        return actual == expected
    if comparator == "lt":
        return actual < expected
    if comparator == "le":
        return actual <= expected
    if comparator == "gt":
        return actual > expected
    return actual >= expected


def _make_check(
    *,
    scope: str,
    metric: str,
    actual: Any,
    condition: Mapping[str, Any],
    cell_id: str | None = None,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "metric": metric,
        "cell_id": cell_id,
        "scenario_id": scenario_id,
        "actual": actual,
        "comparator": condition["comparator"],
        "threshold": condition["value"],
        "passed": compare_threshold(actual, condition),
    }


def _evaluate_metric_map(
    record: Mapping[str, Any],
    conditions: Mapping[str, Mapping[str, Any]],
    scope: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    checks = []
    missing = []
    for metric, condition in conditions.items():
        if metric not in record["metrics"]:
            if metric not in record["unavailable_metrics"]:
                missing.append(f"{record['cell_id']}:{metric}")
            continue
        checks.append(
            _make_check(
                scope=scope,
                metric=metric,
                actual=record["metrics"][metric],
                condition=condition,
                cell_id=record["cell_id"],
                scenario_id=record["scenario_id"],
            )
        )
    return checks, missing


def _evaluate_paired_efficiency(
    records: Sequence[Mapping[str, Any]],
    scenarios: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
    missing_metrics: list[str],
    blockers: list[str],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    by_cell = {
        (record["scenario_id"], record["replicate"]): record for record in records
    }
    for candidate in scenarios.values():
        if (
            candidate["tier"] != "qualification"
            or candidate["purpose"] != "candidate"
            or candidate.get("calibration_kind") is not None
        ):
            continue
        paired_ratios: dict[str, list[float]] = {
            metric: [] for metric in PAIRED_EFFICIENCY_METRICS
        }
        for replicate in range(candidate["replicates"]):
            candidate_record = by_cell[(candidate["scenario_id"], replicate)]
            control_record = by_cell[(candidate["control_id"], replicate)]
            if (
                candidate_record["execution_status"] != "completed"
                or control_record["execution_status"] != "completed"
            ):
                continue
            for derived_metric, (
                raw_metric,
                direction,
            ) in PAIRED_EFFICIENCY_METRICS.items():
                raw_missing = False
                for record in (candidate_record, control_record):
                    if raw_metric in record["metrics"]:
                        continue
                    raw_missing = True
                    if raw_metric not in record["unavailable_metrics"]:
                        missing_metrics.append(f"{record['cell_id']}:{raw_metric}")
                if raw_missing:
                    continue
                candidate_value = candidate_record["metrics"][raw_metric]
                control_value = control_record["metrics"][raw_metric]
                if direction == "control_over_candidate":
                    ratio = control_value / candidate_value
                else:
                    ratio = candidate_value / control_value
                paired_ratios[derived_metric].append(ratio)
                pair_check = _make_check(
                    scope="control_pair_immediate",
                    metric=derived_metric,
                    actual=ratio,
                    condition=policy["immediate_no_go"][derived_metric],
                    cell_id=candidate_record["cell_id"],
                    scenario_id=candidate["scenario_id"],
                )
                pair_check["control_cell_id"] = control_record["cell_id"]
                checks.append(pair_check)
                if not pair_check["passed"]:
                    blockers.append(
                        f"{candidate_record['cell_id']}:{derived_metric}:control-pair"
                    )
        for derived_metric, ratios in paired_ratios.items():
            if len(ratios) != candidate["replicates"]:
                continue
            check = _make_check(
                scope="control_paired_median",
                metric=derived_metric,
                actual=median(ratios),
                condition=policy["control_paired_fit"][derived_metric],
                scenario_id=candidate["scenario_id"],
            )
            check["paired_replicates"] = len(ratios)
            checks.append(check)
    return checks


def _applicable_gradient_contracts(
    record: Mapping[str, Any],
    scenario: Mapping[str, Any],
    analysis_policy: Mapping[str, Any],
) -> tuple[str, ...]:
    contract = analysis_policy["gradient_contract"]
    evaluation = contract["evaluation"]
    if (
        scenario["tier"] not in evaluation["tiers"]
        or record["replicate"] != evaluation["scenario_replicate"]
        or (
            scenario.get("posterior_pair_id") is not None
            and scenario["sampler"] != evaluation["posterior_pair_owner_sampler"]
        )
    ):
        return ()
    contract_names = ["finite_difference", "pytensor_jax"]
    if scenario["layer"] in evaluation["bambi_isomorphism_layers"]:
        contract_names.append("bambi_isomorphism")
    if scenario["layer"] == "hssm" and scenario["model"] == "lba2_b":
        contract_names.append("likelihood_pytensor_jax")
    return tuple(contract_names)


def _gradient_contract_conditions(
    record: Mapping[str, Any],
    scenario: Mapping[str, Any],
    analysis_policy: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    contract_names = _applicable_gradient_contracts(record, scenario, analysis_policy)
    conditions: dict[str, Mapping[str, Any]] = {}
    for contract_name in contract_names:
        for metric in GRADIENT_CONTRACT_GATE_METRICS[contract_name]:
            conditions[metric] = {
                "comparator": "le",
                "value": 1.0,
            }
    return conditions


def _gradient_contract_required_metrics(
    record: Mapping[str, Any],
    scenario: Mapping[str, Any],
    analysis_policy: Mapping[str, Any],
) -> set[str]:
    required: set[str] = set()
    for contract_name in _applicable_gradient_contracts(
        record, scenario, analysis_policy
    ):
        required.update(GRADIENT_CONTRACT_GATE_METRICS[contract_name])
        required.update(GRADIENT_CONTRACT_METRICS[contract_name].values())
    return required


def _evaluate_gradient_contract(
    record: Mapping[str, Any],
    scenario: Mapping[str, Any],
    analysis_policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    conditions = _gradient_contract_conditions(record, scenario, analysis_policy)
    checks, missing = _evaluate_metric_map(record, conditions, "gradient_contract")
    descriptive_metrics = (
        _gradient_contract_required_metrics(record, scenario, analysis_policy)
        - conditions.keys()
    )
    missing.extend(
        f"{record['cell_id']}:{metric}"
        for metric in sorted(descriptive_metrics)
        if metric not in record["metrics"]
        and metric not in record["unavailable_metrics"]
    )
    for check in checks:
        metric = check["metric"]
        if metric.startswith("bambi_isomorphism_"):
            failure_class = "bambi-isomorphism-contract"
        elif metric.startswith("likelihood_pytensor_jax_"):
            failure_class = "likelihood/backend-contract"
        else:
            failure_class = "prior-gradient-contract"
        check["failure_class"] = failure_class
    return checks, missing


def _parameter_summaries_from_record(
    record: Mapping[str, Any], *, require_sbc: bool = False
) -> tuple[ParameterSummary, ...]:
    return tuple(
        validate_parameter_summary(
            summary,
            require_sbc=require_sbc,
            expected_sbc_tie_seed=(record["sbc_tie_seed"] if require_sbc else None),
        )
        for summary in record["parameter_summaries"]
    )


def _bias_checks(
    summaries: Sequence[ParameterSummary],
    *,
    family: Literal["candidate", "control"],
    expected_replicates: int,
    expected_units: Sequence[tuple[str, str]],
    scope: str,
    analysis_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results = evaluate_bias_family(
        summaries,
        family=family,
        expected_replicates=expected_replicates,
        expected_units=expected_units,
        bias_limit=analysis_policy["fixed_recovery_abs_mean_standardized_error_max"],
        familywise_alpha=analysis_policy["familywise_alpha"],
    )
    checks = []
    for result in results:
        check = {
            "scope": scope,
            "metric": "abs_mean_standardized_error",
            "cell_id": None,
            "scenario_id": result.scenario_id,
            "parameter_id": result.parameter_id,
            "family": result.family,
            "actual": result.abs_mean_standardized_error,
            "comparator": "le",
            "threshold": analysis_policy[
                "fixed_recovery_abs_mean_standardized_error_max"
            ],
            "passed": result.magnitude_passed,
            "mean_standardized_error": result.mean_standardized_error,
            "median_standardized_error": result.median_standardized_error,
            "standardized_rmse": result.standardized_rmse,
            "sign_test_pvalue": result.sign_test_pvalue,
            "holm_rejected": result.holm_rejected,
            "sign_test_role": "descriptive-only",
        }
        checks.append(check)
    return checks


def _evaluate_recovery_statistics(
    records: Sequence[Mapping[str, Any]],
    scenarios: Mapping[str, Mapping[str, Any]],
    analysis_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for family in ("candidate", "control"):
        family_records = [
            record
            for record in records
            if scenarios[record["scenario_id"]]["purpose"] == family
            and scenarios[record["scenario_id"]]["recovery"]
            and scenarios[record["scenario_id"]].get("calibration_kind") is None
        ]
        if not family_records or any(
            record["execution_status"] != "completed" for record in family_records
        ):
            continue
        replicate_counts = {
            scenarios[record["scenario_id"]]["replicates"] for record in family_records
        }
        if len(replicate_counts) != 1:
            raise QualificationError(
                f"fixed-recovery {family} scenarios change replicate count"
            )
        summaries = tuple(
            summary
            for record in family_records
            for summary in _parameter_summaries_from_record(record)
        )
        expected_units = tuple(
            (scenario["scenario_id"], parameter_id)
            for scenario in scenarios.values()
            if scenario["tier"] == "qualification"
            and scenario["purpose"] == family
            and scenario["recovery"]
            and scenario.get("calibration_kind") is None
            for parameter_id in analysis_policy["monitored_parameters"]
        )
        try:
            checks.extend(
                _bias_checks(
                    summaries,
                    family=family,
                    expected_replicates=replicate_counts.pop(),
                    expected_units=expected_units,
                    scope="fixed_recovery_bias",
                    analysis_policy=analysis_policy,
                )
            )
        except QualificationStatisticsError as error:
            raise QualificationError(
                f"fixed-recovery {family} statistics are invalid: {error}"
            ) from error

    calibration_records = [
        record
        for record in records
        if scenarios[record["scenario_id"]].get("calibration_kind") == "sbc"
    ]
    if not calibration_records or any(
        record["execution_status"] != "completed" for record in calibration_records
    ):
        return checks
    calibration_replicates = {
        scenarios[record["scenario_id"]]["replicates"] for record in calibration_records
    }
    if calibration_replicates != {analysis_policy["sbc_replicates"]}:
        raise QualificationError("calibration replicate count changes analysis policy")
    calibration_summaries = tuple(
        summary
        for record in calibration_records
        for summary in _parameter_summaries_from_record(record, require_sbc=True)
    )
    calibration_scenario_ids = analysis_policy["coverage_power_design"][
        "candidate_scenario_ids"
    ]
    expected_calibration_units = tuple(
        (scenario_id, parameter_id)
        for scenario_id in calibration_scenario_ids
        for parameter_id in analysis_policy["monitored_parameters"]
    )
    try:
        coverage = evaluate_coverage_family(
            calibration_summaries,
            family="candidate",
            expected_replicates=analysis_policy["sbc_replicates"],
            expected_units=expected_calibration_units,
            familywise_alpha=analysis_policy["familywise_alpha"],
        )
        ranks = evaluate_sbc_rank_family(
            calibration_summaries,
            family="candidate",
            expected_replicates=analysis_policy["sbc_replicates"],
            expected_units=expected_calibration_units,
            familywise_alpha=analysis_policy["familywise_alpha"],
        )
        checks.extend(
            _bias_checks(
                calibration_summaries,
                family="candidate",
                expected_replicates=analysis_policy["sbc_replicates"],
                expected_units=expected_calibration_units,
                scope="calibration_bias",
                analysis_policy=analysis_policy,
            )
        )
    except QualificationStatisticsError as error:
        raise QualificationError(
            f"calibration statistics are invalid: {error}"
        ) from error
    for coverage_result in coverage:
        checks.append(
            {
                "scope": "calibration_coverage",
                "metric": f"coverage_{int(coverage_result.nominal * 100)}pct",
                "cell_id": None,
                "scenario_id": coverage_result.scenario_id,
                "parameter_id": coverage_result.parameter_id,
                "family": coverage_result.family,
                "actual": coverage_result.successes / coverage_result.replicates,
                "comparator": "clopper_pearson_contains",
                "threshold": coverage_result.nominal,
                "passed": coverage_result.passed,
                "successes": coverage_result.successes,
                "replicates": coverage_result.replicates,
                "family_comparisons": coverage_result.family_comparisons,
                "alpha_per_comparison": coverage_result.alpha_per_comparison,
                "interval": [
                    coverage_result.interval.lower,
                    coverage_result.interval.upper,
                ],
            }
        )
    for rank_result in ranks:
        checks.append(
            {
                "scope": "calibration_sbc_rank",
                "metric": "sbc_rank_ecdf_max_abs",
                "cell_id": None,
                "scenario_id": rank_result.scenario_id,
                "parameter_id": rank_result.parameter_id,
                "family": rank_result.family,
                "actual": rank_result.max_abs_deviation,
                "comparator": "le",
                "threshold": rank_result.epsilon,
                "passed": rank_result.passed,
                "replicates": rank_result.replicates,
                "rank_draw_count": rank_result.rank_draw_count,
                "family_curves": rank_result.family_curves,
            }
        )
    return checks


def _evaluate_backend_pairs(
    records: Sequence[Mapping[str, Any]],
    scenarios: Mapping[str, Mapping[str, Any]],
    analysis_policy: Mapping[str, Any],
    pending_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_pair: dict[str, list[Mapping[str, Any]]] = {}
    for scenario in scenarios.values():
        pair_id = scenario.get("posterior_pair_id")
        if scenario["tier"] == "qualification" and pair_id is not None:
            by_pair.setdefault(pair_id, []).append(scenario)
    by_cell = {
        (record["scenario_id"], record["replicate"]): record for record in records
    }
    checks = []
    for pair_id, pair in sorted(by_pair.items()):
        pending_evidence.append(
            {
                "scope": "backend_pair",
                "posterior_pair_id": pair_id,
                "metric": "backend_combined_rank_rhat_max",
                "comparator": "lt",
                "threshold": analysis_policy["backend_combined_rank_rhat_max"],
                "reason": "pending raw-chain runner artifact",
            }
        )
        left_scenario, right_scenario = sorted(pair, key=lambda item: item["sampler"])
        for replicate in range(left_scenario["replicates"]):
            left = by_cell[(left_scenario["scenario_id"], replicate)]
            right = by_cell[(right_scenario["scenario_id"], replicate)]
            if (
                left["execution_status"] != "completed"
                or right["execution_status"] != "completed"
            ):
                continue
            left_summaries = {
                summary.parameter_id: summary
                for summary in _parameter_summaries_from_record(left)
            }
            right_summaries = {
                summary.parameter_id: summary
                for summary in _parameter_summaries_from_record(right)
            }
            for parameter_id in sorted(left_summaries):
                try:
                    result = paired_backend_mean_check(
                        left_summaries[parameter_id],
                        right_summaries[parameter_id],
                        limit=analysis_policy["backend_posterior_mean_mcse_z_max"],
                    )
                except QualificationStatisticsError as error:
                    raise QualificationError(
                        f"backend pair {pair_id} is invalid: {error}"
                    ) from error
                checks.append(
                    {
                        "scope": "backend_pair",
                        "metric": "posterior_mean_mcse_z",
                        "cell_id": left["cell_id"],
                        "paired_cell_id": right["cell_id"],
                        "scenario_id": left["scenario_id"],
                        "posterior_pair_id": pair_id,
                        "parameter_id": parameter_id,
                        "family": result.family,
                        "actual": result.mcse_z,
                        "comparator": "le",
                        "threshold": result.limit,
                        "passed": result.passed,
                    }
                )
    return checks


def assess_results(
    records: Sequence[Mapping[str, Any]],
    plan: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    tier: str,
    environment_catalog: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen gate without allowing diagnostics to alter it."""
    validate_plan(plan, manifest, tier)
    catalog = validate_environment_catalog(environment_catalog, manifest)
    if len(records) != len(plan):
        raise QualificationError("assessment requires one aggregate row per plan cell")
    by_scenario = {
        scenario["scenario_id"]: scenario for scenario in manifest["scenarios"]
    }
    checked_records = []
    for record, entry in zip(records, plan, strict=True):
        validate_result_record(record, entry, catalog, manifest, allow_missing=True)
        checked_records.append(record)

    missing_cells = [
        record["cell_id"]
        for record in checked_records
        if record["execution_status"] == "missing"
    ]
    failed_cells = [
        record["cell_id"]
        for record in checked_records
        if record["execution_status"] == "failed"
    ]
    completed = [
        record
        for record in checked_records
        if record["execution_status"] == "completed"
    ]
    checks: list[dict[str, Any]] = []
    missing_metrics: list[str] = []
    analysis_policy = manifest["analysis_policy"]
    unavailable_metrics = []
    blocking_unavailable_metrics = []
    for record in checked_records:
        scenario = by_scenario[record["scenario_id"]]
        required_contract_metrics = _gradient_contract_required_metrics(
            record, scenario, analysis_policy
        )
        for metric, reason in sorted(record["unavailable_metrics"].items()):
            item = {
                "cell_id": record["cell_id"],
                "metric": metric,
                "reason": reason,
            }
            unavailable_metrics.append(item)
            scenario_level_nonowner = (
                metric in GRADIENT_CONTRACT_ALL_METRICS
                and metric not in required_contract_metrics
                and reason == SCENARIO_LEVEL_CONTRACT_REASON
            )
            if not scenario_level_nonowner:
                blocking_unavailable_metrics.append(item)
    blockers: list[str] = []
    pending_evidence: list[dict[str, Any]] = []

    if tier == "smoke":
        conditions = manifest["thresholds"]["screening"]["per_fit"]
        for record in completed:
            cell_checks, absent = _evaluate_metric_map(record, conditions, "per_fit")
            checks.extend(cell_checks)
            missing_metrics.extend(absent)
            contract_checks, absent = _evaluate_gradient_contract(
                record,
                by_scenario[record["scenario_id"]],
                analysis_policy,
            )
            checks.extend(contract_checks)
            missing_metrics.extend(absent)
    elif tier == "qualification":
        policy = manifest["thresholds"]["qualification"]
        fit_pass: dict[str, bool] = {}
        for record in completed:
            scenario = by_scenario[record["scenario_id"]]
            cell_checks, absent = _evaluate_metric_map(
                record, policy["per_fit"], "per_fit"
            )
            cell_absent = list(absent)
            checks.extend(cell_checks)
            missing_metrics.extend(absent)
            contract_checks, absent = _evaluate_gradient_contract(
                record, scenario, analysis_policy
            )
            checks.extend(contract_checks)
            missing_metrics.extend(absent)
            for metric in ("divergence_count", "posterior_draw_count"):
                if metric not in record["metrics"]:
                    if metric not in record["unavailable_metrics"]:
                        missing_metrics.append(f"{record['cell_id']}:{metric}")
                    cell_absent.append(metric)
            if scenario["canonical"]:
                extra, absent = _evaluate_metric_map(
                    record, policy["canonical_fit"], "canonical_fit"
                )
                checks.extend(extra)
                cell_checks.extend(extra)
                missing_metrics.extend(absent)
                cell_absent.extend(absent)
            fit_pass[record["cell_id"]] = (
                all(check["passed"] for check in cell_checks) and not cell_absent
            )

            immediate_conditions = dict(policy["immediate_no_go"])
            for derived_metric in PAIRED_EFFICIENCY_METRICS:
                immediate_conditions.pop(derived_metric)
            immediate, absent = _evaluate_metric_map(
                record, immediate_conditions, "immediate_no_go"
            )
            missing_metrics.extend(absent)
            blockers.extend(
                f"{record['cell_id']}:{check['metric']}"
                for check in immediate
                if not check["passed"]
            )

        checks.extend(
            _evaluate_paired_efficiency(
                checked_records,
                by_scenario,
                policy,
                missing_metrics,
                blockers,
            )
        )
        checks.extend(
            _evaluate_recovery_statistics(checked_records, by_scenario, analysis_policy)
        )
        checks.extend(
            _evaluate_backend_pairs(
                checked_records,
                by_scenario,
                analysis_policy,
                pending_evidence,
            )
        )

        repeated = policy["repeated_no_go"]
        for scenario_id in sorted({record["scenario_id"] for record in completed}):
            scenario_records = [
                record for record in completed if record["scenario_id"] == scenario_id
            ]
            for metric, condition in repeated["conditions"].items():
                violations = sum(
                    metric in record["metrics"]
                    and not compare_threshold(record["metrics"][metric], condition)
                    for record in scenario_records
                )
                if violations >= repeated["minimum_failing_fits"]:
                    blockers.append(f"{scenario_id}:repeated:{metric}")

        if (
            not missing_cells
            and not failed_cells
            and not missing_metrics
            and not blocking_unavailable_metrics
        ):
            noncanonical = [
                record
                for record in completed
                if not by_scenario[record["scenario_id"]]["canonical"]
            ]
            total_draws = sum(
                record["metrics"]["posterior_draw_count"] for record in noncanonical
            )
            total_divergences = sum(
                record["metrics"]["divergence_count"] for record in noncanonical
            )
            failed_counts = Counter(
                record["scenario_id"]
                for record in completed
                if not fit_pass[record["cell_id"]]
            )
            aggregate_values = {
                "noncanonical_divergence_rate": total_divergences / total_draws,
                "passing_fit_fraction": sum(fit_pass.values()) / len(plan),
                "max_failed_fit_count_per_scenario": max(
                    failed_counts.values(), default=0
                ),
            }
            for metric, condition in policy["primary_aggregate"].items():
                checks.append(
                    _make_check(
                        scope="primary_aggregate",
                        metric=metric,
                        actual=aggregate_values[metric],
                        condition=condition,
                    )
                )

    else:
        conditions = manifest["thresholds"]["diagnostic"]["per_fit"]
        for record in completed:
            cell_checks, absent = _evaluate_metric_map(
                record, conditions, "diagnostic_per_fit"
            )
            checks.extend(cell_checks)
            missing_metrics.extend(absent)

    failed_checks = [check for check in checks if not check["passed"]]
    incomplete = bool(
        missing_cells
        or missing_metrics
        or blocking_unavailable_metrics
        or pending_evidence
    )
    failed = bool(failed_cells or failed_checks or blockers)
    if tier == "qualification":
        outcome = "fail" if failed else "incomplete" if incomplete else "pass"
    elif tier == "smoke":
        outcome = (
            "screening-fail"
            if failed
            else "incomplete"
            if incomplete
            else "screening-pass"
        )
    else:
        outcome = (
            "diagnostic-failed"
            if failed
            else "incomplete"
            if incomplete
            else "diagnostic-complete"
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "study_id": manifest["study_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "tier": tier,
        "qualifies_default": tier == "qualification" and outcome == "pass",
        "outcome": outcome,
        "counts": {
            "planned": len(plan),
            "completed": len(completed),
            "failed": len(failed_cells),
            "missing": len(missing_cells),
            "checks": len(checks),
            "failed_checks": len(failed_checks),
        },
        "missing_cells": missing_cells,
        "failed_cells": failed_cells,
        "missing_metrics": sorted(set(missing_metrics)),
        "unavailable_metrics": unavailable_metrics,
        "pending_evidence": pending_evidence,
        "blockers": sorted(set(blockers)),
        "checks": checks,
    }
    _assert_finite_json(result, "assessment")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("validate", help="validate the frozen manifest")
    plan = commands.add_parser("plan", help="emit a deterministic no-sampling plan")
    plan.add_argument("--tier", choices=sorted(ALLOWED_TIERS), required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument(
        "--dependency-profile",
        choices=("current-resolved", "bambi-0.19"),
        default=DEFAULT_DEPENDENCY_PROFILE,
    )

    aggregate = commands.add_parser(
        "aggregate", help="order cell JSON and materialize missing rows"
    )
    aggregate.add_argument("--tier", choices=sorted(ALLOWED_TIERS), required=True)
    aggregate.add_argument("--results-dir", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--environment", type=Path, action="append", required=True)

    assess = commands.add_parser("assess", help="apply the frozen decision rule")
    assess.add_argument("--tier", choices=sorted(ALLOWED_TIERS), required=True)
    assess.add_argument("--results", type=Path, required=True)
    assess.add_argument("--environment", type=Path, action="append", required=True)
    assess.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the no-sampling qualification harness CLI."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "validate":
            summary = {
                "study_id": manifest["study_id"],
                "manifest_sha256": manifest_sha256(manifest),
                "status": manifest["status"],
                "scenarios": len(manifest["scenarios"]),
            }
            print(_canonical_json(summary))
            return 0
        plan = expand_plan(manifest, args.tier)
        if args.command == "plan":
            plan_paths = write_plan(
                plan,
                manifest,
                args.tier,
                args.output_dir,
                args.dependency_profile,
            )
            print("\n".join(map(str, plan_paths)))
            return 0
        environment_catalog = load_environment_catalog(args.environment, manifest)
        if args.command == "aggregate":
            records = load_cell_results(args.results_dir)
            aggregate = aggregate_results(
                plan, records, manifest, args.tier, environment_catalog
            )
            expected_directory = PurePosixPath(
                manifest["artifact_policy"]["cell_path"]
            ).parts[0]
            if args.results_dir.name != expected_directory:
                raise QualificationError(
                    f"results directory must be named {expected_directory!r}"
                )
            for record in aggregate:
                verify_result_artifacts(record, args.results_dir.parent)
            aggregate_paths = write_aggregate(aggregate, args.output_dir)
            print("\n".join(map(str, aggregate_paths)))
            return 0
        records = load_jsonl(args.results)
        assessment = assess_results(
            records, plan, manifest, args.tier, environment_catalog
        )
        rendered = (
            f"{json.dumps(assessment, allow_nan=False, indent=2, sort_keys=True)}\n"
        )
        if args.output is None:
            print(rendered, end="")
        else:
            _atomic_write_text(args.output, rendered)
            print(args.output)
        successful_outcomes = {"pass", "screening-pass", "diagnostic-complete"}
        return 0 if assessment["outcome"] in successful_outcomes else 1
    except QualificationError as error:
        parser.exit(2, f"qualification contract error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

"""Authenticate and verify the archived fixed-PSDM compact evidence.

The three JSON files contain producer summaries, not raw datasets, traces, or
predictive draws. This network-free verifier reads each file once, authenticates
its bytes before parsing, and recomputes every reportable compact quantity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Never

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = Path("benchmarks/specs/jeam_fixed_psdm_recovery_v1.json")
ADDENDUM_PATH = Path("benchmarks/specs/jeam_fixed_psdm_recovery_v1_addendum.json")
RESULT_PATH = Path("benchmarks/results/jeam_fixed_psdm_recovery_v1.json")
PINS = {
    SPEC_PATH: "2a9fabe13e612a59f7c2138e4e36ae4e01d4bde5e226c16c8572d5ebe3594198",
    ADDENDUM_PATH: "42d4627f7e4eadd9c1ba656095cb3edf5af17d04bd03ad11ede476f78149d8f1",
    RESULT_PATH: "cede87d5a5a2c9789939b66962ebb025b270a13966aa2d657d5b0cbb95e9c2c4",
}
PARAMETER_ORDER = ("v_x", "v_y", "a", "t")
SCENARIO_ORDER = (
    "baseline_asymmetric",
    "reverse_axial_weak_radial",
    "high_threshold_strong_radial",
    "low_threshold_balanced_drift",
)
HISTORICAL_JEAM_REVISION = "1d7112757d8b2d27a31437255fc679194d39ab89"
CURRENT_SAFETY_JEAM_REVISION = "ede7a4f4faf226e4dae52c84dfb01012939cccdc"
RESULT_COMMIT = "d76f995d501603cc56f895e5fa429ce2be14e468"
CANONICAL_START = "2026-08-17T04:15:03.738704+00:00"
EXPECTED_DATA_HASHES = (
    "5b39ad8f2453871a15f574437b1b62d20372476e814019caa9e028f85c0f9726",
    "e2228ec7b121758e5a1599cdda54018caea95940caaa33cc10ae4beafebcba82",
    "a598a0c5f76e54c4019ccefa027714c47f13525668ef3e2ef328a2cb13f21cc7",
    "bba3bae04b0cc329586f9bce82a14aaacf51117c5009f3a5e01942c205857cc8",
)
EXPECTED_TRACE_HASHES = (
    "6bda44d1412175eab0531bf36a199af79133535bc23e6fc259f3d55343ce91cf",
    "cf2c244998de9dc284cdcdbb0c747e249d90ce77cd2f7036e5a6caad6dc05c6e",
    "ca4070fef701cc011ce78155763da76444f2d642f1178ddf5c6b6b8274565f68",
    "048a07fda29a3e25d9125449fc8ae8c1fe59688da8daf68a8343a2e0a66b99fc",
)
EXPECTED_FAILURES = (
    "high_threshold_strong_radial/a: R-hat gate failed",
    "high_threshold_strong_radial/a: bulk ESS gate failed",
    "high_threshold_strong_radial/a: MCSE/SD gate failed",
    "high_threshold_strong_radial/t: R-hat gate failed",
    "high_threshold_strong_radial/t: bulk ESS gate failed",
    "high_threshold_strong_radial/t: MCSE/SD gate failed",
    "low_threshold_balanced_drift/v_y: optimizer recovery error 0.631521 exceeds 0.45",
    "v_y: optimizer RMSE gate failed",
)
EXPECTED_BOUNDARY = {
    "dataset_hashes_recorded": True,
    "trace_hashes_recorded": True,
    "raw_datasets_retained_in_git": 0,
    "raw_traces_retained_in_git": 0,
    "raw_prior_predictive_draws_retained": False,
    "raw_posterior_predictive_draws_retained": False,
    "independent_raw_reverification": "blocked",
    "fixed_psdm_public_support_or_promotion": "blocked",
}


class EvidenceIntegrityError(ValueError):
    """Raised when archived evidence differs from its authenticated contract."""


def _fail(message: str) -> Never:
    raise EvidenceIntegrityError(message)


def _require(condition: object, message: str) -> None:
    if not condition:
        _fail(message)


def _reject_constant(token: str) -> Never:
    _fail(f"Non-finite JSON constant {token!r} is forbidden.")


def _finite_float(token: str) -> float:
    value = float(token)
    _require(math.isfinite(value), f"Non-finite JSON number {token!r} is forbidden.")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON member {key!r} is forbidden.")
        result[key] = value
    return result


def _load_authenticated_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Read once, hash before parsing, and return one strict JSON object."""
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise EvidenceIntegrityError(f"Cannot snapshot {path}.") from error
    observed = hashlib.sha256(payload).hexdigest()
    _require(
        observed == expected_sha256,
        f"SHA256 mismatch for {path.name}: expected {expected_sha256}, "
        f"observed {observed}.",
    )
    try:
        value = json.loads(
            payload,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            object_pairs_hook=_unique_object,
        )
    except EvidenceIntegrityError:
        raise
    except (TypeError, ValueError) as error:
        raise EvidenceIntegrityError(f"Invalid JSON in {path.name}.") from error
    _require(isinstance(value, dict), f"{path.name} must contain a JSON object.")
    return value


def _close(observed: float, expected: float, label: str) -> None:
    _require(
        math.isfinite(observed)
        and math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"Derived value mismatch for {label}.",
    )


def _close_list(
    observed: Sequence[float], expected: Sequence[float], label: str
) -> None:
    _require(len(observed) == len(expected), f"Derived sequence mismatch for {label}.")
    for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
        _close(left, right, f"{label}[{index}]")


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _rmse(errors: Sequence[float]) -> float:
    return math.sqrt(_mean([error * error for error in errors]))


def _add_failure(
    records: list[dict[str, Any]],
    message: str,
    category: str,
    scenario: str | None = None,
    parameter: str | None = None,
) -> None:
    records.append(
        {
            "order": len(records) + 1,
            "message": message,
            "category": category,
            "scenario": scenario,
            "parameter": parameter,
        }
    )


def _verify_core(
    spec: Mapping[str, Any],
    addendum: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> None:
    """Bind the pinned bytes to the historical scope and provenance."""
    study = "jeam-fixed-psdm-recovery-v1"
    _require(
        (spec["study_id"], addendum["study_id"], artifact["study_id"])
        == (study, study, study)
        and (
            spec["schema_version"],
            addendum["schema_version"],
            artifact["schema_version"],
        )
        == (1, 1, 1),
        "Fixed-PSDM study identity or schema changed.",
    )
    scope = spec["scope"]
    _require(
        artifact["canonical"] is True
        and artifact["model"] == scope["model"] == "projected_spherical_diffusion"
        and tuple(artifact["parameter_order"]) == PARAMETER_ORDER
        and tuple(scope["parameter_order"]) == PARAMETER_ORDER
        and artifact["spec_path"] == SPEC_PATH.as_posix()
        and artifact["spec_sha256"] == PINS[SPEC_PATH]
        and addendum["scope"]["formula_or_regression_support_evaluated"] is False,
        "Compact result identity or scalar model scope changed.",
    )
    outcome = addendum["known_v1_outcome"]
    _require(
        addendum["immutable_protocol"]["sha256"] == PINS[SPEC_PATH]
        and outcome["result_commit"] == RESULT_COMMIT
        and outcome["result_sha256"] == PINS[RESULT_PATH]
        and outcome["canonical_started_at_utc"] == CANONICAL_START
        and outcome["overall_pass"] is False
        and (outcome["truth_in_hdi"], outcome["truth_total"]) == (14, 16)
        and addendum["evidence_boundary"] == EXPECTED_BOUNDARY,
        "Archived outcome or compact-only boundary changed.",
    )
    provenance = artifact["provenance"]
    current = addendum["current_safety_revision"]
    _require(
        spec["provenance"]["jeam_revision"] == HISTORICAL_JEAM_REVISION
        and addendum["historical_execution"]["jeam_revision"]
        == HISTORICAL_JEAM_REVISION
        and provenance["jeam_revision"] == HISTORICAL_JEAM_REVISION
        and provenance["hssm_revision"] == "ebbd68ee6dcaad644505ae7f3739b7b1f0ba3794"
        and provenance["spec_commit"] == "c1c68ef3c0ebdf78b4a950c4e62e61bee55b0961"
        and current
        == {
            "jeam_revision": CURRENT_SAFETY_JEAM_REVISION,
            "v1_recovery_rerun": False,
        }
        and artifact["started_at_utc"] == CANONICAL_START,
        "Historical or current-safety provenance changed.",
    )
    runner = addendum["runner_archive"]
    _require(
        "fixed-budget differential-evolution endpoints"
        in runner["optimizer_interpretation"]
        and "not demonstrated converged optima or MLEs"
        in runner["optimizer_interpretation"]
        and "same historical JEAM producer" in runner["objective_parity_interpretation"]
        and runner["ordered_slice_identity_independently_authenticated"] is False
        and runner["runtime_hssm_import_bound_to_recorded_checkout"] is False
        and addendum["successor_policy"]
        == {
            "v2a_and_v2b_require_new_preregistrations": True,
            "mixing_and_identifiability_are_hypotheses_not_v1_conclusions": True,
        },
        "Archival interpretation or successor boundary changed.",
    )
    _require(
        tuple(row["name"] for row in spec["scenarios"]) == SCENARIO_ORDER
        and tuple(row["scenario"] for row in artifact["scenarios"]) == SCENARIO_ORDER,
        "Scenario order or membership changed.",
    )


def _derive(
    spec: Mapping[str, Any],
    addendum: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the stable report records and exact scientific-gate failures."""
    limits = spec["scientific_acceptance"]
    opt_limits = limits["optimizer_recovery"]
    post_limits = limits["posterior_recovery"]
    pred_limits = limits["posterior_predictive"]
    prior_limits = spec["preflight_gates"]["prior_predictive"]
    spec_scenarios = {row["name"]: row for row in spec["scenarios"]}
    parameters: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    objectives: list[dict[str, Any]] = []
    predictives: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(artifact["scenarios"]):
        name = scenario["scenario"]
        direct = scenario["optimizer"]["direct_jeam"]
        compiled = scenario["optimizer"]["compiled_hssm"]
        _require(
            scenario["status"] == "completed"
            and scenario["smoke"] is False
            and scenario["trials"] == 400
            and scenario["truth"] == spec_scenarios[name]["truth"]
            and scenario["data"]["sha256"] == EXPECTED_DATA_HASHES[scenario_index]
            and scenario["data"]["shape"] == [400, 2]
            and scenario["data"]["dtype"] == "float64"
            and scenario["trace"]["sha256"] == EXPECTED_TRACE_HASHES[scenario_index]
            and scenario["trace"]["saved_before_summary"] is True
            and set(scenario["sampler_diagnostics"]["sample_stats"])
            == {"nstep_in", "nstep_out"}
            and (direct["iterations"], direct["evaluations"]) == (20, 1260)
            and (compiled["iterations"], compiled["evaluations"]) == (20, 1260),
            f"Scenario execution or compact metadata changed: {name}.",
        )

        prior = scenario["prior_predictive"]
        prior_passed = all(
            (
                prior["all_values_finite"],
                prior["all_rt_strictly_positive"],
                prior["all_angles_in_closed_domain"],
            )
        ) and all(
            prior_limits["rt_quantile_ratio_to_observed_lower"]
            <= ratio
            <= prior_limits["rt_quantile_ratio_to_observed_upper"]
            for ratio in prior["prior_to_observed_ratios"]
        )
        _require(prior_passed == prior["passed"], f"Prior gate changed: {name}.")
        if not prior_passed:
            _add_failure(
                failures, f"{name}: prior predictive gate failed", "prior", name
            )

        scenario_objectives = []
        _require(
            len(scenario["objective_candidates"]) == 3,
            f"Objective grid changed: {name}.",
        )
        for candidate_index, candidate in enumerate(
            scenario["objective_candidates"], 1
        ):
            error = abs(candidate["direct_jeam"] - candidate["compiled_hssm"])
            _close(
                error,
                candidate["absolute_error"],
                f"{name} objective {candidate_index}",
            )
            scenario_objectives.append(error)
            objectives.append(
                {
                    "scenario": name,
                    "candidate": candidate_index,
                    "direct_jeam": candidate["direct_jeam"],
                    "compiled_hssm": candidate["compiled_hssm"],
                    "absolute_error": error,
                    "absolute_error_limit": limits["maximum_objective_absolute_error"],
                    "passed": error <= limits["maximum_objective_absolute_error"],
                }
            )
        maximum_objective_error = max(scenario_objectives)
        _close(
            maximum_objective_error,
            scenario["maximum_objective_absolute_error"],
            f"{name} maximum objective error",
        )
        objective_passed = (
            maximum_objective_error <= limits["maximum_objective_absolute_error"]
        )
        if not objective_passed:
            _add_failure(
                failures, f"{name}: objective parity failed", "objective", name
            )

        parameter_parity_error = max(
            abs(left - right)
            for left, right in zip(
                direct["parameters"], compiled["parameters"], strict=True
            )
        )
        optimizer_objective_error = abs(direct["objective"] - compiled["objective"])
        recovery_errors = [
            abs(endpoint - truth)
            for endpoint, truth in zip(
                direct["parameters"], scenario["truth"], strict=True
            )
        ]
        optimizer = scenario["optimizer"]
        _close(
            parameter_parity_error,
            optimizer["maximum_parameter_absolute_error"],
            f"{name} optimizer parameters",
        )
        _close(
            optimizer_objective_error,
            optimizer["objective_absolute_error"],
            f"{name} optimizer objective",
        )
        _close_list(
            optimizer["direct_recovery_absolute_error"],
            recovery_errors,
            f"{name} optimizer recovery",
        )
        parameter_parity_passed = (
            parameter_parity_error
            <= limits["maximum_optimizer_parameter_absolute_error"]
        )
        fit_parity_passed = (
            optimizer_objective_error
            <= limits["maximum_optimizer_objective_absolute_error"]
        )
        if not parameter_parity_passed:
            _add_failure(
                failures,
                f"{name}: optimizer parameter parity failed",
                "optimizer_parameter_parity",
                name,
            )
        if not fit_parity_passed:
            _add_failure(
                failures,
                f"{name}: optimizer objective parity failed",
                "optimizer_objective_parity",
                name,
            )

        _require(
            tuple(row["name"] for row in scenario["parameters"]) == PARAMETER_ORDER,
            f"Parameter order changed: {name}.",
        )
        scenario_hdi = 0
        for parameter_index, row in enumerate(scenario["parameters"]):
            parameter = row["name"]
            _require(
                row["truth"] == scenario["truth"][parameter_index]
                and row["optimizer_estimate"] == direct["parameters"][parameter_index],
                f"Parameter vectors disagree: {name}/{parameter}.",
            )
            truth_in_hdi = row["hdi_lower"] <= row["truth"] <= row["hdi_upper"]
            _require(
                truth_in_hdi == row["truth_in_hdi"],
                f"Stored HDI inclusion changed: {name}/{parameter}.",
            )
            scenario_hdi += truth_in_hdi
            optimizer_passed = (
                recovery_errors[parameter_index]
                <= opt_limits["maximum_absolute_error"][parameter]
            )
            rhat_passed = row["rhat"] < post_limits["maximum_rhat_exclusive"]
            bulk_passed = row["ess_bulk"] > post_limits["minimum_bulk_ess_exclusive"]
            tail_passed = row["ess_tail"] > post_limits["minimum_tail_ess_exclusive"]
            mcse_passed = (
                row["mcse_over_posterior_sd"]
                < post_limits["maximum_mcse_over_posterior_sd_exclusive"]
            )
            if not optimizer_passed:
                limit = opt_limits["maximum_absolute_error"][parameter]
                _add_failure(
                    failures,
                    f"{name}/{parameter}: optimizer recovery error "
                    f"{recovery_errors[parameter_index]:.6g} exceeds {limit:.6g}",
                    "optimizer_absolute_error",
                    name,
                    parameter,
                )
            for passed, message, category in (
                (rhat_passed, "R-hat gate failed", "rhat"),
                (bulk_passed, "bulk ESS gate failed", "bulk_ess"),
                (tail_passed, "tail ESS gate failed", "tail_ess"),
                (mcse_passed, "MCSE/SD gate failed", "mcse_over_posterior_sd"),
            ):
                if not passed:
                    _add_failure(
                        failures,
                        f"{name}/{parameter}: {message}",
                        category,
                        name,
                        parameter,
                    )
            parameters.append(
                {
                    "scenario": name,
                    "parameter": parameter,
                    "truth": row["truth"],
                    "optimizer_endpoint": row["optimizer_estimate"],
                    "optimizer_absolute_error": recovery_errors[parameter_index],
                    "optimizer_absolute_error_limit": opt_limits[
                        "maximum_absolute_error"
                    ][parameter],
                    "optimizer_recovery_passed": optimizer_passed,
                    "posterior_mean": row["posterior_mean"],
                    "posterior_sd": row["posterior_sd"],
                    "posterior_absolute_error": abs(
                        row["posterior_mean"] - row["truth"]
                    ),
                    "hdi_lower": row["hdi_lower"],
                    "hdi_upper": row["hdi_upper"],
                    "truth_in_hdi": truth_in_hdi,
                    "rhat": row["rhat"],
                    "rhat_limit": post_limits["maximum_rhat_exclusive"],
                    "rhat_passed": rhat_passed,
                    "ess_bulk": row["ess_bulk"],
                    "bulk_ess_limit": post_limits["minimum_bulk_ess_exclusive"],
                    "bulk_ess_passed": bulk_passed,
                    "ess_tail": row["ess_tail"],
                    "tail_ess_limit": post_limits["minimum_tail_ess_exclusive"],
                    "tail_ess_passed": tail_passed,
                    "mcse_over_posterior_sd": row["mcse_over_posterior_sd"],
                    "mcse_over_posterior_sd_limit": post_limits[
                        "maximum_mcse_over_posterior_sd_exclusive"
                    ],
                    "mcse_passed": mcse_passed,
                    "diagnostics_passed": all(
                        (rhat_passed, bulk_passed, tail_passed, mcse_passed)
                    ),
                }
            )

        predictive = scenario["posterior_predictive"]
        rt_errors = [
            abs(left - right)
            for left, right in zip(
                predictive["predictive_rt_quantiles"],
                predictive["observed_rt_quantiles"],
                strict=True,
            )
        ]
        polar_errors = [
            abs(left - right)
            for left, right in zip(
                predictive["predictive_mean_polar_unit_vector"],
                predictive["observed_mean_polar_unit_vector"],
                strict=True,
            )
        ]
        _close_list(
            predictive["rt_quantile_absolute_errors"],
            rt_errors,
            f"{name} predictive RT",
        )
        _close_list(
            predictive["polar_unit_vector_component_absolute_errors"],
            polar_errors,
            f"{name} predictive polar",
        )
        maximum_rt_error, maximum_polar_error = max(rt_errors), max(polar_errors)
        predictive_passed = (
            maximum_rt_error <= pred_limits["maximum_rt_quantile_absolute_error"]
            and maximum_polar_error
            <= pred_limits["maximum_polar_unit_vector_component_absolute_error"]
        )
        if maximum_rt_error > pred_limits["maximum_rt_quantile_absolute_error"]:
            _add_failure(
                failures,
                f"{name}: predictive RT quantile gate failed",
                "predictive_rt",
                name,
            )
        if (
            maximum_polar_error
            > pred_limits["maximum_polar_unit_vector_component_absolute_error"]
        ):
            _add_failure(
                failures,
                f"{name}: predictive polar unit-vector gate failed",
                "predictive_polar",
                name,
            )
        predictives.append(
            {
                "scenario": name,
                "rt_probabilities": list(predictive["rt_probabilities"]),
                "observed_rt_quantiles": list(predictive["observed_rt_quantiles"]),
                "predictive_rt_quantiles": list(predictive["predictive_rt_quantiles"]),
                "rt_quantile_absolute_errors": rt_errors,
                "maximum_rt_quantile_absolute_error": maximum_rt_error,
                "maximum_rt_quantile_absolute_error_limit": pred_limits[
                    "maximum_rt_quantile_absolute_error"
                ],
                "observed_mean_polar_unit_vector": list(
                    predictive["observed_mean_polar_unit_vector"]
                ),
                "predictive_mean_polar_unit_vector": list(
                    predictive["predictive_mean_polar_unit_vector"]
                ),
                "polar_unit_vector_component_absolute_errors": polar_errors,
                "maximum_polar_unit_vector_component_absolute_error": (
                    maximum_polar_error
                ),
                "maximum_polar_unit_vector_component_absolute_error_limit": pred_limits[
                    "maximum_polar_unit_vector_component_absolute_error"
                ],
                "passed": predictive_passed,
                "evidence_scope": "authenticated producer summaries; raw draws absent",
            }
        )
        runtime, diagnostics = (
            scenario["runtime_seconds"],
            scenario["sampler_diagnostics"],
        )
        scenarios.append(
            {
                "scenario": name,
                "trials": scenario["trials"],
                "status": scenario["status"],
                "prior_predictive_passed": prior_passed,
                "maximum_objective_absolute_error": maximum_objective_error,
                "objective_absolute_error_limit": limits[
                    "maximum_objective_absolute_error"
                ],
                "objective_parity_passed": objective_passed,
                "maximum_optimizer_parameter_absolute_error": parameter_parity_error,
                "optimizer_parameter_absolute_error_limit": limits[
                    "maximum_optimizer_parameter_absolute_error"
                ],
                "optimizer_parameter_parity_passed": parameter_parity_passed,
                "optimizer_objective_absolute_error": optimizer_objective_error,
                "optimizer_objective_absolute_error_limit": limits[
                    "maximum_optimizer_objective_absolute_error"
                ],
                "optimizer_objective_parity_passed": fit_parity_passed,
                "optimizer_iterations": direct["iterations"],
                "optimizer_evaluations": direct["evaluations"],
                "hdi_inclusions": scenario_hdi,
                "hdi_total": len(PARAMETER_ORDER),
                "data_sha256": scenario["data"]["sha256"],
                "trace_sha256": scenario["trace"]["sha256"],
                "trace_bytes": scenario["trace"]["bytes"],
                "mean_nstep_in": diagnostics["mean_nstep_in"],
                "mean_nstep_out": diagnostics["mean_nstep_out"],
                "sampling_seconds": runtime["sampling"],
                "total_seconds": runtime["total"],
                "timing_scope": "descriptive recorded-machine telemetry only",
            }
        )

    aggregates: list[dict[str, Any]] = []
    _require(
        tuple(row["name"] for row in artifact["aggregate"]) == PARAMETER_ORDER,
        "Aggregate order changed.",
    )
    for parameter_index, (parameter, stored) in enumerate(
        zip(PARAMETER_ORDER, artifact["aggregate"], strict=True)
    ):
        rows = parameters[parameter_index :: len(PARAMETER_ORDER)]
        optimizer_errors = [row["optimizer_endpoint"] - row["truth"] for row in rows]
        posterior_errors = [row["posterior_mean"] - row["truth"] for row in rows]
        values = {
            "optimizer_bias": _mean(optimizer_errors),
            "optimizer_rmse": _rmse(optimizer_errors),
            "posterior_bias": _mean(posterior_errors),
            "posterior_rmse": _rmse(posterior_errors),
            "hdi_coverage": _mean([float(row["truth_in_hdi"]) for row in rows]),
            "maximum_rhat": max(row["rhat"] for row in rows),
            "minimum_bulk_ess": min(row["ess_bulk"] for row in rows),
            "minimum_tail_ess": min(row["ess_tail"] for row in rows),
            "maximum_mcse_over_posterior_sd": max(
                row["mcse_over_posterior_sd"] for row in rows
            ),
        }
        for key, value in values.items():
            _close(value, stored[key], f"{parameter} aggregate {key}")
        optimizer_passed = (
            values["optimizer_rmse"] <= opt_limits["maximum_rmse"][parameter]
        )
        bias_passed = (
            abs(values["posterior_bias"])
            <= post_limits["maximum_absolute_bias"][parameter]
        )
        posterior_passed = (
            values["posterior_rmse"] <= post_limits["maximum_rmse"][parameter]
        )
        coverage_passed = (
            values["hdi_coverage"]
            >= post_limits["minimum_94_percent_hdi_coverage_per_parameter"]
        )
        diagnostics_passed = (
            values["maximum_rhat"] < post_limits["maximum_rhat_exclusive"]
            and values["minimum_bulk_ess"] > post_limits["minimum_bulk_ess_exclusive"]
            and values["minimum_tail_ess"] > post_limits["minimum_tail_ess_exclusive"]
            and values["maximum_mcse_over_posterior_sd"]
            < post_limits["maximum_mcse_over_posterior_sd_exclusive"]
        )
        for passed, message, category in (
            (optimizer_passed, "optimizer RMSE gate failed", "optimizer_rmse"),
            (bias_passed, "posterior bias gate failed", "posterior_bias"),
            (posterior_passed, "posterior RMSE gate failed", "posterior_rmse"),
            (coverage_passed, "HDI coverage gate failed", "hdi_coverage"),
        ):
            if not passed:
                _add_failure(
                    failures, f"{parameter}: {message}", category, parameter=parameter
                )
        aggregates.append(
            {
                "parameter": parameter,
                "scenarios": len(rows),
                **values,
                "optimizer_rmse_limit": opt_limits["maximum_rmse"][parameter],
                "optimizer_rmse_passed": optimizer_passed,
                "posterior_absolute_bias_limit": post_limits["maximum_absolute_bias"][
                    parameter
                ],
                "posterior_bias_passed": bias_passed,
                "posterior_rmse_limit": post_limits["maximum_rmse"][parameter],
                "posterior_rmse_passed": posterior_passed,
                "hdi_coverage_limit": post_limits[
                    "minimum_94_percent_hdi_coverage_per_parameter"
                ],
                "hdi_coverage_passed": coverage_passed,
                "diagnostics_passed": diagnostics_passed,
            }
        )

    truth_in_hdi, truth_total = (
        sum(row["truth_in_hdi"] for row in parameters),
        len(parameters),
    )
    coverage = truth_in_hdi / truth_total
    _close(coverage, artifact["overall_hdi_coverage"], "overall HDI coverage")
    if coverage < post_limits["minimum_overall_94_percent_hdi_coverage"]:
        _add_failure(
            failures, "overall HDI coverage gate failed", "overall_hdi_coverage"
        )
    messages = tuple(row["message"] for row in failures)
    _require(messages == EXPECTED_FAILURES, "Recomputed scientific failures changed.")
    _require(
        artifact["gate"]
        == {"evaluated": True, "passed": False, "failures": list(messages)},
        "Stored gate disagrees with recomputation.",
    )

    runner = addendum["runner_archive"]
    summary = {
        "study_id": artifact["study_id"],
        "model": artifact["model"],
        "canonical": artifact["canonical"],
        "scenario_count": len(scenarios),
        "parameter_count": len(PARAMETER_ORDER),
        "unique_truth_count": len(
            {(row["scenario"], row["parameter"]) for row in parameters}
        ),
        "truth_in_hdi": truth_in_hdi,
        "truth_total": truth_total,
        "hdi_coverage": coverage,
        "failure_count": len(failures),
        "overall_pass": not failures,
        "ecosystem_promotion_blocked": True,
        "maximum_rhat": max(row["rhat"] for row in parameters),
        "minimum_bulk_ess": min(row["ess_bulk"] for row in parameters),
        "minimum_tail_ess": min(row["ess_tail"] for row in parameters),
        "maximum_mcse_over_posterior_sd": max(
            row["mcse_over_posterior_sd"] for row in parameters
        ),
        "maximum_objective_absolute_error": max(
            row["absolute_error"] for row in objectives
        ),
        "total_runtime_seconds": artifact["total_runtime_seconds"],
        "optimizer_endpoint_label": "fixed-budget DE endpoint (not MLE)",
        "evidence_class": "authenticated compact producer summaries only",
    }
    boundary = {
        "evidence_class": summary["evidence_class"],
        "retention": {
            **EXPECTED_BOUNDARY,
            "sampler_backend_trace_attributes_retained": False,
            "historical_uv_lock_bytes_retained": False,
        },
        "independent_raw_reverification": "blocked",
        "ecosystem_promotion_blocked": True,
        "public_support_or_promotion": "blocked",
        "promotion_blockers": [
            "the overall frozen v1 recovery gate failed",
            "raw datasets and traces are absent",
            "raw prior- and posterior-predictive draws are absent",
            "ordered-Slice identity and backend trace attributes are not authenticated",
            "runtime HSSM source is not bound to the recorded checkout",
            "historical uv.lock bytes are absent",
            "the current JEAM safety revision was not rerun",
        ],
        "optimizer_interpretation": runner["optimizer_interpretation"],
        "objective_parity_interpretation": runner["objective_parity_interpretation"],
        "ordered_slice_identity_independently_authenticated": False,
        "runtime_hssm_import_bound_to_recorded_checkout": False,
        "telemetry_interpretation": (
            "descriptive recorded-machine observations; never a pass/fail criterion"
        ),
        "successor_interpretation": (
            "mixing and identifiability are v2a/v2b hypotheses, not v1 conclusions"
        ),
    }
    provenance = {
        "spec_sha256": PINS[SPEC_PATH],
        "addendum_sha256": PINS[ADDENDUM_PATH],
        "result_sha256": PINS[RESULT_PATH],
        "result_commit": RESULT_COMMIT,
        "hssm_revision": artifact["provenance"]["hssm_revision"],
        "spec_commit": artifact["provenance"]["spec_commit"],
        "historical_jeam_revision": HISTORICAL_JEAM_REVISION,
        "current_safety_jeam_revision": CURRENT_SAFETY_JEAM_REVISION,
        "current_safety_revision_rerun": False,
        "python_version": artifact["provenance"]["python_version"],
        "package_versions": dict(artifact["provenance"]["package_versions"]),
        "canonical_started_at_utc": artifact["started_at_utc"],
        "generated_at_utc": artifact["generated_at_utc"],
    }
    return {
        "study_id": artifact["study_id"],
        "summary": summary,
        "parameter_records": parameters,
        "scenario_records": scenarios,
        "aggregate_records": aggregates,
        "objective_records": objectives,
        "predictive_records": predictives,
        "failure_records": failures,
        "evidence_boundary": boundary,
        "provenance": provenance,
    }


def verify_psdm_recovery_documents(
    spec: Mapping[str, Any],
    addendum: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify authenticated documents and derive their compact report records."""
    try:
        _verify_core(spec, addendum, artifact)
        return _derive(spec, addendum, artifact)
    except EvidenceIntegrityError:
        raise
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise EvidenceIntegrityError(
            "Malformed fixed-PSDM compact evidence."
        ) from error


def load_verified_psdm_recovery_evidence(
    root: str | Path = REPO_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load one authenticated snapshot per frozen file and verify its science."""
    repository = Path(root)
    spec = _load_authenticated_json(repository / SPEC_PATH, PINS[SPEC_PATH])
    addendum = _load_authenticated_json(repository / ADDENDUM_PATH, PINS[ADDENDUM_PATH])
    artifact = _load_authenticated_json(repository / RESULT_PATH, PINS[RESULT_PATH])
    return (
        verify_psdm_recovery_documents(spec, addendum, artifact),
        spec,
        addendum,
        artifact,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the archive and emit a compact summary without tracebacks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        verification, *_ = load_verified_psdm_recovery_evidence(args.root)
    except EvidenceIntegrityError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(verification["summary"], sort_keys=True))
    return 0


__all__ = [
    "ADDENDUM_PATH",
    "CURRENT_SAFETY_JEAM_REVISION",
    "EXPECTED_FAILURES",
    "EvidenceIntegrityError",
    "HISTORICAL_JEAM_REVISION",
    "PARAMETER_ORDER",
    "PINS",
    "REPO_ROOT",
    "RESULT_PATH",
    "SCENARIO_ORDER",
    "SPEC_PATH",
    "load_verified_psdm_recovery_evidence",
    "main",
    "verify_psdm_recovery_documents",
]


if __name__ == "__main__":
    raise SystemExit(main())

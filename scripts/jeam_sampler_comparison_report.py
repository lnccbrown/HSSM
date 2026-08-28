"""Verifier-backed reporting data for the fixed-CDM sampler study."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scripts.verify_jeam_sampler_comparison_evidence import (
    REPO_ROOT,
    load_verified_sampler_report_evidence,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

SAMPLER_ORDER = ("slice", "pymc_nuts", "numpyro_nuts")
SAMPLER_LABELS = {
    "slice": "Blackbox + PyMC Slice",
    "pymc_nuts": "Analytical + PyMC NUTS",
    "numpyro_nuts": "Analytical + NumPyro NUTS",
}
SAMPLER_COLORS = {
    "slice": "#7F8C8D",
    "pymc_nuts": "#4472C4",
    "numpyro_nuts": "#E67E22",
}
REVISION_ROLES = {
    "historical_analytical_result",
    "durable_blackbox_reference",
    "current_safety_revision",
}


def validate_report_contract(
    verification: Mapping[str, Any],
    spec: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> None:
    """Reject evidence that cannot support the bounded compact-only report."""
    if spec.get("study_id") != "jeam-fixed-cdm-sampler-comparison-v1":
        raise ValueError("Unexpected fixed-CDM sampler specification.")
    if (
        artifact.get("study_id") != spec["study_id"]
        or verification.get("study_id") != spec["study_id"]
    ):
        raise ValueError("The result does not belong to the verified specification.")
    if artifact.get("status") != "canonical-complete":
        raise ValueError("The compact fixed-CDM result is not canonically complete.")

    scenario_names = [row["name"] for row in spec["scenarios"]]
    sampler_names = [row["id"] for row in spec["samplers"]]
    if artifact.get("selected_scenarios") != scenario_names:
        raise ValueError("The result does not contain the preregistered scenarios.")
    if artifact.get("selected_samplers") != sampler_names:
        raise ValueError("The result does not contain the preregistered samplers.")

    expected_fits = {
        (scenario_name, sampler_name)
        for scenario_name in scenario_names
        for sampler_name in sampler_names
    }
    observed_fits = {
        (fit["scenario"], fit["sampler"]) for fit in artifact.get("fits", [])
    }
    if observed_fits != expected_fits or len(artifact.get("fits", [])) != len(
        expected_fits
    ):
        raise ValueError(
            "The result does not contain one fit per scenario and sampler."
        )

    canonical_scenarios = {
        row["name"] for row in spec["scenarios"] if row["role"] == "canonical"
    }
    parameter_names = spec["scope"]["scientific_parameter_order"]
    unique_truth_count = len(canonical_scenarios) * len(parameter_names)
    route_rows = [
        parameter
        for fit in artifact["fits"]
        if fit["scenario"] in canonical_scenarios
        for parameter in fit["parameters"]
    ]
    counts = verification["counts"]
    if (
        counts["canonical_scenario_parameter_truths"] != unique_truth_count
        or counts["canonical_route_hdi_checks"] != len(route_rows)
        or counts["canonical_route_hdi_inclusions"]
        != sum(row["truth_in_hdi"] for row in route_rows)
    ):
        raise ValueError("The verified truth and route-HDI cardinalities disagree.")

    retention = verification["retention"]
    missing_evidence = (
        retention["raw_trace_files_retained"] == 0
        and retention["raw_prior_predictive_draws_retained"] is False
        and retention["raw_posterior_predictive_draws_retained"] is False
        and retention["sampler_backend_trace_attributes_retained"] is False
        and retention["historical_uv_lock_bytes_retained"] is False
    )
    if not missing_evidence:
        raise ValueError("The report's compact-only evidence boundary has changed.")
    if verification["ecosystem_promotion"]["blocked"] is not True:
        raise ValueError("The report must retain the ecosystem promotion block.")
    if set(verification["jeam_revisions"]) != REVISION_ROLES:
        raise ValueError("The report must distinguish all three JEAM revisions.")


def build_parameter_records(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one route-specific HDI row per canonical fit and parameter."""
    records = []
    for fit in artifact["fits"]:
        if fit["role"] != "canonical":
            continue
        for parameter in fit["parameters"]:
            records.append(
                {
                    "scenario": fit["scenario"],
                    "sampler": fit["sampler"],
                    "route": SAMPLER_LABELS[fit["sampler"]],
                    "parameter": parameter["name"],
                    "truth": parameter["truth"],
                    "posterior_mean": parameter["posterior_mean"],
                    "posterior_error": parameter["posterior_mean"] - parameter["truth"],
                    "hdi_lower": parameter["hdi_lower"],
                    "hdi_upper": parameter["hdi_upper"],
                    "relative_hdi_lower": parameter["hdi_lower"] - parameter["truth"],
                    "relative_hdi_upper": parameter["hdi_upper"] - parameter["truth"],
                    "truth_in_hdi": parameter["truth_in_hdi"],
                    "rhat": parameter["rhat"],
                    "ess_bulk": parameter["ess_bulk"],
                    "ess_tail": parameter["ess_tail"],
                    "mcse_over_posterior_sd": parameter["mcse_over_posterior_sd"],
                    "ess_bulk_per_second": parameter["ess_bulk_per_sampling_second"],
                }
            )
    return records


def build_fit_records(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Aggregate diagnostics and recorded-machine timing by fit."""
    records = []
    for fit in artifact["fits"]:
        parameters = fit["parameters"]
        runtime = fit["runtime_seconds"]
        records.append(
            {
                "scenario": fit["scenario"],
                "role": fit["role"],
                "trials": fit["trials"],
                "sampler": fit["sampler"],
                "route": SAMPLER_LABELS[fit["sampler"]],
                "likelihood": fit["likelihood"],
                "backend": fit["backend"],
                "route_hdi_inclusions": sum(
                    parameter["truth_in_hdi"] for parameter in parameters
                ),
                "route_hdi_checks": len(parameters),
                "maximum_rhat": max(parameter["rhat"] for parameter in parameters),
                "minimum_bulk_ess": min(
                    parameter["ess_bulk"] for parameter in parameters
                ),
                "minimum_tail_ess": min(
                    parameter["ess_tail"] for parameter in parameters
                ),
                "maximum_mcse_over_sd": max(
                    parameter["mcse_over_posterior_sd"] for parameter in parameters
                ),
                "minimum_bulk_ess_per_second": min(
                    parameter["ess_bulk_per_sampling_second"]
                    for parameter in parameters
                ),
                "divergences": fit["sampler_diagnostics"]["divergences"],
                "model_build_seconds": runtime["model_build"],
                "objective_compile_seconds": runtime[
                    "objective_compile_and_first_eval"
                ],
                "gradient_compile_seconds": runtime["gradient_compile_and_first_eval"],
                "sampling_seconds": runtime[
                    "sampling_call_including_backend_kernel_compilation"
                ],
                "predictive_seconds": runtime["posterior_predictive"],
                "total_seconds": runtime["total"],
                "timing_scope": "recorded-machine only",
            }
        )
    return records


def build_objective_records(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten every preregistered pointwise objective comparison."""
    records = []
    for scenario in artifact["shared_preflight"]:
        for candidate_index, candidate in enumerate(
            scenario["objective_candidates"], start=1
        ):
            records.append(
                {
                    "scenario": scenario["scenario"],
                    "candidate": candidate_index,
                    "direct_numpy": candidate["direct_numpy"],
                    "direct_jax": candidate["direct_jax"],
                    "compiled_hssm": candidate["compiled_hssm"],
                    "maximum_absolute_error": candidate["maximum_absolute_error"],
                }
            )
    return records


def build_predictive_records(
    spec: Mapping[str, Any], artifact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return canonical predictive errors normalized by preregistered limits."""
    thresholds = spec["scientific_acceptance"]
    records = []
    for fit in artifact["fits"]:
        if fit["role"] != "canonical":
            continue
        predictive = fit["predictive"]
        maximum_rt_error = max(predictive["rt_quantile_absolute_errors"])
        mean_angle_error = predictive["mean_angle_distance"]
        resultant_error = predictive["mean_resultant_length_absolute_error"]
        records.append(
            {
                "scenario": fit["scenario"],
                "sampler": fit["sampler"],
                "route": SAMPLER_LABELS[fit["sampler"]],
                "maximum_rt_quantile_error": maximum_rt_error,
                "mean_angle_error": mean_angle_error,
                "resultant_length_error": resultant_error,
                "rt_fraction_of_limit": maximum_rt_error
                / thresholds["maximum_rt_quantile_absolute_error"],
                "angle_fraction_of_limit": mean_angle_error
                / thresholds["maximum_circular_mean_angle_distance"],
                "resultant_fraction_of_limit": resultant_error
                / thresholds["maximum_mean_resultant_length_absolute_error"],
            }
        )
    return records


def build_efficiency_records(
    verification: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return efficiency ratios explicitly scoped to the recorded machine."""
    efficiency = verification["recorded_machine_efficiency"]
    records = []
    for sampler in SAMPLER_ORDER[1:]:
        row = efficiency["by_sampler"][sampler]
        records.append(
            {
                "sampler": sampler,
                "route": SAMPLER_LABELS[sampler],
                "canonical_median_minimum_bulk_ess_per_second": row[
                    "canonical_median_minimum_bulk_ess_per_second"
                ],
                "slice_canonical_median_minimum_bulk_ess_per_second": efficiency[
                    "slice_canonical_median_minimum_bulk_ess_per_second"
                ],
                "efficiency_ratio_vs_slice": row["canonical_efficiency_ratio_vs_slice"],
                "maximum_total_time_ratio_vs_slice": max(
                    row["canonical_total_seconds_ratios_vs_slice"].values()
                ),
                "total_time_ratios_vs_slice": row[
                    "canonical_total_seconds_ratios_vs_slice"
                ],
                "scale_normalized_efficiency_ratio": row[
                    "scale_normalized_efficiency_ratio_vs_reference"
                ],
                "recorded_machine_gate_passed": row["passed"],
                "evidence_scope": "recorded-machine only",
                "portable_performance_claim": False,
            }
        )
    return records


def summarize_artifact(
    artifact: Mapping[str, Any], verification: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute headline science without conflating truths and route checks."""
    canonical_fits = [fit for fit in artifact["fits"] if fit["role"] == "canonical"]
    canonical_parameters = [
        parameter for fit in canonical_fits for parameter in fit["parameters"]
    ]
    unique_truths = {
        (fit["scenario"], parameter["name"])
        for fit in canonical_fits
        for parameter in fit["parameters"]
    }
    nuts_fits = [fit for fit in artifact["fits"] if fit["sampler"] != "slice"]
    return {
        "fit_count": len(artifact["fits"]),
        "canonical_fit_count": len(canonical_fits),
        "unique_truth_count": len(unique_truths),
        "route_hdi_inclusions": sum(
            parameter["truth_in_hdi"] for parameter in canonical_parameters
        ),
        "route_hdi_checks": len(canonical_parameters),
        "nuts_divergences": sum(
            fit["sampler_diagnostics"]["divergences"] for fit in nuts_fits
        ),
        "maximum_rhat": max(parameter["rhat"] for parameter in canonical_parameters),
        "minimum_bulk_ess": min(
            parameter["ess_bulk"] for parameter in canonical_parameters
        ),
        "minimum_tail_ess": min(
            parameter["ess_tail"] for parameter in canonical_parameters
        ),
        "maximum_mcse_over_sd": max(
            parameter["mcse_over_posterior_sd"] for parameter in canonical_parameters
        ),
        "maximum_objective_absolute_error": max(
            row["maximum_absolute_error"]
            for scenario in artifact["shared_preflight"]
            for row in scenario["objective_candidates"]
        ),
        "ecosystem_promotion_blocked": verification["ecosystem_promotion"]["blocked"],
    }


def build_evidence_boundary(
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose the authenticated limitations alongside reportable claims."""
    return {
        "evidence_class": verification["evidence_class"],
        "efficiency_scope": "recorded-machine only",
        "portable_performance_claim": False,
        "jeam_revisions": dict(verification["jeam_revisions"]),
        "retention": dict(verification["retention"]),
        "ecosystem_promotion_blocked": verification["ecosystem_promotion"]["blocked"],
        "ecosystem_promotion_blockers": list(
            verification["ecosystem_promotion"]["blockers"]
        ),
    }


def load_sampler_comparison_report(
    root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build every report table from one authenticated verifier load."""
    verification, spec, artifact = load_verified_sampler_report_evidence(root)
    validate_report_contract(verification, spec, artifact)
    return {
        "verification": verification,
        "specification": spec,
        "compact_result": artifact,
        "summary": summarize_artifact(artifact, verification),
        "parameter_records": build_parameter_records(artifact),
        "fit_records": build_fit_records(artifact),
        "objective_records": build_objective_records(artifact),
        "predictive_records": build_predictive_records(spec, artifact),
        "efficiency_records": build_efficiency_records(verification),
        "evidence_boundary": build_evidence_boundary(verification),
    }


__all__ = [
    "SAMPLER_COLORS",
    "SAMPLER_LABELS",
    "SAMPLER_ORDER",
    "build_efficiency_records",
    "build_evidence_boundary",
    "build_fit_records",
    "build_objective_records",
    "build_parameter_records",
    "build_predictive_records",
    "load_sampler_comparison_report",
    "summarize_artifact",
    "validate_report_contract",
]

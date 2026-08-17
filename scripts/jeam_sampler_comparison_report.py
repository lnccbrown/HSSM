"""Dependency-light reporting helpers for the fixed-CDM sampler study."""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

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


def validate_report_contract(
    spec: Mapping[str, Any], artifact: Mapping[str, Any]
) -> None:
    """Reject a mismatched or incomplete study before constructing a report."""
    if spec.get("study_id") != "jeam-fixed-cdm-sampler-comparison-v1":
        raise ValueError("Unexpected fixed-CDM sampler specification.")
    if artifact.get("study_id") != spec["study_id"]:
        raise ValueError("The result does not belong to the loaded specification.")
    if artifact.get("status") != "canonical-complete":
        raise ValueError("The fixed-CDM sampler study is not canonically complete.")

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
    if observed_fits != expected_fits:
        raise ValueError(
            "The result does not contain one fit per scenario and sampler."
        )


def build_parameter_records(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one presentation row per canonical fit and parameter."""
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
    """Aggregate diagnostics and timing without hiding their scenario origin."""
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
                "truths_covered": sum(
                    parameter["truth_in_hdi"] for parameter in parameters
                ),
                "truth_count": len(parameters),
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


def summarize_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the headline scientific summary from raw artifact rows."""
    canonical_fits = [fit for fit in artifact["fits"] if fit["role"] == "canonical"]
    canonical_parameters = [
        parameter for fit in canonical_fits for parameter in fit["parameters"]
    ]
    nuts_fits = [fit for fit in artifact["fits"] if fit["sampler"] != "slice"]
    return {
        "fit_count": len(artifact["fits"]),
        "canonical_fit_count": len(canonical_fits),
        "canonical_truths_covered": sum(
            parameter["truth_in_hdi"] for parameter in canonical_parameters
        ),
        "canonical_truth_count": len(canonical_parameters),
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
    }


def build_promotion_records(
    spec: Mapping[str, Any], artifact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Recompute machine-local promotion metrics from individual fit rows."""
    fits = {(fit["scenario"], fit["sampler"]): fit for fit in artifact["fits"]}
    canonical_names = [
        row["name"] for row in spec["scenarios"] if row["role"] == "canonical"
    ]

    def minimum_efficiency(fit: Mapping[str, Any]) -> float:
        return min(
            parameter["ess_bulk_per_sampling_second"] for parameter in fit["parameters"]
        )

    slice_median = statistics.median(
        minimum_efficiency(fits[(name, "slice")]) for name in canonical_names
    )
    reference_name = spec["promotion_decision"]["scale_reference_scenario"]
    scale_name = next(
        row["name"] for row in spec["scenarios"] if row["role"] == "scale"
    )
    records = []
    for sampler in SAMPLER_ORDER[1:]:
        sampler_median = statistics.median(
            minimum_efficiency(fits[(name, sampler)]) for name in canonical_names
        )
        total_ratios = {
            name: fits[(name, sampler)]["runtime_seconds"]["total"]
            / fits[(name, "slice")]["runtime_seconds"]["total"]
            for name in canonical_names
        }
        scale_fit = fits[(scale_name, sampler)]
        reference_fit = fits[(reference_name, sampler)]
        scale_normalized_efficiency = (
            min(parameter["ess_bulk"] for parameter in scale_fit["parameters"])
            * scale_fit["trials"]
            / scale_fit["runtime_seconds"][
                "sampling_call_including_backend_kernel_compilation"
            ]
        )
        reference_normalized_efficiency = (
            min(parameter["ess_bulk"] for parameter in reference_fit["parameters"])
            * reference_fit["trials"]
            / reference_fit["runtime_seconds"][
                "sampling_call_including_backend_kernel_compilation"
            ]
        )
        records.append(
            {
                "sampler": sampler,
                "route": SAMPLER_LABELS[sampler],
                "canonical_median_minimum_bulk_ess_per_second": sampler_median,
                "slice_canonical_median_minimum_bulk_ess_per_second": slice_median,
                "efficiency_ratio_vs_slice": sampler_median / slice_median,
                "maximum_total_time_ratio_vs_slice": max(total_ratios.values()),
                "total_time_ratios_vs_slice": total_ratios,
                "scale_normalized_efficiency_ratio": scale_normalized_efficiency
                / reference_normalized_efficiency,
            }
        )
    return records


__all__ = [
    "SAMPLER_COLORS",
    "SAMPLER_LABELS",
    "SAMPLER_ORDER",
    "build_fit_records",
    "build_objective_records",
    "build_parameter_records",
    "build_predictive_records",
    "build_promotion_records",
    "summarize_artifact",
    "validate_report_contract",
]

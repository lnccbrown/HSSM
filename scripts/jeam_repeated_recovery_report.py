"""Build concise report views from verified JEAM repeated-recovery evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from scripts.verify_jeam_repeated_recovery_evidence import (
    DEFAULT_BUNDLE,
    load_verified_evidence,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def summarize_science(science: Mapping[str, Any]) -> dict[str, float | int | bool]:
    """Reduce verified science to the headline recovery and parity diagnostics."""
    scenarios = science["scenarios"]
    parameters = [row for scenario in scenarios for row in scenario["parameters"]]
    aggregate = science["aggregate"]
    return {
        "passed": science["gate"] == {"passed": True, "failures": []},
        "scenarios": len(scenarios),
        "hdi_inclusions": sum(row["hdi_contains_truth"] for row in parameters),
        "hdi_total": len(parameters),
        "maximum_rhat": max(row["maximum_rhat"] for row in aggregate),
        "minimum_bulk_ess": min(row["minimum_bulk_ess"] for row in aggregate),
        "minimum_tail_ess": min(row["minimum_tail_ess"] for row in aggregate),
        "maximum_mcse_sd_ratio": max(row["maximum_mcse_sd_ratio"] for row in aggregate),
        "maximum_objective_error": max(
            row["maximum_objective_absolute_error"] for row in scenarios
        ),
        "maximum_optimizer_parameter_error": max(
            row["maximum_optimizer_parameter_absolute_error"] for row in scenarios
        ),
    }


def build_report_frames(
    science: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """Create presentation frames without changing scientific values or thresholds."""
    parameter_rows = [
        {"scenario": scenario["name"], **parameter}
        for scenario in science["scenarios"]
        for parameter in scenario["parameters"]
    ]
    scenario_rows = []
    for scenario in science["scenarios"]:
        predictive = scenario["predictive"]
        prior = scenario["prior_predictive"]
        runtime = scenario["runtime"]
        scenario_rows.append(
            {
                "scenario": scenario["name"],
                "objective_error": scenario["maximum_objective_absolute_error"],
                "optimizer_parameter_error": scenario[
                    "maximum_optimizer_parameter_absolute_error"
                ],
                "optimizer_objective_error": scenario[
                    "optimizer_objective_absolute_error"
                ],
                "initial_logp": scenario["initial_logp"],
                "sampling_seconds": runtime["hssm_sampling_seconds"],
                "prior_predictive_seconds": runtime["hssm_prior_predictive_seconds"],
                "posterior_predictive_seconds": runtime["hssm_predictive_seconds"],
                "mean_steps_in": scenario["slice_diagnostics"]["mean_steps_in"],
                "mean_steps_out": scenario["slice_diagnostics"]["mean_steps_out"],
                "maximum_rt_quantile_error": float(
                    np.max(
                        np.abs(
                            np.asarray(predictive["observed_rt_quantiles"])
                            - np.asarray(predictive["predictive_rt_quantiles"])
                        )
                    )
                ),
                "mean_angle_distance": predictive["mean_angle_distance"],
                "resultant_length_error": abs(
                    predictive["observed_resultant_length"]
                    - predictive["predictive_resultant_length"]
                ),
                "minimum_prior_rt_ratio": min(prior["prior_to_observed_rt_ratios"]),
                "maximum_prior_rt_ratio": max(prior["prior_to_observed_rt_ratios"]),
            }
        )

    thresholds = science["thresholds"]
    aggregate_rows = []
    for row in science["aggregate"]:
        name = row["name"]
        bias_limit = thresholds["maximum_absolute_bias"][name]
        rmse_limit = thresholds["maximum_rmse"][name]
        aggregate_rows.append(
            {
                **row,
                "jeam_fixed_budget_bias_ratio": abs(row["jeam_fixed_budget_bias"])
                / bias_limit,
                "jeam_fixed_budget_rmse_ratio": row["jeam_fixed_budget_rmse"]
                / rmse_limit,
                "hssm_posterior_bias_ratio": abs(row["hssm_posterior_bias"])
                / bias_limit,
                "hssm_posterior_rmse_ratio": row["hssm_posterior_rmse"] / rmse_limit,
            }
        )

    return {
        "parameters": pd.DataFrame(parameter_rows),
        "scenarios": pd.DataFrame(scenario_rows),
        "aggregate": pd.DataFrame(aggregate_rows),
    }


def load_report(
    root: str | Path = DEFAULT_BUNDLE,
) -> tuple[dict[str, object], dict[str, Any], dict[str, pd.DataFrame]]:
    """Authenticate the evidence once and return its recomputed report views."""
    manifest, science = load_verified_evidence(root)
    return manifest, science, build_report_frames(science)

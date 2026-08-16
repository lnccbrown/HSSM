"""Independent regression checks for the committed JEAM recovery artifact."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ARTIFACT_PATH = (
    Path(__file__).parents[1] / "benchmarks" / "results" / "jeam_repeated_recovery.json"
)
PINNED_JEAM_REVISION = "a9f547b3630ae8ff31ccec1b904e0c02fdba6d99"
PARAMETER_ORDER = ("a", "t", "v_x", "v_y")
EXPECTED_SCENARIOS = (
    "baseline_asymmetric",
    "reversed_drift",
    "high_threshold_strong_drift",
    "low_threshold_negative_drift",
)


@pytest.fixture(scope="module")
def artifact():
    """Load the frozen result without importing HSSM or optional JEAM."""
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_artifact_records_the_predeclared_repeated_run(artifact):
    """The result should identify the exact scientific and dependency setup."""
    assert artifact["schema_version"] == 1
    assert artifact["jeam_revision"] == PINNED_JEAM_REVISION
    assert artifact["pytensor_floatx"] == "float64"
    assert artifact["parameter_order"] == list(PARAMETER_ORDER)
    assert artifact["sampler"] == "pymc.Slice[a,t,v_x,v_y]"
    assert artifact["trials_per_scenario"] == 300
    assert artifact["chains"] == 4
    assert artifact["tune"] == artifact["draws"] == 1_000
    assert artifact["predictive_draws"] == 40
    assert tuple(row["name"] for row in artifact["scenarios"]) == EXPECTED_SCENARIOS
    assert datetime.fromisoformat(artifact["generated_at_utc"]).tzinfo is not None
    assert artifact["total_runtime_seconds"] > 0.0


def test_every_scenario_preserves_the_direct_jeam_objective(artifact):
    """Re-evaluate parity fields without trusting the benchmark's stored gate."""
    thresholds = artifact["thresholds"]
    for scenario in artifact["scenarios"]:
        direct = np.asarray(scenario["direct_objectives"])
        compiled = np.asarray(scenario["compiled_objectives"])
        observed_objective_error = float(np.max(np.abs(direct - compiled)))
        direct_fit = scenario["direct_jeam_fit"]
        compiled_fit = scenario["compiled_hssm_fit"]
        observed_parameter_error = float(
            np.max(
                np.abs(
                    np.asarray(direct_fit["parameters"])
                    - np.asarray(compiled_fit["parameters"])
                )
            )
        )
        observed_fit_objective_error = abs(
            direct_fit["objective"] - compiled_fit["objective"]
        )

        assert observed_objective_error == pytest.approx(
            scenario["maximum_objective_absolute_error"]
        )
        assert observed_parameter_error == pytest.approx(
            scenario["maximum_optimizer_parameter_absolute_error"]
        )
        assert observed_fit_objective_error == pytest.approx(
            scenario["optimizer_objective_absolute_error"]
        )
        assert observed_objective_error <= thresholds["objective_absolute_error"]
        assert (
            observed_parameter_error <= thresholds["optimizer_parameter_absolute_error"]
        )
        assert (
            observed_fit_objective_error
            <= thresholds["optimizer_objective_absolute_error"]
        )


def test_aggregate_recovery_metrics_are_recomputed_and_pass(artifact):
    """Bias, RMSE, coverage, and convergence should agree with raw scenarios."""
    thresholds = artifact["thresholds"]
    aggregate = {row["name"]: row for row in artifact["aggregate"]}
    for parameter_index, name in enumerate(PARAMETER_ORDER):
        rows = [
            next(row for row in scenario["parameters"] if row["name"] == name)
            for scenario in artifact["scenarios"]
        ]
        truth = np.asarray(
            [scenario["truth"][parameter_index] for scenario in artifact["scenarios"]]
        )
        mle = np.asarray([row["jeam_mle"] for row in rows])
        posterior = np.asarray([row["posterior_mean"] for row in rows])
        coverage = np.asarray(
            [
                row["interval_lower"] <= row["truth"] <= row["interval_upper"]
                for row in rows
            ]
        )
        observed = aggregate[name]

        assert all(
            row["truth"] == pytest.approx(value)
            for row, value in zip(rows, truth, strict=True)
        )
        assert coverage.tolist() == [row["interval_covers_truth"] for row in rows]
        assert observed["jeam_mle_bias"] == pytest.approx(np.mean(mle - truth))
        assert observed["jeam_mle_rmse"] == pytest.approx(
            np.sqrt(np.mean(np.square(mle - truth)))
        )
        assert observed["hssm_posterior_bias"] == pytest.approx(
            np.mean(posterior - truth)
        )
        assert observed["hssm_posterior_rmse"] == pytest.approx(
            np.sqrt(np.mean(np.square(posterior - truth)))
        )
        assert observed["interval_coverage"] == pytest.approx(np.mean(coverage))
        assert observed["maximum_rhat"] == pytest.approx(
            max(row["rhat"] for row in rows)
        )
        assert observed["minimum_bulk_ess"] == pytest.approx(
            min(row["ess_bulk"] for row in rows)
        )
        assert observed["minimum_tail_ess"] == pytest.approx(
            min(row["ess_tail"] for row in rows)
        )
        assert observed["mean_bulk_ess_per_second"] == pytest.approx(
            np.mean([row["ess_bulk_per_second"] for row in rows])
        )

        bias_limit = thresholds["maximum_absolute_bias"][name]
        rmse_limit = thresholds["maximum_rmse"][name]
        assert abs(observed["jeam_mle_bias"]) <= bias_limit
        assert abs(observed["hssm_posterior_bias"]) <= bias_limit
        assert observed["jeam_mle_rmse"] <= rmse_limit
        assert observed["hssm_posterior_rmse"] <= rmse_limit
        assert observed["interval_coverage"] >= thresholds["minimum_interval_coverage"]
        assert observed["maximum_rhat"] <= thresholds["maximum_rhat"]
        assert observed["minimum_bulk_ess"] >= thresholds["minimum_bulk_ess"]
        assert observed["minimum_tail_ess"] >= thresholds["minimum_tail_ess"]

    assert artifact["gate"] == {"passed": True, "failures": []}

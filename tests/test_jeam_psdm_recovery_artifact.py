"""Independent regression checks for the fixed-PSDM recovery artifact."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parents[1]
SPEC_PATH = REPO_ROOT / "benchmarks" / "specs" / "jeam_fixed_psdm_recovery_v1.json"
ARTIFACT_PATH = (
    REPO_ROOT / "benchmarks" / "results" / "jeam_fixed_psdm_recovery_v1.json"
)
PARAMETER_ORDER = ("v_x", "v_y", "a", "t")
EXPECTED_SCENARIOS = (
    "baseline_asymmetric",
    "reverse_axial_weak_radial",
    "high_threshold_strong_radial",
    "low_threshold_balanced_drift",
)
EXPECTED_DATA_HASHES = {
    "baseline_asymmetric": (
        "5b39ad8f2453871a15f574437b1b62d20372476e814019caa9e028f85c0f9726"
    ),
    "reverse_axial_weak_radial": (
        "e2228ec7b121758e5a1599cdda54018caea95940caaa33cc10ae4beafebcba82"
    ),
    "high_threshold_strong_radial": (
        "a598a0c5f76e54c4019ccefa027714c47f13525668ef3e2ef328a2cb13f21cc7"
    ),
    "low_threshold_balanced_drift": (
        "bba3bae04b0cc329586f9bce82a14aaacf51117c5009f3a5e01942c205857cc8"
    ),
}
EXPECTED_TRACE_HASHES = {
    "baseline_asymmetric": (
        "6bda44d1412175eab0531bf36a199af79133535bc23e6fc259f3d55343ce91cf"
    ),
    "reverse_axial_weak_radial": (
        "cf2c244998de9dc284cdcdbb0c747e249d90ce77cd2f7036e5a6caad6dc05c6e"
    ),
    "high_threshold_strong_radial": (
        "ca4070fef701cc011ce78155763da76444f2d642f1178ddf5c6b6b8274565f68"
    ),
    "low_threshold_balanced_drift": (
        "048a07fda29a3e25d9125449fc8ae8c1fe59688da8daf68a8343a2e0a66b99fc"
    ),
}
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


@pytest.fixture(scope="module")
def spec():
    """Load the preregistered protocol without importing HSSM or JEAM."""
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifact():
    """Load the frozen compact result without importing HSSM or JEAM."""
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _rows_by_parameter(artifact, name):
    """Return the four raw scenario rows for one parameter."""
    return [
        next(row for row in scenario["parameters"] if row["name"] == name)
        for scenario in artifact["scenarios"]
    ]


def _recompute_failures(artifact, spec):
    """Evaluate frozen thresholds without trusting the stored gate."""
    acceptance = spec["scientific_acceptance"]
    posterior = acceptance["posterior_recovery"]
    predictive = acceptance["posterior_predictive"]
    failures = []
    for scenario in artifact["scenarios"]:
        name = scenario["scenario"]
        if scenario["status"] != "completed":
            failures.append(f"{name}: scenario status is {scenario['status']!r}")
            continue
        if not scenario["prior_predictive"]["passed"]:
            failures.append(f"{name}: prior predictive gate failed")
        if (
            scenario["maximum_objective_absolute_error"]
            > acceptance["maximum_objective_absolute_error"]
        ):
            failures.append(f"{name}: objective parity failed")
        optimizer = scenario["optimizer"]
        if (
            optimizer["maximum_parameter_absolute_error"]
            > acceptance["maximum_optimizer_parameter_absolute_error"]
        ):
            failures.append(f"{name}: optimizer parameter parity failed")
        if (
            optimizer["objective_absolute_error"]
            > acceptance["maximum_optimizer_objective_absolute_error"]
        ):
            failures.append(f"{name}: optimizer objective parity failed")
        for parameter, error in zip(
            PARAMETER_ORDER,
            optimizer["direct_recovery_absolute_error"],
            strict=True,
        ):
            limit = acceptance["optimizer_recovery"]["maximum_absolute_error"][
                parameter
            ]
            if error > limit:
                failures.append(
                    f"{name}/{parameter}: optimizer recovery error {error:.6g} "
                    f"exceeds {limit:.6g}"
                )
        for parameter in scenario["parameters"]:
            prefix = f"{name}/{parameter['name']}"
            if not parameter["rhat"] < posterior["maximum_rhat_exclusive"]:
                failures.append(f"{prefix}: R-hat gate failed")
            if not parameter["ess_bulk"] > posterior["minimum_bulk_ess_exclusive"]:
                failures.append(f"{prefix}: bulk ESS gate failed")
            if not parameter["ess_tail"] > posterior["minimum_tail_ess_exclusive"]:
                failures.append(f"{prefix}: tail ESS gate failed")
            if not (
                parameter["mcse_over_posterior_sd"]
                < posterior["maximum_mcse_over_posterior_sd_exclusive"]
            ):
                failures.append(f"{prefix}: MCSE/SD gate failed")
        if set(scenario["sampler_diagnostics"]["sample_stats"]) != {
            "nstep_in",
            "nstep_out",
        }:
            failures.append(f"{name}: sample statistics do not identify ordered Slice")
        if (
            max(scenario["posterior_predictive"]["rt_quantile_absolute_errors"])
            > predictive["maximum_rt_quantile_absolute_error"]
        ):
            failures.append(f"{name}: predictive RT quantile gate failed")
        if (
            max(
                scenario["posterior_predictive"][
                    "polar_unit_vector_component_absolute_errors"
                ]
            )
            > predictive["maximum_polar_unit_vector_component_absolute_error"]
        ):
            failures.append(f"{name}: predictive polar unit-vector gate failed")

    aggregate = {row["name"]: row for row in artifact["aggregate"]}
    for name in PARAMETER_ORDER:
        row = aggregate[name]
        if (
            row["optimizer_rmse"]
            > acceptance["optimizer_recovery"]["maximum_rmse"][name]
        ):
            failures.append(f"{name}: optimizer RMSE gate failed")
        if abs(row["posterior_bias"]) > posterior["maximum_absolute_bias"][name]:
            failures.append(f"{name}: posterior bias gate failed")
        if row["posterior_rmse"] > posterior["maximum_rmse"][name]:
            failures.append(f"{name}: posterior RMSE gate failed")
        if (
            row["hdi_coverage"]
            < posterior["minimum_94_percent_hdi_coverage_per_parameter"]
        ):
            failures.append(f"{name}: HDI coverage gate failed")
    if (
        artifact["overall_hdi_coverage"]
        < posterior["minimum_overall_94_percent_hdi_coverage"]
    ):
        failures.append("overall HDI coverage gate failed")
    return failures


def test_artifact_records_exact_canonical_provenance(artifact, spec):
    """The result should name committed code, data, traces, and software versions."""
    assert artifact["schema_version"] == 1
    assert artifact["study_id"] == spec["study_id"]
    assert artifact["canonical"] is True
    assert artifact["model"] == "projected_spherical_diffusion"
    assert tuple(artifact["parameter_order"]) == PARAMETER_ORDER
    assert artifact["spec_path"] == (
        "benchmarks/specs/jeam_fixed_psdm_recovery_v1.json"
    )
    assert artifact["provenance"]["hssm_revision"] == (
        "ebbd68ee6dcaad644505ae7f3739b7b1f0ba3794"
    )
    assert artifact["provenance"]["spec_commit"] == (
        "c1c68ef3c0ebdf78b4a950c4e62e61bee55b0961"
    )
    assert (
        artifact["provenance"]["jeam_revision"] == spec["provenance"]["jeam_revision"]
    )
    assert artifact["provenance"]["python_version"].startswith("3.12.")
    assert artifact["provenance"]["package_versions"]["hssm"] == "0.4.0"
    generated_at = datetime.fromisoformat(artifact["generated_at_utc"])
    frozen_at = datetime.fromisoformat(spec["frozen_at_utc"])
    assert generated_at > frozen_at
    assert artifact["total_runtime_seconds"] > 0.0

    assert tuple(row["scenario"] for row in artifact["scenarios"]) == EXPECTED_SCENARIOS
    for row in artifact["scenarios"]:
        name = row["scenario"]
        assert row["status"] == "completed"
        assert row["smoke"] is False
        assert row["data"]["sha256"] == EXPECTED_DATA_HASHES[name]
        assert row["data"]["shape"] == [400, 2]
        assert row["data"]["dtype"] == "float64"
        assert row["trace"]["sha256"] == EXPECTED_TRACE_HASHES[name]
        assert row["trace"]["saved_before_summary"] is True
        assert row["trace"]["bytes"] > 0


def test_every_scenario_preserves_direct_jeam_inside_hssm(artifact, spec):
    """Recompute objective and optimizer parity from stored raw values."""
    acceptance = spec["scientific_acceptance"]
    for scenario in artifact["scenarios"]:
        objective_errors = [
            abs(row["direct_jeam"] - row["compiled_hssm"])
            for row in scenario["objective_candidates"]
        ]
        observed_objective_error = max(objective_errors)
        direct = scenario["optimizer"]["direct_jeam"]
        compiled = scenario["optimizer"]["compiled_hssm"]
        observed_parameter_error = float(
            np.max(
                np.abs(
                    np.asarray(direct["parameters"])
                    - np.asarray(compiled["parameters"])
                )
            )
        )
        observed_fit_error = abs(direct["objective"] - compiled["objective"])

        assert observed_objective_error == pytest.approx(
            scenario["maximum_objective_absolute_error"]
        )
        assert observed_parameter_error == pytest.approx(
            scenario["optimizer"]["maximum_parameter_absolute_error"]
        )
        assert observed_fit_error == pytest.approx(
            scenario["optimizer"]["objective_absolute_error"]
        )
        assert (
            observed_objective_error <= acceptance["maximum_objective_absolute_error"]
        )
        assert (
            observed_parameter_error
            <= acceptance["maximum_optimizer_parameter_absolute_error"]
        )
        assert (
            observed_fit_error
            <= acceptance["maximum_optimizer_objective_absolute_error"]
        )


def test_aggregate_recovery_is_recomputed_from_scenarios(artifact):
    """Bias, RMSE, coverage, and diagnostics should not trust stored aggregates."""
    aggregate = {row["name"]: row for row in artifact["aggregate"]}
    all_coverage = []
    for name in PARAMETER_ORDER:
        rows = _rows_by_parameter(artifact, name)
        truth = np.asarray([row["truth"] for row in rows])
        optimizer = np.asarray([row["optimizer_estimate"] for row in rows])
        posterior = np.asarray([row["posterior_mean"] for row in rows])
        coverage = np.asarray([row["truth_in_hdi"] for row in rows])
        all_coverage.extend(coverage.tolist())
        observed = aggregate[name]

        assert observed["optimizer_bias"] == pytest.approx(np.mean(optimizer - truth))
        assert observed["optimizer_rmse"] == pytest.approx(
            np.sqrt(np.mean(np.square(optimizer - truth)))
        )
        assert observed["posterior_bias"] == pytest.approx(np.mean(posterior - truth))
        assert observed["posterior_rmse"] == pytest.approx(
            np.sqrt(np.mean(np.square(posterior - truth)))
        )
        assert observed["hdi_coverage"] == pytest.approx(np.mean(coverage))
        assert observed["maximum_rhat"] == pytest.approx(
            max(row["rhat"] for row in rows)
        )
        assert observed["minimum_bulk_ess"] == pytest.approx(
            min(row["ess_bulk"] for row in rows)
        )
        assert observed["minimum_tail_ess"] == pytest.approx(
            min(row["ess_tail"] for row in rows)
        )
        assert observed["maximum_mcse_over_posterior_sd"] == pytest.approx(
            max(row["mcse_over_posterior_sd"] for row in rows)
        )
    assert artifact["overall_hdi_coverage"] == pytest.approx(np.mean(all_coverage))
    assert artifact["overall_hdi_coverage"] == pytest.approx(14.0 / 16.0)


def test_posterior_and_predictive_science_pass_despite_isolated_failures(
    artifact,
    spec,
):
    """Record the useful evidence separately from the exact failed claims."""
    acceptance = spec["scientific_acceptance"]
    posterior = acceptance["posterior_recovery"]
    predictive = acceptance["posterior_predictive"]
    aggregate = {row["name"]: row for row in artifact["aggregate"]}

    for name in PARAMETER_ORDER:
        row = aggregate[name]
        assert abs(row["posterior_bias"]) <= posterior["maximum_absolute_bias"][name]
        assert row["posterior_rmse"] <= posterior["maximum_rmse"][name]
        assert (
            row["hdi_coverage"]
            >= posterior["minimum_94_percent_hdi_coverage_per_parameter"]
        )
    assert (
        artifact["overall_hdi_coverage"]
        >= posterior["minimum_overall_94_percent_hdi_coverage"]
    )
    for scenario in artifact["scenarios"]:
        assert scenario["prior_predictive"]["passed"] is True
        assert (
            max(scenario["posterior_predictive"]["rt_quantile_absolute_errors"])
            <= predictive["maximum_rt_quantile_absolute_error"]
        )
        assert (
            max(
                scenario["posterior_predictive"][
                    "polar_unit_vector_component_absolute_errors"
                ]
            )
            <= predictive["maximum_polar_unit_vector_component_absolute_error"]
        )


def test_stored_failure_is_exactly_the_independently_recomputed_failure(
    artifact,
    spec,
):
    """The failed gate must remain visible and limited to the observed claims."""
    observed = tuple(_recompute_failures(artifact, spec))

    assert observed == EXPECTED_FAILURES
    assert tuple(artifact["gate"]["failures"]) == EXPECTED_FAILURES
    assert artifact["gate"] == {
        "evaluated": True,
        "passed": False,
        "failures": list(EXPECTED_FAILURES),
    }


def test_compact_artifact_contains_no_machine_local_paths(artifact):
    """Committed evidence should remain portable across development machines."""
    serialized = json.dumps(artifact)

    for marker in ("/Users/", "/home/", "/private/", "/tmp/"):
        assert marker not in serialized

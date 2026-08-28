"""Independent checks for the archived fixed-PSDM compact result."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parents[1]
SPEC_PATH = REPO_ROOT / "benchmarks" / "specs" / "jeam_fixed_psdm_recovery_v1.json"
ADDENDUM_PATH = SPEC_PATH.with_name("jeam_fixed_psdm_recovery_v1_addendum.json")
ARTIFACT_PATH = (
    REPO_ROOT / "benchmarks" / "results" / "jeam_fixed_psdm_recovery_v1.json"
)
EXPECTED_SPEC_SHA256 = (
    "2a9fabe13e612a59f7c2138e4e36ae4e01d4bde5e226c16c8572d5ebe3594198"
)
EXPECTED_ADDENDUM_SHA256 = (
    "42d4627f7e4eadd9c1ba656095cb3edf5af17d04bd03ad11ede476f78149d8f1"
)
EXPECTED_ARTIFACT_SHA256 = (
    "cede87d5a5a2c9789939b66962ebb025b270a13966aa2d657d5b0cbb95e9c2c4"
)
EXPECTED_ARTIFACT_COMMIT = "d76f995d501603cc56f895e5fa429ce2be14e468"
EXPECTED_CANONICAL_START = "2026-08-17T04:15:03.738704+00:00"
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


class ArtifactIntegrityError(ValueError):
    """Raised before parsing when an archived file no longer matches its pin."""


def _load_authenticated_json(path, expected_sha256):
    """Authenticate one byte snapshot before parsing it as JSON."""
    payload = path.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ArtifactIntegrityError(
            f"SHA256 mismatch for {path.name}: expected {expected_sha256}, "
            f"observed {observed_sha256}"
        )
    return json.loads(payload)


@pytest.fixture(scope="module")
def spec():
    """Load the preregistered protocol without importing HSSM or JEAM."""
    return _load_authenticated_json(SPEC_PATH, EXPECTED_SPEC_SHA256)


@pytest.fixture(scope="module")
def addendum():
    """Load the authenticated post-hoc archive boundary."""
    return _load_authenticated_json(ADDENDUM_PATH, EXPECTED_ADDENDUM_SHA256)


@pytest.fixture(scope="module")
def artifact():
    """Load the frozen compact result without importing HSSM or JEAM."""
    return _load_authenticated_json(ARTIFACT_PATH, EXPECTED_ARTIFACT_SHA256)


def _rows_by_parameter(artifact, name):
    """Return the four compact scenario rows for one parameter."""
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
        direct_recovery_errors = np.abs(
            np.asarray(optimizer["direct_jeam"]["parameters"])
            - np.asarray(scenario["truth"])
        )
        if not np.allclose(
            direct_recovery_errors,
            optimizer["direct_recovery_absolute_error"],
            rtol=0.0,
            atol=1e-12,
        ):
            failures.append(f"{name}: archived optimizer recovery errors changed")
        for parameter, error in zip(
            PARAMETER_ORDER, direct_recovery_errors, strict=True
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
            failures.append(f"{name}: archived sample-statistic schema changed")
        posterior_predictive = scenario["posterior_predictive"]
        rt_errors = np.abs(
            np.asarray(posterior_predictive["predictive_rt_quantiles"])
            - np.asarray(posterior_predictive["observed_rt_quantiles"])
        )
        polar_errors = np.abs(
            np.asarray(posterior_predictive["predictive_mean_polar_unit_vector"])
            - np.asarray(posterior_predictive["observed_mean_polar_unit_vector"])
        )
        if not np.allclose(
            rt_errors,
            posterior_predictive["rt_quantile_absolute_errors"],
            rtol=0.0,
            atol=1e-12,
        ) or not np.allclose(
            polar_errors,
            posterior_predictive["polar_unit_vector_component_absolute_errors"],
            rtol=0.0,
            atol=1e-12,
        ):
            failures.append(f"{name}: archived predictive errors changed")
        if max(rt_errors) > predictive["maximum_rt_quantile_absolute_error"]:
            failures.append(f"{name}: predictive RT quantile gate failed")
        if (
            max(polar_errors)
            > predictive["maximum_polar_unit_vector_component_absolute_error"]
        ):
            failures.append(f"{name}: predictive polar unit-vector gate failed")

    all_coverage = []
    for name in PARAMETER_ORDER:
        rows = _rows_by_parameter(artifact, name)
        truth = np.asarray([row["truth"] for row in rows])
        optimizer = np.asarray([row["optimizer_estimate"] for row in rows])
        posterior_mean = np.asarray([row["posterior_mean"] for row in rows])
        coverage = np.asarray(
            [row["hdi_lower"] <= row["truth"] <= row["hdi_upper"] for row in rows]
        )
        all_coverage.extend(coverage.tolist())
        if (
            np.sqrt(np.mean(np.square(optimizer - truth)))
            > acceptance["optimizer_recovery"]["maximum_rmse"][name]
        ):
            failures.append(f"{name}: optimizer RMSE gate failed")
        if (
            abs(np.mean(posterior_mean - truth))
            > posterior["maximum_absolute_bias"][name]
        ):
            failures.append(f"{name}: posterior bias gate failed")
        if (
            np.sqrt(np.mean(np.square(posterior_mean - truth)))
            > posterior["maximum_rmse"][name]
        ):
            failures.append(f"{name}: posterior RMSE gate failed")
        if (
            np.mean(coverage)
            < posterior["minimum_94_percent_hdi_coverage_per_parameter"]
        ):
            failures.append(f"{name}: HDI coverage gate failed")
    if np.mean(all_coverage) < posterior["minimum_overall_94_percent_hdi_coverage"]:
        failures.append("overall HDI coverage gate failed")
    return failures


def test_authentication_precedes_json_parsing(tmp_path):
    """Corrupt bytes must fail their hash pin before JSON parsing begins."""
    corrupt = tmp_path / ARTIFACT_PATH.name
    corrupt.write_bytes(b"{")

    with pytest.raises(ArtifactIntegrityError, match="SHA256 mismatch"):
        _load_authenticated_json(corrupt, EXPECTED_ARTIFACT_SHA256)


def test_artifact_records_exact_historical_provenance(artifact, spec, addendum):
    """The compact result should record historical identifiers and software."""
    assert artifact["schema_version"] == 1
    assert artifact["study_id"] == spec["study_id"]
    assert artifact["canonical"] is True
    assert artifact["model"] == "projected_spherical_diffusion"
    assert tuple(artifact["parameter_order"]) == PARAMETER_ORDER
    assert artifact["spec_path"] == (
        "benchmarks/specs/jeam_fixed_psdm_recovery_v1.json"
    )
    assert artifact["spec_sha256"] == EXPECTED_SPEC_SHA256
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
    started_at = datetime.fromisoformat(artifact["started_at_utc"])
    frozen_at = datetime.fromisoformat(spec["frozen_at_utc"])
    assert artifact["started_at_utc"] == EXPECTED_CANONICAL_START
    assert generated_at > started_at > frozen_at
    assert artifact["total_runtime_seconds"] > 0.0

    outcome = addendum["known_v1_outcome"]
    assert outcome["result_commit"] == EXPECTED_ARTIFACT_COMMIT
    assert outcome["result_sha256"] == EXPECTED_ARTIFACT_SHA256
    assert outcome["canonical_started_at_utc"] == EXPECTED_CANONICAL_START
    assert outcome["overall_pass"] is False
    assert (outcome["truth_in_hdi"], outcome["truth_total"]) == (14, 16)

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


def test_archive_exposes_compact_only_evidence_boundary(artifact, addendum):
    """Recorded hashes and summaries must not masquerade as retained raw evidence."""
    boundary = addendum["evidence_boundary"]
    archive = addendum["runner_archive"]

    assert boundary == {
        "dataset_hashes_recorded": True,
        "trace_hashes_recorded": True,
        "raw_datasets_retained_in_git": 0,
        "raw_traces_retained_in_git": 0,
        "raw_prior_predictive_draws_retained": False,
        "raw_posterior_predictive_draws_retained": False,
        "independent_raw_reverification": "blocked",
        "fixed_psdm_public_support_or_promotion": "blocked",
    }
    assert archive["ordered_slice_identity_independently_authenticated"] is False
    assert archive["runtime_hssm_import_bound_to_recorded_checkout"] is False
    assert addendum["current_safety_revision"]["v1_recovery_rerun"] is False

    retained_names = {path.name for path in REPO_ROOT.rglob("*") if path.is_file()}
    for scenario in artifact["scenarios"]:
        assert scenario["data"]["basename"] not in retained_names
        assert scenario["trace"]["basename"] not in retained_names
        assert set(scenario["sampler_diagnostics"]["sample_stats"]) == {
            "nstep_in",
            "nstep_out",
        }


def test_optimizer_rows_are_fixed_budget_endpoints(artifact, addendum):
    """The archived optimizer rows are neither MLEs nor converged-fit claims."""
    interpretation = addendum["runner_archive"]["optimizer_interpretation"]

    assert "fixed-budget differential-evolution endpoints" in interpretation
    assert "not demonstrated converged optima or MLEs" in interpretation
    for scenario in artifact["scenarios"]:
        for fit in ("direct_jeam", "compiled_hssm"):
            assert scenario["optimizer"][fit]["iterations"] == 20
            assert scenario["optimizer"][fit]["evaluations"] == 1260


def test_every_scenario_preserves_direct_jeam_inside_hssm(artifact, spec):
    """Recompute same-producer adapter parity from stored compact values."""
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
        observed_recovery_error = np.abs(
            np.asarray(direct["parameters"]) - np.asarray(scenario["truth"])
        )

        assert observed_objective_error == pytest.approx(
            scenario["maximum_objective_absolute_error"]
        )
        assert observed_parameter_error == pytest.approx(
            scenario["optimizer"]["maximum_parameter_absolute_error"]
        )
        assert observed_fit_error == pytest.approx(
            scenario["optimizer"]["objective_absolute_error"]
        )
        assert observed_recovery_error == pytest.approx(
            scenario["optimizer"]["direct_recovery_absolute_error"]
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
        coverage = np.asarray(
            [row["hdi_lower"] <= row["truth"] <= row["hdi_upper"] for row in rows]
        )
        assert coverage.tolist() == [row["truth_in_hdi"] for row in rows]
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


def test_compact_posterior_and_predictive_subgates_match_stored_summaries(
    artifact,
    spec,
):
    """Keep passing compact subgates distinct from the failed overall gate."""
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
        # Raw prior and predictive draws are absent; these are authenticated
        # producer summaries, not independently recomputed raw diagnostics.
        assert scenario["prior_predictive"]["passed"] is True
        predictive_row = scenario["posterior_predictive"]
        rt_errors = np.abs(
            np.asarray(predictive_row["predictive_rt_quantiles"])
            - np.asarray(predictive_row["observed_rt_quantiles"])
        )
        polar_errors = np.abs(
            np.asarray(predictive_row["predictive_mean_polar_unit_vector"])
            - np.asarray(predictive_row["observed_mean_polar_unit_vector"])
        )
        assert max(rt_errors) <= predictive["maximum_rt_quantile_absolute_error"]
        assert (
            max(polar_errors)
            <= predictive["maximum_polar_unit_vector_component_absolute_error"]
        )


def test_stored_gate_matches_authenticated_compact_summary_recomputation(
    artifact,
    spec,
):
    """The failed gate must match recomputation from authenticated summaries."""
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

"""Independent regression checks for the fixed-CDM sampler study artifact."""

import hashlib
import json
import shutil
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from scripts.verify_jeam_sampler_comparison_evidence import (
    SamplerEvidenceMismatch,
    load_verified_sampler_comparison,
)
from scripts.verify_jeam_sampler_comparison_evidence import (
    main as verify_sampler_evidence,
)

REPO_ROOT = Path(__file__).parents[1]
SPEC_PATH = (
    REPO_ROOT / "benchmarks" / "specs" / "jeam_fixed_cdm_sampler_comparison_v1.json"
)
ARTIFACT_PATH = (
    REPO_ROOT / "benchmarks" / "results" / "jeam_fixed_cdm_sampler_comparison_v1.json"
)
EXPECTED_SPEC_REVISION = "e8d0e45de1a72d9a3682cf165bb13f3d5b57c474"
EXPECTED_PRODUCER_REVISION = "8bd70a873e856d2b876acddf1371e2776a8f0cca"
EXPECTED_JEAM_REVISION = "0c0ef8b834dd062ad8aea5ff8e7a09dfb55492ce"
EXPECTED_ARTIFACT_SHA256 = (
    "35b154b55228ca179c14d89f39491ba9d1a9d0b27a3c8b4131e842f99abf5d39"
)
EXPECTED_ADDENDUM_SHA256 = (
    "07ca04c8aa52f672b654835e63ff35af31f4ff197a3c95d2cb256ef2ab91b33f"
)
EXPECTED_SCALE_SHA256 = (
    "253a16585d6c2bb0b0aa91f8b6fbaabd5609e284a1d2d2bad61bc97266d9e826"
)
ADDENDUM_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "jeam_fixed_cdm_sampler_comparison_v1_addendum.json"
)
SCALE_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "evidence"
    / "jeam_fixed_cdm_sampler_comparison_v1"
    / "baseline_asymmetric_scale_1500.npy"
)


@pytest.fixture(scope="module")
def spec():
    """Load the preregistered protocol."""
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifact():
    """Load compact evidence without importing HSSM or optional JEAM."""
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def verified_evidence():
    """Authenticate and independently derive the compact evidence boundary."""
    return load_verified_sampler_comparison()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_map(artifact) -> dict[tuple[str, str], dict]:
    return {(fit["scenario"], fit["sampler"]): fit for fit in artifact["fits"]}


def _circular_distance(first: float, second: float) -> float:
    return float(abs(np.angle(np.exp(1j * (first - second)))))


def test_artifact_postdates_and_identifies_the_frozen_protocol(spec, artifact):
    """The result must be complete and tied to immutable producer inputs."""
    assert _sha256(ARTIFACT_PATH) == EXPECTED_ARTIFACT_SHA256
    assert artifact["schema_version"] == spec["schema_version"] == 1
    assert artifact["study_id"] == spec["study_id"]
    assert artifact["status"] == "canonical-complete"
    assert artifact["spec_sha256"] == _sha256(SPEC_PATH)
    assert datetime.fromisoformat(
        artifact["generated_at_utc"]
    ) > datetime.fromisoformat(spec["frozen_at_utc"])

    provenance = artifact["provenance"]
    assert provenance["spec_revision"] == EXPECTED_SPEC_REVISION
    assert provenance["hssm_revision"] == EXPECTED_PRODUCER_REVISION
    assert provenance["jeam_revision"] == EXPECTED_JEAM_REVISION
    assert provenance["python_version"].startswith("3.12.")
    assert provenance["package_versions"]["pymc"] == "6.1.0"
    assert provenance["package_versions"]["numpyro"] == "0.21.0"
    assert provenance["jax_backend"] == "cpu"
    assert provenance["jax_devices"] == ["cpu:0"]
    assert artifact["execution"] == spec["execution"]


def test_data_and_shared_preflight_are_recomputed_from_compact_rows(spec, artifact):
    """Compact dataset, prior, and objective rows must not trust PASS flags."""
    scenarios = {row["name"]: row for row in spec["scenarios"]}
    data_rows = {row["scenario"]: row for row in artifact["data"]}
    preflight_rows = {row["scenario"]: row for row in artifact["shared_preflight"]}
    assert set(data_rows) == set(preflight_rows) == set(scenarios)
    assert artifact["scale_prefix_check_passed"] is True

    prior_gate = spec["preflight_gates"]["prior_predictive"]
    objective_gate = spec["objective_parity"]
    for name, scenario in scenarios.items():
        data = data_rows[name]
        assert data["shape"] == [scenario["trials"], 2]
        assert data["dtype"] == "float64"
        assert data["all_finite"] is True
        assert data["all_rt_strictly_positive"] is True
        assert data["all_angles_in_half_open_domain"] is True
        assert len(data["sha256"]) == 64

        preflight = preflight_rows[name]
        assert (
            preflight["model_contract"]["priors"]
            == spec["priors_and_initialization"]["priors"]
        )
        assert (
            preflight["model_contract"]["initvals"]
            == spec["priors_and_initialization"]["resolved_untransformed_initvals"]
        )
        prior = preflight["prior_predictive"]
        assert prior["all_values_finite"] is True
        assert prior["all_rt_strictly_positive"] is True
        assert prior["all_angles_in_half_open_domain"] is True
        ratios = np.asarray(prior["prior_rt_quantiles"]) / np.asarray(
            prior["observed_rt_quantiles"]
        )
        np.testing.assert_allclose(
            ratios, prior["prior_to_observed_ratios"], rtol=1e-12, atol=0.0
        )
        assert np.all(ratios >= prior_gate["rt_quantile_ratio_to_observed_lower"])
        assert np.all(ratios <= prior_gate["rt_quantile_ratio_to_observed_upper"])

        observed_maximum = 0.0
        truth = np.asarray(scenario["truth"])
        for offset, candidate in zip(
            objective_gate["offsets"],
            preflight["objective_candidates"],
            strict=True,
        ):
            np.testing.assert_allclose(
                candidate["parameters"], truth + np.asarray(offset), rtol=0.0, atol=0.0
            )
            values = np.asarray(
                [
                    candidate["direct_numpy"],
                    candidate["direct_jax"],
                    candidate["compiled_hssm"],
                ]
            )
            maximum = float(np.max(np.abs(values[:, None] - values[None, :])))
            assert candidate["maximum_absolute_error"] == pytest.approx(maximum)
            observed_maximum = max(observed_maximum, maximum)
        assert preflight["maximum_objective_absolute_error"] == pytest.approx(
            observed_maximum
        )
        assert observed_maximum <= objective_gate["maximum_absolute_error"]


def test_every_fit_preserves_compact_diagnostics_and_trace_metadata(spec, artifact):
    """Recompute compact fit fields while treating trace entries as metadata."""
    scenarios = {row["name"]: row for row in spec["scenarios"]}
    samplers = {row["id"]: row for row in spec["samplers"]}
    data_hashes = {row["scenario"]: row["sha256"] for row in artifact["data"]}
    fits = _fit_map(artifact)
    expected = {(scenario, sampler) for scenario in scenarios for sampler in samplers}
    assert set(fits) == expected
    assert len(artifact["fits"]) == len(expected) == 15

    for (scenario_name, sampler_name), fit in fits.items():
        scenario = scenarios[scenario_name]
        sampler = samplers[sampler_name]
        assert fit["status"] == "completed"
        assert fit["smoke"] is False
        assert fit["role"] == scenario["role"]
        assert fit["truth"] == scenario["truth"]
        assert fit["trials"] == scenario["trials"]
        assert fit["likelihood"] == sampler["likelihood"]
        assert fit["backend"] == sampler["backend"]
        assert fit["data_sha256"] == data_hashes[scenario_name]
        assert fit["settings"] == {
            "chains": 4,
            "tune": 1000,
            "draws": 1000,
            "posterior_predictive_draws": 40,
        }
        assert np.isfinite(fit["initial_joint_logp"])
        if sampler_name == "slice":
            assert fit["initial_gradient"] is None
            assert fit["sampler_diagnostics"]["sample_stats"] == [
                "nstep_in",
                "nstep_out",
            ]
        else:
            gradient = np.asarray(fit["initial_gradient"])
            assert gradient.shape == (4,)
            assert np.all(np.isfinite(gradient))
            assert np.any(gradient != 0.0)
            assert fit["sampler_diagnostics"]["divergences"] == 0

        trace = fit["trace"]
        assert trace["saved_before_summary"] is True
        assert trace["basename"] == f"trace-{scenario_name}-{sampler_name}.nc"
        assert len(trace["sha256"]) == 64
        assert trace["bytes"] > 0
        runtime = fit["runtime_seconds"]
        core_fields = [
            "model_build",
            "objective_compile_and_first_eval",
            "gradient_compile_and_first_eval",
            "sampling_call_including_backend_kernel_compilation",
            "posterior_predictive",
        ]
        assert all(runtime[field] >= 0.0 for field in core_fields)
        assert runtime["total"] == pytest.approx(
            sum(runtime[field] for field in core_fields)
        )

        assert [row["name"] for row in fit["parameters"]] == spec["scope"][
            "scientific_parameter_order"
        ]
        truth_by_name = dict(
            zip(
                spec["scope"]["scientific_parameter_order"],
                scenario["truth"],
                strict=True,
            )
        )
        for parameter in fit["parameters"]:
            assert parameter["truth"] == truth_by_name[parameter["name"]]
            assert parameter["truth_in_hdi"] is (
                parameter["hdi_lower"] <= parameter["truth"] <= parameter["hdi_upper"]
            )
            assert parameter["mcse_over_posterior_sd"] == pytest.approx(
                parameter["mcse_mean"] / parameter["posterior_sd"]
            )
            assert parameter["ess_bulk_per_sampling_second"] == pytest.approx(
                parameter["ess_bulk"]
                / runtime["sampling_call_including_backend_kernel_compilation"]
            )

        if scenario["role"] == "scale":
            assert fit["predictive"] is None
        else:
            predictive = fit["predictive"]
            observed_rt = np.asarray(predictive["observed_rt_quantiles"])
            predicted_rt = np.asarray(predictive["predictive_rt_quantiles"])
            np.testing.assert_allclose(
                predictive["rt_quantile_absolute_errors"],
                np.abs(predicted_rt - observed_rt),
                rtol=1e-12,
                atol=0.0,
            )
            assert predictive["mean_angle_distance"] == pytest.approx(
                _circular_distance(
                    predictive["observed_mean_angle"],
                    predictive["predictive_mean_angle"],
                )
            )
            assert predictive["mean_resultant_length_absolute_error"] == pytest.approx(
                abs(
                    predictive["predictive_mean_resultant_length"]
                    - predictive["observed_mean_resultant_length"]
                )
            )


def test_scientific_gate_passes_when_recomputed_without_production_helpers(
    spec, artifact
):
    """All convergence, recovery, and prediction thresholds pass from raw rows."""
    scientific = spec["scientific_acceptance"]
    failures = []
    for fit in artifact["fits"]:
        thresholds = (
            scientific if fit["role"] == "canonical" else scientific["scale_scenario"]
        )
        if fit["sampler"] != "slice" and fit["sampler_diagnostics"]["divergences"]:
            failures.append(f"{fit['scenario']}/{fit['sampler']}: divergences")
        for parameter in fit["parameters"]:
            prefix = f"{fit['scenario']}/{fit['sampler']}/{parameter['name']}"
            if not parameter["rhat"] < thresholds["maximum_rhat_exclusive"]:
                failures.append(f"{prefix}: R-hat")
            if not parameter["ess_bulk"] > thresholds["minimum_bulk_ess_exclusive"]:
                failures.append(f"{prefix}: bulk ESS")
            if not parameter["ess_tail"] > thresholds["minimum_tail_ess_exclusive"]:
                failures.append(f"{prefix}: tail ESS")
            if not (
                parameter["mcse_over_posterior_sd"]
                < thresholds["maximum_mcse_over_posterior_sd_exclusive"]
            ):
                failures.append(f"{prefix}: MCSE/SD")
            if fit["role"] == "canonical" and not parameter["truth_in_hdi"]:
                failures.append(f"{prefix}: truth outside HDI")
        if fit["role"] == "canonical":
            predictive = fit["predictive"]
            if (
                max(predictive["rt_quantile_absolute_errors"])
                > scientific["maximum_rt_quantile_absolute_error"]
            ):
                failures.append(f"{fit['scenario']}/{fit['sampler']}: RT prediction")
            if (
                predictive["mean_angle_distance"]
                > scientific["maximum_circular_mean_angle_distance"]
            ):
                failures.append(f"{fit['scenario']}/{fit['sampler']}: mean angle")
            if (
                predictive["mean_resultant_length_absolute_error"]
                > scientific["maximum_mean_resultant_length_absolute_error"]
            ):
                failures.append(f"{fit['scenario']}/{fit['sampler']}: resultant")

    assert failures == []
    assert artifact["scientific_gate"] == {
        "passed": True,
        "by_sampler": {
            "slice": {"passed": True, "failures": []},
            "pymc_nuts": {"passed": True, "failures": []},
            "numpyro_nuts": {"passed": True, "failures": []},
        },
        "failures": [],
    }


def test_promotion_gate_is_recomputed_from_local_efficiency_rows(spec, artifact):
    """Both NUTS backends independently exceed every machine-local promotion gate."""
    fits = _fit_map(artifact)
    promotion = spec["promotion_decision"]
    canonical_names = [
        row["name"] for row in spec["scenarios"] if row["role"] == "canonical"
    ]

    def minimum_efficiency(fit: dict) -> float:
        return min(row["ess_bulk_per_sampling_second"] for row in fit["parameters"])

    slice_median = statistics.median(
        minimum_efficiency(fits[(name, "slice")]) for name in canonical_names
    )
    assert artifact["promotion_gate"]["passed"] is True
    for sampler in ("pymc_nuts", "numpyro_nuts"):
        stored = artifact["promotion_gate"]["by_sampler"][sampler]
        sampler_median = statistics.median(
            minimum_efficiency(fits[(name, sampler)]) for name in canonical_names
        )
        efficiency_ratio = sampler_median / slice_median
        total_ratios = {
            name: fits[(name, sampler)]["runtime_seconds"]["total"]
            / fits[(name, "slice")]["runtime_seconds"]["total"]
            for name in canonical_names
        }
        scale_fit = fits[("baseline_asymmetric_scale_1500", sampler)]
        reference_fit = fits[(promotion["scale_reference_scenario"], sampler)]
        scale_normalized = (
            min(row["ess_bulk"] for row in scale_fit["parameters"])
            * scale_fit["trials"]
            / scale_fit["runtime_seconds"][
                "sampling_call_including_backend_kernel_compilation"
            ]
        )
        reference_normalized = (
            min(row["ess_bulk"] for row in reference_fit["parameters"])
            * reference_fit["trials"]
            / reference_fit["runtime_seconds"][
                "sampling_call_including_backend_kernel_compilation"
            ]
        )
        scale_ratio = scale_normalized / reference_normalized

        assert (
            efficiency_ratio
            >= promotion[
                "canonical_minimum_median_bulk_ess_per_sampling_second_ratio_vs_slice"
            ]
        )
        assert all(
            ratio
            <= promotion["canonical_maximum_total_seconds_ratio_vs_slice_per_scenario"]
            for ratio in total_ratios.values()
        )
        assert (
            scale_ratio
            >= promotion["scale_minimum_normalized_efficiency_ratio_vs_reference"]
        )
        assert stored["canonical_median_minimum_bulk_ess_per_second"] == pytest.approx(
            sampler_median
        )
        assert stored[
            "slice_canonical_median_minimum_bulk_ess_per_second"
        ] == pytest.approx(slice_median)
        assert stored["canonical_efficiency_ratio_vs_slice"] == pytest.approx(
            efficiency_ratio
        )
        assert stored["canonical_total_seconds_ratios_vs_slice"] == pytest.approx(
            total_ratios
        )
        assert stored[
            "scale_normalized_efficiency_ratio_vs_reference"
        ] == pytest.approx(scale_ratio)
        assert stored["passed"] is True
        assert stored["decision"] == "recommend default"
        assert stored["failures"] == []


def test_addendum_freezes_historical_and_current_evidence_scopes(verified_evidence):
    """The sidecar must not turn the historical result into a current-stack claim."""
    assert _sha256(ADDENDUM_PATH) == EXPECTED_ADDENDUM_SHA256
    assert verified_evidence["evidence_class"] == (
        "authenticated compact-only smoke benchmark"
    )
    assert verified_evidence["jeam_revisions"] == {
        "historical_analytical_result": ("0c0ef8b834dd062ad8aea5ff8e7a09dfb55492ce"),
        "durable_blackbox_reference": ("a9f547b3630ae8ff31ccec1b904e0c02fdba6d99"),
        "current_safety_revision": ("ede7a4f4faf226e4dae52c84dfb01012939cccdc"),
    }
    assert verified_evidence["counts"] == {
        "scenarios": 5,
        "canonical_scenarios": 4,
        "samplers": 3,
        "fits": 15,
        "canonical_scenario_parameter_truths": 16,
        "canonical_route_hdi_checks": 48,
        "canonical_route_hdi_inclusions": 48,
        "canonical_nuts_fits": 8,
        "all_nuts_fits": 10,
        "all_nuts_divergences": 0,
        "trace_records": 15,
        "retained_trace_files": 0,
    }
    promotion = verified_evidence["ecosystem_promotion"]
    assert promotion["blocked"] is True
    assert any("backend identity" in item for item in promotion["blockers"])
    assert any(
        "prior- and posterior-predictive" in item for item in promotion["blockers"]
    )


def test_addendum_binds_durable_canonical_and_reconstructed_scale_data(
    verified_evidence,
):
    """All study inputs are durable without mislabeling reconstructed scale bytes."""
    assert _sha256(SCALE_PATH) == EXPECTED_SCALE_SHA256
    scale = np.load(SCALE_PATH, allow_pickle=False)
    baseline_path = (
        REPO_ROOT
        / "benchmarks/evidence/jeam_repeated_recovery_v2/scenarios"
        / "baseline_asymmetric/dataset.npy"
    )
    baseline = np.load(baseline_path, allow_pickle=False)
    assert scale.shape == (1500, 2)
    assert scale.dtype.str == "<f8"
    np.testing.assert_array_equal(scale[:300], baseline)
    assert verified_evidence["reconstructed_scale_dataset_sha256"] == (
        EXPECTED_SCALE_SHA256
    )
    retention = verified_evidence["retention"]
    assert retention["raw_prior_predictive_draws_retained"] is False
    assert retention["raw_posterior_predictive_draws_retained"] is False
    assert retention["sampler_backend_trace_attributes_retained"] is False
    assert retention["historical_scale_dataset_bytes_retained"] is False
    assert retention["post_hoc_scale_dataset_bytes_retained"] is True


def test_verifier_authenticates_before_parsing_mutated_result(tmp_path):
    """A coordinated-looking compact edit cannot cross the pinned hash boundary."""
    copied = tmp_path / "benchmarks"
    shutil.copytree(REPO_ROOT / "benchmarks", copied)
    result_path = copied / "results/jeam_fixed_cdm_sampler_comparison_v1.json"
    result_path.write_bytes(b"{")

    with pytest.raises(SamplerEvidenceMismatch, match="SHA256 mismatch"):
        load_verified_sampler_comparison(tmp_path)


def test_verifier_cli_never_upgrades_compact_evidence(capsys):
    """The user-facing summary must retain the promotion blocker."""
    assert verify_sampler_evidence([]) == 0
    output = capsys.readouterr().out
    assert "compact-only sampler evidence PASS" in output
    assert "promotion blocked" in output
    assert "durable analytical" not in output

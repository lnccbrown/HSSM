"""Independent checks for the predeclared fixed-CDM sampler comparison."""

import json
from datetime import datetime
from pathlib import Path

import pytest

SPEC_PATH = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "specs"
    / "jeam_fixed_cdm_sampler_comparison_v1.json"
)
PARAMETER_ORDER = ["a", "t", "v_x", "v_y"]
CANONICAL_SCENARIOS = {
    "baseline_asymmetric": {
        "truth": [1.2, 0.15, 0.8, -0.5],
        "data_seed": 1492,
        "chain_seeds": [3101, 3102, 3103, 3104],
        "prior_predictive_seed": 6101,
        "posterior_predictive_seed": 7291,
    },
    "reversed_drift": {
        "truth": [1.05, 0.1, -0.9, 0.65],
        "data_seed": 2603,
        "chain_seeds": [4201, 4202, 4203, 4204],
        "prior_predictive_seed": 7101,
        "posterior_predictive_seed": 8291,
    },
    "high_threshold_strong_drift": {
        "truth": [1.6, 0.22, 1.25, 0.3],
        "data_seed": 3714,
        "chain_seeds": [5301, 5302, 5303, 5304],
        "prior_predictive_seed": 8101,
        "posterior_predictive_seed": 9291,
    },
    "low_threshold_negative_drift": {
        "truth": [0.75, 0.07, -0.45, -1.15],
        "data_seed": 4825,
        "chain_seeds": [6401, 6402, 6403, 6404],
        "prior_predictive_seed": 9101,
        "posterior_predictive_seed": 10291,
    },
}


@pytest.fixture(scope="module")
def spec():
    """Load the protocol without importing HSSM or optional JEAM."""
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def test_protocol_is_frozen_before_results(spec):
    """The benchmark inputs must identify a pre-result, versioned protocol."""
    assert spec["schema_version"] == 1
    assert spec["study_id"] == "jeam-fixed-cdm-sampler-comparison-v1"
    assert spec["status"] == "preregistered-before-canonical-run"
    assert spec["canonical_results_must_postdate_this_file_commit"] is True
    assert datetime.fromisoformat(spec["frozen_at_utc"]).tzinfo is not None
    assert len(spec["provenance"]["hssm_stack_base_revision"]) == 40
    assert spec["provenance"]["jeam_revision"] == (
        "0c0ef8b834dd062ad8aea5ff8e7a09dfb55492ce"
    )
    assert spec["provenance"]["python_minor"] == "3.12"
    assert spec["provenance"]["pytensor_floatx"] == "float64"


def test_protocol_reuses_the_four_predeclared_recovery_scenarios(spec):
    """The canonical cases and all stochastic streams stay fixed to H09c."""
    assert spec["scope"]["scientific_parameter_order"] == PARAMETER_ORDER
    observed = {
        scenario["name"]: scenario
        for scenario in spec["scenarios"]
        if scenario["role"] == "canonical"
    }
    assert set(observed) == set(CANONICAL_SCENARIOS)
    for name, expected in CANONICAL_SCENARIOS.items():
        assert observed[name]["trials"] == 300
        for field, value in expected.items():
            assert observed[name][field] == value


def test_scale_case_is_a_controlled_extension_of_the_baseline(spec):
    """The only scaling case changes trial count while freezing data and chains."""
    scenarios = {scenario["name"]: scenario for scenario in spec["scenarios"]}
    baseline = scenarios["baseline_asymmetric"]
    scale = scenarios["baseline_asymmetric_scale_1500"]

    assert scale["role"] == "scale"
    assert scale["trials"] == 5 * baseline["trials"] == 1500
    for field in (
        "truth",
        "data_seed",
        "chain_seeds",
        "prior_predictive_seed",
        "posterior_predictive_seed",
    ):
        assert scale[field] == baseline[field]
    assert scale["exact_prefix_of"] == baseline["name"]
    assert scale["exact_prefix_rows"] == baseline["trials"]


def test_sampler_comparison_freezes_fair_execution(spec):
    """All routes receive equal draws, serial resources, data, and initialization."""
    execution = spec["execution"]
    assert execution["chains"] == 4
    assert execution["tune"] == execution["draws"] == 1000
    assert execution["cores"] == 1
    assert execution["blas_cores"] == 1
    assert execution["compute_convergence_checks_during_sampling"] is True
    assert execution["fresh_subprocess_per_scenario_sampler"] is True
    assert execution["posterior_hdi_probability"] == pytest.approx(0.94)
    assert execution["posterior_predictive_draws"] == 40
    assert execution["sampler_order"] == [
        "slice",
        "pymc_nuts",
        "numpyro_nuts",
    ]
    assert spec["priors_and_initialization"]["resolved_untransformed_initvals"] == {
        "v_x": 0.0,
        "v_y": 0.0,
        "a": 1.5,
        "t": 0.025,
    }

    samplers = {sampler["id"]: sampler for sampler in spec["samplers"]}
    assert set(samplers) == set(execution["sampler_order"])
    assert (samplers["slice"]["likelihood"], samplers["slice"]["backend"]) == (
        "blackbox",
        "pymc",
    )
    for name, backend in (("pymc_nuts", "pymc"), ("numpyro_nuts", "numpyro")):
        assert samplers[name]["likelihood"] == "analytical"
        assert samplers[name]["backend"] == backend
        assert samplers[name]["target_accept"] == pytest.approx(0.9)
    assert samplers["numpyro_nuts"]["nuts_sampler_kwargs"] == {
        "chain_method": "sequential"
    }


def test_scientific_and_promotion_gates_are_separate_and_exact(spec):
    """Scientific validity cannot be replaced by machine-specific speed evidence."""
    scientific = spec["scientific_acceptance"]
    assert scientific["zero_divergences_for_nuts"] is True
    assert scientific["maximum_rhat_exclusive"] == pytest.approx(1.01)
    assert scientific["minimum_bulk_ess_exclusive"] == pytest.approx(400.0)
    assert scientific["minimum_tail_ess_exclusive"] == pytest.approx(400.0)
    assert scientific["maximum_mcse_over_posterior_sd_exclusive"] == pytest.approx(0.05)
    assert scientific["every_parameter_truth_in_94_percent_hdi"] is True
    assert scientific["maximum_rt_quantile_absolute_error"] == pytest.approx(0.12)
    assert scientific["maximum_circular_mean_angle_distance"] == pytest.approx(0.1)
    assert scientific["maximum_mean_resultant_length_absolute_error"] == pytest.approx(
        0.08
    )
    assert scientific["objective_parity_maximum_absolute_error"] == pytest.approx(1e-8)

    promotion = spec["promotion_decision"]
    assert promotion["separate_from_scientific_acceptance"] is True
    assert promotion["evaluate_each_nuts_backend_independently"] is True
    assert promotion[
        "canonical_minimum_median_bulk_ess_per_sampling_second_ratio_vs_slice"
    ] == pytest.approx(1.5)
    assert promotion[
        "canonical_maximum_total_seconds_ratio_vs_slice_per_scenario"
    ] == pytest.approx(1.25)
    assert promotion[
        "scale_minimum_normalized_efficiency_ratio_vs_reference"
    ] == pytest.approx(0.8)


def test_protocol_preserves_raw_evidence_and_forbids_result_selection(spec):
    """Every fit must remain auditable even though large traces stay local."""
    execution = spec["execution"]
    reporting = spec["reporting_and_deviation_policy"]
    assert "SHA256" in execution["data_policy"]
    assert "immediately after sampling" in execution["trace_policy"]
    assert reporting["raw_traces_committed"] is False
    assert reporting["compact_result_committed"] is True
    assert reporting["no_manual_chain_exclusions"] is True
    assert reporting["no_interim_stopping"] is True
    assert reporting["failed_sampling_is_a_recorded_result"] is True
    assert reporting["result_path"].endswith(
        "jeam_fixed_cdm_sampler_comparison_v1.json"
    )

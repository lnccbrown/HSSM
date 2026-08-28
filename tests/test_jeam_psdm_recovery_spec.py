"""Independent checks for the archived fixed-PSDM recovery protocol."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

SPEC_PATH = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "specs"
    / "jeam_fixed_psdm_recovery_v1.json"
)
ADDENDUM_PATH = SPEC_PATH.with_name("jeam_fixed_psdm_recovery_v1_addendum.json")
EXPECTED_ORIGINAL_FREEZE_COMMIT = "c1c68ef3c0ebdf78b4a950c4e62e61bee55b0961"
EXPECTED_SPEC_SHA256 = (
    "2a9fabe13e612a59f7c2138e4e36ae4e01d4bde5e226c16c8572d5ebe3594198"
)
EXPECTED_HSSM_BASE = "11fdc2236711f46d3bd6ba19589947fc20285ff1"
EXPECTED_JEAM_REVISION = "1d7112757d8b2d27a31437255fc679194d39ab89"
PARAMETER_ORDER = ("v_x", "v_y", "a", "t")
EXPECTED_SCENARIOS = {
    "baseline_asymmetric": (0.6, 1.0, 1.1, 0.2),
    "reverse_axial_weak_radial": (-0.7, 0.45, 0.85, 0.1),
    "high_threshold_strong_radial": (0.3, 1.25, 1.5, 0.22),
    "low_threshold_balanced_drift": (0.9, 0.75, 0.7, 0.07),
}
EXPECTED_SEEDS = {
    "baseline_asymmetric": (1592, 8695309, (7101, 7102, 7103, 7104), 11101, 11291),
    "reverse_axial_weak_radial": (
        2703,
        54221,
        (8201, 8202, 8203, 8204),
        12101,
        12291,
    ),
    "high_threshold_strong_radial": (
        3814,
        64231,
        (9301, 9302, 9303, 9304),
        13101,
        13291,
    ),
    "low_threshold_balanced_drift": (
        4925,
        74241,
        (10401, 10402, 10403, 10404),
        14101,
        14291,
    ),
}


def _load_spec():
    """Load the frozen protocol without importing HSSM or JEAM."""
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _load_addendum():
    """Load the post-hoc scope statement without changing the v1 protocol."""
    return json.loads(ADDENDUM_PATH.read_text(encoding="utf-8"))


def test_archived_protocol_matches_the_original_preregistration():
    """The replay must preserve the exact bytes that predated canonical v1."""
    assert hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest() == EXPECTED_SPEC_SHA256

    addendum = _load_addendum()
    immutable = addendum["immutable_protocol"]
    assert addendum["schema_version"] == 1
    assert addendum["study_id"] == "jeam-fixed-psdm-recovery-v1"
    assert addendum["status"] == "historical-preregistration-archived-promotion-blocked"
    assert immutable["original_freeze_commit"] == EXPECTED_ORIGINAL_FREEZE_COMMIT
    assert immutable["sha256"] == EXPECTED_SPEC_SHA256
    assert "not a new preregistration" in immutable["replay_policy"]


def test_archive_states_the_historical_scope_and_negative_outcome():
    """Archival wording must not promote or broaden the failed v1 smoke."""
    addendum = _load_addendum()

    assert addendum["document_role"].startswith("Post-hoc archival interpretation")
    assert addendum["historical_execution"] == {
        "hssm_stack_base_revision": EXPECTED_HSSM_BASE,
        "hssm_runner_revision": "ebbd68ee6dcaad644505ae7f3739b7b1f0ba3794",
        "jeam_revision": EXPECTED_JEAM_REVISION,
        "python_minor": "3.12",
        "pytensor_floatx": "float64",
    }
    assert addendum["current_safety_revision"] == {
        "jeam_revision": "ede7a4f4faf226e4dae52c84dfb01012939cccdc",
        "v1_recovery_rerun": False,
    }
    scope = addendum["scope"]
    assert scope["model_contract"] == (
        "ordinary scalar/intercept-only public-default HSSM model"
    )
    assert scope["formula_or_regression_support_evaluated"] is False
    assert scope["t_prior"] == "untruncated HalfNormal(sigma=2)"
    assert scope["configured_t_bounds"] == [0.0, 2.0]
    assert scope["prior_and_bounds_are_distinct_contracts"] is True
    assert scope["configured_t_bound_truncated_the_prior"] is False
    assert scope["posterior_likelihood_support"] == "t < minimum observed rt"
    outcome = addendum["known_v1_outcome"]
    assert {key: outcome[key] for key in outcome if key != "interpretation"} == {
        "result_path": "benchmarks/results/jeam_fixed_psdm_recovery_v1.json",
        "result_commit": "d76f995d501603cc56f895e5fa429ce2be14e468",
        "result_sha256": (
            "cede87d5a5a2c9789939b66962ebb025b270a13966aa2d657d5b0cbb95e9c2c4"
        ),
        "canonical_started_at_utc": "2026-08-17T04:15:03.738704+00:00",
        "overall_pass": False,
        "truth_in_hdi": 14,
        "truth_total": 16,
    }
    assert "was not demonstrated" in outcome["interpretation"]
    boundary = addendum["evidence_boundary"]
    assert boundary["dataset_hashes_recorded"] is True
    assert boundary["trace_hashes_recorded"] is True
    assert boundary["raw_datasets_retained_in_git"] == 0
    assert boundary["raw_traces_retained_in_git"] == 0
    assert boundary["raw_prior_predictive_draws_retained"] is False
    assert boundary["raw_posterior_predictive_draws_retained"] is False
    assert boundary["independent_raw_reverification"] == "blocked"
    assert boundary["fixed_psdm_public_support_or_promotion"] == "blocked"
    assert addendum["successor_policy"] == {
        "v2a_and_v2b_require_new_preregistrations": True,
        "mixing_and_identifiability_are_hypotheses_not_v1_conclusions": True,
    }


def test_protocol_freezes_model_measure_and_provenance():
    """The scientific question should identify the exact code and density."""
    spec = _load_spec()

    assert spec["schema_version"] == 1
    assert spec["study_id"] == "jeam-fixed-psdm-recovery-v1"
    assert spec["status"] == "preregistered-before-canonical-run"
    assert spec["canonical_results_must_postdate_this_file_commit"] is True
    assert datetime.fromisoformat(spec["frozen_at_utc"]).tzinfo is not None
    assert spec["scope"]["model"] == "projected_spherical_diffusion"
    assert spec["scope"]["response_columns"] == ["rt", "response"]
    assert spec["scope"]["density_measure"] == "d(rt) d(response)"
    assert tuple(spec["scope"]["parameter_order"]) == PARAMETER_ORDER
    assert spec["provenance"]["hssm_stack_base_revision"] == EXPECTED_HSSM_BASE
    assert spec["provenance"]["jeam_revision"] == EXPECTED_JEAM_REVISION
    assert spec["provenance"]["pytensor_floatx"] == "float64"


def test_protocol_uses_the_public_hssm_model_contract():
    """Recovery should test ordinary HSSM defaults, not bespoke recovery priors."""
    spec = _load_spec()
    setup = spec["priors_and_initialization"]

    assert setup["priors"] == {
        "v_x": {"distribution": "Uniform", "lower": -3.0, "upper": 3.0},
        "v_y": {"distribution": "Uniform", "lower": 0.0, "upper": 3.0},
        "a": {"distribution": "Uniform", "lower": 0.1, "upper": 3.0},
        "t": {"distribution": "HalfNormal", "sigma": 2.0},
    }
    assert setup["configured_bounds"] == {
        "v_x": [-3.0, 3.0],
        "v_y": [0.0, 3.0],
        "a": [0.1, 3.0],
        "t": [0.0, 2.0],
    }
    assert setup["resolved_untransformed_initvals"] == {
        "v_x": 0.0,
        "v_y": 1.5,
        "a": 1.5,
        "t": 0.025,
    }
    assert spec["scope"]["fixed_model_settings"] == {
        "sigma": 1.0,
        "s_v": 0.0,
        "s_t": 0.0,
        "threshold_dynamic": "fixed",
        "decay": 0.0,
        "threshold_function": None,
        "p_outlier": None,
    }


def test_protocol_freezes_four_interior_and_distinct_scenarios():
    """Scenarios should vary all scientific axes without hugging model bounds."""
    spec = _load_spec()
    scenarios = spec["scenarios"]
    observed = {row["name"]: tuple(row["truth"]) for row in scenarios}
    observed_seeds = {
        row["name"]: (
            row["data_seed"],
            row["optimizer_seed"],
            tuple(row["chain_seeds"]),
            row["prior_predictive_seed"],
            row["posterior_predictive_seed"],
        )
        for row in scenarios
    }

    assert observed == EXPECTED_SCENARIOS
    assert observed_seeds == EXPECTED_SEEDS
    assert all(row["trials"] == 400 for row in scenarios)
    truth = np.asarray([row["truth"] for row in scenarios], dtype=np.float64)
    assert np.any(truth[:, 0] < 0.0) and np.any(truth[:, 0] > 0.0)
    assert np.ptp(truth[:, 1]) >= 0.8
    assert np.ptp(truth[:, 2]) >= 0.8
    assert np.ptp(truth[:, 3]) >= 0.15
    assert np.all((-2.0 < truth[:, 0]) & (truth[:, 0] < 2.0))
    assert np.all((0.02 < truth[:, 1]) & (truth[:, 1] < 2.0))
    assert np.all((0.4 < truth[:, 2]) & (truth[:, 2] < 2.0))
    assert np.all(truth[:, 3] > 0.02)

    scalar_seeds = [
        row[key]
        for row in scenarios
        for key in (
            "data_seed",
            "optimizer_seed",
            "prior_predictive_seed",
            "posterior_predictive_seed",
        )
    ]
    chain_seeds = [seed for row in scenarios for seed in row["chain_seeds"]]
    assert len(scalar_seeds + chain_seeds) == len(set(scalar_seeds + chain_seeds))
    assert all(len(row["chain_seeds"]) == 4 for row in scenarios)


def test_protocol_freezes_ordered_slice_and_artifact_policy():
    """The run should separate scientific evidence from wall-clock reporting."""
    spec = _load_spec()
    execution = spec["execution"]
    acceptance = spec["scientific_acceptance"]
    reporting = spec["reporting_and_deviation_policy"]

    assert execution["sampler"] == (
        "one pymc.Slice step per parameter in parameter_order"
    )
    assert execution["likelihood"] == "blackbox"
    assert execution["backend"] == "pymc"
    assert execution["chains"] == 4
    assert execution["tune"] == execution["draws"] == 1000
    assert execution["cores"] == execution["blas_cores"] == 1
    assert execution["posterior_hdi_probability"] == 0.94
    assert execution["predictive_angle_summary"].startswith(
        "mean polar unit-vector components"
    )
    assert "time" not in json.dumps(acceptance).lower()
    assert reporting["raw_traces_committed"] is False
    assert reporting["compact_result_committed"] is True
    assert reporting["no_manual_chain_exclusions"] is True
    assert reporting["no_interim_stopping"] is True


def test_protocol_defines_optimizer_posterior_and_predictive_gates():
    """Every claimed layer should have a numerical criterion before execution."""
    spec = _load_spec()
    acceptance = spec["scientific_acceptance"]

    assert acceptance["maximum_objective_absolute_error"] == 1e-8
    assert acceptance["maximum_optimizer_parameter_absolute_error"] == 1e-9
    assert acceptance["maximum_optimizer_objective_absolute_error"] == 1e-6
    assert set(acceptance["optimizer_recovery"]["maximum_absolute_error"]) == set(
        PARAMETER_ORDER
    )
    assert set(acceptance["optimizer_recovery"]["maximum_rmse"]) == set(PARAMETER_ORDER)
    posterior = acceptance["posterior_recovery"]
    assert posterior["maximum_rhat_exclusive"] == 1.01
    assert posterior["minimum_bulk_ess_exclusive"] == 400.0
    assert posterior["minimum_tail_ess_exclusive"] == 400.0
    assert posterior["minimum_94_percent_hdi_coverage_per_parameter"] == 0.75
    assert posterior["minimum_overall_94_percent_hdi_coverage"] == 0.75
    predictive = acceptance["posterior_predictive"]
    assert predictive["maximum_rt_quantile_absolute_error"] == 0.15
    assert predictive["maximum_polar_unit_vector_component_absolute_error"] == 0.08

"""Independent checks for the preregistered fixed-PSDM recovery study."""

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
EXPECTED_HSSM_BASE = "11fdc2236711f46d3bd6ba19589947fc20285ff1"
EXPECTED_JEAM_REVISION = "1d7112757d8b2d27a31437255fc679194d39ab89"
PARAMETER_ORDER = ("v_x", "v_y", "a", "t")
EXPECTED_SCENARIOS = {
    "baseline_asymmetric": (0.6, 1.0, 1.1, 0.2),
    "reverse_axial_weak_radial": (-0.7, 0.45, 0.85, 0.1),
    "high_threshold_strong_radial": (0.3, 1.25, 1.5, 0.22),
    "low_threshold_balanced_drift": (0.9, 0.75, 0.7, 0.07),
}


def _load_spec():
    """Load the frozen protocol without importing HSSM or JEAM."""
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


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

    assert observed == EXPECTED_SCENARIOS
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

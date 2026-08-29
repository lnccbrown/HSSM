"""Static contract tests for the fixed-PSDM successor preregistration."""

import json
import math
from datetime import datetime
from hashlib import sha256
from pathlib import Path

SPEC_PATH = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "specs"
    / "jeam_fixed_psdm_recovery_v2.json"
)
PARAMETER_ORDER = ("v_x", "v_y", "a", "t")
V1_SPEC_SHA256 = "2a9fabe13e612a59f7c2138e4e36ae4e01d4bde5e226c16c8572d5ebe3594198"
V1_ADDENDUM_SHA256 = "42d4627f7e4eadd9c1ba656095cb3edf5af17d04bd03ad11ede476f78149d8f1"
V1_RESULT_SHA256 = "cede87d5a5a2c9789939b66962ebb025b270a13966aa2d657d5b0cbb95e9c2c4"
CURRENT_JEAM_REVISION = "ede7a4f4faf226e4dae52c84dfb01012939cccdc"
V2_SPEC_SHA256 = "ba19b38bfaf6bb3167e9ca3e7fc37b62696633bf9f07ba1160607b8e6e3825fa"
EXPECTED_DATASETS = {
    "baseline_asymmetric": (
        (0.6, 1.0, 1.1, 0.2),
        1592,
        "5b39ad8f2453871a15f574437b1b62d20372476e814019caa9e028f85c0f9726",
    ),
    "reverse_axial_weak_radial": (
        (-0.7, 0.45, 0.85, 0.1),
        2703,
        "e2228ec7b121758e5a1599cdda54018caea95940caaa33cc10ae4beafebcba82",
    ),
    "high_threshold_strong_radial": (
        (0.3, 1.25, 1.5, 0.22),
        3814,
        "a598a0c5f76e54c4019ccefa027714c47f13525668ef3e2ef328a2cb13f21cc7",
    ),
    "low_threshold_balanced_drift": (
        (0.9, 0.75, 0.7, 0.07),
        4925,
        "bba3bae04b0cc329586f9bce82a14aaacf51117c5009f3a5e01942c205857cc8",
    ),
}
V1_SEEDS = {
    1592,
    2703,
    3814,
    4925,
    8695309,
    54221,
    64231,
    74241,
    7101,
    7102,
    7103,
    7104,
    8201,
    8202,
    8203,
    8204,
    9301,
    9302,
    9303,
    9304,
    10401,
    10402,
    10403,
    10404,
    11101,
    11291,
    12101,
    12291,
    13101,
    13291,
    14101,
    14291,
}


def _load_spec():
    """Load the preregistration without importing any inference package."""

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        SPEC_PATH.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _allocated_seeds(spec):
    """Separate generated-data seeds from all new inference seeds."""
    data_seeds = []
    inference_seeds = []
    for scenario in spec["studies"]["mixing_v2a"]["scenarios"]:
        inference_seeds.append(scenario["prior_predictive_seed"])
        for block in scenario["sampling_blocks"]:
            inference_seeds.extend(block["chain_seeds"])
            inference_seeds.append(block["posterior_predictive_seed"])
    for cell in spec["studies"]["information_v2b"]["design_cells"]:
        for replicate in cell["replicates"]:
            data_seeds.append(replicate["source_data_seed"])
            inference_seeds.extend(
                (
                    replicate["prior_predictive_seed"],
                    replicate["posterior_predictive_seed"],
                    *replicate["chain_seeds"],
                )
            )
    return data_seeds, inference_seeds


def test_protocol_bytes_are_immutable_and_strict_json():
    """Any committed protocol-byte change must require a new version."""
    payload = SPEC_PATH.read_bytes()

    assert sha256(payload).hexdigest() == V2_SPEC_SHA256
    assert payload.endswith(b"\n")
    assert not payload.endswith(b"\n\n")
    assert _load_spec()["schema_version"] == 1


def test_successor_contract_is_frozen_before_any_evidence():
    """Both successor questions must predate reconstruction and execution."""
    spec = _load_spec()

    assert spec["schema_version"] == 1
    assert spec["study_id"] == "jeam-fixed-psdm-recovery-v2-decision-contract"
    assert spec["status"] == "preregistered-before-dataset-recovery-or-execution"
    assert spec["canonical_results_must_postdate_this_file_commit"] is True
    assert datetime.fromisoformat(spec["frozen_at_utc"]).tzinfo is not None
    assert tuple(spec["execution_order"]) == (
        "recover_v1_datasets",
        "mixing_v2a",
        "information_v2b",
    )
    assert tuple(spec["studies"]) == ("mixing_v2a", "information_v2b")
    assert spec["questions"]["information_v2b"] == (
        "Conditional on every v2a full window passing, does the seeded "
        "trial-count by threshold by radial-drift factorial design meet the "
        "frozen v_y recovery, posterior-contraction, and paired trial-count gates "
        "under the same ordered-Slice contract?"
    )

    policy = spec["reporting_and_deviation_policy"]
    assert policy["v1_files_immutable"] is True
    assert policy["no_threshold_changes_after_freeze"] is True
    assert policy["no_manual_chain_exclusions"] is True
    assert policy["no_unpreregistered_interim_stopping"] is True
    assert policy["no_stopping_within_a_started_arm"] is True
    assert policy["only_preregistered_between_arm_gate"] == (
        "information_v2b starts only if every mixing_v2a full window passes "
        "every common diagnostic gate"
    )
    assert policy["protocol_change"] == (
        "Never edit a committed v2 specification; add a new versioned "
        "specification for every change."
    )
    assert policy["failed_attempts_retained"] is True
    assert policy["calibration_claim_permitted"] is False
    assert policy["fixed_psdm_public_support_or_promotion"] == "blocked"


def test_historical_inputs_are_cryptographically_bound_and_fail_closed():
    """V2a must use all four exact v1 datasets or not execute."""
    spec = _load_spec()
    historical = spec["historical_bindings"]

    assert historical["v1_spec"] == {
        "path": "benchmarks/specs/jeam_fixed_psdm_recovery_v1.json",
        "original_freeze_commit": "c1c68ef3c0ebdf78b4a950c4e62e61bee55b0961",
        "sha256": V1_SPEC_SHA256,
    }
    assert historical["v1_addendum"]["sha256"] == V1_ADDENDUM_SHA256
    assert historical["v1_result"] == {
        "path": "benchmarks/results/jeam_fixed_psdm_recovery_v1.json",
        "commit": "d76f995d501603cc56f895e5fa429ce2be14e468",
        "sha256": V1_RESULT_SHA256,
    }
    assert historical["historical_jeam_revision"] == (
        "1d7112757d8b2d27a31437255fc679194d39ab89"
    )
    assert historical["current_safety_jeam_revision"] == CURRENT_JEAM_REVISION
    assert historical["known_v1_outcome"] == {
        "overall_pass": False,
        "truth_in_hdi": 14,
        "truth_total": 16,
        "mixing_and_identifiability_are_hypotheses": True,
    }

    observed = {
        row["name"]: (
            tuple(row["truth"]),
            row["data_seed"],
            row["artifact"]["sha256"],
        )
        for row in historical["datasets"]
    }
    assert observed == EXPECTED_DATASETS
    assert all(
        row["artifact"]
        == {
            "bytes": 6528,
            "dtype": "float64",
            "shape": [400, 2],
            "sha256": EXPECTED_DATASETS[row["name"]][2],
        }
        for row in historical["datasets"]
    )

    gate = spec["studies"]["mixing_v2a"]["dataset_recovery_gate"]
    assert gate["required_exact_matches"] == 4
    assert gate["comparison"] == "complete .npy bytes and SHA256"
    assert gate["allclose_permitted"] is False
    assert gate["partial_acceptance_permitted"] is False
    assert gate["ad_hoc_environment_tuning_permitted"] is False
    assert gate["on_any_mismatch"] == "stop mixing_v2a without model construction"


def test_common_model_environment_initialization_and_sampler_are_exact():
    """Successor arms should differ in data, not hidden runtime choices."""
    common = _load_spec()["common_contract"]

    assert tuple(common["model"]["parameter_order"]) == PARAMETER_ORDER
    assert common["model"]["likelihood"] == "blackbox"
    assert common["model"]["fixed_settings"] == {
        "sigma": 1.0,
        "s_v": 0.0,
        "s_t": 0.0,
        "threshold_dynamic": "fixed",
        "decay": 0.0,
        "threshold_function": None,
        "p_outlier": None,
    }
    assert common["priors"] == {
        "v_x": {"distribution": "Uniform", "lower": -3.0, "upper": 3.0},
        "v_y": {"distribution": "Uniform", "lower": 0.0, "upper": 3.0},
        "a": {"distribution": "Uniform", "lower": 0.1, "upper": 3.0},
        "t": {"distribution": "HalfNormal", "sigma": 2.0},
    }
    assert common["configured_bounds"] == {
        "v_x": [-3.0, 3.0],
        "v_y": [0.0, 3.0],
        "a": [0.1, 3.0],
        "t": [0.0, 2.0],
    }
    assert common["prior_and_bound_interpretation"] == (
        "the configured t bound is model metadata and does not truncate the "
        "HalfNormal(2) prior; posterior likelihood support additionally requires "
        "t to remain below every observed response time"
    )
    initialization = common["initialization"]
    assert initialization["untransformed"] == {
        "v_x": 0.0,
        "v_y": 1.5,
        "a": 1.5,
        "t": 0.025,
    }
    assert initialization["jitter"] is False
    assert initialization["hssm_sample_init_argument"] is None
    assert initialization["resolved_pymc_init_argument"] == "adapt_diag"
    assert initialization["minimum_rt_minus_t_at_least"] == 1e-6
    assert initialization["finite_transformed_point_required"] is True
    assert initialization["finite_joint_logp_required"] is True

    sampling = common["sampling"]
    assert sampling == {
        "backend": "pymc",
        "sampler": "one pymc.Slice step per parameter in parameter_order",
        "chains": 4,
        "tune": 1000,
        "draws": 4000,
        "cores": 1,
        "blas_cores": 1,
        "discard_tuned_samples": False,
        "progressbar": False,
        "compute_convergence_checks_during_sampling": True,
        "compute_log_likelihood_during_sampling": False,
    }
    assert common["posterior_predictive"] == {
        "draws_per_chain": 100,
        "draw_selection": (
            "integer draws=100 resolves to zero-based posterior draw indices "
            "[0, 100) in every chain"
        ),
        "kind": "response",
        "safe_mode": True,
        "include_group_specific": True,
        "inplace": False,
        "random_seed": (
            "the preregistered posterior_predictive_seed for the scenario or run"
        ),
    }
    assert common["diagnostic_gates"] == {
        "maximum_rhat_exclusive": 1.01,
        "minimum_bulk_ess_exclusive": 400.0,
        "minimum_tail_ess_exclusive": 400.0,
        "maximum_mcse_over_posterior_sd_exclusive": 0.05,
    }
    assert common["summaries"]["arviz_summary_call"] == (
        "arviz.summary(data, var_names=[v_x, v_y, a, t], sample_dims=[chain, "
        "draw], kind=all, ci_prob=0.94, ci_kind=hdi, round_to=none, skipna=false)"
    )
    assert common["summaries"]["hdi_truth_containment"] == (
        "inclusive lower <= truth <= upper"
    )
    assert common["summaries"]["diagnostic_sample_dims"] == ["chain", "draw"]
    assert common["summaries"]["diagnostic_columns"] == {
        "rhat": "r_hat",
        "bulk_ess": "ess_bulk",
        "tail_ess": "ess_tail",
        "mcse_over_posterior_sd": "mcse_mean / sd",
    }
    assert common["summaries"]["diagnostic_estimators"] == {
        "rhat": (
            "arviz.rhat(data, var_names=[v_x, v_y, a, t], sample_dims=[chain, "
            "draw], method=rank)"
        ),
        "bulk_ess": (
            "arviz.ess(data, var_names=[v_x, v_y, a, t], sample_dims=[chain, "
            "draw], method=bulk, relative=false, prob=null)"
        ),
        "tail_ess": (
            "arviz.ess(data, var_names=[v_x, v_y, a, t], sample_dims=[chain, "
            "draw], method=tail, relative=false, prob=null)"
        ),
        "mcse_mean": (
            "arviz.mcse(data, var_names=[v_x, v_y, a, t], sample_dims=[chain, "
            "draw], method=mean, prob=null)"
        ),
        "posterior_sd": "numpy.std over pooled chain and draw values with ddof=1",
    }

    environment = common["environment"]
    assert environment["development_base_revision"] == (
        "cdc6e8841bdfc72b4fadac0b7f6e8db7fa386374"
    )
    assert environment["current_pyproject_sha256"] == (
        "16825707e994ecf68c2ccad2d30916993ce449878e97ecf67866c85f481b1b4a"
    )
    assert environment["protocol_freeze_revision_policy"] == (
        "record the commit that first contains these exact protocol bytes before "
        "dataset recovery"
    )
    assert environment["hssm_revision_policy"] == (
        "clean execution commit descending from the recorded protocol freeze commit"
    )
    assert environment["jeam_revision"] == CURRENT_JEAM_REVISION
    assert environment["python_version"] == "3.12.13"
    assert environment["pytensor_floatx"] == "float64"
    assert environment["jax_enable_x64"] is True
    assert environment["package_versions"] == {
        "hssm": "0.4.0",
        "jeam": "0.1.0",
        "pymc": "6.3.1",
        "arviz": "1.3.0",
        "bambi": "0.20.0",
        "pytensor": "3.3.0",
        "numpy": "2.4.6",
        "pandas": "3.0.5",
        "scipy": "1.18.0",
        "xarray": "2026.7.0",
        "numba": "0.66.0",
        "llvmlite": "0.48.0",
        "h5netcdf": "1.8.1",
        "h5py": "3.16.0",
        "jax": "0.11.0",
        "numpyro": "0.21.0",
    }
    assert environment["lock_file"] == "uv.lock"
    assert environment["lock_policy"].startswith(
        "resolve the frozen pyproject and JEAM revision into uv.lock"
    )
    assert environment["preexecution_manifest_file"] == "preexecution.json"
    assert environment["preexecution_freeze_policy"] == (
        "before dataset recovery or v2b generation, commit the executor, verifier, "
        "protocol, tests, manifest schema, and uv.lock; record that commit, tree, "
        "and every source SHA256 in immutable preexecution.json"
    )
    assert environment["post_observation_change_policy"].startswith(
        "after any generated or recovered dataset is inspected"
    )


def test_mixing_v2a_uses_paired_prefixes_and_three_new_seed_blocks():
    """V2a should isolate retained-draw budget from dataset information."""
    study = _load_spec()["studies"]["mixing_v2a"]

    assert study["preregistered_hypothesis"] == (
        "The v1 a/t diagnostic failures may be sensitive to retained-draw count "
        "or to changed random streams and runtime under ordered Slice; this arm "
        "compares gate status without identifying causality."
    )
    assert study["data_source"] == "all four byte-identical v1 datasets"
    assert study["dataset_recovery_method"] == {
        "generator": (
            "hssm.integrations.jeam.simulate_projected_spherical_diffusion backed "
            "by the pinned JEAM revision"
        ),
        "scenario_inputs": (
            "use each historical_bindings.datasets truth and data_seed with "
            "n_replicas=400"
        ),
        "resolved_jeam_simulator_arguments": {
            "threshold_dynamic": "fixed",
            "decay": 0.0,
            "threshold_function": None,
            "s_v": 0.0,
            "s_t": 0.0,
            "sigma": 1.0,
            "dt": 0.001,
            "n_sample": 400,
            "max_time": 20.0,
            "random_state": "the historical data_seed",
        },
        "attempts_per_scenario": 1,
        "serialization": "numpy.save(path, float64_array, allow_pickle=false)",
        "write_and_hash_before_array_inspection": True,
        "alternate_environment_seed_or_algorithm_permitted": False,
        "omitted_nonfinite_or_invalid_output_policy": (
            "fail and retain the recovery attempt; do not resimulate"
        ),
    }
    assert study["optimizer_run"] is False
    assert study["objective_parity_rerun"] is False
    assert study["prior_predictive_draws"] == 100
    assert study["posterior_predictive_draws_per_chain"] == 100
    assert study["windows"] == {
        "paired_prefix_draws": 1000,
        "full_draws": 4000,
        "prefix_is_first_retained_draws_of_full_run": True,
        "prefix_extraction": (
            "for each chain, posterior and sample_stats "
            "isel(draw=slice(0, 1000)) after warmup"
        ),
        "thinning_or_resampling": False,
    }
    assert study["prefix_gate_interpretation"] == (
        "compare only frozen diagnostic-gate status; fixed ESS floors make "
        "draw-count dependence partly mechanical and do not identify the cause "
        "of the historical v1 failure"
    )

    scenarios = study["scenarios"]
    assert [row["name"] for row in scenarios] == list(EXPECTED_DATASETS)
    for scenario_index, scenario in enumerate(scenarios, start=1):
        assert scenario["prior_predictive_seed"] == 411000 + scenario_index
        assert [block["id"] for block in scenario["sampling_blocks"]] == [1, 2, 3]
        for block in scenario["sampling_blocks"]:
            block_id = block["id"]
            assert block["chain_seeds"] == [
                210000 + scenario_index * 1000 + block_id * 10 + chain
                for chain in range(1, 5)
            ]
            assert block["posterior_predictive_seed"] == (
                310000 + scenario_index * 1000 + block_id * 10 + 1
            )

    assert study["decision_matrix"] == {
        "any_full_window_fails": "ordered-Slice mixing remains unresolved",
        "all_full_windows_pass_and_any_prefix_fails": (
            "at least one frozen diagnostic gate is sensitive to retained-draw "
            "count in the new traces; the cause of the v1 failure remains unresolved"
        ),
        "all_full_and_prefix_windows_pass": (
            "the new seed and runtime combination passes both windows; the cause "
            "of the v1 failure remains unresolved"
        ),
    }
    assert study["claims_excluded"] == [
        "v_y information or identifiability resolved",
        "simulation-based calibration",
        "fixed-PSDM public support or promotion",
    ]


def test_information_v2b_freezes_factorial_design_and_diagnostics():
    """V2b should vary information axes only after the sampler prerequisite."""
    study = _load_spec()["studies"]["information_v2b"]

    assert study["execution_prerequisite"] == (
        "every mixing_v2a full window passes every diagnostic gate"
    )
    assert study["preregistered_hypothesis"] == (
        "Increasing from the exact 400-row prefix to 1,200 rows may contract "
        "marginal v_y posterior uncertainty across the frozen a and v_y grid "
        "once the sampler prerequisite is satisfied."
    )
    assert study["data_source"] == (
        "eight independent 1,200-row source datasets generated across a, v_y, "
        "and replicate only after this protocol and preexecution manifest are "
        "committed; each n400 dataset is nested within its matched n1200 source"
    )
    assert study["optimizer_run"] is False
    assert study["factors"] == {
        "trials": [400, 1200],
        "a": [0.7, 1.5],
        "v_y": [0.45, 1.25],
        "independent_data_replicates_per_a_v_y_pair": 2,
        "trial_count_pairing": (
            "n400 is the exact first-400-row prefix of its matched n1200 dataset"
        ),
    }
    assert study["fixed_truth"] == {"v_x": 0.6, "t": 0.1}
    assert study["data_generation"] == {
        "generator": (
            "hssm.integrations.jeam.simulate_projected_spherical_diffusion backed "
            "by the pinned JEAM revision"
        ),
        "generate_rows_per_a_v_y_replicate": 1200,
        "hssm_adapter_arguments": {
            "theta": "one [v_x, v_y, a, t] truth row for the a, v_y pair",
            "random_state": "the source_data_seed",
            "n_replicas": 1200,
        },
        "resolved_jeam_simulator_arguments": {
            "threshold_dynamic": "fixed",
            "decay": 0.0,
            "threshold_function": None,
            "s_v": 0.0,
            "s_t": 0.0,
            "sigma": 1.0,
            "dt": 0.001,
            "n_sample": 1200,
            "max_time": 20.0,
            "random_state": "the source_data_seed",
        },
        "source_rows_semantics": "zero-based half-open [start, stop)",
        "n400_policy": "take source[:400] from the retained n1200 source dataset",
        "n1200_policy": "take source[:1200] from the retained source dataset",
        "one_generator_call_per_a_v_y_replicate": True,
        "dtype": "float64",
        "columns": ["rt", "response"],
        "write_and_hash_source_and_both_slices_before_model_construction": True,
        "source_hash_binds_both_trial_count_artifacts": True,
        "resimulation_after_an_array_is_returned": False,
        "omitted_nonfinite_or_invalid_output_policy": (
            "fail and retain the source attempt; never resimulate under the same "
            "or a replacement seed"
        ),
    }

    expected_names = [
        f"n{trials}_a{str(a).replace('.', 'p')}_vy{str(v_y).replace('.', 'p')}"
        for trials in (400, 1200)
        for a in (0.7, 1.5)
        for v_y in (0.45, 1.25)
    ]
    cells = study["design_cells"]
    assert [cell["name"] for cell in cells] == expected_names
    factor_pairs = ((0.7, 0.45), (0.7, 1.25), (1.5, 0.45), (1.5, 1.25))
    for cell_index, cell in enumerate(cells, start=1):
        assert len(cell["replicates"]) == 2
        assert cell["trials"] in (400, 1200)
        factor_pair_index = (cell_index - 1) % 4 + 1
        a, v_y = factor_pairs[factor_pair_index - 1]
        assert cell["truth"] == {"v_x": 0.6, "v_y": v_y, "a": a, "t": 0.1}
        for replicate_index, replicate in enumerate(cell["replicates"], start=1):
            assert replicate == {
                "id": replicate_index,
                "source_data_seed": (
                    610000 + factor_pair_index * 100 + replicate_index
                ),
                "source_rows": [0, cell["trials"]],
                "chain_seeds": [
                    710000 + cell_index * 100 + replicate_index * 10 + chain
                    for chain in range(1, 5)
                ],
                "prior_predictive_seed": (810000 + cell_index * 100 + replicate_index),
                "posterior_predictive_seed": (
                    910000 + cell_index * 100 + replicate_index
                ),
            }

    assert study["posterior_information_diagnostics"] == {
        "marginal_v_y_sd_over_uniform_prior_sd": True,
        "posterior_sd_required_finite_and_positive": True,
        "absolute_v_y_correlations_with": ["v_x", "a", "t"],
        "posterior_mean_and_sd_calculation": (
            "for each run, pool the aligned 4 by 4,000 full-window retained v_y "
            "draws; use the arithmetic mean and sample standard deviation with ddof=1"
        ),
        "correlation_calculation": (
            "for each run and nuisance parameter, pool its aligned retained draws "
            "with v_y, compute the ordinary Pearson correlation, and report its "
            "absolute value"
        ),
        "correlation_role": (
            "descriptive ridge geometry only; not an acceptance gate or an "
            "identifiability test"
        ),
        "hdi_calculation": (
            "for each run, flatten the 4 by 4,000 full-window retained v_y draws "
            "and call arviz.hdi with prob=0.94, method=nearest, circular=false, "
            "and skipna=false"
        ),
        "truth_in_hdi": "inclusive lower <= generating v_y <= upper",
        "matched_trial_count_comparison": (
            "compare n1200 with its exact n400 data prefix within each a, v_y, "
            "and replicate"
        ),
        "likelihood_profile_or_optimizer": False,
    }
    assert study["gate_aggregation"] == {
        "diagnostics": "every parameter in every run passes separately",
        "cell_absolute_bias": (
            "for each of the eight trials by a by v_y cells, absolute value of "
            "the mean of its two replicate posterior means minus generating v_y; "
            "gate the maximum cell value"
        ),
        "cell_rmse": (
            "for each of the eight trials by a by v_y cells, square root of the "
            "mean squared posterior-mean error across its two replicates; gate the "
            "maximum cell value"
        ),
        "overall_hdi_coverage": "16 v_y intervals",
        "per_design_cell_hdi_coverage": (
            "two v_y intervals per trials, a, and v_y cell"
        ),
        "cell_posterior_sd_over_prior_sd": (
            "for each of the eight trials by a by v_y cells, median of its two "
            "replicate posterior-SD to prior-SD ratios; gate the maximum cell value"
        ),
        "trial_count_ratio": (
            "eight separate ratios, each matched replicate n1200 posterior SD "
            "divided by the posterior SD from its exact n400 data prefix; every "
            "ratio must pass"
        ),
        "nuisance_correlation": (
            "report all 48 run-by-nuisance absolute correlations without gating"
        ),
    }
    assert study["gate_membership"] == {
        "global_prerequisite": ["all_sampling_diagnostics_pass"],
        "recovery": [
            "minimum_overall_94_percent_hdi_coverage",
            "minimum_94_percent_hdi_coverage_per_design_cell",
            "maximum_cell_absolute_bias",
            "maximum_cell_rmse",
        ],
        "trial_information": [
            "maximum_cell_median_posterior_sd_over_prior_sd",
            "maximum_each_matched_replicate_1200_to_400_posterior_sd_ratio",
        ],
    }
    assert study["diagnostic_failure_policy"] == (
        "if all_sampling_diagnostics_pass is false, set both recovery and "
        "trial_information false and select otherwise without interpreting "
        "posterior information metrics"
    )
    assert study["gate_value_policy"] == (
        "recompute from raw retained draws without rounding; any missing, NaN, "
        "or nonfinite required metric fails its containing gate and the attempt "
        "is retained"
    )
    assert study["acceptance_threshold_semantics"] == (
        "all v_y recovery and trial-information minimum and maximum thresholds "
        "are inclusive; common sampling diagnostics retain their common exclusive "
        "semantics"
    )
    assert math.isclose(study["v_y_acceptance"]["prior_sd"], math.sqrt(0.75))
    assert study["v_y_acceptance"] == {
        "all_sampling_diagnostics_pass": True,
        "minimum_overall_94_percent_hdi_coverage": 0.75,
        "minimum_94_percent_hdi_coverage_per_design_cell": 0.5,
        "maximum_cell_absolute_bias": 0.2,
        "maximum_cell_rmse": 0.3,
        "prior_sd": math.sqrt(0.75),
        "maximum_cell_median_posterior_sd_over_prior_sd": 0.6,
        "maximum_each_matched_replicate_1200_to_400_posterior_sd_ratio": 0.8,
    }
    assert study["decision_matrix"] == {
        "recovery_and_trial_information_gates_pass": (
            "seeded v_y recovery plus posterior contraction and trial-count "
            "sensitivity are supported within this factorial design; "
            "identifiability, calibration, and promotion remain unestablished"
        ),
        "trial_information_only": (
            "posterior contraction and trial-count sensitivity pass, but seeded "
            "v_y recovery is not demonstrated"
        ),
        "recovery_without_trial_information_contrast": (
            "seeded v_y recovery passes, but the preregistered contraction or "
            "trial-count sensitivity gate does not"
        ),
        "otherwise": (
            "v_y recovery and trial-count information remain unresolved within "
            "this design"
        ),
    }
    assert study["claims_excluded"] == [
        "simulation-based calibration",
        "portable sampler superiority",
        "structural or practical identifiability within or beyond the frozen grid",
        "fixed-PSDM public support or promotion",
    ]


def test_new_seed_allocation_is_exact_unique_and_historical_disjoint():
    """Only the preregistered nested datasets may share a random stream."""
    data_seeds, inference_seeds = _allocated_seeds(_load_spec())

    assert len(data_seeds) == 16
    assert len(set(data_seeds)) == 8
    assert all(data_seeds.count(seed) == 2 for seed in set(data_seeds))
    assert len(inference_seeds) == 160
    assert len(inference_seeds) == len(set(inference_seeds))
    assert set(data_seeds).isdisjoint(inference_seeds)
    assert (set(data_seeds) | set(inference_seeds)).isdisjoint(V1_SEEDS)


def test_raw_evidence_and_failure_retention_are_preregistered():
    """A compact summary alone must never be accepted as v2 evidence."""
    evidence = _load_spec()["common_contract"]["evidence"]

    assert evidence["atomic_checkpoints"] is True
    assert evidence["retain_every_failed_attempt"] is True
    assert evidence["canonical_root"] == (
        "benchmarks/evidence/jeam_fixed_psdm_recovery_v2"
    )
    assert evidence["arm_directories"] == {
        "mixing_v2a": "mixing_v2a",
        "information_v2b": "information_v2b",
    }
    assert evidence["publication_policy"] == (
        "canonical evidence is incomplete until every payload plus its "
        "network-free verifier is committed under canonical_root; local-only or "
        "compact-only retention is insufficient"
    )
    assert evidence["completed_attempt_required_groups"] == [
        "prior",
        "prior_predictive",
        "warmup_posterior",
        "warmup_sample_stats",
        "posterior",
        "sample_stats",
        "posterior_predictive",
        "observed_data",
    ]
    assert evidence["failed_attempt_checkpoint_policy"] == (
        "retain every successfully written stage before failure, record the "
        "failed stage, and never treat a partial checkpoint as gate-passing evidence"
    )
    assert evidence["required_bundle_files"] == [
        "manifest.json",
        "preexecution.json",
        "environment.txt",
        "uv.lock",
        "result.json",
    ]
    assert evidence["required_scenario_files"] == [
        "dataset.npy",
        "measurements.json",
        "raw.nc",
    ]
    assert evidence["record_sampler_classes_and_variable_order"] is True
    assert evidence["record_runtime_import_paths_and_source_hashes"] is True
    assert evidence["record_complete_environment_and_lock"] is True
    assert evidence["compact_only_result_is_sufficient"] is False
    assert evidence["verifier_contract"] == {
        "json": (
            "UTF-8, reject duplicate keys and nonfinite constants, and serialize "
            "generated JSON with sort_keys=true, ensure_ascii=false, "
            "separators=[comma,colon], allow_nan=false, and exactly one terminal "
            "newline"
        ),
        "authentication": (
            "freeze the protocol SHA256 and exact safe inventory, require regular "
            "non-symlink files, snapshot and hash every byte once before parsing, "
            "and reject size or digest mismatch"
        ),
        "recomputation": (
            "recompute every scientific summary and gate from authenticated "
            "datasets and raw groups using verifier-owned thresholds; never trust "
            "stored gate booleans or thresholds"
        ),
        "execution_boundary": (
            "network-free and import no HSSM, JEAM, PyMC, benchmark runner, or "
            "model stack"
        ),
    }

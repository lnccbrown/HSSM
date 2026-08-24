"""Fast contracts for the multi-scenario JEAM recovery protocol."""

import json
from dataclasses import asdict, astuple, replace
from inspect import signature

import numpy as np
import pytest

pytest.importorskip("jeam")

import scripts.benchmark_jeam_repeated_recovery as repeated
from scripts.benchmark_jeam_bayesian_recovery import (
    PredictiveRecovery,
    PriorPredictiveCheck,
    SliceDiagnostics,
)
from scripts.benchmark_jeam_objective_parity import (
    PARAMETER_ORDER,
    FitSummary,
)
from scripts.benchmark_jeam_repeated_recovery import (
    DEFAULT_MAXITER,
    DEFAULT_POPSIZE,
    DEFAULT_REPEATED_DRAWS,
    DEFAULT_REPEATED_PREDICTIVE_DRAWS,
    DEFAULT_REPEATED_PRIOR_DRAWS,
    DEFAULT_REPEATED_TUNE,
    DEFAULT_SCENARIOS,
    DEFAULT_THRESHOLDS,
    DEFAULT_TRIALS,
    HDI_PROBABILITY,
    PRIOR_RT_QUANTILES,
    RT_QUANTILES,
    GateResult,
    ScenarioParameterResult,
    ScenarioResult,
    ScenarioRuntime,
    _mcse_sd_ratio,
    aggregate_results,
    evaluate_gate,
)

LIMITS = DEFAULT_THRESHOLDS


def _parameter(name: str, truth: float, error: float) -> ScenarioParameterResult:
    posterior_sd = 0.2
    mcse_mean = posterior_sd * LIMITS.maximum_mcse_sd_ratio
    return ScenarioParameterResult(
        name=name,
        truth=truth,
        jeam_fixed_budget_estimate=truth + error,
        posterior_mean=truth - error,
        posterior_sd=posterior_sd,
        interval_lower=truth - 0.2,
        interval_upper=truth + 0.2,
        hdi_contains_truth=True,
        rhat=LIMITS.maximum_rhat,
        ess_bulk=LIMITS.minimum_bulk_ess,
        ess_tail=LIMITS.minimum_tail_ess,
        mcse_mean=mcse_mean,
        mcse_sd_ratio=LIMITS.maximum_mcse_sd_ratio,
        ess_bulk_per_second=10.0,
    )


def _scenario(name: str, offset: float = 0.0, error: float = 0.0) -> ScenarioResult:
    truth = tuple(1.0 + offset + index for index in range(len(PARAMETER_ORDER)))
    parameters = tuple(
        _parameter(parameter_name, value, error)
        for parameter_name, value in zip(PARAMETER_ORDER, truth, strict=True)
    )
    predictive = PredictiveRecovery(
        rt_probabilities=(0.1, 0.5, 0.9),
        observed_rt_quantiles=(0.0, 0.0, 0.0),
        predictive_rt_quantiles=(LIMITS.maximum_rt_quantile_absolute_error,) * 3,
        observed_mean_angle=0.0,
        predictive_mean_angle=LIMITS.maximum_mean_angle_distance,
        mean_angle_distance=LIMITS.maximum_mean_angle_distance,
        observed_resultant_length=0.0,
        predictive_resultant_length=LIMITS.maximum_resultant_length_absolute_error,
    )
    fit = FitSummary(truth, 10.0, 900, 14)
    return ScenarioResult(
        name=name,
        truth=truth,
        data_seed=101,
        optimizer_seed=202,
        chain_seeds=(301, 302, 303, 304),
        prior_seed=501,
        predictive_seed=401,
        direct_objectives=(10.0,),
        compiled_objectives=(10.0,),
        objective_candidates=(truth,),
        maximum_objective_absolute_error=LIMITS.objective_absolute_error,
        direct_jeam_fixed_budget_optimizer=fit,
        compiled_hssm_fixed_budget_optimizer=fit,
        maximum_optimizer_parameter_absolute_error=(
            LIMITS.optimizer_parameter_absolute_error
        ),
        optimizer_objective_absolute_error=(LIMITS.optimizer_objective_absolute_error),
        minimum_observed_rt=0.2,
        initial_point=(1.0, 0.1, 0.0, 0.0),
        initial_logp=-10.0,
        parameters=parameters,
        slice_diagnostics=SliceDiagnostics(("nstep_in", "nstep_out"), 1.0, 1.0),
        prior_predictive=PriorPredictiveCheck(
            shape=(1, 100, 20, 2),
            all_finite=True,
            minimum_rt=np.nextafter(0.0, 1.0),
            maximum_rt=2.0,
            minimum_angle=-np.pi,
            maximum_angle=np.nextafter(np.pi, -np.inf),
            rt_probabilities=(0.5, 0.9),
            observed_rt_quantiles=(1.0, 1.0),
            prior_rt_quantiles=(0.1, 20.0),
            prior_to_observed_rt_ratios=(0.1, 20.0),
        ),
        predictive=predictive,
        runtime=ScenarioRuntime(1.0, 1.0, 0.25, 2.0, 0.5),
    )


def test_protocol_defaults_are_frozen_independently():
    """The protocol must not silently change scenarios or gate policy."""
    actual = tuple(
        (
            scenario.name,
            scenario.truth,
            scenario.data_seed,
            scenario.optimizer_seed,
            scenario.chain_seeds,
            scenario.prior_seed,
            scenario.predictive_seed,
        )
        for scenario in DEFAULT_SCENARIOS
    )
    # fmt: off
    expected = (
        ("baseline_asymmetric", (1.20, 0.15, 0.80, -0.50), 1492, 8675309, (3101, 3102, 3103, 3104), 6101, 7291),  # noqa: E501
        ("reversed_drift", (1.05, 0.10, -0.90, 0.65), 2603, 54021, (4201, 4202, 4203, 4204), 7101, 8291),  # noqa: E501
        ("high_threshold_strong_drift", (1.60, 0.22, 1.25, 0.30), 3714, 64031, (5301, 5302, 5303, 5304), 8101, 9291),  # noqa: E501
        ("low_threshold_negative_drift", (0.75, 0.07, -0.45, -1.15), 4825, 74041, (6401, 6402, 6403, 6404), 9101, 10291),  # noqa: E501
    )
    # fmt: on
    assert actual == expected
    protocol = (
        DEFAULT_TRIALS,
        DEFAULT_REPEATED_TUNE,
        DEFAULT_REPEATED_DRAWS,
        DEFAULT_REPEATED_PRIOR_DRAWS,
        DEFAULT_REPEATED_PREDICTIVE_DRAWS,
        DEFAULT_MAXITER,
        DEFAULT_POPSIZE,
        HDI_PROBABILITY,
        tuple(PRIOR_RT_QUANTILES),
        tuple(RT_QUANTILES),
    )
    assert protocol == (
        300,
        1_000,
        1_000,
        100,
        40,
        14,
        15,
        0.94,
        (0.5, 0.9),
        (0.1, 0.5, 0.9),
    )
    # fmt: off
    assert astuple(LIMITS) == (
        5e-5, 1e-12, 5e-5, 1.01, 500.0, 500.0, 0.75, 0.05, 0.1, 20.0, 0.12, 0.10, 0.08,
        (0.12, 0.04, 0.20, 0.20), (0.18, 0.05, 0.28, 0.28),
    )
    # fmt: on
    limits = asdict(LIMITS)
    assert (limits["maximum_absolute_bias"], limits["maximum_rmse"]) == (
        {"a": 0.12, "t": 0.04, "v_x": 0.20, "v_y": 0.20},
        {"a": 0.18, "t": 0.05, "v_x": 0.28, "v_y": 0.28},
    )
    with pytest.raises(KeyError):
        LIMITS.maximum_absolute_bias["unknown"]


def test_aggregate_results_use_fixed_budget_and_hdi_terminology():
    """Aggregation must separate optimizer and posterior summaries."""
    scenarios = (
        _scenario("first", error=0.1),
        _scenario("second", offset=1.0, error=0.1),
    )
    aggregate = aggregate_results(scenarios)

    for parameter in aggregate:
        assert (
            parameter.jeam_fixed_budget_bias,
            parameter.jeam_fixed_budget_rmse,
            parameter.hssm_posterior_bias,
            parameter.hssm_posterior_rmse,
            parameter.hdi_inclusion_fraction,
            parameter.maximum_mcse_sd_ratio,
        ) == pytest.approx((0.1, 0.1, -0.1, 0.1, 1.0, 0.05))
    assert _mcse_sd_ratio(0.01, 0.2) == pytest.approx(0.05)
    assert all(np.isinf(_mcse_sd_ratio(0.1, scale)) for scale in (0.0, -1.0, np.nan))
    with pytest.raises(ValueError, match="At least one completed scenario"):
        aggregate_results(())


def test_gate_accepts_exact_boundaries_and_retained_diagnostics():
    """Every declared inclusive boundary must pass with diagnostics retained."""
    scenarios = (_scenario("boundary"),)
    aggregate = tuple(
        replace(
            parameter,
            jeam_fixed_budget_bias=LIMITS.maximum_absolute_bias[parameter.name],
            jeam_fixed_budget_rmse=LIMITS.maximum_rmse[parameter.name],
            hssm_posterior_bias=-LIMITS.maximum_absolute_bias[parameter.name],
            hssm_posterior_rmse=LIMITS.maximum_rmse[parameter.name],
            hdi_inclusion_fraction=LIMITS.minimum_hdi_inclusion_fraction,
        )
        for parameter in aggregate_results(scenarios)
    )
    gate = evaluate_gate(scenarios, aggregate, LIMITS, total_runtime_seconds=4.0)

    assert gate == GateResult(passed=True, failures=())
    assert scenarios[0].slice_diagnostics.sample_stats == ("nstep_in", "nstep_out")
    assert scenarios[0].predictive.rt_probabilities == (0.1, 0.5, 0.9)


@pytest.mark.parametrize(
    "predictive",
    (
        replace(_scenario("grid").predictive, rt_probabilities=(0.2, 0.5, 0.8)),
        replace(_scenario("length").predictive, observed_rt_quantiles=(0.0, 0.0)),
    ),
)
def test_gate_rejects_predictive_quantile_schema(predictive):
    """Wrong probability grids and mismatched quantile lengths must fail cleanly."""
    scenario = replace(_scenario("invalid predictive schema"), predictive=predictive)

    gate = evaluate_gate((scenario,), aggregate_results((scenario,)), LIMITS)

    assert not gate.passed
    assert gate.failures == (
        "invalid predictive schema: posterior predictive RT quantile schema",
    )


def test_gate_rejects_invalid_slice_mcse_and_nonfinite_metrics():
    """Invalid sampler, precision, and finite-value diagnostics must fail."""
    scenario = _scenario("invalid")
    epsilon = 1e-6
    first = replace(
        scenario.parameters[0],
        posterior_sd=float("nan"),
        mcse_sd_ratio=LIMITS.maximum_mcse_sd_ratio + epsilon,
    )
    scenario = replace(
        scenario,
        maximum_objective_absolute_error=LIMITS.objective_absolute_error + epsilon,
        maximum_optimizer_parameter_absolute_error=(
            LIMITS.optimizer_parameter_absolute_error + epsilon
        ),
        optimizer_objective_absolute_error=(
            LIMITS.optimizer_objective_absolute_error + epsilon
        ),
        initial_point=(1.55, 2.0, 0.0, 0.0),
        initial_logp=float("nan"),
        parameters=(first, *scenario.parameters[1:]),
        predictive=replace(
            scenario.predictive,
            predictive_rt_quantiles=(
                LIMITS.maximum_rt_quantile_absolute_error + epsilon,
            )
            * 3,
            mean_angle_distance=LIMITS.maximum_mean_angle_distance + epsilon,
            predictive_resultant_length=(
                LIMITS.maximum_resultant_length_absolute_error + epsilon
            ),
        ),
        slice_diagnostics=replace(
            scenario.slice_diagnostics,
            sample_stats=("nstep_in",),
            mean_steps_out=0.0,
        ),
        runtime=replace(scenario.runtime, hssm_predictive_seconds=float("inf")),
        prior_predictive=replace(
            scenario.prior_predictive,
            shape=(5, 20, 2),
            all_finite=False,
            minimum_rt=0.0,
            maximum_angle=np.pi,
            rt_probabilities=(0.5,),
            prior_to_observed_rt_ratios=(
                np.nextafter(0.1, 0.0),
                np.nextafter(20.0, np.inf),
            ),
        ),
    )
    first, *remaining = aggregate_results((scenario,))
    bias_limit = LIMITS.maximum_absolute_bias[first.name]
    rmse_limit = LIMITS.maximum_rmse[first.name]
    aggregate = (
        replace(
            first,
            jeam_fixed_budget_bias=bias_limit + epsilon,
            jeam_fixed_budget_rmse=rmse_limit + epsilon,
            hssm_posterior_bias=-(bias_limit + epsilon),
            hssm_posterior_rmse=rmse_limit + epsilon,
            hdi_inclusion_fraction=LIMITS.minimum_hdi_inclusion_fraction - epsilon,
            maximum_rhat=LIMITS.maximum_rhat + epsilon,
            minimum_bulk_ess=LIMITS.minimum_bulk_ess - epsilon,
            minimum_tail_ess=LIMITS.minimum_tail_ess - epsilon,
        ),
        *remaining,
    )

    gate = evaluate_gate(
        (scenario,), aggregate, LIMITS, total_runtime_seconds=float("nan")
    )

    failure_text = "\n".join(gate.failures)
    # fmt: off
    expected_failures = (
        "Slice sample statistics", "Slice steps", "MCSE/SD", "resolved initial point support",  # noqa: E501
        "objective parity", "optimizer parameter parity", "optimizer objective parity",  # noqa: E501
        "posterior predictive RT", "posterior predictive mean angle", "posterior predictive resultant length", "initial logp",  # noqa: E501
        "JEAM fixed-budget optimizer bias", "JEAM fixed-budget optimizer RMSE",  # noqa: E501
        "HSSM posterior bias", "HSSM posterior RMSE", "HDI inclusion fraction", "R-hat", "bulk ESS", "tail ESS",  # noqa: E501
        "prior predictive finiteness", "prior predictive shape", "prior predictive RT support", "prior predictive RT quantiles",  # noqa: E501
        "prior predictive RT scale", "prior predictive angular support", "nonfinite metric",  # noqa: E501
    )
    # fmt: on
    assert all(expected in failure_text for expected in expected_failures)


@pytest.fixture
def benchmark_execution(monkeypatch):
    """Run canonical orchestration with every expensive scenario stubbed."""
    template = _scenario("serialized")
    calls = []

    def fake_run_scenario(scenario, **kwargs):
        calls.append((scenario, kwargs))
        return replace(template, name=scenario.name)

    monkeypatch.setattr(repeated, "run_scenario", fake_run_scenario)
    monkeypatch.setattr(
        repeated.objective, "_installed_jeam_revision", lambda: "revision"
    )
    return repeated.run_benchmark(), calls


def test_schema_v2_serialization_is_canonical_and_finite(
    benchmark_execution, monkeypatch
):
    """The schema must state and serialize the exact canonical execution."""
    result, calls = benchmark_execution
    assert not signature(repeated.run_benchmark).parameters
    assert tuple(scenario.name for scenario, _ in calls) == tuple(
        scenario.name for scenario in DEFAULT_SCENARIOS
    )
    expected_kwargs = {
        "trials": DEFAULT_TRIALS,
        "tune": DEFAULT_REPEATED_TUNE,
        "draws": DEFAULT_REPEATED_DRAWS,
        "prior_draws": DEFAULT_REPEATED_PRIOR_DRAWS,
        "predictive_draws": DEFAULT_REPEATED_PREDICTIVE_DRAWS,
        "optimizer_maxiter": DEFAULT_MAXITER,
        "optimizer_popsize": DEFAULT_POPSIZE,
    }
    assert all(kwargs == expected_kwargs for _, kwargs in calls)

    monkeypatch.setattr("sys.argv", ["benchmark"])
    assert vars(repeated._parse_args()) == {"output": None, "compact": False}
    monkeypatch.setattr("sys.argv", ["benchmark", "--scenario", "baseline"])
    with pytest.raises(SystemExit):
        repeated._parse_args()

    payload = json.loads(repeated._json_payload(result, compact=True))
    assert [
        payload[name]
        for name in (
            "schema_version",
            "trials_per_scenario",
            "chains",
            "tune",
            "draws",
            "hdi_probability",
            "prior_draws",
            "predictive_draws",
            "optimizer_maxiter",
            "optimizer_popsize",
        )
    ] == [2, 300, 4, 1_000, 1_000, 0.94, 100, 40, 14, 15]
    assert "four-scenario deterministic recovery smoke" in payload["benchmark"]
    assert "not a calibration study" in payload["interpretation"]
    assert payload["reproduction_command"].endswith(
        "benchmarks/results/jeam_repeated_recovery_v2.json"
    )
    assert tuple(item["name"] for item in payload["scenarios"]) == tuple(
        scenario.name for scenario in DEFAULT_SCENARIOS
    )
    serialized = payload["scenarios"][0]
    assert (serialized["prior_seed"], serialized["predictive_seed"]) == (501, 401)
    assert serialized["initial_point"] == [1.0, 0.1, 0.0, 0.0]
    assert {"prior_predictive", "predictive", "slice_diagnostics"} <= serialized.keys()
    assert "mle" not in json.dumps(payload).lower()

    with pytest.raises(ValueError, match="Out of range float values"):
        repeated._json_payload(
            replace(result, total_runtime_seconds=np.nan), compact=True
        )
    assert "posterior_sd" in asdict(result.scenarios[0].parameters[0])


def test_cli_writes_compact_output_and_enforces_the_gate(
    benchmark_execution, monkeypatch, tmp_path, capsys
):
    """The CLI must write strict JSON before reporting a failed gate."""
    result, _ = benchmark_execution
    output = tmp_path / "nested" / "recovery.json"
    monkeypatch.setattr(repeated, "run_benchmark", lambda: result)
    monkeypatch.setattr("sys.argv", ["benchmark", "--output", str(output), "--compact"])

    repeated.main()

    text = output.read_text(encoding="utf-8")
    assert capsys.readouterr().out.strip() == str(output)
    assert text.endswith("\n") and text.count("\n") == 1
    assert json.loads(text)["gate"] == {"passed": True, "failures": []}

    failed = replace(result, gate=GateResult(False, ("forced failure",)))
    monkeypatch.setattr(repeated, "run_benchmark", lambda: failed)
    with pytest.raises(SystemExit, match="forced failure"):
        repeated.main()

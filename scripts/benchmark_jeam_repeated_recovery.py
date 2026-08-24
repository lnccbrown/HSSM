"""Define the JEAM/HSSM four-scenario deterministic recovery-smoke protocol.

The frozen scenarios span opposing drifts and different threshold/nondecision-time
scales. Fixed-budget JEAM estimates remain distinct from HSSM posterior means; the
compiled optimizer checks objective preservation only. This is a non-calibration
protocol definition, not durable evidence or an execution entry point.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from numbers import Real

import numpy as np

from scripts import benchmark_jeam_bayesian_recovery as bayesian
from scripts import benchmark_jeam_objective_parity as objective

DEFAULT_TRIALS = objective.DEFAULT_TRIALS
DEFAULT_MAXITER = objective.DEFAULT_MAXITER
DEFAULT_POPSIZE = objective.DEFAULT_POPSIZE
HDI_PROBABILITY = bayesian.HDI_PROBABILITY
PRIOR_RT_QUANTILES = bayesian.PRIOR_RT_QUANTILES
RT_QUANTILES = bayesian.RT_QUANTILES
DEFAULT_REPEATED_TUNE = 1_000
DEFAULT_REPEATED_DRAWS = 1_000
DEFAULT_REPEATED_PRIOR_DRAWS = bayesian.DEFAULT_PRIOR_DRAWS
DEFAULT_REPEATED_PREDICTIVE_DRAWS = 40


@dataclass(frozen=True)
class Scenario:
    """One deterministic data-generating and inference configuration."""

    name: str
    truth: tuple[float, float, float, float]
    data_seed: int
    optimizer_seed: int
    chain_seeds: tuple[int, int, int, int]
    prior_seed: int
    predictive_seed: int


DEFAULT_SCENARIOS = (
    Scenario(
        name="baseline_asymmetric",
        truth=(1.20, 0.15, 0.80, -0.50),
        data_seed=1492,
        optimizer_seed=8675309,
        chain_seeds=(3101, 3102, 3103, 3104),
        prior_seed=6101,
        predictive_seed=7291,
    ),
    Scenario(
        name="reversed_drift",
        truth=(1.05, 0.10, -0.90, 0.65),
        data_seed=2603,
        optimizer_seed=54021,
        chain_seeds=(4201, 4202, 4203, 4204),
        prior_seed=7101,
        predictive_seed=8291,
    ),
    Scenario(
        name="high_threshold_strong_drift",
        truth=(1.60, 0.22, 1.25, 0.30),
        data_seed=3714,
        optimizer_seed=64031,
        chain_seeds=(5301, 5302, 5303, 5304),
        prior_seed=8101,
        predictive_seed=9291,
    ),
    Scenario(
        name="low_threshold_negative_drift",
        truth=(0.75, 0.07, -0.45, -1.15),
        data_seed=4825,
        optimizer_seed=74041,
        chain_seeds=(6401, 6402, 6403, 6404),
        prior_seed=9101,
        predictive_seed=10291,
    ),
)


@dataclass(frozen=True)
class ParameterLimits:
    """Immutable parameter-specific gate limits with name lookup."""

    a: float
    t: float
    v_x: float
    v_y: float

    def __getitem__(self, name: str) -> float:
        """Return the limit for a named model parameter."""
        return {"a": self.a, "t": self.t, "v_x": self.v_x, "v_y": self.v_y}[name]


@dataclass(frozen=True)
class RecoveryThresholds:
    """Predeclared smoke-test and numerical-parity criteria."""

    objective_absolute_error: float
    optimizer_parameter_absolute_error: float
    optimizer_objective_absolute_error: float
    maximum_rhat: float
    minimum_bulk_ess: float
    minimum_tail_ess: float
    minimum_hdi_inclusion_fraction: float
    maximum_mcse_sd_ratio: float
    minimum_prior_to_observed_rt_ratio: float
    maximum_prior_to_observed_rt_ratio: float
    maximum_rt_quantile_absolute_error: float
    maximum_mean_angle_distance: float
    maximum_resultant_length_absolute_error: float
    maximum_absolute_bias: ParameterLimits
    maximum_rmse: ParameterLimits


DEFAULT_THRESHOLDS = RecoveryThresholds(
    objective_absolute_error=5e-5,
    optimizer_parameter_absolute_error=1e-12,
    optimizer_objective_absolute_error=5e-5,
    maximum_rhat=1.01,
    minimum_bulk_ess=500.0,
    minimum_tail_ess=500.0,
    minimum_hdi_inclusion_fraction=0.75,
    maximum_mcse_sd_ratio=0.05,
    minimum_prior_to_observed_rt_ratio=bayesian.PRIOR_RT_RATIO_BOUNDS[0],
    maximum_prior_to_observed_rt_ratio=bayesian.PRIOR_RT_RATIO_BOUNDS[1],
    maximum_rt_quantile_absolute_error=0.12,
    maximum_mean_angle_distance=0.10,
    maximum_resultant_length_absolute_error=0.08,
    maximum_absolute_bias=ParameterLimits(a=0.12, t=0.04, v_x=0.20, v_y=0.20),
    maximum_rmse=ParameterLimits(a=0.18, t=0.05, v_x=0.28, v_y=0.28),
)


@dataclass(frozen=True)
class ScenarioParameterResult:
    """Fixed-budget optimizer and posterior summaries for one parameter."""

    name: str
    truth: float
    jeam_fixed_budget_estimate: float
    posterior_mean: float
    posterior_sd: float
    interval_lower: float
    interval_upper: float
    hdi_contains_truth: bool
    rhat: float
    ess_bulk: float
    ess_tail: float
    mcse_mean: float
    mcse_sd_ratio: float
    ess_bulk_per_second: float


@dataclass(frozen=True)
class ScenarioRuntime:
    """Descriptive wall-clock timings for one scenario."""

    direct_jeam_fixed_budget_optimizer_seconds: float
    compiled_hssm_fixed_budget_optimizer_seconds: float
    hssm_prior_predictive_seconds: float
    hssm_sampling_seconds: float
    hssm_predictive_seconds: float


@dataclass(frozen=True)
class ScenarioResult:
    """Complete recovery-smoke result for one generating scenario."""

    name: str
    truth: tuple[float, ...]
    data_seed: int
    optimizer_seed: int
    chain_seeds: tuple[int, ...]
    prior_seed: int
    predictive_seed: int
    direct_objectives: tuple[float, ...]
    compiled_objectives: tuple[float, ...]
    objective_candidates: tuple[tuple[float, ...], ...]
    maximum_objective_absolute_error: float
    direct_jeam_fixed_budget_optimizer: objective.FitSummary
    compiled_hssm_fixed_budget_optimizer: objective.FitSummary
    maximum_optimizer_parameter_absolute_error: float
    optimizer_objective_absolute_error: float
    minimum_observed_rt: float
    initial_point: tuple[float, ...]
    initial_logp: float
    parameters: tuple[ScenarioParameterResult, ...]
    slice_diagnostics: bayesian.SliceDiagnostics
    prior_predictive: bayesian.PriorPredictiveCheck
    predictive: bayesian.PredictiveRecovery
    runtime: ScenarioRuntime


@dataclass(frozen=True)
class AggregateParameterResult:
    """Multi-scenario descriptive metrics for one parameter."""

    name: str
    scenarios: int
    jeam_fixed_budget_bias: float
    jeam_fixed_budget_rmse: float
    hssm_posterior_bias: float
    hssm_posterior_rmse: float
    hdi_inclusion_fraction: float
    maximum_rhat: float
    minimum_bulk_ess: float
    minimum_tail_ess: float
    maximum_mcse_sd_ratio: float
    mean_bulk_ess_per_second: float


@dataclass(frozen=True)
class GateResult:
    """Outcome of all predeclared numerical and scientific checks."""

    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class MultiScenarioRecoveryResult:
    """Machine-readable multi-scenario recovery-smoke protocol result."""

    schema_version: int
    benchmark: str
    interpretation: str
    generated_at_utc: str
    reproduction_command: str
    parameter_order: tuple[str, ...]
    jeam_revision: str
    pytensor_floatx: str
    sampler: str
    trials_per_scenario: int
    chains: int
    tune: int
    draws: int
    hdi_probability: float
    prior_draws: int
    predictive_draws: int
    optimizer_maxiter: int
    optimizer_popsize: int
    thresholds: RecoveryThresholds
    scenarios: tuple[ScenarioResult, ...]
    aggregate: tuple[AggregateParameterResult, ...]
    total_runtime_seconds: float
    gate: GateResult


def _mcse_sd_ratio(mcse_mean: float, posterior_sd: float) -> float:
    """Return a scale-aware Monte Carlo error, invalidating nonpositive scales."""
    return mcse_mean / posterior_sd if posterior_sd > 0.0 else math.inf


def aggregate_results(
    scenarios: Sequence[ScenarioResult],
) -> tuple[AggregateParameterResult, ...]:
    """Aggregate descriptive errors, HDI inclusion, and diagnostics by parameter."""
    if not scenarios:
        raise ValueError("At least one completed scenario is required.")
    by_name = {
        name: [
            next(
                parameter for parameter in scenario.parameters if parameter.name == name
            )
            for scenario in scenarios
        ]
        for name in objective.PARAMETER_ORDER
    }
    aggregate: list[AggregateParameterResult] = []
    for name, parameters in by_name.items():
        truth = np.asarray([parameter.truth for parameter in parameters])
        fixed_budget = np.asarray(
            [parameter.jeam_fixed_budget_estimate for parameter in parameters]
        )
        posterior = np.asarray([parameter.posterior_mean for parameter in parameters])
        fixed_budget_error = fixed_budget - truth
        posterior_error = posterior - truth
        aggregate.append(
            AggregateParameterResult(
                name=name,
                scenarios=len(parameters),
                jeam_fixed_budget_bias=float(np.mean(fixed_budget_error)),
                jeam_fixed_budget_rmse=float(
                    np.sqrt(np.mean(np.square(fixed_budget_error)))
                ),
                hssm_posterior_bias=float(np.mean(posterior_error)),
                hssm_posterior_rmse=float(np.sqrt(np.mean(np.square(posterior_error)))),
                hdi_inclusion_fraction=float(
                    np.mean([parameter.hdi_contains_truth for parameter in parameters])
                ),
                maximum_rhat=float(
                    np.max([parameter.rhat for parameter in parameters])
                ),
                minimum_bulk_ess=float(
                    np.min([parameter.ess_bulk for parameter in parameters])
                ),
                minimum_tail_ess=float(
                    np.min([parameter.ess_tail for parameter in parameters])
                ),
                maximum_mcse_sd_ratio=float(
                    np.max([parameter.mcse_sd_ratio for parameter in parameters])
                ),
                mean_bulk_ess_per_second=float(
                    np.mean([parameter.ess_bulk_per_second for parameter in parameters])
                ),
            )
        )
    return tuple(aggregate)


def _nonfinite_metric_paths(value: object, path: str) -> Iterator[str]:
    """Yield paths to nonfinite numeric values in nested protocol results."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        for name, item in value.items():
            yield from _nonfinite_metric_paths(item, f"{path}.{name}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            yield from _nonfinite_metric_paths(item, f"{path}[{index}]")
    elif isinstance(value, Real) and not isinstance(value, (bool, int)):
        if not math.isfinite(float(value)):
            yield path


def _scenario_gate_failures(
    scenario: ScenarioResult,
    thresholds: RecoveryThresholds,
) -> list[str]:
    """Return parity, predictive, sampler, and MCSE failures for one scenario."""
    predictive = scenario.predictive
    rt_error = max(
        abs(predicted - observed)
        for predicted, observed in zip(
            predictive.predictive_rt_quantiles,
            predictive.observed_rt_quantiles,
            strict=True,
        )
    )
    checks = {
        "objective parity": (
            scenario.maximum_objective_absolute_error,
            thresholds.objective_absolute_error,
        ),
        "optimizer parameter parity": (
            scenario.maximum_optimizer_parameter_absolute_error,
            thresholds.optimizer_parameter_absolute_error,
        ),
        "optimizer objective parity": (
            scenario.optimizer_objective_absolute_error,
            thresholds.optimizer_objective_absolute_error,
        ),
        "posterior predictive RT quantiles": (
            rt_error,
            thresholds.maximum_rt_quantile_absolute_error,
        ),
        "posterior predictive mean angle": (
            predictive.mean_angle_distance,
            thresholds.maximum_mean_angle_distance,
        ),
        "posterior predictive resultant length": (
            abs(
                predictive.predictive_resultant_length
                - predictive.observed_resultant_length
            ),
            thresholds.maximum_resultant_length_absolute_error,
        ),
    }
    failures = [
        f"{scenario.name}: {label}"
        for label, (value, limit) in checks.items()
        if value > limit
    ]
    diagnostics = scenario.slice_diagnostics
    if diagnostics.sample_stats != ("nstep_in", "nstep_out"):
        failures.append(f"{scenario.name}: Slice sample statistics")
    if diagnostics.mean_steps_in <= 0.0 or diagnostics.mean_steps_out <= 0.0:
        failures.append(f"{scenario.name}: Slice steps")
    failures.extend(
        f"{scenario.name}: {parameter.name} MCSE/SD"
        for parameter in scenario.parameters
        if parameter.mcse_sd_ratio > thresholds.maximum_mcse_sd_ratio
    )
    failures.extend(
        f"{scenario.name}: {failure}"
        for failure in bayesian._preflight_failures(
            scenario.minimum_observed_rt,
            scenario.initial_point,
            scenario.initial_logp,
            scenario.prior_predictive,
            prior_rt_ratio_bounds=(
                thresholds.minimum_prior_to_observed_rt_ratio,
                thresholds.maximum_prior_to_observed_rt_ratio,
            ),
        )
    )
    return failures


def evaluate_gate(
    scenarios: Sequence[ScenarioResult],
    aggregate: Sequence[AggregateParameterResult],
    thresholds: RecoveryThresholds,
    *,
    total_runtime_seconds: float | None = None,
) -> GateResult:
    """Evaluate every predeclared criterion and retain all failure messages."""
    failures = [
        failure
        for scenario in scenarios
        for failure in _scenario_gate_failures(scenario, thresholds)
    ]
    for parameter in aggregate:
        bias_limit = thresholds.maximum_absolute_bias[parameter.name]
        rmse_limit = thresholds.maximum_rmse[parameter.name]
        for estimator, bias, rmse in (
            (
                "JEAM fixed-budget optimizer",
                parameter.jeam_fixed_budget_bias,
                parameter.jeam_fixed_budget_rmse,
            ),
            (
                "HSSM posterior",
                parameter.hssm_posterior_bias,
                parameter.hssm_posterior_rmse,
            ),
        ):
            if abs(bias) > bias_limit:
                failures.append(f"{parameter.name}: {estimator} bias")
            if rmse > rmse_limit:
                failures.append(f"{parameter.name}: {estimator} RMSE")
        if parameter.hdi_inclusion_fraction < thresholds.minimum_hdi_inclusion_fraction:
            failures.append(f"{parameter.name}: HDI inclusion fraction")
        if parameter.maximum_rhat > thresholds.maximum_rhat:
            failures.append(f"{parameter.name}: R-hat")
        if parameter.minimum_bulk_ess < thresholds.minimum_bulk_ess:
            failures.append(f"{parameter.name}: bulk ESS")
        if parameter.minimum_tail_ess < thresholds.minimum_tail_ess:
            failures.append(f"{parameter.name}: tail ESS")
    finite_metrics = {
        "thresholds": thresholds,
        "scenarios": tuple(scenarios),
        "aggregate": tuple(aggregate),
    }
    if total_runtime_seconds is not None:
        finite_metrics["total_runtime_seconds"] = total_runtime_seconds
    failures.extend(
        f"nonfinite metric: {path}"
        for path in _nonfinite_metric_paths(finite_metrics, "result")
    )
    return GateResult(passed=not failures, failures=tuple(failures))

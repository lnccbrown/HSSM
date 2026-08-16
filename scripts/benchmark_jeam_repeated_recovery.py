"""Run repeated JEAM MLE and HSSM Bayesian recovery benchmarks.

Run from the HSSM repository with the prototype dependency group installed::

    uv run --group jeam-prototype python \
        -m scripts.benchmark_jeam_repeated_recovery \
        --output benchmarks/results/jeam_repeated_recovery.json

The four default scenarios are deliberately deterministic and span opposing drift
directions, different threshold/nondecision-time scales, and values away from the
default prior centers. JEAM maximum likelihood and the HSSM posterior mean are reported
as distinct estimators. The compiled HSSM optimizer is used only to verify that HSSM
preserves JEAM's objective; it is not treated as a third estimator.

This is a scientific benchmark, not an ordinary per-PR CI test. The command emits a
machine-readable result and exits nonzero when a predeclared gate fails.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import pytensor

import hssm
from scripts.benchmark_jeam_bayesian_recovery import (
    RecoveryResult,
    run_recovery,
)
from scripts.benchmark_jeam_objective_parity import (
    DEFAULT_MAXITER,
    DEFAULT_POPSIZE,
    DEFAULT_TRIALS,
    PARAMETER_ORDER,
    FitSummary,
    _fit,
    _installed_jeam_revision,
    make_compiled_hssm_objective,
    make_direct_objective,
    optimization_bounds,
    simulate_dataset,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_REPEATED_TUNE = 1_000
DEFAULT_REPEATED_DRAWS = 1_000
DEFAULT_REPEATED_PREDICTIVE_DRAWS = 40


@dataclass(frozen=True)
class Scenario:
    """One deterministic data-generating and inference configuration."""

    name: str
    truth: tuple[float, float, float, float]
    data_seed: int
    optimizer_seed: int
    chain_seeds: tuple[int, int, int, int]
    predictive_seed: int


DEFAULT_SCENARIOS = (
    Scenario(
        name="baseline_asymmetric",
        truth=(1.20, 0.15, 0.80, -0.50),
        data_seed=1492,
        optimizer_seed=8675309,
        chain_seeds=(3101, 3102, 3103, 3104),
        predictive_seed=7291,
    ),
    Scenario(
        name="reversed_drift",
        truth=(1.05, 0.10, -0.90, 0.65),
        data_seed=2603,
        optimizer_seed=54021,
        chain_seeds=(4201, 4202, 4203, 4204),
        predictive_seed=8291,
    ),
    Scenario(
        name="high_threshold_strong_drift",
        truth=(1.60, 0.22, 1.25, 0.30),
        data_seed=3714,
        optimizer_seed=64031,
        chain_seeds=(5301, 5302, 5303, 5304),
        predictive_seed=9291,
    ),
    Scenario(
        name="low_threshold_negative_drift",
        truth=(0.75, 0.07, -0.45, -1.15),
        data_seed=4825,
        optimizer_seed=74041,
        chain_seeds=(6401, 6402, 6403, 6404),
        predictive_seed=10291,
    ),
)


@dataclass(frozen=True)
class RecoveryThresholds:
    """Predeclared repeated-recovery and numerical-parity criteria."""

    objective_absolute_error: float
    optimizer_parameter_absolute_error: float
    optimizer_objective_absolute_error: float
    maximum_rhat: float
    minimum_bulk_ess: float
    minimum_tail_ess: float
    minimum_interval_coverage: float
    maximum_absolute_bias: dict[str, float]
    maximum_rmse: dict[str, float]


DEFAULT_THRESHOLDS = RecoveryThresholds(
    objective_absolute_error=5e-5,
    optimizer_parameter_absolute_error=1e-12,
    optimizer_objective_absolute_error=5e-5,
    maximum_rhat=1.01,
    minimum_bulk_ess=500.0,
    minimum_tail_ess=500.0,
    minimum_interval_coverage=0.75,
    maximum_absolute_bias={"a": 0.12, "t": 0.04, "v_x": 0.20, "v_y": 0.20},
    maximum_rmse={"a": 0.18, "t": 0.05, "v_x": 0.28, "v_y": 0.28},
)


@dataclass(frozen=True)
class ScenarioParameterResult:
    """MLE and posterior recovery for one parameter in one scenario."""

    name: str
    truth: float
    jeam_mle: float
    jeam_mle_error: float
    posterior_mean: float
    posterior_mean_error: float
    interval_lower: float
    interval_upper: float
    interval_covers_truth: bool
    rhat: float
    ess_bulk: float
    ess_tail: float
    mcse_mean: float
    ess_bulk_per_second: float


@dataclass(frozen=True)
class ScenarioRuntime:
    """Descriptive wall-clock timings for one scenario."""

    direct_jeam_mle_seconds: float
    compiled_hssm_mle_seconds: float
    hssm_sampling_seconds: float
    hssm_predictive_seconds: float


@dataclass(frozen=True)
class ScenarioResult:
    """Complete repeated-recovery result for one generating scenario."""

    name: str
    truth: tuple[float, ...]
    data_seed: int
    optimizer_seed: int
    chain_seeds: tuple[int, ...]
    predictive_seed: int
    direct_objectives: tuple[float, ...]
    compiled_objectives: tuple[float, ...]
    objective_candidates: tuple[tuple[float, ...], ...]
    maximum_objective_absolute_error: float
    direct_jeam_fit: FitSummary
    compiled_hssm_fit: FitSummary
    maximum_optimizer_parameter_absolute_error: float
    optimizer_objective_absolute_error: float
    parameters: tuple[ScenarioParameterResult, ...]
    runtime: ScenarioRuntime


@dataclass(frozen=True)
class AggregateParameterResult:
    """Repeated-simulation metrics for one parameter and two estimators."""

    name: str
    scenarios: int
    jeam_mle_bias: float
    jeam_mle_rmse: float
    jeam_mle_r_squared: float | None
    hssm_posterior_bias: float
    hssm_posterior_rmse: float
    hssm_posterior_r_squared: float | None
    interval_coverage: float
    maximum_rhat: float
    minimum_bulk_ess: float
    minimum_tail_ess: float
    mean_bulk_ess_per_second: float


@dataclass(frozen=True)
class GateResult:
    """Outcome of all predeclared numerical and scientific checks."""

    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class RepeatedRecoveryResult:
    """Machine-readable repeated-recovery benchmark artifact."""

    schema_version: int
    benchmark: str
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
    predictive_draws: int
    optimizer_maxiter: int
    optimizer_popsize: int
    thresholds: RecoveryThresholds
    scenarios: tuple[ScenarioResult, ...]
    aggregate: tuple[AggregateParameterResult, ...]
    total_runtime_seconds: float
    gate: GateResult


def _r_squared(truth: np.ndarray, estimates: np.ndarray) -> float | None:
    """Return supplementary coefficient of determination across scenarios."""
    residual_sum = float(np.sum(np.square(estimates - truth)))
    total_sum = float(np.sum(np.square(truth - np.mean(truth))))
    return float(1.0 - residual_sum / total_sum) if total_sum > 0.0 else None


def _scenario_parameters(
    recovery: RecoveryResult,
    mle: Sequence[float],
) -> tuple[ScenarioParameterResult, ...]:
    """Combine JEAM MLE and HSSM posterior summaries without conflating them."""
    mle_by_name = dict(zip(PARAMETER_ORDER, mle, strict=True))
    return tuple(
        ScenarioParameterResult(
            name=parameter.name,
            truth=parameter.truth,
            jeam_mle=float(mle_by_name[parameter.name]),
            jeam_mle_error=float(mle_by_name[parameter.name] - parameter.truth),
            posterior_mean=parameter.posterior_mean,
            posterior_mean_error=parameter.mean_error,
            interval_lower=parameter.interval_lower,
            interval_upper=parameter.interval_upper,
            interval_covers_truth=(
                parameter.interval_lower <= parameter.truth <= parameter.interval_upper
            ),
            rhat=parameter.rhat,
            ess_bulk=parameter.ess_bulk,
            ess_tail=parameter.ess_tail,
            mcse_mean=parameter.mcse_mean,
            ess_bulk_per_second=parameter.ess_bulk_per_second,
        )
        for parameter in recovery.parameters
    )


def run_scenario(
    scenario: Scenario,
    *,
    trials: int,
    tune: int,
    draws: int,
    predictive_draws: int,
    optimizer_maxiter: int,
    optimizer_popsize: int,
) -> ScenarioResult:
    """Run direct JEAM MLE, compiled-objective parity, and HSSM recovery."""
    truth = np.asarray(scenario.truth, dtype=np.float64)
    data = simulate_dataset(truth=truth, trials=trials, seed=scenario.data_seed)
    direct_objective = make_direct_objective(data)
    compiled_objective = make_compiled_hssm_objective(data)
    bounds = optimization_bounds(data)

    direct_started = perf_counter()
    direct_fit = _fit(
        direct_objective,
        bounds,
        seed=scenario.optimizer_seed,
        maxiter=optimizer_maxiter,
        popsize=optimizer_popsize,
    )
    direct_seconds = perf_counter() - direct_started
    compiled_started = perf_counter()
    compiled_fit = _fit(
        compiled_objective,
        bounds,
        seed=scenario.optimizer_seed,
        maxiter=optimizer_maxiter,
        popsize=optimizer_popsize,
    )
    compiled_seconds = perf_counter() - compiled_started

    candidates = (
        tuple(float(value) for value in truth),
        direct_fit.parameters,
        compiled_fit.parameters,
    )
    candidate_arrays = tuple(np.asarray(candidate) for candidate in candidates)
    direct_objectives = tuple(direct_objective(value) for value in candidate_arrays)
    compiled_objectives = tuple(compiled_objective(value) for value in candidate_arrays)

    recovery = run_recovery(
        truth=scenario.truth,
        trials=trials,
        data_seed=scenario.data_seed,
        chains=len(scenario.chain_seeds),
        tune=tune,
        draws=draws,
        chain_seeds=scenario.chain_seeds,
        predictive_draws=predictive_draws,
        predictive_seed=scenario.predictive_seed,
    )
    if recovery.sampler != "pymc.Slice[a,t,v_x,v_y]":
        raise RuntimeError(f"Unexpected sampler: {recovery.sampler}")

    objective_error = float(
        np.max(np.abs(np.subtract(direct_objectives, compiled_objectives)))
    )
    optimizer_parameter_error = float(
        np.max(np.abs(np.subtract(direct_fit.parameters, compiled_fit.parameters)))
    )
    return ScenarioResult(
        name=scenario.name,
        truth=scenario.truth,
        data_seed=scenario.data_seed,
        optimizer_seed=scenario.optimizer_seed,
        chain_seeds=scenario.chain_seeds,
        predictive_seed=scenario.predictive_seed,
        direct_objectives=direct_objectives,
        compiled_objectives=compiled_objectives,
        objective_candidates=candidates,
        maximum_objective_absolute_error=objective_error,
        direct_jeam_fit=direct_fit,
        compiled_hssm_fit=compiled_fit,
        maximum_optimizer_parameter_absolute_error=optimizer_parameter_error,
        optimizer_objective_absolute_error=abs(
            direct_fit.objective - compiled_fit.objective
        ),
        parameters=_scenario_parameters(recovery, direct_fit.parameters),
        runtime=ScenarioRuntime(
            direct_jeam_mle_seconds=direct_seconds,
            compiled_hssm_mle_seconds=compiled_seconds,
            hssm_sampling_seconds=recovery.sampling_seconds,
            hssm_predictive_seconds=recovery.predictive_seconds,
        ),
    )


def aggregate_results(
    scenarios: Sequence[ScenarioResult],
) -> tuple[AggregateParameterResult, ...]:
    """Aggregate bias, RMSE, coverage, convergence, and efficiency by parameter."""
    if not scenarios:
        raise ValueError("At least one completed scenario is required.")
    by_name = {
        name: [
            next(
                parameter for parameter in scenario.parameters if parameter.name == name
            )
            for scenario in scenarios
        ]
        for name in PARAMETER_ORDER
    }
    aggregate = []
    for name, parameters in by_name.items():
        truth = np.asarray([parameter.truth for parameter in parameters])
        mle = np.asarray([parameter.jeam_mle for parameter in parameters])
        posterior = np.asarray([parameter.posterior_mean for parameter in parameters])
        mle_error = mle - truth
        posterior_error = posterior - truth
        aggregate.append(
            AggregateParameterResult(
                name=name,
                scenarios=len(parameters),
                jeam_mle_bias=float(np.mean(mle_error)),
                jeam_mle_rmse=float(np.sqrt(np.mean(np.square(mle_error)))),
                jeam_mle_r_squared=_r_squared(truth, mle),
                hssm_posterior_bias=float(np.mean(posterior_error)),
                hssm_posterior_rmse=float(np.sqrt(np.mean(np.square(posterior_error)))),
                hssm_posterior_r_squared=_r_squared(truth, posterior),
                interval_coverage=float(
                    np.mean(
                        [parameter.interval_covers_truth for parameter in parameters]
                    )
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
                mean_bulk_ess_per_second=float(
                    np.mean([parameter.ess_bulk_per_second for parameter in parameters])
                ),
            )
        )
    return tuple(aggregate)


def evaluate_gate(
    scenarios: Sequence[ScenarioResult],
    aggregate: Sequence[AggregateParameterResult],
    thresholds: RecoveryThresholds,
) -> GateResult:
    """Evaluate every predeclared criterion and retain all failure messages."""
    failures: list[str] = []
    for scenario in scenarios:
        if (
            scenario.maximum_objective_absolute_error
            > thresholds.objective_absolute_error
        ):
            failures.append(f"{scenario.name}: objective parity")
        if (
            scenario.maximum_optimizer_parameter_absolute_error
            > thresholds.optimizer_parameter_absolute_error
        ):
            failures.append(f"{scenario.name}: optimizer parameter parity")
        if (
            scenario.optimizer_objective_absolute_error
            > thresholds.optimizer_objective_absolute_error
        ):
            failures.append(f"{scenario.name}: optimizer objective parity")
    for parameter in aggregate:
        bias_limit = thresholds.maximum_absolute_bias[parameter.name]
        rmse_limit = thresholds.maximum_rmse[parameter.name]
        for estimator, bias, rmse in (
            ("JEAM MLE", parameter.jeam_mle_bias, parameter.jeam_mle_rmse),
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
        if parameter.interval_coverage < thresholds.minimum_interval_coverage:
            failures.append(f"{parameter.name}: interval coverage")
        if parameter.maximum_rhat > thresholds.maximum_rhat:
            failures.append(f"{parameter.name}: R-hat")
        if parameter.minimum_bulk_ess < thresholds.minimum_bulk_ess:
            failures.append(f"{parameter.name}: bulk ESS")
        if parameter.minimum_tail_ess < thresholds.minimum_tail_ess:
            failures.append(f"{parameter.name}: tail ESS")
    return GateResult(passed=not failures, failures=tuple(failures))


def run_benchmark(
    *,
    scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
    trials: int = DEFAULT_TRIALS,
    tune: int = DEFAULT_REPEATED_TUNE,
    draws: int = DEFAULT_REPEATED_DRAWS,
    predictive_draws: int = DEFAULT_REPEATED_PREDICTIVE_DRAWS,
    optimizer_maxiter: int = DEFAULT_MAXITER,
    optimizer_popsize: int = DEFAULT_POPSIZE,
    thresholds: RecoveryThresholds = DEFAULT_THRESHOLDS,
) -> RepeatedRecoveryResult:
    """Run all scenarios under float64 and return the complete artifact."""
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    scenario_names = [scenario.name for scenario in scenarios]
    if len(scenario_names) != len(set(scenario_names)):
        raise ValueError("Scenario names must be unique.")
    chain_counts = {len(scenario.chain_seeds) for scenario in scenarios}
    if len(chain_counts) != 1:
        raise ValueError("Every scenario must use the same number of chains.")
    started = perf_counter()
    original_floatx = pytensor.config.floatX
    hssm.set_floatX("float64")
    try:
        results = tuple(
            run_scenario(
                scenario,
                trials=trials,
                tune=tune,
                draws=draws,
                predictive_draws=predictive_draws,
                optimizer_maxiter=optimizer_maxiter,
                optimizer_popsize=optimizer_popsize,
            )
            for scenario in scenarios
        )
    finally:
        hssm.set_floatX(original_floatx)
    aggregate = aggregate_results(results)
    gate = evaluate_gate(results, aggregate, thresholds)
    return RepeatedRecoveryResult(
        schema_version=1,
        benchmark="JEAM fixed circular diffusion repeated recovery",
        generated_at_utc=datetime.now(UTC).isoformat(),
        reproduction_command=(
            "uv run --group jeam-prototype python "
            "-m scripts.benchmark_jeam_repeated_recovery "
            "--output benchmarks/results/jeam_repeated_recovery.json"
        ),
        parameter_order=PARAMETER_ORDER,
        jeam_revision=_installed_jeam_revision(),
        pytensor_floatx="float64",
        sampler="pymc.Slice[a,t,v_x,v_y]",
        trials_per_scenario=trials,
        chains=len(scenarios[0].chain_seeds),
        tune=tune,
        draws=draws,
        predictive_draws=predictive_draws,
        optimizer_maxiter=optimizer_maxiter,
        optimizer_popsize=optimizer_popsize,
        thresholds=thresholds,
        scenarios=results,
        aggregate=aggregate,
        total_runtime_seconds=perf_counter() - started,
        gate=gate,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--tune", type=int, default=DEFAULT_REPEATED_TUNE)
    parser.add_argument("--draws", type=int, default=DEFAULT_REPEATED_DRAWS)
    parser.add_argument(
        "--predictive-draws",
        type=int,
        default=DEFAULT_REPEATED_PREDICTIVE_DRAWS,
    )
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--popsize", type=int, default=DEFAULT_POPSIZE)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.name for scenario in DEFAULT_SCENARIOS],
        help="Run only this named scenario; repeat to select more than one.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def _json_payload(result: RepeatedRecoveryResult, *, compact: bool) -> str:
    """Serialize a finite standards-compliant result payload."""
    return json.dumps(
        asdict(result),
        indent=None if compact else 2,
        allow_nan=False,
    )


def main() -> None:
    """Run the repeated benchmark, write optional JSON, and enforce its gate."""
    args = _parse_args()
    selected = (
        tuple(
            scenario for scenario in DEFAULT_SCENARIOS if scenario.name in args.scenario
        )
        if args.scenario
        else DEFAULT_SCENARIOS
    )
    result = run_benchmark(
        scenarios=selected,
        trials=args.trials,
        tune=args.tune,
        draws=args.draws,
        predictive_draws=args.predictive_draws,
        optimizer_maxiter=args.maxiter,
        optimizer_popsize=args.popsize,
    )
    payload = _json_payload(result, compact=args.compact)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{payload}\n", encoding="utf-8")
        print(args.output)
    if not result.gate.passed:
        raise SystemExit(
            "Repeated recovery gate failed: " + "; ".join(result.gate.failures)
        )


if __name__ == "__main__":
    main()

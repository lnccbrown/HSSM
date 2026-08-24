"""Run the marked-slow Bayesian recovery benchmark for circular JEAM in HSSM.

Run from the HSSM repository with the prototype dependency group installed::

    uv run --group jeam-prototype python -m scripts.benchmark_jeam_bayesian_recovery

The result is emitted as JSON. It includes recovery, convergence, sampler-step,
posterior-predictive, runtime, and ESS-per-second diagnostics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import xarray as xr

import hssm
from scripts.benchmark_jeam_objective_parity import (
    DEFAULT_DATA_SEED,
    DEFAULT_TRIALS,
    DEFAULT_TRUTH,
    HSSM_BOUNDS,
    PARAMETER_ORDER,
    _installed_jeam_revision,
    simulate_dataset,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_CHAINS = 4
DEFAULT_TUNE = 1_500
DEFAULT_DRAWS = 1_500
HDI_PROBABILITY = 0.94
DEFAULT_PRIOR_DRAWS = 100
DEFAULT_PREDICTIVE_DRAWS = 100
DEFAULT_CHAIN_SEEDS = (3101, 3102, 3103, 3104)
DEFAULT_PRIOR_SEED = 6101
DEFAULT_PREDICTIVE_SEED = 7291
RT_QUANTILES = np.array([0.1, 0.5, 0.9])
PRIOR_RT_QUANTILES = np.array([0.5, 0.9])
PRIOR_RT_RATIO_BOUNDS = (0.1, 20.0)


@dataclass(frozen=True)
class ParameterRecovery:
    """Diagnostics for one recovered model parameter."""

    name: str
    truth: float
    posterior_mean: float
    mean_error: float
    posterior_sd: float
    interval_lower: float
    interval_upper: float
    rhat: float
    ess_bulk: float
    ess_tail: float
    mcse_mean: float
    ess_bulk_per_second: float


@dataclass(frozen=True)
class PredictiveRecovery:
    """Observed-versus-predictive RT and circular summaries."""

    rt_probabilities: tuple[float, ...]
    observed_rt_quantiles: tuple[float, ...]
    predictive_rt_quantiles: tuple[float, ...]
    observed_mean_angle: float
    predictive_mean_angle: float
    mean_angle_distance: float
    observed_resultant_length: float
    predictive_resultant_length: float


@dataclass(frozen=True)
class SliceDiagnostics:
    """Sampler-specific step statistics proving Slice, rather than NUTS, ran."""

    sample_stats: tuple[str, ...]
    mean_steps_in: float
    mean_steps_out: float


@dataclass(frozen=True)
class PriorPredictiveCheck:
    """Shape, support, and broad RT-scale prior-predictive checks."""

    shape: tuple[int, ...]
    all_finite: bool
    minimum_rt: float
    maximum_rt: float
    minimum_angle: float
    maximum_angle: float
    rt_probabilities: tuple[float, ...]
    observed_rt_quantiles: tuple[float, ...]
    prior_rt_quantiles: tuple[float, ...]
    prior_to_observed_rt_ratios: tuple[float, ...]


@dataclass(frozen=True)
class RecoveryResult:
    """Machine-readable HSSM Bayesian recovery benchmark."""

    parameter_order: tuple[str, ...]
    truth: tuple[float, ...]
    trials: int
    data_seed: int
    jeam_revision: str
    pytensor_floatx: str
    sampler: str
    chains: int
    tune: int
    draws: int
    hdi_probability: float
    chain_seeds: tuple[int, ...]
    prior_draws: int
    prior_seed: int
    predictive_draws: int
    predictive_seed: int
    minimum_observed_rt: float
    initial_point: tuple[float, ...]
    initial_logp: float
    prior_predictive_seconds: float
    sampling_seconds: float
    predictive_seconds: float
    parameters: tuple[ParameterRecovery, ...]
    slice_diagnostics: SliceDiagnostics
    prior_predictive: PriorPredictiveCheck
    predictive: PredictiveRecovery


def _circular_summary(angles: np.ndarray) -> tuple[float, float]:
    """Return circular mean direction and mean resultant length."""
    resultant = np.mean(np.exp(1j * np.asarray(angles)))
    return float(np.angle(resultant)), float(np.abs(resultant))


def _circular_distance(first: float, second: float) -> float:
    """Return the absolute shortest angular distance in radians."""
    return float(abs(np.angle(np.exp(1j * (first - second)))))


def _summary_interval_columns(summary: pd.DataFrame) -> tuple[str, str]:
    """Find ArviZ's version-specific HDI column labels."""
    lower = [
        column
        for column in summary.columns
        if column.startswith("hdi") and column.endswith("_lb")
    ]
    upper = [
        column
        for column in summary.columns
        if column.startswith("hdi") and column.endswith("_ub")
    ]
    if len(lower) != 1 or len(upper) != 1:
        raise RuntimeError("Expected one lower and one upper HDI column from ArviZ.")
    return lower[0], upper[0]


def _resolved_initial_point(minimum_observed_rt: float) -> tuple[float, ...]:
    """Return the frozen untransformed point inside model and data support."""
    values = (1.0, min(0.1, minimum_observed_rt / 2.0), 0.0, 0.0)
    in_model_bounds = all(
        HSSM_BOUNDS[name][0] < value < HSSM_BOUNDS[name][1]
        for name, value in zip(PARAMETER_ORDER, values, strict=True)
    )
    if not (
        np.isfinite(minimum_observed_rt)
        and 0.0 < values[1] < minimum_observed_rt
        and in_model_bounds
    ):
        raise ValueError("Data leave no valid deterministic initial point.")
    return values


def _preflight_failures(
    minimum_observed_rt: float,
    initial_point: tuple[float, ...],
    initial_logp: float,
    prior: PriorPredictiveCheck,
    *,
    prior_rt_ratio_bounds: tuple[float, float] = PRIOR_RT_RATIO_BOUNDS,
) -> tuple[str, ...]:
    """Return initialization and prior-predictive protocol failures."""
    failures = []
    try:
        if initial_point != _resolved_initial_point(minimum_observed_rt):
            failures.append("resolved initial point support")
    except ValueError:
        failures.append("resolved initial point support")
    if not np.isfinite(initial_logp):
        failures.append("initial logp")
    if not prior.all_finite:
        failures.append("prior predictive finiteness")
    if not (
        len(prior.shape) == 4
        and prior.shape[0] == 1
        and prior.shape[-1] == 2
        and prior.shape[1] > 0
        and prior.shape[2] > 0
    ):
        failures.append("prior predictive shape")
    if prior.minimum_rt <= 0.0 or prior.maximum_rt < prior.minimum_rt:
        failures.append("prior predictive RT support")
    if not (
        prior.rt_probabilities == tuple(PRIOR_RT_QUANTILES)
        and len(prior.observed_rt_quantiles)
        == len(prior.prior_rt_quantiles)
        == len(prior.prior_to_observed_rt_ratios)
        == len(PRIOR_RT_QUANTILES)
    ):
        failures.append("prior predictive RT quantiles")
    lower_ratio, upper_ratio = prior_rt_ratio_bounds
    if any(
        ratio < lower_ratio or ratio > upper_ratio
        for ratio in prior.prior_to_observed_rt_ratios
    ):
        failures.append("prior predictive RT scale")
    if (
        prior.minimum_angle < -np.pi
        or prior.maximum_angle >= np.pi
        or prior.maximum_angle < prior.minimum_angle
    ):
        failures.append("prior predictive angular support")
    return tuple(failures)


def _sample_with_resolved_init(
    model: hssm.HSSM,
    slice_steps: Sequence[object],
    initial_point: tuple[float, ...],
    *,
    minimum_observed_rt: float,
    initial_logp: float,
    prior_predictive: PriorPredictiveCheck,
    chains: int,
    tune: int,
    draws: int,
    chain_seeds: tuple[int, ...],
) -> object:
    """Run Slice with the same explicit supported point for every chain."""
    if failures := _preflight_failures(
        minimum_observed_rt,
        initial_point,
        initial_logp,
        prior_predictive,
    ):
        raise RuntimeError(f"Recovery preflight failed: {'; '.join(failures)}")
    return model.sample(
        sampler="pymc",
        step=slice_steps,
        initvals=dict(zip(PARAMETER_ORDER, initial_point, strict=True)),
        chains=chains,
        cores=1,
        tune=tune,
        draws=draws,
        random_seed=list(chain_seeds),
        progressbar=False,
        idata_kwargs={"log_likelihood": False},
    )


def run_recovery(
    *,
    truth: Sequence[float] = tuple(DEFAULT_TRUTH),
    trials: int = DEFAULT_TRIALS,
    data_seed: int = DEFAULT_DATA_SEED,
    chains: int = DEFAULT_CHAINS,
    tune: int = DEFAULT_TUNE,
    draws: int = DEFAULT_DRAWS,
    chain_seeds: tuple[int, ...] = DEFAULT_CHAIN_SEEDS,
    prior_draws: int = DEFAULT_PRIOR_DRAWS,
    prior_seed: int = DEFAULT_PRIOR_SEED,
    predictive_draws: int = DEFAULT_PREDICTIVE_DRAWS,
    predictive_seed: int = DEFAULT_PREDICTIVE_SEED,
) -> RecoveryResult:
    """Fit the fixed circular model and collect recovery diagnostics."""
    if len(chain_seeds) != chains:
        raise ValueError("Provide exactly one random seed per MCMC chain.")
    truth_values = np.asarray(truth, dtype=np.float64)
    if truth_values.shape != (len(PARAMETER_ORDER),):
        raise ValueError(
            f"truth must follow {PARAMETER_ORDER} and have shape "
            f"({len(PARAMETER_ORDER)},)."
        )

    data = simulate_dataset(truth=truth_values, trials=trials, seed=data_seed)
    model = hssm.HSSM(
        data=pd.DataFrame(data, columns=["rt", "response"]),
        model="circular_diffusion",
        p_outlier=None,
    )
    minimum_observed_rt = float(np.min(data[:, 0]))
    initial_point = _resolved_initial_point(minimum_observed_rt)
    initial_values = dict(zip(PARAMETER_ORDER, initial_point, strict=True))
    initial_logp = float(model.compile_logp()(initial_values))
    started = perf_counter()
    prior = model.sample_prior_predictive(
        draws=prior_draws,
        random_seed=prior_seed,
    )
    prior_predictive_seconds = perf_counter() - started
    prior_values = np.asarray(
        prior["prior_predictive"].to_dataset()["rt,response"].values
    )
    expected_prior_shape = (1, prior_draws, trials, 2)
    if prior_values.shape != expected_prior_shape:
        raise RuntimeError(
            f"Expected prior-predictive shape {expected_prior_shape}, "
            f"received {prior_values.shape}."
        )
    observed_rt = np.quantile(data[:, 0], PRIOR_RT_QUANTILES)
    prior_rt = np.quantile(prior_values[..., 0], PRIOR_RT_QUANTILES)
    prior_predictive = PriorPredictiveCheck(
        shape=prior_values.shape,
        all_finite=bool(np.all(np.isfinite(prior_values))),
        minimum_rt=float(np.min(prior_values[..., 0])),
        maximum_rt=float(np.max(prior_values[..., 0])),
        minimum_angle=float(np.min(prior_values[..., 1])),
        maximum_angle=float(np.max(prior_values[..., 1])),
        rt_probabilities=tuple(float(value) for value in PRIOR_RT_QUANTILES),
        observed_rt_quantiles=tuple(float(value) for value in observed_rt),
        prior_rt_quantiles=tuple(float(value) for value in prior_rt),
        prior_to_observed_rt_ratios=tuple(
            float(prior_value / observed_value)
            for prior_value, observed_value in zip(prior_rt, observed_rt, strict=True)
        ),
    )
    slice_steps = [
        pm.Slice(vars=[model.pymc_model[name]], model=model.pymc_model)
        for name in PARAMETER_ORDER
    ]
    started = perf_counter()
    traces = _sample_with_resolved_init(
        model,
        slice_steps,
        initial_point,
        minimum_observed_rt=minimum_observed_rt,
        initial_logp=initial_logp,
        prior_predictive=prior_predictive,
        chains=chains,
        tune=tune,
        draws=draws,
        chain_seeds=chain_seeds,
    )
    sampling_seconds = perf_counter() - started
    if not isinstance(traces, xr.DataTree):
        raise RuntimeError("Expected PyMC sampling to return an xarray DataTree.")

    summary = az.summary(
        traces,
        var_names=list(PARAMETER_ORDER),
        ci_prob=HDI_PROBABILITY,
        ci_kind="hdi",
        round_to=8,
    )
    interval_lower, interval_upper = _summary_interval_columns(summary)
    truth_by_name = dict(zip(PARAMETER_ORDER, truth_values, strict=True))
    parameter_recovery = tuple(
        ParameterRecovery(
            name=name,
            truth=float(truth_by_name[name]),
            posterior_mean=float(summary.loc[name, "mean"]),
            mean_error=float(summary.loc[name, "mean"] - truth_by_name[name]),
            posterior_sd=float(summary.loc[name, "sd"]),
            interval_lower=float(summary.loc[name, interval_lower]),
            interval_upper=float(summary.loc[name, interval_upper]),
            rhat=float(summary.loc[name, "r_hat"]),
            ess_bulk=float(summary.loc[name, "ess_bulk"]),
            ess_tail=float(summary.loc[name, "ess_tail"]),
            mcse_mean=float(summary.loc[name, "mcse_mean"]),
            ess_bulk_per_second=float(summary.loc[name, "ess_bulk"] / sampling_seconds),
        )
        for name in PARAMETER_ORDER
    )

    sample_stats = traces["sample_stats"]
    sample_stat_names = tuple(sorted(str(name) for name in sample_stats.data_vars))
    slice_diagnostics = SliceDiagnostics(
        sample_stats=sample_stat_names,
        mean_steps_in=float(np.asarray(sample_stats["nstep_in"]).mean()),
        mean_steps_out=float(np.asarray(sample_stats["nstep_out"]).mean()),
    )

    started = perf_counter()
    predictive = model.sample_posterior_predictive(
        dt=traces,
        inplace=False,
        kind="response",
        draws=predictive_draws,
        safe_mode=True,
        random_seed=predictive_seed,
    )
    predictive_seconds = perf_counter() - started
    if predictive is None:
        raise RuntimeError("Expected non-inplace posterior predictive output.")
    predictive_dataset = predictive["posterior_predictive"].to_dataset()
    values = predictive_dataset["rt,response"].values
    observed_angle, observed_resultant = _circular_summary(data[:, 1])
    predictive_angle, predictive_resultant = _circular_summary(values[..., 1])
    observed_rt = np.quantile(data[:, 0], RT_QUANTILES)
    predictive_rt = np.quantile(values[..., 0], RT_QUANTILES)
    predictive_recovery = PredictiveRecovery(
        rt_probabilities=tuple(float(value) for value in RT_QUANTILES),
        observed_rt_quantiles=tuple(float(value) for value in observed_rt),
        predictive_rt_quantiles=tuple(float(value) for value in predictive_rt),
        observed_mean_angle=observed_angle,
        predictive_mean_angle=predictive_angle,
        mean_angle_distance=_circular_distance(observed_angle, predictive_angle),
        observed_resultant_length=observed_resultant,
        predictive_resultant_length=predictive_resultant,
    )

    return RecoveryResult(
        parameter_order=PARAMETER_ORDER,
        truth=tuple(float(value) for value in truth_values),
        trials=trials,
        data_seed=data_seed,
        jeam_revision=_installed_jeam_revision(),
        pytensor_floatx=pytensor.config.floatX,
        sampler="pymc.Slice[a,t,v_x,v_y]",
        chains=chains,
        tune=tune,
        draws=draws,
        hdi_probability=HDI_PROBABILITY,
        chain_seeds=chain_seeds,
        prior_draws=prior_draws,
        prior_seed=prior_seed,
        predictive_draws=predictive_draws,
        predictive_seed=predictive_seed,
        minimum_observed_rt=minimum_observed_rt,
        initial_point=initial_point,
        initial_logp=initial_logp,
        prior_predictive_seconds=prior_predictive_seconds,
        sampling_seconds=sampling_seconds,
        predictive_seconds=predictive_seconds,
        parameters=parameter_recovery,
        slice_diagnostics=slice_diagnostics,
        prior_predictive=prior_predictive,
        predictive=predictive_recovery,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--tune", type=int, default=DEFAULT_TUNE)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--prior-draws", type=int, default=DEFAULT_PRIOR_DRAWS)
    parser.add_argument(
        "--predictive-draws", type=int, default=DEFAULT_PREDICTIVE_DRAWS
    )
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the recovery benchmark and emit JSON."""
    args = _parse_args()
    result = run_recovery(
        trials=args.trials,
        tune=args.tune,
        draws=args.draws,
        prior_draws=args.prior_draws,
        predictive_draws=args.predictive_draws,
    )
    print(json.dumps(asdict(result), indent=None if args.compact else 2))


if __name__ == "__main__":
    main()

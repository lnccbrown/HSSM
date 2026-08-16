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
    PARAMETER_ORDER,
    _installed_jeam_revision,
    simulate_dataset,
)

DEFAULT_CHAINS = 4
DEFAULT_TUNE = 1_500
DEFAULT_DRAWS = 1_500
DEFAULT_PREDICTIVE_DRAWS = 100
DEFAULT_CHAIN_SEEDS = (3101, 3102, 3103, 3104)
DEFAULT_PREDICTIVE_SEED = 7291
RT_QUANTILES = np.array([0.1, 0.5, 0.9])


@dataclass(frozen=True)
class ParameterRecovery:
    """Diagnostics for one recovered model parameter."""

    name: str
    truth: float
    posterior_mean: float
    mean_error: float
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
    chain_seeds: tuple[int, ...]
    sampling_seconds: float
    predictive_seconds: float
    parameters: tuple[ParameterRecovery, ...]
    slice_diagnostics: SliceDiagnostics
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


def run_recovery(
    *,
    trials: int = DEFAULT_TRIALS,
    data_seed: int = DEFAULT_DATA_SEED,
    chains: int = DEFAULT_CHAINS,
    tune: int = DEFAULT_TUNE,
    draws: int = DEFAULT_DRAWS,
    chain_seeds: tuple[int, ...] = DEFAULT_CHAIN_SEEDS,
    predictive_draws: int = DEFAULT_PREDICTIVE_DRAWS,
    predictive_seed: int = DEFAULT_PREDICTIVE_SEED,
) -> RecoveryResult:
    """Fit the fixed circular model and collect recovery diagnostics."""
    if len(chain_seeds) != chains:
        raise ValueError("Provide exactly one random seed per MCMC chain.")

    data = simulate_dataset(trials=trials, seed=data_seed)
    model = hssm.HSSM(
        data=pd.DataFrame(data, columns=["rt", "response"]),
        model="circular_diffusion",
        p_outlier=None,
    )
    slice_steps = [
        pm.Slice(vars=[model.pymc_model[name]], model=model.pymc_model)
        for name in PARAMETER_ORDER
    ]
    started = perf_counter()
    traces = model.sample(
        sampler="pymc",
        step=slice_steps,
        chains=chains,
        cores=1,
        tune=tune,
        draws=draws,
        random_seed=list(chain_seeds),
        progressbar=False,
        idata_kwargs={"log_likelihood": False},
    )
    sampling_seconds = perf_counter() - started
    if not isinstance(traces, xr.DataTree):
        raise RuntimeError("Expected PyMC sampling to return an xarray DataTree.")

    summary = az.summary(
        traces,
        var_names=list(PARAMETER_ORDER),
        ci_prob=0.94,
        ci_kind="hdi",
        round_to=8,
    )
    interval_lower, interval_upper = _summary_interval_columns(summary)
    truth_by_name = dict(zip(PARAMETER_ORDER, DEFAULT_TRUTH, strict=True))
    parameter_recovery = tuple(
        ParameterRecovery(
            name=name,
            truth=float(truth_by_name[name]),
            posterior_mean=float(summary.loc[name, "mean"]),
            mean_error=float(summary.loc[name, "mean"] - truth_by_name[name]),
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
        truth=tuple(float(value) for value in DEFAULT_TRUTH),
        trials=trials,
        data_seed=data_seed,
        jeam_revision=_installed_jeam_revision(),
        pytensor_floatx=pytensor.config.floatX,
        sampler="pymc.Slice[a,t,v_x,v_y]",
        chains=chains,
        tune=tune,
        draws=draws,
        chain_seeds=chain_seeds,
        sampling_seconds=sampling_seconds,
        predictive_seconds=predictive_seconds,
        parameters=parameter_recovery,
        slice_diagnostics=slice_diagnostics,
        predictive=predictive_recovery,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--tune", type=int, default=DEFAULT_TUNE)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
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
        predictive_draws=args.predictive_draws,
    )
    print(json.dumps(asdict(result), indent=None if args.compact else 2))


if __name__ == "__main__":
    main()

"""Benchmark direct JEAM against the compiled HSSM circular likelihood.

Run from the HSSM repository with the prototype dependency group installed::

    uv run --group jeam-prototype python scripts/benchmark_jeam_objective_parity.py

The command writes one JSON object to stdout. Runtime values are descriptive only; the
scientific invariant is numerical objective and bounded-optimizer parity.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pytensor
import pytensor.tensor as pt
from jeam.Models.Circular import CircularDiffusionModel
from scipy.optimize import differential_evolution

from hssm.distribution_utils.dist import make_distribution, make_likelihood_callable
from hssm.integrations.jeam import (
    logp_circular_diffusion,
    simulate_circular_diffusion,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

PARAMETER_ORDER = ("a", "t", "v_x", "v_y")
DEFAULT_TRUTH = np.array([1.20, 0.15, 0.80, -0.50], dtype=np.float64)
DEFAULT_DATA_SEED = 1492
DEFAULT_OPTIMIZER_SEED = 8675309
DEFAULT_TRIALS = 300
DEFAULT_MAXITER = 14
DEFAULT_POPSIZE = 15
NDT_SUPPORT_MARGIN = 1e-12

HSSM_PARAMETER_ORDER = ["v_x", "v_y", "a", "t"]
HSSM_BOUNDS = {
    "v_x": (-3.0, 3.0),
    "v_y": (-3.0, 3.0),
    "a": (0.1, 3.0),
    "t": (0.0, 2.0),
}


@dataclass(frozen=True)
class FitSummary:
    """Serializable optimizer result."""

    parameters: tuple[float, ...]
    objective: float
    evaluations: int
    iterations: int


@dataclass(frozen=True)
class RuntimeSummary:
    """Descriptive median pointwise objective runtimes in milliseconds."""

    direct_jeam_ms: float
    compiled_hssm_ms: float


@dataclass(frozen=True)
class BenchmarkResult:
    """Machine-readable objective and optimizer parity result."""

    parameter_order: tuple[str, ...]
    truth: tuple[float, ...]
    trials: int
    data_seed: int
    optimizer_seed: int
    jeam_revision: str
    candidate_parameters: tuple[tuple[float, ...], ...]
    direct_objectives: tuple[float, ...]
    compiled_objectives: tuple[float, ...]
    direct_fit: FitSummary
    compiled_fit: FitSummary
    runtime: RuntimeSummary


def _installed_jeam_revision() -> str:
    """Return the installed VCS revision when direct-url metadata is available."""
    distribution = importlib.metadata.distribution("jeam")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is not None:
        direct_url = json.loads(direct_url_text)
        commit_id = direct_url.get("vcs_info", {}).get("commit_id")
        if isinstance(commit_id, str):
            return commit_id
    return distribution.version


def simulate_dataset(
    truth: np.ndarray = DEFAULT_TRUTH,
    *,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_DATA_SEED,
) -> np.ndarray:
    """Simulate one deterministic fixed-CDM dataset through pinned JEAM."""
    a, t, v_x, v_y = np.asarray(truth, dtype=np.float64)
    theta = np.array([v_x, v_y, a, t], dtype=np.float64)
    return simulate_circular_diffusion(
        theta=theta,
        random_state=seed,
        n_replicas=trials,
    )


def make_direct_objective(data: np.ndarray) -> Callable[[np.ndarray], float]:
    """Return the negative summed log likelihood evaluated directly by JEAM."""
    observations = np.asarray(data, dtype=np.float64)
    model = CircularDiffusionModel(threshold_dynamic="fixed")

    def objective(parameters: np.ndarray) -> float:
        a, t, v_x, v_y = np.asarray(parameters, dtype=np.float64)
        logp = model.joint_lpdf(
            rt=observations[:, 0],
            theta=observations[:, 1],
            drift_vec=np.column_stack(
                (
                    np.full(observations.shape[0], v_x),
                    np.full(observations.shape[0], v_y),
                )
            ),
            ndt=np.full(observations.shape[0], t),
            threshold=float(a),
            decay=0.0,
            threshold_function=None,
            dt_threshold_function=None,
            s_v=0.0,
            s_t=0.0,
            sigma=1.0,
        )
        return -float(np.sum(logp, dtype=np.float64))

    return objective


def make_compiled_hssm_objective(
    data: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """Compile the actual HSSM distribution into a scalar objective."""
    likelihood = make_likelihood_callable(
        loglik=logp_circular_diffusion,
        loglik_kind="blackbox",
        backend="pytensor",
    )
    distribution = make_distribution(
        rv=simulate_circular_diffusion,
        loglik=likelihood,
        list_params=HSSM_PARAMETER_ORDER.copy(),
        bounds=HSSM_BOUNDS,
    )
    parameters = pt.dvector("parameters")
    a, t, v_x, v_y = (parameters[index] for index in range(4))
    pointwise = distribution.logp(  # pyrefly: ignore[missing-attribute]
        pt.as_tensor_variable(np.asarray(data, dtype=np.float64)),
        v_x,
        v_y,
        a,
        t,
    )
    compiled = pytensor.function([parameters], -pt.sum(pointwise))

    def objective(values: np.ndarray) -> float:
        return float(compiled(np.asarray(values, dtype=np.float64)))

    return objective


def optimization_bounds(data: np.ndarray) -> tuple[tuple[float, float], ...]:
    """Return bounds wholly inside both HSSM and dataset-specific RT support."""
    min_rt = float(np.min(np.asarray(data)[:, 0]))
    # HSSM excludes rt - t <= 1e-15 before evaluating the model likelihood.
    # Keep the optimizer endpoint comfortably inside that shared support.
    t_upper = min(HSSM_BOUNDS["t"][1], min_rt - NDT_SUPPORT_MARGIN)
    if t_upper <= 0.05:
        raise ValueError(
            "The simulated dataset leaves no valid optimization range for t."
        )
    return ((0.60, 1.80), (0.05, t_upper), (-1.50, 1.50), (-1.50, 1.50))


def _fit(
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]],
    *,
    seed: int,
    maxiter: int,
    popsize: int,
) -> FitSummary:
    """Run a deterministic fixed-budget differential-evolution fit."""
    fit: Any = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        polish=False,
        tol=0.0,
        atol=0.0,
        workers=1,
        updating="immediate",
    )
    return FitSummary(
        parameters=tuple(float(value) for value in fit.x),
        objective=float(fit.fun),
        evaluations=int(fit.nfev),
        iterations=int(fit.nit),
    )


def _median_runtime_ms(
    objective: Callable[[np.ndarray], float],
    parameters: np.ndarray,
    *,
    repeats: int,
) -> float:
    """Measure a warmed objective without turning speed into an assertion."""
    objective(parameters)
    durations = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        start = perf_counter()
        objective(parameters)
        durations[index] = perf_counter() - start
    return float(np.median(durations) * 1_000.0)


def run_benchmark(
    *,
    trials: int = DEFAULT_TRIALS,
    data_seed: int = DEFAULT_DATA_SEED,
    optimizer_seed: int = DEFAULT_OPTIMIZER_SEED,
    maxiter: int = DEFAULT_MAXITER,
    popsize: int = DEFAULT_POPSIZE,
    timing_repeats: int = 51,
) -> BenchmarkResult:
    """Run objective-grid, optimizer, and descriptive runtime comparisons."""
    data = simulate_dataset(trials=trials, seed=data_seed)
    direct = make_direct_objective(data)
    compiled = make_compiled_hssm_objective(data)
    bounds = optimization_bounds(data)
    candidates = (
        tuple(float(value) for value in DEFAULT_TRUTH),
        (1.00, 0.10, 0.40, -0.30),
        (1.50, 0.12, 1.10, -0.80),
        (
            float(DEFAULT_TRUTH[0]),
            bounds[1][1],
            float(DEFAULT_TRUTH[2]),
            float(DEFAULT_TRUTH[3]),
        ),
    )
    candidate_arrays = tuple(np.asarray(candidate) for candidate in candidates)
    direct_objectives = tuple(direct(candidate) for candidate in candidate_arrays)
    compiled_objectives = tuple(compiled(candidate) for candidate in candidate_arrays)
    direct_fit = _fit(
        direct,
        bounds,
        seed=optimizer_seed,
        maxiter=maxiter,
        popsize=popsize,
    )
    compiled_fit = _fit(
        compiled,
        bounds,
        seed=optimizer_seed,
        maxiter=maxiter,
        popsize=popsize,
    )
    runtime = RuntimeSummary(
        direct_jeam_ms=_median_runtime_ms(
            direct,
            np.asarray(direct_fit.parameters),
            repeats=timing_repeats,
        ),
        compiled_hssm_ms=_median_runtime_ms(
            compiled,
            np.asarray(compiled_fit.parameters),
            repeats=timing_repeats,
        ),
    )
    return BenchmarkResult(
        parameter_order=PARAMETER_ORDER,
        truth=tuple(float(value) for value in DEFAULT_TRUTH),
        trials=trials,
        data_seed=data_seed,
        optimizer_seed=optimizer_seed,
        jeam_revision=_installed_jeam_revision(),
        candidate_parameters=candidates,
        direct_objectives=direct_objectives,
        compiled_objectives=compiled_objectives,
        direct_fit=direct_fit,
        compiled_fit=compiled_fit,
        runtime=runtime,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--optimizer-seed", type=int, default=DEFAULT_OPTIMIZER_SEED)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--popsize", type=int, default=DEFAULT_POPSIZE)
    parser.add_argument("--timing-repeats", type=int, default=51)
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and print machine-readable JSON."""
    args = _parse_args()
    result = run_benchmark(
        trials=args.trials,
        data_seed=args.data_seed,
        optimizer_seed=args.optimizer_seed,
        maxiter=args.maxiter,
        popsize=args.popsize,
        timing_repeats=args.timing_repeats,
    )
    print(json.dumps(asdict(result), indent=None if args.compact else 2))


if __name__ == "__main__":
    main()

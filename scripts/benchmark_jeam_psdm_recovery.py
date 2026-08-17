"""Run the preregistered fixed-PSDM optimizer and Slice recovery study.

The canonical protocol lives in
``benchmarks/specs/jeam_fixed_psdm_recovery_v1.json``. The runner generates each
dataset once, preserves its bytes, compares direct JEAM and compiled HSSM
objectives/optimizers, saves raw posterior traces before post-processing, and writes a
compact JSON artifact suitable for version control.

Run one explicitly non-canonical smoke scenario first::

    python -m scripts.benchmark_jeam_psdm_recovery \
        --smoke --scenario baseline_asymmetric \
        --work-dir /tmp/jeam-psdm-smoke \
        --output /tmp/jeam-psdm-smoke.json

Run the canonical study only from a clean commit after publishing the protocol::

    python -m scripts.benchmark_jeam_psdm_recovery \
        --work-dir /tmp/jeam-psdm-recovery-v1 \
        --output benchmarks/results/jeam_fixed_psdm_recovery_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

REPO_ROOT = Path(__file__).parents[1]
DEFAULT_SPEC_PATH = (
    REPO_ROOT / "benchmarks" / "specs" / "jeam_fixed_psdm_recovery_v1.json"
)
PACKAGE_NAMES = (
    "hssm",
    "jeam",
    "pymc",
    "arviz",
    "bambi",
    "pytensor",
    "numpy",
    "pandas",
    "scipy",
    "xarray",
)


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict, stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    """Return the SHA256 of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    """Run one read-only git query from the repository root."""
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


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


def _optional_float(value: Any) -> float | None:
    """Return a finite float or ``None`` for an unavailable diagnostic."""
    result = float(value)
    return result if np.isfinite(result) else None


def _scenario(spec: dict[str, Any], name: str) -> dict[str, Any]:
    """Resolve one named scenario."""
    try:
        return next(row for row in spec["scenarios"] if row["name"] == name)
    except StopIteration as exc:
        raise ValueError(f"Unknown scenario {name!r}.") from exc


def _configure_precision() -> None:
    """Select the protocol float64 mode before constructing a PyMC model."""
    import hssm

    hssm.set_floatX("float64")


def _simulate_dataset(scenario: dict[str, Any]) -> np.ndarray:
    """Simulate one scenario through HSSM's public fixed-PSDM adapter."""
    from hssm.integrations.jeam import simulate_projected_spherical_diffusion

    data = simulate_projected_spherical_diffusion(
        theta=np.asarray(scenario["truth"], dtype=np.float64),
        random_state=scenario["data_seed"],
        n_replicas=scenario["trials"],
    )
    data = np.asarray(data, dtype=np.float64)
    expected_shape = (scenario["trials"], 2)
    if data.shape != expected_shape:
        raise RuntimeError(
            f"Expected simulated shape {expected_shape}, got {data.shape}."
        )
    return data


def _make_model(data: np.ndarray):
    """Construct the ordinary public fixed-PSDM HSSM model."""
    import pandas as pd

    import hssm

    return hssm.HSSM(
        data=pd.DataFrame(data, columns=["rt", "response"]),
        model="projected_spherical_diffusion",
        p_outlier=None,
    )


def _prior_signature(model) -> dict[str, dict[str, float | str]]:
    """Return JSON-safe structural prior metadata from an HSSM model."""
    signature: dict[str, dict[str, float | str]] = {}
    for name in model.list_params:
        prior = model.params[name].prior
        if not hasattr(prior, "name") or not hasattr(prior, "args"):
            raise TypeError(f"Expected a Bambi prior for {name!r}.")
        signature[name] = {
            "distribution": str(prior.name),
            **{
                key: float(np.asarray(value))
                for key, value in sorted(prior.args.items())
            },
        }
    return signature


def _initval_signature(model) -> dict[str, float]:
    """Return untransformed scalar initial values in public parameter order."""
    return {name: float(np.asarray(model.initvals[name])) for name in model.list_params}


def _bounds_signature(model) -> dict[str, list[float]]:
    """Return JSON-safe registered parameter bounds."""
    return {
        name: [float(value) for value in model.params[name].bounds]
        for name in model.list_params
    }


def _assert_model_contract(model, spec: dict[str, Any]) -> dict[str, Any]:
    """Check the runtime model against the frozen public defaults."""
    expected = spec["priors_and_initialization"]
    contract = {
        "priors": _prior_signature(model),
        "initvals": _initval_signature(model),
        "bounds": _bounds_signature(model),
    }
    if contract["priors"] != expected["priors"]:
        raise AssertionError(
            f"Resolved priors differ from the protocol: {contract['priors']!r}"
        )
    if contract["initvals"] != expected["resolved_untransformed_initvals"]:
        raise AssertionError(
            f"Resolved initvals differ from the protocol: {contract['initvals']!r}"
        )
    if contract["bounds"] != expected["configured_bounds"]:
        raise AssertionError(
            f"Resolved bounds differ from the protocol: {contract['bounds']!r}"
        )
    return contract


def _make_objectives(
    data: np.ndarray,
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], float]]:
    """Return direct-JEAM and compiled-HSSM negative log likelihoods."""
    import pytensor
    import pytensor.tensor as pt
    from jeam.Models.Spherical import ProjectedSphericalDiffusionModel

    observations = np.asarray(data, dtype=np.float64)
    producer = ProjectedSphericalDiffusionModel(threshold_dynamic="fixed")

    def direct(values: np.ndarray) -> float:
        v_x, v_y, threshold, ndt = np.asarray(values, dtype=np.float64)
        pointwise = producer.joint_lpdf(
            rt=observations[:, 0],
            theta=observations[:, 1],
            drift_vec=np.column_stack(
                (
                    np.full(observations.shape[0], v_x),
                    np.full(observations.shape[0], v_y),
                )
            ),
            ndt=np.full(observations.shape[0], ndt),
            threshold=float(threshold),
            decay=0.0,
            threshold_function=None,
            dt_threshold_function=None,
            s_v=0.0,
            s_t=0.0,
            sigma=1.0,
        )
        return -float(np.sum(pointwise, dtype=np.float64))

    model = _make_model(observations)
    parameters = pt.dvector("psdm_recovery_parameters")
    pointwise = model.model_distribution.logp(
        pt.as_tensor_variable(observations),
        parameters[0],
        parameters[1],
        parameters[2],
        parameters[3],
    )
    compiled_function = pytensor.function([parameters], -pt.sum(pointwise))

    def compiled(values: np.ndarray) -> float:
        return float(compiled_function(np.asarray(values, dtype=np.float64)))

    return direct, compiled


def _optimization_bounds(data: np.ndarray, spec: dict[str, Any]):
    """Resolve optimizer bounds wholly inside observation-specific RT support."""
    frozen = spec["objective_and_optimizer"]["bounds"]
    t_upper = float(np.nextafter(np.min(data[:, 0]), -np.inf))
    if t_upper <= frozen["t_lower"]:
        raise ValueError("The dataset leaves no valid optimization interval for t.")
    return (
        tuple(frozen["v_x"]),
        tuple(frozen["v_y"]),
        tuple(frozen["a"]),
        (float(frozen["t_lower"]), t_upper),
    )


def _fit(
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]],
    *,
    seed: int,
    maxiter: int,
    popsize: int,
) -> dict[str, Any]:
    """Run the frozen fixed-budget differential-evolution optimizer."""
    from scipy.optimize import differential_evolution

    started = perf_counter()
    fit = differential_evolution(
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
    return {
        "parameters": [float(value) for value in fit.x],
        "objective": float(fit.fun),
        "evaluations": int(fit.nfev),
        "iterations": int(fit.nit),
        "seconds": perf_counter() - started,
    }


def _summary_interval_columns(summary) -> tuple[str, str]:
    """Find ArviZ's version-specific HDI column names."""
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
        raise RuntimeError("Expected one lower and one upper HDI column.")
    return lower[0], upper[0]


def _polar_unit_vector_summary(angles: np.ndarray) -> tuple[float, float]:
    """Return mean meridional unit-vector components for polar angles."""
    values = np.asarray(angles, dtype=np.float64)
    return float(np.mean(np.sin(values))), float(np.mean(np.cos(values)))


def _worker_settings(spec: dict[str, Any], smoke: bool) -> dict[str, int]:
    """Return canonical settings or an unmistakably non-canonical smoke budget."""
    execution = spec["execution"]
    optimizer = spec["objective_and_optimizer"]
    if not smoke:
        return {
            "chains": execution["chains"],
            "tune": execution["tune"],
            "draws": execution["draws"],
            "prior_predictive_draws": execution["prior_predictive_draws"],
            "posterior_predictive_draws": execution["posterior_predictive_draws"],
            "optimizer_maxiter": optimizer["maxiter"],
            "optimizer_popsize": optimizer["popsize"],
        }
    return {
        "chains": 1,
        "tune": 5,
        "draws": 5,
        "prior_predictive_draws": 2,
        "posterior_predictive_draws": 2,
        "optimizer_maxiter": 1,
        "optimizer_popsize": 5,
    }


def _run_scenario(
    spec: dict[str, Any],
    scenario: dict[str, Any],
    work_dir: Path,
    *,
    smoke: bool,
) -> dict[str, Any]:
    """Run one complete optimizer, posterior, and predictive recovery scenario."""
    import arviz as az
    import pymc as pm

    total_started = perf_counter()
    settings = _worker_settings(spec, smoke)
    data = _simulate_dataset(scenario)
    data_path = work_dir / f"data-{scenario['name']}.npy"
    np.save(data_path, data, allow_pickle=False)
    data_sha256 = _sha256(data_path)

    prior_model = _make_model(data)
    prior_contract = _assert_model_contract(prior_model, spec)
    prior_started = perf_counter()
    prior = prior_model.sample_prior_predictive(
        draws=settings["prior_predictive_draws"],
        random_seed=scenario["prior_predictive_seed"],
    )
    prior_seconds = perf_counter() - prior_started
    prior_values = np.asarray(prior["prior_predictive"]["rt,response"].values)
    ratio_probabilities = np.asarray(
        spec["preflight_gates"]["prior_predictive"]["ratio_probabilities"]
    )
    observed_rt = np.quantile(data[:, 0], ratio_probabilities)
    prior_rt = np.quantile(prior_values[..., 0], ratio_probabilities)
    prior_ratios = prior_rt / observed_rt
    prior_gate = spec["preflight_gates"]["prior_predictive"]
    prior_checks = {
        "all_values_finite": bool(np.all(np.isfinite(prior_values))),
        "all_rt_strictly_positive": bool(np.all(prior_values[..., 0] > 0.0)),
        "all_angles_in_closed_domain": bool(
            np.all((0.0 <= prior_values[..., 1]) & (prior_values[..., 1] <= np.pi))
        ),
        "rt_quantile_probabilities": ratio_probabilities.tolist(),
        "observed_rt_quantiles": observed_rt.tolist(),
        "prior_rt_quantiles": prior_rt.tolist(),
        "prior_to_observed_ratios": prior_ratios.tolist(),
        "seconds": prior_seconds,
    }
    prior_checks["passed"] = bool(
        prior_checks["all_values_finite"]
        and prior_checks["all_rt_strictly_positive"]
        and prior_checks["all_angles_in_closed_domain"]
        and np.all(prior_ratios >= prior_gate["rt_quantile_ratio_to_observed_lower"])
        and np.all(prior_ratios <= prior_gate["rt_quantile_ratio_to_observed_upper"])
    )

    model_started = perf_counter()
    model = _make_model(data)
    contract = _assert_model_contract(model, spec)
    model_build_seconds = perf_counter() - model_started
    if contract != prior_contract:
        raise AssertionError("Prior-check and sampling model contracts differ.")

    initial_point = model.initial_point()
    initial_logp = float(model.compile_logp()(initial_point))
    if not np.isfinite(initial_logp):
        raise FloatingPointError(f"Initial joint logp is not finite: {initial_logp}")

    direct_objective, compiled_objective = _make_objectives(data)
    objective_rows = []
    truth = np.asarray(scenario["truth"], dtype=np.float64)
    for offset in spec["objective_and_optimizer"]["candidate_offsets"]:
        candidate = truth + np.asarray(offset, dtype=np.float64)
        direct_value = direct_objective(candidate)
        compiled_value = compiled_objective(candidate)
        objective_rows.append(
            {
                "parameters": candidate.tolist(),
                "direct_jeam": direct_value,
                "compiled_hssm": compiled_value,
                "absolute_error": abs(direct_value - compiled_value),
            }
        )

    bounds = _optimization_bounds(data, spec)
    direct_fit = _fit(
        direct_objective,
        bounds,
        seed=scenario["optimizer_seed"],
        maxiter=settings["optimizer_maxiter"],
        popsize=settings["optimizer_popsize"],
    )
    compiled_fit = _fit(
        compiled_objective,
        bounds,
        seed=scenario["optimizer_seed"],
        maxiter=settings["optimizer_maxiter"],
        popsize=settings["optimizer_popsize"],
    )
    direct_parameters = np.asarray(direct_fit["parameters"])
    compiled_parameters = np.asarray(compiled_fit["parameters"])
    optimizer = {
        "bounds": [[float(low), float(high)] for low, high in bounds],
        "direct_jeam": direct_fit,
        "compiled_hssm": compiled_fit,
        "maximum_parameter_absolute_error": float(
            np.max(np.abs(direct_parameters - compiled_parameters))
        ),
        "objective_absolute_error": abs(
            direct_fit["objective"] - compiled_fit["objective"]
        ),
        "direct_recovery_absolute_error": np.abs(direct_parameters - truth).tolist(),
    }

    slice_steps = [
        pm.Slice(vars=[model.pymc_model[name]], model=model.pymc_model)
        for name in spec["scope"]["parameter_order"]
    ]
    sampling_started = perf_counter()
    traces = model.sample(
        sampler="pymc",
        step=slice_steps,
        init=spec["execution"]["init"],
        initvals=model.initvals,
        chains=settings["chains"],
        cores=spec["execution"]["cores"],
        blas_cores=spec["execution"]["blas_cores"],
        tune=settings["tune"],
        draws=settings["draws"],
        random_seed=scenario["chain_seeds"][: settings["chains"]],
        progressbar=spec["execution"]["progressbar"],
        compute_convergence_checks=spec["execution"][
            "compute_convergence_checks_during_sampling"
        ],
        idata_kwargs={
            "log_likelihood": spec["execution"][
                "compute_log_likelihood_during_sampling"
            ]
        },
    )
    sampling_seconds = perf_counter() - sampling_started

    trace_name = f"trace-{scenario['name']}-slice.nc"
    trace_path = work_dir / trace_name
    traces.to_netcdf(trace_path)
    trace_metadata = {
        "basename": trace_name,
        "sha256": _sha256(trace_path),
        "bytes": trace_path.stat().st_size,
        "saved_before_summary": True,
    }

    summary = az.summary(
        traces,
        var_names=spec["scope"]["parameter_order"],
        ci_prob=spec["execution"]["posterior_hdi_probability"],
        ci_kind="hdi",
        round_to=10,
    )
    interval_lower, interval_upper = _summary_interval_columns(summary)
    truth_by_name = dict(
        zip(spec["scope"]["parameter_order"], scenario["truth"], strict=True)
    )
    parameters = []
    for name in spec["scope"]["parameter_order"]:
        posterior_sd = float(summary.loc[name, "sd"])
        mcse_mean = float(summary.loc[name, "mcse_mean"])
        parameters.append(
            {
                "name": name,
                "truth": truth_by_name[name],
                "optimizer_estimate": float(
                    direct_parameters[spec["scope"]["parameter_order"].index(name)]
                ),
                "posterior_mean": float(summary.loc[name, "mean"]),
                "posterior_sd": posterior_sd,
                "hdi_lower": float(summary.loc[name, interval_lower]),
                "hdi_upper": float(summary.loc[name, interval_upper]),
                "truth_in_hdi": bool(
                    summary.loc[name, interval_lower]
                    <= truth_by_name[name]
                    <= summary.loc[name, interval_upper]
                ),
                "rhat": _optional_float(summary.loc[name, "r_hat"]),
                "ess_bulk": _optional_float(summary.loc[name, "ess_bulk"]),
                "ess_tail": _optional_float(summary.loc[name, "ess_tail"]),
                "mcse_mean": _optional_float(mcse_mean),
                "mcse_over_posterior_sd": (
                    _optional_float(mcse_mean / posterior_sd)
                    if posterior_sd > 0.0
                    else None
                ),
            }
        )

    sample_stats = traces["sample_stats"].to_dataset()
    sampler_diagnostics = {
        "sample_stats": sorted(str(name) for name in sample_stats.data_vars),
        "mean_nstep_in": float(np.asarray(sample_stats["nstep_in"]).mean()),
        "mean_nstep_out": float(np.asarray(sample_stats["nstep_out"]).mean()),
    }

    predictive_started = perf_counter()
    predictive = model.sample_posterior_predictive(
        dt=traces,
        inplace=False,
        kind="response",
        draws=settings["posterior_predictive_draws"],
        safe_mode=True,
        random_seed=scenario["posterior_predictive_seed"],
    )
    predictive_seconds = perf_counter() - predictive_started
    if predictive is None:
        raise RuntimeError("Expected non-inplace posterior predictive output.")
    predictive_values = np.asarray(
        predictive["posterior_predictive"]["rt,response"].values
    )
    probabilities = np.asarray(spec["execution"]["rt_quantile_probabilities"])
    observed_rt = np.quantile(data[:, 0], probabilities)
    predictive_rt = np.quantile(predictive_values[..., 0], probabilities)
    observed_direction = _polar_unit_vector_summary(data[:, 1])
    predictive_direction = _polar_unit_vector_summary(predictive_values[..., 1])
    predictive_summary = {
        "rt_probabilities": probabilities.tolist(),
        "observed_rt_quantiles": observed_rt.tolist(),
        "predictive_rt_quantiles": predictive_rt.tolist(),
        "rt_quantile_absolute_errors": np.abs(predictive_rt - observed_rt).tolist(),
        "observed_mean_polar_unit_vector": list(observed_direction),
        "predictive_mean_polar_unit_vector": list(predictive_direction),
        "polar_unit_vector_component_absolute_errors": np.abs(
            np.asarray(predictive_direction) - np.asarray(observed_direction)
        ).tolist(),
    }

    runtime = {
        "model_build": model_build_seconds,
        "prior_predictive": prior_seconds,
        "direct_optimizer": direct_fit["seconds"],
        "compiled_optimizer": compiled_fit["seconds"],
        "sampling": sampling_seconds,
        "posterior_predictive": predictive_seconds,
    }
    return {
        "scenario": scenario["name"],
        "truth": scenario["truth"],
        "trials": scenario["trials"],
        "data_seed": scenario["data_seed"],
        "optimizer_seed": scenario["optimizer_seed"],
        "chain_seeds": scenario["chain_seeds"][: settings["chains"]],
        "prior_predictive_seed": scenario["prior_predictive_seed"],
        "posterior_predictive_seed": scenario["posterior_predictive_seed"],
        "status": "completed",
        "smoke": smoke,
        "settings": settings,
        "data": {
            "basename": data_path.name,
            "sha256": data_sha256,
            "bytes": data_path.stat().st_size,
            "dtype": str(data.dtype),
            "shape": list(data.shape),
        },
        "model_contract": contract,
        "initial_joint_logp": initial_logp,
        "prior_predictive": prior_checks,
        "objective_candidates": objective_rows,
        "maximum_objective_absolute_error": max(
            row["absolute_error"] for row in objective_rows
        ),
        "optimizer": optimizer,
        "trace": trace_metadata,
        "parameters": parameters,
        "sampler_diagnostics": sampler_diagnostics,
        "posterior_predictive": predictive_summary,
        "runtime_seconds": {
            **runtime,
            "total": perf_counter() - total_started,
        },
    }


def _scenario_failures(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Evaluate scenario-level preflight, parity, recovery, and predictive gates."""
    name = row["scenario"]
    if row["status"] != "completed":
        return [f"{name}: scenario status is {row['status']!r}"]
    acceptance = spec["scientific_acceptance"]
    failures = []
    if not row["prior_predictive"]["passed"]:
        failures.append(f"{name}: prior predictive gate failed")
    if (
        row["maximum_objective_absolute_error"]
        > acceptance["maximum_objective_absolute_error"]
    ):
        failures.append(f"{name}: objective parity failed")
    optimizer = row["optimizer"]
    if (
        optimizer["maximum_parameter_absolute_error"]
        > acceptance["maximum_optimizer_parameter_absolute_error"]
    ):
        failures.append(f"{name}: optimizer parameter parity failed")
    if (
        optimizer["objective_absolute_error"]
        > acceptance["maximum_optimizer_objective_absolute_error"]
    ):
        failures.append(f"{name}: optimizer objective parity failed")

    parameter_order = spec["scope"]["parameter_order"]
    optimizer_limits = acceptance["optimizer_recovery"]["maximum_absolute_error"]
    for parameter, error in zip(
        parameter_order,
        optimizer["direct_recovery_absolute_error"],
        strict=True,
    ):
        if error > optimizer_limits[parameter]:
            failures.append(
                f"{name}/{parameter}: optimizer recovery error {error:.6g} exceeds "
                f"{optimizer_limits[parameter]:.6g}"
            )

    posterior = acceptance["posterior_recovery"]
    for parameter in row["parameters"]:
        prefix = f"{name}/{parameter['name']}"
        if parameter["rhat"] is None or not (
            parameter["rhat"] < posterior["maximum_rhat_exclusive"]
        ):
            failures.append(f"{prefix}: R-hat gate failed")
        if parameter["ess_bulk"] is None or not (
            parameter["ess_bulk"] > posterior["minimum_bulk_ess_exclusive"]
        ):
            failures.append(f"{prefix}: bulk ESS gate failed")
        if parameter["ess_tail"] is None or not (
            parameter["ess_tail"] > posterior["minimum_tail_ess_exclusive"]
        ):
            failures.append(f"{prefix}: tail ESS gate failed")
        if parameter["mcse_over_posterior_sd"] is None or not (
            parameter["mcse_over_posterior_sd"]
            < posterior["maximum_mcse_over_posterior_sd_exclusive"]
        ):
            failures.append(f"{prefix}: MCSE/SD gate failed")

    if set(row["sampler_diagnostics"]["sample_stats"]) != {
        "nstep_in",
        "nstep_out",
    }:
        failures.append(f"{name}: sample statistics do not identify ordered Slice")
    predictive = acceptance["posterior_predictive"]
    if (
        max(row["posterior_predictive"]["rt_quantile_absolute_errors"])
        > predictive["maximum_rt_quantile_absolute_error"]
    ):
        failures.append(f"{name}: predictive RT quantile gate failed")
    if (
        max(row["posterior_predictive"]["polar_unit_vector_component_absolute_errors"])
        > predictive["maximum_polar_unit_vector_component_absolute_error"]
    ):
        failures.append(f"{name}: predictive polar unit-vector gate failed")
    return failures


def _aggregate(
    rows: list[dict[str, Any]], spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], float]:
    """Aggregate optimizer and posterior recovery across completed scenarios."""
    aggregate = []
    coverage_values = []
    for name in spec["scope"]["parameter_order"]:
        parameter_rows = [
            next(
                parameter
                for parameter in row["parameters"]
                if parameter["name"] == name
            )
            for row in rows
        ]
        truth = np.asarray([row["truth"] for row in parameter_rows])
        optimizer = np.asarray([row["optimizer_estimate"] for row in parameter_rows])
        posterior = np.asarray([row["posterior_mean"] for row in parameter_rows])
        coverage = np.asarray([row["truth_in_hdi"] for row in parameter_rows])
        coverage_values.extend(coverage.tolist())
        aggregate.append(
            {
                "name": name,
                "scenarios": len(parameter_rows),
                "optimizer_bias": float(np.mean(optimizer - truth)),
                "optimizer_rmse": float(np.sqrt(np.mean(np.square(optimizer - truth)))),
                "posterior_bias": float(np.mean(posterior - truth)),
                "posterior_rmse": float(np.sqrt(np.mean(np.square(posterior - truth)))),
                "hdi_coverage": float(np.mean(coverage)),
                "maximum_rhat": max(row["rhat"] for row in parameter_rows),
                "minimum_bulk_ess": min(row["ess_bulk"] for row in parameter_rows),
                "minimum_tail_ess": min(row["ess_tail"] for row in parameter_rows),
                "maximum_mcse_over_posterior_sd": max(
                    row["mcse_over_posterior_sd"] for row in parameter_rows
                ),
            }
        )
    return aggregate, float(np.mean(coverage_values))


def _aggregate_failures(
    aggregate: list[dict[str, Any]],
    overall_coverage: float,
    spec: dict[str, Any],
) -> list[str]:
    """Evaluate preregistered cross-scenario recovery gates."""
    acceptance = spec["scientific_acceptance"]
    optimizer = acceptance["optimizer_recovery"]
    posterior = acceptance["posterior_recovery"]
    failures = []
    for row in aggregate:
        name = row["name"]
        if row["optimizer_rmse"] > optimizer["maximum_rmse"][name]:
            failures.append(f"{name}: optimizer RMSE gate failed")
        if abs(row["posterior_bias"]) > posterior["maximum_absolute_bias"][name]:
            failures.append(f"{name}: posterior bias gate failed")
        if row["posterior_rmse"] > posterior["maximum_rmse"][name]:
            failures.append(f"{name}: posterior RMSE gate failed")
        if (
            row["hdi_coverage"]
            < posterior["minimum_94_percent_hdi_coverage_per_parameter"]
        ):
            failures.append(f"{name}: HDI coverage gate failed")
    if overall_coverage < posterior["minimum_overall_94_percent_hdi_coverage"]:
        failures.append("overall HDI coverage gate failed")
    return failures


def _provenance(spec_path: Path) -> dict[str, Any]:
    """Collect exact software and machine metadata for the result."""
    relative_spec = spec_path.resolve().relative_to(REPO_ROOT.resolve())
    package_versions = {
        name: importlib.metadata.version(name) for name in PACKAGE_NAMES
    }
    return {
        "hssm_revision": _git("rev-parse", "HEAD"),
        "spec_commit": _git("log", "-1", "--format=%H", "--", str(relative_spec)),
        "jeam_revision": _installed_jeam_revision(),
        "pyproject_sha256": _sha256(REPO_ROOT / "pyproject.toml"),
        "python_version": platform.python_version(),
        "package_versions": package_versions,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }


def _assert_canonical_preconditions(spec_path: Path, spec: dict[str, Any]) -> None:
    """Refuse a canonical run that cannot support exact provenance."""
    if _git("status", "--porcelain"):
        raise RuntimeError("Canonical recovery requires a clean HSSM worktree.")
    provenance = _provenance(spec_path)
    frozen = spec["provenance"]
    if provenance["jeam_revision"] != frozen["jeam_revision"]:
        raise RuntimeError(
            "Installed JEAM revision does not match the protocol: "
            f"{provenance['jeam_revision']} != {frozen['jeam_revision']}"
        )
    _git("merge-base", "--is-ancestor", provenance["spec_commit"], "HEAD")


def run_study(
    *,
    spec_path: Path = DEFAULT_SPEC_PATH,
    output_path: Path,
    work_dir: Path,
    scenario_names: Sequence[str] | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run selected smoke scenarios or the complete canonical study."""
    spec = _load_json(spec_path)
    if not smoke:
        _assert_canonical_preconditions(spec_path, spec)
    _configure_precision()
    work_dir.mkdir(parents=True, exist_ok=True)
    selected = (
        [_scenario(spec, name) for name in scenario_names]
        if scenario_names
        else list(spec["scenarios"])
    )
    if not smoke and len(selected) != len(spec["scenarios"]):
        raise ValueError("Canonical recovery must run every frozen scenario.")

    started_at = datetime.now(UTC)
    rows = []
    for scenario in selected:
        try:
            rows.append(_run_scenario(spec, scenario, work_dir, smoke=smoke))
        except Exception as exc:  # noqa: BLE001 - failures are scientific output
            rows.append(
                {
                    "scenario": scenario["name"],
                    "truth": scenario["truth"],
                    "trials": scenario["trials"],
                    "status": "failed",
                    "smoke": smoke,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    completed = [row for row in rows if row["status"] == "completed"]
    aggregate: list[dict[str, Any]] = []
    overall_coverage: float | None = None
    failures = [failure for row in rows for failure in _scenario_failures(row, spec)]
    if len(completed) == len(rows):
        aggregate, overall_coverage = _aggregate(completed, spec)
        failures.extend(_aggregate_failures(aggregate, overall_coverage, spec))
    else:
        failures.append("aggregate recovery unavailable because a scenario failed")

    canonical = not smoke
    payload = {
        "schema_version": 1,
        "study_id": spec["study_id"],
        "canonical": canonical,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "started_at_utc": started_at.isoformat(),
        "spec_path": str(spec_path.resolve().relative_to(REPO_ROOT.resolve())),
        "spec_sha256": _sha256(spec_path),
        "provenance": _provenance(spec_path),
        "parameter_order": spec["scope"]["parameter_order"],
        "model": spec["scope"]["model"],
        "settings": _worker_settings(spec, smoke),
        "scenarios": rows,
        "aggregate": aggregate,
        "overall_hdi_coverage": overall_coverage,
        "total_runtime_seconds": (datetime.now(UTC) - started_at).total_seconds(),
        "gate": {
            "evaluated": canonical,
            "passed": bool(canonical and not failures),
            "failures": failures,
        },
    }
    _write_json(output_path, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the requested study and exit nonzero when a canonical gate fails."""
    args = _parse_args()
    result = run_study(
        spec_path=args.spec,
        output_path=args.output,
        work_dir=args.work_dir,
        scenario_names=args.scenarios,
        smoke=args.smoke,
    )
    status = "PASS" if result["gate"]["passed"] else "NOT PASSED"
    print(f"Wrote {args.output.name}; canonical gate: {status}")
    if result["canonical"] and not result["gate"]["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

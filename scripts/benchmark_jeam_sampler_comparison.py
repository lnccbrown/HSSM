"""Run the preregistered fixed-CDM Slice and NUTS comparison.

The canonical study is defined in
``benchmarks/specs/jeam_fixed_cdm_sampler_comparison_v1.json``. The parent process
generates and hashes each dataset, performs shared prior/objective checks, and launches
every scenario-sampler fit in a fresh Python subprocess. Raw DataTrees stay in the
requested work directory; the compact JSON result is suitable for version control.

Run a single non-canonical smoke fit first::

    uv run --python 3.12 --group jeam-prototype python -m \
      scripts.benchmark_jeam_sampler_comparison \
      --smoke --scenario baseline_asymmetric --sampler numpyro_nuts \
      --work-dir /tmp/jeam-sampler-smoke --output /tmp/jeam-sampler-smoke.json

Run the complete canonical study only after the specification commit is public::

    uv run --python 3.12 --group jeam-prototype python -m \
      scripts.benchmark_jeam_sampler_comparison \
      --work-dir /tmp/jeam-sampler-v1 \
      --output benchmarks/results/jeam_fixed_cdm_sampler_comparison_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).parents[1]
DEFAULT_SPEC_PATH = (
    REPO_ROOT / "benchmarks" / "specs" / "jeam_fixed_cdm_sampler_comparison_v1.json"
)
PACKAGE_NAMES = (
    "hssm",
    "jeam",
    "pymc",
    "jax",
    "numpyro",
    "arviz",
    "bambi",
    "pytensor",
    "numpy",
    "pandas",
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
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    """Return the SHA256 of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _sampler(spec: dict[str, Any], sampler_id: str) -> dict[str, Any]:
    """Resolve one named sampler route."""
    try:
        return next(row for row in spec["samplers"] if row["id"] == sampler_id)
    except StopIteration as exc:
        raise ValueError(f"Unknown sampler {sampler_id!r}.") from exc


def _configure_precision() -> None:
    """Select the protocol's float64 execution mode before JAX computation."""
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    host_device_flag = "--xla_force_host_platform_device_count=1"
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if xla_flags and host_device_flag not in xla_flags.split():
        raise RuntimeError(
            "The benchmark requires one visible JAX host device, but XLA_FLAGS is "
            f"already set to {xla_flags!r}."
        )
    os.environ["XLA_FLAGS"] = host_device_flag
    import jax

    import hssm

    jax.config.update("jax_enable_x64", True)
    if jax.local_device_count() != 1:
        raise RuntimeError(
            "The benchmark requires exactly one visible JAX device, found "
            f"{jax.local_device_count()}."
        )
    hssm.set_floatX("float64")


def _simulate_dataset(scenario: dict[str, Any]) -> np.ndarray:
    """Simulate one scenario through HSSM's public JEAM predictive adapter."""
    from hssm.integrations.jeam import simulate_circular_diffusion

    a, t, v_x, v_y = np.asarray(scenario["truth"], dtype=np.float64)
    data = simulate_circular_diffusion(
        theta=np.asarray([v_x, v_y, a, t], dtype=np.float64),
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


def _make_model(data: np.ndarray, likelihood: str):
    """Construct the ordinary intercept-only HSSM model used by the study."""
    import pandas as pd

    import hssm

    return hssm.HSSM(
        data=pd.DataFrame(data, columns=["rt", "response"]),
        model="circular_diffusion",
        loglik_kind=likelihood,
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
    """Return untransformed scalar initial values."""
    return {name: float(np.asarray(value)) for name, value in model.initvals.items()}


def _assert_model_contract(model, spec: dict[str, Any]) -> dict[str, Any]:
    """Check priors and initial values against the frozen protocol."""
    priors = _prior_signature(model)
    initvals = _initval_signature(model)
    if priors != spec["priors_and_initialization"]["priors"]:
        raise AssertionError(f"Resolved priors differ from the protocol: {priors!r}")
    if initvals != spec["priors_and_initialization"]["resolved_untransformed_initvals"]:
        raise AssertionError(
            f"Resolved initial values differ from the protocol: {initvals!r}"
        )
    return {"priors": priors, "initvals": initvals}


def _circular_summary(angles: np.ndarray) -> tuple[float, float]:
    """Return circular mean direction and mean resultant length."""
    resultant = np.mean(np.exp(1j * np.asarray(angles)))
    return float(np.angle(resultant)), float(np.abs(resultant))


def _circular_distance(first: float, second: float) -> float:
    """Return the absolute shortest angular distance."""
    return float(abs(np.angle(np.exp(1j * (first - second)))))


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


def _run_shared_preflight(
    spec: dict[str, Any], scenario: dict[str, Any], data_path: Path
) -> dict[str, Any]:
    """Run prior-predictive and three-way objective checks once per scenario."""
    import jax.numpy as jnp
    from jeam.likelihoods.jax_fixed import fixed_cdm_logpdf
    from jeam.Models.Circular import CircularDiffusionModel

    data = np.load(data_path, allow_pickle=False)
    blackbox_model = _make_model(data, "blackbox")
    analytical_model = _make_model(data, "analytical")
    blackbox_contract = _assert_model_contract(blackbox_model, spec)
    analytical_contract = _assert_model_contract(analytical_model, spec)
    if blackbox_contract != analytical_contract:
        raise AssertionError("Blackbox and analytical model contracts differ.")

    prior_started = perf_counter()
    prior = blackbox_model.sample_prior_predictive(
        draws=spec["execution"]["prior_predictive_draws"],
        random_seed=scenario["prior_predictive_seed"],
    )
    prior_seconds = perf_counter() - prior_started
    prior_values = np.asarray(prior["prior_predictive"]["rt,response"].values)
    ratio_probabilities = np.asarray(
        spec["preflight_gates"]["prior_predictive"]["ratio_probabilities"]
    )
    observed_quantiles = np.quantile(data[:, 0], ratio_probabilities)
    prior_quantiles = np.quantile(prior_values[..., 0], ratio_probabilities)
    ratios = prior_quantiles / observed_quantiles
    prior_gate = spec["preflight_gates"]["prior_predictive"]
    prior_checks = {
        "all_values_finite": bool(np.all(np.isfinite(prior_values))),
        "all_rt_strictly_positive": bool(np.all(prior_values[..., 0] > 0.0)),
        "all_angles_in_half_open_domain": bool(
            np.all((-np.pi <= prior_values[..., 1]) & (prior_values[..., 1] < np.pi))
        ),
        "rt_quantile_probabilities": ratio_probabilities.tolist(),
        "observed_rt_quantiles": observed_quantiles.tolist(),
        "prior_rt_quantiles": prior_quantiles.tolist(),
        "prior_to_observed_ratios": ratios.tolist(),
        "seconds": prior_seconds,
    }
    prior_checks["passed"] = bool(
        prior_checks["all_values_finite"]
        and prior_checks["all_rt_strictly_positive"]
        and prior_checks["all_angles_in_half_open_domain"]
        and np.all(ratios >= prior_gate["rt_quantile_ratio_to_observed_lower"])
        and np.all(ratios <= prior_gate["rt_quantile_ratio_to_observed_upper"])
    )

    direct_model = CircularDiffusionModel(threshold_dynamic="fixed")
    objective_rows = []
    truth = np.asarray(scenario["truth"], dtype=np.float64)
    for offset in spec["objective_parity"]["offsets"]:
        parameters = truth + np.asarray(offset, dtype=np.float64)
        a, t, v_x, v_y = parameters
        direct = np.asarray(
            direct_model.joint_lpdf(
                rt=data[:, 0],
                theta=data[:, 1],
                drift_vec=np.column_stack(
                    (
                        np.full(data.shape[0], v_x),
                        np.full(data.shape[0], v_y),
                    )
                ),
                ndt=np.full(data.shape[0], t),
                threshold=float(a),
                decay=0.0,
                threshold_function=None,
                dt_threshold_function=None,
                s_v=0.0,
                s_t=0.0,
                sigma=1.0,
            ),
            dtype=np.float64,
        ).sum()
        direct_jax = np.asarray(
            fixed_cdm_logpdf(
                jnp.asarray(data[:, 0]),
                jnp.asarray(data[:, 1]),
                v_x,
                v_y,
                a,
                t,
                sigma=1.0,
            ),
            dtype=np.float64,
        ).sum()
        compiled = np.asarray(
            analytical_model.model_distribution.logp(
                data,
                v_x,
                v_y,
                a,
                t,
            ).eval(),
            dtype=np.float64,
        ).sum()
        values = np.asarray([direct, direct_jax, compiled], dtype=np.float64)
        objective_rows.append(
            {
                "parameters": parameters.tolist(),
                "direct_numpy": float(direct),
                "direct_jax": float(direct_jax),
                "compiled_hssm": float(compiled),
                "maximum_absolute_error": float(
                    np.max(np.abs(values[:, None] - values[None, :]))
                ),
            }
        )

    return {
        "scenario": scenario["name"],
        "model_contract": blackbox_contract,
        "prior_predictive": prior_checks,
        "objective_candidates": objective_rows,
        "maximum_objective_absolute_error": max(
            row["maximum_absolute_error"] for row in objective_rows
        ),
    }


def _worker_settings(spec: dict[str, Any], smoke: bool) -> dict[str, Any]:
    """Return canonical settings or a clearly non-canonical smoke reduction."""
    execution = spec["execution"]
    if not smoke:
        return {
            "chains": execution["chains"],
            "tune": execution["tune"],
            "draws": execution["draws"],
            "posterior_predictive_draws": execution["posterior_predictive_draws"],
        }
    return {
        "chains": 1,
        "tune": 5,
        "draws": 5,
        "posterior_predictive_draws": 2,
    }


def _run_worker(
    spec: dict[str, Any],
    scenario: dict[str, Any],
    sampler: dict[str, Any],
    data_path: Path,
    work_dir: Path,
    *,
    smoke: bool,
) -> dict[str, Any]:
    """Run one isolated scenario-sampler fit and summarize its raw trace."""
    import arviz as az
    import pymc as pm

    data = np.load(data_path, allow_pickle=False)
    settings = _worker_settings(spec, smoke)
    total_started = perf_counter()
    model_started = perf_counter()
    model = _make_model(data, sampler["likelihood"])
    model_build_seconds = perf_counter() - model_started
    contract = _assert_model_contract(model, spec)

    transformed_initial_point = pm.initial_point.make_initial_point_fn(
        model=model.pymc_model,
        overrides=model.initvals,
        return_transformed=True,
    )(None)
    objective_started = perf_counter()
    initial_logp = float(
        np.asarray(model.pymc_model.compile_logp()(transformed_initial_point))
    )
    objective_seconds = perf_counter() - objective_started
    if not np.isfinite(initial_logp):
        raise FloatingPointError(f"Initial joint logp is not finite: {initial_logp}")

    gradient_seconds = 0.0
    initial_gradient: list[float] | None = None
    if sampler["id"] != "slice":
        gradient_started = perf_counter()
        gradient = np.asarray(
            model.pymc_model.compile_dlogp()(transformed_initial_point),
            dtype=np.float64,
        )
        gradient_seconds = perf_counter() - gradient_started
        expected_shape = tuple(
            spec["preflight_gates"]["initial_point"]["analytical_gradient_shape"]
        )
        if gradient.shape != expected_shape:
            raise RuntimeError(
                "Expected initial gradient shape "
                f"{expected_shape}, got {gradient.shape}."
            )
        if not np.all(np.isfinite(gradient)) or not np.any(gradient != 0.0):
            raise FloatingPointError(f"Invalid initial analytical gradient: {gradient}")
        initial_gradient = gradient.tolist()

    sample_kwargs: dict[str, Any] = {
        "sampler": sampler["backend"],
        "init": spec["execution"]["init"],
        "initvals": model.initvals,
        "chains": settings["chains"],
        "cores": spec["execution"]["cores"],
        "blas_cores": spec["execution"]["blas_cores"],
        "tune": settings["tune"],
        "draws": settings["draws"],
        "random_seed": scenario["chain_seeds"][: settings["chains"]],
        "progressbar": spec["execution"]["progressbar"],
        "compute_convergence_checks": spec["execution"][
            "compute_convergence_checks_during_sampling"
        ],
        "idata_kwargs": {
            "log_likelihood": spec["execution"][
                "compute_log_likelihood_during_sampling"
            ]
        },
    }
    if sampler["id"] == "slice":
        sample_kwargs["step"] = [
            pm.Slice(vars=[model.pymc_model[name]], model=model.pymc_model)
            for name in spec["scope"]["scientific_parameter_order"]
        ]
    else:
        sample_kwargs["target_accept"] = sampler["target_accept"]
        if sampler["nuts_sampler_kwargs"] is not None:
            sample_kwargs["nuts_sampler_kwargs"] = sampler["nuts_sampler_kwargs"]

    sampling_started = perf_counter()
    traces = model.sample(**sample_kwargs)
    sampling_seconds = perf_counter() - sampling_started

    trace_name = f"trace-{scenario['name']}-{sampler['id']}.nc"
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
        var_names=spec["scope"]["scientific_parameter_order"],
        ci_prob=spec["execution"]["posterior_hdi_probability"],
        ci_kind="hdi",
        round_to=10,
    )
    interval_lower, interval_upper = _summary_interval_columns(summary)
    truth_by_name = dict(
        zip(
            spec["scope"]["scientific_parameter_order"],
            scenario["truth"],
            strict=True,
        )
    )
    parameters = []
    for name in spec["scope"]["scientific_parameter_order"]:
        posterior_sd = float(summary.loc[name, "sd"])
        mcse_mean = float(summary.loc[name, "mcse_mean"])
        parameters.append(
            {
                "name": name,
                "truth": truth_by_name[name],
                "posterior_mean": float(summary.loc[name, "mean"]),
                "posterior_sd": posterior_sd,
                "hdi_lower": float(summary.loc[name, interval_lower]),
                "hdi_upper": float(summary.loc[name, interval_upper]),
                "rhat": _optional_float(summary.loc[name, "r_hat"]),
                "ess_bulk": _optional_float(summary.loc[name, "ess_bulk"]),
                "ess_tail": _optional_float(summary.loc[name, "ess_tail"]),
                "mcse_mean": _optional_float(mcse_mean),
                "mcse_over_posterior_sd": (
                    _optional_float(mcse_mean / posterior_sd)
                    if posterior_sd > 0.0
                    else None
                ),
                "truth_in_hdi": bool(
                    summary.loc[name, interval_lower]
                    <= truth_by_name[name]
                    <= summary.loc[name, interval_upper]
                ),
                "ess_bulk_per_sampling_second": (
                    _optional_float(summary.loc[name, "ess_bulk"] / sampling_seconds)
                ),
            }
        )

    sample_stats = traces["sample_stats"].to_dataset()
    stat_names = sorted(str(name) for name in sample_stats.data_vars)
    sampler_diagnostics: dict[str, Any] = {"sample_stats": stat_names}
    if "diverging" in sample_stats:
        sampler_diagnostics["divergences"] = int(
            np.asarray(sample_stats["diverging"]).sum()
        )
    else:
        sampler_diagnostics["divergences"] = None
    for stat_name in ("acceptance_rate", "tree_depth", "n_steps"):
        if stat_name in sample_stats:
            values = np.asarray(sample_stats[stat_name])
            sampler_diagnostics[f"mean_{stat_name}"] = float(values.mean())
            sampler_diagnostics[f"maximum_{stat_name}"] = float(values.max())
    for stat_name in ("nstep_in", "nstep_out"):
        if stat_name in sample_stats:
            sampler_diagnostics[f"mean_{stat_name}"] = float(
                np.asarray(sample_stats[stat_name]).mean()
            )

    predictive_seconds = 0.0
    predictive_summary = None
    if not (
        scenario["role"] == "scale"
        and spec["execution"]["scale_scenario_posterior_predictive"] is False
    ):
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
        values = np.asarray(predictive["posterior_predictive"]["rt,response"].values)
        probabilities = np.asarray(spec["execution"]["rt_quantile_probabilities"])
        observed_rt = np.quantile(data[:, 0], probabilities)
        predictive_rt = np.quantile(values[..., 0], probabilities)
        observed_angle, observed_resultant = _circular_summary(data[:, 1])
        predictive_angle, predictive_resultant = _circular_summary(values[..., 1])
        predictive_summary = {
            "rt_probabilities": probabilities.tolist(),
            "observed_rt_quantiles": observed_rt.tolist(),
            "predictive_rt_quantiles": predictive_rt.tolist(),
            "rt_quantile_absolute_errors": np.abs(predictive_rt - observed_rt).tolist(),
            "observed_mean_angle": observed_angle,
            "predictive_mean_angle": predictive_angle,
            "mean_angle_distance": _circular_distance(observed_angle, predictive_angle),
            "observed_mean_resultant_length": observed_resultant,
            "predictive_mean_resultant_length": predictive_resultant,
            "mean_resultant_length_absolute_error": abs(
                predictive_resultant - observed_resultant
            ),
        }

    runtime = {
        "model_build": model_build_seconds,
        "objective_compile_and_first_eval": objective_seconds,
        "gradient_compile_and_first_eval": gradient_seconds,
        "sampling_call_including_backend_kernel_compilation": sampling_seconds,
        "posterior_predictive": predictive_seconds,
    }
    runtime["total"] = sum(runtime.values())
    return {
        "scenario": scenario["name"],
        "role": scenario["role"],
        "truth": scenario["truth"],
        "trials": scenario["trials"],
        "sampler": sampler["id"],
        "likelihood": sampler["likelihood"],
        "backend": sampler["backend"],
        "status": "completed",
        "smoke": smoke,
        "data_sha256": _sha256(data_path),
        "settings": settings,
        "model_contract": contract,
        "initial_joint_logp": initial_logp,
        "initial_gradient": initial_gradient,
        "trace": trace_metadata,
        "parameters": parameters,
        "sampler_diagnostics": sampler_diagnostics,
        "predictive": predictive_summary,
        "runtime_seconds": runtime,
        "worker_wall_seconds": perf_counter() - total_started,
    }


def _fit_failures(fit: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Evaluate one fit using only compact result fields."""
    prefix = f"{fit['scenario']}/{fit['sampler']}"
    if fit["status"] != "completed":
        return [f"{prefix}: worker status is {fit['status']!r}"]
    thresholds = (
        spec["scientific_acceptance"]
        if fit["role"] == "canonical"
        else spec["scientific_acceptance"]["scale_scenario"]
    )
    failures = []
    if fit["sampler"] != "slice" and fit["sampler_diagnostics"]["divergences"] != 0:
        failures.append(f"{prefix}: divergences are not zero")
    for parameter in fit["parameters"]:
        name = parameter["name"]
        if parameter["rhat"] is None or not (
            parameter["rhat"] < thresholds["maximum_rhat_exclusive"]
        ):
            failures.append(f"{prefix}/{name}: R-hat gate failed")
        if parameter["ess_bulk"] is None or not (
            parameter["ess_bulk"] > thresholds["minimum_bulk_ess_exclusive"]
        ):
            failures.append(f"{prefix}/{name}: bulk ESS gate failed")
        if parameter["ess_tail"] is None or not (
            parameter["ess_tail"] > thresholds["minimum_tail_ess_exclusive"]
        ):
            failures.append(f"{prefix}/{name}: tail ESS gate failed")
        if parameter["mcse_over_posterior_sd"] is None or not (
            parameter["mcse_over_posterior_sd"]
            < thresholds["maximum_mcse_over_posterior_sd_exclusive"]
        ):
            failures.append(f"{prefix}/{name}: MCSE/SD gate failed")
        if fit["role"] == "canonical" and not parameter["truth_in_hdi"]:
            failures.append(f"{prefix}/{name}: truth is outside the 94% HDI")
    if fit["role"] == "canonical":
        predictive = fit["predictive"]
        if predictive is None:
            failures.append(f"{prefix}: predictive summary is missing")
        else:
            if (
                max(predictive["rt_quantile_absolute_errors"])
                > thresholds["maximum_rt_quantile_absolute_error"]
            ):
                failures.append(f"{prefix}: predictive RT gate failed")
            if (
                predictive["mean_angle_distance"]
                > thresholds["maximum_circular_mean_angle_distance"]
            ):
                failures.append(f"{prefix}: predictive mean-angle gate failed")
            if (
                predictive["mean_resultant_length_absolute_error"]
                > thresholds["maximum_mean_resultant_length_absolute_error"]
            ):
                failures.append(f"{prefix}: predictive resultant-length gate failed")
    return failures


def _scientific_gate(
    fits: list[dict[str, Any]],
    preflight: list[dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate preflight and sampler-specific scientific gates."""
    shared_failures = []
    objective_limit = spec["objective_parity"]["maximum_absolute_error"]
    for row in preflight:
        if not row["prior_predictive"]["passed"]:
            shared_failures.append(
                f"{row['scenario']}: prior-predictive support/scale gate failed"
            )
        if row["maximum_objective_absolute_error"] > objective_limit:
            shared_failures.append(f"{row['scenario']}: objective parity gate failed")

    by_sampler = {}
    expected_scenarios = {row["name"] for row in spec["scenarios"]}
    for sampler in spec["samplers"]:
        sampler_fits = [fit for fit in fits if fit["sampler"] == sampler["id"]]
        observed_scenarios = [fit["scenario"] for fit in sampler_fits]
        failures = []
        if set(observed_scenarios) != expected_scenarios or len(
            observed_scenarios
        ) != len(expected_scenarios):
            failures.append(
                f"{sampler['id']}: expected exactly one fit for every scenario"
            )
        failures.extend(
            failure for fit in sampler_fits for failure in _fit_failures(fit, spec)
        )
        by_sampler[sampler["id"]] = {
            "passed": not (shared_failures or failures),
            "failures": [*shared_failures, *failures],
        }
    all_failures = [
        failure for gate in by_sampler.values() for failure in gate["failures"]
    ]
    return {
        "passed": all(gate["passed"] for gate in by_sampler.values()),
        "by_sampler": by_sampler,
        "failures": list(dict.fromkeys(all_failures)),
    }


def _promotion_gate(
    fits: list[dict[str, Any]],
    scientific_gate: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate local efficiency promotion independently for each NUTS backend."""
    promotion = spec["promotion_decision"]
    canonical = [fit for fit in fits if fit["role"] == "canonical"]
    scale = [fit for fit in fits if fit["role"] == "scale"]

    def minimum_efficiency(fit: dict[str, Any]) -> float:
        return min(
            parameter["ess_bulk_per_sampling_second"] for parameter in fit["parameters"]
        )

    slice_fits = [fit for fit in canonical if fit["sampler"] == "slice"]
    if len(slice_fits) != 4 or any(fit["status"] != "completed" for fit in slice_fits):
        return {
            "passed": False,
            "by_sampler": {},
            "failures": ["Slice reference fits are incomplete."],
        }
    slice_median = statistics.median(minimum_efficiency(fit) for fit in slice_fits)
    by_sampler = {}
    for sampler_id in ("pymc_nuts", "numpyro_nuts"):
        failures = []
        sampler_fits = [fit for fit in canonical if fit["sampler"] == sampler_id]
        if len(sampler_fits) != 4 or any(
            fit["status"] != "completed" for fit in sampler_fits
        ):
            by_sampler[sampler_id] = {
                "passed": False,
                "failures": ["Canonical fits are incomplete."],
            }
            continue
        median_efficiency = statistics.median(
            minimum_efficiency(fit) for fit in sampler_fits
        )
        efficiency_ratio = median_efficiency / slice_median
        if (
            efficiency_ratio
            < promotion[
                "canonical_minimum_median_bulk_ess_per_sampling_second_ratio_vs_slice"
            ]
        ):
            failures.append("canonical median minimum bulk ESS/s ratio failed")

        slice_by_scenario = {fit["scenario"]: fit for fit in slice_fits}
        total_ratios = {
            fit["scenario"]: fit["runtime_seconds"]["total"]
            / slice_by_scenario[fit["scenario"]]["runtime_seconds"]["total"]
            for fit in sampler_fits
        }
        if any(
            ratio
            > promotion["canonical_maximum_total_seconds_ratio_vs_slice_per_scenario"]
            for ratio in total_ratios.values()
        ):
            failures.append("canonical total-time ratio failed")

        scale_fit = next((fit for fit in scale if fit["sampler"] == sampler_id), None)
        reference_fit = next(
            (
                fit
                for fit in sampler_fits
                if fit["scenario"] == promotion["scale_reference_scenario"]
            ),
            None,
        )
        scale_ratio = None
        if (
            scale_fit is None
            or reference_fit is None
            or scale_fit["status"] != "completed"
        ):
            failures.append("scale fit or reference fit is incomplete")
        else:
            scale_normalized = (
                min(parameter["ess_bulk"] for parameter in scale_fit["parameters"])
                * scale_fit["trials"]
                / scale_fit["runtime_seconds"][
                    "sampling_call_including_backend_kernel_compilation"
                ]
            )
            reference_normalized = (
                min(parameter["ess_bulk"] for parameter in reference_fit["parameters"])
                * reference_fit["trials"]
                / reference_fit["runtime_seconds"][
                    "sampling_call_including_backend_kernel_compilation"
                ]
            )
            scale_ratio = scale_normalized / reference_normalized
            if (
                scale_ratio
                < promotion["scale_minimum_normalized_efficiency_ratio_vs_reference"]
            ):
                failures.append("scale normalized-efficiency ratio failed")

        scientific_passed = scientific_gate["by_sampler"][sampler_id]["passed"]
        if not scientific_passed:
            failures.append("scientific gate failed")
        by_sampler[sampler_id] = {
            "passed": not failures,
            "scientific_passed": scientific_passed,
            "canonical_median_minimum_bulk_ess_per_second": median_efficiency,
            "slice_canonical_median_minimum_bulk_ess_per_second": slice_median,
            "canonical_efficiency_ratio_vs_slice": efficiency_ratio,
            "canonical_total_seconds_ratios_vs_slice": total_ratios,
            "scale_normalized_efficiency_ratio_vs_reference": scale_ratio,
            "decision": "recommend default"
            if not failures
            else (
                "opt-in only" if scientific_passed else "not scientifically supported"
            ),
            "failures": failures,
        }
    return {
        "passed": any(gate["passed"] for gate in by_sampler.values()),
        "by_sampler": by_sampler,
        "failures": [
            f"{sampler_id}: {failure}"
            for sampler_id, gate in by_sampler.items()
            for failure in gate["failures"]
        ],
    }


def _git_revision_for_path(path: Path) -> str:
    """Return the latest commit touching a tracked path."""
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path)],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _environment_metadata(spec_path: Path) -> dict[str, Any]:
    """Collect the preregistered environment and hardware provenance."""
    import jax

    package_versions = {
        name: importlib.metadata.version(name) for name in PACKAGE_NAMES
    }
    return {
        "hssm_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "spec_revision": _git_revision_for_path(spec_path),
        "jeam_revision": _installed_jeam_revision(),
        "uv_lock_sha256": _sha256(REPO_ROOT / "uv.lock"),
        "python_version": platform.python_version(),
        "package_versions": package_versions,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }


def _installed_jeam_revision() -> str:
    """Read the installed immutable VCS revision."""
    distribution = importlib.metadata.distribution("jeam")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        return distribution.version
    direct_url = json.loads(direct_url_text)
    return direct_url.get("vcs_info", {}).get("commit_id", distribution.version)


def _worker_command(
    *,
    spec_path: Path,
    scenario_name: str,
    sampler_id: str,
    data_path: Path,
    work_dir: Path,
    result_path: Path,
    smoke: bool,
) -> list[str]:
    """Build the exact isolated worker invocation."""
    command = [
        sys.executable,
        "-m",
        "scripts.benchmark_jeam_sampler_comparison",
        "--worker",
        "--spec",
        str(spec_path),
        "--scenario",
        scenario_name,
        "--sampler",
        sampler_id,
        "--data-file",
        str(data_path),
        "--work-dir",
        str(work_dir),
        "--worker-result",
        str(result_path),
    ]
    if smoke:
        command.append("--smoke")
    return command


def _sanitize_failure(message: str, work_dir: Path) -> str:
    """Remove machine-local paths from a recorded failure message."""
    return message.replace(str(REPO_ROOT), "<repo>").replace(
        str(work_dir), "<work-dir>"
    )


def _worker_entry(args: argparse.Namespace) -> int:
    """Execute a worker and always leave a compact success or failure record."""
    spec = _load_json(args.spec)
    scenario = _scenario(spec, args.scenario[0])
    sampler = _sampler(spec, args.sampler[0])
    try:
        _configure_precision()
        result = _run_worker(
            spec,
            scenario,
            sampler,
            args.data_file,
            args.work_dir,
            smoke=args.smoke,
        )
        exit_code = 0
    except Exception as exc:  # noqa: BLE001 - scientific failures must be recorded
        result = {
            "scenario": scenario["name"],
            "role": scenario["role"],
            "truth": scenario["truth"],
            "trials": scenario["trials"],
            "sampler": sampler["id"],
            "likelihood": sampler["likelihood"],
            "backend": sampler["backend"],
            "status": "failed",
            "smoke": args.smoke,
            "error_type": type(exc).__name__,
            "error_message": _sanitize_failure(str(exc), args.work_dir),
            "traceback_tail": [
                _sanitize_failure(line, args.work_dir)
                for line in traceback.format_exc().splitlines()[-12:]
            ],
        }
        exit_code = 1
    _write_json(args.worker_result, result)
    return exit_code


def _parent_entry(args: argparse.Namespace) -> int:
    """Generate shared evidence and orchestrate isolated worker processes."""
    _configure_precision()
    spec = _load_json(args.spec)
    if spec["status"] != "preregistered-before-canonical-run":
        raise RuntimeError("Refusing to run an unfrozen benchmark specification.")
    installed_jeam = _installed_jeam_revision()
    expected_jeam = spec["provenance"]["jeam_revision"]
    if installed_jeam != expected_jeam:
        raise RuntimeError(
            f"Expected JEAM revision {expected_jeam}, found {installed_jeam}."
        )

    selected_scenarios = [
        row
        for row in spec["scenarios"]
        if not args.scenario or row["name"] in args.scenario
    ]
    selected_samplers = [
        row for row in spec["samplers"] if not args.sampler or row["id"] in args.sampler
    ]
    if not selected_scenarios or not selected_samplers:
        raise ValueError("The scenario and sampler selections must be non-empty.")
    canonical_complete = (
        not args.smoke
        and len(selected_scenarios) == len(spec["scenarios"])
        and len(selected_samplers) == len(spec["samplers"])
    )

    work_dir = args.work_dir or Path(
        tempfile.mkdtemp(prefix="jeam-fixed-cdm-sampler-v1-")
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output
    if output_path is None:
        output_path = (
            work_dir / "smoke-result.json"
            if args.smoke or not canonical_complete
            else REPO_ROOT / spec["reporting_and_deviation_policy"]["result_path"]
        )

    data_rows = []
    data_paths = {}
    data_arrays = {}
    for scenario in selected_scenarios:
        data = _simulate_dataset(scenario)
        data_path = work_dir / f"data-{scenario['name']}.npy"
        np.save(data_path, data, allow_pickle=False)
        data_paths[scenario["name"]] = data_path
        data_arrays[scenario["name"]] = data
        data_rows.append(
            {
                "scenario": scenario["name"],
                "basename": data_path.name,
                "sha256": _sha256(data_path),
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "all_finite": bool(np.all(np.isfinite(data))),
                "all_rt_strictly_positive": bool(np.all(data[:, 0] > 0.0)),
                "all_angles_in_half_open_domain": bool(
                    np.all((-np.pi <= data[:, 1]) & (data[:, 1] < np.pi))
                ),
            }
        )
    prefix_check = None
    if {
        "baseline_asymmetric",
        "baseline_asymmetric_scale_1500",
    } <= data_arrays.keys():
        prefix_check = bool(
            np.array_equal(
                data_arrays["baseline_asymmetric"],
                data_arrays["baseline_asymmetric_scale_1500"][:300],
            )
        )
        if not prefix_check:
            raise AssertionError(
                "The scale dataset is not the preregistered prefix extension."
            )

    shared_preflight = []
    for scenario in selected_scenarios:
        print(f"preflight {scenario['name']}", flush=True)
        shared_preflight.append(
            _run_shared_preflight(spec, scenario, data_paths[scenario["name"]])
        )

    fits = []
    for scenario in selected_scenarios:
        for sampler in selected_samplers:
            worker_result = work_dir / f"worker-{scenario['name']}-{sampler['id']}.json"
            print(f"fit {scenario['name']} / {sampler['id']}", flush=True)
            completed = subprocess.run(
                _worker_command(
                    spec_path=args.spec,
                    scenario_name=scenario["name"],
                    sampler_id=sampler["id"],
                    data_path=data_paths[scenario["name"]],
                    work_dir=work_dir,
                    result_path=worker_result,
                    smoke=args.smoke,
                ),
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            if not worker_result.exists():
                raise RuntimeError(
                    f"Worker exited {completed.returncode} without a result for "
                    f"{scenario['name']}/{sampler['id']}."
                )
            fit = _load_json(worker_result)
            if completed.returncode != 0 and fit["status"] != "failed":
                raise RuntimeError("Nonzero worker exit did not record failure status.")
            fits.append(fit)
            print(
                f"completed {scenario['name']} / {sampler['id']}: {fit['status']}",
                flush=True,
            )

    status = (
        "canonical-complete"
        if canonical_complete
        else ("smoke" if args.smoke else "partial")
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "study_id": spec["study_id"],
        "status": status,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "spec_sha256": _sha256(args.spec),
        "provenance": _environment_metadata(args.spec),
        "execution": spec["execution"],
        "selected_scenarios": [row["name"] for row in selected_scenarios],
        "selected_samplers": [row["id"] for row in selected_samplers],
        "data": data_rows,
        "scale_prefix_check_passed": prefix_check,
        "shared_preflight": shared_preflight,
        "fits": fits,
        "scientific_gate": None,
        "promotion_gate": None,
    }
    if canonical_complete:
        payload["scientific_gate"] = _scientific_gate(fits, shared_preflight, spec)
        payload["promotion_gate"] = _promotion_gate(
            fits, payload["scientific_gate"], spec
        )
    _write_json(output_path, payload)
    print(f"result {output_path}", flush=True)
    print(f"work directory {work_dir}", flush=True)
    return 0 if all(fit["status"] == "completed" for fit in fits) else 1


def _parse_args() -> argparse.Namespace:
    """Parse parent and private worker arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--sampler", action="append")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--data-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.spec = args.spec.resolve()
    if args.work_dir is not None:
        args.work_dir = args.work_dir.resolve()
    if args.output is not None:
        args.output = args.output.resolve()
    if args.data_file is not None:
        args.data_file = args.data_file.resolve()
    if args.worker_result is not None:
        args.worker_result = args.worker_result.resolve()
    if args.worker:
        required = {
            "scenario": args.scenario,
            "sampler": args.sampler,
            "data_file": args.data_file,
            "work_dir": args.work_dir,
            "worker_result": args.worker_result,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"worker mode requires: {', '.join(missing)}")
        if len(args.scenario) != 1 or len(args.sampler) != 1:
            parser.error("worker mode requires exactly one scenario and sampler")
    return args


def main() -> None:
    """Run the parent study or one private isolated worker."""
    args = _parse_args()
    exit_code = _worker_entry(args) if args.worker else _parent_entry(args)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

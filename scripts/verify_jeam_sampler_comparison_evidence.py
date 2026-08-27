"""Verify the fixed-CDM sampler study's compact-only retained evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence  # noqa: TC003
from io import BytesIO
from numbers import Real
from pathlib import Path
from typing import Any, Never

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ADDENDUM = "benchmarks/results/jeam_fixed_cdm_sampler_comparison_v1_addendum.json"
SPEC = "benchmarks/specs/jeam_fixed_cdm_sampler_comparison_v1.json"
RESULT = "benchmarks/results/jeam_fixed_cdm_sampler_comparison_v1.json"
SCALE = "benchmarks/evidence/jeam_fixed_cdm_sampler_comparison_v1/baseline_asymmetric_scale_1500.npy"  # noqa: E501
BUNDLE = "benchmarks/evidence/jeam_repeated_recovery_v2"
PINS = {
    ADDENDUM: "07ca04c8aa52f672b654835e63ff35af31f4ff197a3c95d2cb256ef2ab91b33f",
    SPEC: "a09b5760ede85bae4b1869c0cb98d8ba9bb78d09032f9e62221dbafe74b3d0e5",
    RESULT: "35b154b55228ca179c14d89f39491ba9d1a9d0b27a3c8b4131e842f99abf5d39",
    SCALE: "253a16585d6c2bb0b0aa91f8b6fbaabd5609e284a1d2d2bad61bc97266d9e826",
}
MANIFEST_PIN = "d8a5c458d2194f1fb7031f6bc5ca5add3cd67afabd028880dc0bfed887ef9972"
REVISIONS = {
    "historical_analytical_result": "0c0ef8b834dd062ad8aea5ff8e7a09dfb55492ce",
    "durable_blackbox_reference": "a9f547b3630ae8ff31ccec1b904e0c02fdba6d99",
    "current_safety_revision": "ede7a4f4faf226e4dae52c84dfb01012939cccdc",
}
PARAMETERS = ("a", "t", "v_x", "v_y")
SAMPLERS = ("slice", "pymc_nuts", "numpyro_nuts")


class SamplerEvidenceMismatch(ValueError):
    """Raised for any compact sampler evidence mismatch."""


def _fail(message: str) -> Never:
    raise SamplerEvidenceMismatch(message)


def _snapshot(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise SamplerEvidenceMismatch(f"Cannot read evidence: {path}.") from error


def _parse(payload: bytes, label: str) -> dict[str, Any]:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SamplerEvidenceMismatch(f"Invalid JSON: {label}.") from error
    if not isinstance(document, dict):
        _fail(f"JSON root is not an object: {label}.")
    return document


def _num(value: object, label: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        _fail(f"Expected a finite number: {label}.")
    return float(value)


def _vector(value: object, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        _fail(f"Unexpected vector shape: {label}.")
    return [_num(item, label) for item in value]


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _rows(rows: object, key: str, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        _fail(f"Invalid row list: {label}.")
    indexed = {row.get(key): row for row in rows}
    if len(indexed) != len(rows) or not all(isinstance(name, str) for name in indexed):
        _fail(f"Invalid row keys: {label}.")
    return indexed  # type: ignore[return-value]


def _authenticate(root: Path) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    snapshots = {name: _snapshot(root / name) for name in PINS}
    for name, digest in PINS.items():
        if hashlib.sha256(snapshots[name]).hexdigest() != digest:
            _fail(f"SHA256 mismatch: {name}.")
    return snapshots, {
        name: _parse(snapshots[name], name) for name in (ADDENDUM, SPEC, RESULT)
    }


def _verify_bundle(root: Path) -> dict[str, Any]:
    try:
        from scripts.verify_jeam_repeated_recovery_evidence import (
            verify_integrity as _verify_integrity,
        )
    except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
        from verify_jeam_repeated_recovery_evidence import (  # type: ignore[no-redef]
            verify_integrity as _verify_integrity,
        )

    manifest = _verify_integrity(root / BUNDLE)
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("result_schema_version") != 2:
        _fail("Durable reference is not the expected schema-v2 result bundle.")
    return dict(manifest)


def _check_provenance(addendum: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    manifest = addendum["authenticated_inputs"]["durable_blackbox_manifest"]
    for role, revision in REVISIONS.items():
        if addendum["jeam_revisions"][role]["revision"] != revision:
            _fail(f"JEAM revision mismatch: {role}.")
    if (
        manifest["sha256"] != MANIFEST_PIN
        or addendum["jeam_revisions"]["current_safety_revision"][
            "sampler_comparison_rerun"
        ]
        or addendum["promotion"]["ecosystem_promotion"] != "blocked"
        or result["provenance"]["jeam_revision"]
        != REVISIONS["historical_analytical_result"]
    ):
        _fail("Historical/current provenance boundary mismatch.")


def _bind_datasets(
    root: Path,
    scale_payload: bytes,
    addendum: Mapping[str, Any],
    spec: Mapping[str, Any],
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    scenarios = _rows(spec["scenarios"], "name", "scenarios")
    compact = _rows(result["data"], "scenario", "datasets")
    canonical = addendum["dataset_binding"]["canonical_scenario_sha256"]
    durable = {
        row["path"].split("/")[1]: row["sha256"]
        for row in manifest["artifacts"]
        if row["role"] == "dataset"
    }
    expected = {**canonical, "baseline_asymmetric_scale_1500": PINS[SCALE]}
    observed = {name: row["sha256"] for name, row in compact.items()}
    if canonical != durable or set(compact) != set(scenarios) or observed != expected:
        _fail("Canonical dataset binding mismatch.")
    try:
        scale = np.load(BytesIO(scale_payload), allow_pickle=False)
        baseline_payload = _snapshot(
            root / BUNDLE / "scenarios/baseline_asymmetric/dataset.npy"
        )
        baseline = np.load(BytesIO(baseline_payload), allow_pickle=False)
    except (EOFError, OSError, TypeError, ValueError) as error:
        raise SamplerEvidenceMismatch("Invalid retained dataset bytes.") from error
    if (
        hashlib.sha256(baseline_payload).hexdigest() != canonical["baseline_asymmetric"]
        or scale.shape != (1500, 2)
        or scale.dtype.str != "<f8"
        or not np.isfinite(scale).all()
        or np.any(scale[:, 0] <= 0.0)
        or np.any(scale[:, 1] < -np.pi)
        or np.any(scale[:, 1] >= np.pi)
        or not np.array_equal(scale[:300], baseline)
    ):
        _fail("Reconstructed scale dataset contract mismatch.")
    return expected


def _preflight_failures(
    spec: Mapping[str, Any], result: Mapping[str, Any]
) -> list[str]:
    scenarios = _rows(spec["scenarios"], "name", "scenarios")
    preflights = _rows(result["shared_preflight"], "scenario", "preflights")
    if set(preflights) != set(scenarios):
        _fail("Preflight inventory mismatch.")
    prior_gate = spec["preflight_gates"]["prior_predictive"]
    objective_gate = spec["objective_parity"]
    failures: list[str] = []
    for name, scenario in scenarios.items():
        row, prior = preflights[name], preflights[name]["prior_predictive"]
        observed = _vector(prior["observed_rt_quantiles"], 2, name)
        predicted = _vector(prior["prior_rt_quantiles"], 2, name)
        ratios = [x / y for x, y in zip(predicted, observed, strict=True)]
        stored = _vector(prior["prior_to_observed_ratios"], 2, name)
        if (
            row["model_contract"]["priors"]
            != spec["priors_and_initialization"]["priors"]
            or row["model_contract"]["initvals"]
            != spec["priors_and_initialization"]["resolved_untransformed_initvals"]
            or not all(
                prior[field]
                for field in (
                    "all_values_finite",
                    "all_rt_strictly_positive",
                    "all_angles_in_half_open_domain",
                )
            )
            or not all(_close(x, y) for x, y in zip(ratios, stored, strict=True))
            or any(
                x < prior_gate["rt_quantile_ratio_to_observed_lower"]
                or x > prior_gate["rt_quantile_ratio_to_observed_upper"]
                for x in ratios
            )
        ):
            failures.append(f"{name}: prior/model preflight")
        maximum = 0.0
        for candidate in row["objective_candidates"]:
            values = [
                _num(candidate[key], name)
                for key in ("direct_numpy", "direct_jax", "compiled_hssm")
            ]
            error = max(abs(x - y) for x in values for y in values)
            maximum = max(maximum, error)
            if not _close(error, _num(candidate["maximum_absolute_error"], name)):
                _fail(f"Derived objective error mismatch: {name}.")
        if (
            not _close(maximum, _num(row["maximum_objective_absolute_error"], name))
            or maximum > objective_gate["maximum_absolute_error"]
        ):
            failures.append(f"{name}: objective parity")
    return failures


def _fit_science(
    spec: Mapping[str, Any], result: Mapping[str, Any]
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str], dict[str, int]]:
    scenarios = _rows(spec["scenarios"], "name", "scenarios")
    routes = _rows(spec["samplers"], "id", "samplers")
    data = _rows(result["data"], "scenario", "datasets")
    fits = {(row["scenario"], row["sampler"]): row for row in result["fits"]}
    expected = {(name, sampler) for name in scenarios for sampler in routes}
    if len(fits) != len(result["fits"]) or set(fits) != expected:
        _fail("Fit cross-product mismatch.")
    failures = _preflight_failures(spec, result)
    counts = {"hdi": 0, "included": 0, "nuts": 0, "canonical_nuts": 0, "div": 0}
    science = spec["scientific_acceptance"]
    for (name, sampler), fit in fits.items():
        scenario, route = scenarios[name], routes[sampler]
        prefix = f"{name}/{sampler}"
        if (
            fit["status"] != "completed"
            or fit["smoke"] is not False
            or fit["role"] != scenario["role"]
            or fit["truth"] != scenario["truth"]
            or fit["likelihood"] != route["likelihood"]
            or fit["backend"] != route["backend"]
            or fit["data_sha256"] != data[name]["sha256"]
        ):
            _fail(f"Fit header mismatch: {prefix}.")
        seconds = _num(
            fit["runtime_seconds"][
                "sampling_call_including_backend_kernel_compilation"
            ],
            prefix,
        )
        trace = fit["trace"]
        if seconds <= 0.0 or trace["bytes"] <= 0 or not trace["saved_before_summary"]:
            _fail(f"Runtime/trace record mismatch: {prefix}.")
        divergences = fit["sampler_diagnostics"]["divergences"]
        if sampler == "slice":
            if fit["initial_gradient"] is not None or divergences is not None:
                _fail(f"Slice diagnostic mismatch: {prefix}.")
        else:
            gradient = _vector(fit["initial_gradient"], 4, prefix)
            if not isinstance(divergences, int) or not any(gradient):
                _fail(f"NUTS diagnostic mismatch: {prefix}.")
            counts["nuts"] += 1
            counts["div"] += divergences
            counts["canonical_nuts"] += scenario["role"] == "canonical"
            if divergences:
                failures.append(f"{prefix}: divergences")
        thresholds = (
            science if scenario["role"] == "canonical" else science["scale_scenario"]
        )
        if [row["name"] for row in fit["parameters"]] != list(PARAMETERS):
            _fail(f"Parameter inventory mismatch: {prefix}.")
        truth = dict(zip(PARAMETERS, scenario["truth"], strict=True))
        for row in fit["parameters"]:
            label, value = f"{prefix}/{row['name']}", truth[row["name"]]
            lower, upper = _num(row["hdi_lower"], label), _num(row["hdi_upper"], label)
            included = lower <= value <= upper
            sd, ess = _num(row["posterior_sd"], label), _num(row["ess_bulk"], label)
            mcse = _num(row["mcse_mean"], label) / sd
            if (
                not _close(_num(row["truth"], label), value)
                or row["truth_in_hdi"] is not included
                or not _close(mcse, _num(row["mcse_over_posterior_sd"], label))
                or not _close(
                    ess / seconds, _num(row["ess_bulk_per_sampling_second"], label)
                )
            ):
                _fail(f"Derived parameter row mismatch: {label}.")
            tests = (
                _num(row["rhat"], label) < thresholds["maximum_rhat_exclusive"],
                ess > thresholds["minimum_bulk_ess_exclusive"],
                _num(row["ess_tail"], label) > thresholds["minimum_tail_ess_exclusive"],
                mcse < thresholds["maximum_mcse_over_posterior_sd_exclusive"],
            )
            if not all(tests):
                failures.append(f"{label}: convergence")
            if scenario["role"] == "canonical":
                counts["hdi"] += 1
                counts["included"] += included
                if not included:
                    failures.append(f"{label}: truth outside HDI")
        predictive = fit["predictive"]
        if scenario["role"] == "scale":
            if predictive is not None:
                _fail(f"Scale predictive row mismatch: {prefix}.")
            continue
        observed = _vector(predictive["observed_rt_quantiles"], 3, prefix)
        predicted = _vector(predictive["predictive_rt_quantiles"], 3, prefix)
        errors = [abs(x - y) for x, y in zip(predicted, observed, strict=True)]
        angle = abs(
            math.atan2(
                math.sin(
                    predictive["observed_mean_angle"]
                    - predictive["predictive_mean_angle"]
                ),
                math.cos(
                    predictive["observed_mean_angle"]
                    - predictive["predictive_mean_angle"]
                ),
            )
        )
        resultant = abs(
            predictive["observed_mean_resultant_length"]
            - predictive["predictive_mean_resultant_length"]
        )
        if (
            not all(
                _close(x, y)
                for x, y in zip(
                    errors,
                    _vector(predictive["rt_quantile_absolute_errors"], 3, prefix),
                )
            )
            or not _close(angle, predictive["mean_angle_distance"])
            or not _close(resultant, predictive["mean_resultant_length_absolute_error"])
        ):
            _fail(f"Derived predictive row mismatch: {prefix}.")
        if (
            max(errors) > science["maximum_rt_quantile_absolute_error"]
            or angle > science["maximum_circular_mean_angle_distance"]
            or resultant > science["maximum_mean_resultant_length_absolute_error"]
        ):
            failures.append(f"{prefix}: predictive gate")
    return fits, failures, counts


def _ratios(
    spec: Mapping[str, Any], fits: Mapping[tuple[str, str], Mapping[str, Any]]
) -> dict[str, Any]:
    canonical = [row["name"] for row in spec["scenarios"] if row["role"] == "canonical"]
    gate = spec["promotion_decision"]

    def efficiency(fit: Mapping[str, Any]) -> float:
        seconds = fit["runtime_seconds"][
            "sampling_call_including_backend_kernel_compilation"
        ]
        return min(row["ess_bulk"] for row in fit["parameters"]) / seconds

    slice_median = statistics.median(
        efficiency(fits[(name, "slice")]) for name in canonical
    )
    by_sampler = {}
    for sampler in SAMPLERS[1:]:
        median = statistics.median(
            efficiency(fits[(name, sampler)]) for name in canonical
        )
        totals = {
            name: fits[(name, sampler)]["runtime_seconds"]["total"]
            / fits[(name, "slice")]["runtime_seconds"]["total"]
            for name in canonical
        }

        def normalized(name: str) -> float:
            fit = fits[(name, sampler)]
            return efficiency(fit) * fit["trials"]

        efficiency_ratio = median / slice_median
        scale_ratio = normalized("baseline_asymmetric_scale_1500") / normalized(
            gate["scale_reference_scenario"]
        )
        passed = (
            efficiency_ratio
            >= gate[
                "canonical_minimum_median_bulk_ess_per_sampling_second_ratio_vs_slice"
            ]
            and max(totals.values())
            <= gate["canonical_maximum_total_seconds_ratio_vs_slice_per_scenario"]
            and scale_ratio
            >= gate["scale_minimum_normalized_efficiency_ratio_vs_reference"]
        )
        by_sampler[sampler] = {
            "canonical_median_minimum_bulk_ess_per_second": median,
            "canonical_efficiency_ratio_vs_slice": efficiency_ratio,
            "canonical_total_seconds_ratios_vs_slice": totals,
            "scale_normalized_efficiency_ratio_vs_reference": scale_ratio,
            "passed": passed,
        }
    return {
        "slice_canonical_median_minimum_bulk_ess_per_second": slice_median,
        "by_sampler": by_sampler,
        "passed": all(row["passed"] for row in by_sampler.values()),
    }


def _load(root: Path) -> dict[str, Any]:
    snapshots, documents = _authenticate(root)
    addendum, spec, result = documents[ADDENDUM], documents[SPEC], documents[RESULT]
    _check_provenance(addendum, result)
    if (
        spec["provenance"]["pytensor_floatx"] != "float64"
        or result["spec_sha256"] != PINS[SPEC]
        or result["execution"] != spec["execution"]
    ):
        _fail("Frozen execution header mismatch.")
    manifest = _verify_bundle(root)
    hashes = _bind_datasets(root, snapshots[SCALE], addendum, spec, result, manifest)
    fits, failures, derived = _fit_science(spec, result)
    ratios = _ratios(spec, fits)
    if failures:
        _fail(f"Compact scientific gate failed: {'; '.join(failures)}")
    if not ratios["passed"]:
        _fail("Recorded-machine efficiency gate failed.")
    canonical_scenarios = sum(row["role"] == "canonical" for row in spec["scenarios"])
    counts = {
        "scenarios": len(spec["scenarios"]),
        "canonical_scenarios": canonical_scenarios,
        "samplers": len(spec["samplers"]),
        "fits": len(fits),
        "canonical_scenario_parameter_truths": canonical_scenarios * len(PARAMETERS),
        "canonical_route_hdi_checks": derived["hdi"],
        "canonical_route_hdi_inclusions": derived["included"],
        "canonical_nuts_fits": derived["canonical_nuts"],
        "all_nuts_fits": derived["nuts"],
        "all_nuts_divergences": derived["div"],
        "trace_records": len(fits),
        "retained_trace_files": 0,
    }
    expected = (5, 4, 3, 15, 16, 48, 48, 8, 10, 0, 15, 0)
    if tuple(counts.values()) != expected:
        _fail("Recomputed evidence counts mismatch.")
    return {
        "study_id": spec["study_id"],
        "evidence_class": "authenticated compact-only smoke benchmark",
        "counts": counts,
        "canonical_dataset_sha256": addendum["dataset_binding"][
            "canonical_scenario_sha256"
        ],  # noqa: E501
        "reconstructed_scale_dataset_sha256": hashes["baseline_asymmetric_scale_1500"],
        "scientific_gate": {"passed": True, "failures": []},
        "recorded_machine_efficiency": ratios,
        "jeam_revisions": REVISIONS,
        "retention": addendum["evidence_accounting"],
        "ecosystem_promotion": {
            "blocked": True,
            "blockers": addendum["promotion"]["blockers"],
        },
    }


def load_verified_sampler_comparison(
    root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Authenticate compact evidence and return independently derived boundaries."""
    try:
        return _load(Path(root))
    except SamplerEvidenceMismatch:
        raise
    except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
        raise SamplerEvidenceMismatch(f"Invalid sampler evidence: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    """Run the compact-only verifier as a terse CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=REPO_ROOT)
    try:
        report = load_verified_sampler_comparison(parser.parse_args(argv).root)
    except SamplerEvidenceMismatch as error:
        print(f"compact-only sampler evidence FAILED: {error}", file=sys.stderr)
        return 1
    counts = report["counts"]
    print(
        "compact-only sampler evidence PASS: "
        f"{counts['fits']} fits, {counts['canonical_route_hdi_inclusions']}/"
        f"{counts['canonical_route_hdi_checks']} route HDIs, "
        f"{counts['all_nuts_divergences']} NUTS divergences; promotion blocked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

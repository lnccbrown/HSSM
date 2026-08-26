"""Independently verify the canonical JEAM repeated-recovery evidence bundle.

This module is intentionally network-free and imports neither HSSM nor JEAM. It
authenticates every byte before parsing payloads, then recomputes the scientific
result from the retained NumPy and xarray artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Never

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = REPOSITORY / "benchmarks/evidence/jeam_repeated_recovery_v2"
MANIFEST_SHA256 = "d8a5c458d2194f1fb7031f6bc5ca5add3cd67afabd028880dc0bfed887ef9972"
PROTOCOL_SHA256 = "8c68a1169e9f539c54e4d5cb23e99784702bdc45a536ffea50b7cca7eac5c8e7"
PARAMETERS = ("a", "t", "v_x", "v_y")
RAW_GROUPS = (
    "prior",
    "prior_predictive",
    "observed_data",
    "posterior",
    "sample_stats",
    "posterior_predictive",
)
SCENARIOS = (
    (
        "baseline_asymmetric",
        (1.2, 0.15, 0.8, -0.5),
        1492,
        8675309,
        (3101, 3102, 3103, 3104),
        6101,
        7291,
    ),
    (
        "reversed_drift",
        (1.05, 0.1, -0.9, 0.65),
        2603,
        54021,
        (4201, 4202, 4203, 4204),
        7101,
        8291,
    ),
    (
        "high_threshold_strong_drift",
        (1.6, 0.22, 1.25, 0.3),
        3714,
        64031,
        (5301, 5302, 5303, 5304),
        8101,
        9291,
    ),
    (
        "low_threshold_negative_drift",
        (0.75, 0.07, -0.45, -1.15),
        4825,
        74041,
        (6401, 6402, 6403, 6404),
        9101,
        10291,
    ),
)
THRESHOLDS = {
    "objective_absolute_error": 5e-5,
    "optimizer_parameter_absolute_error": 1e-12,
    "optimizer_objective_absolute_error": 5e-5,
    "maximum_rhat": 1.01,
    "minimum_bulk_ess": 500.0,
    "minimum_tail_ess": 500.0,
    "minimum_hdi_inclusion_fraction": 0.75,
    "maximum_mcse_sd_ratio": 0.05,
    "minimum_prior_to_observed_rt_ratio": 0.1,
    "maximum_prior_to_observed_rt_ratio": 20.0,
    "maximum_rt_quantile_absolute_error": 0.12,
    "maximum_mean_angle_distance": 0.1,
    "maximum_resultant_length_absolute_error": 0.08,
    "maximum_absolute_bias": {"a": 0.12, "t": 0.04, "v_x": 0.2, "v_y": 0.2},
    "maximum_rmse": {"a": 0.18, "t": 0.05, "v_x": 0.28, "v_y": 0.28},
}
EXPECTED_PROVENANCE = {
    "clean_worktree": True,
    "environment_sha256": (
        "c6afcb78635abbfa9fbefb0f30509f1aa2dafcb9a7e6bead6e4c362581951c6e"
    ),
    "jeam_revision": "a9f547b3630ae8ff31ccec1b904e0c02fdba6d99",
    "platform": {"machine": "arm64", "release": "25.4.0", "system": "Darwin"},
    "producer_revision": "518ced3194414ea9cfbcb2748afa3e0710b51a0c",
    "producer_tree": "925a895cd3a55b12485f79618411bce602873473",
    "protocol_base_revision": "cd3fe0a1decc963a9db160a72597e6715b895a5c",
    "python": {"implementation": "CPython", "version": "3.12.13"},
    "source_sha256": {
        "pyproject.toml": (
            "26ec800ba6e71d59774d152ae04d9aab72336fa62ec6f64a4449df6217171e04"
        ),
        "scripts/benchmark_jeam_bayesian_recovery.py": (
            "f6ba695b77e221c8887ee165d09e1157600569d99854f54f3c8fe675198b6ca3"
        ),
        "scripts/benchmark_jeam_objective_parity.py": (
            "406b723e00dc4bfd407cda44e42bf6cc604ad9c4f916846fe72894633cefdef3"
        ),
        "scripts/benchmark_jeam_recovery_bundle.py": (
            "8ba4565126fd1a13f88786d631f5fb02f36c161663e2bd496e11e05697a05fc7"
        ),
        "scripts/benchmark_jeam_recovery_evidence.py": (
            "bb7f59ac2c0112e53aadfa9b8e1c51b13d5b60e01da0ea392e18569899969a7d"
        ),
        "scripts/benchmark_jeam_repeated_recovery.py": (
            "3e1b09192f763b1eb536236bbf8f30b0bd3eb580ab33c17cb87f8223dee62ac4"
        ),
    },
}


class EvidenceMismatch(ValueError):
    """Raised when retained evidence differs from the canonical contract."""


def _scenario_documents() -> list[dict[str, object]]:
    """Return the frozen scenario documents used in the protocol."""
    return [
        {
            "name": name,
            "truth": list(truth),
            "data_seed": data_seed,
            "optimizer_seed": optimizer_seed,
            "chain_seeds": list(chain_seeds),
            "prior_seed": prior_seed,
            "predictive_seed": predictive_seed,
        }
        for (
            name,
            truth,
            data_seed,
            optimizer_seed,
            chain_seeds,
            prior_seed,
            predictive_seed,
        ) in SCENARIOS
    ]


def _expected_protocol() -> dict[str, object]:
    """Return the independently frozen canonical protocol."""
    return {
        "schema_version": 1,
        "result_schema_version": 2,
        "parameter_order": list(PARAMETERS),
        "sampler": "pymc.Slice[a,t,v_x,v_y]",
        "pytensor_floatx": "float64",
        "trials_per_scenario": 300,
        "chains": 4,
        "tune": 1000,
        "draws": 1000,
        "hdi_probability": 0.94,
        "prior_draws": 100,
        "predictive_draws": 40,
        "optimizer_maxiter": 14,
        "optimizer_popsize": 15,
        "thresholds": THRESHOLDS,
        "scenarios": _scenario_documents(),
    }


def _expected_artifacts() -> dict[str, str]:
    """Return the exact safe inventory, excluding the externally pinned manifest."""
    artifacts = {"environment.txt": "environment", "result.json": "derived_result"}
    for name, *_ in SCENARIOS:
        prefix = f"scenarios/{name}"
        artifacts[f"{prefix}/dataset.npy"] = "dataset"
        artifacts[f"{prefix}/measurements.json"] = "measurements"
        artifacts[f"{prefix}/raw.nc"] = "raw_datatree"
    return dict(sorted(artifacts.items()))


def _expected_directories() -> set[str]:
    """Return the only directories allowed below the evidence root."""
    return {"scenarios", *(f"scenarios/{name}" for name, *_ in SCENARIOS)}


def _fail(message: str) -> Never:
    """Raise a consistently typed evidence error."""
    raise EvidenceMismatch(message)


def _canonical_json_bytes(value: object) -> bytes:
    """Encode one finite JSON value in the producer's canonical form."""
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise EvidenceMismatch(
            "JSON value is not finite and canonicalizable."
        ) from error
    return f"{payload}\n".encode()


def _finite_float(token: str) -> float:
    """Parse a JSON float while rejecting overflow to infinity."""
    value = float(token)
    if not math.isfinite(value):
        _fail(f"Non-finite JSON number {token!r} is forbidden.")
    return value


def _reject_constant(token: str) -> None:
    """Reject JavaScript-style non-finite constants accepted by ``json``."""
    _fail(f"Non-finite JSON constant {token!r} is forbidden.")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Construct a JSON object while rejecting duplicate member names."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"Duplicate JSON member {key!r} is forbidden.")
        result[key] = value
    return result


def _parse_strict_json(
    payload: bytes, label: str, *, canonical: bool = True
) -> dict[str, object]:
    """Parse one authenticated finite duplicate-free UTF-8 JSON object."""
    try:
        value = json.loads(
            payload,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            object_pairs_hook=_unique_object,
        )
    except EvidenceMismatch:
        raise
    except ValueError as error:
        raise EvidenceMismatch(f"Invalid JSON payload: {label}.") from error
    if not isinstance(value, dict):
        _fail(f"JSON payload must be an object: {label}.")
    if canonical and payload != _canonical_json_bytes(value):
        _fail(f"JSON payload is not canonical: {label}.")
    return value


def _read_snapshot(path: Path) -> bytes:
    """Read one no-follow regular file exactly once for hashing and parsing."""
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                _fail(f"Artifact is not a regular file: {path.name}.")
            return stream.read()
    except OSError as error:
        raise EvidenceMismatch(f"Cannot snapshot artifact {path.name}.") from error


def _load_strict_json(path: Path, *, canonical: bool = True) -> dict[str, object]:
    """Snapshot and strictly parse one standalone JSON file."""
    return _parse_strict_json(_read_snapshot(path), path.name, canonical=canonical)


def _safe_relative_path(value: object) -> str:
    """Return one canonical POSIX path confined below a bundle root."""
    if not isinstance(value, str) or not value or "\0" in value or "\\" in value:
        _fail(f"Unsafe artifact path: {value!r}.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or re.match(r"^[A-Za-z]:", value)
    ):
        _fail(f"Unsafe artifact path: {value!r}.")
    return value


def _verify_inventory(root: Path) -> None:
    """Reject missing, extra, non-regular, and symlinked bundle entries."""
    if root.is_symlink() or not root.is_dir():
        _fail(f"Evidence root is not a real directory: {root}.")
    files: set[str] = set()
    directories: set[str] = set()
    for directory, names, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in names:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.is_dir():
                _fail(f"Unsafe bundle directory: {relative}.")
            directories.add(relative)
        for name in filenames:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.is_file():
                _fail(f"Unsafe bundle artifact: {relative}.")
            files.add(relative)
    expected_files = {*_expected_artifacts(), "manifest.json"}
    if files != expected_files:
        _fail(
            "Bundle file inventory mismatch: "
            f"missing={sorted(expected_files - files)}, "
            f"extra={sorted(files - expected_files)}."
        )
    if directories != _expected_directories():
        _fail("Bundle directory inventory mismatch.")


def _artifact_records(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Validate and index the manifest's exact ordered artifact records."""
    raw_records = manifest.get("artifacts")
    if not isinstance(raw_records, list):
        _fail("Manifest artifacts must be a list.")
    records: dict[str, dict[str, object]] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            _fail("Every manifest artifact must be an object.")
        path = _safe_relative_path(raw_record.get("path"))
        if path in records:
            _fail(f"Duplicate manifest artifact path: {path}.")
        records[path] = raw_record
    expected = _expected_artifacts()
    if list(records) != sorted(expected) or set(records) != set(expected):
        _fail("Manifest artifact inventory or ordering mismatch.")
    for path, role in expected.items():
        record = records[path]
        extra = {"shape", "dtype"} if role == "dataset" else set()
        extra = {"groups"} if role == "raw_datatree" else extra
        if set(record) != {"path", "role", "bytes", "sha256", *extra}:
            _fail(f"Manifest record schema mismatch: {path}.")
        size, digest = record["bytes"], record["sha256"]
        if (
            record["role"] != role
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            _fail(f"Invalid manifest record: {path}.")
        if role == "dataset" and (
            record["shape"] != [300, 2] or record["dtype"] != "<f8"
        ):
            _fail(f"Dataset manifest contract mismatch: {path}.")
        if role == "raw_datatree" and record["groups"] != list(RAW_GROUPS):
            _fail(f"DataTree manifest contract mismatch: {path}.")
    return records


def _validate_manifest(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Validate the pinned manifest semantics against local frozen constants."""
    if set(manifest) != {
        "schema_version",
        "bundle",
        "protocol",
        "protocol_sha256",
        "provenance",
        "artifacts",
    }:
        _fail("Manifest schema mismatch.")
    protocol = manifest.get("protocol")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("bundle") != "JEAM repeated-recovery durable evidence"
        or protocol != _expected_protocol()
        or manifest.get("provenance") != EXPECTED_PROVENANCE
    ):
        _fail("Manifest protocol or provenance mismatch.")
    protocol_hash = hashlib.sha256(_canonical_json_bytes(protocol)).hexdigest()
    if manifest.get("protocol_sha256") != PROTOCOL_SHA256 or (
        protocol_hash != PROTOCOL_SHA256
    ):
        _fail("Manifest protocol digest mismatch.")
    return _artifact_records(manifest)


def _authenticate_files(root: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    """Authenticate the manifest and every payload before parsing any payload."""
    _verify_inventory(root)
    manifest_payload = _read_snapshot(root / "manifest.json")
    if hashlib.sha256(manifest_payload).hexdigest() != MANIFEST_SHA256:
        _fail("Manifest SHA256 mismatch.")
    manifest = _parse_strict_json(manifest_payload, "manifest.json")
    records = _validate_manifest(manifest)
    snapshots: dict[str, bytes] = {}
    for name, record in records.items():
        payload = _read_snapshot(root / name)
        snapshots[name] = payload
        if len(payload) != record["bytes"]:
            _fail(f"Artifact size mismatch: {name}.")
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            _fail(f"Artifact SHA256 mismatch: {name}.")
    return manifest, snapshots


def _source_payload(source: bytes | Path) -> tuple[bytes, str]:
    """Resolve a test path or return an already authenticated snapshot."""
    if isinstance(source, bytes):
        return source, "snapshot"
    return _read_snapshot(source), source.name


def _load_dataset(source: bytes | Path) -> np.ndarray:
    """Load and validate one canonical dataset without permitting pickle."""
    payload, label = _source_payload(source)
    try:
        dataset = np.load(BytesIO(payload), allow_pickle=False)
    except (EOFError, OSError, ValueError, TypeError) as error:
        raise EvidenceMismatch(f"Invalid NumPy dataset: {label}.") from error
    if not isinstance(dataset, np.ndarray):
        dataset.close()
        _fail(f"NumPy dataset must be an array: {label}.")
    if dataset.dtype != np.dtype("<f8") or dataset.shape != (300, 2):
        _fail(f"Dataset dtype or shape mismatch: {label}.")
    if (
        not np.all(np.isfinite(dataset))
        or np.any(dataset[:, 0] <= 0.0)
        or np.any(dataset[:, 1] < -np.pi)
        or np.any(dataset[:, 1] >= np.pi)
    ):
        _fail(f"Dataset support mismatch: {label}.")
    return dataset


def _array(
    dataset: xr.Dataset,
    name: str,
    dims: tuple[str, ...],
    shape: tuple[int, ...],
    dtype: str,
) -> np.ndarray:
    """Extract one exact finite DataTree variable."""
    variable = dataset[name]
    values = np.asarray(variable.values)
    if (
        variable.dims != dims
        or values.shape != shape
        or values.dtype != np.dtype(dtype)
    ):
        _fail(f"Raw variable contract mismatch: {name}.")
    if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
        _fail(f"Raw variable contains non-finite values: {name}.")
    return values


def _validate_raw(source: bytes | Path, observed: np.ndarray) -> dict[str, xr.Dataset]:
    """Load one DataTree and enforce the exact six-group raw contract."""
    payload, label = _source_payload(source)
    try:
        with xr.open_datatree(BytesIO(payload), engine="h5netcdf") as tree:
            paths = tuple(node.path.removeprefix("/") for node in tree.subtree)
            if paths != ("", *RAW_GROUPS) or tuple(tree.children) != RAW_GROUPS:
                _fail(f"Raw DataTree group contract mismatch: {label}.")
            groups = {
                name: tree.children[name].to_dataset(inherit=False).load()
                for name in RAW_GROUPS
            }
    except (OSError, ValueError) as error:
        if isinstance(error, EvidenceMismatch):
            raise
        raise EvidenceMismatch(f"Invalid raw DataTree: {label}.") from error

    prior = groups["prior"]
    if set(prior.data_vars) != {*PARAMETERS, "v_x_mean"}:
        _fail("Prior variable contract mismatch.")
    for name in PARAMETERS:
        _array(prior, name, ("chain", "draw"), (1, 100), "<f8")
    _array(prior, "v_x_mean", ("chain", "draw", "__obs__"), (1, 100, 300), "<f8")

    contracts = {
        "prior_predictive": (
            ("chain", "draw", "__obs__", "rt,response_dim"),
            (1, 100, 300, 2),
        ),
        "observed_data": (("__obs__", "rt,response_extra_dim_0"), (300, 2)),
        "posterior_predictive": (
            ("chain", "draw", "__obs__", "rt,response_dim"),
            (4, 40, 300, 2),
        ),
    }
    for group, (dims, shape) in contracts.items():
        if tuple(groups[group].data_vars) != ("rt,response",):
            _fail(f"{group} variable contract mismatch.")
        _array(groups[group], "rt,response", dims, shape, "<f8")

    posterior = groups["posterior"]
    if set(posterior.data_vars) != set(PARAMETERS):
        _fail("Posterior variable contract mismatch.")
    for name in PARAMETERS:
        _array(posterior, name, ("chain", "draw"), (4, 1000), "<f8")

    stats = groups["sample_stats"]
    if set(stats.data_vars) != {"nstep_in", "nstep_out"}:
        _fail("Sample-stat variable contract mismatch.")
    for name in ("nstep_in", "nstep_out"):
        _array(
            stats,
            name,
            ("chain", "draw", f"{name}_dim_0"),
            (4, 1000, 4),
            "<i8",
        )
    raw_observed = np.asarray(groups["observed_data"]["rt,response"].values)
    if not np.array_equal(raw_observed, observed):
        _fail("observed_data does not exactly bind the retained dataset.")
    return groups


def _load_verified_bundle(
    root: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, np.ndarray],
    dict[str, dict[str, xr.Dataset]],
]:
    """Authenticate first, then parse all canonical evidence payloads."""
    manifest, snapshots = _authenticate_files(root)
    result = _parse_strict_json(snapshots["result.json"], "result.json")
    measurements: dict[str, dict[str, object]] = {}
    datasets: dict[str, np.ndarray] = {}
    groups: dict[str, dict[str, xr.Dataset]] = {}
    for name, *_ in SCENARIOS:
        prefix = f"scenarios/{name}"
        measurements[name] = _parse_strict_json(
            snapshots[f"{prefix}/measurements.json"], f"{name}/measurements.json"
        )
        datasets[name] = _load_dataset(snapshots[f"{prefix}/dataset.npy"])
        groups[name] = _validate_raw(snapshots[f"{prefix}/raw.nc"], datasets[name])
    try:
        environment = snapshots["environment.txt"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceMismatch("Environment snapshot is not UTF-8.") from error
    if not environment.endswith("\n") or any(
        marker in environment
        for marker in ("/Users/", "/home/", "/private/", "file://")
    ):
        _fail("Environment snapshot is not portable canonical text.")
    return manifest, result, measurements, datasets, groups


def verify_integrity(root: str | Path = DEFAULT_BUNDLE) -> dict[str, object]:
    """Verify every canonical byte and raw storage contract."""
    manifest, *_ = _load_verified_bundle(Path(root))
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the standalone verifier CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", type=Path, default=DEFAULT_BUNDLE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the bundle and return a process-compatible status."""
    root = _parse_args(argv).bundle
    try:
        verify_integrity(root)
    except (EvidenceMismatch, OSError) as error:
        print(f"JEAM evidence verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"artifacts": 14, "integrity": "passed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

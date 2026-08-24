"""Provenance and manifest handling for durable JEAM recovery evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import platform
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from scripts.benchmark_jeam_recovery_evidence import (
    _RAW_GROUPS,
    _atomic_bytes,
    _canonical_json_bytes,
    _sha256_file,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_HSSM_REPOSITORY_URL = "https://github.com/lnccbrown/HSSM.git"
_JEAM_REPOSITORY_URL = "https://github.com/AlexanderFengler/JEAM.git"


def _git_output(repository: Path, *arguments: str) -> str:
    """Run one read-only Git query and return stripped standard output."""
    return subprocess.check_output(
        ("git", *arguments), cwd=repository, text=True
    ).strip()


def _environment_snapshot(*, hssm_revision: str, jeam_revision: str) -> bytes:
    """Return a path-free, normalized installed-distribution snapshot."""
    packages: dict[str, str] = {}
    for distribution in importlib_metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        version = distribution.version
        previous = packages.setdefault(name, version)
        if previous != version:
            raise RuntimeError(
                f"Conflicting installed versions for {name!r}: "
                f"{previous!r} and {version!r}."
            )
    packages.update(
        hssm=f"git+{_HSSM_REPOSITORY_URL}@{hssm_revision}",
        jeam=f"git+{_JEAM_REPOSITORY_URL}@{jeam_revision}",
    )
    return "".join(
        f"{name} @ {value}\n" if value.startswith("git+") else f"{name}=={value}\n"
        for name, value in sorted(packages.items())
    ).encode()


def prepare_evidence_bundle(
    directory: str | Path,
    *,
    repository: str | Path,
    imported_hssm_file: str | Path,
    protocol_base_revision: str,
    jeam_revision: str,
    source_paths: Sequence[str | Path],
) -> dict[str, object]:
    """Preflight a clean source tree and create a provenance-bound bundle root."""
    target, repo = Path(directory), Path(repository).resolve()
    imported = Path(imported_hssm_file).resolve()
    if target.exists():
        raise FileExistsError(target)
    if Path(_git_output(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise RuntimeError("repository is not the resolved Git worktree root.")
    if not imported.is_file() or not imported.is_relative_to(repo / "src" / "hssm"):
        raise RuntimeError("Imported hssm does not resolve from the capture worktree.")
    if status := _git_output(repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError(f"Evidence capture requires a clean worktree: {status}")
    _git_output(repo, "merge-base", "--is-ancestor", protocol_base_revision, "HEAD")
    revision = _git_output(repo, "rev-parse", "HEAD")
    tree = _git_output(repo, "rev-parse", "HEAD^{tree}")

    source_sha256: dict[str, str] = {}
    for source_path in source_paths:
        source = (repo / source_path).resolve()
        if not source.is_relative_to(repo) or not source.is_file():
            raise ValueError(f"Invalid evidence source path: {source_path!s}")
        source_sha256[source.relative_to(repo).as_posix()] = _sha256_file(source)
    environment = _environment_snapshot(
        hssm_revision=revision, jeam_revision=jeam_revision
    )

    target.mkdir(parents=True)
    (target / "scenarios").mkdir()
    _atomic_bytes(target / "environment.txt", environment)
    return {
        "producer_revision": revision,
        "producer_tree": tree,
        "protocol_base_revision": protocol_base_revision,
        "clean_worktree": True,
        "jeam_revision": jeam_revision,
        "source_sha256": dict(sorted(source_sha256.items())),
        "environment_sha256": hashlib.sha256(environment).hexdigest(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }


def _scenario_names(protocol: Mapping[str, object]) -> list[str]:
    scenarios = protocol.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("protocol must declare a non-empty scenario list.")
    names: list[str] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict) or not isinstance(scenario.get("name"), str):
            raise TypeError("Each protocol scenario must be a mapping with a name.")
        name = scenario["name"]
        if name in names or name in {"", ".", ".."} or re.search(r"[/\\\0]", name):
            raise ValueError(f"Invalid or duplicate scenario name: {name!r}.")
        names.append(name)
    return names


def _artifact_record(root: Path, path: Path, role: str) -> dict[str, object]:
    """Describe one hash-bound bundle artifact without absolute paths."""
    record: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if role == "dataset":
        dataset = np.load(path, allow_pickle=False)
        record.update(shape=list(dataset.shape), dtype=dataset.dtype.str)
    elif role == "raw_datatree":
        with xr.open_datatree(path, engine="h5netcdf") as tree:
            groups = tuple(
                node.path.removeprefix("/") for node in tree.subtree if node.path != "/"
            )
        if groups != _RAW_GROUPS:
            raise RuntimeError(
                f"Unexpected raw evidence groups in {path.name}: {groups}."
            )
        record["groups"] = list(groups)
    return record


def finalize_evidence_bundle(
    directory: str | Path,
    *,
    result: Mapping[str, object],
    protocol: Mapping[str, object],
    provenance: Mapping[str, object],
) -> Path:
    """Write the derived result and final hash manifest after all scenarios finish."""
    root = Path(directory)
    names = _scenario_names(protocol)
    result_payload = _canonical_json_bytes(result)
    protocol_sha256 = hashlib.sha256(_canonical_json_bytes(protocol)).hexdigest()
    result_path = root / "result.json"
    _atomic_bytes(result_path, result_payload)

    expected: dict[Path, str] = {
        root / "environment.txt": "environment",
        result_path: "derived_result",
        **{
            root / "scenarios" / name / filename: role
            for name in names
            for filename, role in (
                ("dataset.npy", "dataset"),
                ("measurements.json", "measurements"),
                ("raw.nc", "raw_datatree"),
            )
        },
    }
    missing = [path for path in expected if not path.is_file()]
    if missing:
        relative = ", ".join(path.relative_to(root).as_posix() for path in missing)
        raise RuntimeError(f"Evidence bundle is incomplete: {relative}")
    actual = {
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if unexpected := actual - expected.keys():
        relative = ", ".join(
            path.relative_to(root).as_posix() for path in sorted(unexpected)
        )
        raise RuntimeError(f"Evidence bundle has unexpected files: {relative}")

    artifacts = [
        _artifact_record(root, path, role) for path, role in sorted(expected.items())
    ]
    manifest = {
        "schema_version": 1,
        "bundle": "JEAM repeated-recovery durable evidence",
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "provenance": provenance,
        "artifacts": artifacts,
    }
    manifest_path = root / "manifest.json"
    _atomic_bytes(manifest_path, _canonical_json_bytes(manifest))
    return manifest_path

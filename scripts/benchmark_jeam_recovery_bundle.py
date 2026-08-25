"""Provenance and manifest handling for durable JEAM recovery evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import platform
import re
import subprocess
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from scripts.benchmark_jeam_recovery_evidence import (
    _RAW_GROUPS,
    _atomic_bytes,
    _canonical_json_bytes,
    _fsync_directory,
    _require_posix_durability,
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


def _durable_mkdir(directory: Path) -> None:
    """Create a directory chain and flush every new parent entry."""
    missing: list[Path] = []
    cursor = directory
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if not cursor.is_dir():
        raise NotADirectoryError(cursor)
    for path in reversed(missing):
        path.mkdir()
        _fsync_directory(path.parent)


def _declared_jeam_revision(repository: Path) -> str:
    """Return the exact JEAM commit declared by the HSSM source tree."""
    with (repository / "pyproject.toml").open("rb") as source_file:
        configuration = tomllib.load(source_file)
    try:
        source = configuration["tool"]["uv"]["sources"]["jeam"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("HSSM must declare an exact JEAM source pin.") from error
    if (
        not isinstance(source, dict)
        or source.get("git") != _JEAM_REPOSITORY_URL
        or not isinstance(revision := source.get("rev"), str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
    ):
        raise RuntimeError("HSSM must declare an exact JEAM Git revision.")
    return revision


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
    _require_posix_durability()
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
    declared_jeam_revision = _declared_jeam_revision(repo)
    if jeam_revision != declared_jeam_revision:
        raise RuntimeError(
            "Installed JEAM revision does not match HSSM's declared pin: "
            f"{jeam_revision!r} != {declared_jeam_revision!r}."
        )
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

    _durable_mkdir(target)
    _durable_mkdir(target / "scenarios")
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


def _write_or_attest(target: Path, payload: bytes) -> None:
    """Create a final document, or accept an identical retry checkpoint."""
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError(f"{target} already records different content.")
        return
    _atomic_bytes(target, payload)


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
    expected: dict[Path, str] = {
        root / "environment.txt": "environment",
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
    environment_hash = _sha256_file(root / "environment.txt")
    if provenance.get("environment_sha256") != environment_hash:
        raise RuntimeError("Evidence environment no longer matches its provenance.")
    actual = {
        path
        for path in root.rglob("*")
        if path.is_file() and path not in {root / "manifest.json", result_path}
    }
    if unexpected := actual - expected.keys():
        relative = ", ".join(
            path.relative_to(root).as_posix() for path in sorted(unexpected)
        )
        raise RuntimeError(f"Evidence bundle has unexpected files: {relative}")

    records = {
        path: _artifact_record(root, path, role) for path, role in expected.items()
    }
    _write_or_attest(result_path, result_payload)
    expected[result_path] = "derived_result"
    records[result_path] = _artifact_record(root, result_path, "derived_result")
    artifacts = [records[path] for path in sorted(expected)]
    manifest = {
        "schema_version": 1,
        "bundle": "JEAM repeated-recovery durable evidence",
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "provenance": provenance,
        "artifacts": artifacts,
    }
    manifest_path = root / "manifest.json"
    _write_or_attest(manifest_path, _canonical_json_bytes(manifest))
    return manifest_path

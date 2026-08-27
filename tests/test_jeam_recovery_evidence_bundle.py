"""Provenance and manifest contracts for JEAM recovery evidence bundles."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

import scripts.benchmark_jeam_recovery_bundle as bundle
import scripts.benchmark_jeam_recovery_evidence as evidence

if TYPE_CHECKING:
    from pathlib import Path


JEAM_REVISION = "ede7a4f4faf226e4dae52c84dfb01012939cccdc"


def _repository(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repository"
    package = repository / "src" / "hssm" / "__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text("", encoding="utf-8")
    source = repository / "pyproject.toml"
    source.write_text(
        "[project]\n"
        "name='hssm'\n"
        "[tool.uv.sources]\n"
        "jeam = { git = "
        "'https://github.com/AlexanderFengler/JEAM.git', "
        f"rev = '{JEAM_REVISION}' }}\n",
        encoding="utf-8",
    )
    return repository, package, source


def _clean_environment(
    monkeypatch: pytest.MonkeyPatch, repository: Path, status: str = ""
) -> None:
    """Install deterministic source and environment answers."""
    answers = {
        ("rev-parse", "--show-toplevel"): str(repository),
        ("status", "--porcelain", "--untracked-files=all"): status,
        ("merge-base", "--is-ancestor", "base", "HEAD"): "",
        ("rev-parse", "HEAD"): "producer",
        ("rev-parse", "HEAD^{tree}"): "tree",
    }
    monkeypatch.setattr(
        bundle,
        "_git_output",
        lambda received, *args: (
            answers[args] if received == repository else pytest.fail("wrong repository")
        ),
    )
    monkeypatch.setattr(
        bundle.importlib_metadata,
        "distributions",
        lambda: (
            SimpleNamespace(metadata={"Name": "NumPy"}, version="2.4.0"),
            SimpleNamespace(metadata={"Name": "hssm"}, version="0.3.0"),
            SimpleNamespace(metadata={"Name": "JEAM"}, version="0.1.0"),
        ),
    )


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    repository, package, _ = _repository(tmp_path)
    _clean_environment(monkeypatch, repository)
    root = tmp_path / "bundle"
    provenance = bundle.prepare_evidence_bundle(
        root,
        repository=repository,
        imported_hssm_file=package,
        protocol_base_revision="base",
        jeam_revision=JEAM_REVISION,
        source_paths=("pyproject.toml",),
    )
    return root, provenance


def _complete_scenario(root: Path, name: str) -> None:
    scenario = root / "scenarios" / name
    scenario.mkdir()
    with (scenario / "dataset.npy").open("wb") as target:
        np.save(target, np.ones((2, 2)), allow_pickle=False)
    xr.DataTree.from_dict(
        {
            group: xr.Dataset({"value": ("draw", [1.0, 2.0])})
            for group in evidence._RAW_GROUPS
        }
    ).to_netcdf(scenario / "raw.nc", engine="h5netcdf")
    (scenario / "measurements.json").write_text('{"schema_version":1}\n')


def test_prepare_attests_clean_source_and_path_free_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bind clean source, source hashes, and a path-free environment."""
    synced = []
    monkeypatch.setattr(bundle, "_fsync_directory", synced.append)
    root, provenance = _prepare(tmp_path, monkeypatch)
    environment = (root / "environment.txt").read_bytes()

    assert environment.splitlines() == [
        b"hssm @ git+https://github.com/lnccbrown/HSSM.git@producer",
        (
            b"jeam @ git+https://github.com/AlexanderFengler/JEAM.git@"
            + JEAM_REVISION.encode()
        ),
        b"numpy==2.4.0",
    ]
    assert str(tmp_path).encode() not in environment
    assert tuple(
        provenance[key]
        for key in ("producer_revision", "producer_tree", "clean_worktree")
    ) == ("producer", "tree", True)
    source = tmp_path / "repository" / "pyproject.toml"
    assert provenance["source_sha256"] == {
        "pyproject.toml": hashlib.sha256(source.read_bytes()).hexdigest()
    }
    assert provenance["environment_sha256"] == hashlib.sha256(environment).hexdigest()
    assert synced == [tmp_path, root]


def test_prepare_rejects_drifted_jeam_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never start canonical capture with a JEAM revision other than the source pin."""
    repository, package, _ = _repository(tmp_path)
    _clean_environment(monkeypatch, repository)
    root = tmp_path / "bundle"

    with pytest.raises(RuntimeError, match="Installed JEAM revision"):
        bundle.prepare_evidence_bundle(
            root,
            repository=repository,
            imported_hssm_file=package,
            protocol_base_revision="base",
            jeam_revision="deadbeef",
            source_paths=("pyproject.toml",),
        )

    assert not root.exists()


def test_prepare_rejects_unsupported_platform_without_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject unsupported durability before creating any bundle parent."""
    target = tmp_path / "absent" / "bundle"

    def reject() -> None:
        raise RuntimeError("POSIX durability required")

    monkeypatch.setattr(bundle, "_require_posix_durability", reject)

    with pytest.raises(RuntimeError, match="POSIX durability"):
        bundle.prepare_evidence_bundle(
            target,
            repository=tmp_path,
            imported_hssm_file=tmp_path / "hssm.py",
            protocol_base_revision="base",
            jeam_revision=JEAM_REVISION,
            source_paths=(),
        )

    assert not target.parent.exists()


@pytest.mark.parametrize(
    ("inside", "status", "message"),
    [
        (False, "", "Imported hssm"),
        (True, " M tracked.py", "clean worktree"),
    ],
)
def test_prepare_rejects_wrong_or_dirty_source_before_writing(
    inside: bool,
    status: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject wrong imports and dirty source before creating the bundle."""
    repository, package, _ = _repository(tmp_path)
    _clean_environment(monkeypatch, repository, status)
    root = tmp_path / "bundle"
    with pytest.raises(RuntimeError, match=message):
        bundle.prepare_evidence_bundle(
            root,
            repository=repository,
            imported_hssm_file=package if inside else tmp_path / "elsewhere.py",
            protocol_base_revision="base",
            jeam_revision="jeam",
            source_paths=(),
        )
    assert not root.exists()


def test_manifest_binds_exact_inventory_hashes_and_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bind every exact artifact, schema descriptor, size, and hash."""
    root, provenance = _prepare(tmp_path, monkeypatch)
    _complete_scenario(root, "first")
    protocol = {"trials": 2, "scenarios": [{"name": "first", "seed": 7}]}
    result = {"schema_version": 2, "gate": {"passed": True, "failures": []}}

    manifest_path = bundle.finalize_evidence_bundle(
        root, result=result, protocol=protocol, provenance=provenance
    )
    manifest = json.loads(manifest_path.read_bytes())

    assert manifest["schema_version"] == 1
    assert manifest["protocol"] == protocol
    assert (
        manifest["protocol_sha256"]
        == hashlib.sha256(evidence._canonical_json_bytes(protocol)).hexdigest()
    )
    assert [(item["path"], item["role"]) for item in manifest["artifacts"]] == [
        ("environment.txt", "environment"),
        ("result.json", "derived_result"),
        ("scenarios/first/dataset.npy", "dataset"),
        ("scenarios/first/measurements.json", "measurements"),
        ("scenarios/first/raw.nc", "raw_datatree"),
    ]
    for item in manifest["artifacts"]:
        path = root / item["path"]
        assert (item["bytes"], item["sha256"]) == (
            path.stat().st_size,
            evidence._sha256_file(path),
        )
        assert str(tmp_path) not in item["path"]
    assert (
        manifest["artifacts"][2] | {"shape": [2, 2], "dtype": "<f8"}
        == manifest["artifacts"][2]
    )
    assert manifest["artifacts"][-1]["groups"] == list(evidence._RAW_GROUPS)
    assert json.loads((root / "result.json").read_bytes()) == result


@pytest.mark.parametrize("fault", ["missing", "extra"])
def test_manifest_rejects_missing_or_extra_files(fault: str, tmp_path: Path) -> None:
    """Never finalize an incomplete or contaminated inventory."""
    root = tmp_path / fault
    (root / "scenarios").mkdir(parents=True)
    (root / "environment.txt").write_text("numpy==2\n")
    name = "missing"
    if fault == "extra":
        name = "first"
        _complete_scenario(root, name)
        (root / "undeclared.txt").write_text("unexpected")

    with pytest.raises(RuntimeError, match="incomplete|unexpected"):
        bundle.finalize_evidence_bundle(
            root,
            result={"schema_version": 2},
            protocol={"scenarios": [{"name": name}]},
            provenance={
                "environment_sha256": hashlib.sha256(b"numpy==2\n").hexdigest()
            },
        )
    assert not (root / "result.json").exists()
    assert not (root / "manifest.json").exists()


def test_manifest_retry_attests_the_completed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume finalization after a manifest-write fault without replacing output."""
    root, provenance = _prepare(tmp_path, monkeypatch)
    _complete_scenario(root, "first")
    protocol = {"scenarios": [{"name": "first"}]}
    result = {"schema_version": 2, "gate": {"passed": True}}
    atomic_bytes = bundle._atomic_bytes
    failed = False

    def fail_manifest(path: Path, payload: bytes) -> None:
        nonlocal failed
        if path.name == "manifest.json" and not failed:
            failed = True
            raise OSError("injected manifest failure")
        atomic_bytes(path, payload)

    monkeypatch.setattr(bundle, "_atomic_bytes", fail_manifest)
    with pytest.raises(OSError, match="injected manifest failure"):
        bundle.finalize_evidence_bundle(
            root, result=result, protocol=protocol, provenance=provenance
        )
    assert (root / "result.json").is_file()
    assert not (root / "manifest.json").exists()

    manifest = bundle.finalize_evidence_bundle(
        root, result=result, protocol=protocol, provenance=provenance
    )
    assert manifest.is_file()


def test_manifest_rejects_environment_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a changed environment record instead of authenticating two hashes."""
    root, provenance = _prepare(tmp_path, monkeypatch)
    _complete_scenario(root, "first")
    (root / "environment.txt").write_text("changed\n")

    with pytest.raises(RuntimeError, match="environment"):
        bundle.finalize_evidence_bundle(
            root,
            result={"schema_version": 2},
            protocol={"scenarios": [{"name": "first"}]},
            provenance=provenance,
        )

    assert not (root / "result.json").exists()

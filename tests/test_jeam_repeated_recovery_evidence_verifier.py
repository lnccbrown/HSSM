"""Independent tests for the durable JEAM repeated-recovery verifier."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from scripts import verify_jeam_repeated_recovery_evidence as verifier

BUNDLE = (
    Path(__file__).parents[1] / "benchmarks" / "evidence" / "jeam_repeated_recovery_v2"
)


def _copy_bundle(tmp_path: Path, name: str = "bundle") -> Path:
    target = tmp_path / name
    return Path(shutil.copytree(BUNDLE, target))


def test_canonical_integrity_and_exact_raw_contract() -> None:
    """Authenticate all 14 artifacts and bind all six groups to each dataset."""
    manifest, result, measurements, datasets, groups = verifier._load_verified_bundle(
        BUNDLE
    )

    assert manifest["protocol"] == verifier._expected_protocol()
    assert manifest["provenance"] == verifier.EXPECTED_PROVENANCE
    assert result["schema_version"] == 2
    assert tuple(measurements) == tuple(name for name, *_ in verifier.SCENARIOS)
    for name, *_ in verifier.SCENARIOS:
        assert datasets[name].shape == (300, 2)
        assert tuple(groups[name]) == verifier.RAW_GROUPS
        np.testing.assert_array_equal(
            groups[name]["observed_data"]["rt,response"], datasets[name]
        )


def test_payload_hashing_precedes_payload_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-size corrupt JSON payload must fail rehashing before parsing."""
    root = _copy_bundle(tmp_path)
    result = root / "result.json"
    result.write_bytes(result.read_bytes().replace(b'"benchmark"', b'"benchmArk"', 1))
    loaded: list[str] = []
    original = verifier._parse_strict_json

    def recording_parser(payload: bytes, label: str, *, canonical: bool = True):
        loaded.append(label)
        return original(payload, label, canonical=canonical)

    monkeypatch.setattr(verifier, "_parse_strict_json", recording_parser)

    with pytest.raises(verifier.EvidenceMismatch, match="SHA256.*result.json"):
        verifier.verify_integrity(root)
    assert loaded == ["manifest.json"]


def test_manifest_hashing_precedes_manifest_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The external root-of-trust digest must gate manifest interpretation."""
    root = _copy_bundle(tmp_path)
    manifest = root / "manifest.json"
    manifest.write_bytes(manifest.read_bytes().replace(b'"bundle"', b'"bundlE"', 1))
    loaded: list[str] = []
    original = verifier._parse_strict_json

    def recording_parser(payload: bytes, label: str, *, canonical: bool = True):
        loaded.append(label)
        return original(payload, label, canonical=canonical)

    monkeypatch.setattr(verifier, "_parse_strict_json", recording_parser)

    with pytest.raises(verifier.EvidenceMismatch, match="Manifest SHA256 mismatch"):
        verifier.verify_integrity(root)
    assert loaded == []


@pytest.mark.parametrize("target_name", ("manifest.json", "result.json"))
def test_parsers_consume_the_authenticated_snapshot_after_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_name: str
) -> None:
    """Replacing a path after its read cannot change the bytes later parsed."""
    root = _copy_bundle(tmp_path)
    target = root / target_name
    original = verifier._read_snapshot
    reads = 0

    def replacing_reader(path: Path) -> bytes:
        nonlocal reads
        payload = original(path)
        if path == target:
            reads += 1
            target.replace(root / f"authenticated-{target_name}")
            target.write_bytes(b"x" * len(payload))
        return payload

    monkeypatch.setattr(verifier, "_read_snapshot", replacing_reader)

    assert verifier.verify_integrity(root)["schema_version"] == 1
    assert reads == 1
    assert target.read_bytes().startswith(b"x")


@pytest.mark.parametrize(
    "payload, message",
    (
        (b'{"x":1,"x":2}\n', "Duplicate JSON member"),
        (b'{"x":NaN}\n', "Non-finite JSON constant"),
        (b'{"x":1e999}\n', "Non-finite JSON number"),
        (b'{"x":' + b"1" * 5000 + b"}\n", "Invalid JSON payload"),
    ),
    ids=("duplicate", "nan", "overflow", "huge-integer"),
)
def test_strict_json_rejects_duplicates_and_nonfinite_numbers(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    """Reject JSON extensions that could change scientific interpretation."""
    path = tmp_path / "invalid.json"
    path.write_bytes(payload)

    with pytest.raises(verifier.EvidenceMismatch, match=message):
        verifier._load_strict_json(path, canonical=False)


@pytest.mark.parametrize(
    "value",
    ("../escape", "/absolute", "a/../b", "a\\b", "C:/escape", "./relative"),
)
def test_unsafe_manifest_paths_are_rejected(value: str) -> None:
    """Manifest paths must be canonical POSIX paths below the evidence root."""
    with pytest.raises(verifier.EvidenceMismatch, match="Unsafe artifact path"):
        verifier._safe_relative_path(value)


@pytest.mark.parametrize(
    ("array", "message"),
    (
        (np.array([[object(), object()]], dtype=object), "Invalid NumPy dataset"),
        (np.zeros((299, 2), dtype=np.float64), "dtype or shape mismatch"),
    ),
)
def test_unsafe_numpy_datasets_are_rejected(
    tmp_path: Path, array: np.ndarray, message: str
) -> None:
    """Never unpickle evidence arrays or accept a noncanonical shape."""
    path = tmp_path / "dataset.npy"
    np.save(path, array)

    with pytest.raises(verifier.EvidenceMismatch, match=message):
        verifier._load_dataset(path)


def test_non_array_and_truncated_numpy_payloads_are_normalized(tmp_path: Path) -> None:
    """NPZ archives and EOF failures must fail closed as evidence mismatches."""
    archive = tmp_path / "dataset.npz"
    np.savez(archive, data=np.zeros((300, 2)))
    with pytest.raises(verifier.EvidenceMismatch, match="must be an array"):
        verifier._load_dataset(archive)

    empty = tmp_path / "empty.npy"
    empty.write_bytes(b"")
    with pytest.raises(verifier.EvidenceMismatch, match="Invalid NumPy dataset"):
        verifier._load_dataset(empty)


def test_exact_inventory_rejects_extra_artifacts(tmp_path: Path) -> None:
    """The externally pinned manifest is the only file outside its 14 records."""
    root = _copy_bundle(tmp_path)
    (root / "extra.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(verifier.EvidenceMismatch, match="inventory mismatch"):
        verifier.verify_integrity(root)


def test_root_and_nested_symlinks_are_rejected(tmp_path: Path) -> None:
    """No evidence root, nested directory, or artifact may redirect access."""
    root_link = tmp_path / "root-link"
    root_link.symlink_to(BUNDLE, target_is_directory=True)
    with pytest.raises(verifier.EvidenceMismatch, match="not a real directory"):
        verifier.verify_integrity(root_link)

    nested_root = _copy_bundle(tmp_path, "nested-bundle")
    scenario = nested_root / "scenarios" / "baseline_asymmetric"
    shutil.rmtree(scenario)
    scenario.symlink_to(
        BUNDLE / "scenarios" / "baseline_asymmetric", target_is_directory=True
    )
    with pytest.raises(verifier.EvidenceMismatch, match="Unsafe bundle directory"):
        verifier.verify_integrity(nested_root)

    artifact_root = _copy_bundle(tmp_path, "artifact-bundle")
    dataset = artifact_root / "scenarios" / "baseline_asymmetric" / "dataset.npy"
    dataset.unlink()
    dataset.symlink_to(BUNDLE / "scenarios" / "baseline_asymmetric" / "dataset.npy")
    with pytest.raises(verifier.EvidenceMismatch, match="Unsafe bundle artifact"):
        verifier.verify_integrity(artifact_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_group", "group contract"),
        ("wrong_shape", "Raw variable contract"),
        ("wrong_dtype", "Raw variable contract"),
        ("observed_binding", "exactly bind"),
    ),
)
def test_malformed_raw_contract_is_rejected(
    tmp_path: Path, mutation: str, message: str
) -> None:
    """The six groups, dimensions, dtypes, and observed binding are exact."""
    scenario = BUNDLE / "scenarios" / "baseline_asymmetric"
    with xr.open_datatree(scenario / "raw.nc", engine="h5netcdf") as tree:
        groups = {
            name: tree.children[name].to_dataset(inherit=False).load().copy(deep=True)
            for name in verifier.RAW_GROUPS
        }
    if mutation == "missing_group":
        del groups["posterior_predictive"]
    elif mutation == "wrong_shape":
        groups["posterior"] = groups["posterior"].isel(draw=slice(None, -1))
    elif mutation == "wrong_dtype":
        groups["posterior"]["a"] = groups["posterior"]["a"].astype(np.float32)
    else:
        observed = groups["observed_data"]["rt,response"].values.copy()
        observed[0, 0] += 1.0
        groups["observed_data"]["rt,response"] = (
            groups["observed_data"]["rt,response"].dims,
            observed,
        )
    raw = tmp_path / "raw.nc"
    xr.DataTree.from_dict(groups).to_netcdf(raw, engine="h5netcdf")
    dataset = np.load(scenario / "dataset.npy", allow_pickle=False)

    with pytest.raises(verifier.EvidenceMismatch, match=message):
        verifier._validate_raw(raw, dataset)

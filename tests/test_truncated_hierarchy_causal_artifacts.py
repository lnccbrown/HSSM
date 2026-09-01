"""Tests for immutable causal-experiment artifact publication."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.truncated_hierarchy_causal_artifacts import (
    ArtifactRef,
    ArtifactStore,
    CausalArtifactError,
    canonical_json_bytes,
    decode_canonical_json,
    merge_run_directories,
)


def test_canonical_json_is_finite_sorted_and_round_trips() -> None:
    """Canonical encoding fixes ordering, whitespace, Unicode, and newline."""
    payload = {"z": [3, True, None], "a": {"unicode": "μ", "value": 1.25}}
    encoded = canonical_json_bytes(payload)

    assert encoded == (b'{"a":{"unicode":"\xce\xbc","value":1.25},"z":[3,true,null]}\n')
    assert decode_canonical_json(encoded) == payload


@pytest.mark.parametrize(
    "payload",
    [
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": Path("not-json")},
        {1: "non-string-key"},
    ],
)
def test_canonical_json_rejects_ambiguous_values(payload) -> None:
    """Non-finite, non-string-keyed, and non-JSON values are invalid."""
    with pytest.raises(CausalArtifactError):
        canonical_json_bytes(payload)


@pytest.mark.parametrize(
    "encoded",
    [
        b'{"x":NaN}\n',
        b'{"x":1,"x":2}\n',
        b'{ "x": 1 }\n',
        b'{"x":1}',
    ],
)
def test_decoder_rejects_noncanonical_or_nonstrict_json(encoded: bytes) -> None:
    """Reading enforces the same strict byte representation as writing."""
    with pytest.raises(CausalArtifactError):
        decode_canonical_json(encoded)


def test_store_publishes_without_overwrite_and_verifies_bytes(tmp_path: Path) -> None:
    """An artifact path is an immutable commit rather than a mutable filename."""
    store = ArtifactStore(tmp_path.resolve())
    reference = store.write_json("data/example.json", {"seed": 1282})

    assert reference == ArtifactRef(
        path="data/example.json",
        sha256=hashlib.sha256(b'{"seed":1282}\n').hexdigest(),
        size_bytes=len(b'{"seed":1282}\n'),
    )
    assert store.read_json(reference) == {"seed": 1282}
    with pytest.raises(CausalArtifactError, match="overwrite"):
        store.write_json("data/example.json", {"seed": 1283})


def test_store_detects_tampering_before_decode(tmp_path: Path) -> None:
    """Verification rejects changed bytes before their payload is trusted."""
    store = ArtifactStore(tmp_path.resolve())
    reference = store.write_json("cells/cell.json", {"status": "completed"})
    (tmp_path / reference.path).write_bytes(b'{"status":"failed"}\n')

    with pytest.raises(CausalArtifactError, match="wrong"):
        store.read_json(reference)


def test_ensure_accepts_only_an_identical_concurrent_input(tmp_path: Path) -> None:
    """Idempotent input materialization never silently reuses different bytes."""
    store = ArtifactStore(tmp_path.resolve())
    first = store.ensure_json("data/shared.json", {"seed": 1282})

    assert store.ensure_json("data/shared.json", {"seed": 1282}) == first
    with pytest.raises(CausalArtifactError, match="overwrite"):
        store.ensure_json("data/shared.json", {"seed": 1283})


@pytest.mark.parametrize(
    "path",
    ["", "/absolute.json", "../escape.json", "a/../escape.json", "a\\b.json"],
)
def test_store_rejects_unsafe_paths(tmp_path: Path, path: str) -> None:
    """Recorded paths cannot be absolute, traversing, or platform-dependent."""
    store = ArtifactStore(tmp_path.resolve())
    with pytest.raises(CausalArtifactError):
        store.write_bytes(path, b"payload")


def test_store_rejects_symlink_escape(tmp_path: Path) -> None:
    """A symlinked parent cannot redirect an artifact outside the run root."""
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    store = ArtifactStore(root.resolve())

    with pytest.raises(CausalArtifactError, match="escapes"):
        store.write_bytes("linked/payload.bin", b"payload")


def test_merge_accepts_identical_shared_inputs_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    """Downloaded block roots merge only when duplicate bytes are identical."""
    downloads = tmp_path / "downloads"
    first = downloads / "block-a"
    second = downloads / "block-b"
    (first / "data").mkdir(parents=True)
    (second / "data").mkdir(parents=True)
    (first / "cells").mkdir()
    (second / "cells").mkdir()
    (first / "data/shared.json").write_bytes(b'{"seed":1282}\n')
    (second / "data/shared.json").write_bytes(b'{"seed":1282}\n')
    (first / "cells/a.json").write_bytes(b'{"cell":"a"}\n')
    (second / "cells/b.json").write_bytes(b'{"cell":"b"}\n')

    summary = merge_run_directories(downloads, tmp_path / "merged")

    assert summary == {"published": 3, "identical": 1}
    assert (tmp_path / "merged/cells/a.json").is_file()
    (second / "data/shared.json").write_bytes(b'{"seed":1283}\n')
    with pytest.raises(CausalArtifactError, match="overwrite"):
        merge_run_directories(downloads, tmp_path / "conflicting")


def test_merge_rejects_symlinks_and_unexpected_roots(tmp_path: Path) -> None:
    """A download cannot smuggle links or non-contract paths into the run root."""
    downloads = tmp_path / "downloads"
    block = downloads / "block"
    block.mkdir(parents=True)
    (block / "unknown.txt").write_text("no")
    with pytest.raises(CausalArtifactError, match="unexpected"):
        merge_run_directories(downloads, tmp_path / "merged")

    (block / "unknown.txt").unlink()
    (block / "data").mkdir()
    (block / "data/link.json").symlink_to(tmp_path / "outside")
    with pytest.raises(CausalArtifactError, match="symlink"):
        merge_run_directories(downloads, tmp_path / "merged-links")

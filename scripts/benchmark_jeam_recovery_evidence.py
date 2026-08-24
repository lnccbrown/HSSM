"""Atomic on-disk checkpoints for one JEAM recovery scenario."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

_PRIOR_GROUPS = ("prior", "prior_predictive", "observed_data")
_POSTERIOR_GROUPS = ("posterior", "sample_stats")
_RAW_GROUPS = (*_PRIOR_GROUPS, *_POSTERIOR_GROUPS, "posterior_predictive")
_COMPRESSION = {"zlib": True, "complevel": 4, "shuffle": True}


def _sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA256 digest of one file."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    """Encode finite JSON deterministically with a terminating newline."""
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{payload}\n".encode()


def _atomic_write(
    target: Path,
    write: Callable[[Path], object],
    *,
    replace: bool = False,
) -> None:
    """Write through a same-directory temporary and atomically install it."""
    if target.exists() and not replace:
        raise FileExistsError(target)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write(temporary)
        with temporary.open("rb") as staged:
            os.fsync(staged.fileno())
        install = os.replace if replace else os.link
        install(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(target: Path, payload: bytes) -> None:
    """Atomically create one byte-exact file without overwriting."""

    def write(temporary: Path) -> None:
        temporary.write_bytes(payload)
        if temporary.read_bytes() != payload:
            raise RuntimeError(f"{target.name} failed its byte round trip.")

    _atomic_write(target, write)


def _group_dataset(tree: xr.DataTree, group: str) -> xr.Dataset:
    """Load an owned copy of one required, non-inherited DataTree group."""
    try:
        node = tree[group]
    except KeyError as error:
        raise ValueError(f"DataTree is missing required group {group!r}.") from error
    if not isinstance(node, xr.DataTree):
        raise ValueError(f"DataTree path {group!r} is not a group.")
    return node.to_dataset(inherit=False).load().copy(deep=True)


def _assert_same_group(
    expected: xr.DataTree, candidate: xr.DataTree, group: str
) -> None:
    """Reject duplicate groups differing beyond the volatile creation timestamp."""
    datasets = (_group_dataset(expected, group), _group_dataset(candidate, group))
    for dataset in datasets:
        dataset.attrs.pop("created_at", None)
    try:
        xr.testing.assert_identical(*datasets)
    except AssertionError as error:
        raise ValueError(f"Duplicate {group!r} group is not identical.") from error


def _compression_encoding(tree: xr.DataTree) -> dict[str, dict[str, dict[str, object]]]:
    """Return h5netcdf compression settings for all non-scalar data variables."""
    encoding: dict[str, dict[str, dict[str, object]]] = {}
    for node in tree.subtree:
        if any(not isinstance(name, str) for name in node.data_vars):
            raise TypeError("Evidence variable names must be strings.")
        variables: dict[str, dict[str, object]] = {
            str(name): dict(_COMPRESSION)
            for name, variable in node.data_vars.items()
            if variable.ndim
        }
        if node.path != "/" and variables:
            encoding[node.path] = variables
    return encoding


def _arrays_identical(first: np.ndarray, second: np.ndarray) -> bool:
    """Compare concrete dtype, shape, values, and NaN positions."""
    return (
        first.dtype == second.dtype
        and first.shape == second.shape
        and bool(
            np.array_equal(first, second, equal_nan=first.dtype.kind in {"c", "f"})
        )
    )


class ScenarioEvidenceWriter:
    """Build one scenario directory through durable ordered checkpoints."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir()
        self.dataset_path = self.directory / "dataset.npy"
        self.raw_path = self.directory / "raw.nc"
        self.measurements_path = self.directory / "measurements.json"

    def record_dataset(self, data: np.ndarray) -> Path:
        """Record data, accepting only an identical repeated attestation."""
        array = np.array(data, copy=True, subok=False)
        if array.dtype.hasobject:
            raise TypeError("The evidence dataset cannot use an object dtype.")
        if self.dataset_path.exists():
            if _arrays_identical(np.load(self.dataset_path, allow_pickle=False), array):
                return self.dataset_path
            raise FileExistsError(
                f"{self.dataset_path} already records different data."
            )

        def write(temporary: Path) -> None:
            with temporary.open("wb") as target:
                np.save(target, array, allow_pickle=False)
            if not _arrays_identical(np.load(temporary, allow_pickle=False), array):
                raise RuntimeError("The dataset checkpoint failed its round trip.")

        _atomic_write(self.dataset_path, write)
        return self.dataset_path

    def record_prior(self, tree: xr.DataTree) -> Path:
        """Create the first raw checkpoint from prior and observed groups."""
        return self._checkpoint(tree, _PRIOR_GROUPS)

    def record_posterior(self, tree: xr.DataTree) -> Path:
        """Extend the raw checkpoint with posterior and sampler groups."""
        return self._checkpoint(tree, _POSTERIOR_GROUPS, ("observed_data",))

    def record_predictive(self, tree: xr.DataTree) -> Path:
        """Complete the raw checkpoint with posterior-predictive draws."""
        return self._checkpoint(tree, _RAW_GROUPS[-1:], ("observed_data", "posterior"))

    def write_measurements(self, measurements: Mapping[str, object]) -> Path:
        """Write the final strict canonical measurements document."""
        self._require_dataset()
        self._read_raw(_RAW_GROUPS)
        _atomic_bytes(self.measurements_path, _canonical_json_bytes(measurements))
        return self.measurements_path

    def _require_dataset(self) -> None:
        if not self.dataset_path.is_file():
            raise RuntimeError("record_dataset() must complete first.")

    def _read_raw(self, expected_groups: Sequence[str]) -> xr.DataTree:
        if not self.raw_path.is_file():
            raise RuntimeError("record_prior() must complete first.")
        loaded = xr.load_datatree(self.raw_path, engine="h5netcdf")
        actual = tuple(loaded.children)
        if actual != tuple(expected_groups):
            raise RuntimeError(
                f"Unexpected raw checkpoint groups: expected {tuple(expected_groups)}, "
                f"found {actual}."
            )
        return loaded

    def _checkpoint(
        self,
        source: xr.DataTree,
        additions: Sequence[str],
        duplicates: Sequence[str] = (),
    ) -> Path:
        start = _RAW_GROUPS.index(additions[0])
        existing = _RAW_GROUPS[:start]
        self._require_dataset()
        if not existing and self.raw_path.exists():
            raise FileExistsError(self.raw_path)
        current = self._read_raw(existing) if existing else None
        if current is not None:
            for group in duplicates:
                _assert_same_group(current, source, group)
        base = source if current is None else current
        datasets = {group: _group_dataset(base, group) for group in existing}
        datasets.update({group: _group_dataset(source, group) for group in additions})
        self._write_raw(xr.DataTree.from_dict(datasets), replace=bool(existing))
        return self.raw_path

    def _write_raw(self, tree: xr.DataTree, *, replace: bool) -> None:
        def write(temporary: Path) -> None:
            tree.to_netcdf(
                temporary,
                engine="h5netcdf",
                encoding=_compression_encoding(tree),
            )
            xr.testing.assert_identical(
                tree, xr.load_datatree(temporary, engine="h5netcdf")
            )

        _atomic_write(self.raw_path, write, replace=replace)

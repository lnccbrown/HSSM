"""Durability contracts for one JEAM recovery evidence scenario."""

from __future__ import annotations

from pathlib import Path

import h5netcdf
import numpy as np
import pytest
import xarray as xr

from scripts.benchmark_jeam_recovery_evidence import (
    ScenarioEvidenceWriter,
    _canonical_json_bytes,
    _sha256_file,
)

GROUPS = tuple(
    "prior prior_predictive observed_data posterior sample_stats "
    "posterior_predictive".split()
)
DATA = np.array([[0.3, -0.2], [0.7, 0.4]], dtype=np.float64)


def _trees() -> tuple[xr.DataTree, xr.DataTree, xr.DataTree]:
    datasets = {
        "prior": xr.Dataset({"a": ("draw", [0.8, 1.0])}),
        "prior_predictive": xr.Dataset(
            {"response": (("draw", "obs"), [[0.2, 0.4], [0.3, 0.5]])}
        ),
        "observed_data": xr.Dataset({"rt,response": (("obs", "response_dim"), DATA)}),
        "posterior": xr.Dataset({"a": (("chain", "draw"), [[1.0, 1.1], [0.9, 1.0]])}),
        "sample_stats": xr.Dataset({"nstep_in": (("chain", "draw"), [[2, 3], [4, 5]])}),
        "posterior_predictive": xr.Dataset(
            {"response": (("chain", "draw", "obs"), np.arange(8).reshape(2, 2, 2))}
        ),
    }

    def tree(groups, created_at):
        selected = {group: datasets[group] for group in groups}
        for duplicate in {"observed_data", "posterior"} & selected.keys():
            selected[duplicate] = selected[duplicate].assign_attrs(
                created_at=created_at
            )
        return xr.DataTree.from_dict(selected)

    return (
        tree(GROUPS[:3], "prior"),
        tree(("posterior", "sample_stats", "observed_data"), "sampling"),
        tree(("posterior", "observed_data", "posterior_predictive"), "predictive"),
    )


def test_raw_checkpoint_progresses_exactly_and_is_compressed(tmp_path: Path) -> None:
    """Every stage extends one compressed tree and accepts volatile timestamps."""
    writer = ScenarioEvidenceWriter(tmp_path / "scenario")
    writer.record_dataset(DATA)
    trees = _trees()

    for method, tree, expected_groups in (
        (writer.record_prior, trees[0], GROUPS[:3]),
        (writer.record_posterior, trees[1], GROUPS[:5]),
        (writer.record_predictive, trees[2], GROUPS),
    ):
        method(tree)
        with xr.open_datatree(writer.raw_path, engine="h5netcdf") as checkpoint:
            checkpoint.load()
            assert tuple(checkpoint.children) == expected_groups

    with h5netcdf.File(writer.raw_path, "r") as raw_file:
        for group in GROUPS:
            for variable in raw_file.groups[group].variables.values():
                if variable.dimensions:
                    filters = variable.filters()
                    assert (
                        filters["zlib"],
                        filters["complevel"],
                        filters["shuffle"],
                    ) == (
                        True,
                        4,
                        True,
                    )


def test_dataset_and_final_json_round_trip_strictly_without_overwrite(
    tmp_path: Path,
) -> None:
    """Round-trip arrays and strict JSON while refusing replacement."""
    writer = ScenarioEvidenceWriter(tmp_path / "scenario")
    assert writer.record_dataset(DATA) == writer.dataset_path
    assert writer.record_dataset(DATA.copy()) == writer.dataset_path
    np.testing.assert_array_equal(
        np.load(writer.dataset_path, allow_pickle=False), DATA
    )

    with pytest.raises(FileExistsError, match="different data"):
        writer.record_dataset(DATA.astype(np.float32))
    with pytest.raises(TypeError, match="object dtype"):
        ScenarioEvidenceWriter(tmp_path / "object").record_dataset(
            np.array([object()], dtype=object)
        )

    for method, tree in zip(
        (writer.record_prior, writer.record_posterior, writer.record_predictive),
        _trees(),
        strict=True,
    ):
        method(tree)
    with pytest.raises(ValueError, match="Out of range"):
        writer.write_measurements({"bad": np.nan})
    assert not writer.measurements_path.exists()

    value = {"z": 3, "a": {"unicode": "μ", "finite": 1.25}}
    expected = b'{"a":{"finite":1.25,"unicode":"\xce\xbc"},"z":3}\n'
    writer.write_measurements(value)
    assert writer.measurements_path.read_bytes() == expected
    with pytest.raises(FileExistsError):
        writer.write_measurements({"z": 4})
    with pytest.raises(FileExistsError):
        ScenarioEvidenceWriter(writer.directory)
    assert _canonical_json_bytes(value) == expected


def test_invalid_order_or_group_never_creates_a_raw_checkpoint(tmp_path: Path) -> None:
    """Reject invalid progression without leaving a partial checkpoint."""
    writer = ScenarioEvidenceWriter(tmp_path / "scenario")
    prior, posterior, _ = _trees()
    with pytest.raises(RuntimeError, match="record_dataset"):
        writer.record_prior(prior)
    writer.record_dataset(DATA)
    with pytest.raises(RuntimeError, match="record_prior"):
        writer.record_posterior(posterior)
    with pytest.raises(ValueError, match="observed_data"):
        writer.record_prior(
            xr.DataTree.from_dict(
                {group: prior[group].to_dataset() for group in GROUPS[:2]}
            )
        )
    assert not writer.raw_path.exists()
    assert not tuple(writer.directory.glob(".*.tmp"))


@pytest.mark.parametrize("group", ["observed_data", "posterior"])
def test_substantive_duplicate_group_mismatch_preserves_checkpoint(
    group: str, tmp_path: Path
) -> None:
    """Reject value changes while tolerating only volatile timestamps."""
    prior, posterior, predictive = _trees()
    writer = ScenarioEvidenceWriter(tmp_path / "scenario")
    writer.record_dataset(DATA)
    writer.record_prior(prior)
    if group == "observed_data":
        posterior[group]["rt,response"][0, 0] = 99.0
        action = lambda: writer.record_posterior(posterior)
    else:
        writer.record_posterior(posterior)
        predictive[group]["a"][0, 0] = 99.0
        action = lambda: writer.record_predictive(predictive)
    original = _sha256_file(writer.raw_path)

    with pytest.raises(ValueError, match=group):
        action()
    assert _sha256_file(writer.raw_path) == original


def test_failed_serialization_preserves_checkpoint_and_removes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the last valid checkpoint when serialization fails."""
    prior, posterior, _ = _trees()
    writer = ScenarioEvidenceWriter(tmp_path / "scenario")
    writer.record_dataset(DATA)
    writer.record_prior(prior)
    original = _sha256_file(writer.raw_path)

    def fail(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("injected failure")

    monkeypatch.setattr(xr.DataTree, "to_netcdf", fail)
    with pytest.raises(OSError, match="injected failure"):
        writer.record_posterior(posterior)
    assert _sha256_file(writer.raw_path) == original
    assert not tuple(writer.directory.glob(".*.tmp"))

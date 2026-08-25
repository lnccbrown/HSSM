"""Independent tests for the durable JEAM repeated-recovery verifier."""

from __future__ import annotations

import copy
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


@pytest.fixture(scope="module")
def recomputed_science():
    """Load authenticated primary evidence and recompute its scientific result."""
    _, stored, measurements, datasets, groups = verifier._load_verified_bundle(BUNDLE)
    return stored, verifier._recompute_science(measurements, datasets, groups)


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


def test_raw_evidence_recomputes_16_of_16_truths_and_the_full_gate(
    recomputed_science,
) -> None:
    """Recompute posterior summaries, PPCs, aggregation, and every science gate."""
    stored, science = recomputed_science
    verifier._assert_same_science(
        verifier.scientific_projection(science),
        verifier.scientific_projection(stored),
    )
    assert [row["runtime"] for row in science["scenarios"]] == [
        row["runtime"] for row in stored["scenarios"]
    ]
    assert [row["mean_bulk_ess_per_second"] for row in science["aggregate"]] == (
        pytest.approx([row["mean_bulk_ess_per_second"] for row in stored["aggregate"]])
    )

    assert science["gate"] == {"passed": True, "failures": []}
    assert (
        sum(
            row["hdi_contains_truth"]
            for scenario in science["scenarios"]
            for row in scenario["parameters"]
        )
        == 16
    )
    assert all(row["hdi_inclusion_fraction"] == 1.0 for row in science["aggregate"])
    assert verifier.verify_evidence(BUNDLE) == {
        "artifacts": 14,
        "scenarios": 4,
        "hdi_inclusions": 16,
        "hdi_total": 16,
        "gate": "passed",
    }


def test_primary_objective_measurements_are_not_rerun_or_trusted_inconsistently(
    recomputed_science,
) -> None:
    """Hash-bound optimizer values remain primary but must agree internally."""
    _, science = recomputed_science
    scenario = science["scenarios"][0]
    direct = np.asarray(scenario["direct_objectives"])
    compiled = np.asarray(scenario["compiled_objectives"])

    assert scenario["maximum_objective_absolute_error"] == pytest.approx(
        np.max(np.abs(direct - compiled))
    )
    assert scenario["maximum_optimizer_parameter_absolute_error"] == 0.0
    assert scenario["optimizer_objective_absolute_error"] == 0.0

    _, _, measurements, datasets, _ = verifier._load_verified_bundle(BUNDLE)
    altered = copy.deepcopy(measurements["baseline_asymmetric"])
    objective = altered["objective"]
    assert isinstance(objective, dict)
    direct_fit = objective["direct_fixed_budget_optimizer"]
    assert isinstance(direct_fit, dict)
    direct_objective = direct_fit["objective"]
    assert isinstance(direct_objective, float)
    direct_fit["objective"] = direct_objective + 1.0
    with pytest.raises(verifier.EvidenceMismatch, match="internally inconsistent"):
        verifier._measurement_science(
            "baseline_asymmetric", altered, datasets["baseline_asymmetric"]
        )


def test_scientific_gate_boundaries_are_inclusive(recomputed_science) -> None:
    """Every exact preregistered threshold must remain a passing boundary."""
    _, original = recomputed_science
    science = copy.deepcopy(original)
    scenario = science["scenarios"][0]
    scenario["maximum_objective_absolute_error"] = 5e-5
    scenario["maximum_optimizer_parameter_absolute_error"] = 1e-12
    scenario["optimizer_objective_absolute_error"] = 5e-5
    scenario["predictive"].update(
        observed_rt_quantiles=[0.0, 0.0, 0.0],
        predictive_rt_quantiles=[0.12, 0.12, 0.12],
        mean_angle_distance=0.1,
        observed_resultant_length=0.0,
        predictive_resultant_length=0.08,
    )
    scenario["prior_predictive"]["prior_to_observed_rt_ratios"] = [0.1, 20.0]
    for parameter in scenario["parameters"]:
        parameter["mcse_sd_ratio"] = 0.05
    for aggregate in science["aggregate"]:
        name = aggregate["name"]
        aggregate.update(
            jeam_fixed_budget_bias=verifier.THRESHOLDS["maximum_absolute_bias"][name],
            jeam_fixed_budget_rmse=verifier.THRESHOLDS["maximum_rmse"][name],
            hssm_posterior_bias=-verifier.THRESHOLDS["maximum_absolute_bias"][name],
            hssm_posterior_rmse=verifier.THRESHOLDS["maximum_rmse"][name],
            hdi_inclusion_fraction=0.75,
            maximum_rhat=1.01,
            minimum_bulk_ess=500.0,
            minimum_tail_ess=500.0,
        )

    assert verifier._science_failures(science) == []


def test_stored_thresholds_cannot_relax_independent_gate(recomputed_science) -> None:
    """The evaluator must ignore artifact-owned limits and reject a true overrun."""
    stored, original = recomputed_science
    science = copy.deepcopy(original)
    science["thresholds"]["maximum_rhat"] = 1e9
    science["aggregate"][0]["maximum_rhat"] = np.nextafter(1.01, np.inf)

    assert "a: R-hat" in verifier._science_failures(science)

    altered_stored = copy.deepcopy(stored)
    altered_stored["thresholds"]["maximum_rhat"] = 1e9
    with pytest.raises(verifier.EvidenceMismatch, match="thresholds"):
        verifier._assert_same_science(
            verifier.scientific_projection(original),
            verifier.scientific_projection(altered_stored),
        )


def test_descriptive_telemetry_is_excluded_from_science(recomputed_science) -> None:
    """Timestamps, runtime, and ESS/second cannot change scientific conclusions."""
    stored, _ = recomputed_science
    altered = copy.deepcopy(stored)
    altered["generated_at_utc"] = "2099-01-01T00:00:00+00:00"
    altered["total_runtime_seconds"] = 1.0
    for scenario in altered["scenarios"]:
        scenario["runtime"] = {key: 1.0 for key in scenario["runtime"]}
        for parameter in scenario["parameters"]:
            parameter["ess_bulk_per_second"] = 1.0
    for aggregate in altered["aggregate"]:
        aggregate["mean_bulk_ess_per_second"] = 1.0

    assert verifier.scientific_projection(altered) == verifier.scientific_projection(
        stored
    )


def test_cli_defaults_to_canonical_and_returns_nonzero_on_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The standalone command verifies the canonical default and fails closed."""
    assert verifier.main([]) == 0
    assert '"gate": "passed"' in capsys.readouterr().out

    root = _copy_bundle(tmp_path)
    dataset = root / "scenarios" / "baseline_asymmetric" / "dataset.npy"
    payload = bytearray(dataset.read_bytes())
    payload[-1] ^= 1
    dataset.write_bytes(payload)

    assert verifier.main([str(root)]) == 1
    assert "SHA256 mismatch" in capsys.readouterr().err

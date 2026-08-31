"""Artifact-producing one-cell runner for HSSM qualification issue #1282.

The frozen v2 manifest is the only source of experimental choices.  This module
selects one canonical plan cell, generates or verifies its shared data artifact,
materializes the exact transformed starts, runs the requested NUTS backend, and
standardizes the retained evidence before publishing the cell result last.

Sampling and numerical diagnostics are deliberately separate functions.  The
timed sampler therefore cannot be warmed by finite-difference, PyTensor/JAX, or
likelihood-parity probes.  The ``run`` command invokes ``sample`` and ``finalize``
as separate children with fresh compilation caches; :func:`finalize_cell` runs
diagnostics only after the immutable chain artifact exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, cast

import arviz as az
import jax
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import xarray as xr
from pymc.initial_point import make_initial_point_fns_per_chain
from pymc.sampling.jax import sample_numpyro_nuts
from scipy.special import expit
from scipy.stats import truncnorm

import hssm

# Preserve direct ``python scripts/...py`` usability while the orchestrator uses
# the cleaner module form below. A clean process otherwise puts only ``scripts/``
# on ``sys.path``, which cannot resolve the repository's ``scripts`` namespace.
if not __package__:  # pragma: no cover - covered by the module subprocess test
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.truncated_hierarchy_hssm import (
    DEFAULT_DDM_NETWORK,
    HSSMBuild,
    SamplerStartArtifact,
    build_hssm_model,
    evaluate_hssm_gradients,
    extract_actual_sampler_starts,
    lba2_pytensor_jax_parity,
    validate_actual_sampler_starts,
)
from scripts.truncated_hierarchy_models import (
    Bounds,
    GeometryModel,
    LinkedNormalPrior,
    NativeTruncatedPrior,
    SyntheticHierarchyData,
    ToyDataSpec,
    build_bambi_model,
    build_direct_pymc_model,
    compare_isomorphic_models,
    evaluate_transformed_geometry,
    generate_synthetic_data,
    support_inverse,
)
from scripts.truncated_hierarchy_qualification import (
    DEFAULT_DEPENDENCY_PROFILE,
    DEFAULT_MANIFEST,
    REPO_ROOT,
    RUNNER_VERSION,
    QualificationError,
    collect_environment,
    derive_numpyro_chain_keys,
    environment_sha256,
    expand_plan,
    load_environment_catalog,
    load_manifest,
    strict_json_loads,
    validate_result_record,
    verify_result_artifacts,
    write_cell_result,
)
from scripts.truncated_hierarchy_statistics import derive_sbc_rank_tie_index

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

SamplerName = Literal["pymc", "numpyro"]
PhaseName = Literal["sample", "finalize"]

_IDENTITY_FIELDS = (
    "schema_version",
    "study_id",
    "manifest_sha256",
    "cell_id",
    "scenario_id",
    "replicate",
    "data_seed",
    "truth_seed",
    "group_seed",
    "observation_seed",
    "initialization_seed",
    "start_seeds",
    "sampler_seed",
    "sbc_draw_seed",
    "sbc_tie_seed",
    "chain_seeds",
)
_SAMPLE_STAT_ALIASES = {
    "diverging": ("diverging",),
    "energy": ("energy",),
    "tree_depth": ("tree_depth", "depth"),
    "n_steps": ("n_steps", "num_steps"),
    "step_size": ("step_size",),
    "acceptance_rate": (
        "acceptance_rate",
        "acceptance_probability",
        "mean_tree_accept",
    ),
}
_MONITORED = (
    "group_location",
    "group_scale",
    "group_first",
    "group_middle",
    "group_last",
)
_CHAIN_POSTERIOR = (*_MONITORED, "group_effect")
_MAX_TREE_DEPTH = 10
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PHASE_CONTEXT_ENV = "HSSM_TRUNCATED_HIERARCHY_PHASE_CONTEXT"
_PHASES: tuple[PhaseName, PhaseName] = ("sample", "finalize")


class RunnerError(QualificationError):
    """Raised when one cell cannot satisfy the frozen execution contract."""


@dataclass(frozen=True, slots=True)
class ExecutionIdentity:
    """Opaque parent-minted identity bound into one final cell record."""

    execution_attempt_id: str
    pair_execution_id: str | None
    pair_position: int | None
    worker_identity_sha256: str

    def __post_init__(self) -> None:
        """Require exact digests and coherent nullable pair fields."""
        if not isinstance(self.execution_attempt_id, str) or not _SHA256.fullmatch(
            self.execution_attempt_id
        ):
            raise RunnerError("execution_attempt_id must be a lowercase SHA-256")
        if not isinstance(self.worker_identity_sha256, str) or not _SHA256.fullmatch(
            self.worker_identity_sha256
        ):
            raise RunnerError("worker_identity_sha256 must be a lowercase SHA-256")
        if (self.pair_execution_id is None) != (self.pair_position is None):
            raise RunnerError(
                "pair_execution_id and pair_position must be null together"
            )
        if self.pair_execution_id is not None and (
            not isinstance(self.pair_execution_id, str)
            or not _SHA256.fullmatch(self.pair_execution_id)
        ):
            raise RunnerError("pair_execution_id must be a lowercase SHA-256")
        if self.pair_position is not None and (
            isinstance(self.pair_position, bool) or self.pair_position not in {0, 1}
        ):
            raise RunnerError("pair_position must be zero, one, or null")


@dataclass(frozen=True, slots=True)
class PhaseContext:
    """Strict one-child launch context created only by the parent orchestrator."""

    phase: PhaseName
    cell_id: str
    identity: ExecutionIdentity
    pytensor_cache: Path
    jax_cache: Path
    matplotlib_cache: Path
    xdg_cache: Path

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "phase": self.phase,
            "cell_id": self.cell_id,
            "execution_attempt_id": self.identity.execution_attempt_id,
            "pair_execution_id": self.identity.pair_execution_id,
            "pair_position": self.identity.pair_position,
            "worker_identity_sha256": self.identity.worker_identity_sha256,
            "cache_paths": {
                "pytensor": str(self.pytensor_cache),
                "jax": str(self.jax_cache),
                "matplotlib": str(self.matplotlib_cache),
                "xdg": str(self.xdg_cache),
            },
        }

    def as_jsonable(self) -> dict[str, Any]:
        """Return a self-checking canonical environment payload."""
        payload = self._payload()
        return {
            **payload,
            "context_sha256": hashlib.sha256(
                _canonical_json_bytes(payload)
            ).hexdigest(),
        }

    @classmethod
    def from_jsonable(cls, value: Any) -> PhaseContext:
        """Parse a phase context without accepting unknown or ambiguous fields."""
        if not isinstance(value, dict):
            raise RunnerError("orchestrator phase context must be a JSON object")
        expected = {
            "schema_version",
            "phase",
            "cell_id",
            "execution_attempt_id",
            "pair_execution_id",
            "pair_position",
            "worker_identity_sha256",
            "cache_paths",
            "context_sha256",
        }
        if set(value) != expected or value.get("schema_version") != 1:
            raise RunnerError("orchestrator phase context schema is invalid")
        supplied_digest = value["context_sha256"]
        if not isinstance(supplied_digest, str) or not _SHA256.fullmatch(
            supplied_digest
        ):
            raise RunnerError("orchestrator phase context digest is invalid")
        payload = {key: item for key, item in value.items() if key != "context_sha256"}
        expected_digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        if supplied_digest != expected_digest:
            raise RunnerError("orchestrator phase context digest is invalid")
        phase = value["phase"]
        cell_id = value["cell_id"]
        if phase not in {"sample", "finalize"} or not isinstance(cell_id, str):
            raise RunnerError("orchestrator phase context identity is invalid")
        cache_paths = value["cache_paths"]
        if not isinstance(cache_paths, dict) or set(cache_paths) != {
            "pytensor",
            "jax",
            "matplotlib",
            "xdg",
        }:
            raise RunnerError("orchestrator phase cache paths are invalid")
        if any(
            not isinstance(path, str) or not Path(path).is_absolute()
            for path in cache_paths.values()
        ):
            raise RunnerError("orchestrator phase cache paths must be absolute")
        return cls(
            phase=cast("PhaseName", phase),
            cell_id=cell_id,
            identity=ExecutionIdentity(
                execution_attempt_id=value["execution_attempt_id"],
                pair_execution_id=value["pair_execution_id"],
                pair_position=value["pair_position"],
                worker_identity_sha256=value["worker_identity_sha256"],
            ),
            pytensor_cache=Path(cache_paths["pytensor"]),
            jax_cache=Path(cache_paths["jax"]),
            matplotlib_cache=Path(cache_paths["matplotlib"]),
            xdg_cache=Path(cache_paths["xdg"]),
        )


@dataclass(frozen=True, slots=True)
class BuiltCell:
    """One built PyMC graph and the metadata needed to standardize it."""

    plan_entry: Mapping[str, Any]
    data_payload: Mapping[str, Any]
    pymc_model: pm.Model
    geometry: GeometryModel | None = None
    hssm_build: HSSMBuild | None = None

    def __post_init__(self) -> None:
        """Require exactly one construction source."""
        if (self.geometry is None) == (self.hssm_build is None):
            raise RunnerError("built cell requires exactly one model source")


@dataclass(slots=True)
class ArtifactState:
    """Only fully published artifacts that may be referenced by a result."""

    data_artifact: str | None = None
    data_sha256: str | None = None
    actual_start_artifact: str | None = None
    actual_start_sha256: str | None = None
    raw_chain_artifact: str | None = None
    raw_chain_sha256: str | None = None


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize strict JSON to byte-stable canonical bytes."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    """Hash exact file bytes without loading a chain artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _derive_execution_hash(nonce: str, purpose: str, *parts: str) -> str:
    """Derive one domain-separated opaque identifier from a parent nonce."""
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "contract": "hssm-truncated-hierarchy-execution-v1",
                "nonce": nonce,
                "purpose": purpose,
                "parts": list(parts),
            }
        )
    ).hexdigest()


def _worker_identity_sha256(nonce: str) -> str:
    """Bind all children from this attempt to one parent process and host."""
    return _derive_execution_hash(
        nonce,
        "worker",
        socket.gethostname(),
        str(os.getpid()),
    )


def _requires_paired_execution(plan_entry: Mapping[str, Any]) -> bool:
    """Return whether the frozen policy requires candidate/control co-location."""
    scenario = plan_entry["scenario"]
    return bool(
        scenario["tier"] == "qualification"
        and scenario["purpose"] in {"candidate", "control"}
        and scenario.get("calibration_kind") is None
    )


def _validate_execution_identity(
    plan_entry: Mapping[str, Any], identity: ExecutionIdentity
) -> None:
    """Require paired fields exactly when this planned cell belongs to a pair."""
    paired = _requires_paired_execution(plan_entry)
    if paired and identity.pair_execution_id is None:
        raise RunnerError("qualification candidate/control cells require run-pair")
    if not paired and identity.pair_execution_id is not None:
        raise RunnerError("unpaired cells must have null pair execution fields")


def _nonpaired_execution_identity(
    plan_entry: Mapping[str, Any], *, nonce: str
) -> ExecutionIdentity:
    """Derive one non-paired cell identity from a fresh parent nonce."""
    if _requires_paired_execution(plan_entry):
        raise RunnerError("this cell requires paired execution via run-pair")
    return ExecutionIdentity(
        execution_attempt_id=_derive_execution_hash(
            nonce, "cell-attempt", str(plan_entry["cell_id"])
        ),
        pair_execution_id=None,
        pair_position=None,
        worker_identity_sha256=_worker_identity_sha256(nonce),
    )


def _publish_bytes_once(path: Path, payload: bytes) -> str:
    """Create an immutable artifact or verify an existing identical writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if not path.is_file() or _file_sha256(path) != digest:
            raise RunnerError(f"existing artifact differs from canonical bytes: {path}")
        return digest
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or _file_sha256(path) != digest:
                raise RunnerError(
                    f"concurrent artifact writer produced different bytes: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _publish_file_once(path: Path, temporary: Path) -> str:
    """Publish a completed temporary file without replacing prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = _file_sha256(temporary)
    if path.exists():
        if not path.is_file() or _file_sha256(path) != digest:
            raise RunnerError(f"existing artifact differs from generated bytes: {path}")
        temporary.unlink(missing_ok=True)
        return digest
    try:
        os.link(temporary, path)
    except FileExistsError:
        if not path.is_file() or _file_sha256(path) != digest:
            raise RunnerError(
                f"concurrent artifact writer produced different bytes: {path}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def select_plan_cell(
    manifest: Mapping[str, Any], *, tier: str, cell_id: str
) -> Mapping[str, Any]:
    """Select exactly one canonical cell from the generated frozen plan."""
    matches = [
        entry for entry in expand_plan(manifest, tier) if entry["cell_id"] == cell_id
    ]
    if len(matches) != 1:
        raise RunnerError(
            f"cell_id {cell_id!r} selected {len(matches)} cells from tier {tier!r}"
        )
    return matches[0]


def resolve_qualification_pair(
    plan_entry: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Resolve one non-SBC qualification entry to candidate and control cells."""
    if not _requires_paired_execution(plan_entry):
        raise RunnerError("run-pair requires a non-SBC qualification pair member")
    replicate = int(plan_entry["replicate"])
    plan = expand_plan(manifest, "qualification")
    if plan_entry["scenario"]["purpose"] == "candidate":
        candidate = plan_entry
        control_scenario_id = candidate["scenario"]["control_id"]
    else:
        control_scenario_id = plan_entry["scenario_id"]
        candidate_matches = [
            entry
            for entry in plan
            if entry["replicate"] == replicate
            and entry["scenario"]["purpose"] == "candidate"
            and entry["scenario"].get("control_id") == control_scenario_id
        ]
        if len(candidate_matches) != 1:
            raise RunnerError(
                "control cell does not resolve to exactly one qualification candidate"
            )
        candidate = candidate_matches[0]
    control_matches = [
        entry
        for entry in plan
        if entry["replicate"] == replicate
        and entry["scenario_id"] == control_scenario_id
        and entry["scenario"]["purpose"] == "control"
    ]
    if len(control_matches) != 1:
        raise RunnerError(
            "candidate cell does not resolve to exactly one qualification control"
        )
    control = control_matches[0]
    for field in (
        "data_seed",
        "truth_seed",
        "group_seed",
        "observation_seed",
    ):
        if candidate[field] != control[field]:
            raise RunnerError(f"qualification pair changes shared {field}")
    return candidate, control


def ordered_qualification_pair(
    plan_entry: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return the frozen counterbalanced execution order for one replicate pair."""
    candidate, control = resolve_qualification_pair(plan_entry, manifest)
    return (
        (candidate, control)
        if int(plan_entry["replicate"]) % 2 == 0
        else (control, candidate)
    )


def _bounds(scenario: Mapping[str, Any]) -> Bounds:
    return Bounds(lower=scenario["lower"], upper=scenario["upper"])


def _draw_truth(plan_entry: Mapping[str, Any]) -> tuple[float, float]:
    """Draw or read the exact group hyperparameters for this shared dataset."""
    scenario = plan_entry["scenario"]
    if scenario["truth_kind"] == "fixed":
        return float(scenario["truth_group_location"]), float(
            scenario["truth_group_scale"]
        )
    if scenario["truth_kind"] != "prior_predictive":
        raise RunnerError(f"unsupported truth_kind: {scenario['truth_kind']}")
    bounds = _bounds(scenario)
    lower = (
        -np.inf
        if bounds.lower is None
        else (bounds.lower - float(scenario["prior_hyper_location"])) / 0.25
    )
    upper = (
        np.inf
        if bounds.upper is None
        else (bounds.upper - float(scenario["prior_hyper_location"])) / 0.25
    )
    rng = np.random.default_rng(int(plan_entry["truth_seed"]))
    location = float(
        truncnorm.rvs(
            lower,
            upper,
            loc=float(scenario["prior_hyper_location"]),
            scale=0.25,
            random_state=rng,
        )
    )
    scale = float(0.3 * rng.weibull(1.5))
    if not math.isfinite(location) or not math.isfinite(scale) or scale <= 0:
        raise RunnerError("prior-predictive truth draw was invalid")
    return location, scale


def _base_hierarchy_data(
    plan_entry: Mapping[str, Any], *, group_location: float, group_scale: float
) -> SyntheticHierarchyData:
    scenario = plan_entry["scenario"]
    spec = ToyDataSpec(
        bounds=_bounds(scenario),
        group_location=group_location,
        group_scale=group_scale,
        n_groups=int(scenario["n_groups"]),
        n_per_group=int(scenario["n_per_group"]),
        floatx=cast("Literal['float32', 'float64']", scenario["floatx"]),
    )
    return generate_synthetic_data(
        spec,
        group_seed=int(plan_entry["group_seed"]),
        observation_seed=int(plan_entry["observation_seed"]),
    )


def _simulate_hssm_observations(
    *, model: str, theta: dict[str, Any], observation_seed: int
) -> pd.DataFrame:
    """Run an HSSM DGP without leaking its legacy global NumPy RNG use.

    The Cython LBA simulator currently consumes NumPy's legacy global stream for
    part of its draw even when ``random_state`` seeds its explicit simulator RNG.
    One qualification cell owns its process, so temporarily seeding that legacy
    stream is safe; restoring it in ``finally`` keeps this helper referentially
    transparent for model construction and diagnostics in the same process.
    """
    legacy_state = np.random.get_state()
    try:
        np.random.seed(observation_seed)
        simulated = hssm.simulate_data(
            model=model,
            theta=theta,
            size=1,
            random_state=observation_seed,
        )
    finally:
        np.random.set_state(legacy_state)
    if not isinstance(simulated, pd.DataFrame):  # pragma: no cover - API invariant
        raise RunnerError("HSSM simulator did not return a DataFrame")
    return simulated


def generate_data_payload(plan_entry: Mapping[str, Any]) -> dict[str, Any]:
    """Generate the canonical truth/observation payload for one data owner."""
    scenario = plan_entry["scenario"]
    location, scale = _draw_truth(plan_entry)
    hierarchy = _base_hierarchy_data(
        plan_entry, group_location=location, group_scale=scale
    )
    group_index = hierarchy.group_index
    model_key = str(scenario["model"])
    if model_key == "toy_gaussian":
        observations: dict[str, Any] = {"y": hierarchy.y.tolist()}
    elif model_key in {"lba2_b", "approx_ddm_z"}:
        natural = hierarchy.group_effect[group_index]
        if model_key == "lba2_b":
            simulated = _simulate_hssm_observations(
                model="lba2",
                theta={
                    "A": np.full(group_index.size, 0.1),
                    "b": natural,
                    "v0": np.full(group_index.size, 1.0),
                    "v1": np.full(group_index.size, 1.2),
                },
                observation_seed=int(plan_entry["observation_seed"]),
            )
        else:
            simulated = _simulate_hssm_observations(
                model="ddm",
                theta={
                    "v": np.full(group_index.size, 0.5),
                    "a": np.full(group_index.size, 1.5),
                    "z": natural,
                    "t": np.full(group_index.size, 0.3),
                },
                observation_seed=int(plan_entry["observation_seed"]),
            )
        observations = {
            "rt": simulated["rt"].to_numpy().tolist(),
            "response": simulated["response"].to_numpy().tolist(),
        }
    elif model_key == "softmax_beta":
        rng = np.random.default_rng(int(plan_entry["observation_seed"]))
        probability = expit(hierarchy.group_effect[group_index])
        response = np.where(rng.random(group_index.size) < probability, 1, -1)
        observations = {"response": response.tolist()}
    else:  # pragma: no cover - manifest validation owns the model enumeration
        raise RunnerError(f"unsupported model contract: {model_key}")
    payload = {
        "schema_version": 1,
        "study_id": plan_entry["study_id"],
        "data_id": scenario["data_id"],
        "replicate": plan_entry["replicate"],
        "model": model_key,
        "floatx": scenario["floatx"],
        "bounds": {"lower": scenario["lower"], "upper": scenario["upper"]},
        "truth_kind": scenario["truth_kind"],
        "n_groups": scenario["n_groups"],
        "n_per_group": scenario["n_per_group"],
        "seeds": {
            "data": plan_entry["data_seed"],
            "truth": plan_entry["truth_seed"],
            "group": plan_entry["group_seed"],
            "observation": plan_entry["observation_seed"],
        },
        "group_labels": list(hierarchy.group_labels),
        "group_index": hierarchy.group_index.tolist(),
        "truth": {
            "group_location": location,
            "group_scale": scale,
            "group_effect": hierarchy.group_effect.tolist(),
        },
        "observations": observations,
    }
    # Fail at generation time rather than much later in a result validator.
    _canonical_json_bytes(payload)
    return payload


def materialize_data_artifact(
    plan_entry: Mapping[str, Any], manifest: Mapping[str, Any], artifact_root: Path
) -> tuple[Mapping[str, Any], str, str]:
    """Generate/reuse the shared data artifact, verifying byte identity."""
    payload = generate_data_payload(plan_entry)
    relative = (
        manifest["artifact_policy"]["data_path"]
        .replace("<data_id>", str(plan_entry["scenario"]["data_id"]))
        .replace("<replicate>", str(plan_entry["replicate"]))
    )
    path = artifact_root / PurePosixPath(relative)
    canonical = _canonical_json_bytes(payload)
    digest = _publish_bytes_once(path, canonical)
    loaded = strict_json_loads(path.read_text(encoding="utf-8"), source=str(path))
    if loaded != payload:
        raise RunnerError("published data artifact did not round-trip exactly")
    return payload, relative, digest


def load_data_artifact(
    plan_entry: Mapping[str, Any], manifest: Mapping[str, Any], artifact_root: Path
) -> tuple[Mapping[str, Any], str, str]:
    """Load a shared artifact only if it still equals the canonical DGP bytes."""
    payload = generate_data_payload(plan_entry)
    relative = (
        manifest["artifact_policy"]["data_path"]
        .replace("<data_id>", str(plan_entry["scenario"]["data_id"]))
        .replace("<replicate>", str(plan_entry["replicate"]))
    )
    path = artifact_root / PurePosixPath(relative)
    if not path.is_file():
        raise RunnerError(f"data artifact is unavailable: {path}")
    expected = _canonical_json_bytes(payload)
    if path.read_bytes() != expected:
        raise RunnerError("data artifact differs from the canonical DGP bytes")
    return payload, relative, hashlib.sha256(expected).hexdigest()


def _toy_data_from_payload(
    plan_entry: Mapping[str, Any], payload: Mapping[str, Any]
) -> SyntheticHierarchyData:
    scenario = plan_entry["scenario"]
    truth = cast("Mapping[str, Any]", payload["truth"])
    spec = ToyDataSpec(
        bounds=_bounds(scenario),
        group_location=float(truth["group_location"]),
        group_scale=float(truth["group_scale"]),
        n_groups=int(scenario["n_groups"]),
        n_per_group=int(scenario["n_per_group"]),
        floatx=cast("Literal['float32', 'float64']", scenario["floatx"]),
    )
    dtype = np.dtype(str(scenario["floatx"]))
    return SyntheticHierarchyData(
        group_seed=int(plan_entry["group_seed"]),
        observation_seed=int(plan_entry["observation_seed"]),
        spec=spec,
        group_labels=tuple(cast("Sequence[str]", payload["group_labels"])),
        group_index=np.asarray(payload["group_index"], dtype=np.int64),
        y=np.asarray(
            cast("Mapping[str, Any]", payload["observations"])["y"], dtype=dtype
        ),
        group_effect=np.asarray(truth["group_effect"], dtype=dtype),
    )


def _hssm_data_from_payload(payload: Mapping[str, Any]) -> pd.DataFrame:
    group_labels = cast("Sequence[str]", payload["group_labels"])
    group_index = np.asarray(payload["group_index"], dtype=np.int64)
    participants = np.asarray([group_labels[index] for index in group_index])
    observations = cast("Mapping[str, Any]", payload["observations"])
    columns = {key: np.asarray(value) for key, value in observations.items()}
    columns["participant_id"] = participants
    return pd.DataFrame(columns)


def _prior_from_scenario(
    scenario: Mapping[str, Any], bounds: Bounds
) -> NativeTruncatedPrior | LinkedNormalPrior:
    if scenario["prior"] == "truncated_normal":
        return NativeTruncatedPrior(
            bounds=bounds,
            location_base_mean=float(scenario["prior_hyper_location"]),
        )
    if scenario["prior"] == "linked_normal":
        return LinkedNormalPrior(
            bounds=bounds,
            location_base_mean_eta=float(scenario["prior_hyper_location"]),
        )
    raise RunnerError(f"unsupported prior family: {scenario['prior']}")


def build_cell_model(
    plan_entry: Mapping[str, Any],
    data_payload: Mapping[str, Any],
    *,
    ddm_network_path: Path = DEFAULT_DDM_NETWORK,
) -> BuiltCell:
    """Build the exact direct-PyMC, Bambi, or HSSM graph for one cell."""
    scenario = plan_entry["scenario"]
    layer = scenario["layer"]
    if layer in {"pymc", "bambi"}:
        data = _toy_data_from_payload(plan_entry, data_payload)
        prior = _prior_from_scenario(scenario, data.spec.bounds)
        geometry = (
            build_direct_pymc_model(prior, data)
            if layer == "pymc"
            else build_bambi_model(prior, data)
        )
        return BuiltCell(
            plan_entry=plan_entry,
            data_payload=data_payload,
            pymc_model=geometry.model,
            geometry=geometry,
        )
    if layer == "hssm":
        build = build_hssm_model(
            scenario,
            _hssm_data_from_payload(data_payload),
            initval_seed=int(plan_entry["initialization_seed"]),
            ddm_network_path=ddm_network_path,
        )
        return BuiltCell(
            plan_entry=plan_entry,
            data_payload=data_payload,
            pymc_model=build.model.pymc_model,
            hssm_build=build,
        )
    raise RunnerError(f"unsupported construction layer: {layer}")


def validate_runtime_contract(
    plan_entry: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[str, bool]:
    """Require the precision, device, and single-thread process contract."""
    validate_hssm_checkout()
    scenario = plan_entry["scenario"]
    observed_floatx = str(pytensor.config.floatX)
    observed_jax_x64 = bool(jax.config.x64_enabled)
    expected_floatx = str(scenario["floatx"])
    if observed_floatx != expected_floatx:
        raise RunnerError(
            f"PyTensor floatX is {observed_floatx}, expected {expected_floatx}"
        )
    if observed_jax_x64 != (expected_floatx == "float64"):
        raise RunnerError("JAX x64 setting does not match the planned precision")
    if jax.default_backend() != manifest["execution_policy"]["required_device"]:
        raise RunnerError("JAX backend does not match the required CPU device")
    for name, expected in manifest["execution_policy"]["thread_environment"].items():
        if os.environ.get(name) != expected:
            raise RunnerError(f"{name} must be set to {expected!r}")
    return observed_floatx, observed_jax_x64


def validate_hssm_checkout() -> Path:
    """Require the imported HSSM package to come from this exact checkout."""
    source_root = (REPO_ROOT / "src" / "hssm").resolve()
    module_file_raw = getattr(hssm, "__file__", None)
    if not isinstance(module_file_raw, str) or not module_file_raw:
        raise RunnerError("imported hssm package has no concrete source path")
    module_file = Path(module_file_raw).resolve()
    if source_root != module_file.parent and source_root not in module_file.parents:
        raise RunnerError(
            "imported hssm package is not from this checkout: "
            f"expected below {source_root}, found {module_file}"
        )
    return module_file


def materialize_exact_starts(built: BuiltCell) -> tuple[SamplerStartArtifact, bool]:
    """Materialize and validate the exact starts frozen for this cell."""
    entry = built.plan_entry
    scenario = entry["scenario"]
    sampler = cast("SamplerName", scenario["sampler"])
    chains = int(scenario["chains"])
    logps: Sequence[float]
    if scenario["initialization_policy"] == "hssm-default":
        if built.hssm_build is None:
            raise RunnerError("hssm-default starts require a built HSSM model")
        artifact = extract_actual_sampler_starts(
            built.hssm_build, sampler=sampler, chains=chains
        )
        logps = validate_actual_sampler_starts(built.hssm_build, artifact)
    elif scenario["initialization_policy"] == "backend-default":
        if len(entry["start_seeds"]) != chains:
            raise RunnerError("backend-default start seed count is inconsistent")
        functions = make_initial_point_fns_per_chain(
            model=built.pymc_model,
            overrides=None,
            jitter_rvs=set(built.pymc_model.free_RVs),
            chains=chains,
        )
        points = tuple(
            {name: np.asarray(value) for name, value in function(seed).items()}
            for function, seed in zip(functions, entry["start_seeds"], strict=True)
        )
        artifact = SamplerStartArtifact(
            sampler=sampler,
            initialization_seed=int(entry["initialization_seed"]),
            start_seeds=tuple(int(seed) for seed in entry["start_seeds"]),
            transformed_points=points,
        )
        mode = "JAX" if sampler == "numpyro" else None
        logp_fn = built.pymc_model.compile_logp(mode=mode)
        direct_logps: list[float] = []
        for point in artifact.transformed_points:
            built.pymc_model.check_start_vals(point, mode=mode)
            direct_logps.append(float(np.asarray(logp_fn(point))))
        logps = direct_logps
    else:  # pragma: no cover - manifest validation owns the enumeration
        raise RunnerError("unsupported initialization policy")
    finite = bool(np.isfinite(logps).all())
    if not finite:
        raise RunnerError("one or more exact starts has non-finite full logp")
    return artifact, finite


def sample_cell_model(built: BuiltCell, starts: SamplerStartArtifact) -> xr.DataTree:
    """Sample once with the exact frozen budget, starts, and seed contract."""
    entry = built.plan_entry
    scenario = entry["scenario"]
    common: dict[str, Any] = {
        "draws": int(scenario["draws"]),
        "tune": int(scenario["tune"]),
        "chains": int(scenario["chains"]),
        "target_accept": float(scenario["target_accept"]),
        "initvals": list(starts.transformed_points),
        "progressbar": False,
        "compute_convergence_checks": False,
        "model": built.pymc_model,
        "idata_kwargs": {"log_likelihood": False},
    }
    if scenario["sampler"] == "pymc":
        trace = pm.sample(
            **common,
            cores=1,
            random_seed=list(entry["chain_seeds"]),
            init="adapt_diag",
            return_inferencedata=True,
            discard_tuned_samples=True,
        )
    else:
        trace = sample_numpyro_nuts(
            **common,
            random_seed=int(entry["sampler_seed"]),
            jitter=False,
            # NumPyro accepts sequential even though PyMC's public annotation has
            # not yet caught up with the underlying MCMC API.
            chain_method="sequential",  # pyrefly: ignore[bad-argument-type]
        )
    if not isinstance(trace, xr.DataTree):
        raise RunnerError("sampler did not return an xarray DataTree")
    return trace


def _dataset(trace: xr.DataTree, group: str) -> xr.Dataset:
    if group not in trace:
        raise RunnerError(f"sampler trace lacks {group!r}")
    return trace[group].to_dataset()


def _chain_draw_array(variable: xr.DataArray, *, name: str) -> NDArray[Any]:
    if "chain" not in variable.dims or "draw" not in variable.dims:
        raise RunnerError(f"{name} lacks chain/draw dimensions")
    trailing = [
        dimension for dimension in variable.dims if dimension not in {"chain", "draw"}
    ]
    return np.asarray(variable.transpose("chain", "draw", *trailing))


def _raw_hierarchy_draws(
    built: BuiltCell, posterior: xr.Dataset
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    scenario = built.plan_entry["scenario"]
    linked = scenario["prior"] == "linked_normal"
    if built.geometry is not None:
        blocks = {block.canonical_name: block for block in built.geometry.blocks}
        location = _chain_draw_array(
            posterior[blocks["group_location"].random_variable_name],
            name="group_location",
        )
        scale = _chain_draw_array(
            posterior[blocks["group_scale"].random_variable_name],
            name="group_scale",
        )
        group = _chain_draw_array(
            posterior[blocks["group_effect"].random_variable_name],
            name="group_effect",
        )
    else:
        build = cast("HSSMBuild", built.hssm_build)
        location = _chain_draw_array(
            posterior[build.group_location_name], name="group_location"
        )
        scale = _chain_draw_array(posterior[build.group_scale_name], name="group_scale")
        group = _chain_draw_array(posterior[build.group_rv_name], name="group_effect")
    chains = int(scenario["chains"])
    draws = int(scenario["draws"])
    location = np.asarray(location).reshape(chains, draws, -1)
    scale = np.asarray(scale).reshape(chains, draws, -1)
    group = np.asarray(group).reshape(chains, draws, -1)
    if location.shape[-1] != 1 or scale.shape[-1] != 1:
        raise RunnerError("hierarchy hyperparameters must be scalar per draw")
    if group.shape[-1] != int(scenario["n_groups"]):
        raise RunnerError("group posterior size does not match n_groups")
    location = location[..., 0]
    scale = scale[..., 0]
    if linked:
        bounds = _bounds(scenario)
        location = support_inverse(location, bounds)
        group = support_inverse(group, bounds)
    return location, scale, group


def standardize_chain(built: BuiltCell, trace: xr.DataTree) -> xr.DataTree:
    """Reduce a backend trace to the exact monitored posterior/stat contract."""
    scenario = built.plan_entry["scenario"]
    posterior = _dataset(trace, "posterior")
    sample_stats = _dataset(trace, "sample_stats")
    location, scale, group = _raw_hierarchy_draws(built, posterior)
    chains, draws = location.shape
    if (chains, draws) != (int(scenario["chains"]), int(scenario["draws"])):
        raise RunnerError("retained posterior shape differs from the planned budget")
    group_indices = [int(index) for index in scenario["group_indices"]]
    standardized_posterior = xr.Dataset(
        {
            "group_location": (("chain", "draw"), location),
            "group_scale": (("chain", "draw"), scale),
            "group_first": (("chain", "draw"), group[..., group_indices[0]]),
            "group_middle": (("chain", "draw"), group[..., group_indices[1]]),
            "group_last": (("chain", "draw"), group[..., group_indices[2]]),
            "group_effect": (("chain", "draw", "group"), group),
        },
        coords={
            "chain": np.arange(chains),
            "draw": np.arange(draws),
            "group": np.arange(int(scenario["n_groups"])),
        },
    )
    standardized_stats: dict[str, tuple[tuple[str, str], NDArray[Any]]] = {}
    for target, aliases in _SAMPLE_STAT_ALIASES.items():
        source = next((name for name in aliases if name in sample_stats), None)
        if source is None:
            raise RunnerError(f"sample_stats lacks the frozen {target!r} statistic")
        values = _chain_draw_array(sample_stats[source], name=target)
        values = np.asarray(values).reshape(chains, draws, -1)
        if values.shape[-1] != 1:
            raise RunnerError(f"sample statistic {target!r} is not scalar per draw")
        standardized_stats[target] = (("chain", "draw"), values[..., 0])
    standardized = xr.DataTree.from_dict(
        {
            "posterior": standardized_posterior,
            "sample_stats": xr.Dataset(
                standardized_stats,
                coords={"chain": np.arange(chains), "draw": np.arange(draws)},
            ),
        }
    )
    validate_standardized_chain(standardized, scenario)
    return standardized


def validate_standardized_chain(
    chain: xr.DataTree, scenario: Mapping[str, Any]
) -> None:
    """Require exact groups, variables, dimensions, shapes, and finite evidence."""
    if set(chain.children) != {"posterior", "sample_stats"}:
        raise RunnerError("standardized chain has unexpected groups")
    posterior = _dataset(chain, "posterior")
    stats = _dataset(chain, "sample_stats")
    if set(posterior.data_vars) != set(_CHAIN_POSTERIOR):
        raise RunnerError("standardized posterior variables changed")
    if set(stats.data_vars) != set(_SAMPLE_STAT_ALIASES):
        raise RunnerError("standardized sample-stat variables changed")
    expected_shape = (int(scenario["chains"]), int(scenario["draws"]))
    expected_group_shape = (*expected_shape, int(scenario["n_groups"]))
    for name, variable in posterior.data_vars.items():
        expected_dims = (
            ("chain", "draw", "group") if name == "group_effect" else ("chain", "draw")
        )
        expected_variable_shape = (
            expected_group_shape if name == "group_effect" else expected_shape
        )
        if variable.dims != expected_dims or variable.shape != expected_variable_shape:
            raise RunnerError("standardized posterior violates its dimension contract")
        if not np.isfinite(np.asarray(variable)).all():
            raise RunnerError("standardized chain contains non-finite values")
    for variable in stats.data_vars.values():
        if variable.dims != ("chain", "draw") or variable.shape != expected_shape:
            raise RunnerError("standardized sample stat violates chain/draw contract")
        if not np.isfinite(np.asarray(variable)).all():
            raise RunnerError("standardized chain contains non-finite values")
    group_indices = [int(index) for index in scenario["group_indices"]]
    for name, index in zip(_MONITORED[2:], group_indices, strict=True):
        if not np.array_equal(
            np.asarray(posterior[name]),
            np.asarray(posterior["group_effect"].isel(group=index)),
        ):
            raise RunnerError(
                f"{name} is not the exact group_effect slice at group index {index}"
            )


def write_start_artifact(
    starts: SamplerStartArtifact,
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifact_root: Path,
) -> tuple[str, str]:
    """Publish exact transformed starts after the timed region."""
    relative = manifest["artifact_policy"]["start_path"].replace(
        "<cell_id>", str(plan_entry["cell_id"])
    )
    payload = _canonical_json_bytes(starts.as_jsonable())
    digest = _publish_bytes_once(artifact_root / PurePosixPath(relative), payload)
    if digest != starts.sha256():
        raise RunnerError("start artifact digest disagrees with canonical start object")
    return relative, digest


def load_start_artifact(
    built: BuiltCell,
    manifest: Mapping[str, Any],
    artifact_root: Path,
) -> tuple[SamplerStartArtifact, str, str]:
    """Load and rederive exact starts so diagnostics use the sampled point."""
    entry = built.plan_entry
    relative = manifest["artifact_policy"]["start_path"].replace(
        "<cell_id>", str(entry["cell_id"])
    )
    path = artifact_root / PurePosixPath(relative)
    if not path.is_file():
        raise RunnerError(f"start artifact is unavailable: {path}")
    expected, finite = materialize_exact_starts(built)
    if not finite or path.read_bytes() != _canonical_json_bytes(expected.as_jsonable()):
        raise RunnerError("start artifact differs from the rederived exact starts")
    return expected, relative, _file_sha256(path)


def write_chain_artifact(
    chain: xr.DataTree,
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifact_root: Path,
) -> tuple[str, str]:
    """Validate, serialize, reopen, and atomically publish the NetCDF chain."""
    scenario = plan_entry["scenario"]
    validate_standardized_chain(chain, scenario)
    relative = manifest["artifact_policy"]["chain_path"].replace(
        "<cell_id>", str(plan_entry["cell_id"])
    )
    destination = artifact_root / PurePosixPath(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".nc"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        chain.to_netcdf(temporary)
        reopened = xr.open_datatree(temporary)
        try:
            reopened.load()
            validate_standardized_chain(reopened, scenario)
        finally:
            reopened.close()
        return relative, _publish_file_once(destination, temporary)
    finally:
        temporary.unlink(missing_ok=True)


def load_chain_artifact(
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifact_root: Path,
) -> tuple[xr.DataTree, str, str]:
    """Load the completed standardized chain and detach it from the NetCDF file."""
    relative = manifest["artifact_policy"]["chain_path"].replace(
        "<cell_id>", str(plan_entry["cell_id"])
    )
    path = artifact_root / PurePosixPath(relative)
    if not path.is_file():
        raise RunnerError(f"chain artifact is unavailable: {path}")
    chain = xr.open_datatree(path)
    try:
        chain.load()
        validate_standardized_chain(chain, plan_entry["scenario"])
        detached = chain.copy(deep=True)
    finally:
        chain.close()
    return detached, relative, _file_sha256(path)


def _finite_values(dataset: xr.Dataset) -> NDArray[np.float64]:
    pieces = [
        np.asarray(variable).reshape(-1) for variable in dataset.data_vars.values()
    ]
    if not pieces:
        raise RunnerError("diagnostic dataset is empty")
    values = np.concatenate(pieces).astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise RunnerError("posterior diagnostic is non-finite")
    return values


def compute_sampler_metrics(
    chain: xr.DataTree, *, sampling_elapsed_seconds: float
) -> dict[str, bool | int | float]:
    """Compute all raw per-cell diagnostics and efficiency primitives."""
    if not math.isfinite(sampling_elapsed_seconds) or sampling_elapsed_seconds <= 0:
        raise RunnerError("sampling elapsed time must be positive and finite")
    posterior = _dataset(chain, "posterior")
    stats = _dataset(chain, "sample_stats")
    hyper = posterior[["group_location", "group_scale"]]
    groups = posterior[["group_effect"]]
    hyper_rhat = _finite_values(az.rhat(hyper, method="rank"))
    hyper_bulk = _finite_values(az.ess(hyper, method="bulk"))
    hyper_tail = _finite_values(az.ess(hyper, method="tail"))
    hyper_mcse = _finite_values(az.mcse(hyper, method="mean"))
    group_rhat = _finite_values(az.rhat(groups, method="rank"))
    group_bulk = _finite_values(az.ess(groups, method="bulk"))
    group_tail = _finite_values(az.ess(groups, method="tail"))
    hyper_sd = np.asarray(
        [
            np.std(np.asarray(hyper[name]).reshape(-1), ddof=1)
            for name in ("group_location", "group_scale")
        ]
    )
    if not np.isfinite(hyper_sd).all() or np.any(hyper_sd <= 0):
        raise RunnerError("hyperparameter posterior SD is invalid")
    energy = np.asarray(stats["energy"])
    bfmi_values = np.mean(np.diff(energy, axis=1) ** 2, axis=1) / np.var(
        energy, axis=1, ddof=1
    )
    if not np.isfinite(bfmi_values).all():
        raise RunnerError("BFMI is non-finite")
    divergences = np.asarray(stats["diverging"], dtype=bool)
    draw_count = int(divergences.size)
    divergence_count = int(np.count_nonzero(divergences))
    n_steps = np.asarray(stats["n_steps"], dtype=np.int64)
    if np.any(n_steps <= 0):
        raise RunnerError("n_steps must be positive")
    leapfrog_steps = int(np.sum(n_steps))
    tree_depth = np.asarray(stats["tree_depth"])
    step_size = np.asarray(stats["step_size"], dtype=np.float64)
    hyper_ess_per_second = hyper_bulk / sampling_elapsed_seconds
    leapfrog_per_ess = leapfrog_steps / hyper_bulk
    return {
        "sampling_success": True,
        "divergence_count": divergence_count,
        "posterior_draw_count": draw_count,
        "divergence_rate": divergence_count / draw_count,
        "hyper_rhat_max": float(np.max(hyper_rhat)),
        "hyper_ess_bulk_min": float(np.min(hyper_bulk)),
        "hyper_ess_tail_min": float(np.min(hyper_tail)),
        "bfmi_min": float(np.min(bfmi_values)),
        "treedepth_saturation_rate": float(np.mean(tree_depth >= _MAX_TREE_DEPTH)),
        "hyper_mcse_over_sd_max": float(np.max(hyper_mcse / hyper_sd)),
        "group_rhat_max": float(np.max(group_rhat)),
        "group_ess_bulk_fraction_ge_400": float(np.mean(group_bulk >= 400)),
        "group_ess_tail_fraction_ge_400": float(np.mean(group_tail >= 400)),
        "sampling_elapsed_seconds": float(sampling_elapsed_seconds),
        "step_size_median": float(np.median(step_size)),
        # One retained NUTS leapfrog step entails one gradient evaluation in the
        # frozen cross-backend cost proxy. Warmup cost remains in elapsed time.
        "gradient_evaluation_count": leapfrog_steps,
        "leapfrog_step_count": leapfrog_steps,
        "hyper_ess_per_second_median": float(np.median(hyper_ess_per_second)),
        "hyper_leapfrog_steps_per_effective_sample_median": float(
            np.median(leapfrog_per_ess)
        ),
    }


def _gradient_tolerances(
    scenario: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    contract = manifest["analysis_policy"]["gradient_contract"]
    precision = str(scenario["floatx"])
    finite_key = (
        "float32_narrow_stress"
        if precision == "float32"
        and scenario["bound_kind"] == "narrow"
        and scenario["tier"] == "stress"
        else precision
    )
    return (
        contract["finite_difference"][finite_key],
        contract["pytensor_jax"][precision],
    )


def _contract_applies(
    plan_entry: Mapping[str, Any], manifest: Mapping[str, Any]
) -> bool:
    scenario = plan_entry["scenario"]
    evaluation = manifest["analysis_policy"]["gradient_contract"]["evaluation"]
    return bool(
        scenario["tier"] in evaluation["tiers"]
        and plan_entry["replicate"] == evaluation["scenario_replicate"]
        and (
            scenario.get("posterior_pair_id") is None
            or scenario["sampler"] == evaluation["posterior_pair_owner_sampler"]
        )
    )


def _hssm_lba_start_b(built: BuiltCell, starts: SamplerStartArtifact) -> NDArray[Any]:
    build = cast("HSSMBuild", built.hssm_build)
    group_rv = build.model.pymc_model.named_vars[build.group_rv_name]
    constrained = build.model.pymc_model.compile_fn(group_rv)(
        starts.transformed_points[0]
    )
    group_values = np.asarray(constrained).reshape(-1)
    labels = {
        label: index
        for index, label in enumerate(sorted(build.data["participant_id"].unique()))
    }
    group_index = np.asarray([labels[value] for value in build.data["participant_id"]])
    return group_values[group_index]


def run_diagnostics(
    built: BuiltCell,
    starts: SamplerStartArtifact,
    manifest: Mapping[str, Any],
) -> dict[str, bool | float]:
    """Run post-sampling geometry/parity probes without touching a sampler."""
    entry = built.plan_entry
    scenario = entry["scenario"]
    finite_tolerance, jax_tolerance = _gradient_tolerances(scenario, manifest)
    applicable = _contract_applies(entry, manifest)
    metrics: dict[str, bool | float]
    if built.geometry is not None:
        vector = built.geometry.pack_point(starts.transformed_points[0])
        geometry = evaluate_transformed_geometry(built.geometry, vector)
        metrics = {
            "compile_success": True,
            "initialization_success": True,
            "logp_finite": geometry.all_finite,
            "gradient_finite": geometry.all_finite,
        }
        if applicable:
            metrics.update(
                geometry.qualification_metrics(
                    finite_difference_absolute_tolerance=float(
                        finite_tolerance["absolute_tolerance"]
                    ),
                    finite_difference_relative_tolerance=float(
                        finite_tolerance["relative_tolerance"]
                    ),
                    pytensor_jax_absolute_tolerance=float(
                        jax_tolerance["absolute_tolerance"]
                    ),
                    pytensor_jax_relative_tolerance=float(
                        jax_tolerance["relative_tolerance"]
                    ),
                )
            )
            if scenario["layer"] == "bambi":
                data = _toy_data_from_payload(entry, built.data_payload)
                prior = _prior_from_scenario(scenario, data.spec.bounds)
                direct = build_direct_pymc_model(prior, data)
                comparison = compare_isomorphic_models(direct, built.geometry, vector)
                tolerance = manifest["analysis_policy"]["gradient_contract"][
                    "bambi_isomorphism"
                ][scenario["floatx"]]
                metrics.update(
                    comparison.qualification_metrics(
                        absolute_tolerance=float(tolerance["absolute_tolerance"]),
                        relative_tolerance=float(tolerance["relative_tolerance"]),
                    )
                )
    else:
        build = cast("HSSMBuild", built.hssm_build)
        diagnostics = evaluate_hssm_gradients(build, starts.transformed_points[0])
        metrics = {
            "initialization_success": True,
            **diagnostics.qualification_metrics(
                finite_difference_absolute_tolerance=float(
                    finite_tolerance["absolute_tolerance"]
                ),
                finite_difference_relative_tolerance=float(
                    finite_tolerance["relative_tolerance"]
                ),
                pytensor_jax_absolute_tolerance=float(
                    jax_tolerance["absolute_tolerance"]
                ),
                pytensor_jax_relative_tolerance=float(
                    jax_tolerance["relative_tolerance"]
                ),
            ),
        }
        if not applicable:
            for name in tuple(metrics):
                if name.startswith(("finite_difference_", "pytensor_jax_")):
                    metrics.pop(name)
        if applicable and scenario["model"] == "lba2_b":
            parity = lba2_pytensor_jax_parity(
                build.data,
                b=_hssm_lba_start_b(built, starts),
                floatx=build.floatx,
            )
            tolerance = manifest["analysis_policy"]["gradient_contract"][
                "likelihood_pytensor_jax"
            ][scenario["floatx"]]
            metrics.update(
                parity.qualification_metrics(
                    value_absolute_tolerance=float(
                        tolerance["value_absolute_tolerance"]
                    ),
                    value_relative_tolerance=float(
                        tolerance["value_relative_tolerance"]
                    ),
                    gradient_absolute_tolerance=float(
                        tolerance["gradient_absolute_tolerance"]
                    ),
                    gradient_relative_tolerance=float(
                        tolerance["gradient_relative_tolerance"]
                    ),
                )
            )
            metrics["logp_finite"] = bool(metrics["logp_finite"] and parity.all_finite)
            metrics["gradient_finite"] = bool(
                metrics["gradient_finite"] and parity.all_finite
            )
    if not bool(metrics["logp_finite"]) or not bool(metrics["gradient_finite"]):
        raise RunnerError("post-sampling numerical diagnostics were non-finite")
    return metrics


def _sbc_draw_indices(
    *, seed: int, parameter_id: str, available: int, count: int
) -> NDArray[np.int64]:
    if count > available:
        raise RunnerError("SBC draw count exceeds retained posterior draws")
    scores = []
    for source_index in range(available):
        payload = _canonical_json_bytes(
            {
                "contract": "hssm-truncated-hierarchy-sbc-draw-v1",
                "parameter_id": parameter_id,
                "seed": seed,
                "source_index": source_index,
            }
        )
        scores.append((hashlib.sha256(payload).digest(), source_index))
    selected = sorted(scores)[:count]
    return np.asarray([index for _, index in selected], dtype=np.int64)


def compute_parameter_summaries(
    chain: xr.DataTree,
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    data_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compute fixed-recovery or SBC summaries from all retained draws."""
    scenario = plan_entry["scenario"]
    if not scenario["recovery"] or scenario["purpose"] not in {"candidate", "control"}:
        return []
    posterior = _dataset(chain, "posterior")
    truth = cast("Mapping[str, Any]", data_payload["truth"])
    group_truth = cast("Sequence[float]", truth["group_effect"])
    group_indices = [int(index) for index in scenario["group_indices"]]
    truths = {
        "group_location": float(truth["group_location"]),
        "group_scale": float(truth["group_scale"]),
        "group_first": float(group_truth[group_indices[0]]),
        "group_middle": float(group_truth[group_indices[1]]),
        "group_last": float(group_truth[group_indices[2]]),
    }
    mcse_dataset = az.mcse(posterior[list(_MONITORED)], method="mean")
    summaries: list[dict[str, Any]] = []
    calibration = scenario.get("calibration_kind") == "sbc"
    rank_count = int(manifest["analysis_policy"]["sbc_rank_draw_count"])
    for parameter_id in manifest["analysis_policy"]["monitored_parameters"]:
        draws = np.asarray(posterior[parameter_id]).reshape(-1)
        quantiles = np.quantile(draws, [0.025, 0.05, 0.5, 0.95, 0.975], method="linear")
        summary: dict[str, Any] = {
            "family": scenario["purpose"],
            "scenario_id": plan_entry["scenario_id"],
            "parameter_id": parameter_id,
            "replicate": plan_entry["replicate"],
            "truth": truths[parameter_id],
            "posterior_mean": float(np.mean(draws)),
            "posterior_sd": float(np.std(draws, ddof=1)),
            "posterior_mcse": float(np.asarray(mcse_dataset[parameter_id])),
            "q025": float(quantiles[0]),
            "q05": float(quantiles[1]),
            "q50": float(quantiles[2]),
            "q95": float(quantiles[3]),
            "q975": float(quantiles[4]),
        }
        if calibration:
            selected = draws[
                _sbc_draw_indices(
                    seed=int(plan_entry["sbc_draw_seed"]),
                    parameter_id=parameter_id,
                    available=draws.size,
                    count=rank_count,
                )
            ]
            rank_less = int(np.count_nonzero(selected < truths[parameter_id]))
            rank_equal = int(np.count_nonzero(selected == truths[parameter_id]))
            tie_index = derive_sbc_rank_tie_index(
                tie_seed=int(plan_entry["sbc_tie_seed"]),
                family=cast("Literal['candidate', 'control']", scenario["purpose"]),
                scenario_id=str(plan_entry["scenario_id"]),
                parameter_id=parameter_id,
                replicate=int(plan_entry["replicate"]),
                rank_less=rank_less,
                rank_equal=rank_equal,
                rank_draw_count=rank_count,
            )
            summary.update(
                {
                    "rank_less": rank_less,
                    "rank_equal": rank_equal,
                    "rank_tie_index": tie_index,
                    "rank": rank_less + tie_index,
                    "rank_draw_count": rank_count,
                }
            )
        summaries.append(summary)
    return summaries


def _stable_environment_contract(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Select profile facts that must be identical across fresh hosted workers."""
    runtime = environment["runtime"]
    return {
        "schema_version": environment["schema_version"],
        "study_id": environment["study_id"],
        "manifest_sha256": environment["manifest_sha256"],
        "runner_version": environment["runner_version"],
        "dependency_profile": environment["dependency_profile"],
        "git": {
            "commit": environment["git"]["commit"],
            "dirty": environment["git"]["dirty"],
        },
        "project": dict(environment["project"]),
        "runtime": {
            "python": ".".join(str(runtime["python"]).split(".")[:2]),
            "implementation": runtime["implementation"],
        },
        "packages": dict(environment["packages"]),
    }


def _environment_for_cell(
    plan_entry: Mapping[str, Any],
    environment_catalog: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    profile = (
        plan_entry["scenario"].get("dependency_profile") or DEFAULT_DEPENDENCY_PROFILE
    )
    matches = [
        environment
        for environment in environment_catalog.values()
        if environment["dependency_profile"] == profile
    ]
    if len(matches) != 1:
        raise RunnerError(
            f"environment catalog does not select profile {profile!r} once"
        )
    supplied = matches[0]
    observed = collect_environment(manifest, profile)
    # GitHub-hosted jobs are fresh VMs. A runner-image rollout can legitimately
    # change the kernel string or Python patch release between the profile-attestation
    # job and a sampling shard. Bind the worker to the frozen source, lock/project,
    # exact package set, interpreter family, and Python minor version; validate and
    # record cell precision separately. Exact platform strings remain descriptive in
    # the supplied sidecar and are not a cross-job identity requirement.
    if _stable_environment_contract(observed) != _stable_environment_contract(supplied):
        raise RunnerError(
            "fresh worker environment does not match the stable profile contract"
        )
    return supplied


def _provenance(
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    environment: Mapping[str, Any],
    artifacts: ArtifactState,
    execution_identity: ExecutionIdentity,
    *,
    pytensor_floatx: str | None,
    jax_enable_x64: bool | None,
    numpyro_started: bool,
) -> dict[str, Any]:
    scenario = plan_entry["scenario"]
    effective_keys = None
    if scenario["sampler"] == "numpyro" and numpyro_started:
        effective_keys = [
            list(key)
            for key in derive_numpyro_chain_keys(
                int(plan_entry["sampler_seed"]), int(scenario["chains"])
            )
        ]
    return {
        "runner_version": RUNNER_VERSION,
        "sampler": scenario["sampler"],
        "device": "cpu",
        "floatx": scenario["floatx"],
        "pytensor_floatx": pytensor_floatx,
        "jax_enable_x64": jax_enable_x64,
        "data_artifact": artifacts.data_artifact,
        "data_sha256": artifacts.data_sha256,
        "effective_numpyro_chain_keys": effective_keys,
        "actual_start_artifact": artifacts.actual_start_artifact,
        "actual_start_sha256": artifacts.actual_start_sha256,
        "raw_chain_artifact": artifacts.raw_chain_artifact,
        "raw_chain_sha256": artifacts.raw_chain_sha256,
        "git_commit": environment["git"]["commit"],
        "environment_sha256": environment_sha256(environment, manifest),
        "execution_attempt_id": execution_identity.execution_attempt_id,
        "pair_execution_id": execution_identity.pair_execution_id,
        "pair_position": execution_identity.pair_position,
        "worker_identity_sha256": execution_identity.worker_identity_sha256,
    }


def _result_record(
    plan_entry: Mapping[str, Any],
    *,
    status: Literal["completed", "failed"],
    metrics: Mapping[str, Any],
    parameter_summaries: Sequence[Mapping[str, Any]],
    failure: Mapping[str, str] | None,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **{field: plan_entry[field] for field in _IDENTITY_FIELDS},
        "execution_status": status,
        "metrics": dict(metrics),
        "unavailable_metrics": {},
        "parameter_summaries": [dict(summary) for summary in parameter_summaries],
        "failure": None if failure is None else dict(failure),
        "provenance": dict(provenance),
    }


def _cells_dir(manifest: Mapping[str, Any], artifact_root: Path) -> Path:
    return (
        artifact_root / PurePosixPath(manifest["artifact_policy"]["cell_path"]).parts[0]
    )


def _publish_failed_result(
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    environment_catalog: Mapping[str, Mapping[str, Any]],
    artifact_root: Path,
    *,
    artifacts: ArtifactState,
    execution_identity: ExecutionIdentity,
    metrics: Mapping[str, Any],
    stage: str,
    error: BaseException,
    pytensor_floatx: str | None,
    jax_enable_x64: bool | None,
    numpyro_started: bool,
) -> Mapping[str, Any]:
    """Publish a failed cell as the final marker after preserving prior artifacts."""
    environment = _environment_for_cell(plan_entry, environment_catalog, manifest)
    failed_metrics = dict(metrics)
    failed_metrics.setdefault("sampling_success", False)
    record = _result_record(
        plan_entry,
        status="failed",
        metrics=failed_metrics,
        parameter_summaries=[],
        failure={
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error) or repr(error),
        },
        provenance=_provenance(
            plan_entry,
            manifest,
            environment,
            artifacts,
            execution_identity,
            pytensor_floatx=pytensor_floatx,
            jax_enable_x64=jax_enable_x64,
            numpyro_started=numpyro_started,
        ),
    )
    cells_dir = _cells_dir(manifest, artifact_root)
    validate_result_record(record, plan_entry, environment_catalog, manifest)
    write_cell_result(record, plan_entry, cells_dir, environment_catalog, manifest)
    return record


def run_sample_phase(
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    environment_catalog: Mapping[str, Mapping[str, Any]],
    artifact_root: Path,
    *,
    phase_context: PhaseContext,
) -> Mapping[str, Any]:
    """Run only the timed sampling phase and publish data/start/chain artifacts."""
    _validate_phase_context(phase_context, plan_entry, artifact_root, "sample")
    execution_identity = phase_context.identity
    _validate_execution_identity(plan_entry, execution_identity)
    _environment_for_cell(plan_entry, environment_catalog, manifest)
    cells_dir = _cells_dir(manifest, artifact_root)
    result_path = cells_dir / f"{plan_entry['cell_id']}.json"
    if result_path.exists():
        raise RunnerError(f"cell completion marker already exists: {result_path}")
    artifacts = ArtifactState()
    metrics: dict[str, Any] = {}
    observed_floatx: str | None = None
    observed_jax_x64: bool | None = None
    numpyro_started = False
    stage = "runtime-contract"
    starts: SamplerStartArtifact | None = None
    try:
        observed_floatx, observed_jax_x64 = validate_runtime_contract(
            plan_entry, manifest
        )
        stage = "data"
        payload, artifacts.data_artifact, artifacts.data_sha256 = (
            materialize_data_artifact(plan_entry, manifest, artifact_root)
        )
        stage = "build"
        built = build_cell_model(plan_entry, payload)

        started = time.perf_counter()
        try:
            stage = "initialization"
            starts, logp_finite = materialize_exact_starts(built)
            metrics["initialization_success"] = True
            metrics["logp_finite"] = logp_finite
            stage = "sampling"
            numpyro_started = plan_entry["scenario"]["sampler"] == "numpyro"
            raw_trace = sample_cell_model(built, starts)
        finally:
            elapsed = max(time.perf_counter() - started, sys.float_info.min)

        stage = "start-artifact"
        artifacts.actual_start_artifact, artifacts.actual_start_sha256 = (
            write_start_artifact(starts, plan_entry, manifest, artifact_root)
        )
        stage = "chain-standardization"
        chain = standardize_chain(built, raw_trace)
        chain.attrs["sampling_elapsed_seconds"] = elapsed
        # Validate all raw sampler diagnostics before publishing the trace. The
        # fresh finalize process recomputes them from these exact bytes.
        metrics.update(compute_sampler_metrics(chain, sampling_elapsed_seconds=elapsed))
        stage = "chain-artifact"
        artifacts.raw_chain_artifact, artifacts.raw_chain_sha256 = write_chain_artifact(
            chain, plan_entry, manifest, artifact_root
        )
        return {
            "phase_status": "sampled",
            "cell_id": plan_entry["cell_id"],
            "data_artifact": artifacts.data_artifact,
            "actual_start_artifact": artifacts.actual_start_artifact,
            "raw_chain_artifact": artifacts.raw_chain_artifact,
        }
    except Exception as error:
        failure_error = error
        if starts is not None and artifacts.actual_start_artifact is None:
            try:
                artifacts.actual_start_artifact, artifacts.actual_start_sha256 = (
                    write_start_artifact(starts, plan_entry, manifest, artifact_root)
                )
            except Exception as artifact_error:
                stage = "start-artifact"
                failure_error = artifact_error
        return _publish_failed_result(
            plan_entry,
            manifest,
            environment_catalog,
            artifact_root,
            artifacts=artifacts,
            execution_identity=execution_identity,
            metrics=metrics,
            stage=stage,
            error=failure_error,
            pytensor_floatx=observed_floatx,
            jax_enable_x64=observed_jax_x64,
            numpyro_started=numpyro_started,
        )


def finalize_cell(
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    environment_catalog: Mapping[str, Mapping[str, Any]],
    artifact_root: Path,
    *,
    phase_context: PhaseContext,
    diagnostics_fn: Callable[
        [BuiltCell, SamplerStartArtifact, Mapping[str, Any]],
        Mapping[str, bool | float],
    ] = run_diagnostics,
) -> Mapping[str, Any]:
    """In a fresh process, diagnose sampled bytes and publish the final marker."""
    _validate_phase_context(phase_context, plan_entry, artifact_root, "finalize")
    execution_identity = phase_context.identity
    _validate_execution_identity(plan_entry, execution_identity)
    environment = _environment_for_cell(plan_entry, environment_catalog, manifest)
    cells_dir = _cells_dir(manifest, artifact_root)
    result_path = cells_dir / f"{plan_entry['cell_id']}.json"
    if result_path.exists():
        raise RunnerError(f"cell completion marker already exists: {result_path}")
    artifacts = ArtifactState()
    metrics: dict[str, Any] = {}
    observed_floatx: str | None = None
    observed_jax_x64: bool | None = None
    stage = "runtime-contract"
    try:
        observed_floatx, observed_jax_x64 = validate_runtime_contract(
            plan_entry, manifest
        )
        stage = "data-reload"
        payload, artifacts.data_artifact, artifacts.data_sha256 = load_data_artifact(
            plan_entry, manifest, artifact_root
        )
        stage = "diagnostic-build"
        built = build_cell_model(plan_entry, payload)
        stage = "start-reload"
        starts, artifacts.actual_start_artifact, artifacts.actual_start_sha256 = (
            load_start_artifact(built, manifest, artifact_root)
        )
        stage = "chain-reload"
        chain, artifacts.raw_chain_artifact, artifacts.raw_chain_sha256 = (
            load_chain_artifact(plan_entry, manifest, artifact_root)
        )
        elapsed = chain.attrs.get("sampling_elapsed_seconds")
        if isinstance(elapsed, bool) or not isinstance(elapsed, int | float):
            raise RunnerError("chain lacks numeric sampling_elapsed_seconds metadata")
        metrics.update(
            compute_sampler_metrics(chain, sampling_elapsed_seconds=float(elapsed))
        )
        summaries = compute_parameter_summaries(chain, plan_entry, manifest, payload)
        stage = "diagnostics"
        metrics.update(diagnostics_fn(built, starts, manifest))
        record = _result_record(
            plan_entry,
            status="completed",
            metrics=metrics,
            parameter_summaries=summaries,
            failure=None,
            provenance=_provenance(
                plan_entry,
                manifest,
                environment,
                artifacts,
                execution_identity,
                pytensor_floatx=observed_floatx,
                jax_enable_x64=observed_jax_x64,
                numpyro_started=plan_entry["scenario"]["sampler"] == "numpyro",
            ),
        )
        validate_result_record(record, plan_entry, environment_catalog, manifest)
        write_cell_result(record, plan_entry, cells_dir, environment_catalog, manifest)
        return record
    except Exception as error:
        return _publish_failed_result(
            plan_entry,
            manifest,
            environment_catalog,
            artifact_root,
            artifacts=artifacts,
            execution_identity=execution_identity,
            metrics=metrics,
            stage=stage,
            error=error,
            pytensor_floatx=observed_floatx,
            jax_enable_x64=observed_jax_x64,
            numpyro_started=False,
        )


def _load_existing_result(
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    environment_catalog: Mapping[str, Mapping[str, Any]],
    artifact_root: Path,
    *,
    execution_identity: ExecutionIdentity | None = None,
) -> Mapping[str, Any] | None:
    path = _cells_dir(manifest, artifact_root) / f"{plan_entry['cell_id']}.json"
    if not path.is_file():
        return None
    record = strict_json_loads(path.read_text(encoding="utf-8"), source=str(path))
    validate_result_record(record, plan_entry, environment_catalog, manifest)
    verify_result_artifacts(cast("Mapping[str, Any]", record), artifact_root)
    if execution_identity is not None:
        provenance = cast("Mapping[str, Any]", record)["provenance"]
        expected = {
            "execution_attempt_id": execution_identity.execution_attempt_id,
            "pair_execution_id": execution_identity.pair_execution_id,
            "pair_position": execution_identity.pair_position,
            "worker_identity_sha256": execution_identity.worker_identity_sha256,
        }
        if any(provenance[field] != value for field, value in expected.items()):
            raise RunnerError(
                "existing result belongs to a different execution attempt"
            )
    return cast("Mapping[str, Any]", record)


def _discover_artifacts(
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifact_root: Path,
) -> ArtifactState:
    """Discover only independently valid artifacts after an unhandled child exit."""
    artifacts = ArtifactState()
    try:
        _, artifacts.data_artifact, artifacts.data_sha256 = load_data_artifact(
            plan_entry, manifest, artifact_root
        )
    except Exception:
        pass
    start_relative = manifest["artifact_policy"]["start_path"].replace(
        "<cell_id>", str(plan_entry["cell_id"])
    )
    start_path = artifact_root / PurePosixPath(start_relative)
    if start_path.is_file():
        try:
            start = strict_json_loads(
                start_path.read_text(encoding="utf-8"), source=str(start_path)
            )
            if (
                isinstance(start, dict)
                and set(start)
                == {
                    "schema_version",
                    "coordinate_system",
                    "sampler",
                    "initialization_seed",
                    "start_seeds",
                    "chains",
                }
                and start.get("schema_version") == 1
                and start.get("coordinate_system") == "pymc-transformed-value-variables"
                and start.get("sampler") == plan_entry["scenario"]["sampler"]
                and start.get("initialization_seed")
                == plan_entry["initialization_seed"]
                and start.get("start_seeds") == plan_entry["start_seeds"]
                and isinstance(start.get("chains"), list)
                and len(start["chains"]) == plan_entry["scenario"]["chains"]
            ):
                artifacts.actual_start_artifact = start_relative
                artifacts.actual_start_sha256 = _file_sha256(start_path)
        except Exception:
            pass
    try:
        chain, artifacts.raw_chain_artifact, artifacts.raw_chain_sha256 = (
            load_chain_artifact(plan_entry, manifest, artifact_root)
        )
        chain.close()
    except Exception:
        pass
    return artifacts


def synthesize_child_failure(
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    environment_catalog: Mapping[str, Mapping[str, Any]],
    artifact_root: Path,
    *,
    execution_identity: ExecutionIdentity,
    stage: str,
    error: BaseException,
) -> Mapping[str, Any]:
    """Publish a deterministic failed marker after a crash or timeout."""
    existing = _load_existing_result(
        plan_entry,
        manifest,
        environment_catalog,
        artifact_root,
        execution_identity=execution_identity,
    )
    if existing is not None:
        return existing
    return _publish_failed_result(
        plan_entry,
        manifest,
        environment_catalog,
        artifact_root,
        artifacts=_discover_artifacts(plan_entry, manifest, artifact_root),
        execution_identity=execution_identity,
        metrics={},
        stage=stage,
        error=error,
        pytensor_floatx=None,
        jax_enable_x64=None,
        numpyro_started=False,
    )


def _phase_context(
    plan_entry: Mapping[str, Any],
    artifact_root: Path,
    phase: PhaseName,
    execution_identity: ExecutionIdentity,
) -> PhaseContext:
    """Build the exact attempt-local cache identity for one child phase."""
    cache_root = (
        artifact_root.resolve()
        / ".cache"
        / execution_identity.execution_attempt_id
        / str(plan_entry["cell_id"])
        / phase
    )
    return PhaseContext(
        phase=phase,
        cell_id=str(plan_entry["cell_id"]),
        identity=execution_identity,
        pytensor_cache=cache_root / "pytensor",
        jax_cache=cache_root / "jax",
        matplotlib_cache=cache_root / "matplotlib",
        xdg_cache=cache_root / "xdg",
    )


def _validate_phase_context(
    context: PhaseContext,
    plan_entry: Mapping[str, Any],
    artifact_root: Path,
    phase: PhaseName,
) -> None:
    """Bind a child to its exact parent-minted phase, cell, attempt, and caches."""
    _validate_execution_identity(plan_entry, context.identity)
    expected = _phase_context(plan_entry, artifact_root, phase, context.identity)
    if context != expected:
        raise RunnerError(
            "orchestrator phase context does not match the requested phase and cell"
        )
    expected_environment = {
        "PYTENSOR_FLAGS": (
            f"floatX={plan_entry['scenario']['floatx']},"
            f"base_compiledir={context.pytensor_cache}"
        ),
        "JAX_ENABLE_X64": (
            "true" if plan_entry["scenario"]["floatx"] == "float64" else "false"
        ),
        "JAX_PLATFORMS": "cpu",
        "JAX_COMPILATION_CACHE_DIR": str(context.jax_cache),
        "MPLCONFIGDIR": str(context.matplotlib_cache),
        "XDG_CACHE_HOME": str(context.xdg_cache),
    }
    for name, expected_value in expected_environment.items():
        if os.environ.get(name) != expected_value:
            raise RunnerError(
                f"orchestrator phase context does not match process {name}"
            )
    for path in (
        context.pytensor_cache,
        context.jax_cache,
        context.matplotlib_cache,
        context.xdg_cache,
    ):
        if not path.is_dir():
            raise RunnerError(f"orchestrator phase cache is unavailable: {path}")


def _load_phase_context(
    plan_entry: Mapping[str, Any], artifact_root: Path, phase: PhaseName
) -> PhaseContext:
    """Load and verify the unexposed context required by a child-only CLI phase."""
    raw = os.environ.get(_PHASE_CONTEXT_ENV)
    if raw is None:
        raise RunnerError(f"{phase} is an orchestrator-only phase; use run or run-pair")
    parsed = strict_json_loads(raw, source=_PHASE_CONTEXT_ENV)
    context = PhaseContext.from_jsonable(parsed)
    _validate_phase_context(context, plan_entry, artifact_root, phase)
    return context


def _child_environment(
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifact_root: Path,
    phase: PhaseName,
    execution_identity: ExecutionIdentity,
) -> dict[str, str]:
    """Create fresh caches and the exact private context for one child launch."""
    _validate_execution_identity(plan_entry, execution_identity)
    context = _phase_context(plan_entry, artifact_root, phase, execution_identity)
    cache_root = context.pytensor_cache.parent
    try:
        cache_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise RunnerError(
            f"refusing to reuse a phase compilation cache: {cache_root}"
        ) from None
    for path in (
        context.pytensor_cache,
        context.jax_cache,
        context.matplotlib_cache,
        context.xdg_cache,
    ):
        path.mkdir()
    environment = dict(os.environ)
    floatx = str(plan_entry["scenario"]["floatx"])
    environment.update(manifest["execution_policy"]["thread_environment"])
    environment.update(
        {
            "PYTENSOR_FLAGS": (
                f"floatX={floatx},base_compiledir={context.pytensor_cache}"
            ),
            "JAX_ENABLE_X64": "true" if floatx == "float64" else "false",
            "JAX_PLATFORMS": "cpu",
            "JAX_COMPILATION_CACHE_DIR": str(context.jax_cache),
            "MPLCONFIGDIR": str(context.matplotlib_cache),
            "XDG_CACHE_HOME": str(context.xdg_cache),
            _PHASE_CONTEXT_ENV: json.dumps(
                context.as_jsonable(),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
    )
    return environment


def _orchestrate_cell(
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    environment_paths: Sequence[Path],
    environment_catalog: Mapping[str, Mapping[str, Any]],
    artifact_root: Path,
    *,
    execution_identity: ExecutionIdentity,
    timeout_seconds: float | None = None,
) -> Mapping[str, Any]:
    """Run sample and finalize in separate fresh children and contain crashes."""
    _validate_execution_identity(plan_entry, execution_identity)
    common = [
        "--manifest",
        str(manifest_path),
        "--tier",
        str(plan_entry["scenario"]["tier"]),
        "--cell-id",
        str(plan_entry["cell_id"]),
        "--artifact-root",
        str(artifact_root),
    ]
    for path in environment_paths:
        common.extend(("--environment", str(path)))
    for phase in _PHASES:
        command = [
            sys.executable,
            "-m",
            "scripts.truncated_hierarchy_runner",
            phase,
            *common,
        ]
        try:
            child_environment = _child_environment(
                plan_entry,
                manifest,
                artifact_root,
                phase,
                execution_identity,
            )
        except Exception as error:
            return synthesize_child_failure(
                plan_entry,
                manifest,
                environment_catalog,
                artifact_root,
                execution_identity=execution_identity,
                stage=f"{phase}-cache-preparation",
                error=error,
            )
        try:
            completed = subprocess.run(
                command,
                check=False,
                cwd=REPO_ROOT,
                env=child_environment,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            return synthesize_child_failure(
                plan_entry,
                manifest,
                environment_catalog,
                artifact_root,
                execution_identity=execution_identity,
                stage=f"{phase}-child-timeout",
                error=error,
            )
        if completed.returncode != 0:
            existing = _load_existing_result(
                plan_entry,
                manifest,
                environment_catalog,
                artifact_root,
                execution_identity=execution_identity,
            )
            if existing is not None:
                return existing
            return synthesize_child_failure(
                plan_entry,
                manifest,
                environment_catalog,
                artifact_root,
                execution_identity=execution_identity,
                stage=f"{phase}-child-crash",
                error=RunnerError(
                    f"{phase} child exited with status {completed.returncode}"
                ),
            )
    result = _load_existing_result(
        plan_entry,
        manifest,
        environment_catalog,
        artifact_root,
        execution_identity=execution_identity,
    )
    if result is None:
        return synthesize_child_failure(
            plan_entry,
            manifest,
            environment_catalog,
            artifact_root,
            execution_identity=execution_identity,
            stage="finalize-child-crash",
            error=RunnerError("finalize child exited without a cell marker"),
        )
    return result


def orchestrate_cell(
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    environment_paths: Sequence[Path],
    environment_catalog: Mapping[str, Mapping[str, Any]],
    artifact_root: Path,
    *,
    timeout_seconds: float | None = None,
) -> Mapping[str, Any]:
    """Mint one unpaired attempt and run its isolated sample/finalize children."""
    if _requires_paired_execution(plan_entry):
        raise RunnerError("this qualification cell must be launched with run-pair")
    nonce = uuid.uuid4().hex
    identity = _nonpaired_execution_identity(plan_entry, nonce=nonce)
    return _orchestrate_cell(
        plan_entry,
        manifest,
        manifest_path,
        environment_paths,
        environment_catalog,
        artifact_root,
        execution_identity=identity,
        timeout_seconds=timeout_seconds,
    )


def _pair_execution_entries(
    plan_entry: Mapping[str, Any], manifest: Mapping[str, Any], *, nonce: str
) -> tuple[
    tuple[Mapping[str, Any], ExecutionIdentity],
    tuple[Mapping[str, Any], ExecutionIdentity],
]:
    """Derive ordered cells and their linked identities from one parent nonce."""
    candidate, control = resolve_qualification_pair(plan_entry, manifest)
    ordered = ordered_qualification_pair(plan_entry, manifest)
    pair_execution_id = _derive_execution_hash(
        nonce,
        "pair-execution",
        str(candidate["cell_id"]),
        str(control["cell_id"]),
    )
    worker_identity = _worker_identity_sha256(nonce)
    paired: list[tuple[Mapping[str, Any], ExecutionIdentity]] = []
    for position, entry in enumerate(ordered):
        identity = ExecutionIdentity(
            execution_attempt_id=_derive_execution_hash(
                nonce, "cell-attempt", str(entry["cell_id"])
            ),
            pair_execution_id=pair_execution_id,
            pair_position=position,
            worker_identity_sha256=worker_identity,
        )
        _validate_execution_identity(entry, identity)
        paired.append((entry, identity))
    return paired[0], paired[1]


def orchestrate_pair(
    plan_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    environment_paths: Sequence[Path],
    environment_catalog: Mapping[str, Mapping[str, Any]],
    artifact_root: Path,
    *,
    timeout_seconds: float | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Run both members on this parent/worker, continuing after the first fails."""
    entries = _pair_execution_entries(plan_entry, manifest, nonce=uuid.uuid4().hex)
    results: list[Mapping[str, Any]] = []
    uncontained: list[BaseException] = []
    for entry, identity in entries:
        try:
            result = _orchestrate_cell(
                entry,
                manifest,
                manifest_path,
                environment_paths,
                environment_catalog,
                artifact_root,
                execution_identity=identity,
                timeout_seconds=timeout_seconds,
            )
        except Exception as error:
            try:
                result = synthesize_child_failure(
                    entry,
                    manifest,
                    environment_catalog,
                    artifact_root,
                    execution_identity=identity,
                    stage="pair-parent-crash",
                    error=error,
                )
            except Exception as synthesis_error:
                uncontained.append(synthesis_error)
                continue
        results.append(result)
    if uncontained:
        raise RunnerError(
            "pair parent attempted both cells but could not publish every result"
        ) from uncontained[0]
    if len(results) != 2:  # pragma: no cover - guarded by the branch above
        raise RunnerError("pair parent did not produce two cell results")
    return results[0], results[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exactly one frozen hierarchical-TN cell or execution pair"
    )
    parser.add_argument("phase", choices=("sample", "finalize", "run", "run-pair"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--tier",
        required=True,
        choices=("smoke", "qualification", "stress"),
    )
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument(
        "--environment",
        required=True,
        action="append",
        type=Path,
        help="profile-specific environment sidecar (repeatable)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="run-parent timeout for each fresh child (no timeout by default)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; return one when any requested cell failed."""
    args = _parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    plan_entry = select_plan_cell(manifest, tier=args.tier, cell_id=args.cell_id)
    phase_context = None
    if args.phase in {"sample", "finalize"}:
        phase_context = _load_phase_context(
            plan_entry,
            args.artifact_root,
            cast("PhaseName", args.phase),
        )
    catalog = load_environment_catalog(args.environment, manifest)
    if args.phase == "sample":
        assert phase_context is not None
        outcome = run_sample_phase(
            plan_entry,
            manifest,
            catalog,
            args.artifact_root,
            phase_context=phase_context,
        )
        print(json.dumps(outcome, allow_nan=False, sort_keys=True))
        return 1 if outcome.get("execution_status") == "failed" else 0
    if args.phase == "finalize":
        assert phase_context is not None
        result = finalize_cell(
            plan_entry,
            manifest,
            catalog,
            args.artifact_root,
            phase_context=phase_context,
        )
    elif args.phase == "run":
        result = orchestrate_cell(
            plan_entry,
            manifest,
            args.manifest,
            args.environment,
            catalog,
            args.artifact_root,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        results = orchestrate_pair(
            plan_entry,
            manifest,
            args.manifest,
            args.environment,
            catalog,
            args.artifact_root,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(list(results), allow_nan=False, sort_keys=True))
        return (
            0 if all(item["execution_status"] == "completed" for item in results) else 1
        )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["execution_status"] == "completed" else 1


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())

"""Execute backend-paired five-form blocks for the #1282 causal experiment.

This runner is intentionally separate from the frozen v2 qualification harness.
Its unit of scheduling is a backend pair containing two five-representation
blocks for one tier, regime, and replicate.  A parent process mints a run
context, materializes byte-identical data and natural starts, launches all ten
cells in isolated processes/caches, and publishes each cell JSON last.
Scientific failures are evidence and do not suppress paired controls; contract,
environment, artifact, programming, and interrupt failures leave the pair
incomplete instead of manufacturing scientific evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

# Public CI commands historically invoke scripts by file path.  Preserve that
# surface while child workers use module mode, which otherwise has the more
# reliable package import semantics.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor
import xarray as xr
from pymc.initial_point import make_initial_point_fns_per_chain
from pymc.sampling.jax import get_jaxified_graph, sample_numpyro_nuts
from pymc.util import get_random_generator
from pytensor.link.c.exceptions import CompileError
from scipy.special import log_ndtr, ndtri_exp

from scripts.truncated_hierarchy_causal_artifacts import (
    ArtifactRef,
    ArtifactStore,
    CausalArtifactError,
    canonical_json_bytes,
    decode_canonical_json,
    merge_run_directories,
    sha256_bytes,
)
from scripts.truncated_hierarchy_causal_contract import (
    ALLOWED_TIERS,
    DEFAULT_MANIFEST,
    ORACLE_METRICS,
    RUNNER_VERSION,
    CausalContractError,
    RunContext,
    UnitSpec,
    build_plan,
    collect_environment,
    environment_digest,
    load_manifest,
    manifest_digest,
    pair_units,
    plan_unit_by_id,
    validate_environment,
    validate_manifest,
    validate_result_record,
    validate_run_context,
)
from scripts.truncated_hierarchy_causal_models import (
    Parameterization,
    build_causal_model,
)
from scripts.truncated_hierarchy_causal_oracle import (
    CausalParameterization,
    HierarchicalPosteriorSpec,
    TruncationBounds,
    hierarchical_natural_values,
    hierarchical_posterior_components,
    positive_inverse,
    support_inverse,
)
from scripts.truncated_hierarchy_models import (
    Bounds,
    NativeTruncatedPrior,
    SyntheticHierarchyData,
    ToyDataSpec,
    generate_synthetic_data,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

ScientificStage = Literal[
    "data", "build", "initialize", "compile", "sample", "summarize", "diagnose"
]

SCIENTIFIC_STAGES: frozenset[str] = frozenset(
    {"data", "build", "initialize", "compile", "sample", "summarize", "diagnose"}
)
REPRESENTATION_TO_PARAMETERIZATION: dict[str, CausalParameterization] = {
    "native-centered": "centered",
    "manual-centered": "centered",
    "group-icdf-noncentered": "group_icdf_noncentered",
    "location-icdf-noncentered": "location_icdf_noncentered",
    "full-icdf-noncentered": "full_icdf_noncentered",
}
BUILDER_TO_PARAMETERIZATION: dict[str, Parameterization] = {
    "native_centered": "native_centered",
    "manual_centered": "manual_centered",
    "group_icdf_noncentered": "group_icdf_noncentered",
    "location_icdf_noncentered": "location_icdf_noncentered",
    "full_icdf_noncentered": "full_icdf_noncentered",
}
TRAJECTORY_POINTS_PER_CHAIN = 1
CELL_HEARTBEAT_SECONDS = 30.0
CELL_CLEANUP_SECONDS = 2.0
CELL_OBSERVATION_PREFIX = "hssm-causal-observation "


class CausalRunnerError(RuntimeError):
    """Base class for runner contract and infrastructure failures."""


class IncompleteBlockError(CausalRunnerError):
    """Raised when infrastructure prevents a complete ten-cell backend pair."""


class ScientificCellFailure(RuntimeError):
    """A deliberately classified model/compiler/sampler failure."""

    def __init__(
        self,
        stage: ScientificStage,
        message: str,
        *,
        error_type: str | None = None,
    ) -> None:
        if stage not in SCIENTIFIC_STAGES:
            raise ValueError(f"unknown scientific failure stage {stage!r}")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("scientific failure message must be non-empty")
        super().__init__(message)
        self.stage = stage
        self.error_type = error_type or type(self).__name__

    def as_dict(self) -> dict[str, str]:
        """Return the exact failure evidence included in a cell record."""
        return {
            "stage": self.stage,
            "error_type": self.error_type,
            "message": str(self),
        }


@dataclass(frozen=True, slots=True)
class CellExecution:
    """In-memory evidence returned by one isolated cell execution."""

    status: Literal["completed", "failed"]
    coordinate_starts: Mapping[str, Any] | None = None
    chain_dataset: xr.Dataset | None = None
    diagnostics: Mapping[str, Any] | None = None
    metrics: Mapping[str, Any] | None = None
    parameter_summaries: Sequence[Mapping[str, Any]] | None = None
    failure: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Enforce completed-versus-failed evidence separation."""
        if self.status == "completed":
            if (
                self.coordinate_starts is None
                or self.chain_dataset is None
                or self.diagnostics is None
                or self.metrics is None
                or self.parameter_summaries is None
                or self.failure is not None
            ):
                raise CausalRunnerError("completed cell lacks required evidence")
        elif self.status == "failed":
            if self.failure is None:
                raise CausalRunnerError("failed cell lacks classified failure")
            stage = self.failure.get("stage")
            if stage in {"summarize", "diagnose"} and self.chain_dataset is None:
                raise CausalRunnerError(
                    "late scientific failure must preserve the completed chain"
                )
            if (
                stage not in {"summarize", "diagnose"}
                and self.chain_dataset is not None
            ):
                raise CausalRunnerError(
                    "pre-summary scientific failure cannot publish a chain"
                )
            if self.parameter_summaries:
                raise CausalRunnerError(
                    "failed cell cannot publish posterior summaries"
                )
        else:
            raise CausalRunnerError(f"unknown cell status {self.status!r}")


def _regime_bounds(unit: UnitSpec) -> Bounds:
    return Bounds(unit.regime["lower"], unit.regime["upper"])


def _toy_spec(unit: UnitSpec) -> ToyDataSpec:
    """Construct the public B1 data specification from one planned unit."""
    return ToyDataSpec(
        bounds=_regime_bounds(unit),
        group_location=float(unit.regime["truth_group_location"]),
        group_scale=float(unit.regime["truth_group_scale"]),
        n_groups=int(unit.regime["n_groups"]),
        n_per_group=int(unit.regime["n_per_group"]),
        floatx=cast("Literal['float32', 'float64']", unit.floatx),
        observation_sigma=float(unit.natural_model["observation_sigma"]),
    )


def _prior(unit: UnitSpec) -> NativeTruncatedPrior:
    """Construct the one common natural prior from the frozen plan entry."""
    return NativeTruncatedPrior(
        bounds=_regime_bounds(unit),
        location_base_mean=float(unit.regime["prior_hyper_location"]),
        location_prior_sigma=float(unit.natural_model["location_prior_sigma"]),
        scale_prior_alpha=float(unit.natural_model["scale_prior_alpha"]),
        scale_prior_beta=float(unit.natural_model["scale_prior_beta"]),
    )


def _oracle_spec(
    unit: UnitSpec, prior: NativeTruncatedPrior, data: SyntheticHierarchyData
) -> HierarchicalPosteriorSpec:
    return HierarchicalPosteriorSpec(
        bounds=TruncationBounds(prior.bounds.lower, prior.bounds.upper),
        location_base_mean=prior.location_base_mean,
        location_prior_scale=prior.location_prior_sigma,
        scale_prior_shape=prior.scale_prior_alpha,
        scale_prior_scale=prior.scale_prior_beta,
        n_groups=int(unit.regime["n_groups"]),
        group_index=data.group_index,
        observations=data.y,
        observation_scale=data.spec.observation_sigma,
    )


def _data_payload(unit: UnitSpec, data: SyntheticHierarchyData) -> dict[str, Any]:
    """Serialize every generated byte needed to rebuild a dataset exactly."""
    return {
        "schema_version": unit.schema_version,
        "study_id": unit.study_id,
        "manifest_sha256": unit.manifest_sha256,
        "data_id": unit.data_id,
        "tier": unit.tier,
        "regime_id": unit.regime_id,
        "replicate": unit.replicate,
        "seeds": {
            "data_seed": unit.data_seed,
            "truth_seed": unit.truth_seed,
            "group_seed": unit.group_seed,
            "observation_seed": unit.observation_seed,
        },
        "spec": {
            "lower": data.spec.bounds.lower,
            "upper": data.spec.bounds.upper,
            "truth_group_location": data.spec.group_location,
            "truth_group_scale": data.spec.group_scale,
            "n_groups": data.spec.n_groups,
            "n_per_group": data.spec.n_per_group,
            "floatx": data.spec.floatx,
            "observation_sigma": data.spec.observation_sigma,
        },
        "group_labels": list(data.group_labels),
        "group_index": [int(value) for value in data.group_index],
        "group_effect": [float(value) for value in data.group_effect],
        "observations": [float(value) for value in data.y],
    }


def _data_from_payload(
    payload: Mapping[str, Any], unit: UnitSpec
) -> SyntheticHierarchyData:
    """Strictly reconstruct and re-derive a shared input artifact."""
    if payload.get("data_id") != unit.data_id:
        raise CausalRunnerError("data artifact identity does not match the plan")
    expected = generate_synthetic_data(
        _toy_spec(unit),
        group_seed=unit.group_seed,
        observation_seed=unit.observation_seed,
    )
    if canonical_json_bytes(_data_payload(unit, expected)) != canonical_json_bytes(
        payload
    ):
        raise CausalRunnerError(
            "data artifact does not match deterministic regeneration"
        )
    return expected


def _pack_model_point(model: pm.Model, point: Mapping[str, Any]) -> np.ndarray:
    """Pack B1 value variables in their frozen location/scale/groups order."""
    pieces: list[np.ndarray] = []
    for variable in model.value_vars:
        if variable.name not in point:
            raise CausalRunnerError(f"point lacks value variable {variable.name!r}")
        pieces.append(np.asarray(point[variable.name]).reshape(-1))
    vector = np.concatenate(pieces).astype(model.value_vars[0].dtype, copy=False)
    if not np.all(np.isfinite(vector)):
        raise ScientificCellFailure("initialize", "initial coordinate is non-finite")
    return vector


def _point_from_vector(model: pm.Model, vector: np.ndarray) -> dict[str, np.ndarray]:
    """Split one canonical coordinate vector into a concrete PyMC point."""
    initial = model.initial_point()
    cursor = 0
    result: dict[str, np.ndarray] = {}
    for variable in model.value_vars:
        shape = np.asarray(initial[variable.name]).shape
        size = int(np.prod(shape, dtype=int)) if shape else 1
        result[variable.name] = np.asarray(
            vector[cursor : cursor + size], dtype=variable.dtype
        ).reshape(shape)
        cursor += size
    if cursor != vector.size:
        raise CausalRunnerError("coordinate dimension does not match model graph")
    return result


def _logdiffexp(larger: float, smaller: float) -> float:
    if not larger > smaller:
        raise ScientificCellFailure("initialize", "degenerate truncated probability")
    return larger + math.log(-math.expm1(smaller - larger))


def _normal_interval_logmass(lower: float, upper: float) -> float:
    """Return log(Phi(upper)-Phi(lower)) without tail cancellation."""
    if not lower < upper:
        raise ScientificCellFailure("initialize", "truncated interval is empty")
    if lower >= 0.0:
        return _logdiffexp(float(log_ndtr(-lower)), float(log_ndtr(-upper)))
    return _logdiffexp(float(log_ndtr(upper)), float(log_ndtr(lower)))


def _truncated_offset_from_natural(
    value: float,
    *,
    location: float,
    scale: float,
    bounds: Bounds,
) -> float:
    """Invert the B1 TN-ICDF map using stable interval probabilities."""
    if not bounds.contains(value) or not math.isfinite(scale) or scale <= 0.0:
        raise ScientificCellFailure("initialize", "natural start is outside support")
    lower = -math.inf if bounds.lower is None else (bounds.lower - location) / scale
    upper = math.inf if bounds.upper is None else (bounds.upper - location) / scale
    standardized = (value - location) / scale
    total = _normal_interval_logmass(lower, upper)
    log_quantile = _normal_interval_logmass(lower, standardized) - total
    log_survival = _normal_interval_logmass(standardized, upper) - total
    offset = (
        float(ndtri_exp(log_quantile))
        if log_quantile <= log_survival
        else -float(ndtri_exp(log_survival))
    )
    if not math.isfinite(offset):
        raise ScientificCellFailure("initialize", "TN inverse start is non-finite")
    return offset


def natural_to_coordinate(
    natural: Mapping[str, Any],
    *,
    prior: NativeTruncatedPrior,
    oracle_spec: HierarchicalPosteriorSpec,
    representation_id: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map one common natural hierarchy into a representation and round-trip it."""
    try:
        parameterization = REPRESENTATION_TO_PARAMETERIZATION[representation_id]
    except KeyError as error:
        raise CausalRunnerError(
            f"unknown representation {representation_id!r}"
        ) from error
    location = float(natural["group_location"])
    scale = float(natural["group_scale"])
    groups = np.asarray(natural["group_effect"], dtype=np.float64)
    if groups.shape != (oracle_spec.n_groups,):
        raise CausalRunnerError("natural group start has the wrong shape")
    location_noncentered = parameterization in {
        "location_icdf_noncentered",
        "full_icdf_noncentered",
    }
    group_noncentered = parameterization in {
        "group_icdf_noncentered",
        "full_icdf_noncentered",
    }
    if location_noncentered:
        location_coordinate = _truncated_offset_from_natural(
            location,
            location=prior.location_base_mean,
            scale=prior.location_prior_sigma,
            bounds=prior.bounds,
        )
    else:
        location_coordinate = support_inverse(location, oracle_spec.bounds)
    if group_noncentered:
        group_coordinates = [
            _truncated_offset_from_natural(
                float(group), location=location, scale=scale, bounds=prior.bounds
            )
            for group in groups
        ]
    else:
        group_coordinates = [
            support_inverse(float(group), oracle_spec.bounds) for group in groups
        ]
    vector = np.asarray(
        [location_coordinate, positive_inverse(scale), *group_coordinates],
        dtype=np.float64,
    )

    return vector, _natural_coordinate_roundtrip(
        natural,
        vector,
        oracle_spec=oracle_spec,
        parameterization=parameterization,
    )


def _natural_coordinate_roundtrip(
    natural: Mapping[str, Any],
    vector: np.ndarray,
    *,
    oracle_spec: HierarchicalPosteriorSpec,
    parameterization: CausalParameterization,
) -> dict[str, Any]:
    """Round-trip the exact graph-representable coordinate to natural scale."""
    location = float(natural["group_location"])
    scale = float(natural["group_scale"])
    groups = np.asarray(natural["group_effect"], dtype=np.float64)
    restored = hierarchical_natural_values(vector, oracle_spec, parameterization)
    restored_groups = np.asarray(
        [group.value for group in restored.group_effect], dtype=np.float64
    )
    differences = np.concatenate(
        [
            [abs(restored.location.value - location)],
            [abs(restored.scale.value - scale)],
            np.abs(restored_groups - groups),
        ]
    )
    roundtrip = {
        "group_location": restored.location.value,
        "group_scale": restored.scale.value,
        "group_effect": restored_groups.tolist(),
        "absolute_error_max": float(np.max(differences)),
    }
    return roundtrip


def _natural_start_payload(
    unit: UnitSpec,
    data: SyntheticHierarchyData,
    prior: NativeTruncatedPrior,
) -> dict[str, Any]:
    """Generate the common starts once from native-centered PyMC jitter.

    This makes the previously implicit ``jitter+adapt_diag`` initialization
    explicit and inspectable.  Only the starting point generation is reused;
    sampling receives the resulting exact points with ``init='adapt_diag'`` so
    neither backend adds a second, representation-specific jitter.
    """
    if pytensor.config.floatX != unit.floatx:
        raise CausalRunnerError(
            "natural-start source process floatX differs from the planned regime"
        )
    model = build_causal_model("native_centered", prior, data)
    value_variable_dtypes = [str(variable.dtype) for variable in model.value_vars]
    if not value_variable_dtypes or any(
        dtype != unit.floatx for dtype in value_variable_dtypes
    ):
        raise CausalRunnerError(
            "natural-start source graph contains an unplanned value-variable dtype"
        )
    functions = make_initial_point_fns_per_chain(
        model=model,
        overrides=None,
        jitter_rvs=set(model.free_RVs),
        chains=unit.chains,
    )
    oracle_spec = _oracle_spec(unit, prior, data)
    chains: list[dict[str, Any]] = []
    for chain, (function, seed) in enumerate(
        zip(functions, unit.natural_start_chain_seeds, strict=True)
    ):
        point = function(seed)
        vector = _pack_model_point(model, point).astype(np.float64)
        natural = hierarchical_natural_values(vector, oracle_spec, "centered")
        groups = [float(item.value) for item in natural.group_effect]
        chains.append(
            {
                "chain": chain,
                "seed": seed,
                "group_location": float(natural.location.value),
                "group_scale": float(natural.scale.value),
                "group_effect": groups,
            }
        )
    return {
        "schema_version": unit.schema_version,
        "study_id": unit.study_id,
        "manifest_sha256": unit.manifest_sha256,
        "start_id": unit.start_id,
        "data_id": unit.data_id,
        "tier": unit.tier,
        "regime_id": unit.regime_id,
        "replicate": unit.replicate,
        "policy": "native-centered-support-point-plus-uniform-jitter-v3",
        "source_graph": {
            "builder": "native_centered",
            "pytensor_floatx": pytensor.config.floatX,
            "value_variable_dtypes": value_variable_dtypes,
        },
        "natural_start_seed": unit.natural_start_seed,
        "natural_start_chain_seeds": list(unit.natural_start_chain_seeds),
        "chains": chains,
    }


def _validate_natural_start_payload(
    payload: Mapping[str, Any],
    unit: UnitSpec,
    prior: NativeTruncatedPrior,
    data: SyntheticHierarchyData,
) -> None:
    """Validate identity, seeds, shapes, support, and finite natural values."""
    identity = {
        "schema_version": unit.schema_version,
        "study_id": unit.study_id,
        "manifest_sha256": unit.manifest_sha256,
        "start_id": unit.start_id,
        "data_id": unit.data_id,
        "tier": unit.tier,
        "regime_id": unit.regime_id,
        "replicate": unit.replicate,
    }
    for name, expected in identity.items():
        if payload.get(name) != expected:
            raise CausalRunnerError(f"natural start {name} does not match the plan")
    if payload.get("natural_start_chain_seeds") != list(unit.natural_start_chain_seeds):
        raise CausalRunnerError("natural start chain seeds do not match the plan")
    source_graph = payload.get("source_graph")
    if not isinstance(source_graph, dict) or set(source_graph) != {
        "builder",
        "pytensor_floatx",
        "value_variable_dtypes",
    }:
        raise CausalRunnerError("natural start source graph attestation is malformed")
    if (
        source_graph["builder"] != "native_centered"
        or source_graph["pytensor_floatx"] != unit.floatx
        or not isinstance(source_graph["value_variable_dtypes"], list)
        or not source_graph["value_variable_dtypes"]
        or any(dtype != unit.floatx for dtype in source_graph["value_variable_dtypes"])
    ):
        raise CausalRunnerError("natural start source graph precision is wrong")
    chains = payload.get("chains")
    if not isinstance(chains, list) or len(chains) != unit.chains:
        raise CausalRunnerError("natural start artifact has the wrong chain count")
    for index, chain in enumerate(chains):
        if not isinstance(chain, dict):
            raise CausalRunnerError("natural start chain must be an object")
        if chain.get("chain") != index:
            raise CausalRunnerError("natural start chains are out of order")
        if chain.get("seed") != unit.natural_start_chain_seeds[index]:
            raise CausalRunnerError("natural start seed does not match the plan")
        location = float(chain.get("group_location", math.nan))
        scale = float(chain.get("group_scale", math.nan))
        groups = np.asarray(chain.get("group_effect", []), dtype=np.float64)
        if not prior.bounds.contains(location) or not scale > 0.0:
            raise CausalRunnerError("natural start hyperparameters are invalid")
        if groups.shape != (data.spec.n_groups,) or not prior.bounds.contains(groups):
            raise CausalRunnerError("natural group start is invalid")
    expected = _natural_start_payload(unit, data, prior)
    if canonical_json_bytes(expected) != canonical_json_bytes(payload):
        raise CausalRunnerError(
            "natural start artifact does not match deterministic regeneration"
        )


def _materialize_inputs_for_unit_current_process(
    unit: UnitSpec, store: ArtifactStore
) -> tuple[ArtifactRef, ArtifactRef]:
    """Publish one pair after the caller has activated its planned precision."""
    if pytensor.config.floatX != unit.floatx:
        raise CausalRunnerError(
            "input materialization process floatX differs from the planned regime"
        )
    data = generate_synthetic_data(
        _toy_spec(unit),
        group_seed=unit.group_seed,
        observation_seed=unit.observation_seed,
    )
    data_payload = _data_payload(unit, data)
    data_reference = store.ensure_json(f"data/{unit.data_id}.json", data_payload)
    prior = _prior(unit)
    start_payload = _natural_start_payload(unit, data, prior)
    _validate_natural_start_payload(start_payload, unit, prior, data)
    start_reference = store.ensure_json(
        f"starts/natural/{unit.start_id}.json", start_payload
    )
    return data_reference, start_reference


def _materialize_inputs_subprocess(
    unit: UnitSpec,
    store: ArtifactStore,
    *,
    manifest_path: Path,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Materialize one shared pair in a fresh precision-specific interpreter."""
    with tempfile.TemporaryDirectory(
        prefix="hssm-causal-materialize-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        (temporary / "pytensor").mkdir()
        (temporary / "jax").mkdir()
        (temporary / "matplotlib").mkdir()
        (temporary / "xdg-cache").mkdir()
        environment = os.environ.copy()
        environment.update(
            {
                "PYTENSOR_FLAGS": (
                    f"base_compiledir={temporary / 'pytensor'},floatX={unit.floatx}"
                ),
                "JAX_COMPILATION_CACHE_DIR": str(temporary / "jax"),
                "JAX_ENABLE_X64": "true" if unit.floatx == "float64" else "false",
                "JAX_PLATFORMS": "cpu",
                "MPLCONFIGDIR": str(temporary / "matplotlib"),
                "XDG_CACHE_HOME": str(temporary / "xdg-cache"),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        command = [
            sys.executable,
            "-m",
            "scripts.truncated_hierarchy_causal_runner",
            "--manifest",
            str(manifest_path.resolve()),
            "_materialize-unit",
            "--tier",
            unit.tier,
            "--cell-id",
            unit.cell_id,
            "--run-dir",
            str(store.root),
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise IncompleteBlockError(
            f"precision-specific input materialization failed for {unit.data_id}: "
            f"{completed.stderr[-4000:]}"
        )
    try:
        payload = decode_canonical_json(completed.stdout.encode("utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"data", "natural_start"}:
            raise CausalArtifactError("materialization response has unexpected fields")
        data_reference = ArtifactRef.from_dict(payload["data"])
        start_reference = ArtifactRef.from_dict(payload["natural_start"])
    except (CausalArtifactError, TypeError, UnicodeError) as error:
        raise IncompleteBlockError(
            "precision-specific materialization returned invalid evidence"
        ) from error
    if (
        data_reference.path != f"data/{unit.data_id}.json"
        or start_reference.path != f"starts/natural/{unit.start_id}.json"
    ):
        raise IncompleteBlockError("materialization returned unplanned artifact paths")
    store.verify(data_reference)
    store.verify(start_reference)
    return data_reference, start_reference


def materialize_inputs_for_unit(
    unit: UnitSpec,
    store: ArtifactStore,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Publish inputs directly only when this process has the planned floatX."""
    if pytensor.config.floatX == unit.floatx:
        return _materialize_inputs_for_unit_current_process(unit, store)
    return _materialize_inputs_subprocess(unit, store, manifest_path=manifest_path)


def materialize_inputs(
    manifest: Mapping[str, Any],
    tier: str,
    store: ArtifactStore,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> tuple[tuple[ArtifactRef, ArtifactRef], ...]:
    """Materialize each distinct data/start pair in a tier exactly once."""
    representatives: dict[str, UnitSpec] = {}
    for unit in build_plan(manifest, tier):
        representatives.setdefault(unit.data_id, unit)
    return tuple(
        materialize_inputs_for_unit(unit, store, manifest_path=manifest_path)
        for unit in representatives.values()
    )


def _coordinate_start_payload(
    unit: UnitSpec,
    model: pm.Model,
    starts: Mapping[str, Any],
    prior: NativeTruncatedPrior,
    oracle_spec: HierarchicalPosteriorSpec,
) -> tuple[dict[str, Any], list[dict[str, np.ndarray]]]:
    """Map every shared natural chain start into one graph's coordinates."""
    _, chain_rng_provenance = _sampler_rng_provenance(unit)
    records: list[dict[str, Any]] = []
    initvals: list[dict[str, np.ndarray]] = []
    for chain in starts["chains"]:
        natural = {
            "group_location": chain["group_location"],
            "group_scale": chain["group_scale"],
            "group_effect": chain["group_effect"],
        }
        candidate_vector, _ = natural_to_coordinate(
            natural,
            prior=prior,
            oracle_spec=oracle_spec,
            representation_id=unit.representation_id,
        )
        point = _point_from_vector(model, candidate_vector)
        vector = _pack_model_point(model, point)
        roundtrip = _natural_coordinate_roundtrip(
            natural,
            vector,
            oracle_spec=oracle_spec,
            parameterization=REPRESENTATION_TO_PARAMETERIZATION[unit.representation_id],
        )
        initvals.append(point)
        records.append(
            {
                "chain": chain["chain"],
                "natural_start_seed": chain["seed"],
                "chain_rng_provenance": chain_rng_provenance[chain["chain"]],
                "natural": natural,
                "coordinate_vector": vector.tolist(),
                "value_variables": {
                    name: np.asarray(value).tolist() for name, value in point.items()
                },
                "roundtrip": roundtrip,
            }
        )
    return (
        {
            "schema_version": unit.schema_version,
            "study_id": unit.study_id,
            "manifest_sha256": unit.manifest_sha256,
            "cell_id": unit.cell_id,
            "start_id": unit.start_id,
            "representation_id": unit.representation_id,
            "backend_id": unit.backend_id,
            "chains": records,
        },
        initvals,
    )


def _sampler_rng_provenance(unit: UnitSpec) -> tuple[Any, list[dict[str, Any]]]:
    """Describe exact backend RNG inputs and chain streams on the pinned stack."""
    if unit.backend_id == "pymc":
        if unit.sampler_seed is not None or len(unit.chain_seeds) != unit.chains:
            raise CausalRunnerError("PyMC seed shape differs from the plan")
        # PyMC 6.3.1 treats this sequence as SeedSequence entropy, spawns one
        # Generator per chain, draws one init/step integer, and hands the advanced
        # Generator to the sampler.  Record both the draw and exact handoff state.
        generators = get_random_generator(list(unit.chain_seeds)).spawn(unit.chains)
        records: list[dict[str, Any]] = []
        for chain, generator in enumerate(generators):
            init_step_seed = int(generator.integers(2**30))
            seed_sequence = generator.bit_generator.seed_seq
            records.append(
                {
                    "chain": chain,
                    "rng": "numpy.random.Generator(PCG64)",
                    "spawn_key": list(getattr(seed_sequence, "spawn_key")),
                    "seed_sequence_pool_size": getattr(seed_sequence, "pool_size"),
                    "init_step_seed": init_step_seed,
                    "post_init_draw_state_sha256": sha256_bytes(
                        canonical_json_bytes(generator.bit_generator.state)
                    ),
                }
            )
        return list(unit.chain_seeds), records
    if unit.backend_id == "numpyro":
        if unit.sampler_seed is None or unit.chain_seeds:
            raise CausalRunnerError("NumPyro seed shape differs from the plan")
        key = jax.random.PRNGKey(unit.sampler_seed)
        keys = key[None, :] if unit.chains == 1 else jax.random.split(key, unit.chains)
        key_lists = np.asarray(keys, dtype=np.uint32).astype(np.uint64).tolist()
        return unit.sampler_seed, [
            {"chain": chain, "rng": "jax-prng-key", "key": chain_key}
            for chain, chain_key in enumerate(key_lists)
        ]
    raise CausalRunnerError(f"unknown backend {unit.backend_id!r}")


def _finite_initial_evidence(
    model: pm.Model, initvals: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Evaluate log density and gradient at every exact sampler start."""
    try:
        outputs: list[Any] = [model.logp(jacobian=True), model.dlogp(jacobian=True)]
        function = model.compile_fn(
            outputs,
            inputs=model.value_vars,
            on_unused_input="ignore",
            point_fn=False,
        )
    except (CompileError, NotImplementedError) as error:
        raise ScientificCellFailure(
            "compile", str(error), error_type=type(error).__name__
        ) from error
    records: list[dict[str, Any]] = []
    for chain, point in enumerate(initvals):
        arguments = [point[variable.name] for variable in model.value_vars]
        value_raw, derivative_raw = function(*arguments)
        value = float(value_raw)
        derivative = np.asarray(derivative_raw, dtype=np.float64)
        finite = bool(math.isfinite(value) and np.all(np.isfinite(derivative)))
        records.append(
            {
                "chain": chain,
                "logp": value,
                "gradient_finite": bool(np.all(np.isfinite(derivative))),
                "gradient_absolute_max": float(np.max(np.abs(derivative))),
                "finite": finite,
            }
        )
    if not all(record["finite"] for record in records):
        raise ScientificCellFailure(
            "initialize", "at least one exact shared start has non-finite logp/gradient"
        )
    return {"chains": records, "all_finite": True}


def _sample_model(
    unit: UnitSpec,
    model: pm.Model,
    initvals: Sequence[Mapping[str, Any]],
) -> tuple[Any, float]:
    """Sample one graph with exact starts and frozen backend-specific RNG input."""
    started = time.perf_counter()
    exact_initvals = cast("Any", [dict(point) for point in initvals])
    try:
        with cast("Any", model):
            if unit.backend_id == "pymc":
                if (
                    unit.sampler_seed is not None
                    or len(unit.chain_seeds) != unit.chains
                ):
                    raise CausalRunnerError("PyMC seed shape differs from the plan")
                inference = pm.sample(
                    draws=unit.draws,
                    tune=unit.tune,
                    chains=unit.chains,
                    cores=1,
                    random_seed=list(unit.chain_seeds),
                    nuts_sampler="pymc",
                    initvals=exact_initvals,
                    init="adapt_diag",
                    progressbar=False,
                    compute_convergence_checks=False,
                    discard_tuned_samples=False,
                    return_inferencedata=True,
                    nuts={
                        "target_accept": unit.target_accept,
                        "max_treedepth": unit.max_treedepth,
                    },
                )
            elif unit.backend_id == "numpyro":
                if unit.sampler_seed is None or unit.chain_seeds:
                    raise CausalRunnerError("NumPyro seed shape differs from the plan")
                inference = sample_numpyro_nuts(
                    draws=unit.draws,
                    tune=unit.tune,
                    chains=unit.chains,
                    target_accept=unit.target_accept,
                    random_seed=unit.sampler_seed,
                    initvals=exact_initvals,
                    jitter=False,
                    model=model,
                    progressbar=False,
                    quiet=True,
                    chain_method="sequential",  # type: ignore[arg-type]
                    nuts_kwargs={"max_tree_depth": unit.max_treedepth},
                    compute_convergence_checks=False,
                )
            else:
                raise CausalRunnerError(f"unknown backend {unit.backend_id!r}")
    except (CompileError, NotImplementedError) as error:
        raise ScientificCellFailure(
            "compile", str(error), error_type=type(error).__name__
        ) from error
    except (
        FloatingPointError,
        np.linalg.LinAlgError,
        pm.exceptions.SamplingError,
    ) as error:
        raise ScientificCellFailure(
            "sample", str(error), error_type=type(error).__name__
        ) from error
    elapsed = time.perf_counter() - started
    return inference, elapsed


def _dataset_view(group: Any) -> xr.Dataset:
    """Normalize an ArviZ DataTree group or xarray dataset."""
    if isinstance(group, xr.Dataset):
        return group
    if hasattr(group, "to_dataset"):
        return group.to_dataset()
    if hasattr(group, "dataset"):
        return xr.Dataset(group.dataset)
    raise CausalRunnerError("inference output lacks an xarray dataset group")


def _chain_identity_attributes(unit: UnitSpec) -> dict[str, Any]:
    """Return the exact plan and RNG identity carried by every chain artifact."""
    sampler_seed_input, chain_rng_provenance = _sampler_rng_provenance(unit)
    return {
        "schema_version": unit.schema_version,
        "study_id": unit.study_id,
        "manifest_sha256": unit.manifest_sha256,
        "cell_id": unit.cell_id,
        "pair_id": unit.pair_id,
        "pair_position": unit.pair_position,
        "block_id": unit.block_id,
        "block_position": unit.block_position,
        "tier": unit.tier,
        "regime_id": unit.regime_id,
        "backend_id": unit.backend_id,
        "representation_id": unit.representation_id,
        "replicate": unit.replicate,
        "chains": unit.chains,
        "draws": unit.draws,
        "pymc_seed_entropy_words_json": json.dumps(
            list(unit.chain_seeds), separators=(",", ":")
        ),
        "sampler_seed_input_json": json.dumps(
            sampler_seed_input, separators=(",", ":")
        ),
        "chain_rng_provenance_json": json.dumps(
            chain_rng_provenance, separators=(",", ":"), sort_keys=True
        ),
    }


def _validate_standardized_natural_chains(unit: UnitSpec, chains: xr.Dataset) -> None:
    """Fail closed unless a canonical chain is complete and plan-bound."""
    expected_attrs = _chain_identity_attributes(unit)
    for name, expected in expected_attrs.items():
        if chains.attrs.get(name) != expected:
            raise CausalRunnerError(
                f"standardized chain attribute {name!r} is not plan-bound"
            )
    expected_sizes = {
        "chain": unit.chains,
        "draw": unit.draws,
        "group": int(unit.regime["n_groups"]),
    }
    for dimension, size in expected_sizes.items():
        if chains.sizes.get(dimension) != size:
            raise CausalRunnerError(
                f"standardized chain dimension {dimension!r} has the wrong size"
            )
        if not np.array_equal(
            np.asarray(chains.coords[dimension]), np.arange(size, dtype=np.int64)
        ):
            raise CausalRunnerError(
                f"standardized chain coordinate {dimension!r} is not canonical"
            )
    expected_natural_dims = {
        "group_location": ("chain", "draw"),
        "group_scale": ("chain", "draw"),
        "group_effect": ("chain", "draw", "group"),
    }
    required_stats = {
        "acceptance_rate",
        "diverging",
        "energy",
        "n_steps",
        "step_size",
        "tree_depth",
    }
    for name, dimensions in expected_natural_dims.items():
        if name not in chains or chains[name].dims != dimensions:
            raise CausalRunnerError(
                f"standardized chain variable {name!r} has the wrong dimensions"
            )
        if not np.all(np.isfinite(np.asarray(chains[name]))):
            raise CausalRunnerError(
                f"standardized chain variable {name!r} is non-finite"
            )
    for name in required_stats:
        variable = f"sample_stat__{name}"
        if variable not in chains or chains[variable].dims != ("chain", "draw"):
            raise CausalRunnerError(
                f"standardized chain lacks canonical statistic {name!r}"
            )
        if not np.all(np.isfinite(np.asarray(chains[variable]))):
            raise CausalRunnerError(
                f"standardized chain statistic {name!r} is non-finite"
            )
    warmup_retained = chains.attrs.get("warmup_sample_stats_retained")
    if warmup_retained not in {0, 1}:
        raise CausalRunnerError("standardized chain warmup attestation is invalid")
    warmup_variables = [
        name
        for name in chains.data_vars
        if isinstance(name, str) and name.startswith("warmup_sample_stat__")
    ]
    if warmup_retained:
        if chains.sizes.get("warmup_draw") != unit.tune or not warmup_variables:
            raise CausalRunnerError("standardized warmup evidence is incomplete")
        if not np.array_equal(
            np.asarray(chains.coords["warmup_draw"]),
            np.arange(unit.tune, dtype=np.int64),
        ):
            raise CausalRunnerError("standardized warmup coordinate is not canonical")
        if any(
            chains[name].dims != ("chain", "warmup_draw") for name in warmup_variables
        ):
            raise CausalRunnerError(
                "standardized warmup statistic has wrong dimensions"
            )
    elif warmup_variables or "warmup_draw" in chains.dims:
        raise CausalRunnerError("standardized chain has unattested warmup evidence")


def _standardize_natural_chains(unit: UnitSpec, inference: Any) -> xr.Dataset:
    """Retain canonical natural draws and auditable per-draw sampler evidence."""
    try:
        posterior = _dataset_view(inference.posterior)
    except AttributeError as error:
        raise ScientificCellFailure(
            "summarize", "sampler output lacks posterior chains"
        ) from error
    required = {"group_location", "group_scale", "group_effect"}
    missing = required.difference(posterior.data_vars)
    if missing:
        raise ScientificCellFailure(
            "summarize", f"posterior lacks natural variables: {sorted(missing)}"
        )
    location = posterior["group_location"].transpose("chain", "draw")
    scale = posterior["group_scale"].transpose("chain", "draw")
    effect = posterior["group_effect"].transpose("chain", "draw", "group")
    expected_shapes = {
        "location": (unit.chains, unit.draws),
        "scale": (unit.chains, unit.draws),
        "effect": (unit.chains, unit.draws, int(unit.regime["n_groups"])),
    }
    observed_shapes = {
        "location": location.shape,
        "scale": scale.shape,
        "effect": effect.shape,
    }
    if observed_shapes != expected_shapes:
        message = (
            f"posterior natural-chain shapes are {observed_shapes}, "
            f"expected {expected_shapes}"
        )
        raise ScientificCellFailure(
            "summarize",
            message,
        )
    values = np.concatenate(
        [
            np.asarray(location).reshape(-1),
            np.asarray(scale).reshape(-1),
            np.asarray(effect).reshape(-1),
        ]
    )
    if not np.all(np.isfinite(values)):
        raise ScientificCellFailure("summarize", "posterior contains non-finite values")
    try:
        sample_stats = _dataset_view(inference.sample_stats)
    except AttributeError as error:
        raise ScientificCellFailure(
            "sample", "sampler output lacks per-draw statistics"
        ) from error
    required_stats = {
        "acceptance_rate",
        "diverging",
        "energy",
        "n_steps",
        "step_size",
        "tree_depth",
    }
    missing_stats = required_stats.difference(sample_stats.data_vars)
    if missing_stats:
        raise ScientificCellFailure(
            "sample", f"sample_stats lacks required fields: {sorted(missing_stats)}"
        )
    data_vars: dict[str, Any] = {
        "group_location": (("chain", "draw"), np.asarray(location)),
        "group_scale": (("chain", "draw"), np.asarray(scale)),
        "group_effect": (("chain", "draw", "group"), np.asarray(effect)),
    }

    def retain_stats(
        source: xr.Dataset,
        *,
        prefix: str,
        draw_dimension: str,
        expected_draws: int,
    ) -> None:
        for raw_name in sorted(source.data_vars, key=str):
            if not isinstance(raw_name, str):
                raise ScientificCellFailure(
                    "sample", "sample statistic names must be strings"
                )
            name = raw_name
            value = source[name]
            if set(value.dims) != {"chain", "draw"} or len(value.dims) != 2:
                raise ScientificCellFailure(
                    "sample", f"sample statistic {name!r} has unsupported dimensions"
                )
            ordered = value.transpose("chain", "draw")
            if ordered.shape != (unit.chains, expected_draws):
                raise ScientificCellFailure(
                    "sample", f"sample statistic {name!r} has the wrong shape"
                )
            array = np.asarray(ordered)
            if array.dtype.kind not in "biuf":
                raise ScientificCellFailure(
                    "sample", f"sample statistic {name!r} is not numeric"
                )
            if name in required_stats and not np.all(np.isfinite(array)):
                raise ScientificCellFailure(
                    "sample", f"required sample statistic {name!r} is non-finite"
                )
            data_vars[f"{prefix}{name}"] = (
                ("chain", draw_dimension),
                array,
            )

    retain_stats(
        sample_stats,
        prefix="sample_stat__",
        draw_dimension="draw",
        expected_draws=unit.draws,
    )
    warmup_retained = False
    if hasattr(inference, "warmup_sample_stats"):
        warmup_stats = _dataset_view(inference.warmup_sample_stats)
        retain_stats(
            warmup_stats,
            prefix="warmup_sample_stat__",
            draw_dimension="warmup_draw",
            expected_draws=unit.tune,
        )
        warmup_retained = True
    group_index = np.arange(int(unit.regime["n_groups"]), dtype=np.int64)
    result = xr.Dataset(
        data_vars=data_vars,
        coords={
            "chain": np.arange(unit.chains, dtype=np.int64),
            "draw": np.arange(unit.draws, dtype=np.int64),
            "group": group_index,
            **(
                {"warmup_draw": np.arange(unit.tune, dtype=np.int64)}
                if warmup_retained
                else {}
            ),
        },
        attrs={
            **_chain_identity_attributes(unit),
            "warmup_sample_stats_retained": int(warmup_retained),
        },
    )
    _validate_standardized_natural_chains(unit, result)
    return result


def _dataarray_values(dataset: xr.Dataset) -> np.ndarray:
    """Flatten all values in an ArviZ diagnostic dataset."""
    pieces = [np.asarray(value).reshape(-1) for value in dataset.data_vars.values()]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float64)


def _sampler_metrics(
    unit: UnitSpec,
    chains: xr.Dataset,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Recompute every registered sampler metric from the retained chain artifact."""
    _validate_standardized_natural_chains(unit, chains)

    def statistic(name: str) -> xr.DataArray:
        variable = f"sample_stat__{name}"
        if variable not in chains:
            raise ScientificCellFailure(
                "summarize", f"retained chain lacks sample statistic {name!r}"
            )
        value = chains[variable]
        if value.dims != ("chain", "draw") or value.shape != (
            unit.chains,
            unit.draws,
        ):
            raise ScientificCellFailure(
                "summarize", f"retained sample statistic {name!r} has wrong shape"
            )
        return value

    divergences = np.asarray(statistic("diverging"), dtype=np.int64)
    draws_total = int(unit.chains * unit.draws)
    divergence_count = int(np.sum(divergences))
    if "sample_stat__reached_max_treedepth" in chains:
        saturated = np.asarray(statistic("reached_max_treedepth"), dtype=bool)
    else:
        saturated = np.asarray(statistic("tree_depth")) >= unit.max_treedepth

    hyper = chains[["group_location", "group_scale"]]
    group = chains[["group_effect"]]
    hyper_rhat = _dataarray_values(az.rhat(hyper, method="rank"))
    group_rhat = _dataarray_values(az.rhat(group, method="rank"))
    hyper_bulk = _dataarray_values(az.ess(hyper, method="bulk"))
    group_bulk = _dataarray_values(az.ess(group, method="bulk"))
    hyper_tail = _dataarray_values(az.ess(hyper, method="tail"))
    group_tail = _dataarray_values(az.ess(group, method="tail"))
    hyper_mcse = _dataarray_values(az.mcse(hyper, method="mean"))
    hyper_sd = np.asarray(
        [float(chains[name].std(dim=("chain", "draw"), ddof=1)) for name in hyper]
    )
    mcse_over_sd = np.divide(
        hyper_mcse,
        hyper_sd,
        out=np.full_like(hyper_mcse, np.inf),
        where=hyper_sd > 0,
    )
    try:
        bfmi_result = az.bfmi(np.asarray(statistic("energy"), dtype=np.float64))
        bfmi_values = np.asarray(bfmi_result, dtype=np.float64).reshape(-1)
    except Exception as error:  # ArviZ adapters differ across InferenceData/DataTree.
        raise ScientificCellFailure(
            "summarize",
            f"BFMI calculation failed: {error}",
            error_type=type(error).__name__,
        ) from error

    def optional_stat(name: str) -> list[float] | None:
        if f"sample_stat__{name}" not in chains:
            return None
        array = np.asarray(statistic(name), dtype=np.float64)
        return [float(value) for value in np.mean(array, axis=1)]

    step_size = np.asarray(statistic("step_size"), dtype=np.float64)
    n_steps = np.asarray(statistic("n_steps"), dtype=np.float64)
    if step_size.shape != (unit.chains, unit.draws):
        raise ScientificCellFailure("summarize", "step_size has the wrong shape")
    if n_steps.shape != (unit.chains, unit.draws):
        raise ScientificCellFailure("summarize", "n_steps has the wrong shape")
    result = {
        "wall_seconds": float(elapsed_seconds),
        "chains": unit.chains,
        "draws_per_chain": unit.draws,
        "draws_total": draws_total,
        "divergence_count": divergence_count,
        "divergence_count_by_chain": [
            int(value) for value in np.sum(divergences, axis=1)
        ],
        "divergence_rate": divergence_count / draws_total,
        "treedepth_saturation_count": int(np.sum(saturated)),
        "treedepth_saturation_rate": float(np.mean(saturated)),
        "hyper_rhat": hyper_rhat.tolist(),
        "hyper_rhat_max": float(np.max(hyper_rhat)),
        "group_rhat": group_rhat.tolist(),
        "group_rhat_max": float(np.max(group_rhat)),
        "hyper_ess_bulk": hyper_bulk.tolist(),
        "hyper_ess_bulk_min": float(np.min(hyper_bulk)),
        "group_ess_bulk": group_bulk.tolist(),
        "group_ess_bulk_fraction_ge_400": float(np.mean(group_bulk >= 400.0)),
        "hyper_ess_tail": hyper_tail.tolist(),
        "hyper_ess_tail_min": float(np.min(hyper_tail)),
        "group_ess_tail": group_tail.tolist(),
        "group_ess_tail_fraction_ge_400": float(np.mean(group_tail >= 400.0)),
        "bfmi_by_chain": bfmi_values.tolist(),
        "bfmi_min": float(np.min(bfmi_values)),
        "hyper_mcse_over_sd": mcse_over_sd.tolist(),
        "hyper_mcse_over_sd_max": float(np.max(mcse_over_sd)),
        "mean_step_size_by_chain": optional_stat("step_size"),
        "final_step_size_by_chain": [float(value) for value in step_size[:, -1]],
        "step_size_final_min": float(np.min(step_size[:, -1])),
        "step_size_final_max": float(np.max(step_size[:, -1])),
        "leapfrog_step_count": float(np.sum(n_steps)),
        "mean_acceptance_rate_by_chain": optional_stat("acceptance_rate"),
        "mean_n_steps_by_chain": optional_stat("n_steps"),
        "mean_energy_by_chain": optional_stat("energy"),
    }
    numeric_values = [
        float(item)
        for value in result.values()
        if value is not None
        for item in (value if isinstance(value, list) else [value])
        if not isinstance(item, bool)
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ScientificCellFailure(
            "summarize", "at least one sampler diagnostic is non-finite"
        )
    return result


def _model_arguments(model: pm.Model, vector: np.ndarray) -> list[np.ndarray]:
    """Split a canonical vector into arguments ordered like ``model.value_vars``."""
    point = _point_from_vector(model, vector)
    return [point[variable.name] for variable in model.value_vars]


def _make_graph_evaluator(
    model: pm.Model, backend_id: str
) -> Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]]:
    """Compile the exact scalar value/gradient/Hessian path used by a backend."""
    model_logp = model.logp(jacobian=True)
    if backend_id == "pymc":
        outputs: list[Any] = [
            model_logp,
            model.dlogp(jacobian=True),
            model.d2logp(jacobian=True, negate_output=False),
        ]
        function = model.compile_fn(
            outputs,
            inputs=model.value_vars,
            on_unused_input="ignore",
            point_fn=False,
        )

        def evaluate_pymc(
            vector: np.ndarray,
        ) -> tuple[float, np.ndarray, np.ndarray]:
            value, gradient, hessian = function(*_model_arguments(model, vector))
            return (
                float(value),
                np.asarray(gradient, dtype=np.float64),
                np.asarray(hessian, dtype=np.float64),
            )

        return evaluate_pymc
    if backend_id != "numpyro":
        raise CausalRunnerError(f"unknown backend {backend_id!r}")
    jaxified = get_jaxified_graph(
        inputs=model.value_vars, outputs=cast("Any", [model_logp])
    )
    initial = model.initial_point()
    layouts: list[tuple[int, int, tuple[int, ...]]] = []
    cursor = 0
    for variable in model.value_vars:
        shape = np.asarray(initial[variable.name]).shape
        size = int(np.prod(shape, dtype=int)) if shape else 1
        layouts.append((cursor, cursor + size, shape))
        cursor += size

    def scalar(vector: Any) -> Any:
        arguments = [
            vector[start:stop].reshape(shape) for start, stop, shape in layouts
        ]
        return jaxified(*arguments)[0]

    value_and_gradient = jax.jit(jax.value_and_grad(scalar))
    hessian_function = jax.jit(jax.hessian(scalar))

    def evaluate_numpyro(
        vector: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        point = jnp.asarray(vector)
        value, gradient = value_and_gradient(point)
        hessian = hessian_function(point)
        return (
            float(value),
            np.asarray(gradient, dtype=np.float64),
            np.asarray(hessian, dtype=np.float64),
        )

    return evaluate_numpyro


def _scaled_error(
    observed: np.ndarray | float,
    expected: np.ndarray | float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, float]:
    """Return raw and combined-tolerance errors for one diagnostic component."""
    observed_array = np.asarray(observed, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    difference = np.abs(observed_array - expected_array)
    scale = absolute_tolerance + relative_tolerance * np.maximum(
        np.abs(observed_array), np.abs(expected_array)
    )
    return {
        "absolute_max": float(np.max(difference)),
        "scaled_max": float(np.max(difference / scale)),
    }


def _oracle_tolerances(
    analysis_policy: Mapping[str, Any], floatx: str
) -> Mapping[str, Mapping[str, float]]:
    """Read the frozen component tolerances, rejecting an undefined gate."""
    gate = analysis_policy["oracle_gate"]
    tolerances = gate.get("component_tolerances")
    if not isinstance(tolerances, dict) or floatx not in tolerances:
        raise CausalRunnerError(
            f"manifest lacks oracle component tolerances for {floatx}"
        )
    result = tolerances[floatx]
    if set(result) != {"logp", "gradient", "hessian"}:
        raise CausalRunnerError("oracle tolerances must cover logp/gradient/Hessian")
    for component in result.values():
        if set(component) != {"absolute_tolerance", "relative_tolerance"}:
            raise CausalRunnerError("oracle component tolerance is malformed")
    return result


def _diagnostic_natural_points(
    unit: UnitSpec,
    prior: NativeTruncatedPrior,
    data: SyntheticHierarchyData,
    starts: Mapping[str, Any],
    chains: xr.Dataset | None,
    *,
    include_static: bool,
) -> list[dict[str, Any]]:
    """Select fixed, every-start, and hash-selected trajectory points."""
    points: list[dict[str, Any]] = []
    if include_static and unit.replicate == 0:
        points.append(
            {
                "point_id": "fixed-truth",
                "kind": "fixed-grid",
                "group_location": data.spec.group_location,
                "group_scale": data.spec.group_scale,
                "group_effect": [float(value) for value in data.group_effect],
            }
        )
    if include_static:
        for chain in starts["chains"]:
            points.append(
                {
                    "point_id": f"start-chain-{int(chain['chain']):02d}",
                    "kind": "shared-natural-start",
                    "group_location": chain["group_location"],
                    "group_scale": chain["group_scale"],
                    "group_effect": chain["group_effect"],
                }
            )
    if chains is not None:
        for chain in range(unit.chains):
            ranked = sorted(
                range(unit.draws),
                key=lambda draw: hashlib.sha256(
                    f"{unit.cell_id}:trajectory:{chain}:{draw}".encode()
                ).digest(),
            )
            for selection, draw in enumerate(ranked[:TRAJECTORY_POINTS_PER_CHAIN]):
                point_id = f"trajectory-chain-{chain:02d}-selection-{selection:02d}"
                points.append(
                    {
                        "point_id": point_id,
                        "kind": "hash-selected-posterior-trajectory",
                        "chain": chain,
                        "draw": draw,
                        "selection_sha256": hashlib.sha256(
                            f"{unit.cell_id}:trajectory:{chain}:{draw}".encode()
                        ).hexdigest(),
                        "group_location": float(chains["group_location"][chain, draw]),
                        "group_scale": float(chains["group_scale"][chain, draw]),
                        "group_effect": np.asarray(
                            chains["group_effect"][chain, draw], dtype=np.float64
                        ).tolist(),
                    }
                )
    parameterization = REPRESENTATION_TO_PARAMETERIZATION[unit.representation_id]
    nc_indices: list[int] = []
    if include_static and parameterization in {
        "location_icdf_noncentered",
        "full_icdf_noncentered",
    }:
        nc_indices.append(0)
    if include_static and parameterization in {
        "group_icdf_noncentered",
        "full_icdf_noncentered",
    }:
        nc_indices.append(2)
    if nc_indices:
        spec = _oracle_spec(unit, prior, data)
        base, _ = natural_to_coordinate(
            starts["chains"][0],
            prior=prior,
            oracle_spec=spec,
            representation_id=unit.representation_id,
        )
        branch_epsilon = 8.0 * math.sqrt(np.finfo(unit.floatx).eps)
        for coordinate_index in nc_indices:
            for label, value in (
                ("branch-left", -branch_epsilon),
                ("branch-zero", 0.0),
                ("branch-right", branch_epsilon),
                ("tail-low", -6.0),
                ("tail-high", 6.0),
            ):
                coordinate = base.copy()
                coordinate[coordinate_index] = value
                natural = hierarchical_natural_values(
                    coordinate, spec, parameterization
                )
                points.append(
                    {
                        "point_id": f"icdf-{coordinate_index}-{label}",
                        "kind": f"icdf-{label}",
                        "group_location": natural.location.value,
                        "group_scale": natural.scale.value,
                        "group_effect": [item.value for item in natural.group_effect],
                        "diagnostic_coordinate_vector": coordinate.tolist(),
                        "branch_epsilon": branch_epsilon,
                        "coordinate_index": coordinate_index,
                    }
                )
    return points


def _oracle_diagnostics(
    unit: UnitSpec,
    model: pm.Model,
    prior: NativeTruncatedPrior,
    data: SyntheticHierarchyData,
    starts: Mapping[str, Any],
    chains: xr.Dataset | None,
    analysis_policy: Mapping[str, Any],
    *,
    evaluator: Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]]
    | None = None,
    prior_records: Sequence[Mapping[str, Any]] = (),
    include_static: bool = True,
) -> dict[str, Any]:
    """Evaluate backend graphs against the independent oracle through order two."""
    parameterization = REPRESENTATION_TO_PARAMETERIZATION[unit.representation_id]
    spec = _oracle_spec(unit, prior, data)
    tolerances = _oracle_tolerances(analysis_policy, unit.floatx)
    graph_evaluator = (
        _make_graph_evaluator(model, unit.backend_id)
        if evaluator is None
        else evaluator
    )
    allowed_scaled_error = float(analysis_policy["oracle_gate"]["scaled_error_max"])
    roundtrip_limit = float(
        analysis_policy["oracle_gate"]["roundtrip_absolute_error_max"][unit.floatx]
    )
    failed_scaled_error = {
        "absolute_max": float(np.finfo(np.float64).max),
        "scaled_max": allowed_scaled_error + 1.0,
    }
    records = [dict(record) for record in prior_records]
    for point in _diagnostic_natural_points(
        unit, prior, data, starts, chains, include_static=include_static
    ):
        if "diagnostic_coordinate_vector" in point:
            candidate_vector = np.asarray(
                point["diagnostic_coordinate_vector"], dtype=np.float64
            )
        else:
            candidate_vector, _ = natural_to_coordinate(
                point,
                prior=prior,
                oracle_spec=spec,
                representation_id=unit.representation_id,
            )
        graph_point = _point_from_vector(model, candidate_vector)
        vector = _pack_model_point(model, graph_point)
        roundtrip = _natural_coordinate_roundtrip(
            point,
            vector,
            oracle_spec=spec,
            parameterization=parameterization,
        )
        observed_value, observed_gradient, observed_hessian = graph_evaluator(vector)
        expected = hierarchical_posterior_components(
            vector, spec, parameterization
        ).total
        expected_finite = bool(
            math.isfinite(expected.value)
            and np.all(np.isfinite(expected.gradient))
            and np.all(np.isfinite(expected.hessian))
        )
        if not expected_finite:
            raise CausalRunnerError("independent oracle is non-finite at a valid point")
        component_finite = {
            "value": math.isfinite(observed_value),
            "gradient": bool(np.all(np.isfinite(observed_gradient))),
            "hessian": bool(np.all(np.isfinite(observed_hessian))),
        }
        finite = all(component_finite.values())
        errors = {
            "value": (
                _scaled_error(
                    observed_value,
                    expected.value,
                    absolute_tolerance=float(tolerances["logp"]["absolute_tolerance"]),
                    relative_tolerance=float(tolerances["logp"]["relative_tolerance"]),
                )
                if component_finite["value"]
                else dict(failed_scaled_error)
            ),
            "gradient": (
                _scaled_error(
                    observed_gradient,
                    expected.gradient,
                    absolute_tolerance=float(
                        tolerances["gradient"]["absolute_tolerance"]
                    ),
                    relative_tolerance=float(
                        tolerances["gradient"]["relative_tolerance"]
                    ),
                )
                if component_finite["gradient"]
                else dict(failed_scaled_error)
            ),
            "hessian": (
                _scaled_error(
                    observed_hessian,
                    expected.hessian,
                    absolute_tolerance=float(
                        tolerances["hessian"]["absolute_tolerance"]
                    ),
                    relative_tolerance=float(
                        tolerances["hessian"]["relative_tolerance"]
                    ),
                )
                if component_finite["hessian"]
                else dict(failed_scaled_error)
            ),
        }
        passed = bool(
            finite
            and roundtrip["absolute_error_max"] <= roundtrip_limit
            and all(
                component["scaled_max"] <= allowed_scaled_error
                for component in errors.values()
            )
        )
        records.append(
            {
                **{
                    name: value
                    for name, value in point.items()
                    if name
                    in {
                        "point_id",
                        "kind",
                        "chain",
                        "draw",
                        "selection_sha256",
                        "branch_epsilon",
                        "coordinate_index",
                        "group_location",
                        "group_scale",
                        "group_effect",
                    }
                },
                "coordinate_vector": vector.tolist(),
                "roundtrip": roundtrip,
                "observed": {
                    "value": observed_value if component_finite["value"] else None,
                    "gradient": (
                        observed_gradient.tolist()
                        if component_finite["gradient"]
                        else None
                    ),
                    "hessian": (
                        observed_hessian.tolist()
                        if component_finite["hessian"]
                        else None
                    ),
                },
                "oracle": {
                    "value": expected.value,
                    "gradient": expected.gradient.tolist(),
                    "hessian": expected.hessian.tolist(),
                },
                "errors": errors,
                "component_finite": component_finite,
                "finite": finite,
                "passed": passed,
            }
        )
    tail_records = [
        record for record in records if record["kind"].startswith("icdf-tail")
    ]
    branch_checks: list[dict[str, Any]] = []
    coordinate_indices = sorted(
        {
            int(record["coordinate_index"])
            for record in records
            if record["kind"].startswith("icdf-branch")
        }
    )
    for coordinate_index in coordinate_indices:
        by_kind = {
            record["kind"]: record
            for record in records
            if record.get("coordinate_index") == coordinate_index
            and record["kind"].startswith("icdf-branch")
        }
        required_kinds = {
            "icdf-branch-left",
            "icdf-branch-zero",
            "icdf-branch-right",
        }
        if set(by_kind) != required_kinds:
            raise CausalRunnerError("ICDF branch diagnostic triplet is incomplete")
        left = by_kind["icdf-branch-left"]
        zero = by_kind["icdf-branch-zero"]
        right = by_kind["icdf-branch-right"]
        branch_finite = bool(left["finite"] and zero["finite"] and right["finite"])
        oracle_value_jump = right["oracle"]["value"] - left["oracle"]["value"]
        oracle_gradient_jump = np.asarray(
            right["oracle"]["gradient"], dtype=np.float64
        ) - np.asarray(left["oracle"]["gradient"], dtype=np.float64)
        if branch_finite:
            observed_value_jump = right["observed"]["value"] - left["observed"]["value"]
            observed_gradient_jump = np.asarray(
                right["observed"]["gradient"], dtype=np.float64
            ) - np.asarray(left["observed"]["gradient"], dtype=np.float64)
            value_jump_error = _scaled_error(
                observed_value_jump,
                oracle_value_jump,
                absolute_tolerance=float(tolerances["logp"]["absolute_tolerance"]),
                relative_tolerance=float(tolerances["logp"]["relative_tolerance"]),
            )
            gradient_jump_error = _scaled_error(
                observed_gradient_jump,
                oracle_gradient_jump,
                absolute_tolerance=float(tolerances["gradient"]["absolute_tolerance"]),
                relative_tolerance=float(tolerances["gradient"]["relative_tolerance"]),
            )
        else:
            observed_value_jump = None
            observed_gradient_jump = None
            value_jump_error = dict(failed_scaled_error)
            gradient_jump_error = dict(failed_scaled_error)
        branch_checks.append(
            {
                "coordinate_index": coordinate_index,
                "epsilon": left["branch_epsilon"],
                "left_point_id": left["point_id"],
                "zero_point_id": zero["point_id"],
                "right_point_id": right["point_id"],
                "observed_value_jump": observed_value_jump,
                "oracle_value_jump": oracle_value_jump,
                "observed_gradient_jump": (
                    None
                    if observed_gradient_jump is None
                    else observed_gradient_jump.tolist()
                ),
                "oracle_gradient_jump": oracle_gradient_jump.tolist(),
                "value_jump_error": value_jump_error,
                "gradient_jump_error": gradient_jump_error,
                "passed": bool(
                    left["passed"]
                    and zero["passed"]
                    and right["passed"]
                    and value_jump_error["scaled_max"] <= allowed_scaled_error
                    and gradient_jump_error["scaled_max"] <= allowed_scaled_error
                ),
            }
        )
    return {
        "backend_id": unit.backend_id,
        "representation_id": unit.representation_id,
        "parameterization": parameterization,
        "point_selection": {
            "fixed_grid": "replicate-zero-truth",
            "starts": "every-shared-natural-chain-start",
            "trajectory": "lowest-sha256-per-chain",
            "trajectory_points_per_chain": TRAJECTORY_POINTS_PER_CHAIN,
        },
        "posterior_trajectory_evaluated": chains is not None,
        "records": records,
        "icdf_tail_finite": all(record["finite"] for record in tail_records),
        "icdf_branch_checks": branch_checks,
        "icdf_branch_continuous": all(record["passed"] for record in branch_checks),
        "passed": all(record["passed"] for record in records)
        and all(record["passed"] for record in branch_checks),
    }


def _parameter_summaries(chains: xr.Dataset) -> list[dict[str, Any]]:
    """Summarize every natural parameter used for cross-form agreement."""
    summaries: list[dict[str, Any]] = []

    def append(parameter_id: str, values: xr.DataArray, index: int | None) -> None:
        flattened = np.asarray(values, dtype=np.float64).reshape(-1)
        mcse = az.mcse(values, method="mean")
        mcse_value = float(np.asarray(mcse).reshape(-1)[0])
        summaries.append(
            {
                "parameter_id": parameter_id,
                "index": index,
                "mean": float(np.mean(flattened)),
                "sd": float(np.std(flattened, ddof=1)),
                "mcse_mean": mcse_value,
            }
        )

    append("group_location", chains["group_location"], None)
    append("group_scale", chains["group_scale"], None)
    for index in range(chains.sizes["group"]):
        append("group_effect", chains["group_effect"].isel(group=index), index)
    if any(
        not math.isfinite(float(summary[field]))
        for summary in summaries
        for field in ("mean", "sd", "mcse_mean")
    ):
        raise ScientificCellFailure("summarize", "parameter summary is non-finite")
    return summaries


def _registered_metrics(
    unit: UnitSpec,
    raw: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> dict[str, Any]:
    """Project detailed evidence onto the contract's flat registered metrics."""
    result: dict[str, Any] = {
        "compile_success": True,
        "initialization_success": True,
        "logp_finite": True,
        "gradient_finite": True,
        "sampling_success": True,
        "divergence_count": raw["divergence_count"],
        "posterior_draw_count": raw["draws_total"],
        "divergence_rate": raw["divergence_rate"],
        "sampling_elapsed_seconds": raw["wall_seconds"],
        "step_size_final_min": raw["step_size_final_min"],
        "step_size_final_max": raw["step_size_final_max"],
        "leapfrog_step_count": raw["leapfrog_step_count"],
        **_registered_oracle_metrics(oracle),
    }
    if unit.tier == "confirmation":
        result.update(
            {
                "hyper_rhat_max": raw["hyper_rhat_max"],
                "hyper_ess_bulk_min": raw["hyper_ess_bulk_min"],
                "hyper_ess_tail_min": raw["hyper_ess_tail_min"],
                "bfmi_min": raw["bfmi_min"],
                "treedepth_saturation_rate": raw["treedepth_saturation_rate"],
                "hyper_mcse_over_sd_max": raw["hyper_mcse_over_sd_max"],
                "group_rhat_max": raw["group_rhat_max"],
                "group_ess_bulk_fraction_ge_400": raw["group_ess_bulk_fraction_ge_400"],
                "group_ess_tail_fraction_ge_400": raw["group_ess_tail_fraction_ge_400"],
            }
        )
    return result


def _registered_oracle_metrics(oracle: Mapping[str, Any]) -> dict[str, Any]:
    """Project deterministic oracle evidence without requiring sampler output."""
    records = oracle["records"]
    if not records:
        raise CausalRunnerError("oracle diagnostics contain no evaluation records")
    return {
        "oracle_evaluation_count": len(records),
        "oracle_logp_scaled_error_max": max(
            record["errors"]["value"]["scaled_max"] for record in records
        ),
        "oracle_gradient_scaled_error_max": max(
            record["errors"]["gradient"]["scaled_max"] for record in records
        ),
        "oracle_hessian_scaled_error_max": max(
            record["errors"]["hessian"]["scaled_max"] for record in records
        ),
        "roundtrip_absolute_error_max": max(
            record["roundtrip"]["absolute_error_max"] for record in records
        ),
        "icdf_tail_finite": oracle["icdf_tail_finite"],
        "icdf_branch_continuous": oracle["icdf_branch_continuous"],
    }


def _runtime_evidence() -> dict[str, Any]:
    """Capture observed child runtime state for every completed or failed cell."""
    return {
        "process_id": os.getpid(),
        "cache_identity_sha256": os.environ.get("HSSM_CAUSAL_CACHE_ID"),
        "pytensor_floatx": pytensor.config.floatX,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_platform": jax.default_backend(),
    }


def _diagnostics_payload(
    unit: UnitSpec,
    *,
    initial_points: Mapping[str, Any] | None = None,
    oracle: Mapping[str, Any] | None = None,
    sampler: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one identity-bound diagnostics payload with observed runtime state."""
    result: dict[str, Any] = {
        "schema_version": unit.schema_version,
        "study_id": unit.study_id,
        "manifest_sha256": unit.manifest_sha256,
        "cell_id": unit.cell_id,
        "runtime": _runtime_evidence(),
    }
    if initial_points is not None:
        result["initial_points"] = dict(initial_points)
    if oracle is not None:
        result["oracle"] = dict(oracle)
    if sampler is not None:
        result["sampler"] = dict(sampler)
    return result


def _validate_execution_oracle_coherence(
    unit: UnitSpec, execution: CellExecution
) -> None:
    """Bind every registered oracle claim to the retained diagnostic phase."""
    metric_names = set(execution.metrics or {})
    observed_metric_names = metric_names.intersection(ORACLE_METRICS)
    if observed_metric_names and observed_metric_names != ORACLE_METRICS:
        raise CausalRunnerError("cell contains a partial registered oracle metric set")
    oracle = (
        execution.diagnostics.get("oracle")
        if isinstance(execution.diagnostics, Mapping)
        else None
    )
    has_metrics = observed_metric_names == ORACLE_METRICS
    if (oracle is not None) != has_metrics:
        raise CausalRunnerError(
            "retained raw oracle diagnostics and registered metrics disagree"
        )
    if oracle is None:
        return
    if not isinstance(oracle, Mapping) or not isinstance(oracle.get("records"), list):
        raise CausalRunnerError("retained oracle diagnostics are malformed")
    trajectory = oracle.get("posterior_trajectory_evaluated")
    if not isinstance(trajectory, bool):
        raise CausalRunnerError("retained oracle phase is malformed")
    layers = {
        "native-centered": 0,
        "manual-centered": 0,
        "group-icdf-noncentered": 1,
        "location-icdf-noncentered": 1,
        "full-icdf-noncentered": 2,
    }[unit.representation_id]
    expected_count = (
        unit.chains
        + int(unit.replicate == 0)
        + 5 * layers
        + (TRAJECTORY_POINTS_PER_CHAIN * unit.chains if trajectory else 0)
    )
    if (
        len(oracle["records"]) != expected_count
        or (execution.metrics or {}).get("oracle_evaluation_count") != expected_count
    ):
        raise CausalRunnerError("retained oracle phase has the wrong evaluation count")
    if execution.status == "completed":
        if not trajectory:
            raise CausalRunnerError("completed cell lacks trajectory oracle evidence")
        return
    stage = (execution.failure or {}).get("stage")
    if not isinstance(stage, str):
        raise CausalRunnerError("failed cell lacks a classified oracle stage")
    expected_trajectory = {
        "compile": False,
        "sample": False,
        "diagnose": False,
        "summarize": True,
    }.get(stage)
    if expected_trajectory is None or trajectory is not expected_trajectory:
        raise CausalRunnerError(
            "failed cell retains an oracle phase inconsistent with its stage"
        )


def execute_cell(
    unit: UnitSpec,
    data_payload: Mapping[str, Any],
    start_payload: Mapping[str, Any],
    analysis_policy: Mapping[str, Any],
) -> CellExecution:
    """Build, initialize, sample, summarize, and diagnose one planned cell."""
    try:
        data = _data_from_payload(data_payload, unit)
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ScientificCellFailure(
            "data", str(error), error_type=type(error).__name__
        ) from error
    prior = _prior(unit)
    _validate_natural_start_payload(start_payload, unit, prior, data)
    try:
        builder = BUILDER_TO_PARAMETERIZATION[unit.builder]
    except KeyError as error:
        raise CausalRunnerError(f"unsupported builder {unit.builder!r}") from error
    try:
        model = build_causal_model(builder, prior, data)
    except (
        FloatingPointError,
        np.linalg.LinAlgError,
        NotImplementedError,
        pm.exceptions.DtypeError,
        pm.exceptions.ShapeError,
        pm.exceptions.TruncationError,
    ) as error:
        raise ScientificCellFailure(
            "build", str(error), error_type=type(error).__name__
        ) from error
    oracle_spec = _oracle_spec(unit, prior, data)
    coordinate_payload, initvals = _coordinate_start_payload(
        unit, model, start_payload, prior, oracle_spec
    )
    roundtrip_limit = float(
        analysis_policy["oracle_gate"]["roundtrip_absolute_error_max"][unit.floatx]
    )
    if any(
        chain["roundtrip"]["absolute_error_max"] > roundtrip_limit
        for chain in coordinate_payload["chains"]
    ):
        failure = ScientificCellFailure(
            "initialize", "natural-coordinate-natural round-trip exceeded tolerance"
        )
        return CellExecution(
            status="failed",
            coordinate_starts=coordinate_payload,
            diagnostics=_diagnostics_payload(unit),
            metrics={"initialization_success": False},
            parameter_summaries=[],
            failure=failure.as_dict(),
        )
    try:
        initial_evidence = _finite_initial_evidence(model, initvals)
    except ScientificCellFailure as failure:
        metrics = (
            {"compile_success": False}
            if failure.stage == "compile"
            else {"compile_success": True, "initialization_success": False}
        )
        return CellExecution(
            status="failed",
            coordinate_starts=coordinate_payload,
            diagnostics=_diagnostics_payload(unit),
            metrics=metrics,
            parameter_summaries=[],
            failure=failure.as_dict(),
        )
    try:
        evaluator = _make_graph_evaluator(model, unit.backend_id)
        pre_oracle = _oracle_diagnostics(
            unit,
            model,
            prior,
            data,
            start_payload,
            None,
            analysis_policy,
            evaluator=evaluator,
        )
    except (CompileError, NotImplementedError) as error:
        classified_failure = ScientificCellFailure(
            "compile", str(error), error_type=type(error).__name__
        )
        return CellExecution(
            status="failed",
            coordinate_starts=coordinate_payload,
            diagnostics=_diagnostics_payload(unit, initial_points=initial_evidence),
            metrics={
                "compile_success": False,
                "initialization_success": True,
                "logp_finite": True,
                "gradient_finite": True,
            },
            parameter_summaries=[],
            failure=classified_failure.as_dict(),
        )
    except ScientificCellFailure as error:
        classified_failure = ScientificCellFailure(
            "initialize", str(error), error_type=error.error_type
        )
        return CellExecution(
            status="failed",
            coordinate_starts=coordinate_payload,
            diagnostics=_diagnostics_payload(unit, initial_points=initial_evidence),
            metrics={
                "compile_success": True,
                "initialization_success": False,
                "logp_finite": True,
                "gradient_finite": True,
            },
            parameter_summaries=[],
            failure=classified_failure.as_dict(),
        )
    pre_oracle_metrics = _registered_oracle_metrics(pre_oracle)
    pre_diagnostics = _diagnostics_payload(
        unit,
        initial_points=initial_evidence,
        oracle=pre_oracle,
    )
    try:
        inference, elapsed = _sample_model(unit, model, initvals)
    except ScientificCellFailure as failure:
        metrics = (
            {
                "compile_success": False,
                "initialization_success": True,
                "logp_finite": True,
                "gradient_finite": True,
                **pre_oracle_metrics,
            }
            if failure.stage == "compile"
            else {
                "compile_success": True,
                "initialization_success": True,
                "logp_finite": True,
                "gradient_finite": True,
                "sampling_success": False,
                **pre_oracle_metrics,
            }
        )
        return CellExecution(
            status="failed",
            coordinate_starts=coordinate_payload,
            diagnostics=pre_diagnostics,
            metrics=metrics,
            parameter_summaries=[],
            failure=failure.as_dict(),
        )
    try:
        natural_chains = _standardize_natural_chains(unit, inference)
    except ScientificCellFailure as error:
        # The sampler has not produced the experiment's canonical chain artifact
        # until this projection succeeds.  A malformed or non-finite posterior is
        # therefore a sample-stage failure, not a summarize-stage failure with a
        # missing chain.
        classified_failure = ScientificCellFailure(
            "sample", str(error), error_type=error.error_type
        )
        return CellExecution(
            status="failed",
            coordinate_starts=coordinate_payload,
            diagnostics=pre_diagnostics,
            metrics={
                "compile_success": True,
                "initialization_success": True,
                "logp_finite": True,
                "gradient_finite": True,
                "sampling_success": False,
                "sampling_elapsed_seconds": elapsed,
                **pre_oracle_metrics,
            },
            parameter_summaries=[],
            failure=classified_failure.as_dict(),
        )
    try:
        oracle = _oracle_diagnostics(
            unit,
            model,
            prior,
            data,
            start_payload,
            natural_chains,
            analysis_policy,
            evaluator=evaluator,
            prior_records=pre_oracle["records"],
            include_static=False,
        )
    except (CompileError, NotImplementedError, ScientificCellFailure) as error:
        error_type = (
            error.error_type
            if isinstance(error, ScientificCellFailure)
            else type(error).__name__
        )
        classified_failure = ScientificCellFailure(
            "diagnose", str(error), error_type=error_type
        )
        return CellExecution(
            status="failed",
            coordinate_starts=coordinate_payload,
            chain_dataset=natural_chains,
            diagnostics=pre_diagnostics,
            metrics={
                "compile_success": True,
                "initialization_success": True,
                "logp_finite": True,
                "gradient_finite": True,
                "sampling_success": True,
                "sampling_elapsed_seconds": elapsed,
                **pre_oracle_metrics,
            },
            parameter_summaries=[],
            failure=classified_failure.as_dict(),
        )
    try:
        raw_metrics = _sampler_metrics(unit, natural_chains, elapsed)
    except ScientificCellFailure as failure:
        return CellExecution(
            status="failed",
            coordinate_starts=coordinate_payload,
            chain_dataset=natural_chains,
            diagnostics=_diagnostics_payload(
                unit, initial_points=initial_evidence, oracle=oracle
            ),
            metrics={
                "compile_success": True,
                "initialization_success": True,
                "logp_finite": True,
                "gradient_finite": True,
                "sampling_success": True,
                "sampling_elapsed_seconds": elapsed,
                **_registered_oracle_metrics(oracle),
            },
            parameter_summaries=[],
            failure=failure.as_dict(),
        )
    diagnostics = _diagnostics_payload(
        unit,
        initial_points=initial_evidence,
        oracle=oracle,
        sampler=raw_metrics,
    )
    try:
        summaries = _parameter_summaries(natural_chains)
    except ScientificCellFailure as failure:
        return CellExecution(
            status="failed",
            coordinate_starts=coordinate_payload,
            chain_dataset=natural_chains,
            diagnostics=diagnostics,
            metrics=_registered_metrics(unit, raw_metrics, oracle),
            parameter_summaries=[],
            failure=failure.as_dict(),
        )
    metrics = _registered_metrics(unit, raw_metrics, oracle)
    return CellExecution(
        status="completed",
        coordinate_starts=coordinate_payload,
        chain_dataset=natural_chains,
        diagnostics=diagnostics,
        metrics=metrics,
        parameter_summaries=summaries,
    )


def mint_run_context(
    manifest: Mapping[str, Any],
    units: Sequence[UnitSpec],
    *,
    worker_identity: str,
    environment: Mapping[str, Any] | None = None,
    expected_git_commit: str | None = None,
) -> tuple[RunContext, Mapping[str, Any]]:
    """Validate the runtime and mint one parent identity for ten child attempts."""
    if len(units) != 10 or len({unit.pair_id for unit in units}) != 1:
        raise CausalRunnerError(
            "a run context requires exactly one ten-cell backend pair"
        )
    if not isinstance(worker_identity, str) or not worker_identity.strip():
        raise CausalRunnerError("worker identity must be a non-empty opaque string")
    record = collect_environment(manifest) if environment is None else environment
    validate_environment(record, manifest)
    git_commit = str(record["git"]["commit"])
    if expected_git_commit is not None and git_commit != expected_git_commit:
        raise CausalRunnerError("runtime git commit differs from the expected commit")
    worker_digest = sha256_bytes(worker_identity.encode("utf-8"))
    pair_execution_id = sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "hssm-causal-pair-execution-v3",
                "pair_id": units[0].pair_id,
                "manifest_sha256": manifest_digest(manifest),
                "environment_sha256": environment_digest(record),
                "git_commit": git_commit,
                "worker_identity_sha256": worker_digest,
            }
        )
    )
    attempt_ids = tuple(
        sha256_bytes(
            canonical_json_bytes(
                {
                    "domain": "hssm-causal-cell-attempt-v3",
                    "pair_execution_id": pair_execution_id,
                    "cell_id": unit.cell_id,
                    "position": unit.pair_position,
                }
            )
        )
        for unit in units
    )
    context = RunContext(
        schema_version=units[0].schema_version,
        study_id=units[0].study_id,
        manifest_sha256=units[0].manifest_sha256,
        pair_id=units[0].pair_id,
        block_ids=tuple(dict.fromkeys(unit.block_id for unit in units)),
        cell_ids=tuple(unit.cell_id for unit in units),
        execution_order=tuple(unit.cell_id for unit in units),
        environment=dict(record),
        environment_sha256=environment_digest(record),
        git_commit=git_commit,
        worker_identity_sha256=worker_digest,
        pair_execution_id=pair_execution_id,
        execution_attempt_ids=attempt_ids,
    )
    validate_run_context(context, units, manifest)
    return context, record


def _netcdf_bytes(dataset: xr.Dataset) -> bytes:
    """Serialize a standardized natural chain to portable NetCDF3 bytes."""
    payload = dataset.to_netcdf(path=None, engine="scipy", format="NETCDF3_64BIT")
    return bytes(payload)


def _backend_metadata(
    manifest: Mapping[str, Any], backend_id: str
) -> Mapping[str, Any]:
    matches = [
        item for item in manifest["backends"] if item["backend_id"] == backend_id
    ]
    if len(matches) != 1:
        raise CausalRunnerError(f"manifest backend {backend_id!r} is not unique")
    return matches[0]


def _result_provenance(
    unit: UnitSpec,
    context: RunContext,
    manifest: Mapping[str, Any],
    execution: CellExecution,
) -> dict[str, Any]:
    backend = _backend_metadata(manifest, unit.backend_id)
    sampler_seed_input, chain_rng_provenance = _sampler_rng_provenance(unit)
    if execution.diagnostics is None:
        raise CausalRunnerError("cell lacks observed runtime diagnostics")
    runtime = execution.diagnostics.get("runtime")
    if not isinstance(runtime, Mapping):
        raise CausalRunnerError("cell runtime diagnostics are malformed")
    required_runtime = {
        "process_id",
        "cache_identity_sha256",
        "pytensor_floatx",
        "jax_enable_x64",
        "jax_platform",
    }
    if set(runtime) != required_runtime:
        raise CausalRunnerError("cell runtime diagnostics have unexpected fields")
    if runtime["pytensor_floatx"] != unit.floatx:
        raise CausalRunnerError("cell runtime PyTensor precision differs from plan")
    if runtime["jax_enable_x64"] is not (unit.floatx == "float64"):
        raise CausalRunnerError("cell runtime JAX precision differs from plan")
    if runtime["jax_platform"] != backend["device"]:
        raise CausalRunnerError("cell runtime device differs from plan")
    cache_identity = runtime["cache_identity_sha256"]
    if (
        not isinstance(cache_identity, str)
        or len(cache_identity) != 64
        or any(character not in "0123456789abcdef" for character in cache_identity)
    ):
        raise CausalRunnerError("cell runtime lacks a valid cache identity")
    position = context.cell_ids.index(unit.cell_id)
    return {
        "environment_sha256": context.environment_sha256,
        "git_commit": context.git_commit,
        "worker_identity_sha256": context.worker_identity_sha256,
        "pair_execution_id": context.pair_execution_id,
        "execution_attempt_id": context.execution_attempt_ids[position],
        "sampler": backend["sampler"],
        "compiler_path": backend["compiler_path"],
        "device": runtime["jax_platform"],
        "floatx": unit.floatx,
        "pytensor_floatx": runtime["pytensor_floatx"],
        "jax_enable_x64": runtime["jax_enable_x64"],
        "sampler_seed_input": sampler_seed_input,
        "chain_rng_provenance": chain_rng_provenance,
    }


def publish_cell_execution(
    unit: UnitSpec,
    context: RunContext,
    manifest: Mapping[str, Any],
    store: ArtifactStore,
    execution: CellExecution,
    *,
    context_reference: ArtifactRef,
    data_reference: ArtifactRef,
    natural_start_reference: ArtifactRef,
) -> ArtifactRef:
    """Publish all evidence first and the validated result JSON last."""
    _validate_execution_oracle_coherence(unit, execution)
    if execution.chain_dataset is not None:
        _validate_standardized_natural_chains(unit, execution.chain_dataset)
    coordinate_reference = (
        store.write_json(
            f"starts/coordinates/{unit.cell_id}.json", execution.coordinate_starts
        )
        if execution.coordinate_starts is not None
        else None
    )
    chain_reference = (
        store.write_bytes(
            f"chains/{unit.cell_id}.nc", _netcdf_bytes(execution.chain_dataset)
        )
        if execution.chain_dataset is not None
        else None
    )
    diagnostic_reference = (
        store.write_json(f"diagnostics/{unit.cell_id}.json", execution.diagnostics)
        if execution.diagnostics is not None
        else None
    )
    artifacts = {
        "context": context_reference.as_dict(),
        "data": data_reference.as_dict(),
        "natural_start": natural_start_reference.as_dict(),
        "coordinate_start": (
            None if coordinate_reference is None else coordinate_reference.as_dict()
        ),
        "chain": None if chain_reference is None else chain_reference.as_dict(),
        "diagnostics": (
            None if diagnostic_reference is None else diagnostic_reference.as_dict()
        ),
    }
    record = {
        "schema_version": unit.schema_version,
        "runner_version": RUNNER_VERSION,
        "study_id": unit.study_id,
        "manifest_sha256": unit.manifest_sha256,
        "tier": unit.tier,
        "regime_id": unit.regime_id,
        "backend_id": unit.backend_id,
        "representation_id": unit.representation_id,
        "replicate": unit.replicate,
        "pair_id": unit.pair_id,
        "pair_position": unit.pair_position,
        "block_id": unit.block_id,
        "block_position": unit.block_position,
        "cell_id": unit.cell_id,
        "execution_status": execution.status,
        "metrics": dict(execution.metrics or {}),
        "parameter_summaries": [
            dict(summary) for summary in (execution.parameter_summaries or [])
        ],
        "artifacts": artifacts,
        "failure": None if execution.failure is None else dict(execution.failure),
        "provenance": _result_provenance(unit, context, manifest, execution),
    }
    validate_result_record(record, unit, context)
    return store.write_json(f"cells/{unit.cell_id}.json", record)


def _write_staged_execution(directory: Path, execution: CellExecution) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    files: dict[str, str | None] = {
        "coordinate_starts": None,
        "chain": None,
        "diagnostics": None,
    }
    if execution.coordinate_starts is not None:
        (directory / "coordinate-start.json").write_bytes(
            canonical_json_bytes(execution.coordinate_starts)
        )
        files["coordinate_starts"] = "coordinate-start.json"
    if execution.chain_dataset is not None:
        (directory / "chain.nc").write_bytes(_netcdf_bytes(execution.chain_dataset))
        files["chain"] = "chain.nc"
    if execution.diagnostics is not None:
        (directory / "diagnostics.json").write_bytes(
            canonical_json_bytes(execution.diagnostics)
        )
        files["diagnostics"] = "diagnostics.json"
    outcome = {
        "status": execution.status,
        "metrics": dict(execution.metrics or {}),
        "parameter_summaries": [
            dict(summary) for summary in (execution.parameter_summaries or [])
        ],
        "failure": None if execution.failure is None else dict(execution.failure),
        "files": files,
    }
    (directory / "outcome.json").write_bytes(canonical_json_bytes(outcome))


def _load_staged_execution(directory: Path) -> CellExecution:
    outcome = decode_canonical_json((directory / "outcome.json").read_bytes())
    files = outcome["files"]

    def json_file(name: str) -> Any | None:
        relative = files[name]
        return (
            None
            if relative is None
            else decode_canonical_json((directory / relative).read_bytes())
        )

    chain = (
        None
        if files["chain"] is None
        else xr.load_dataset(directory / files["chain"], engine="scipy")
    )
    return CellExecution(
        status=outcome["status"],
        coordinate_starts=json_file("coordinate_starts"),
        chain_dataset=chain,
        diagnostics=json_file("diagnostics"),
        metrics=outcome["metrics"],
        parameter_summaries=outcome["parameter_summaries"],
        failure=outcome["failure"],
    )


def _load_context(
    path: Path, units: Sequence[UnitSpec], manifest: Mapping[str, Any]
) -> RunContext:
    """Load a canonical context and bind it to the exact ten planned units."""
    try:
        payload = decode_canonical_json(path.read_bytes())
    except OSError as error:
        raise CausalRunnerError(f"cannot read parent context {path}") from error
    if not isinstance(payload, Mapping):
        raise CausalRunnerError("parent context must be a JSON object")
    context = RunContext.from_dict(payload)
    return validate_run_context(context, units, manifest)


def _private_sample_cell(
    manifest_path: Path,
    tier: str,
    cell_id: str,
    run_dir: Path,
    context_file: Path,
    output_directory: Path,
) -> int:
    """Private fresh-process entry point; write staging, never final evidence."""
    manifest = load_manifest(manifest_path)
    unit = plan_unit_by_id(manifest, tier, cell_id)
    units = pair_units(manifest, tier, unit.pair_id)
    _load_context(context_file, units, manifest)
    store = ArtifactStore(run_dir)
    try:
        data_reference, start_reference = materialize_inputs_for_unit(
            unit, store, manifest_path=manifest_path
        )
        data_payload = store.read_json(data_reference)
        start_payload = store.read_json(start_reference)
        execution = execute_cell(
            unit, data_payload, start_payload, manifest["analysis_policy"]
        )
    except ScientificCellFailure as failure:
        execution = CellExecution(
            status="failed",
            diagnostics=_diagnostics_payload(unit),
            metrics={},
            parameter_summaries=[],
            failure=failure.as_dict(),
        )
    _write_staged_execution(output_directory, execution)
    return 0 if execution.status == "completed" else 10


def _private_materialize_unit(
    manifest_path: Path,
    tier: str,
    cell_id: str,
    run_dir: Path,
) -> int:
    """Private precision-activated input materializer."""
    manifest = load_manifest(manifest_path)
    unit = plan_unit_by_id(manifest, tier, cell_id)
    data_reference, start_reference = _materialize_inputs_for_unit_current_process(
        unit, ArtifactStore(run_dir)
    )
    print(
        canonical_json_bytes(
            {
                "data": data_reference.as_dict(),
                "natural_start": start_reference.as_dict(),
            }
        ).decode(),
        end="",
    )
    return 0


def _process_resource_snapshot(child_pid: int) -> dict[str, Any]:
    """Return best-effort process and host telemetry for one active child."""
    try:
        import psutil
    except ImportError:
        return {
            "resource_status": "unavailable",
            "resource_error_types": ["ImportError"],
        }

    error_types: set[str] = set()
    snapshot: dict[str, Any] = {}
    try:
        snapshot["parent_rss_bytes"] = psutil.Process(os.getpid()).memory_info().rss
    except (psutil.Error, OSError) as error:
        error_types.add(type(error).__name__)

    processes: list[Any] = []
    try:
        child = psutil.Process(child_pid)
        processes.append(child)
        try:
            processes.extend(child.children(recursive=True))
        except (psutil.Error, OSError) as error:
            error_types.add(type(error).__name__)
    except (psutil.Error, OSError) as error:
        error_types.add(type(error).__name__)

    process_count = 0
    rss_bytes = 0
    vms_bytes = 0
    thread_count = 0
    cpu_user_seconds = 0.0
    cpu_system_seconds = 0.0
    for process in processes:
        try:
            with process.oneshot():
                memory = process.memory_info()
                cpu = process.cpu_times()
                threads = process.num_threads()
        except (psutil.Error, OSError) as error:
            error_types.add(type(error).__name__)
            continue
        process_count += 1
        rss_bytes += int(memory.rss)
        vms_bytes += int(memory.vms)
        thread_count += int(threads)
        cpu_user_seconds += float(cpu.user)
        cpu_system_seconds += float(cpu.system)
    snapshot.update(
        {
            "child_process_count": process_count,
            "child_tree_rss_bytes": rss_bytes,
            "child_tree_vms_bytes": vms_bytes,
            "child_tree_thread_count": thread_count,
            "child_tree_cpu_user_seconds": cpu_user_seconds,
            "child_tree_cpu_system_seconds": cpu_system_seconds,
        }
    )

    try:
        host_memory = psutil.virtual_memory()
        host_swap = psutil.swap_memory()
        snapshot.update(
            {
                "host_available_memory_bytes": int(host_memory.available),
                "host_memory_percent": float(host_memory.percent),
                "host_swap_used_bytes": int(host_swap.used),
            }
        )
    except (psutil.Error, OSError) as error:
        error_types.add(type(error).__name__)

    snapshot["resource_status"] = "partial" if error_types else "available"
    snapshot["resource_error_types"] = sorted(error_types)
    return snapshot


def _emit_cell_observation(unit: UnitSpec, event: str, **values: Any) -> None:
    """Write one fail-soft operational record outside the evidence contract."""
    payload = {
        "event": event,
        "cell_id": unit.cell_id,
        "pair_id": unit.pair_id,
        "pair_position": unit.pair_position,
        "tier": unit.tier,
        "regime_id": unit.regime_id,
        "replicate": unit.replicate,
        "backend_id": unit.backend_id,
        "representation_id": unit.representation_id,
        **values,
    }
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        print(CELL_OBSERVATION_PREFIX + encoded, file=sys.stderr, flush=True)
    except Exception:
        # Broken log pipes and telemetry serialization must never alter a cell.
        return


def _kill_cell_process_tree(process: subprocess.Popen[str]) -> None:
    """Best-effort kill of the isolated child session or non-POSIX tree."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except OSError:
            pass
    else:
        try:
            import psutil

            root = psutil.Process(process.pid)
            descendants = root.children(recursive=True)
            for descendant in reversed(descendants):
                try:
                    descendant.kill()
                except (psutil.Error, OSError):
                    pass
        except (ImportError, OSError):
            pass
        except Exception:
            # psutil is optional; direct-child cleanup remains the fallback.
            pass
    try:
        process.kill()
    except OSError:
        pass


def _cleanup_interrupted_cell_process(process: subprocess.Popen[str]) -> None:
    """Bound cleanup so inherited pipes cannot mask the original interrupt."""
    try:
        _kill_cell_process_tree(process)
    except BaseException:
        pass
    try:
        process.communicate(timeout=CELL_CLEANUP_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    except BaseException:
        pass

    for pipe in (process.stdout, process.stderr):
        if pipe is None:
            continue
        try:
            pipe.close()
        except BaseException:
            pass
    try:
        process.kill()
    except BaseException:
        pass
    try:
        process.wait(timeout=CELL_CLEANUP_SECONDS)
    except BaseException:
        pass


def _run_observed_cell_process(
    unit: UnitSpec,
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    heartbeat_seconds: float = CELL_HEARTBEAT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run one captured child while emitting fail-soft operational heartbeats."""
    if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0.0:
        raise ValueError("heartbeat_seconds must be finite and positive")
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
        creationflags=(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        ),
    )
    try:
        _emit_cell_observation(
            unit,
            "cell-start",
            child_pid=process.pid,
            elapsed_seconds=0.0,
        )
        while True:
            try:
                stdout, stderr = process.communicate(timeout=heartbeat_seconds)
                break
            except subprocess.TimeoutExpired:
                try:
                    resources = _process_resource_snapshot(process.pid)
                except Exception as error:
                    resources = {
                        "resource_status": "unavailable",
                        "resource_error_types": [type(error).__name__],
                    }
                _emit_cell_observation(
                    unit,
                    "cell-heartbeat",
                    child_pid=process.pid,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    **resources,
                )
    except BaseException:
        _cleanup_interrupted_cell_process(process)
        try:
            _emit_cell_observation(
                unit,
                "cell-end",
                child_pid=process.pid,
                elapsed_seconds=round(time.monotonic() - started, 3),
                returncode=process.returncode,
                termination="observer-interrupted",
            )
        except BaseException:
            pass
        raise

    returncode = cast("int", process.returncode)
    _emit_cell_observation(
        unit,
        "cell-end",
        child_pid=process.pid,
        elapsed_seconds=round(time.monotonic() - started, 3),
        returncode=returncode,
        termination="child-exited",
    )
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _subprocess_cell_executor(
    unit: UnitSpec,
    *,
    manifest_path: Path,
    run_dir: Path,
    context_path: Path,
    context: RunContext,
) -> CellExecution:
    """Run one cell in a fresh interpreter and uniquely scoped compiler caches."""
    position = context.cell_ids.index(unit.cell_id)
    cache_identity = sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "hssm-causal-fresh-cache-v3",
                "attempt": context.execution_attempt_ids[position],
            }
        )
    )
    with tempfile.TemporaryDirectory(prefix="hssm-causal-cell-") as temporary_name:
        temporary = Path(temporary_name)
        output_directory = temporary / "result"
        cache = temporary / "cache"
        (cache / "pytensor").mkdir(parents=True)
        (cache / "jax").mkdir(parents=True)
        (cache / "matplotlib").mkdir(parents=True)
        (cache / "xdg").mkdir(parents=True)
        environment = os.environ.copy()
        environment.update(
            {
                "HSSM_CAUSAL_CACHE_ID": cache_identity,
                "PYTENSOR_FLAGS": (
                    f"base_compiledir={cache / 'pytensor'},floatX={unit.floatx}"
                ),
                "JAX_COMPILATION_CACHE_DIR": str(cache / "jax"),
                "JAX_ENABLE_X64": "true" if unit.floatx == "float64" else "false",
                "JAX_PLATFORMS": "cpu",
                "MPLCONFIGDIR": str(cache / "matplotlib"),
                "XDG_CACHE_HOME": str(cache / "xdg"),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        command = [
            sys.executable,
            "-m",
            "scripts.truncated_hierarchy_causal_runner",
            "--manifest",
            str(manifest_path),
            "_sample-cell",
            "--tier",
            unit.tier,
            "--cell-id",
            unit.cell_id,
            "--run-dir",
            str(run_dir),
            "--context",
            str(context_path),
            "--output-dir",
            str(output_directory),
        ]
        completed = _run_observed_cell_process(
            unit,
            command,
            cwd=Path(__file__).resolve().parents[1],
            environment=environment,
        )
        if completed.returncode not in {0, 10}:
            stderr = completed.stderr[-4000:]
            message = (
                f"fresh child for {unit.cell_id} exited "
                f"{completed.returncode}: {stderr}"
            )
            raise IncompleteBlockError(message)
        execution = _load_staged_execution(output_directory)
        expected_status = "completed" if completed.returncode == 0 else "failed"
        if execution.status != expected_status:
            raise IncompleteBlockError("child exit status contradicts staged outcome")
        runtime = (
            execution.diagnostics.get("runtime", {})
            if execution.diagnostics is not None
            else {}
        )
        if runtime.get("cache_identity_sha256") != cache_identity:
            raise IncompleteBlockError("child did not attest its unique compiler cache")
        if runtime.get("pytensor_floatx") != unit.floatx:
            raise IncompleteBlockError("child attested the wrong PyTensor floatX")
        if runtime.get("jax_enable_x64") is not (unit.floatx == "float64"):
            raise IncompleteBlockError("child attested the wrong JAX precision")
        if runtime.get("jax_platform") != "cpu":
            raise IncompleteBlockError("child did not attest the required CPU backend")
        return execution


def run_pair(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    units: Sequence[UnitSpec],
    store: ArtifactStore,
    context: RunContext,
    *,
    context_reference: ArtifactRef,
    executor: Callable[[UnitSpec], CellExecution] | None = None,
) -> tuple[ArtifactRef, ...]:
    """Attempt all ten paired members, then publish independent final markers."""
    validate_run_context(context, units, manifest)
    representative = units[0]
    data_reference, start_reference = materialize_inputs_for_unit(
        representative, store, manifest_path=manifest_path
    )
    executions: list[CellExecution] = []
    for unit in units:
        try:
            execution = (
                _subprocess_cell_executor(
                    unit,
                    manifest_path=manifest_path,
                    run_dir=store.root,
                    context_path=store.root / context_reference.path,
                    context=context,
                )
                if executor is None
                else executor(unit)
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            raise IncompleteBlockError(
                f"infrastructure aborted {unit.cell_id}; no cell markers published"
            ) from error
        executions.append(execution)
    if len(executions) != 10:
        raise IncompleteBlockError("pair did not attempt all ten planned cells")
    return tuple(
        publish_cell_execution(
            unit,
            context,
            manifest,
            store,
            execution,
            context_reference=context_reference,
            data_reference=data_reference,
            natural_start_reference=start_reference,
        )
        for unit, execution in zip(units, executions, strict=True)
    )


def run_unit_parent(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    tier: str,
    pair_id: str,
    run_dir: Path,
    worker_identity: str,
    expected_git_commit: str | None = None,
) -> tuple[tuple[ArtifactRef, ...], bool]:
    """Mint context and execute the selected backend-paired unit."""
    units = pair_units(manifest, tier, pair_id)
    context, _environment = mint_run_context(
        manifest,
        units,
        worker_identity=worker_identity,
        expected_git_commit=expected_git_commit,
    )
    store = ArtifactStore(run_dir)
    context_reference = store.write_json(f"contexts/{pair_id}.json", context.as_dict())
    results = run_pair(
        manifest,
        manifest_path,
        units,
        store,
        context,
        context_reference=context_reference,
    )
    any_failed = any(
        store.read_json(reference)["execution_status"] == "failed"
        for reference in results
    )
    return results, any_failed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser(
        "materialize-inputs", help="publish deterministic shared data and starts"
    )
    materialize.add_argument("--tier", choices=ALLOWED_TIERS, required=True)
    materialize.add_argument("--run-dir", type=Path, required=True)
    run = commands.add_parser(
        "run-unit", help="run one indivisible ten-cell backend pair"
    )
    run.add_argument("--tier", choices=ALLOWED_TIERS, required=True)
    run.add_argument("--pair-id", required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--worker-identity", required=True)
    run.add_argument("--expected-git-commit")
    merge = commands.add_parser(
        "merge-runs", help="merge downloaded pair roots by exact bytes"
    )
    merge.add_argument("--source-dir", type=Path, required=True)
    merge.add_argument("--run-dir", type=Path, required=True)
    private = commands.add_parser("_sample-cell", help=argparse.SUPPRESS)
    private.add_argument("--tier", choices=ALLOWED_TIERS, required=True)
    private.add_argument("--cell-id", required=True)
    private.add_argument("--run-dir", type=Path, required=True)
    private.add_argument("--context", type=Path, required=True)
    private.add_argument("--output-dir", type=Path, required=True)
    private_materialize = commands.add_parser(
        "_materialize-unit", help=argparse.SUPPRESS
    )
    private_materialize.add_argument("--tier", choices=ALLOWED_TIERS, required=True)
    private_materialize.add_argument("--cell-id", required=True)
    private_materialize.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the causal execution CLI with fail-safe exit semantics."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "merge-runs":
            summary = merge_run_directories(args.source_dir, args.run_dir)
            print(canonical_json_bytes(summary).decode(), end="")
            return 0
        manifest = load_manifest(args.manifest)
        validate_manifest(manifest, manifest_path=args.manifest)
        if args.command == "materialize-inputs":
            input_references = materialize_inputs(
                manifest,
                args.tier,
                ArtifactStore(args.run_dir),
                manifest_path=args.manifest,
            )
            print(
                canonical_json_bytes(
                    {"tier": args.tier, "input_pairs": len(input_references)}
                ).decode(),
                end="",
            )
            return 0
        if args.command == "_materialize-unit":
            return _private_materialize_unit(
                args.manifest,
                args.tier,
                args.cell_id,
                args.run_dir,
            )
        if args.command == "_sample-cell":
            return _private_sample_cell(
                args.manifest,
                args.tier,
                args.cell_id,
                args.run_dir,
                args.context,
                args.output_dir,
            )
        result_references, any_failed = run_unit_parent(
            manifest,
            args.manifest,
            tier=args.tier,
            pair_id=args.pair_id,
            run_dir=args.run_dir,
            worker_identity=args.worker_identity,
            expected_git_commit=args.expected_git_commit,
        )
        print(
            canonical_json_bytes(
                {
                    "pair_id": args.pair_id,
                    "cell_results": [
                        reference.as_dict() for reference in result_references
                    ],
                    "scientific_failures": any_failed,
                }
            ).decode(),
            end="",
        )
        return 1 if any_failed else 0
    except (CausalArtifactError, CausalContractError, CausalRunnerError) as error:
        parser.exit(2, f"causal runner error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

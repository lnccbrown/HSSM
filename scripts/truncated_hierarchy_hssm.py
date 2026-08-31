"""HSSM construction and pre-sampling diagnostics for qualification #1282.

The functions in this module intentionally stop before posterior sampling. They
construct the exact HSSM candidate/control graphs, materialize the transformed
starts that HSSM passes to NUTS, and separate prior-factor gradient checks from
full-likelihood checks. In particular, LBA likelihood parity is reported on its
own so that a likelihood-backend disagreement cannot be called a
``TruncatedNormal`` failure.

The executable v2 manifest records the prior hyper-location numerically. Candidate
construction still comes only from HSSM's generated safe prior: the scenario value
is checked against the exact #1277 factory and is never used to recalibrate it.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import bambi as bmb
import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
from pymc.initial_point import StartDict, make_initial_point_fns_per_chain
from pytensor.graph.traversal import ancestors

import hssm
from hssm.param.parameterization_check import find_disconnected_free_rvs
from scripts.truncated_hierarchy_models import (
    five_point_gradient,
    maximum_errors,
    normalized_error_max,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from pytensor.tensor.variable import TensorVariable

SamplerName = Literal["pymc", "numpyro"]
FloatX = Literal["float32", "float64"]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DDM_NETWORK = REPO_ROOT / "tests/fixtures/ddm.onnx"
DDM_NETWORK_SHA256 = "2fbc2199d61cf3ca67725d2262c563636655d05dd1873770825db34313c179f6"

_MODEL_PARAMETER = {
    "lba2_b": "b",
    "approx_ddm_z": "z",
    "softmax_beta": "beta",
}
_MODEL_FORMULA = {
    "lba2_b": "b ~ 0 + (1 | participant_id)",
    "approx_ddm_z": "z ~ 0 + (1 | participant_id)",
    "softmax_beta": "beta ~ 0 + (1 | participant_id)",
}
_CONTROL_PRIOR = {
    "lba2_b": "linked_normal",
    "approx_ddm_z": "linked_normal",
    "softmax_beta": "linked_normal",
}


class HSSMQualificationError(ValueError):
    """Raised when an HSSM qualification input is incomplete or inconsistent."""


@dataclass
class HSSMBuild:
    """A built HSSM graph plus the names needed by downstream diagnostics."""

    scenario_id: str
    model_key: str
    prior_kind: str
    parameter: str
    group_term: str
    group_rv_name: str
    group_location_name: str
    group_scale_name: str
    link_name: str
    bounds: tuple[float, float]
    prior_hyper_location: float
    floatx: FloatX
    initialization_seed: int
    data: pd.DataFrame
    model: hssm.HSSM


@dataclass(frozen=True)
class HierarchyDiagnostics:
    """Observed Bambi/PyMC structure of one generated group hierarchy."""

    prior_family: str
    location_prior_family: str
    scale_prior_family: str
    group_rv_op: str
    location_rv_op: str
    scale_rv_op: str
    prior_noncentered: bool | None
    base_mu: float
    lower: float
    upper: float
    free_rv_names: tuple[str, ...]
    value_var_names: tuple[str, ...]
    offset_present: bool
    group_connected_to_parameter: bool
    disconnected_free_rvs: tuple[str, ...]


@dataclass(frozen=True)
class SamplerStartArtifact:
    """The transformed chain starts produced from HSSM's actual ``_initvals``."""

    sampler: SamplerName
    initialization_seed: int
    start_seeds: tuple[int, ...]
    transformed_points: tuple[dict[str, np.ndarray], ...]

    def as_jsonable(self) -> dict[str, Any]:
        """Return a strict-JSON representation suitable for the raw artifact."""
        chains = []
        for chain_index, point in enumerate(self.transformed_points):
            chains.append(
                {
                    "chain_index": chain_index,
                    "values": {
                        name: np.asarray(value).tolist()
                        for name, value in sorted(point.items())
                    },
                }
            )
        return {
            "schema_version": 1,
            "coordinate_system": "pymc-transformed-value-variables",
            "sampler": self.sampler,
            "initialization_seed": self.initialization_seed,
            "start_seeds": list(self.start_seeds),
            "chains": chains,
        }

    def sha256(self) -> str:
        """Return the canonical digest stored in qualification provenance."""
        payload = json.dumps(
            self.as_jsonable(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GradientDiagnostics:
    """Raw transformed-coordinate diagnostics for one HSSM graph."""

    target_rvs: tuple[str, ...]
    target_value_vars: tuple[str, ...]
    finite_difference_base_step: float
    prior_factor_logp_pytensor: float
    prior_factor_logp_jax: float
    prior_factor_gradient_pytensor: np.ndarray
    prior_factor_gradient_jax: np.ndarray
    prior_factor_gradient_finite_difference: np.ndarray
    finite_difference_gradient_abs_error_max: float
    finite_difference_gradient_rel_error_max: float
    pytensor_jax_gradient_abs_error_max: float
    pytensor_jax_gradient_rel_error_max: float
    full_logp_pytensor: float
    full_logp_jax: float
    full_gradient_size: int
    logp_finite: bool
    gradient_finite: bool

    def qualification_metrics(
        self,
        *,
        finite_difference_absolute_tolerance: float,
        finite_difference_relative_tolerance: float,
        pytensor_jax_absolute_tolerance: float,
        pytensor_jax_relative_tolerance: float,
    ) -> dict[str, bool | float]:
        """Return the pre-sampling fields consumed by the #1282 assessor."""
        return {
            "compile_success": True,
            "logp_finite": self.logp_finite,
            "gradient_finite": self.gradient_finite,
            "finite_difference_gradient_abs_error_max": (
                self.finite_difference_gradient_abs_error_max
            ),
            "finite_difference_gradient_rel_error_max": (
                self.finite_difference_gradient_rel_error_max
            ),
            "finite_difference_gradient_normalized_error_max": normalized_error_max(
                self.prior_factor_gradient_pytensor,
                self.prior_factor_gradient_finite_difference,
                absolute_tolerance=finite_difference_absolute_tolerance,
                relative_tolerance=finite_difference_relative_tolerance,
            ),
            "pytensor_jax_gradient_abs_error_max": (
                self.pytensor_jax_gradient_abs_error_max
            ),
            "pytensor_jax_gradient_rel_error_max": (
                self.pytensor_jax_gradient_rel_error_max
            ),
            "pytensor_jax_gradient_normalized_error_max": normalized_error_max(
                self.prior_factor_gradient_pytensor,
                self.prior_factor_gradient_jax,
                absolute_tolerance=pytensor_jax_absolute_tolerance,
                relative_tolerance=pytensor_jax_relative_tolerance,
            ),
        }


@dataclass(frozen=True)
class LikelihoodParityDiagnostics:
    """PyTensor/JAX value and gradient disagreement for the LBA2 likelihood."""

    value_abs_error_max: float
    value_rel_error_max: float
    gradient_abs_error_max: float
    gradient_rel_error_max: float
    pytensor_values: np.ndarray
    jax_values: np.ndarray
    pytensor_gradient: np.ndarray
    jax_gradient: np.ndarray
    all_finite: bool

    def qualification_metrics(
        self,
        *,
        value_absolute_tolerance: float,
        value_relative_tolerance: float,
        gradient_absolute_tolerance: float,
        gradient_relative_tolerance: float,
    ) -> dict[str, float]:
        """Return raw and normalized likelihood-backend evidence."""
        return {
            "likelihood_pytensor_jax_value_abs_error_max": self.value_abs_error_max,
            "likelihood_pytensor_jax_value_rel_error_max": self.value_rel_error_max,
            "likelihood_pytensor_jax_value_normalized_error_max": (
                normalized_error_max(
                    self.pytensor_values,
                    self.jax_values,
                    absolute_tolerance=value_absolute_tolerance,
                    relative_tolerance=value_relative_tolerance,
                )
            ),
            "likelihood_pytensor_jax_gradient_abs_error_max": (
                self.gradient_abs_error_max
            ),
            "likelihood_pytensor_jax_gradient_rel_error_max": (
                self.gradient_rel_error_max
            ),
            "likelihood_pytensor_jax_gradient_normalized_error_max": (
                normalized_error_max(
                    self.pytensor_gradient,
                    self.jax_gradient,
                    absolute_tolerance=gradient_absolute_tolerance,
                    relative_tolerance=gradient_relative_tolerance,
                )
            ),
        }


def _require_scenario_value(scenario: Mapping[str, Any], key: str) -> Any:
    try:
        return scenario[key]
    except KeyError as error:
        raise HSSMQualificationError(f"scenario is missing {key!r}") from error


def _scenario_bounds(scenario: Mapping[str, Any]) -> tuple[float, float]:
    lower_raw = _require_scenario_value(scenario, "lower")
    upper_raw = _require_scenario_value(scenario, "upper")
    lower = -np.inf if lower_raw is None else float(lower_raw)
    upper = np.inf if upper_raw is None else float(upper_raw)
    if not lower < upper:
        raise HSSMQualificationError("scenario bounds must define a non-empty interval")
    return lower, upper


def _generated_base_mu(bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return float((lower + upper) / 2) if np.isfinite([lower, upper]).all() else 0.0


def _scenario_prior_hyper_location(
    scenario: Mapping[str, Any],
    *,
    prior_kind: str,
    bounds: tuple[float, float],
) -> float:
    raw = _require_scenario_value(scenario, "prior_hyper_location")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise HSSMQualificationError("prior_hyper_location must be a finite number")
    value = float(raw)
    if not np.isfinite(value):
        raise HSSMQualificationError("prior_hyper_location must be a finite number")
    expected = _generated_base_mu(bounds) if prior_kind == "truncated_normal" else 0.0
    if not np.isclose(value, expected, rtol=0.0, atol=1e-12):
        raise HSSMQualificationError(
            "prior_hyper_location does not match the frozen generated-prior contract: "
            f"expected {expected}, got {value}"
        )
    return value


def make_structural_test_data(
    model_key: str,
    *,
    n_groups: int,
    n_per_group: int,
    seed: int,
) -> pd.DataFrame:
    """Create deterministic local data for construction/gradient tests only.

    This helper is not a recovery or SBC generator. The observations are plausible,
    finite inputs that exercise every group in a model graph without calling a
    simulator or any network service.
    """
    if model_key not in _MODEL_PARAMETER:
        raise HSSMQualificationError(f"unsupported HSSM model contract: {model_key}")
    if n_groups <= 0 or n_per_group <= 0:
        raise HSSMQualificationError("structural data dimensions must be positive")
    rng = np.random.default_rng(seed)
    group_index = np.repeat(np.arange(n_groups), n_per_group)
    trial_index = np.tile(np.arange(n_per_group), n_groups)
    participants = np.asarray([f"p{index:03d}" for index in group_index])
    response_index = (group_index + trial_index) % 2

    if model_key == "softmax_beta":
        return pd.DataFrame(
            {
                "response": np.where(response_index == 0, -1, 1),
                "participant_id": participants,
            }
        )

    rt_base = 0.43 if model_key == "lba2_b" else 0.58
    rt = (
        rt_base
        + 0.035 * (trial_index % 5)
        + rng.uniform(-0.002, 0.002, size=len(group_index))
    )
    if model_key == "lba2_b":
        response = response_index
    else:
        response = np.where(response_index == 0, -1, 1)
    return pd.DataFrame(
        {"rt": rt, "response": response, "participant_id": participants}
    )


def _validate_model_data(
    scenario: Mapping[str, Any], model_key: str, data: pd.DataFrame
) -> None:
    expected_columns = (
        {"response", "participant_id"}
        if model_key == "softmax_beta"
        else {"rt", "response", "participant_id"}
    )
    missing = expected_columns - set(data.columns)
    if missing:
        raise HSSMQualificationError(f"HSSM data is missing columns {sorted(missing)}")
    n_groups = int(_require_scenario_value(scenario, "n_groups"))
    n_per_group = int(_require_scenario_value(scenario, "n_per_group"))
    if len(data) != n_groups * n_per_group:
        raise HSSMQualificationError(
            "HSSM data row count does not match n_groups * n_per_group"
        )
    counts = data.groupby("participant_id", sort=False).size().to_numpy()
    if len(counts) != n_groups or not np.all(counts == n_per_group):
        raise HSSMQualificationError(
            "HSSM data does not contain the declared balanced participant panel"
        )


def _shifted_log_link(lower: float) -> hssm.Link:
    def link(value):
        return np.log(np.asarray(value) - lower)

    def linkinv(value):
        return lower + np.exp(value)

    def linkinv_backend(value):
        return lower + pt.exp(value)

    return hssm.Link(
        "shifted_log",
        link=link,
        linkinv=linkinv,
        linkinv_backend=linkinv_backend,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def _seed_legacy_numpy(seed: int) -> Iterator[None]:
    """Seed HSSM's legacy-NumPy init jitter without leaking global RNG state."""
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise HSSMQualificationError("initval_seed must be a uint32-compatible int")
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def build_hssm_model(
    scenario: Mapping[str, Any],
    data: pd.DataFrame,
    *,
    initval_seed: int,
    ddm_network_path: Path = DEFAULT_DDM_NETWORK,
) -> HSSMBuild:
    """Build one exact manifest HSSM candidate or transformed control.

    The native candidates use only ``prior_settings="safe"`` and an absent
    explicit link, exercising the exact generated #1277 default. Controls alter
    only the link scale: shifted log for LBA ``b``, generalized logit for DDM
    ``z``, and log for softmax ``beta``. No posterior sampling occurs.
    """
    if _require_scenario_value(scenario, "layer") != "hssm":
        raise HSSMQualificationError("build_hssm_model requires layer='hssm'")
    model_key = str(_require_scenario_value(scenario, "model"))
    if model_key not in _MODEL_PARAMETER:
        raise HSSMQualificationError(f"unsupported HSSM model contract: {model_key}")
    prior_kind = str(_require_scenario_value(scenario, "prior"))
    allowed_priors = {"truncated_normal", _CONTROL_PRIOR[model_key]}
    if prior_kind not in allowed_priors:
        raise HSSMQualificationError(
            f"{model_key} prior must be one of {sorted(allowed_priors)}"
        )
    floatx = str(_require_scenario_value(scenario, "floatx"))
    if floatx not in {"float32", "float64"}:
        raise HSSMQualificationError("HSSM floatx must be float32 or float64")
    floatx_typed: FloatX = "float32" if floatx == "float32" else "float64"
    bounds = _scenario_bounds(scenario)
    prior_hyper_location = _scenario_prior_hyper_location(
        scenario,
        prior_kind=prior_kind,
        bounds=bounds,
    )
    _validate_model_data(scenario, model_key, data)

    parameter = _MODEL_PARAMETER[model_key]
    include: dict[str, Any] = {
        "name": parameter,
        "formula": _MODEL_FORMULA[model_key],
    }
    if prior_kind == "truncated_normal":
        link_name = "identity"
    elif model_key in {"lba2_b", "softmax_beta"}:
        if not np.isfinite(bounds[0]) or np.isfinite(bounds[1]):
            raise HSSMQualificationError("the shifted-log control is lower-only")
        include["link"] = _shifted_log_link(bounds[0])
        link_name = "shifted_log"
    elif model_key == "approx_ddm_z":
        if not np.isfinite(bounds).all():
            raise HSSMQualificationError(
                "the DDM gen-logit control needs finite bounds"
            )
        include["link"] = hssm.Link("gen_logit", bounds=bounds)
        link_name = "gen_logit"
    hssm.set_floatX(floatx_typed, update_jax=True)
    common_kwargs: dict[str, Any] = {
        "data": data.copy(),
        "include": [include],
        "p_outlier": 0.0,
        "prior_settings": "safe",
        "noncentered": True,
        "process_initvals": True,
    }
    if model_key == "lba2_b":
        model_kwargs = {
            **common_kwargs,
            "model": "lba2",
            "A": 0.1,
            "v0": 1.0,
            "v1": 1.2,
        }
    elif model_key == "approx_ddm_z":
        network_path = ddm_network_path.resolve()
        if not network_path.is_file():
            raise HSSMQualificationError(f"DDM network is unavailable: {network_path}")
        if _file_sha256(network_path) != DDM_NETWORK_SHA256:
            raise HSSMQualificationError(
                "DDM network does not match the frozen fixture"
            )
        model_kwargs = {
            **common_kwargs,
            "model": "ddm",
            "loglik_kind": "approx_differentiable",
            "loglik": network_path,
            "model_config": {"backend": "jax", "bounds": {"z": bounds}},
            "v": 0.5,
            "a": 1.5,
            "t": 0.3,
        }
    else:
        model_kwargs = {
            **common_kwargs,
            "model": "softmax_inv_temperature_2",
            "logit1": 1.0,
        }

    with _seed_legacy_numpy(initval_seed):
        # The scenario-specific dictionary is validated above. Pyrefly cannot
        # narrow a heterogeneous ``**kwargs`` mapping to HSSM's overload.
        # pyrefly: ignore[bad-argument-type]
        model = hssm.HSSM(**model_kwargs)

    group_rv_name = f"{parameter}_1|participant_id"
    build = HSSMBuild(
        scenario_id=str(_require_scenario_value(scenario, "scenario_id")),
        model_key=model_key,
        prior_kind=prior_kind,
        parameter=parameter,
        group_term="1|participant_id",
        group_rv_name=group_rv_name,
        group_location_name=f"{group_rv_name}_mu",
        group_scale_name=f"{group_rv_name}_sigma",
        link_name=link_name,
        bounds=bounds,
        prior_hyper_location=prior_hyper_location,
        floatx=floatx_typed,
        initialization_seed=initval_seed,
        data=data.copy(),
        model=model,
    )
    validate_expected_hierarchy(build)
    return build


def inspect_hierarchy(build: HSSMBuild) -> HierarchyDiagnostics:
    """Inspect the real Bambi priors and PyMC random-variable graph."""
    parameter_prior = build.model.params[build.parameter].prior
    if not isinstance(parameter_prior, dict):
        raise HSSMQualificationError("parameter did not produce term-level priors")
    prior = parameter_prior[build.group_term]
    if not isinstance(prior, bmb.Prior):
        raise HSSMQualificationError("group term did not produce a Bambi prior")
    location_prior = prior.args.get("mu")
    scale_prior = prior.args.get("sigma")
    if not isinstance(location_prior, bmb.Prior) or not isinstance(
        scale_prior, bmb.Prior
    ):
        raise HSSMQualificationError("group prior lost its recursive hyperpriors")

    pymc_model = build.model.pymc_model
    group_rv = pymc_model.named_vars[build.group_rv_name]
    location_rv = pymc_model.named_vars[build.group_location_name]
    scale_rv = pymc_model.named_vars[build.group_scale_name]
    parameter_rv = pymc_model.named_vars[build.parameter]
    parameter_ancestors = set(ancestors([parameter_rv]))
    free_names = tuple(rv.name for rv in pymc_model.free_RVs)
    value_names = tuple(value.name for value in pymc_model.value_vars)

    lower_raw = prior.args.get("lower", -np.inf)
    upper_raw = prior.args.get("upper", np.inf)
    lower = float(np.asarray(-np.inf if lower_raw is None else lower_raw))
    upper = float(np.asarray(np.inf if upper_raw is None else upper_raw))
    base_mu = float(np.asarray(location_prior.args["mu"]))
    return HierarchyDiagnostics(
        prior_family=prior.name,
        location_prior_family=location_prior.name,
        scale_prior_family=scale_prior.name,
        group_rv_op=type(group_rv.owner.op).__name__,
        location_rv_op=type(location_rv.owner.op).__name__,
        scale_rv_op=type(scale_rv.owner.op).__name__,
        prior_noncentered=prior.noncentered,
        base_mu=base_mu,
        lower=lower,
        upper=upper,
        free_rv_names=free_names,
        value_var_names=value_names,
        offset_present=f"{build.group_rv_name}_offset" in free_names,
        group_connected_to_parameter=group_rv in parameter_ancestors,
        disconnected_free_rvs=tuple(find_disconnected_free_rvs(pymc_model)),
    )


def validate_expected_hierarchy(build: HSSMBuild) -> HierarchyDiagnostics:
    """Require the intended centered, connected, no-offset generated structure."""
    observed = inspect_hierarchy(build)
    candidate = build.prior_kind == "truncated_normal"
    expected_family = "TruncatedNormal" if candidate else "Normal"
    expected_op = "TruncatedNormalRV" if candidate else "NormalRV"
    expected_base_mu = build.prior_hyper_location
    required_free = {
        build.group_rv_name,
        build.group_location_name,
        build.group_scale_name,
    }
    errors: list[str] = []
    if observed.prior_family != expected_family:
        errors.append(f"outer family is {observed.prior_family}, not {expected_family}")
    if observed.location_prior_family != expected_family:
        errors.append("location hyperprior has the wrong family")
    if observed.scale_prior_family != "Weibull":
        errors.append("scale hyperprior is not Weibull")
    if observed.group_rv_op != expected_op or observed.location_rv_op != expected_op:
        errors.append("PyMC group/location RV operators do not match the prior family")
    if observed.scale_rv_op != "WeibullBetaRV":
        errors.append("PyMC group scale is not a WeibullBetaRV")
    if observed.prior_noncentered is not False:
        errors.append("location-bearing group term is not explicitly centered")
    if not np.isclose(observed.base_mu, expected_base_mu):
        errors.append("observed base_mu does not match the frozen v2 contract")
    if candidate and not np.allclose(
        [observed.lower, observed.upper], build.bounds, equal_nan=False
    ):
        errors.append("native TruncatedNormal bounds do not match the scenario")
    if not required_free <= set(observed.free_rv_names):
        errors.append("one or more hierarchy random variables are not free")
    if observed.offset_present:
        errors.append("unexpected non-centered offset RV is present")
    if not observed.group_connected_to_parameter:
        errors.append("group RV does not feed the HSSM parameter")
    if observed.disconnected_free_rvs:
        errors.append(f"disconnected free RVs: {observed.disconnected_free_rvs}")
    if errors:
        raise HSSMQualificationError("invalid HSSM hierarchy: " + "; ".join(errors))
    return observed


def extract_actual_sampler_starts(
    build: HSSMBuild,
    *,
    sampler: SamplerName,
    chains: int,
) -> SamplerStartArtifact:
    """Transform HSSM ``_initvals`` exactly as its NUTS sampler paths do.

    HSSM invokes every NUTS backend with ``init="adapt_diag"``. PyMC therefore
    adds no initializer jitter, and HSSM explicitly disables NumPyro's extra
    JAX initializer jitter. The only jitter is the support-aware jitter already
    present in ``model._initvals``; the functions below apply PyMC's value
    transforms to those exact constrained overrides.
    """
    if sampler not in {"pymc", "numpyro"}:
        raise HSSMQualificationError(f"unsupported sampler start path: {sampler}")
    if not isinstance(chains, int) or isinstance(chains, bool) or chains <= 0:
        raise HSSMQualificationError("chains must be a positive integer")
    if not build.model._initvals:
        raise HSSMQualificationError("HSSM did not retain processed _initvals")

    overrides: StartDict = {
        name: np.array(value, copy=True)
        for name, value in build.model._initvals.items()
    }
    functions = make_initial_point_fns_per_chain(
        model=build.model.pymc_model,
        overrides=overrides,
        jitter_rvs=set(),
        chains=1,
    )
    expected_names = {value.name for value in build.model.pymc_model.value_vars}
    point = {
        name: np.asarray(value)
        for name, value in functions[0](build.initialization_seed).items()
    }
    if set(point) != expected_names:
        raise HSSMQualificationError(
            "transformed sampler start does not contain every value variable"
        )
    points = tuple(
        {name: np.array(value, copy=True) for name, value in point.items()}
        for _ in range(chains)
    )
    return SamplerStartArtifact(
        sampler=sampler,
        initialization_seed=build.initialization_seed,
        start_seeds=(),
        transformed_points=points,
    )


def validate_actual_sampler_starts(
    build: HSSMBuild, artifact: SamplerStartArtifact
) -> tuple[float, ...]:
    """Validate every transformed start with the backend it will actually use."""
    hssm.set_floatX(build.floatx, update_jax=True)
    mode = "JAX" if artifact.sampler == "numpyro" else None
    logp_fn = build.model.pymc_model.compile_logp(mode=mode)
    logps = []
    for raw_point in artifact.transformed_points:
        point = {name: np.array(value, copy=True) for name, value in raw_point.items()}
        build.model.pymc_model.check_start_vals(point, mode=mode)
        logp = float(np.asarray(logp_fn(point)))
        if not np.isfinite(logp):
            raise HSSMQualificationError("full logp is non-finite at sampler start")
        logps.append(logp)
    return tuple(logps)


def _max_abs_relative(
    reference: np.ndarray | float, observed: np.ndarray | float
) -> tuple[float, float]:
    errors = maximum_errors(reference, observed)
    return errors.absolute_max, errors.relative_max


def _jaxified_value_and_gradient(
    output: TensorVariable,
    inputs: list[TensorVariable],
    arguments: list[np.ndarray],
    *,
    wrt_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a jaxified graph and differentiate it with native JAX autodiff."""
    import jax
    import jax.numpy as jnp
    from pymc.sampling.jax import get_jaxified_graph

    if not wrt_indices:
        raise HSSMQualificationError("JAX gradient needs at least one input")
    jaxified = cast(
        "Callable[..., Any]",
        get_jaxified_graph(inputs=inputs, outputs=[output]),
    )
    jax_arguments = tuple(jnp.asarray(argument) for argument in arguments)

    def scalar_output(*values):
        return jnp.sum(jaxified(*values)[0])

    raw_value = np.asarray(jaxified(*jax_arguments)[0])
    raw_gradient = jax.grad(scalar_output, argnums=wrt_indices)(*jax_arguments)
    gradient_parts = (
        raw_gradient if isinstance(raw_gradient, tuple) else (raw_gradient,)
    )
    gradient = np.concatenate([np.asarray(part).reshape(-1) for part in gradient_parts])
    return raw_value, gradient


def evaluate_hssm_gradients(
    build: HSSMBuild,
    transformed_start: Mapping[str, np.ndarray],
    *,
    finite_difference_step: float | None = None,
) -> GradientDiagnostics:
    """Evaluate hierarchy-prior accuracy and full-model finiteness at a NUTS start."""
    hssm.set_floatX(build.floatx, update_jax=True)
    pymc_model = build.model.pymc_model
    expected_names = {value.name for value in pymc_model.value_vars}
    if set(transformed_start) != expected_names:
        raise HSSMQualificationError("gradient point must contain every value variable")
    point = {
        name: np.array(value, copy=True) for name, value in transformed_start.items()
    }

    target_rv_names = (
        build.group_location_name,
        build.group_scale_name,
        build.group_rv_name,
    )
    target_rvs = [pymc_model.named_vars[name] for name in target_rv_names]
    target_value_vars = [pymc_model.rvs_to_values[rv] for rv in target_rvs]
    target_value_names: list[str] = []
    target_shapes: list[tuple[int, ...]] = []
    target_sizes: list[int] = []
    target_pieces: list[np.ndarray] = []
    for value_var in target_value_vars:
        if value_var.name is None:
            raise HSSMQualificationError("hierarchy value variable is unnamed")
        value = np.asarray(point[value_var.name])
        target_value_names.append(value_var.name)
        target_shapes.append(value.shape)
        target_sizes.append(value.size)
        target_pieces.append(value.reshape(-1))
    target_vector = np.concatenate(target_pieces).astype(build.floatx, copy=False)

    def point_from_target_vector(vector: np.ndarray) -> dict[str, np.ndarray]:
        if vector.shape != target_vector.shape or not np.isfinite(vector).all():
            raise HSSMQualificationError("hierarchy gradient vector is malformed")
        result = {name: np.array(value, copy=True) for name, value in point.items()}
        cursor = 0
        for name, shape, size in zip(
            target_value_names,
            target_shapes,
            target_sizes,
            strict=True,
        ):
            result[name] = vector[cursor : cursor + size].reshape(shape)
            cursor += size
        return result

    factor_logp = cast(
        "TensorVariable",
        pymc_model.logp(vars=target_rvs, jacobian=True, sum=True),
    )
    factor_gradient = cast(
        "TensorVariable",
        pt.concatenate(
            [
                cast("TensorVariable", pt.grad(factor_logp, value_var)).ravel()
                for value_var in target_value_vars
            ]
        ),
    )
    factor_pytensor_fn = pymc_model.compile_fn(
        [factor_logp, factor_gradient],
        inputs=pymc_model.value_vars,
        mode="FAST_COMPILE",
        on_unused_input="ignore",
    )
    factor_logp_pt, factor_gradient_pt = factor_pytensor_fn(point)
    factor_logp_jax_raw, factor_gradient_jax = _jaxified_value_and_gradient(
        factor_logp,
        target_value_vars,
        [np.asarray(point[name]) for name in target_value_names],
        wrt_indices=tuple(range(len(target_value_vars))),
    )
    factor_logp_jax = float(np.asarray(factor_logp_jax_raw))

    def factor_logp_from_vector(vector: np.ndarray) -> float:
        return float(
            np.asarray(factor_pytensor_fn(point_from_target_vector(vector))[0])
        )

    finite_difference = five_point_gradient(
        factor_logp_from_vector,
        target_vector,
        step=finite_difference_step,
    )
    finite_difference_errors = maximum_errors(factor_gradient_pt, finite_difference)
    jax_errors = maximum_errors(factor_gradient_pt, factor_gradient_jax)
    base_step = (
        float(np.finfo(np.dtype(build.floatx)).eps ** (1 / 5))
        if finite_difference_step is None
        else float(finite_difference_step)
    )

    full_logp_pt = float(np.asarray(pymc_model.compile_logp()(point)))
    full_gradient_pt = np.asarray(pymc_model.compile_dlogp()(point))
    full_inputs = list(pymc_model.value_vars)
    full_logp_jax_raw, full_gradient_jax = _jaxified_value_and_gradient(
        cast("TensorVariable", pymc_model.logp()),
        full_inputs,
        [np.asarray(point[value.name]) for value in full_inputs],
        wrt_indices=tuple(range(len(full_inputs))),
    )
    full_logp_jax = float(np.asarray(full_logp_jax_raw))
    logp_finite = bool(
        np.isfinite(
            [factor_logp_pt, factor_logp_jax, full_logp_pt, full_logp_jax]
        ).all()
    )
    gradient_finite = bool(
        np.isfinite([factor_gradient_pt, factor_gradient_jax, finite_difference]).all()
        and np.isfinite(full_gradient_pt).all()
        and np.isfinite(full_gradient_jax).all()
    )
    return GradientDiagnostics(
        target_rvs=target_rv_names,
        target_value_vars=tuple(target_value_names),
        finite_difference_base_step=base_step,
        prior_factor_logp_pytensor=float(np.asarray(factor_logp_pt)),
        prior_factor_logp_jax=float(np.asarray(factor_logp_jax)),
        prior_factor_gradient_pytensor=np.asarray(factor_gradient_pt),
        prior_factor_gradient_jax=np.asarray(factor_gradient_jax),
        prior_factor_gradient_finite_difference=finite_difference,
        finite_difference_gradient_abs_error_max=finite_difference_errors.absolute_max,
        finite_difference_gradient_rel_error_max=finite_difference_errors.relative_max,
        pytensor_jax_gradient_abs_error_max=jax_errors.absolute_max,
        pytensor_jax_gradient_rel_error_max=jax_errors.relative_max,
        full_logp_pytensor=full_logp_pt,
        full_logp_jax=full_logp_jax,
        full_gradient_size=int(full_gradient_pt.size),
        logp_finite=logp_finite,
        gradient_finite=gradient_finite,
    )


def lba2_pytensor_jax_parity(
    data: pd.DataFrame | np.ndarray,
    *,
    b: float | np.ndarray,
    A: float = 0.1,
    v0: float = 1.0,
    v1: float = 1.2,
    floatx: FloatX = "float64",
) -> LikelihoodParityDiagnostics:
    """Compare LBA2 likelihood values and ``dlogp/db`` across compiler modes.

    The observed data is embedded as a constant so PyTensor's JAX linker sees a
    concrete ``arange`` length, matching a built HSSM model. This function only
    reports raw disagreement; callers decide whether the likelihood layer is
    trustworthy before interpreting hierarchy geometry.
    """
    if isinstance(data, pd.DataFrame):
        values = data[["rt", "response"]].to_numpy()
    else:
        values = np.asarray(data)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise HSSMQualificationError("LBA parity data must have shape (n, 2)")
    hssm.set_floatX(floatx, update_jax=True)
    dtype = np.dtype(floatx)
    values = values.astype(dtype, copy=False)
    b_values = np.asarray(b, dtype=dtype)
    if b_values.ndim == 0:
        b_values = np.full(len(values), b_values, dtype=dtype)
    if b_values.shape != (len(values),):
        raise HSSMQualificationError("LBA b must be scalar or one value per trial")
    if not np.all(b_values > A):
        raise HSSMQualificationError("LBA parity points must satisfy b > A")

    A_symbol = pt.scalar("A", dtype=floatx)
    b_symbol = pt.vector("b", dtype=floatx)
    v0_symbol = pt.scalar("v0", dtype=floatx)
    v1_symbol = pt.scalar("v1", dtype=floatx)
    from hssm.likelihoods.analytical import logp_lba2

    # ``logp_lba2`` supports symbolic parameters at runtime, while its public
    # annotations currently describe only concrete scalar/array callers.
    symbolic_logp_lba2 = cast("Callable[..., TensorVariable]", logp_lba2)
    logp = symbolic_logp_lba2(
        pt.as_tensor_variable(values),
        A_symbol,
        b_symbol,
        v0_symbol,
        v1_symbol,
    )
    gradient = cast("TensorVariable", pt.grad(pt.sum(logp), b_symbol))
    inputs = [A_symbol, b_symbol, v0_symbol, v1_symbol]
    pytensor_fn = pytensor.function(inputs, [logp, gradient], mode="FAST_COMPILE")
    arguments = [
        np.asarray(A, dtype=dtype),
        b_values,
        np.asarray(v0, dtype=dtype),
        np.asarray(v1, dtype=dtype),
    ]
    pytensor_values, pytensor_gradient = pytensor_fn(*arguments)
    jax_values, jax_gradient = _jaxified_value_and_gradient(
        logp,
        inputs,
        arguments,
        wrt_indices=(1,),
    )
    pytensor_values = np.asarray(pytensor_values)
    jax_values = np.asarray(jax_values)
    pytensor_gradient = np.asarray(pytensor_gradient)
    jax_gradient = np.asarray(jax_gradient)
    value_abs, value_rel = _max_abs_relative(pytensor_values, jax_values)
    gradient_abs, gradient_rel = _max_abs_relative(pytensor_gradient, jax_gradient)
    all_finite = bool(
        np.isfinite(pytensor_values).all()
        and np.isfinite(jax_values).all()
        and np.isfinite(pytensor_gradient).all()
        and np.isfinite(jax_gradient).all()
    )
    return LikelihoodParityDiagnostics(
        value_abs_error_max=value_abs,
        value_rel_error_max=value_rel,
        gradient_abs_error_max=gradient_abs,
        gradient_rel_error_max=gradient_rel,
        pytensor_values=pytensor_values,
        jax_values=jax_values,
        pytensor_gradient=pytensor_gradient,
        jax_gradient=jax_gradient,
        all_finite=all_finite,
    )

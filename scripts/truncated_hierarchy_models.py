"""Scenario-independent geometry primitives for HSSM #1282.

The executable v2 manifest resolves the earlier ambiguous categorical anchors into
concrete prior and truth values.  This module remains deliberately
scenario-independent: its API requires those concrete values and builds either of
two centered
hierarchies with the same number of free scalar coordinates:

``NativeTruncatedPrior``
    The location and group values are native PyMC ``TruncatedNormal`` random
    variables on the response scale.  Its base mean may be inside, on, or outside
    support because that is a valid parameter of a truncated distribution.

``LinkedNormalPrior``
    The location and group values are centered ``Normal`` random variables on an
    unconstrained predictor scale.  The complete group predictor is mapped to the
    response scale with HSSM's canonical links: ``lower + exp(eta)``,
    ``upper - exp(eta)``, or the generalized-logit inverse on a finite interval.
    Its predictor-scale base mean must be supplied explicitly.  This is a different
    prior, not an algebraic reparameterization of the candidate.

Direct PyMC and Bambi factories are isomorphic within either prior family.  All
numerical comparisons happen in PyMC's transformed value-variable coordinates, the
coordinates seen by NUTS.  The module provides exact value-block maps, five-point
finite differences, PyTensor/JAX parity, and direct/Bambi isomorphism metrics.  It
contains no posterior sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
from scipy.stats import truncnorm

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

OBSERVATION_SIGMA = 0.5
LOCATION_PRIOR_SIGMA = 0.25
SCALE_PRIOR_ALPHA = 1.5
SCALE_PRIOR_BETA = 0.3
CANONICAL_BLOCK_ORDER = ("group_location", "group_scale", "group_effect")

FloatX = Literal["float32", "float64"]
BoundarySide = Literal["lower", "upper"]


class GeometryContractError(ValueError):
    """Raised when explicit inputs or a built graph violate the contract."""


def _finite_optional(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeometryContractError(f"{name} must be a finite number or None")
    result = float(value)
    if not math.isfinite(result):
        raise GeometryContractError(f"{name} must be finite or None")
    return result


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeometryContractError(f"{name} must be positive and finite")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise GeometryContractError(f"{name} must be positive and finite")
    return result


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GeometryContractError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class Bounds:
    """One- or two-sided open response support."""

    lower: float | None
    upper: float | None

    def __post_init__(self) -> None:
        """Validate and normalize the bounds."""
        lower = _finite_optional(self.lower, "lower")
        upper = _finite_optional(self.upper, "upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        if lower is None and upper is None:
            raise GeometryContractError("at least one finite bound is required")
        if lower is not None and upper is not None and not lower < upper:
            raise GeometryContractError("finite bounds must satisfy lower < upper")

    @property
    def width(self) -> float | None:
        """Return interval width, or ``None`` for one-sided support."""
        if self.lower is None or self.upper is None:
            return None
        return self.upper - self.lower

    def contains(self, value: Any) -> bool:
        """Return whether all values lie strictly inside the open support."""
        array = np.asarray(value)
        if not np.all(np.isfinite(array)):
            return False
        if self.lower is not None and not np.all(array > self.lower):
            return False
        return not (self.upper is not None and not np.all(array < self.upper))

    def pymc_limits(self) -> tuple[float, float]:
        """Return numeric limits accepted by PyMC and Bambi priors."""
        return (
            -np.inf if self.lower is None else self.lower,
            np.inf if self.upper is None else self.upper,
        )


@dataclass(frozen=True, slots=True)
class NativeTruncatedPrior:
    """Explicit centered native-TruncatedNormal hierarchy specification."""

    bounds: Bounds
    location_base_mean: float
    location_prior_sigma: float = LOCATION_PRIOR_SIGMA
    scale_prior_alpha: float = SCALE_PRIOR_ALPHA
    scale_prior_beta: float = SCALE_PRIOR_BETA

    def __post_init__(self) -> None:
        """Validate all explicit prior hyperparameters."""
        if not math.isfinite(float(self.location_base_mean)):
            raise GeometryContractError("location_base_mean must be finite")
        _positive(self.location_prior_sigma, "location_prior_sigma")
        _positive(self.scale_prior_alpha, "scale_prior_alpha")
        _positive(self.scale_prior_beta, "scale_prior_beta")

    @property
    def kind(self) -> Literal["truncated_normal"]:
        """Return the manifest-compatible family name."""
        return "truncated_normal"


@dataclass(frozen=True, slots=True)
class LinkedNormalPrior:
    """Explicit centered predictor-scale Normal hierarchy specification."""

    bounds: Bounds
    location_base_mean_eta: float
    location_prior_sigma: float = LOCATION_PRIOR_SIGMA
    scale_prior_alpha: float = SCALE_PRIOR_ALPHA
    scale_prior_beta: float = SCALE_PRIOR_BETA

    def __post_init__(self) -> None:
        """Validate all explicit prior hyperparameters."""
        if not math.isfinite(float(self.location_base_mean_eta)):
            raise GeometryContractError("location_base_mean_eta must be finite")
        _positive(self.location_prior_sigma, "location_prior_sigma")
        _positive(self.scale_prior_alpha, "scale_prior_alpha")
        _positive(self.scale_prior_beta, "scale_prior_beta")

    @property
    def kind(self) -> Literal["linked_normal"]:
        """Return the manifest-compatible family name."""
        return "linked_normal"


PriorSpec: TypeAlias = NativeTruncatedPrior | LinkedNormalPrior


@dataclass(frozen=True, slots=True)
class ToyDataSpec:
    """Fully explicit natural-scale data-generating specification."""

    bounds: Bounds
    group_location: float
    group_scale: float
    n_groups: int
    n_per_group: int
    floatx: FloatX = "float64"
    observation_sigma: float = OBSERVATION_SIGMA

    def __post_init__(self) -> None:
        """Validate a complete fixed-truth data-generating specification."""
        if not self.bounds.contains(self.group_location):
            raise GeometryContractError(
                "group_location must lie strictly inside response support"
            )
        _positive(self.group_scale, "group_scale")
        if _nonnegative_integer(self.n_groups, "n_groups") == 0:
            raise GeometryContractError("n_groups must be positive")
        _nonnegative_integer(self.n_per_group, "n_per_group")
        _positive(self.observation_sigma, "observation_sigma")
        if self.floatx not in {"float32", "float64"}:
            raise GeometryContractError("floatx must be 'float32' or 'float64'")


@dataclass(frozen=True, slots=True)
class SyntheticHierarchyData:
    """Deterministic natural-scale group truths and Gaussian observations."""

    group_seed: int
    observation_seed: int
    spec: ToyDataSpec
    group_labels: tuple[str, ...]
    group_index: np.ndarray
    y: np.ndarray
    group_effect: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """Return Bambi input with explicit ordered grouping levels."""
        group_values = [self.group_labels[index] for index in self.group_index]
        group_id = pd.Categorical(
            group_values, categories=self.group_labels, ordered=True
        )
        return pd.DataFrame({"y": self.y, "group_id": group_id})


@dataclass(frozen=True, slots=True)
class ValueBlock:
    """One canonical block and its concrete PyMC value variable."""

    canonical_name: str
    random_variable_name: str
    value_variable_name: str
    shape: tuple[int, ...]
    size: int
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class GeometryModel:
    """A PyMC graph plus a checked canonical transformed-value layout."""

    model: pm.Model
    source: Literal["pymc", "bambi"]
    prior_kind: Literal["truncated_normal", "linked_normal"]
    bounds: Bounds
    blocks: tuple[ValueBlock, ...]
    initial_point_values: Mapping[str, np.ndarray]

    @property
    def dimension(self) -> int:
        """Return the number of transformed scalar coordinates."""
        return sum(block.size for block in self.blocks)

    @property
    def canonical_names(self) -> tuple[str, ...]:
        """Return the frozen canonical block order."""
        return tuple(block.canonical_name for block in self.blocks)

    @property
    def value_variable_names(self) -> tuple[str, ...]:
        """Return value-variable names in canonical order."""
        return tuple(block.value_variable_name for block in self.blocks)

    @property
    def model_float_dtype(self) -> np.dtype[Any]:
        """Return the single dtype shared by every free value variable."""
        dtypes = {np.dtype(variable.dtype) for variable in self.model.value_vars}
        if len(dtypes) != 1:
            raise GeometryContractError(f"mixed value-variable dtypes: {dtypes}")
        return next(iter(dtypes))

    def pack_point(self, point: Mapping[str, Any] | None = None) -> np.ndarray:
        """Pack a PyMC point in canonical, not dictionary, order."""
        source = self.initial_point_values if point is None else point
        pieces: list[np.ndarray] = []
        for block in self.blocks:
            if block.value_variable_name not in source:
                raise GeometryContractError(
                    f"point lacks value variable {block.value_variable_name!r}"
                )
            value = np.asarray(source[block.value_variable_name])
            if value.shape != block.shape:
                raise GeometryContractError(
                    f"{block.value_variable_name!r} has shape {value.shape}, "
                    f"expected {block.shape}"
                )
            pieces.append(value.reshape(-1))
        return np.concatenate(pieces).astype(self.model_float_dtype, copy=False)

    def point_from_vector(self, vector: Any) -> dict[str, np.ndarray]:
        """Unpack canonical coordinates into a concrete PyMC point."""
        array = np.asarray(vector, dtype=self.model_float_dtype)
        self.validate_vector(array)
        return {
            block.value_variable_name: array[block.start : block.stop].reshape(
                block.shape
            )
            for block in self.blocks
        }

    def model_args_from_vector(self, vector: Any) -> list[Any]:
        """Return arrays in ``model.value_vars`` order without NumPy coercion."""
        if getattr(vector, "ndim", None) != 1 or vector.shape[0] != self.dimension:
            raise GeometryContractError(
                f"expected a one-dimensional vector of length {self.dimension}"
            )
        by_name = {
            block.value_variable_name: vector[block.start : block.stop].reshape(
                block.shape
            )
            for block in self.blocks
        }
        return [by_name[variable.name] for variable in self.model.value_vars]

    def canonicalize_model_gradient(self, gradient: Any) -> np.ndarray:
        """Reorder PyMC's flattened model gradient into canonical block order."""
        flat = np.asarray(gradient, dtype=self.model_float_dtype).reshape(-1)
        slices: dict[str, slice] = {}
        cursor = 0
        for variable in self.model.value_vars:
            size = int(np.asarray(self.initial_point_values[variable.name]).size)
            slices[variable.name] = slice(cursor, cursor + size)
            cursor += size
        if cursor != flat.size:
            raise GeometryContractError(
                f"gradient length {flat.size} does not match model dimension {cursor}"
            )
        return np.concatenate(
            [flat[slices[block.value_variable_name]] for block in self.blocks]
        )

    def validate_vector(self, vector: np.ndarray) -> None:
        """Reject malformed or non-finite transformed coordinates."""
        if vector.ndim != 1 or vector.size != self.dimension:
            raise GeometryContractError(
                f"expected a one-dimensional vector of length {self.dimension}"
            )
        if not np.all(np.isfinite(vector)):
            raise GeometryContractError("transformed vector must be finite")


@dataclass(frozen=True, slots=True)
class EvaluationPoint:
    """A transformed point with explicit natural-scale boundary distances."""

    vector: np.ndarray
    natural_group_location: float
    natural_group_effect: np.ndarray
    boundary: BoundarySide
    minimum_boundary_distance: float


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    """Maximum elementwise absolute and symmetric relative errors."""

    absolute_max: float
    relative_max: float


@dataclass(frozen=True, slots=True)
class GeometryMetrics:
    """Raw transformed-space log-density and gradient measurements."""

    pytensor_logp: float
    jax_logp: float
    pytensor_gradient: np.ndarray
    jax_gradient: np.ndarray
    finite_difference_gradient: np.ndarray
    finite_difference: ErrorMetrics
    pytensor_jax: ErrorMetrics
    logp_pytensor_jax: ErrorMetrics

    @property
    def all_finite(self) -> bool:
        """Return whether every raw value and discrepancy is finite."""
        scalars = np.asarray(
            [
                self.pytensor_logp,
                self.jax_logp,
                self.finite_difference.absolute_max,
                self.finite_difference.relative_max,
                self.pytensor_jax.absolute_max,
                self.pytensor_jax.relative_max,
                self.logp_pytensor_jax.absolute_max,
                self.logp_pytensor_jax.relative_max,
            ]
        )
        return bool(
            np.all(np.isfinite(scalars))
            and np.all(np.isfinite(self.pytensor_gradient))
            and np.all(np.isfinite(self.jax_gradient))
            and np.all(np.isfinite(self.finite_difference_gradient))
        )

    def finite_difference_normalized_error_max(
        self, *, absolute_tolerance: float, relative_tolerance: float
    ) -> float:
        """Return the combined ratio for autodiff versus finite differences."""
        return normalized_error_max(
            self.pytensor_gradient,
            self.finite_difference_gradient,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )

    def pytensor_jax_normalized_error_max(
        self, *, absolute_tolerance: float, relative_tolerance: float
    ) -> float:
        """Return the combined-tolerance ratio for PyTensor versus native JAX."""
        return normalized_error_max(
            self.pytensor_gradient,
            self.jax_gradient,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )

    def qualification_metrics(
        self,
        *,
        finite_difference_absolute_tolerance: float,
        finite_difference_relative_tolerance: float,
        pytensor_jax_absolute_tolerance: float,
        pytensor_jax_relative_tolerance: float,
    ) -> dict[str, float]:
        """Return raw and normalized gradient evidence for the assessor."""
        return {
            "finite_difference_gradient_abs_error_max": (
                self.finite_difference.absolute_max
            ),
            "finite_difference_gradient_rel_error_max": (
                self.finite_difference.relative_max
            ),
            "finite_difference_gradient_normalized_error_max": (
                self.finite_difference_normalized_error_max(
                    absolute_tolerance=finite_difference_absolute_tolerance,
                    relative_tolerance=finite_difference_relative_tolerance,
                )
            ),
            "pytensor_jax_gradient_abs_error_max": self.pytensor_jax.absolute_max,
            "pytensor_jax_gradient_rel_error_max": self.pytensor_jax.relative_max,
            "pytensor_jax_gradient_normalized_error_max": (
                self.pytensor_jax_normalized_error_max(
                    absolute_tolerance=pytensor_jax_absolute_tolerance,
                    relative_tolerance=pytensor_jax_relative_tolerance,
                )
            ),
        }


@dataclass(frozen=True, slots=True)
class IsomorphismMetrics:
    """Direct/Bambi transformed log-density and gradient discrepancies."""

    direct_logp: float
    bambi_logp: float
    direct_gradient: np.ndarray
    bambi_gradient: np.ndarray
    logp: ErrorMetrics
    gradient: ErrorMetrics

    @property
    def absolute_max(self) -> float:
        """Return the maximum raw absolute discrepancy."""
        return max(self.logp.absolute_max, self.gradient.absolute_max)

    @property
    def relative_max(self) -> float:
        """Return the maximum raw relative discrepancy."""
        return max(self.logp.relative_max, self.gradient.relative_max)

    def normalized_error_max(
        self, *, absolute_tolerance: float, relative_tolerance: float
    ) -> float:
        """Return the largest combined-tolerance ratio across value and gradient."""
        return max(
            normalized_error_max(
                self.direct_logp,
                self.bambi_logp,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            ),
            normalized_error_max(
                self.direct_gradient,
                self.bambi_gradient,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            ),
        )

    def qualification_metrics(
        self, *, absolute_tolerance: float, relative_tolerance: float
    ) -> dict[str, float]:
        """Return raw and normalized direct/Bambi isomorphism evidence."""
        return {
            "bambi_isomorphism_abs_error_max": self.absolute_max,
            "bambi_isomorphism_rel_error_max": self.relative_max,
            "bambi_isomorphism_normalized_error_max": self.normalized_error_max(
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            ),
        }


def support_forward(value: Any, bounds: Bounds) -> np.ndarray:
    """Map response-scale values to HSSM's canonical predictor scale."""
    array = np.asarray(value)
    if not bounds.contains(array):
        raise GeometryContractError("value must lie strictly inside response support")
    lower, upper = bounds.lower, bounds.upper
    if lower is not None and upper is None:
        return np.log(array - lower)
    if lower is None and upper is not None:
        return np.log(upper - array)
    assert lower is not None and upper is not None
    proportion = (array - lower) / (upper - lower)
    return np.log(proportion) - np.log1p(-proportion)


def support_inverse(eta: Any, bounds: Bounds) -> np.ndarray:
    """Map predictor values to support using stable NumPy operations."""
    array = np.asarray(eta)
    lower, upper = bounds.lower, bounds.upper
    if lower is not None and upper is None:
        return lower + np.exp(array)
    if lower is None and upper is not None:
        return upper - np.exp(array)
    assert lower is not None and upper is not None
    sigmoid = np.exp(-np.logaddexp(0.0, -array))
    return lower + (upper - lower) * sigmoid


def _support_inverse_backend(eta: Any, bounds: Bounds) -> Any:
    lower, upper = bounds.lower, bounds.upper
    if lower is not None and upper is None:
        return lower + pt.exp(eta)
    if lower is None and upper is not None:
        return upper - pt.exp(eta)
    assert lower is not None and upper is not None
    return lower + (upper - lower) * pt.sigmoid(eta)


def _draw_truncated_normal(
    rng: np.random.Generator,
    *,
    mu: float,
    sigma: float,
    bounds: Bounds,
    size: int,
) -> np.ndarray:
    """Draw using the exact SciPy parameterization frozen in the v2 manifest."""
    lower = -np.inf if bounds.lower is None else (bounds.lower - mu) / sigma
    upper = np.inf if bounds.upper is None else (bounds.upper - mu) / sigma
    return np.asarray(
        truncnorm.rvs(
            lower,
            upper,
            loc=mu,
            scale=sigma,
            size=size,
            random_state=rng,
        )
    )


def generate_synthetic_data(
    spec: ToyDataSpec,
    *,
    group_seed: int,
    observation_seed: int,
) -> SyntheticHierarchyData:
    """Generate truths and observations from independent frozen RNG streams."""
    for seed, name in (
        (group_seed, "group_seed"),
        (observation_seed, "observation_seed"),
    ):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise GeometryContractError(f"{name} must be a non-negative integer")
    group_rng = np.random.default_rng(group_seed)
    observation_rng = np.random.default_rng(observation_seed)
    group_effect = _draw_truncated_normal(
        group_rng,
        mu=spec.group_location,
        sigma=spec.group_scale,
        bounds=spec.bounds,
        size=spec.n_groups,
    ).astype(spec.floatx, copy=False)
    group_index = np.repeat(np.arange(spec.n_groups, dtype=np.int64), spec.n_per_group)
    y = (
        group_effect[group_index]
        + observation_rng.normal(0.0, spec.observation_sigma, size=group_index.size)
    ).astype(spec.floatx, copy=False)
    return SyntheticHierarchyData(
        group_seed=group_seed,
        observation_seed=observation_seed,
        spec=spec,
        group_labels=tuple(f"g{index:03d}" for index in range(spec.n_groups)),
        group_index=group_index,
        y=y,
        group_effect=group_effect,
    )


def _require_compatible(prior: PriorSpec, data: SyntheticHierarchyData) -> None:
    if prior.bounds != data.spec.bounds:
        raise GeometryContractError("prior and data bounds must be identical")


def build_direct_pymc_model(
    prior: PriorSpec, data: SyntheticHierarchyData
) -> GeometryModel:
    """Build a direct centered PyMC candidate or linked-control hierarchy."""
    _require_compatible(prior, data)
    spec = data.spec
    lower, upper = prior.bounds.pymc_limits()
    coords: dict[str, Any] = {"group": data.group_labels}
    if data.y.size:
        coords["observation"] = np.arange(data.y.size)
    with pytensor.config.change_flags(floatX=spec.floatx):
        with pm.Model(coords=coords) as model:  # pyrefly: ignore[bad-context-manager]
            if isinstance(prior, NativeTruncatedPrior):
                location_rv = pm.TruncatedNormal(
                    "group_location",
                    mu=np.asarray(prior.location_base_mean, dtype=spec.floatx),
                    sigma=np.asarray(prior.location_prior_sigma, dtype=spec.floatx),
                    lower=lower,
                    upper=upper,
                )
            else:
                location_rv = pm.Normal(
                    "group_location_eta",
                    mu=np.asarray(prior.location_base_mean_eta, dtype=spec.floatx),
                    sigma=np.asarray(prior.location_prior_sigma, dtype=spec.floatx),
                )
                pm.Deterministic(
                    "group_location",
                    _support_inverse_backend(location_rv, prior.bounds),
                )
            scale_rv = pm.Weibull(
                "group_scale",
                alpha=np.asarray(prior.scale_prior_alpha, dtype=spec.floatx),
                beta=np.asarray(prior.scale_prior_beta, dtype=spec.floatx),
            )
            if isinstance(prior, NativeTruncatedPrior):
                group_rv = pm.TruncatedNormal(
                    "group_effect",
                    mu=location_rv,
                    sigma=scale_rv,
                    lower=lower,
                    upper=upper,
                    dims="group",
                )
                natural_group = group_rv
                rv_names = {
                    "group_location": "group_location",
                    "group_scale": "group_scale",
                    "group_effect": "group_effect",
                }
            else:
                group_rv = pm.Normal(
                    "group_effect_eta", mu=location_rv, sigma=scale_rv, dims="group"
                )
                natural_group = pm.Deterministic(
                    "group_effect",
                    _support_inverse_backend(group_rv, prior.bounds),
                    dims="group",
                )
                rv_names = {
                    "group_location": "group_location_eta",
                    "group_scale": "group_scale",
                    "group_effect": "group_effect_eta",
                }
            if data.y.size:
                pm.Normal(
                    "y",
                    mu=natural_group[data.group_index],
                    sigma=np.asarray(spec.observation_sigma, dtype=spec.floatx),
                    observed=data.y,
                    dims="observation",
                )
        return _make_geometry_model(model, "pymc", prior, rv_names)


def _fixed_normal_likelihood(name: str, mu: Any, **kwargs: Any) -> Any:
    """Bambi custom likelihood with an externally bound fixed scale."""
    sigma = kwargs.pop("qualification_observation_sigma")
    return pm.Normal(name, mu=mu, sigma=sigma, **kwargs)


def _bambi_link(bounds: Bounds) -> bmb.Link:
    def link(value: Any) -> np.ndarray:
        return support_forward(value, bounds)

    def link_inverse(eta: Any) -> np.ndarray:
        return support_inverse(eta, bounds)

    def link_inverse_backend(eta: Any) -> Any:
        return _support_inverse_backend(eta, bounds)

    return bmb.Link(
        "hssm_qualification_support",
        link=link,
        linkinv=link_inverse,
        linkinv_backend=link_inverse_backend,
    )


def build_bambi_model(prior: PriorSpec, data: SyntheticHierarchyData) -> GeometryModel:
    """Build the Bambi graph isomorphic to the direct graph for one prior family."""
    _require_compatible(prior, data)
    if not data.y.size:
        raise GeometryContractError("Bambi geometry requires at least one observation")
    spec = data.spec
    lower, upper = prior.bounds.pymc_limits()
    if isinstance(prior, NativeTruncatedPrior):
        location_prior = bmb.Prior(
            "TruncatedNormal",
            mu=prior.location_base_mean,
            sigma=prior.location_prior_sigma,
            lower=lower,
            upper=upper,
        )
        group_prior = bmb.Prior(
            "TruncatedNormal",
            mu=location_prior,
            sigma=bmb.Prior(
                "Weibull",
                alpha=prior.scale_prior_alpha,
                beta=prior.scale_prior_beta,
            ),
            lower=lower,
            upper=upper,
            noncentered=False,
        )
        link: str | bmb.Link = "identity"
    else:
        location_prior = bmb.Prior(
            "Normal",
            mu=prior.location_base_mean_eta,
            sigma=prior.location_prior_sigma,
        )
        group_prior = bmb.Prior(
            "Normal",
            mu=location_prior,
            sigma=bmb.Prior(
                "Weibull",
                alpha=prior.scale_prior_alpha,
                beta=prior.scale_prior_beta,
            ),
            noncentered=False,
        )
        link = _bambi_link(prior.bounds)
    likelihood = bmb.Likelihood(
        "FixedNormal",
        params=["mu", "qualification_observation_sigma"],
        parent="mu",
        dist=_fixed_normal_likelihood,
    )
    family = bmb.Family(
        "fixed_normal",
        likelihood=likelihood,
        link=link,  # pyrefly: ignore[bad-argument-type]
    )
    with pytensor.config.change_flags(floatX=spec.floatx):
        bambi_model = bmb.Model(
            "y ~ 0 + (1 | group_id)",
            data.to_frame(),
            family=family,
            priors={
                "1|group_id": group_prior,
                "qualification_observation_sigma": spec.observation_sigma,
            },
            auto_scale=False,
            noncentered=True,
            center_predictors=False,
        )
        bambi_model.build()
        model = bambi_model.backend.model
        rv_names = {
            "group_location": "1|group_id_mu",
            "group_scale": "1|group_id_sigma",
            "group_effect": "1|group_id",
        }
        return _make_geometry_model(model, "bambi", prior, rv_names)


def _make_geometry_model(
    model: pm.Model,
    source: Literal["pymc", "bambi"],
    prior: PriorSpec,
    rv_names: Mapping[str, str],
) -> GeometryModel:
    point = model.initial_point()
    blocks: list[ValueBlock] = []
    cursor = 0
    for canonical_name in CANONICAL_BLOCK_ORDER:
        rv_name = rv_names[canonical_name]
        if rv_name not in model.named_vars:
            raise GeometryContractError(f"model lacks random variable {rv_name!r}")
        rv = model.named_vars[rv_name]
        if rv not in model.rvs_to_values:
            raise GeometryContractError(f"{rv_name!r} is not a free random variable")
        value_var = model.rvs_to_values[rv]
        value = np.asarray(point[value_var.name])
        size = int(value.size)
        blocks.append(
            ValueBlock(
                canonical_name=canonical_name,
                random_variable_name=rv_name,
                value_variable_name=value_var.name,
                shape=value.shape,
                size=size,
                start=cursor,
                stop=cursor + size,
            )
        )
        cursor += size
    mapped = {block.value_variable_name for block in blocks}
    actual = {variable.name for variable in model.value_vars}
    if mapped != actual:
        raise GeometryContractError(
            "model contains unmapped or missing value variables: "
            f"mapped={sorted(mapped)}, actual={sorted(actual)}"
        )
    geometry = GeometryModel(
        model=model,
        source=source,
        prior_kind=prior.kind,
        bounds=prior.bounds,
        blocks=tuple(blocks),
        initial_point_values={
            name: np.asarray(value).copy() for name, value in point.items()
        },
    )
    geometry.pack_point()
    return geometry


def make_near_boundary_evaluation_point(
    geometry: GeometryModel,
    *,
    side: BoundarySide,
    fraction: float = 0.02,
) -> EvaluationPoint:
    """Create a finite point near an explicitly selected, finite boundary."""
    if not math.isfinite(fraction) or not 0 < fraction < 0.25:
        raise GeometryContractError("fraction must lie strictly between 0 and 0.25")
    bound = geometry.bounds.lower if side == "lower" else geometry.bounds.upper
    if bound is None:
        raise GeometryContractError(f"{side} is not a finite boundary")
    width = geometry.bounds.width
    margin = width * fraction if width is not None else LOCATION_PRIOR_SIGMA * fraction
    multipliers = np.linspace(1.25, 2.0, geometry.blocks[-1].size)
    if side == "lower":
        location = bound + margin
        natural_group = bound + margin * multipliers
    else:
        location = bound - margin
        natural_group = bound - margin * multipliers
    if not geometry.bounds.contains(np.r_[location, natural_group]):
        raise GeometryContractError("constructed boundary probe left response support")
    location_coordinate = support_forward(location, geometry.bounds)
    group_coordinate = support_forward(natural_group, geometry.bounds)
    scale_coordinate = np.log(max(margin * 4, 0.05))
    vector = np.concatenate(
        [
            np.asarray(location_coordinate).reshape(-1),
            np.asarray(scale_coordinate).reshape(-1),
            np.asarray(group_coordinate).reshape(-1),
        ]
    ).astype(geometry.model_float_dtype)
    geometry.validate_vector(vector)
    distances: list[float] = []
    for value in np.r_[location, natural_group]:
        if geometry.bounds.lower is not None:
            distances.append(float(value - geometry.bounds.lower))
        if geometry.bounds.upper is not None:
            distances.append(float(geometry.bounds.upper - value))
    return EvaluationPoint(
        vector=vector,
        natural_group_location=float(location),
        natural_group_effect=np.asarray(natural_group),
        boundary=side,
        minimum_boundary_distance=min(distances),
    )


def five_point_gradient(
    function: Callable[[np.ndarray], float],
    vector: Any,
    *,
    step: float | None = None,
) -> np.ndarray:
    """Evaluate a central five-point finite-difference gradient."""
    point = np.asarray(vector)
    if point.ndim != 1 or not np.all(np.isfinite(point)):
        raise GeometryContractError("finite differences require a finite 1-D vector")
    dtype = np.dtype(point.dtype)
    if dtype.kind != "f":
        point = point.astype(np.float64)
        dtype = point.dtype
    if step is not None and (not math.isfinite(step) or step <= 0):
        raise GeometryContractError(
            "finite-difference step must be positive and finite"
        )
    base_step = float(np.finfo(dtype).eps ** (1 / 5)) if step is None else step
    result = np.empty(point.size, dtype=np.float64)
    for index in range(point.size):
        coordinate_step = base_step * max(1.0, abs(float(point[index])))
        offset = np.zeros(point.size, dtype=dtype)
        offset[index] = coordinate_step
        result[index] = (
            -function(point + 2 * offset)
            + 8 * function(point + offset)
            - 8 * function(point - offset)
            + function(point - 2 * offset)
        ) / (12 * coordinate_step)
    return result


def maximum_errors(reference: Any, comparison: Any) -> ErrorMetrics:
    """Return max absolute and symmetric relative errors without pass/fail logic."""
    expected = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(comparison, dtype=np.float64)
    if expected.shape != observed.shape:
        raise GeometryContractError(
            f"comparison shapes differ: {expected.shape} != {observed.shape}"
        )
    absolute = np.abs(expected - observed)
    denominator = np.maximum.reduce(
        [np.abs(expected), np.abs(observed), np.full(expected.shape, 1e-12)]
    )
    relative = absolute / denominator
    return ErrorMetrics(float(np.max(absolute)), float(np.max(relative)))


def normalized_error_max(
    reference: Any,
    comparison: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    """Return the largest standard combined-tolerance error ratio.

    A value at or below one satisfies
    ``abs(observed - reference) <= atol + rtol * max(abs(reference), abs(observed))``
    in every coordinate. Raw absolute and relative maxima remain useful descriptive
    diagnostics, but this combined ratio is the qualification decision statistic.
    """
    expected = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(comparison, dtype=np.float64)
    if expected.shape != observed.shape:
        raise GeometryContractError(
            f"comparison shapes differ: {expected.shape} != {observed.shape}"
        )
    if expected.size == 0:
        raise GeometryContractError("comparison arrays must not be empty")
    tolerances = np.asarray([absolute_tolerance, relative_tolerance], dtype=np.float64)
    if not np.all(np.isfinite(tolerances)) or np.any(tolerances < 0):
        raise GeometryContractError(
            "comparison tolerances must be finite and nonnegative"
        )
    if not np.any(tolerances > 0):
        raise GeometryContractError(
            "at least one comparison tolerance must be positive"
        )
    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(observed)):
        raise GeometryContractError("comparison values must be finite")
    scale = np.maximum(np.abs(expected), np.abs(observed))
    denominator = absolute_tolerance + relative_tolerance * scale
    return float(np.max(np.abs(expected - observed) / denominator))


def _compile_pytensor_functions(
    geometry: GeometryModel,
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    compiled_logp = geometry.model.compile_logp()
    compiled_gradient = geometry.model.compile_dlogp()

    def logp(vector: np.ndarray) -> float:
        return float(compiled_logp(geometry.point_from_vector(vector)))

    def gradient(vector: np.ndarray) -> np.ndarray:
        raw = compiled_gradient(geometry.point_from_vector(vector))
        return geometry.canonicalize_model_gradient(raw)

    return logp, gradient


def _compile_jax_functions(
    geometry: GeometryModel,
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    import jax
    import jax.numpy as jnp
    from pymc.sampling.jax import get_jaxified_graph

    jaxified = get_jaxified_graph(
        inputs=geometry.model.value_vars,
        # PyMC's type annotation permits a list, but sum=True returns one tensor.
        outputs=[geometry.model.logp()],  # pyrefly: ignore[bad-argument-type]
    )

    def scalar_logp(vector: Any) -> Any:
        return jaxified(*geometry.model_args_from_vector(vector))[0]

    gradient_function = jax.grad(scalar_logp)

    def logp(vector: np.ndarray) -> float:
        return float(np.asarray(scalar_logp(jnp.asarray(vector))))

    def gradient(vector: np.ndarray) -> np.ndarray:
        return np.asarray(gradient_function(jnp.asarray(vector)))

    return logp, gradient


def evaluate_transformed_geometry(
    geometry: GeometryModel,
    vector: Any,
    *,
    finite_difference_step: float | None = None,
) -> GeometryMetrics:
    """Measure finite-difference and PyTensor/JAX parity at one transformed point."""
    point = np.asarray(vector, dtype=geometry.model_float_dtype)
    geometry.validate_vector(point)
    pytensor_logp, pytensor_gradient = _compile_pytensor_functions(geometry)
    jax_logp, jax_gradient = _compile_jax_functions(geometry)
    pt_value = pytensor_logp(point)
    pt_gradient = pytensor_gradient(point)
    jax_value = jax_logp(point)
    jax_gradient_value = jax_gradient(point)
    finite_difference = five_point_gradient(
        pytensor_logp, point, step=finite_difference_step
    )
    return GeometryMetrics(
        pytensor_logp=pt_value,
        jax_logp=jax_value,
        pytensor_gradient=pt_gradient,
        jax_gradient=jax_gradient_value,
        finite_difference_gradient=finite_difference,
        finite_difference=maximum_errors(pt_gradient, finite_difference),
        pytensor_jax=maximum_errors(pt_gradient, jax_gradient_value),
        logp_pytensor_jax=maximum_errors(pt_value, jax_value),
    )


def compare_isomorphic_models(
    direct: GeometryModel, bambi: GeometryModel, vector: Any
) -> IsomorphismMetrics:
    """Compare direct-PyMC and Bambi PyTensor logp/gradients at one coordinate."""
    if direct.source != "pymc" or bambi.source != "bambi":
        raise GeometryContractError("comparison requires direct PyMC then Bambi")
    if direct.prior_kind != bambi.prior_kind:
        raise GeometryContractError("models must use the same prior family")
    if direct.bounds != bambi.bounds:
        raise GeometryContractError("models must use the same response support")
    if direct.canonical_names != bambi.canonical_names:
        raise GeometryContractError("models have different canonical blocks")
    if tuple(block.shape for block in direct.blocks) != tuple(
        block.shape for block in bambi.blocks
    ):
        raise GeometryContractError("models have different block shapes")
    point = np.asarray(vector, dtype=direct.model_float_dtype)
    direct.validate_vector(point)
    direct_logp, direct_gradient = _compile_pytensor_functions(direct)
    bambi_logp, bambi_gradient = _compile_pytensor_functions(bambi)
    direct_logp_value = direct_logp(point)
    bambi_logp_value = bambi_logp(point)
    direct_gradient_value = direct_gradient(point)
    bambi_gradient_value = bambi_gradient(point)
    return IsomorphismMetrics(
        direct_logp=direct_logp_value,
        bambi_logp=bambi_logp_value,
        direct_gradient=direct_gradient_value,
        bambi_gradient=bambi_gradient_value,
        logp=maximum_errors(direct_logp_value, bambi_logp_value),
        gradient=maximum_errors(direct_gradient_value, bambi_gradient_value),
    )

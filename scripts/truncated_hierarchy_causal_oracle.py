"""Independent second-order oracle for truncated-hierarchy log densities.

This module is deliberately isolated from PyMC, PyTensor, JAX, Bambi, and the
qualification model builders.  It implements a tiny forward-mode, second-order
``Jet2`` scalar and uses only :mod:`numpy` and :mod:`scipy` special functions.
That makes it suitable as an independent reference when deciding whether a
transformed PyMC graph has the same value, gradient, and curvature as the
mathematical hierarchical TruncatedNormal posterior.

The canonical coordinate order is always location, log scale, then group
effects.  The centered variant uses bounded coordinates for location and group
effects.  Location inverse-CDF non-centering replaces only the location
coordinate with a standard-Normal offset; group inverse-CDF non-centering
replaces only the group coordinates; full inverse-CDF non-centering replaces
both.  Every actual constrained free-variable transform includes its log
absolute Jacobian in the returned posterior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from scipy.special import expit, log_ndtr, ndtr, ndtri, ndtri_exp

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

LOG_2PI = math.log(2.0 * math.pi)
LOG_SQRT_2PI = 0.5 * LOG_2PI

type CausalParameterization = Literal[
    "centered",
    "location_icdf_noncentered",
    "group_icdf_noncentered",
    "full_icdf_noncentered",
]


class OracleDomainError(ValueError):
    """Raised when the oracle is evaluated outside its mathematical domain."""


@dataclass(frozen=True, slots=True)
class Jet2:
    """Value, gradient, and Hessian of one scalar expression.

    ``Jet2`` is intentionally small rather than a general automatic
    differentiation system.  Its arithmetic is sufficient for the probability
    densities and transforms in this module, and keeping it local makes the
    reference independent of the differentiation stacks under investigation.
    """

    value: float
    gradient: NDArray[np.float64]
    hessian: NDArray[np.float64]

    __array_priority__: ClassVar[float] = 1000.0

    def __post_init__(self) -> None:
        """Validate and detach derivative arrays from caller-owned storage."""
        value = float(self.value)
        gradient = np.array(self.gradient, dtype=np.float64, copy=True)
        hessian = np.array(self.hessian, dtype=np.float64, copy=True)
        if gradient.ndim != 1:
            raise ValueError("a Jet2 gradient must be one-dimensional")
        expected_shape = (gradient.size, gradient.size)
        if hessian.shape != expected_shape:
            raise ValueError(
                f"a Jet2 Hessian must have shape {expected_shape}, got {hessian.shape}"
            )
        gradient.setflags(write=False)
        hessian.setflags(write=False)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "hessian", hessian)

    @property
    def dimension(self) -> int:
        """Return the number of independent coordinates represented."""
        return self.gradient.size

    @classmethod
    def constant(cls, value: float, dimension: int) -> Jet2:
        """Construct a constant in a derivative space of ``dimension``."""
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("dimension must be an integer")
        if dimension < 0:
            raise ValueError("dimension must be non-negative")
        return cls(
            float(value),
            np.zeros(dimension, dtype=np.float64),
            np.zeros((dimension, dimension), dtype=np.float64),
        )

    @classmethod
    def variable(cls, value: float, index: int, dimension: int) -> Jet2:
        """Construct one independent variable in a derivative space."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0 or index >= dimension:
            raise ValueError("variable index must lie inside the derivative space")
        gradient = np.zeros(dimension, dtype=np.float64)
        gradient[index] = 1.0
        return cls(
            float(value),
            gradient,
            np.zeros((dimension, dimension), dtype=np.float64),
        )

    def _coerce(self, other: float | Jet2) -> Jet2:
        """Promote a numeric constant and enforce compatible jet dimensions."""
        if isinstance(other, Jet2):
            if other.dimension != self.dimension:
                raise ValueError("Jet2 operands have different dimensions")
            return other
        return Jet2.constant(float(other), self.dimension)

    def compose(self, value: float, first: float, second: float) -> Jet2:
        """Apply a scalar unary function from its first two derivatives."""
        gradient = first * self.gradient
        hessian = first * self.hessian + second * np.outer(self.gradient, self.gradient)
        return Jet2(value, gradient, hessian)

    def __neg__(self) -> Jet2:
        """Negate value and derivatives."""
        return Jet2(-self.value, -self.gradient, -self.hessian)

    def __add__(self, other: float | Jet2) -> Jet2:
        """Add a scalar or compatible jet."""
        right = self._coerce(other)
        return Jet2(
            self.value + right.value,
            self.gradient + right.gradient,
            self.hessian + right.hessian,
        )

    def __radd__(self, other: float | Jet2) -> Jet2:
        """Add this jet to a left-hand scalar or compatible jet."""
        return self + other

    def __sub__(self, other: float | Jet2) -> Jet2:
        """Subtract a scalar or compatible jet."""
        return self + (-self._coerce(other))

    def __rsub__(self, other: float | Jet2) -> Jet2:
        """Subtract this jet from a left-hand scalar or compatible jet."""
        return self._coerce(other) - self

    def __mul__(self, other: float | Jet2) -> Jet2:
        """Multiply by a scalar or compatible jet."""
        right = self._coerce(other)
        hessian = (
            right.value * self.hessian
            + self.value * right.hessian
            + np.outer(self.gradient, right.gradient)
            + np.outer(right.gradient, self.gradient)
        )
        return Jet2(
            self.value * right.value,
            right.value * self.gradient + self.value * right.gradient,
            hessian,
        )

    def __rmul__(self, other: float | Jet2) -> Jet2:
        """Multiply this jet by a left-hand scalar or compatible jet."""
        return self * other

    def reciprocal(self) -> Jet2:
        """Return the multiplicative inverse."""
        if self.value == 0.0:
            raise OracleDomainError("cannot divide by zero")
        inverse = 1.0 / self.value
        return self.compose(inverse, -(inverse**2), 2.0 * inverse**3)

    def __truediv__(self, other: float | Jet2) -> Jet2:
        """Divide by a scalar or compatible jet."""
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other: float | Jet2) -> Jet2:
        """Divide a left-hand scalar or compatible jet by this jet."""
        return self._coerce(other) * self.reciprocal()

    def __pow__(self, exponent: float) -> Jet2:
        """Raise this jet to a constant real power."""
        if not isinstance(exponent, (int, float)) or isinstance(exponent, bool):
            raise TypeError("Jet2 only supports constant real exponents")
        power = float(exponent)
        if self.value == 0.0 and power < 0.0:
            raise OracleDomainError("a negative power requires a nonzero base")
        if self.value <= 0.0 and not power.is_integer():
            raise OracleDomainError("a non-integer Jet2 power requires a positive base")
        value = self.value**power
        if power == 0.0:
            return Jet2.constant(1.0, self.dimension)
        if power == 1.0:
            return self
        first = power * self.value ** (power - 1.0)
        second = power * (power - 1.0) * self.value ** (power - 2.0)
        return self.compose(value, first, second)


def seed_jets(values: ArrayLike) -> tuple[Jet2, ...]:
    """Seed one independent ``Jet2`` for each value in a one-dimensional vector."""
    point = np.asarray(values, dtype=np.float64)
    if point.ndim != 1:
        raise ValueError("jet seed values must be one-dimensional")
    if not np.all(np.isfinite(point)):
        raise ValueError("jet seed values must all be finite")
    return tuple(
        Jet2.variable(float(value), index, point.size)
        for index, value in enumerate(point)
    )


def _promote(*values: float | Jet2) -> tuple[Jet2, ...]:
    """Promote scalars into the single derivative space present in ``values``."""
    dimensions = {value.dimension for value in values if isinstance(value, Jet2)}
    if len(dimensions) > 1:
        raise ValueError("Jet2 operands have different dimensions")
    dimension = next(iter(dimensions), 0)
    return tuple(
        value if isinstance(value, Jet2) else Jet2.constant(float(value), dimension)
        for value in values
    )


def jet_exp(value: Jet2) -> Jet2:
    """Exponentiate a jet."""
    result = math.exp(value.value)
    return value.compose(result, result, result)


def jet_log(value: Jet2) -> Jet2:
    """Take a natural logarithm of a positive jet."""
    if value.value <= 0.0:
        raise OracleDomainError("logarithm requires a positive value")
    inverse = 1.0 / value.value
    return value.compose(math.log(value.value), inverse, -(inverse**2))


def jet_sigmoid(value: Jet2) -> Jet2:
    """Evaluate a numerically stable logistic sigmoid."""
    result = float(expit(value.value))
    first = result * (1.0 - result)
    second = first * (1.0 - 2.0 * result)
    return value.compose(result, first, second)


def jet_softplus(value: Jet2) -> Jet2:
    """Evaluate ``log(1 + exp(value))`` without overflow."""
    result = float(np.logaddexp(0.0, value.value))
    first = float(expit(value.value))
    return value.compose(result, first, first * (1.0 - first))


def jet_log_ndtr(value: Jet2) -> Jet2:
    """Evaluate the standard-normal log CDF and its first two derivatives."""
    result = float(log_ndtr(value.value))
    log_density = -0.5 * value.value**2 - LOG_SQRT_2PI
    inverse_mills = math.exp(log_density - result)
    second = -inverse_mills * (value.value + inverse_mills)
    return value.compose(result, inverse_mills, second)


def jet_ndtr(value: Jet2) -> Jet2:
    """Evaluate the standard-normal CDF and its first two derivatives."""
    result = float(ndtr(value.value))
    density = math.exp(-0.5 * value.value**2 - LOG_SQRT_2PI)
    return value.compose(result, density, -value.value * density)


def jet_ndtri(value: Jet2) -> Jet2:
    """Evaluate the standard-normal quantile and its first two derivatives."""
    if not 0.0 < value.value < 1.0:
        raise OracleDomainError("normal quantile requires a probability in (0, 1)")
    result = float(ndtri(value.value))
    inverse_density = math.exp(0.5 * result**2 + LOG_SQRT_2PI)
    return value.compose(
        result,
        inverse_density,
        result * inverse_density**2,
    )


def jet_ndtri_exp(log_probability: Jet2) -> Jet2:
    """Invert a standard-normal log CDF without exponentiating its value."""
    if not log_probability.value < 0.0:
        raise OracleDomainError("normal log-quantile requires a value below zero")
    result = float(ndtri_exp(log_probability.value))
    log_density = -0.5 * result**2 - LOG_SQRT_2PI
    first = math.exp(log_probability.value - log_density)
    second = first * (1.0 + result * first)
    return log_probability.compose(result, first, second)


def jet_log1mexp(value: Jet2) -> Jet2:
    """Evaluate ``log(1 - exp(value))`` stably for a negative jet."""
    if value.value >= 0.0:
        raise OracleDomainError("log1mexp requires a strictly negative value")
    if value.value < -math.log(2.0):
        result = math.log1p(-math.exp(value.value))
    else:
        result = math.log(-math.expm1(value.value))
    exponential = math.exp(value.value)
    denominator = -math.expm1(value.value)
    first = -exponential / denominator
    second = -exponential / denominator**2
    return value.compose(result, first, second)


def jet_logdiffexp(larger: Jet2, smaller: Jet2) -> Jet2:
    """Return ``log(exp(larger) - exp(smaller))`` without cancellation."""
    if larger.dimension != smaller.dimension:
        raise ValueError("Jet2 operands have different dimensions")
    if not larger.value > smaller.value:
        raise OracleDomainError("logdiffexp requires larger > smaller")
    return larger + jet_log1mexp(smaller - larger)


def jet_logaddexp(left: Jet2, right: Jet2) -> Jet2:
    """Return ``log(exp(left) + exp(right))`` without overflow."""
    if left.dimension != right.dimension:
        raise ValueError("Jet2 operands have different dimensions")
    if left.value >= right.value:
        return left + jet_softplus(right - left)
    return right + jet_softplus(left - right)


@dataclass(frozen=True, slots=True)
class TruncationBounds:
    """One- or two-sided open support used by a TruncatedNormal."""

    lower: float | None
    upper: float | None

    def __post_init__(self) -> None:
        """Validate and normalize support limits."""
        lower = None if self.lower is None else float(self.lower)
        upper = None if self.upper is None else float(self.upper)
        if lower is not None and not math.isfinite(lower):
            raise ValueError("lower must be finite or None")
        if upper is not None and not math.isfinite(upper):
            raise ValueError("upper must be finite or None")
        if lower is None and upper is None:
            raise ValueError("at least one truncation bound is required")
        if lower is not None and upper is not None and not lower < upper:
            raise ValueError("finite bounds must satisfy lower < upper")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def contains(self, value: float) -> bool:
        """Return whether ``value`` lies strictly within the open support."""
        if not math.isfinite(value):
            return False
        if self.lower is not None and value <= self.lower:
            return False
        return self.upper is None or value < self.upper


@dataclass(frozen=True, slots=True)
class TransformedValue:
    """Natural value and log absolute Jacobian of a scalar transform."""

    natural: Jet2
    log_abs_det_jacobian: Jet2


def support_transform(
    unconstrained: Jet2, bounds: TruncationBounds
) -> TransformedValue:
    """Map an unconstrained jet into lower, upper, or finite support."""
    lower = bounds.lower
    upper = bounds.upper
    if lower is not None and upper is None:
        distance = jet_exp(unconstrained)
        return TransformedValue(lower + distance, unconstrained)
    if lower is None and upper is not None:
        distance = jet_exp(unconstrained)
        return TransformedValue(upper - distance, unconstrained)
    if lower is None or upper is None:  # pragma: no cover - validated bounds
        raise AssertionError("unreachable invalid bounds")
    width = upper - lower
    fraction = jet_sigmoid(unconstrained)
    log_jacobian = (
        math.log(width) - jet_softplus(-unconstrained) - jet_softplus(unconstrained)
    )
    return TransformedValue(lower + width * fraction, log_jacobian)


def support_inverse(natural: float, bounds: TruncationBounds) -> float:
    """Map an interior natural value back to its unconstrained coordinate."""
    value = float(natural)
    if not bounds.contains(value):
        raise OracleDomainError("natural value must lie strictly inside support")
    if bounds.lower is not None and bounds.upper is None:
        return math.log(value - bounds.lower)
    if bounds.lower is None and bounds.upper is not None:
        return math.log(bounds.upper - value)
    if bounds.lower is None or bounds.upper is None:  # pragma: no cover
        raise AssertionError("unreachable invalid bounds")
    return math.log(value - bounds.lower) - math.log(bounds.upper - value)


def positive_transform(unconstrained: Jet2) -> TransformedValue:
    """Map an unconstrained jet to the positive reals with an exponential."""
    return TransformedValue(jet_exp(unconstrained), unconstrained)


def positive_inverse(natural: float) -> float:
    """Map a positive natural value back to its log coordinate."""
    value = float(natural)
    if not math.isfinite(value) or value <= 0.0:
        raise OracleDomainError("positive inverse requires a positive finite value")
    return math.log(value)


def normal_logpdf(
    value: float | Jet2,
    location: float | Jet2,
    scale: float | Jet2,
) -> Jet2:
    """Return a Normal log density including its normalization constant."""
    value_jet, location_jet, scale_jet = _promote(value, location, scale)
    if scale_jet.value <= 0.0:
        raise OracleDomainError("Normal scale must be positive")
    standardized = (value_jet - location_jet) / scale_jet
    return -0.5 * standardized**2 - jet_log(scale_jet) - LOG_SQRT_2PI


def weibull_logpdf(
    value: float | Jet2,
    shape: float,
    scale: float,
) -> Jet2:
    """Return the Weibull log density in SciPy/PyMC shape-scale form."""
    value_jet = _promote(value)[0]
    shape_value = float(shape)
    scale_value = float(scale)
    if value_jet.value <= 0.0:
        raise OracleDomainError("Weibull value must be positive")
    if not math.isfinite(shape_value) or shape_value <= 0.0:
        raise OracleDomainError("Weibull shape must be positive and finite")
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise OracleDomainError("Weibull scale must be positive and finite")
    ratio = value_jet / scale_value
    return (
        math.log(shape_value)
        - math.log(scale_value)
        + (shape_value - 1.0) * jet_log(ratio)
        - jet_exp(shape_value * jet_log(ratio))
    )


def truncated_normal_log_normalizer(
    location: float | Jet2,
    scale: float | Jet2,
    bounds: TruncationBounds,
) -> Jet2:
    """Return ``log(Phi(beta) - Phi(alpha))`` with tail-stable branches."""
    location_jet, scale_jet = _promote(location, scale)
    if scale_jet.value <= 0.0:
        raise OracleDomainError("TruncatedNormal scale must be positive")
    alpha = None if bounds.lower is None else (bounds.lower - location_jet) / scale_jet
    beta = None if bounds.upper is None else (bounds.upper - location_jet) / scale_jet
    if alpha is None:
        if beta is None:  # pragma: no cover - validated bounds
            raise AssertionError("unreachable invalid bounds")
        return jet_log_ndtr(beta)
    if beta is None:
        return jet_log_ndtr(-alpha)

    # In the positive tail, subtract survival probabilities.  Elsewhere,
    # subtract CDFs.  This avoids the catastrophic 1 - Phi(x) cancellation.
    if alpha.value >= 0.0:
        return jet_logdiffexp(jet_log_ndtr(-alpha), jet_log_ndtr(-beta))
    return jet_logdiffexp(jet_log_ndtr(beta), jet_log_ndtr(alpha))


def truncated_normal_logpdf(
    value: float | Jet2,
    location: float | Jet2,
    scale: float | Jet2,
    bounds: TruncationBounds,
) -> Jet2:
    """Return a normalized TruncatedNormal log density on open support."""
    value_jet, location_jet, scale_jet = _promote(value, location, scale)
    if not bounds.contains(value_jet.value):
        raise OracleDomainError("TruncatedNormal value lies outside open support")
    return normal_logpdf(value_jet, location_jet, scale_jet) - (
        truncated_normal_log_normalizer(location_jet, scale_jet, bounds)
    )


def truncated_normal_from_standard_normal(
    offset: float | Jet2,
    location: float | Jet2,
    scale: float | Jet2,
    bounds: TruncationBounds,
) -> Jet2:
    """Map a standard-Normal offset through the exact TN inverse CDF.

    The target log CDF and log survival probability are formed independently,
    and the smaller probability is inverted directly from log space.  This
    avoids both ``1 - Phi(x)`` cancellation and ordinary-probability underflow
    while retaining exact second-order dependence on location, scale, and
    offset.
    """
    offset_jet, location_jet, scale_jet = _promote(offset, location, scale)
    if scale_jet.value <= 0.0:
        raise OracleDomainError("TruncatedNormal scale must be positive")

    alpha = None if bounds.lower is None else (bounds.lower - location_jet) / scale_jet
    beta = None if bounds.upper is None else (bounds.upper - location_jet) / scale_jet
    log_quantile = jet_log_ndtr(offset_jet)
    log_quantile_survival = jet_log_ndtr(-offset_jet)

    if alpha is None:
        if beta is None:  # pragma: no cover - validated bounds
            raise AssertionError("unreachable invalid bounds")
        log_target_cdf = log_quantile + jet_log_ndtr(beta)
        log_target_survival = jet_logaddexp(
            log_quantile_survival,
            log_quantile + jet_log_ndtr(-beta),
        )
    elif beta is None:
        log_target_cdf = jet_logaddexp(
            log_quantile_survival + jet_log_ndtr(alpha),
            log_quantile,
        )
        log_target_survival = log_quantile_survival + jet_log_ndtr(-alpha)
    else:
        log_target_cdf = jet_logaddexp(
            log_quantile_survival + jet_log_ndtr(alpha),
            log_quantile + jet_log_ndtr(beta),
        )
        log_target_survival = jet_logaddexp(
            log_quantile_survival + jet_log_ndtr(-alpha),
            log_quantile + jet_log_ndtr(-beta),
        )

    if log_target_cdf.value <= log_target_survival.value:
        standardized = jet_ndtri_exp(log_target_cdf)
    else:
        standardized = -jet_ndtri_exp(log_target_survival)
    natural = location_jet + scale_jet * standardized
    if not bounds.contains(natural.value):
        raise OracleDomainError(
            "finite precision put the TN inverse-CDF value on its boundary"
        )
    return natural


@dataclass(frozen=True, slots=True)
class HierarchicalPosteriorSpec:
    """Natural-scale constants for the centered hierarchical posterior."""

    bounds: TruncationBounds
    location_base_mean: float
    location_prior_scale: float
    scale_prior_shape: float
    scale_prior_scale: float
    n_groups: int
    group_index: NDArray[np.int64]
    observations: NDArray[np.float64]
    observation_scale: float

    def __post_init__(self) -> None:
        """Validate and detach the complete posterior specification."""
        if isinstance(self.n_groups, bool) or not isinstance(self.n_groups, int):
            raise TypeError("n_groups must be an integer")
        if self.n_groups <= 0:
            raise ValueError("n_groups must be positive")
        numeric_constants = {
            "location_base_mean": self.location_base_mean,
            "location_prior_scale": self.location_prior_scale,
            "scale_prior_shape": self.scale_prior_shape,
            "scale_prior_scale": self.scale_prior_scale,
            "observation_scale": self.observation_scale,
        }
        for name, raw_value in numeric_constants.items():
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name != "location_base_mean" and value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

        raw_index = np.asarray(self.group_index)
        if raw_index.ndim != 1 or not np.issubdtype(raw_index.dtype, np.integer):
            raise ValueError("group_index must be a one-dimensional integer array")
        group_index = np.array(raw_index, dtype=np.int64, copy=True)
        observations = np.array(self.observations, dtype=np.float64, copy=True)
        if observations.ndim != 1:
            raise ValueError("observations must be one-dimensional")
        if group_index.size != observations.size:
            raise ValueError("group_index and observations must have equal length")
        if np.any(group_index < 0) or np.any(group_index >= self.n_groups):
            raise ValueError("group_index contains an out-of-range group")
        if not np.all(np.isfinite(observations)):
            raise ValueError("observations must all be finite")
        group_index.setflags(write=False)
        observations.setflags(write=False)
        object.__setattr__(self, "group_index", group_index)
        object.__setattr__(self, "observations", observations)


@dataclass(frozen=True, slots=True)
class PosteriorComponents:
    """Separate transformed-space terms and their summed log posterior."""

    location_prior: Jet2
    scale_prior: Jet2
    group_prior: Jet2
    likelihood: Jet2
    transform_log_jacobian: Jet2

    @property
    def total(self) -> Jet2:
        """Return the complete transformed-coordinate log posterior."""
        return (
            self.location_prior
            + self.scale_prior
            + self.group_prior
            + self.likelihood
            + self.transform_log_jacobian
        )


@dataclass(frozen=True, slots=True)
class NaturalHierarchy:
    """Natural variables induced by one causal parameterization."""

    parameterization: CausalParameterization
    location: Jet2
    scale: Jet2
    group_effect: tuple[Jet2, ...]
    free_rv_log_jacobian: Jet2


def hierarchical_natural_values(
    transformed_point: ArrayLike,
    spec: HierarchicalPosteriorSpec,
    parameterization: CausalParameterization = "centered",
) -> NaturalHierarchy:
    """Map a frozen causal-coordinate vector to the common natural hierarchy."""
    point = np.asarray(transformed_point, dtype=np.float64)
    expected_dimension = spec.n_groups + 2
    if point.shape != (expected_dimension,):
        raise ValueError(
            f"transformed_point must have shape ({expected_dimension},), "
            f"got {point.shape}"
        )
    if parameterization not in {
        "centered",
        "location_icdf_noncentered",
        "group_icdf_noncentered",
        "full_icdf_noncentered",
    }:
        raise ValueError(f"unknown causal parameterization {parameterization!r}")

    coordinates = seed_jets(point)
    scale_transform = positive_transform(coordinates[1])
    free_rv_log_jacobian = scale_transform.log_abs_det_jacobian

    location_noncentered = parameterization in {
        "location_icdf_noncentered",
        "full_icdf_noncentered",
    }
    group_noncentered = parameterization in {
        "group_icdf_noncentered",
        "full_icdf_noncentered",
    }

    if location_noncentered:
        location = truncated_normal_from_standard_normal(
            coordinates[0],
            spec.location_base_mean,
            spec.location_prior_scale,
            spec.bounds,
        )
    else:
        location_transform = support_transform(coordinates[0], spec.bounds)
        location = location_transform.natural
        free_rv_log_jacobian += location_transform.log_abs_det_jacobian

    if not group_noncentered:
        group_transforms = tuple(
            support_transform(coordinate, spec.bounds) for coordinate in coordinates[2:]
        )
        group_effect = tuple(item.natural for item in group_transforms)
        for transformed_group in group_transforms:
            free_rv_log_jacobian += transformed_group.log_abs_det_jacobian
    else:
        group_effect = tuple(
            truncated_normal_from_standard_normal(
                offset,
                location,
                scale_transform.natural,
                spec.bounds,
            )
            for offset in coordinates[2:]
        )

    return NaturalHierarchy(
        parameterization=parameterization,
        location=location,
        scale=scale_transform.natural,
        group_effect=group_effect,
        free_rv_log_jacobian=free_rv_log_jacobian,
    )


def hierarchical_posterior_components(
    transformed_point: ArrayLike,
    spec: HierarchicalPosteriorSpec,
    parameterization: CausalParameterization = "centered",
) -> PosteriorComponents:
    """Evaluate one same-model hierarchy in sampler-visible coordinates.

    Coordinate order is always location, log scale, then one coordinate per
    group.  Centered coordinates use the bounded support transform.  Location-NC
    replaces the location coordinate with a standard-Normal offset; group-NC
    replaces the group coordinates; full-NC replaces both.  The return value
    keeps probability terms separate so a mismatch can be localized before
    inspecting the total.
    """
    point = np.asarray(transformed_point, dtype=np.float64)
    natural = hierarchical_natural_values(point, spec, parameterization)
    expected_dimension = point.size
    coordinates = seed_jets(point)

    location_noncentered = parameterization in {
        "location_icdf_noncentered",
        "full_icdf_noncentered",
    }
    group_noncentered = parameterization in {
        "group_icdf_noncentered",
        "full_icdf_noncentered",
    }

    if location_noncentered:
        location_prior = normal_logpdf(coordinates[0], 0.0, 1.0)
    else:
        location_prior = truncated_normal_logpdf(
            natural.location,
            spec.location_base_mean,
            spec.location_prior_scale,
            spec.bounds,
        )
    scale_prior = weibull_logpdf(
        natural.scale,
        spec.scale_prior_shape,
        spec.scale_prior_scale,
    )
    group_prior = Jet2.constant(0.0, expected_dimension)
    if not group_noncentered:
        for group_effect in natural.group_effect:
            group_prior += truncated_normal_logpdf(
                group_effect,
                natural.location,
                natural.scale,
                spec.bounds,
            )
    else:
        for offset in coordinates[2:]:
            group_prior += normal_logpdf(offset, 0.0, 1.0)

    likelihood = Jet2.constant(0.0, expected_dimension)
    for observation, group in zip(spec.observations, spec.group_index, strict=True):
        likelihood += normal_logpdf(
            float(observation),
            natural.group_effect[int(group)],
            spec.observation_scale,
        )

    return PosteriorComponents(
        location_prior=location_prior,
        scale_prior=scale_prior,
        group_prior=group_prior,
        likelihood=likelihood,
        transform_log_jacobian=natural.free_rv_log_jacobian,
    )

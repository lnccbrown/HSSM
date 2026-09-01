"""Tests for the independent second-order truncated-hierarchy oracle."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit
from scipy.stats import norm, truncnorm, weibull_min

if TYPE_CHECKING:
    from collections.abc import Callable

from scripts.truncated_hierarchy_causal_oracle import (
    CausalParameterization,
    HierarchicalPosteriorSpec,
    Jet2,
    TruncationBounds,
    hierarchical_natural_values,
    hierarchical_posterior_components,
    jet_log_ndtr,
    jet_ndtr,
    jet_ndtri,
    jet_ndtri_exp,
    normal_logpdf,
    positive_inverse,
    positive_transform,
    seed_jets,
    support_inverse,
    support_transform,
    truncated_normal_from_standard_normal,
    truncated_normal_logpdf,
    weibull_logpdf,
)


def _finite_difference_gradient(
    function: Callable[[np.ndarray], float], point: np.ndarray
) -> np.ndarray:
    """Return an independent central-difference gradient."""
    gradient = np.empty_like(point)
    for index in range(point.size):
        step = 2e-6 * max(1.0, abs(float(point[index])))
        displacement = np.zeros_like(point)
        displacement[index] = step
        gradient[index] = (
            function(point + displacement) - function(point - displacement)
        ) / (2.0 * step)
    return gradient


def _finite_difference_hessian(
    function: Callable[[np.ndarray], float], point: np.ndarray
) -> np.ndarray:
    """Return an independent central-difference Hessian."""
    steps = 2e-4 * np.maximum(1.0, np.abs(point))
    hessian = np.empty((point.size, point.size), dtype=np.float64)
    center = function(point)
    for row in range(point.size):
        row_step = np.zeros_like(point)
        row_step[row] = steps[row]
        hessian[row, row] = (
            function(point + row_step) - 2.0 * center + function(point - row_step)
        ) / steps[row] ** 2
        for column in range(row):
            column_step = np.zeros_like(point)
            column_step[column] = steps[column]
            value = (
                function(point + row_step + column_step)
                - function(point + row_step - column_step)
                - function(point - row_step + column_step)
                + function(point - row_step - column_step)
            ) / (4.0 * steps[row] * steps[column])
            hessian[row, column] = value
            hessian[column, row] = value
    return hessian


def _natural_and_log_jacobian(
    unconstrained: float, bounds: TruncationBounds
) -> tuple[float, float]:
    """Evaluate support transforms without using the oracle implementation."""
    if bounds.lower is not None and bounds.upper is None:
        distance = math.exp(unconstrained)
        return bounds.lower + distance, unconstrained
    if bounds.lower is None and bounds.upper is not None:
        distance = math.exp(unconstrained)
        return bounds.upper - distance, unconstrained
    assert bounds.lower is not None
    assert bounds.upper is not None
    width = bounds.upper - bounds.lower
    fraction = float(expit(unconstrained))
    log_jacobian = (
        math.log(width)
        - float(np.logaddexp(0.0, -unconstrained))
        - float(np.logaddexp(0.0, unconstrained))
    )
    return bounds.lower + width * fraction, log_jacobian


def _scipy_truncated_logpdf(
    value: float,
    location: float,
    scale: float,
    bounds: TruncationBounds,
) -> float:
    """Evaluate the corresponding SciPy TruncatedNormal density."""
    standardized_lower = (
        -np.inf if bounds.lower is None else (bounds.lower - location) / scale
    )
    standardized_upper = (
        np.inf if bounds.upper is None else (bounds.upper - location) / scale
    )
    return float(
        truncnorm.logpdf(
            value,
            standardized_lower,
            standardized_upper,
            loc=location,
            scale=scale,
        )
    )


def _scipy_truncated_from_offset(
    offset: float,
    location: float,
    scale: float,
    bounds: TruncationBounds,
) -> float:
    """Push a standard-Normal offset through SciPy's TN quantile."""
    standardized_lower = (
        -np.inf if bounds.lower is None else (bounds.lower - location) / scale
    )
    standardized_upper = (
        np.inf if bounds.upper is None else (bounds.upper - location) / scale
    )
    return float(
        truncnorm.ppf(
            norm.cdf(offset),
            standardized_lower,
            standardized_upper,
            loc=location,
            scale=scale,
        )
    )


def _scipy_offset_from_truncated(
    value: float,
    location: float,
    scale: float,
    bounds: TruncationBounds,
) -> float:
    """Map an interior TN value back to its standard-Normal offset."""
    standardized_lower = (
        -np.inf if bounds.lower is None else (bounds.lower - location) / scale
    )
    standardized_upper = (
        np.inf if bounds.upper is None else (bounds.upper - location) / scale
    )
    quantile = truncnorm.cdf(
        value,
        standardized_lower,
        standardized_upper,
        loc=location,
        scale=scale,
    )
    return float(norm.ppf(quantile))


def _natural_reference(
    point: np.ndarray,
    spec: HierarchicalPosteriorSpec,
    parameterization: CausalParameterization,
) -> tuple[float, float, np.ndarray, float]:
    """Map causal coordinates to natural values using only SciPy and math."""
    scale = math.exp(point[1])
    transform_log_jacobian = point[1]
    location_noncentered = parameterization in {
        "location_icdf_noncentered",
        "full_icdf_noncentered",
    }
    group_noncentered = parameterization in {
        "group_icdf_noncentered",
        "full_icdf_noncentered",
    }
    if location_noncentered:
        location = _scipy_truncated_from_offset(
            point[0],
            spec.location_base_mean,
            spec.location_prior_scale,
            spec.bounds,
        )
    else:
        location, location_jacobian = _natural_and_log_jacobian(point[0], spec.bounds)
        transform_log_jacobian += location_jacobian

    if not group_noncentered:
        groups_and_jacobians = tuple(
            _natural_and_log_jacobian(value, spec.bounds) for value in point[2:]
        )
        groups = np.array([item[0] for item in groups_and_jacobians])
        transform_log_jacobian += sum(item[1] for item in groups_and_jacobians)
    else:
        groups = np.array(
            [
                _scipy_truncated_from_offset(offset, location, scale, spec.bounds)
                for offset in point[2:]
            ]
        )
    return location, scale, groups, transform_log_jacobian


def _point_for_parameterization(
    location: float,
    scale: float,
    groups: list[float],
    spec: HierarchicalPosteriorSpec,
    parameterization: CausalParameterization,
) -> np.ndarray:
    """Construct causal coordinates for one fixed natural hierarchy."""
    location_noncentered = parameterization in {
        "location_icdf_noncentered",
        "full_icdf_noncentered",
    }
    group_noncentered = parameterization in {
        "group_icdf_noncentered",
        "full_icdf_noncentered",
    }
    if location_noncentered:
        location_coordinate = _scipy_offset_from_truncated(
            location,
            spec.location_base_mean,
            spec.location_prior_scale,
            spec.bounds,
        )
    else:
        location_coordinate = support_inverse(location, spec.bounds)
    if not group_noncentered:
        group_coordinates = [support_inverse(group, spec.bounds) for group in groups]
    else:
        group_coordinates = [
            _scipy_offset_from_truncated(group, location, scale, spec.bounds)
            for group in groups
        ]
    return np.array([location_coordinate, positive_inverse(scale), *group_coordinates])


def _posterior_reference(
    point: np.ndarray,
    spec: HierarchicalPosteriorSpec,
    parameterization: CausalParameterization = "centered",
) -> dict[str, float]:
    """Evaluate posterior terms independently with SciPy distributions."""
    location, scale, groups, transform_log_jacobian = _natural_reference(
        point, spec, parameterization
    )
    location_noncentered = parameterization in {
        "location_icdf_noncentered",
        "full_icdf_noncentered",
    }
    group_noncentered = parameterization in {
        "group_icdf_noncentered",
        "full_icdf_noncentered",
    }
    if location_noncentered:
        location_prior = float(norm.logpdf(point[0]))
    else:
        location_prior = _scipy_truncated_logpdf(
            location,
            spec.location_base_mean,
            spec.location_prior_scale,
            spec.bounds,
        )
    scale_prior = float(
        weibull_min.logpdf(
            scale,
            c=spec.scale_prior_shape,
            scale=spec.scale_prior_scale,
        )
    )
    if not group_noncentered:
        group_prior = sum(
            _scipy_truncated_logpdf(group, location, scale, spec.bounds)
            for group in groups
        )
    else:
        group_prior = float(np.sum(norm.logpdf(point[2:])))
    likelihood = float(
        np.sum(
            norm.logpdf(
                spec.observations,
                loc=groups[spec.group_index],
                scale=spec.observation_scale,
            )
        )
    )
    return {
        "location_prior": location_prior,
        "scale_prior": scale_prior,
        "group_prior": group_prior,
        "likelihood": likelihood,
        "transform_log_jacobian": transform_log_jacobian,
    }


def test_second_order_jet_matches_known_mixed_curvature() -> None:
    """Check the jet's product and nonlinear composition rules analytically."""
    x, y = seed_jets([0.7, -0.4])
    result = (x * y + 2.0 * x) ** 3
    inner = 0.7 * -0.4 + 2.0 * 0.7
    inner_gradient = np.array([-0.4 + 2.0, 0.7])
    inner_hessian = np.array([[0.0, 1.0], [1.0, 0.0]])
    expected_hessian = (
        6.0 * inner * np.outer(inner_gradient, inner_gradient)
        + 3.0 * inner**2 * inner_hessian
    )

    assert result.value == pytest.approx(inner**3)
    assert_allclose(result.gradient, 3.0 * inner**2 * inner_gradient)
    assert_allclose(result.hessian, expected_hessian)
    assert_allclose(result.hessian, result.hessian.T)


@pytest.mark.parametrize("value", [-5.0, -2.0, 0.0, 2.0, 5.0])
def test_standard_normal_cdf_quantile_jets_are_inverse(value: float) -> None:
    """Check independent normal CDF and quantile values through second order."""
    coordinate = Jet2.variable(value, 0, 1)
    restored = jet_ndtri(jet_ndtr(coordinate))

    assert restored.value == pytest.approx(value, abs=5e-10)
    assert_allclose(restored.gradient, [1.0], rtol=2e-9, atol=2e-9)
    assert_allclose(restored.hessian, [[0.0]], rtol=0.0, atol=2e-9)


def test_log_normal_quantile_jet_preserves_underflowing_tail() -> None:
    """Round-trip a log CDF whose ordinary probability is exactly zero."""
    coordinate = Jet2.variable(-42.0, 0, 1)
    log_probability = jet_log_ndtr(coordinate)
    restored = jet_ndtri_exp(log_probability)

    assert math.exp(log_probability.value) == 0.0
    assert restored.value == pytest.approx(-42.0, abs=2e-13)
    assert_allclose(restored.gradient, [1.0], rtol=2e-12, atol=2e-12)
    assert_allclose(restored.hessian, [[0.0]], rtol=0.0, atol=5e-12)


@pytest.mark.parametrize(
    ("bounds", "location", "expected"),
    [
        pytest.param(
            TruncationBounds(0.2, None),
            -4.0,
            0.2016490930587551,
            id="lower-alpha-42",
        ),
        pytest.param(
            TruncationBounds(0.2, 0.8),
            -4.0,
            0.2016490930587551,
            id="finite-right-tail",
        ),
        pytest.param(
            TruncationBounds(0.2, 0.8),
            5.0,
            0.7983509069412449,
            id="finite-left-tail",
        ),
    ],
)
def test_truncated_inverse_cdf_is_log_stable_beyond_probability_underflow(
    bounds: TruncationBounds,
    location: float,
    expected: float,
) -> None:
    """Regress alpha=42 and symmetric finite-tail value/derivative failures."""
    point = np.array([0.0, location, math.log(0.1)])

    def evaluate(candidate: np.ndarray):
        offset, location_jet, log_scale = seed_jets(candidate)
        return truncated_normal_from_standard_normal(
            offset,
            location_jet,
            positive_transform(log_scale).natural,
            bounds,
        )

    def scipy_reference(candidate: np.ndarray) -> float:
        offset, candidate_location, log_scale = candidate
        scale = math.exp(log_scale)
        standardized_lower = (
            -np.inf
            if bounds.lower is None
            else (bounds.lower - candidate_location) / scale
        )
        standardized_upper = (
            np.inf
            if bounds.upper is None
            else (bounds.upper - candidate_location) / scale
        )
        return float(
            truncnorm.ppf(
                norm.cdf(offset),
                standardized_lower,
                standardized_upper,
                loc=candidate_location,
                scale=scale,
            )
        )

    result = evaluate(point)

    assert result.value == pytest.approx(expected, rel=0.0, abs=2e-15)
    assert result.value == pytest.approx(scipy_reference(point), abs=2e-15)
    assert np.all(np.isfinite(result.gradient))
    assert np.all(np.isfinite(result.hessian))
    assert_allclose(
        result.gradient,
        _finite_difference_gradient(scipy_reference, point),
        rtol=8e-6,
        atol=3e-9,
    )
    assert_allclose(
        result.hessian,
        _finite_difference_hessian(scipy_reference, point),
        rtol=2e-4,
        atol=5e-8,
    )


@pytest.mark.parametrize(
    ("bounds", "location", "scale"),
    [
        pytest.param(TruncationBounds(0.2, None), 0.0, 0.25, id="lower"),
        pytest.param(TruncationBounds(0.1, 0.9), 0.5, 0.25, id="finite"),
        pytest.param(TruncationBounds(0.1, 0.9), -0.5, 0.1, id="finite-tail"),
    ],
)
@pytest.mark.parametrize("offset", [-4.0, 0.0, 4.0])
def test_truncated_inverse_cdf_matches_scipy_and_change_of_variables(
    bounds: TruncationBounds,
    location: float,
    scale: float,
    offset: float,
) -> None:
    """Verify exact TN quantiles, curvature, and their pushforward Jacobian."""

    def evaluate(candidate: np.ndarray):
        coordinate = Jet2.variable(float(candidate[0]), 0, 1)
        return truncated_normal_from_standard_normal(
            coordinate, location, scale, bounds
        )

    point = np.array([offset])
    result = evaluate(point)
    expected = _scipy_truncated_from_offset(offset, location, scale, bounds)
    log_tn_density = _scipy_truncated_logpdf(expected, location, scale, bounds)
    expected_derivative = math.exp(norm.logpdf(offset) - log_tn_density)
    scalar_function = lambda candidate: evaluate(candidate).value

    assert result.value == pytest.approx(expected, rel=2e-10, abs=2e-12)
    assert result.gradient[0] == pytest.approx(expected_derivative, rel=3e-8, abs=2e-11)
    assert_allclose(
        result.gradient,
        _finite_difference_gradient(scalar_function, point),
        rtol=3e-6,
        atol=2e-8,
    )
    assert_allclose(
        result.hessian,
        _finite_difference_hessian(scalar_function, point),
        rtol=4e-4,
        atol=2e-6,
    )


@pytest.mark.parametrize(
    "bounds",
    [
        pytest.param(TruncationBounds(0.2, None), id="lower"),
        pytest.param(TruncationBounds(None, 0.8), id="upper"),
        pytest.param(TruncationBounds(0.1, 0.9), id="finite"),
    ],
)
@pytest.mark.parametrize("unconstrained", [-15.0, -2.0, 0.0, 3.0, 15.0])
def test_support_transform_round_trip_and_derivatives(
    bounds: TruncationBounds, unconstrained: float
) -> None:
    """Cover both tails of every interval transform and its log Jacobian."""
    coordinate = Jet2.variable(unconstrained, 0, 1)
    transformed = support_transform(coordinate, bounds)
    restored = support_inverse(transformed.natural.value, bounds)

    assert restored == pytest.approx(unconstrained, abs=2e-9)
    assert bounds.contains(transformed.natural.value)

    def natural_function(point: np.ndarray) -> float:
        jet = Jet2.variable(float(point[0]), 0, 1)
        return support_transform(jet, bounds).natural.value

    def jacobian_function(point: np.ndarray) -> float:
        jet = Jet2.variable(float(point[0]), 0, 1)
        return support_transform(jet, bounds).log_abs_det_jacobian.value

    point = np.array([unconstrained])
    assert_allclose(
        transformed.natural.gradient,
        _finite_difference_gradient(natural_function, point),
        rtol=2e-7,
        atol=2e-10,
    )
    assert_allclose(
        transformed.log_abs_det_jacobian.hessian,
        _finite_difference_hessian(jacobian_function, point),
        rtol=2e-5,
        atol=2e-8,
    )


@pytest.mark.parametrize("natural", [1e-8, 0.03, 1.0, 200.0])
def test_positive_transform_round_trip(natural: float) -> None:
    """Check the log transform across several scale regimes."""
    unconstrained = positive_inverse(natural)
    transformed = positive_transform(Jet2.variable(unconstrained, 0, 1))

    assert transformed.natural.value == pytest.approx(natural)
    assert transformed.log_abs_det_jacobian.value == pytest.approx(unconstrained)
    assert_allclose(transformed.natural.gradient, [natural])
    assert_allclose(transformed.natural.hessian, [[natural]])
    assert_allclose(transformed.log_abs_det_jacobian.gradient, [1.0])
    assert_allclose(transformed.log_abs_det_jacobian.hessian, [[0.0]])


@pytest.mark.parametrize(
    ("bounds", "value", "location", "scale"),
    [
        pytest.param(TruncationBounds(0.2, None), 0.200001, 0.0, 0.03, id="lower-tail"),
        pytest.param(TruncationBounds(None, 0.8), 0.799999, 1.0, 0.03, id="upper-tail"),
        pytest.param(
            TruncationBounds(0.0, 1.0), 0.01, -2.0, 0.1, id="finite-right-tail"
        ),
        pytest.param(
            TruncationBounds(-1.0, 0.0), -0.01, 2.0, 0.1, id="finite-left-tail"
        ),
        pytest.param(
            TruncationBounds(0.1, 0.9), 0.100001, 0.5, 100.0, id="broad-scale"
        ),
    ],
)
def test_truncated_normal_matches_scipy_in_boundary_and_scale_regimes(
    bounds: TruncationBounds, value: float, location: float, scale: float
) -> None:
    """Exercise stable one- and two-sided normalizers against SciPy."""
    actual = truncated_normal_logpdf(value, location, scale, bounds).value
    expected = _scipy_truncated_logpdf(value, location, scale, bounds)

    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-12)


@pytest.mark.parametrize(
    ("bounds", "natural_value", "location", "natural_scale"),
    [
        pytest.param(TruncationBounds(0.2, None), 0.2002, 0.23, 0.015, id="lower-near"),
        pytest.param(TruncationBounds(0.1, 0.9), 0.1002, 0.13, 0.06, id="finite-near"),
        pytest.param(TruncationBounds(0.1, 0.9), 0.55, 0.45, 0.8, id="finite-broad"),
    ],
)
def test_transformed_truncated_normal_gradient_and_hessian_match_differences(
    bounds: TruncationBounds,
    natural_value: float,
    location: float,
    natural_scale: float,
) -> None:
    """Verify TN value, gradient, and curvature in transformed coordinates."""
    point = np.array(
        [support_inverse(natural_value, bounds), location, math.log(natural_scale)]
    )

    def evaluate(candidate: np.ndarray):
        value_coordinate, location_jet, scale_coordinate = seed_jets(candidate)
        value_transform = support_transform(value_coordinate, bounds)
        scale_transform = positive_transform(scale_coordinate)
        return (
            truncated_normal_logpdf(
                value_transform.natural,
                location_jet,
                scale_transform.natural,
                bounds,
            )
            + value_transform.log_abs_det_jacobian
            + scale_transform.log_abs_det_jacobian
        )

    result = evaluate(point)
    scalar_function = lambda candidate: evaluate(candidate).value

    assert_allclose(
        result.gradient,
        _finite_difference_gradient(scalar_function, point),
        rtol=2e-5,
        atol=2e-5,
    )
    assert_allclose(
        result.hessian,
        _finite_difference_hessian(scalar_function, point),
        rtol=7e-4,
        atol=5e-4,
    )


def test_normal_and_weibull_components_match_scipy_and_finite_differences() -> None:
    """Check the remaining posterior distributions independently."""
    point = np.array([0.4, -1.7])

    def evaluate(candidate: np.ndarray):
        location, log_scale = seed_jets(candidate)
        scale_transform = positive_transform(log_scale)
        return (
            normal_logpdf(0.7, location, 0.3)
            + weibull_logpdf(scale_transform.natural, 1.5, 0.3)
            + scale_transform.log_abs_det_jacobian
        )

    result = evaluate(point)
    natural_scale = math.exp(point[1])
    expected = (
        norm.logpdf(0.7, loc=point[0], scale=0.3)
        + weibull_min.logpdf(natural_scale, c=1.5, scale=0.3)
        + point[1]
    )
    scalar_function = lambda candidate: evaluate(candidate).value

    assert result.value == pytest.approx(expected, rel=1e-13, abs=1e-13)
    assert_allclose(
        result.gradient,
        _finite_difference_gradient(scalar_function, point),
        rtol=2e-6,
        atol=2e-7,
    )
    assert_allclose(
        result.hessian,
        _finite_difference_hessian(scalar_function, point),
        rtol=2e-6,
        atol=2e-7,
    )


POSTERIOR_CASES = (
    pytest.param(
        TruncationBounds(0.2, None),
        0.205,
        0.015,
        [0.2008, 0.207, 0.23],
        0.0,
        id="lower-boundary-small-scale",
    ),
    pytest.param(
        TruncationBounds(0.2, None),
        0.9,
        0.6,
        [0.35, 0.8, 1.6],
        0.5,
        id="lower-interior-broad-scale",
    ),
    pytest.param(
        TruncationBounds(0.1, 0.9),
        0.13,
        0.06,
        [0.1008, 0.14, 0.27],
        0.5,
        id="finite-boundary-small-scale",
    ),
    pytest.param(
        TruncationBounds(0.1, 0.9),
        0.5,
        0.5,
        [0.2, 0.55, 0.82],
        0.5,
        id="finite-interior-broad-scale",
    ),
)


def _posterior_spec(
    bounds: TruncationBounds,
    groups: list[float],
    base_mean: float,
) -> HierarchicalPosteriorSpec:
    """Build the common deterministic full-posterior test fixture."""
    return HierarchicalPosteriorSpec(
        bounds=bounds,
        location_base_mean=base_mean,
        location_prior_scale=0.25,
        scale_prior_shape=1.5,
        scale_prior_scale=0.3,
        n_groups=3,
        group_index=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        observations=np.array(
            [
                groups[0] - 0.02,
                groups[0] + 0.01,
                groups[1],
                groups[1] + 0.03,
                groups[2] - 0.01,
                groups[2] + 0.02,
            ]
        ),
        observation_scale=0.5,
    )


@pytest.mark.parametrize(
    ("bounds", "location", "scale", "groups", "base_mean"), POSTERIOR_CASES
)
def test_hierarchical_posterior_matches_scipy_and_finite_differences(
    bounds: TruncationBounds,
    location: float,
    scale: float,
    groups: list[float],
    base_mean: float,
) -> None:
    """Validate every term and total Hessian for causal study regimes."""
    spec = _posterior_spec(bounds, groups, base_mean)
    point = _point_for_parameterization(location, scale, groups, spec, "centered")
    components = hierarchical_posterior_components(point, spec)
    reference = _posterior_reference(point, spec)

    for name, expected in reference.items():
        actual = getattr(components, name).value
        assert actual == pytest.approx(expected, rel=3e-11, abs=3e-11), name
    assert components.total.value == pytest.approx(
        sum(reference.values()), rel=3e-11, abs=3e-11
    )

    scalar_function = lambda candidate: (
        hierarchical_posterior_components(candidate, spec).total.value
    )
    assert_allclose(
        components.total.gradient,
        _finite_difference_gradient(scalar_function, point),
        rtol=8e-5,
        atol=5e-5,
    )
    assert_allclose(
        components.total.hessian,
        _finite_difference_hessian(scalar_function, point),
        rtol=1.5e-3,
        atol=2e-3,
    )
    assert_allclose(components.total.hessian, components.total.hessian.T, atol=1e-12)


@pytest.mark.parametrize(
    "parameterization",
    [
        "location_icdf_noncentered",
        "group_icdf_noncentered",
        "full_icdf_noncentered",
    ],
)
@pytest.mark.parametrize(
    ("bounds", "location", "scale", "groups", "base_mean"), POSTERIOR_CASES
)
def test_noncentered_posterior_matches_scipy_and_finite_differences(
    parameterization: CausalParameterization,
    bounds: TruncationBounds,
    location: float,
    scale: float,
    groups: list[float],
    base_mean: float,
) -> None:
    """Validate every exact ICDF parameterization through second order."""
    spec = _posterior_spec(bounds, groups, base_mean)
    point = _point_for_parameterization(location, scale, groups, spec, parameterization)
    components = hierarchical_posterior_components(point, spec, parameterization)
    reference = _posterior_reference(point, spec, parameterization)

    for name, expected in reference.items():
        actual = getattr(components, name).value
        assert actual == pytest.approx(expected, rel=2e-9, abs=2e-9), name
    assert components.total.value == pytest.approx(
        sum(reference.values()), rel=2e-9, abs=2e-9
    )

    scalar_function = lambda candidate: (
        hierarchical_posterior_components(candidate, spec, parameterization).total.value
    )
    assert_allclose(
        components.total.gradient,
        _finite_difference_gradient(scalar_function, point),
        rtol=2e-4,
        atol=8e-5,
    )
    assert_allclose(
        components.total.hessian,
        _finite_difference_hessian(scalar_function, point),
        rtol=3e-3,
        atol=3e-3,
    )
    assert_allclose(components.total.hessian, components.total.hessian.T, atol=1e-12)


@pytest.mark.parametrize(
    "parameterization",
    [
        "location_icdf_noncentered",
        "group_icdf_noncentered",
        "full_icdf_noncentered",
    ],
)
@pytest.mark.parametrize(
    ("bounds", "location", "scale", "groups", "base_mean"),
    [POSTERIOR_CASES[0], POSTERIOR_CASES[2], POSTERIOR_CASES[3]],
)
def test_parameterizations_encode_the_same_natural_model_and_measure(
    parameterization: CausalParameterization,
    bounds: TruncationBounds,
    location: float,
    scale: float,
    groups: list[float],
    base_mean: float,
) -> None:
    """Check natural equality and the exact coordinate-density Jacobian."""
    spec = _posterior_spec(bounds, groups, base_mean)
    centered_point = _point_for_parameterization(
        location, scale, groups, spec, "centered"
    )
    noncentered_point = _point_for_parameterization(
        location, scale, groups, spec, parameterization
    )
    centered_natural = hierarchical_natural_values(centered_point, spec, "centered")
    noncentered_natural = hierarchical_natural_values(
        noncentered_point, spec, parameterization
    )

    assert noncentered_natural.location.value == pytest.approx(
        centered_natural.location.value, rel=2e-10, abs=2e-12
    )
    assert noncentered_natural.scale.value == pytest.approx(
        centered_natural.scale.value, rel=2e-13, abs=2e-13
    )
    assert_allclose(
        [item.value for item in noncentered_natural.group_effect],
        [item.value for item in centered_natural.group_effect],
        rtol=2e-9,
        atol=2e-11,
    )

    log_coordinate_jacobian = 0.0
    if parameterization in {
        "location_icdf_noncentered",
        "full_icdf_noncentered",
    }:
        _, centered_location_jacobian = _natural_and_log_jacobian(
            centered_point[0], bounds
        )
        location_tn_logpdf = _scipy_truncated_logpdf(
            location,
            spec.location_base_mean,
            spec.location_prior_scale,
            bounds,
        )
        log_coordinate_jacobian += (
            norm.logpdf(noncentered_point[0])
            - location_tn_logpdf
            - centered_location_jacobian
        )
    if parameterization in {
        "group_icdf_noncentered",
        "full_icdf_noncentered",
    }:
        for index, group in enumerate(groups):
            _, centered_group_jacobian = _natural_and_log_jacobian(
                centered_point[index + 2], bounds
            )
            group_tn_logpdf = _scipy_truncated_logpdf(group, location, scale, bounds)
            log_coordinate_jacobian += (
                norm.logpdf(noncentered_point[index + 2])
                - group_tn_logpdf
                - centered_group_jacobian
            )

    centered_logp = hierarchical_posterior_components(
        centered_point, spec, "centered"
    ).total.value
    noncentered_logp = hierarchical_posterior_components(
        noncentered_point, spec, parameterization
    ).total.value
    assert noncentered_logp == pytest.approx(
        centered_logp + log_coordinate_jacobian,
        rel=3e-9,
        abs=3e-9,
    )

"""Tests for exact same-model TruncatedNormal causal parameterizations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pymc.sampling.jax import get_jaxified_graph, get_jaxified_logp
from pytensor.compile.mode import Mode
from scipy.stats import truncnorm

from scripts.truncated_hierarchy_causal_models import (
    CausalModelContractError,
    _truncated_normal_from_standard_normal,
    bounded_from_unconstrained,
    bounded_log_jacobian,
    build_full_icdf_noncentered,
    build_group_icdf_noncentered,
    build_location_icdf_noncentered,
    build_manual_centered,
    build_native_centered,
    manual_truncated_normal_logp,
    truncated_normal_icdf,
)
from scripts.truncated_hierarchy_causal_oracle import (
    HierarchicalPosteriorSpec,
    Jet2,
    TruncationBounds,
    hierarchical_posterior_components,
    truncated_normal_from_standard_normal,
)
from scripts.truncated_hierarchy_models import (
    Bounds,
    NativeTruncatedPrior,
    ToyDataSpec,
    generate_synthetic_data,
)

if TYPE_CHECKING:
    from collections.abc import Callable

BUILDERS: tuple[tuple[str, Callable], ...] = (
    ("native", build_native_centered),
    ("manual", build_manual_centered),
    ("location-icdf", build_location_icdf_noncentered),
    ("group-icdf", build_group_icdf_noncentered),
    ("full-icdf", build_full_icdf_noncentered),
)

ORACLE_VARIANTS = (
    pytest.param(
        "native",
        build_native_centered,
        "centered",
        (
            "group_location_rv_interval__",
            "group_scale_rv_log__",
            "group_effect_rv_interval__",
        ),
        id="native",
    ),
    pytest.param(
        "manual",
        build_manual_centered,
        "centered",
        (
            "group_location_coordinate",
            "group_scale_rv_log__",
            "group_effect_coordinate",
        ),
        id="manual",
    ),
    pytest.param(
        "location-icdf",
        build_location_icdf_noncentered,
        "location_icdf_noncentered",
        (
            "group_location_offset",
            "group_scale_rv_log__",
            "group_effect_rv_interval__",
        ),
        id="location-icdf",
    ),
    pytest.param(
        "group-icdf",
        build_group_icdf_noncentered,
        "group_icdf_noncentered",
        (
            "group_location_rv_interval__",
            "group_scale_rv_log__",
            "group_effect_offset",
        ),
        id="group-icdf",
    ),
    pytest.param(
        "full-icdf",
        build_full_icdf_noncentered,
        "full_icdf_noncentered",
        (
            "group_location_offset",
            "group_scale_rv_log__",
            "group_effect_offset",
        ),
        id="full-icdf",
    ),
)

BOUND_CASES = (
    pytest.param(Bounds(0.2, None), 0.0, 0.23, id="lower-only"),
    pytest.param(Bounds(0.1, 0.9), 0.5, 0.13, id="finite"),
)

EXTREME_TAIL_CASES = (
    pytest.param(Bounds(0.2, None), -4.0, 0.1, id="lower-alpha-42"),
    pytest.param(Bounds(0.2, 0.21), -4.0, 0.1, id="finite-right-tail"),
    pytest.param(Bounds(-0.21, -0.2), 4.0, 0.1, id="finite-left-tail"),
)


def _inputs(bounds: Bounds, base_mean: float, location: float):
    spec = ToyDataSpec(
        bounds=bounds,
        group_location=location,
        group_scale=0.3,
        n_groups=3,
        n_per_group=2,
    )
    data = generate_synthetic_data(spec, group_seed=1282, observation_seed=1283)
    return NativeTruncatedPrior(bounds, base_mean), data


def _eval(function, *values):
    return np.asarray(function(*values), dtype=np.float64)


def _oracle_spec(prior, data) -> HierarchicalPosteriorSpec:
    return HierarchicalPosteriorSpec(
        bounds=TruncationBounds(prior.bounds.lower, prior.bounds.upper),
        location_base_mean=prior.location_base_mean,
        location_prior_scale=prior.location_prior_sigma,
        scale_prior_shape=prior.scale_prior_alpha,
        scale_prior_scale=prior.scale_prior_beta,
        n_groups=data.spec.n_groups,
        group_index=data.group_index,
        observations=data.y,
        observation_scale=data.spec.observation_sigma,
    )


def _model_arguments(model, vector: np.ndarray) -> list[np.ndarray]:
    """Split a canonical location/scale/groups vector in value-variable order."""
    initial_point = model.initial_point()
    arguments: list[np.ndarray] = []
    cursor = 0
    for variable in model.value_vars:
        initial = np.asarray(initial_point[variable.name])
        size = initial.size
        arguments.append(vector[cursor : cursor + size].reshape(initial.shape))
        cursor += size
    assert cursor == vector.size
    return arguments


@pytest.mark.parametrize(("bounds", "base_mean", "location"), BOUND_CASES)
@pytest.mark.parametrize(("name", "builder"), BUILDERS)
def test_all_parameterizations_build_the_same_natural_surface(
    bounds, base_mean, location, name, builder
) -> None:
    """Expose identical natural names and the frozen Weibull scale prior."""
    prior, data = _inputs(bounds, base_mean, location)
    model = builder(prior, data)

    deterministic_names = {variable.name for variable in model.deterministics}
    assert {"group_location", "group_scale", "group_effect"} <= deterministic_names
    assert model.named_vars_to_dims["group_effect"] == ("group",)
    assert len(model.coords["group"]) == 3
    assert type(model.named_vars["group_scale_rv"].owner.op).__name__ == (
        "WeibullBetaRV"
    )
    assert {variable.name for variable in model.observed_RVs} == {"y"}

    free_names = {variable.name for variable in model.free_RVs}
    if name == "native":
        assert free_names == {
            "group_location_rv",
            "group_scale_rv",
            "group_effect_rv",
        }
    elif name == "manual":
        assert free_names == {
            "group_location_coordinate",
            "group_scale_rv",
            "group_effect_coordinate",
        }
    elif name == "location-icdf":
        assert free_names == {
            "group_location_offset",
            "group_scale_rv",
            "group_effect_rv",
        }
        assert type(model.named_vars["group_location_offset"].owner.op).__name__ == (
            "NormalRV"
        )
    elif name == "group-icdf":
        assert free_names == {
            "group_location_rv",
            "group_scale_rv",
            "group_effect_offset",
        }
        assert type(model.named_vars["group_effect_offset"].owner.op).__name__ == (
            "NormalRV"
        )
    else:
        assert free_names == {
            "group_location_offset",
            "group_scale_rv",
            "group_effect_offset",
        }
        assert type(model.named_vars["group_location_offset"].owner.op).__name__ == (
            "NormalRV"
        )
        assert type(model.named_vars["group_effect_offset"].owner.op).__name__ == (
            "NormalRV"
        )


@pytest.mark.parametrize(("bounds", "base_mean", "location"), BOUND_CASES)
def test_manual_centered_matches_native_transformed_logp(
    bounds, base_mean, location
) -> None:
    """Compare complete log densities at identical transformed coordinates."""
    prior, data = _inputs(bounds, base_mean, location)
    native = build_native_centered(prior, data)
    manual = build_manual_centered(prior, data)
    native_point = native.initial_point()
    manual_point = manual.initial_point()

    location_coordinate = np.asarray(-0.35)
    effect_coordinate = np.asarray([-0.8, 0.1, 0.9])
    scale_coordinate = np.asarray(-1.1)
    native_point["group_location_rv_interval__"] = location_coordinate
    native_point["group_effect_rv_interval__"] = effect_coordinate
    native_point["group_scale_rv_log__"] = scale_coordinate
    manual_point["group_location_coordinate"] = location_coordinate
    manual_point["group_effect_coordinate"] = effect_coordinate
    manual_point["group_scale_rv_log__"] = scale_coordinate

    native_logp = native.compile_logp()(native_point)
    manual_logp = manual.compile_logp()(manual_point)
    np.testing.assert_allclose(manual_logp, native_logp, rtol=5e-9, atol=5e-8)


@pytest.mark.parametrize(("bounds", "base_mean", "location"), BOUND_CASES)
@pytest.mark.parametrize(
    ("_name", "builder", "oracle_parameterization", "value_names"),
    ORACLE_VARIANTS,
)
def test_transformed_posterior_matches_independent_second_order_oracle(
    bounds,
    base_mean,
    location,
    _name,
    builder,
    oracle_parameterization,
    value_names,
) -> None:
    """Match value, gradient, and Hessian in the oracle's exact coordinate order."""
    prior, data = _inputs(bounds, base_mean, location)
    model = builder(prior, data)
    assert tuple(variable.name for variable in model.value_vars) == value_names

    # The oracle freezes this order for every variant: location coordinate,
    # log scale, then one centered coordinate or standard-Normal offset per group.
    point = np.array([-0.45, math.log(0.22), -1.1, 0.15, 0.85])
    arguments = _model_arguments(model, point)
    model_logp = model.logp(jacobian=True)
    model_gradient = model.dlogp(jacobian=True)
    model_hessian = model.d2logp(
        jacobian=True,
        negate_output=False,
    )
    pytensor_function = model.compile_fn(
        [model_logp, model_gradient, model_hessian],
        inputs=model.value_vars,
        mode=Mode(linker="py", optimizer=None),
        on_unused_input="ignore",
        point_fn=False,
    )
    observed_value, observed_gradient, observed_hessian = pytensor_function(*arguments)

    expected = hierarchical_posterior_components(
        point,
        _oracle_spec(prior, data),
        parameterization=oracle_parameterization,
    ).total
    np.testing.assert_allclose(observed_value, expected.value, rtol=2e-8, atol=5e-8)
    np.testing.assert_allclose(
        observed_gradient, expected.gradient, rtol=3e-7, atol=3e-7
    )
    np.testing.assert_allclose(observed_hessian, expected.hessian, rtol=3e-6, atol=3e-6)

    # NumPyro differentiates one JAXified scalar logp.  Differentiate that same
    # scalar here rather than merely lowering PyTensor's precomputed gradient.
    jaxified_logp = get_jaxified_graph(
        inputs=model.value_vars,
        outputs=[model_logp],
    )
    initial_point = model.initial_point()
    layouts = []
    cursor = 0
    for variable in model.value_vars:
        shape = np.asarray(initial_point[variable.name]).shape
        size = int(np.prod(shape, dtype=int)) if shape else 1
        layouts.append((cursor, cursor + size, shape))
        cursor += size
    assert cursor == point.size

    def scalar_jax_logp(vector):
        ordered = [vector[start:stop].reshape(shape) for start, stop, shape in layouts]
        return jaxified_logp(*ordered)[0]

    jax_point = jnp.asarray(point)
    jax_value, jax_gradient = jax.value_and_grad(scalar_jax_logp)(jax_point)
    jax_hessian = jax.hessian(scalar_jax_logp)(jax_point)
    np.testing.assert_allclose(jax_value, expected.value, rtol=2e-8, atol=5e-8)
    np.testing.assert_allclose(jax_gradient, expected.gradient, rtol=3e-7, atol=3e-7)
    np.testing.assert_allclose(jax_hessian, expected.hessian, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize(("bounds", "base_mean", "_location"), BOUND_CASES)
def test_manual_density_matches_scipy(bounds, base_mean, _location) -> None:
    """Check the independently normalized density at interior natural values."""
    value = pt.dvector("value")
    sigma = pt.dscalar("sigma")
    logp = manual_truncated_normal_logp(value, mu=base_mean, sigma=sigma, bounds=bounds)
    function = pytensor.function([value, sigma], logp)
    if bounds.upper is None:
        values = np.array([0.201, 0.3, 1.2])
    else:
        values = np.array([0.101, 0.4, 0.899])
    scale = 0.25
    lower = (bounds.lower - base_mean) / scale
    upper = np.inf if bounds.upper is None else (bounds.upper - base_mean) / scale
    expected = truncnorm.logpdf(values, lower, upper, loc=base_mean, scale=scale)

    np.testing.assert_allclose(
        function(values, scale), expected, rtol=1e-11, atol=1e-11
    )


@pytest.mark.parametrize(("bounds", "base_mean", "_location"), BOUND_CASES)
def test_inverse_cdf_is_monotone_accurate_and_has_the_target_density(
    bounds, base_mean, _location
) -> None:
    """Exercise central and tail quantiles plus the inverse-Jacobian density."""
    quantile = pt.dvector("quantile")
    scale = 0.25
    natural = truncated_normal_icdf(quantile, mu=base_mean, sigma=scale, bounds=bounds)
    derivative = pt.grad(pt.sum(natural), quantile)
    function = pytensor.function([quantile], [natural, derivative])
    probabilities = np.array([1e-10, 1e-6, 0.01, 0.5, 0.99, 1 - 1e-6, 1 - 1e-10])
    observed, derivatives = function(probabilities)

    lower = (bounds.lower - base_mean) / scale
    upper = np.inf if bounds.upper is None else (bounds.upper - base_mean) / scale
    expected = truncnorm.ppf(probabilities, lower, upper, loc=base_mean, scale=scale)
    expected_logp = truncnorm.logpdf(observed, lower, upper, loc=base_mean, scale=scale)

    assert np.all(np.isfinite(observed))
    assert np.all(np.diff(observed) > 0)
    assert np.all(observed > bounds.lower)
    if bounds.upper is not None:
        assert np.all(observed < bounds.upper)
    np.testing.assert_allclose(observed, expected, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(
        -np.log(derivatives), expected_logp, rtol=2e-7, atol=2e-7
    )


@pytest.mark.parametrize(("bounds", "base_mean", "scale"), EXTREME_TAIL_CASES)
def test_extreme_tail_icdf_and_density_match_scipy_in_pytensor_and_jax(
    bounds, base_mean, scale
) -> None:
    """Keep the causal alternatives valid after ordinary CDF values underflow."""
    quantile = pt.dscalar("quantile")
    value = pt.dscalar("value")
    offset = pt.dscalar("offset")
    natural = truncated_normal_icdf(
        quantile,
        mu=base_mean,
        sigma=scale,
        bounds=bounds,
    )
    logp = manual_truncated_normal_logp(
        value,
        mu=base_mean,
        sigma=scale,
        bounds=bounds,
    )
    offset_natural = _truncated_normal_from_standard_normal(
        offset,
        mu=base_mean,
        sigma=scale,
        bounds=bounds,
    )
    offset_derivative = pt.grad(offset_natural, offset)
    offset_second_derivative = pt.grad(offset_derivative, offset)
    outputs = [
        natural,
        logp,
        offset_natural,
        offset_derivative,
        offset_second_derivative,
    ]
    pytensor_function = pytensor.function([quantile, value, offset], outputs)
    jax_function = pytensor.function(
        [quantile, value, offset],
        outputs,
        mode="JAX",
    )

    probability = 0.5
    lower = (bounds.lower - base_mean) / scale
    upper = np.inf if bounds.upper is None else (bounds.upper - base_mean) / scale
    expected_value = truncnorm.ppf(
        probability,
        lower,
        upper,
        loc=base_mean,
        scale=scale,
    )
    expected_logp = truncnorm.logpdf(
        expected_value,
        lower,
        upper,
        loc=base_mean,
        scale=scale,
    )
    expected_offset_derivative = math.exp(
        -0.5 * math.log(2.0 * math.pi) - expected_logp
    )
    oracle_offset = truncated_normal_from_standard_normal(
        Jet2.variable(0.0, index=0, dimension=1),
        base_mean,
        scale,
        TruncationBounds(bounds.lower, bounds.upper),
    )
    expected_offset_second_derivative = oracle_offset.hessian[0, 0]

    for function in (pytensor_function, jax_function):
        (
            observed_value,
            observed_logp,
            observed_offset_value,
            observed_offset_derivative,
            observed_offset_second_derivative,
        ) = function(probability, expected_value, 0.0)
        np.testing.assert_allclose(
            observed_value, expected_value, rtol=1e-11, atol=1e-12
        )
        np.testing.assert_allclose(observed_logp, expected_logp, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(
            observed_offset_value, expected_value, rtol=1e-11, atol=1e-12
        )
        np.testing.assert_allclose(
            observed_offset_derivative,
            expected_offset_derivative,
            rtol=2e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            observed_offset_second_derivative,
            expected_offset_second_derivative,
            rtol=2e-9,
            atol=1e-12,
        )
        assert observed_offset_derivative > 0.0

    # NumPyro differentiates the scalar JAXified transform itself.  Exercise
    # that route, not only a PyTensor-derived gradient lowered into JAX.
    jaxified_offset = get_jaxified_graph(inputs=[offset], outputs=[offset_natural])

    def scalar_jax_offset(offset_value):
        return jaxified_offset(offset_value)[0]

    jax_offset = jnp.asarray(0.0)
    jax_value, jax_derivative = jax.value_and_grad(scalar_jax_offset)(jax_offset)
    jax_second_derivative = jax.hessian(scalar_jax_offset)(jax_offset)
    np.testing.assert_allclose(jax_value, expected_value, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        jax_derivative,
        expected_offset_derivative,
        rtol=2e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jax_second_derivative,
        expected_offset_second_derivative,
        rtol=2e-9,
        atol=1e-12,
    )
    assert jax_derivative > 0.0


@pytest.mark.parametrize(("bounds", "base_mean", "_location"), BOUND_CASES)
def test_standard_normal_offset_induces_the_target_natural_density(
    bounds, base_mean, _location
) -> None:
    """Verify the exact non-centering change of variables, including tails."""
    offset = pt.dvector("offset")
    scale = 0.25
    quantile = 0.5 * pt.erfc(-offset / np.sqrt(2.0))
    quantile_survival = 0.5 * pt.erfc(offset / np.sqrt(2.0))
    natural = truncated_normal_icdf(
        quantile,
        mu=base_mean,
        sigma=scale,
        bounds=bounds,
        quantile_survival=quantile_survival,
    )
    derivative = pt.grad(pt.sum(natural), offset)
    function = pytensor.function([offset], [natural, derivative])
    offsets = np.array([-6.0, -2.0, 0.0, 2.0, 6.0])
    observed, derivatives = function(offsets)

    lower = (bounds.lower - base_mean) / scale
    upper = np.inf if bounds.upper is None else (bounds.upper - base_mean) / scale
    induced_logp = -0.5 * offsets**2 - 0.5 * np.log(2.0 * np.pi) - np.log(derivatives)
    expected_logp = truncnorm.logpdf(observed, lower, upper, loc=base_mean, scale=scale)

    assert np.all(np.diff(observed) > 0)
    np.testing.assert_allclose(induced_logp, expected_logp, rtol=3e-7, atol=3e-7)


@pytest.mark.parametrize(("bounds", "_base_mean", "_location"), BOUND_CASES)
def test_manual_coordinate_transform_and_jacobian(
    bounds, _base_mean, _location
) -> None:
    """Verify support, monotonicity, and the explicit coordinate Jacobian."""
    coordinate = pt.dvector("coordinate")
    natural = bounded_from_unconstrained(coordinate, bounds)
    derivative = pt.grad(pt.sum(natural), coordinate)
    log_jacobian = bounded_log_jacobian(coordinate, bounds)
    function = pytensor.function([coordinate], [natural, derivative, log_jacobian])
    values, derivatives, observed_log_jacobian = function(
        np.array([-8.0, -1.0, 0.0, 1.0, 8.0])
    )

    assert np.all(np.diff(values) > 0)
    assert np.all(values > bounds.lower)
    if bounds.upper is not None:
        assert np.all(values < bounds.upper)
    np.testing.assert_allclose(
        observed_log_jacobian, np.log(derivatives), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(("bounds", "base_mean", "location"), BOUND_CASES)
@pytest.mark.parametrize(("_name", "builder"), BUILDERS)
def test_model_logp_compiles_for_pytensor(
    bounds, base_mean, location, _name, builder
) -> None:
    """Compile and evaluate every parameterization through PyTensor."""
    prior, data = _inputs(bounds, base_mean, location)
    model = builder(prior, data)
    point = model.initial_point()

    pytensor_logp = np.asarray(model.compile_logp()(point))
    assert np.isfinite(pytensor_logp)


@pytest.mark.parametrize(("bounds", "base_mean", "location"), BOUND_CASES)
@pytest.mark.parametrize(("_name", "builder"), BUILDERS[1:])
def test_manual_and_icdf_logp_compile_for_numpyro_jax_path(
    bounds, base_mean, location, _name, builder
) -> None:
    """Compile the independent candidate graphs through NumPyro's JAX path."""
    prior, data = _inputs(bounds, base_mean, location)
    model = builder(prior, data)
    point = model.initial_point()

    pytensor_logp = np.asarray(model.compile_logp()(point))
    jax_logp = get_jaxified_logp(model)(
        [point[variable.name] for variable in model.value_vars]
    )
    assert np.isfinite(pytensor_logp)
    assert np.isfinite(np.asarray(jax_logp))
    np.testing.assert_allclose(jax_logp, pytensor_logp, rtol=2e-8, atol=2e-8)


def test_inverse_cdf_primitive_lowers_to_jax_without_tfp_fallback() -> None:
    """Prove that the independent quantile uses the NumPyro-compatible path."""
    quantile = pt.dvector("quantile")
    natural = truncated_normal_icdf(
        quantile, mu=0.0, sigma=0.25, bounds=Bounds(0.2, None)
    )
    jax_function = get_jaxified_graph(inputs=[quantile], outputs=[natural])
    values = np.array([1e-6, 0.5, 1 - 1e-6])

    observed = _eval(jax_function, values)[0]
    assert np.all(np.isfinite(observed))
    assert np.all(np.diff(observed) > 0)


def test_causal_builders_reject_upper_only_and_scale_prior_drift() -> None:
    """Fail fast when a caller leaves the frozen same-model comparison."""
    upper_bounds = Bounds(None, 0.8)
    upper_prior, upper_data = _inputs(upper_bounds, 0.5, 0.7)
    with pytest.raises(CausalModelContractError, match="finite lower"):
        build_native_centered(upper_prior, upper_data)

    bounds = Bounds(0.2, None)
    _prior, data = _inputs(bounds, 0.0, 0.23)
    changed_prior = NativeTruncatedPrior(bounds, 0.0, scale_prior_beta=0.4)
    with pytest.raises(CausalModelContractError, match="beta must remain frozen"):
        build_manual_centered(changed_prior, data)

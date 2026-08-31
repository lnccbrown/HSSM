"""Tests for scenario-independent bounded-hierarchy geometry primitives."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Literal

import jax
import numpy as np
import pytest

from scripts.truncated_hierarchy_models import (
    CANONICAL_BLOCK_ORDER,
    Bounds,
    GeometryContractError,
    LinkedNormalPrior,
    NativeTruncatedPrior,
    ToyDataSpec,
    build_bambi_model,
    build_direct_pymc_model,
    compare_isomorphic_models,
    evaluate_transformed_geometry,
    generate_synthetic_data,
    make_near_boundary_evaluation_point,
    maximum_errors,
    normalized_error_max,
    support_forward,
    support_inverse,
)

BOUND_CASES = (
    pytest.param(Bounds(0.2, None), 0.23, 0.3, 0.0, "lower", "float64", id="lower"),
    pytest.param(Bounds(None, 0.8), 0.77, 0.3, 0.0, "upper", "float64", id="upper"),
    pytest.param(Bounds(0.1, 0.9), 0.13, 0.3, 0.5, "lower", "float32", id="two-sided"),
    pytest.param(
        Bounds(0.49, 0.51),
        0.495,
        0.05,
        0.5,
        "lower",
        "float64",
        id="narrow",
    ),
)


def _jax_gradient_tolerances(floatx: str) -> tuple[float, float]:
    """Use the precision JAX actually executes in this test process."""
    if floatx == "float32" or not jax.config.x64_enabled:
        return 2e-5, 5e-5
    return 2e-8, 2e-7


def _data(
    bounds: Bounds,
    location: float,
    scale: float,
    *,
    floatx: Literal["float32", "float64"] = "float64",
    seed: int = 1282,
):
    return generate_synthetic_data(
        ToyDataSpec(
            bounds=bounds,
            group_location=location,
            group_scale=scale,
            n_groups=4,
            n_per_group=2,
            floatx=floatx,
        ),
        group_seed=seed,
        observation_seed=seed + 1,
    )


@pytest.mark.parametrize(
    ("bounds", "values"),
    [
        pytest.param(Bounds(0.2, None), np.array([0.201, 0.5, 2.0]), id="lower"),
        pytest.param(Bounds(None, 0.8), np.array([-1.0, 0.5, 0.799]), id="upper"),
        pytest.param(Bounds(0.1, 0.9), np.array([0.101, 0.5, 0.899]), id="two-sided"),
        pytest.param(Bounds(0.49, 0.51), np.array([0.4901, 0.5, 0.5099]), id="narrow"),
    ],
)
def test_canonical_support_links_round_trip(bounds, values) -> None:
    """Exercise shifted-log and generalized-logit mappings near both boundaries."""
    eta = support_forward(values, bounds)
    restored = support_inverse(eta, bounds)

    assert np.all(np.isfinite(eta))
    np.testing.assert_allclose(restored, values, rtol=1e-12, atol=1e-12)


def test_support_links_reject_the_boundary_instead_of_clipping() -> None:
    """Never disguise an infeasible natural value as an interior point."""
    bounds = Bounds(0.49, 0.51)

    with pytest.raises(GeometryContractError, match="strictly inside"):
        support_forward(0.49, bounds)
    with pytest.raises(GeometryContractError, match="strictly inside"):
        ToyDataSpec(bounds, 0.52, 0.05, 4, 2)


@pytest.mark.parametrize(
    ("bounds", "location", "scale", "_base", "_side", "floatx"), BOUND_CASES
)
def test_synthetic_data_are_seeded_and_within_support(
    bounds, location, scale, _base, _side, floatx
) -> None:
    """Generate identical data for one seed and distinct data for another seed."""
    first = _data(bounds, location, scale, floatx=floatx, seed=111)
    repeated = _data(bounds, location, scale, floatx=floatx, seed=111)
    different = _data(bounds, location, scale, floatx=floatx, seed=112)

    np.testing.assert_array_equal(first.group_effect, repeated.group_effect)
    np.testing.assert_array_equal(first.y, repeated.y)
    assert not np.array_equal(first.y, different.y)
    assert bounds.contains(first.group_effect)
    assert first.group_index.shape == (8,)
    assert first.y.dtype == np.dtype(floatx)


def test_group_and_observation_rng_streams_are_independent() -> None:
    """Changing an observation seed cannot silently redraw group truths."""
    spec = ToyDataSpec(Bounds(0.2, None), 0.23, 0.3, 4, 2)
    first = generate_synthetic_data(
        spec,
        group_seed=111,
        observation_seed=222,
    )
    different_observations = generate_synthetic_data(
        spec,
        group_seed=111,
        observation_seed=223,
    )
    different_groups = generate_synthetic_data(
        spec,
        group_seed=112,
        observation_seed=222,
    )

    np.testing.assert_array_equal(
        first.group_effect, different_observations.group_effect
    )
    assert not np.array_equal(first.y, different_observations.y)
    assert not np.array_equal(first.group_effect, different_groups.group_effect)


@pytest.mark.parametrize(
    ("bounds", "location", "scale", "base", "side", "floatx"), BOUND_CASES
)
def test_direct_candidate_layout_and_gradients_cover_all_bound_shapes(
    bounds, location, scale, base, side, floatx
) -> None:
    """Check transformed ordering, near-boundary probes, FD, and JAX parity."""
    data = _data(bounds, location, scale, floatx=floatx)
    geometry = build_direct_pymc_model(NativeTruncatedPrior(bounds, base), data)

    assert geometry.canonical_names == CANONICAL_BLOCK_ORDER
    assert tuple(block.shape for block in geometry.blocks) == ((), (), (4,))
    assert geometry.dimension == 6
    assert geometry.value_variable_names == (
        "group_location_interval__",
        "group_scale_log__",
        "group_effect_interval__",
    )
    assert type(geometry.model.named_vars["group_location"].owner.op).__name__ == (
        "TruncatedNormalRV"
    )
    assert type(geometry.model.named_vars["group_effect"].owner.op).__name__ == (
        "TruncatedNormalRV"
    )

    initial = geometry.pack_point()
    np.testing.assert_array_equal(
        geometry.pack_point(geometry.point_from_vector(initial)), initial
    )
    probe = make_near_boundary_evaluation_point(geometry, side=side)
    assert probe.minimum_boundary_distance > 0
    assert bounds.contains(probe.natural_group_location)
    assert bounds.contains(probe.natural_group_effect)

    metrics = evaluate_transformed_geometry(geometry, probe.vector)
    assert metrics.all_finite
    if floatx == "float32":
        assert (
            metrics.finite_difference_normalized_error_max(
                absolute_tolerance=0.002,
                relative_tolerance=0.005,
            )
            <= 1
        )
    else:
        assert (
            metrics.finite_difference_normalized_error_max(
                absolute_tolerance=5e-7,
                relative_tolerance=5e-6,
            )
            <= 1
        )
    jax_atol, jax_rtol = _jax_gradient_tolerances(floatx)
    assert (
        metrics.pytensor_jax_normalized_error_max(
            absolute_tolerance=jax_atol,
            relative_tolerance=jax_rtol,
        )
        <= 1
    )


def test_linked_control_is_centered_and_maps_the_complete_predictor() -> None:
    """Keep the control unconstrained in eta and constrained in the likelihood."""
    bounds = Bounds(0.2, None)
    data = _data(bounds, 0.23, 0.3)
    prior = LinkedNormalPrior(
        bounds, location_base_mean_eta=float(support_forward(0.23, bounds))
    )
    geometry = build_direct_pymc_model(prior, data)

    assert geometry.value_variable_names == (
        "group_location_eta",
        "group_scale_log__",
        "group_effect_eta",
    )
    assert "group_effect" in {rv.name for rv in geometry.model.deterministics}
    assert "group_effect_eta" in {rv.name for rv in geometry.model.free_RVs}
    assert "group_effect_eta_offset" not in geometry.model.named_vars

    probe = make_near_boundary_evaluation_point(geometry, side="lower")
    metrics = evaluate_transformed_geometry(geometry, probe.vector)
    assert metrics.all_finite
    assert (
        metrics.finite_difference_normalized_error_max(
            absolute_tolerance=5e-7,
            relative_tolerance=5e-6,
        )
        <= 1
    )
    jax_atol, jax_rtol = _jax_gradient_tolerances("float64")
    assert (
        metrics.pytensor_jax_normalized_error_max(
            absolute_tolerance=jax_atol,
            relative_tolerance=jax_rtol,
        )
        <= 1
    )


@pytest.mark.parametrize(
    ("bounds", "location", "scale", "base", "side", "_floatx"), BOUND_CASES
)
def test_bambi_candidate_isomorphic_for_every_bound_shape(
    bounds, location, scale, base, side, _floatx
) -> None:
    """Compare direct and formula-generated centered native hierarchies."""
    data = _data(bounds, location, scale, floatx="float64")
    prior = NativeTruncatedPrior(bounds, base)
    direct = build_direct_pymc_model(prior, data)
    bambi = build_bambi_model(prior, data)
    probe = make_near_boundary_evaluation_point(direct, side=side)

    assert bambi.canonical_names == CANONICAL_BLOCK_ORDER
    assert tuple(block.shape for block in bambi.blocks) == ((), (), (4,))
    assert bambi.value_variable_names == (
        "1|group_id_mu_interval__",
        "1|group_id_sigma_log__",
        "1|group_id_interval__",
    )
    metrics = compare_isomorphic_models(direct, bambi, probe.vector)
    assert (
        metrics.normalized_error_max(
            absolute_tolerance=2e-7,
            relative_tolerance=2e-7,
        )
        <= 1
    )


def test_bambi_linked_control_is_isomorphic_to_direct_control() -> None:
    """Verify the public Bambi custom-link path against the direct control."""
    bounds = Bounds(0.1, 0.9)
    data = _data(bounds, 0.13, 0.3)
    prior = LinkedNormalPrior(
        bounds, location_base_mean_eta=float(support_forward(0.13, bounds))
    )
    direct = build_direct_pymc_model(prior, data)
    bambi = build_bambi_model(prior, data)
    probe = make_near_boundary_evaluation_point(direct, side="lower")

    assert bambi.value_variable_names == (
        "1|group_id_mu",
        "1|group_id_sigma_log__",
        "1|group_id",
    )
    metrics = compare_isomorphic_models(direct, bambi, probe.vector)
    assert (
        metrics.normalized_error_max(
            absolute_tolerance=2e-7,
            relative_tolerance=2e-7,
        )
        <= 1
    )


def test_isomorphism_metric_detects_deliberately_wrong_bambi_data() -> None:
    """Prove that the parity measurement catches a non-isomorphic likelihood."""
    bounds = Bounds(0.2, None)
    data = _data(bounds, 0.23, 0.3)
    prior = NativeTruncatedPrior(bounds, 0.0)
    direct = build_direct_pymc_model(prior, data)
    correct_bambi = build_bambi_model(prior, data)
    wrong_bambi = build_bambi_model(prior, replace(data, y=data.y + 0.25))
    probe = make_near_boundary_evaluation_point(direct, side="lower")

    correct = compare_isomorphic_models(direct, correct_bambi, probe.vector)
    wrong = compare_isomorphic_models(direct, wrong_bambi, probe.vector)

    assert (
        correct.normalized_error_max(
            absolute_tolerance=2e-7,
            relative_tolerance=2e-7,
        )
        <= 1
    )
    assert (
        wrong.normalized_error_max(
            absolute_tolerance=2e-7,
            relative_tolerance=2e-7,
        )
        > 1
    )


def test_error_metric_detects_a_deliberately_perturbed_gradient() -> None:
    """Keep a cheap, direct regression guard for gradient-error detection."""
    reference = np.array([1.0, -2.0, 0.5])
    perturbed = reference.copy()
    perturbed[1] += 0.1

    metrics = maximum_errors(reference, perturbed)

    assert metrics.absolute_max == pytest.approx(0.1)
    assert metrics.relative_max > 0.04


def test_combined_tolerance_does_not_false_fail_a_near_zero_gradient() -> None:
    """Use atol and rtol together instead of independently gating both maxima."""
    reference = np.array([1.0e-3, -1.4])
    harmless = np.array([1.0e-3 - 2.3e-7, -1.4])
    material = np.array([1.0e-3 - 3.0e-5, -1.4])

    raw = maximum_errors(reference, harmless)

    assert raw.relative_max > 5e-5
    assert (
        normalized_error_max(
            reference,
            harmless,
            absolute_tolerance=2e-5,
            relative_tolerance=5e-5,
        )
        < 1
    )
    assert (
        normalized_error_max(
            reference,
            material,
            absolute_tolerance=2e-5,
            relative_tolerance=5e-5,
        )
        > 1
    )


def test_bambi_float32_cell_reaches_the_real_backend() -> None:
    """Let the stress cell measure Bambi/PyMC instead of a harness rejection."""
    bounds = Bounds(0.1, 0.9)
    data = _data(bounds, 0.13, 0.3, floatx="float32")

    try:
        geometry = build_bambi_model(NativeTruncatedPrior(bounds, 0.5), data)
    except TypeError as error:
        # Some supported Bambi/PyTensor combinations expose a backend dtype
        # mismatch here. That is admissible diagnostic evidence; the harness
        # itself must not reject the cell before reaching the real backend.
        assert re.search(r"Vector\(float64.*Vector\(float32", str(error))
    else:
        assert geometry.source == "bambi"
        assert geometry.model_float_dtype == np.dtype("float32")

"""Pre-sampling HSSM integration checks for qualification issue #1282."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import jax
import numpy as np
import pymc as pm
import pytensor
import pytest

import hssm
from scripts.truncated_hierarchy_hssm import (
    HSSMBuild,
    build_hssm_model,
    evaluate_hssm_gradients,
    extract_actual_sampler_starts,
    inspect_hierarchy,
    lba2_pytensor_jax_parity,
    make_structural_test_data,
    validate_actual_sampler_starts,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


def _scenario(
    model: str,
    prior: str,
) -> dict[str, Any]:
    if model == "lba2_b":
        lower, upper, floatx = 0.2, None, "float64"
    elif model == "approx_ddm_z":
        lower, upper, floatx = 0.1, 0.9, "float32"
    else:
        lower, upper, floatx = 0.0, None, "float64"
    scenario_id = f"structural-{model.replace('_', '-')}-{prior.replace('_', '-')}"
    return {
        "scenario_id": scenario_id,
        "layer": "hssm",
        "model": model,
        "prior": prior,
        "lower": lower,
        "upper": upper,
        "prior_hyper_location": (
            (lower + upper) / 2
            if prior == "truncated_normal" and upper is not None
            else 0.0
        ),
        "floatx": floatx,
        "n_groups": 3,
        "n_per_group": 4,
    }


BUILD_CASES = {
    "lba-candidate": _scenario("lba2_b", "truncated_normal"),
    "lba-control": _scenario("lba2_b", "linked_normal"),
    "ddm-candidate": _scenario("approx_ddm_z", "truncated_normal"),
    "ddm-control": _scenario("approx_ddm_z", "linked_normal"),
    "softmax-candidate": _scenario("softmax_beta", "truncated_normal"),
    "softmax-control": _scenario("softmax_beta", "linked_normal"),
}


@pytest.fixture(scope="module")
def builds() -> Iterator[Mapping[str, HSSMBuild]]:
    """Build each candidate/control once while forbidding network fallback."""
    previous_floatx = pytensor.config.floatX
    previous_jax_x64 = jax.config.read("jax_enable_x64")
    result: dict[str, HSSMBuild] = {}
    with patch(
        "hssm.distribution_utils.onnx_utils.model.hf_hub_download",
        side_effect=AssertionError("qualification models must not use the network"),
    ):
        for index, (case, scenario) in enumerate(BUILD_CASES.items()):
            data = make_structural_test_data(
                scenario["model"],
                n_groups=scenario["n_groups"],
                n_per_group=scenario["n_per_group"],
                seed=1282,
            )
            result[case] = build_hssm_model(
                scenario,
                data,
                initval_seed=1700 + index,
            )
    yield result
    hssm.set_floatX(
        "float32" if previous_floatx == "float32" else "float64",
        update_jax=True,
    )
    jax.config.update("jax_enable_x64", previous_jax_x64)


@pytest.mark.parametrize(
    ("case", "family", "link"),
    [
        ("lba-candidate", "TruncatedNormal", "identity"),
        ("lba-control", "Normal", "shifted_log"),
        ("ddm-candidate", "TruncatedNormal", "identity"),
        ("ddm-control", "Normal", "gen_logit"),
        ("softmax-candidate", "TruncatedNormal", "identity"),
        ("softmax-control", "Normal", "shifted_log"),
    ],
)
def test_exact_hssm_candidate_and_control_graphs(builds, case, family, link):
    """Every HSSM representative has a centered, connected, no-offset graph."""
    build = builds[case]
    observed = inspect_hierarchy(build)

    assert build.link_name == link
    assert observed.prior_family == family
    assert observed.location_prior_family == family
    assert observed.scale_prior_family == "Weibull"
    assert observed.prior_noncentered is False
    assert observed.offset_present is False
    assert observed.group_connected_to_parameter is True
    assert observed.disconnected_free_rvs == ()
    assert {
        build.group_rv_name,
        build.group_location_name,
        build.group_scale_name,
    } <= set(observed.free_rv_names)
    assert len(build.data) == 12

    if family == "TruncatedNormal":
        assert observed.group_rv_op == "TruncatedNormalRV"
        assert observed.location_rv_op == "TruncatedNormalRV"
        assert (observed.lower, observed.upper) == pytest.approx(build.bounds)
    else:
        assert observed.group_rv_op == "NormalRV"
        assert observed.location_rv_op == "NormalRV"


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("lba-candidate", 0.0),
        ("ddm-candidate", 0.5),
        ("softmax-candidate", 0.0),
        ("lba-control", 0.0),
        ("ddm-control", 0.0),
        ("softmax-control", 0.0),
    ],
)
def test_v2_numeric_prior_calibration_matches_generated_graph(builds, case, expected):
    """V2 describes the generated prior numerically without recalibrating it."""
    assert builds[case].prior_hyper_location == pytest.approx(expected)
    assert inspect_hierarchy(builds[case]).base_mu == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, "inside", np.nan, 0.1])
def test_v2_rejects_missing_ambiguous_or_mismatched_prior_calibration(value):
    """A scenario cannot silently move the generated hyper-location."""
    scenario = _scenario("lba2_b", "truncated_normal")
    if value is None:
        scenario.pop("prior_hyper_location")
    else:
        scenario["prior_hyper_location"] = value
    data = make_structural_test_data(
        "lba2_b",
        n_groups=scenario["n_groups"],
        n_per_group=scenario["n_per_group"],
        seed=1282,
    )
    with pytest.raises(ValueError, match="prior_hyper_location"):
        build_hssm_model(scenario, data, initval_seed=1700)


@pytest.mark.parametrize("case", BUILD_CASES)
def test_actual_hssm_starts_are_transformed_valid_and_supported(builds, case):
    """Materialize HSSM's processed overrides, not a fresh model initial point."""
    build = builds[case]
    artifact = extract_actual_sampler_starts(
        build,
        sampler="pymc",
        chains=2,
    )
    repeated = extract_actual_sampler_starts(
        build,
        sampler="pymc",
        chains=2,
    )

    assert artifact.sha256() == repeated.sha256()
    assert artifact.initialization_seed == build.initialization_seed
    assert artifact.start_seeds == ()
    assert len(artifact.transformed_points) == 2
    assert all(
        set(point) == {value.name for value in build.model.pymc_model.value_vars}
        for point in artifact.transformed_points
    )
    for left, right in zip(
        artifact.transformed_points, repeated.transformed_points, strict=True
    ):
        for name in left:
            np.testing.assert_array_equal(left[name], right[name])
    for name, value in artifact.transformed_points[0].items():
        np.testing.assert_array_equal(value, artifact.transformed_points[1][name])
    assert np.isfinite(validate_actual_sampler_starts(build, artifact)).all()

    # Evaluate the response-scale HSSM parameter at the transformed sampler point.
    (parameter_graph,) = build.model.pymc_model.replace_rvs_by_values(
        [build.model.pymc_model.named_vars[build.parameter]]
    )
    parameter_fn = build.model.pymc_model.compile_fn(
        parameter_graph,
        inputs=build.model.pymc_model.value_vars,
        on_unused_input="ignore",
    )
    parameter_values = np.asarray(parameter_fn(artifact.transformed_points[0]))
    lower, upper = build.bounds
    if np.isfinite(lower):
        assert np.all(parameter_values > lower)
    if np.isfinite(upper):
        assert np.all(parameter_values < upper)


def test_numpyro_start_extraction_uses_same_no_jitter_overrides(builds):
    """HSSM's NumPyro wrapper retains its own controlled, transformed start."""
    build = builds["lba-candidate"]
    pymc_artifact = extract_actual_sampler_starts(
        build,
        sampler="pymc",
        chains=1,
    )
    numpyro_artifact = extract_actual_sampler_starts(
        build,
        sampler="numpyro",
        chains=1,
    )
    for name, value in pymc_artifact.transformed_points[0].items():
        np.testing.assert_array_equal(
            value, numpyro_artifact.transformed_points[0][name]
        )
    assert np.isfinite(validate_actual_sampler_starts(build, numpyro_artifact)).all()


@pytest.mark.parametrize("case", BUILD_CASES)
def test_prior_factor_gradients_and_full_models_are_finite(builds, case):
    """Check transformed prior gradients separately from complete likelihoods."""
    build = builds[case]
    artifact = extract_actual_sampler_starts(
        build,
        sampler="pymc",
        chains=1,
    )
    diagnostics = evaluate_hssm_gradients(build, artifact.transformed_points[0])
    if build.floatx == "float32":
        finite_difference_tolerances = (0.002, 0.005)
        pytensor_jax_tolerances = (2e-5, 5e-5)
    else:
        finite_difference_tolerances = (5e-7, 5e-6)
        pytensor_jax_tolerances = (2e-8, 2e-7)
    metrics = diagnostics.qualification_metrics(
        finite_difference_absolute_tolerance=finite_difference_tolerances[0],
        finite_difference_relative_tolerance=finite_difference_tolerances[1],
        pytensor_jax_absolute_tolerance=pytensor_jax_tolerances[0],
        pytensor_jax_relative_tolerance=pytensor_jax_tolerances[1],
    )

    assert metrics["compile_success"] is True
    assert metrics["logp_finite"] is True
    assert metrics["gradient_finite"] is True
    assert metrics["finite_difference_gradient_normalized_error_max"] <= 1
    assert metrics["pytensor_jax_gradient_normalized_error_max"] <= 1
    assert diagnostics.full_gradient_size == 5


def test_lba_likelihood_parity_is_an_independent_diagnostic(builds):
    """Report likelihood-backend errors independently of the prior contract."""
    build = builds["lba-candidate"]
    ordinary = lba2_pytensor_jax_parity(build.data, b=0.5)
    at_generated_start = lba2_pytensor_jax_parity(build.data, b=1.2)

    for observed in (ordinary, at_generated_start):
        metrics = observed.qualification_metrics(
            value_absolute_tolerance=2e-8,
            value_relative_tolerance=2e-7,
            gradient_absolute_tolerance=2e-8,
            gradient_relative_tolerance=2e-7,
        )
        assert observed.all_finite is True
        assert metrics["likelihood_pytensor_jax_value_normalized_error_max"] <= 1
        assert metrics["likelihood_pytensor_jax_gradient_normalized_error_max"] <= 1
    assert at_generated_start.pytensor_values.shape == (12,)
    assert at_generated_start.jax_gradient.shape == (12,)


def test_ddm_factory_uses_only_the_committed_local_network(tmp_path):
    """Reject an unavailable local fixture rather than falling back to a download."""
    scenario = BUILD_CASES["ddm-candidate"]
    data = make_structural_test_data(
        scenario["model"], n_groups=3, n_per_group=4, seed=1282
    )
    unavailable = Path(tmp_path) / "missing.onnx"
    with pytest.raises(ValueError, match="network is unavailable"):
        build_hssm_model(
            scenario,
            data,
            initval_seed=1282,
            ddm_network_path=unavailable,
        )


def test_parameter_function_really_uses_group_rv(builds):
    """Perturbing the group value changes its HSSM response-scale parameter."""
    build = builds["softmax-control"]
    artifact = extract_actual_sampler_starts(
        build,
        sampler="pymc",
        chains=1,
    )
    point = {
        name: np.array(value, copy=True)
        for name, value in artifact.transformed_points[0].items()
    }
    group_value_name = build.model.pymc_model.rvs_to_values[
        build.model.pymc_model.named_vars[build.group_rv_name]
    ].name
    assert group_value_name is not None
    (parameter_graph,) = build.model.pymc_model.replace_rvs_by_values(
        [build.model.pymc_model.named_vars[build.parameter]]
    )
    parameter_fn = build.model.pymc_model.compile_fn(
        parameter_graph,
        inputs=build.model.pymc_model.value_vars,
        on_unused_input="ignore",
    )
    baseline = np.asarray(parameter_fn(point))
    point[group_value_name] = point[group_value_name] + 0.25
    perturbed = np.asarray(parameter_fn(point))
    assert not np.allclose(baseline, perturbed)
    assert np.all(perturbed > 0.0)


def test_start_artifact_is_strict_json(builds):
    """The provenance artifact contains only transformed values and finite JSON."""
    artifact = extract_actual_sampler_starts(
        builds["ddm-candidate"],
        sampler="numpyro",
        chains=2,
    )
    payload = artifact.as_jsonable()
    assert payload["coordinate_system"] == "pymc-transformed-value-variables"
    assert payload["sampler"] == "numpyro"
    assert payload["initialization_seed"] == builds["ddm-candidate"].initialization_seed
    assert payload["start_seeds"] == []
    assert len(artifact.sha256()) == 64
    assert len(payload["chains"]) == 2
    for chain in payload["chains"]:
        assert all(
            np.isfinite(np.asarray(value)).all() for value in chain["values"].values()
        )


def test_no_sampling_api_is_called(monkeypatch, builds):
    """The entire structural layer stops before posterior sampling."""

    def unexpected_sample(*args, **kwargs):
        raise AssertionError("pre-sampling qualification layer called sample")

    monkeypatch.setattr(pm, "sample", unexpected_sample)
    artifact = extract_actual_sampler_starts(
        builds["softmax-candidate"],
        sampler="pymc",
        chains=1,
    )
    assert len(artifact.transformed_points) == 1

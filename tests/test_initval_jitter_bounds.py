"""Regression tests for support-aware hierarchical initial-value jitter."""

import numpy as np
import pytest
from pymc.exceptions import SamplingError
from pymc.initial_point import StartDict, make_initial_point_fns_per_chain

import hssm
from hssm.defaults import INITVAL_JITTER_SETTINGS
from hssm.likelihoods import logp_ddm

hssm.set_floatX("float32", update_jax=True)

GROUP_TERM = "v_1|participant_id"
NARROW_BOUNDS = (0.499, 0.501)


def _build_group_model(cavanagh_test, bounds, **kwargs) -> hssm.HSSM:
    """Build a tiny custom DDM with one generated group-only location."""
    data = cavanagh_test.groupby("participant_id").head(2).copy()
    return hssm.HSSM(
        data=data,
        model="custom",
        model_config={
            "list_params": ["v", "a", "z", "t"],
            "choices": [-1, 1],
            "bounds": {
                "v": bounds,
                "a": (0.1, np.inf),
                "z": (0.0, 1.0),
                "t": (0.0, np.inf),
            },
        },
        loglik=logp_ddm,
        loglik_kind="analytical",
        include=[{"name": "v", "formula": "v ~ 0 + (1 | participant_id)"}],
        a=1.5,
        z=0.5,
        t=0.2,
        p_outlier=0.0,
        prior_settings="safe",
        **kwargs,
    )


def _sampler_initial_point(model: hssm.HSSM) -> dict[str, np.ndarray]:
    """Compile HSSM's constrained overrides through PyMC's sampler path."""
    overrides: StartDict = {
        name: np.asarray(value) for name, value in model._initvals.items()
    }
    initial_point_fn = make_initial_point_fns_per_chain(
        model=model.pymc_model,
        overrides=overrides,
        jitter_rvs=set(),
        chains=1,
    )[0]
    return initial_point_fn(1269)


def _group_value_name(model: hssm.HSSM) -> str:
    """Return the transformed PyMC value name for the group random variable."""
    group_rv = model.pymc_model.named_vars[GROUP_TERM]
    value_name = model.pymc_model.rvs_to_values[group_rv].name
    assert value_name is not None
    return value_name


def _uniform_endpoint(endpoint: str):
    """Return a uniform stub that deterministically selects one endpoint."""

    def select(low, high, size=None):
        selected = low if endpoint == "low" else high
        return np.broadcast_to(np.asarray(selected), size).copy()

    return select


def test_default_jitter_keeps_narrow_group_initvals_strictly_inside_support(
    cavanagh_test, monkeypatch
):
    """Default vector jitter cannot cross narrow generated group bounds."""
    monkeypatch.setattr(np.random, "uniform", _uniform_endpoint("high"))

    model = _build_group_model(cavanagh_test, NARROW_BOUNDS)
    jittered = model._initvals[GROUP_TERM]

    assert model.initval_jitter == INITVAL_JITTER_SETTINGS["jitter_epsilon"]
    assert model.params["v"].prior["1|participant_id"].name == "TruncatedNormal"
    assert model._get_group_initval_bounds(GROUP_TERM) == pytest.approx(NARROW_BOUNDS)
    assert jittered.dtype == np.dtype("float32")
    assert np.all(jittered > NARROW_BOUNDS[0])
    assert np.all(jittered < NARROW_BOUNDS[1])
    assert np.any(jittered != np.float32(0.5))

    sampler_point = _sampler_initial_point(model)
    assert all(np.all(np.isfinite(value)) for value in sampler_point.values())


@pytest.mark.parametrize(
    ("bounds", "endpoint"),
    [
        pytest.param((0.2, np.inf), "low", id="lower-only"),
        pytest.param((-np.inf, 0.2), "high", id="upper-only"),
    ],
)
def test_group_jitter_respects_one_sided_support(
    cavanagh_test, monkeypatch, bounds, endpoint
):
    """One-sided native group priors retain a strict finite boundary."""
    model = _build_group_model(cavanagh_test, bounds, initval_jitter=0.0)
    prior = model.params["v"].prior["1|participant_id"]
    missing_endpoint = "upper" if np.isfinite(bounds[0]) else "lower"
    prior.args[missing_endpoint] = None
    detected_bounds = model._get_group_initval_bounds(GROUP_TERM)
    assert detected_bounds == pytest.approx(bounds)

    dtype = model._initvals[GROUP_TERM].dtype
    if np.isfinite(bounds[0]):
        boundary = np.asarray(bounds[0], dtype=dtype)
        interior = np.nextafter(
            np.nextafter(boundary, np.asarray(np.inf, dtype=dtype)),
            np.asarray(np.inf, dtype=dtype),
        ).item()
    else:
        boundary = np.asarray(bounds[1], dtype=dtype)
        interior = np.nextafter(
            np.nextafter(boundary, np.asarray(-np.inf, dtype=dtype)),
            np.asarray(-np.inf, dtype=dtype),
        ).item()
    model._initvals[GROUP_TERM] = np.full_like(model._initvals[GROUP_TERM], interior)
    monkeypatch.setattr(np.random, "uniform", _uniform_endpoint(endpoint))

    model._jitter_initvals(vector_only=True)
    jittered = model._initvals[GROUP_TERM]

    if np.isfinite(bounds[0]):
        assert np.all(jittered > bounds[0])
    if np.isfinite(bounds[1]):
        assert np.all(jittered < bounds[1])
    sampler_point = _sampler_initial_point(model)
    assert np.all(np.isfinite(sampler_point[_group_value_name(model)]))


def test_zero_jitter_preserves_group_initial_point(cavanagh_test):
    """An explicit zero jitter leaves the generated group vector unchanged."""
    model = _build_group_model(cavanagh_test, NARROW_BOUNDS, initval_jitter=0.0)

    np.testing.assert_array_equal(
        model._initvals[GROUP_TERM], model.initial_point()[GROUP_TERM]
    )
    sampler_point = _sampler_initial_point(model)
    assert np.all(np.isfinite(sampler_point[_group_value_name(model)]))


def test_invalid_group_initvals_are_not_silently_repaired(cavanagh_test, monkeypatch):
    """Boundary, outside, and NaN starts remain invalid for PyMC to report."""
    model = _build_group_model(cavanagh_test, NARROW_BOUNDS, initval_jitter=0.0)
    lower, upper = model._get_group_initval_bounds(GROUP_TERM)
    dtype = model._initvals[GROUP_TERM].dtype
    lower_value = np.asarray(lower, dtype=dtype)
    upper_value = np.asarray(upper, dtype=dtype)
    invalid_values = [
        lower_value.item(),
        np.nextafter(lower_value, np.asarray(-np.inf, dtype=dtype)).item(),
        upper_value.item(),
        np.nextafter(upper_value, np.asarray(np.inf, dtype=dtype)).item(),
        np.nan,
    ]

    def unexpected_uniform(*args, **kwargs):
        raise AssertionError("invalid initial values must not be jittered")

    monkeypatch.setattr(np.random, "uniform", unexpected_uniform)
    for invalid_value in invalid_values:
        supplied = np.full_like(model._initvals[GROUP_TERM], invalid_value)
        model._initvals[GROUP_TERM] = supplied.copy()

        model._jitter_initvals(vector_only=True)

        np.testing.assert_equal(model._initvals[GROUP_TERM], supplied)
        sampler_point = _sampler_initial_point(model)
        assert not np.all(np.isfinite(sampler_point[_group_value_name(model)]))
        with pytest.raises(SamplingError):
            model.pymc_model.check_start_vals(sampler_point, mode="FAST_COMPILE")


def test_unbounded_vector_jitter_matches_legacy_seeded_result(cavanagh_test):
    """Unbounded group vectors retain the prior seeded additive-jitter behavior."""
    model = _build_group_model(cavanagh_test, (-np.inf, np.inf), initval_jitter=0.0)
    assert model.params["v"].prior["1|participant_id"].name == "Normal"
    assert model._get_group_initval_bounds(GROUP_TERM) is None

    starting_value = model._initvals[GROUP_TERM].copy()
    jitter_epsilon = INITVAL_JITTER_SETTINGS["jitter_epsilon"]
    random_state = np.random.get_state()
    try:
        np.random.seed(1269)
        expected = starting_value + np.random.uniform(
            -jitter_epsilon, jitter_epsilon, starting_value.shape
        ).astype(np.float32)
        np.random.seed(1269)
        model._jitter_initvals(
            jitter_epsilon=jitter_epsilon,
            vector_only=True,
        )
    finally:
        np.random.set_state(random_state)

    np.testing.assert_array_equal(model._initvals[GROUP_TERM], expected)
    sampler_point = _sampler_initial_point(model)
    assert np.all(np.isfinite(sampler_point[_group_value_name(model)]))

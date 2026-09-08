"""Tests for response-order-derived observation dimensions."""

from collections.abc import Callable

import cloudpickle
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pytest
import xarray as xr

import hssm
from hssm.distribution_utils import make_distribution, make_hssm_rv


def _synthetic_simulator(obs_dim: int, *, scalar_output: bool = False) -> Callable:
    """Return a deterministic-contract simulator with seeded random values."""

    def simulator(theta, random_state, n_replicas, **kwargs):
        del kwargs
        theta_array = np.asarray(theta)
        n_rows = (
            n_replicas if theta_array.ndim == 1 else theta_array.shape[0] * n_replicas
        )
        values = np.random.default_rng(random_state).normal(size=(n_rows, obs_dim))
        return values[:, 0] if scalar_output else values

    simulator.model_name = f"synthetic_{obs_dim}d"  # type: ignore[attr-defined]
    simulator.choices = ()  # type: ignore[attr-defined]
    simulator.obs_dim = obs_dim  # type: ignore[attr-defined]
    return simulator


def _synthetic_logp(data, v):
    """Return one finite log-likelihood value per observation."""
    if data.ndim == 1:
        return -pt.square(data - v)
    return -pt.sum(pt.square(data - pt.shape_padright(v)), axis=-1)


def _synthetic_model(
    configured_obs_dim: int,
    *,
    simulator_obs_dim: int | None = None,
) -> hssm.HSSM:
    """Build a small custom model with scalar physical response columns."""
    simulator_obs_dim = simulator_obs_dim or configured_obs_dim
    if configured_obs_dim == 1:
        response = ("response",)
        data = pd.DataFrame({"response": [0, 1, 0, 1, 0]})
        response_domains = {"response": {"kind": "categorical", "values": (0, 1)}}
    else:
        response = ("rt",) + tuple(
            f"response_{index}" for index in range(1, configured_obs_dim)
        )
        data = pd.DataFrame(
            {
                "rt": np.linspace(0.3, 0.7, 5),
                **{column: np.linspace(-0.5, 0.5, 5) for column in response[1:]},
            }
        )
        response_domains = {
            column: {"kind": "continuous", "bounds": (-1.0, 1.0)}
            for column in response[1:]
        }

    return hssm.HSSM(
        data=data,
        model="custom",
        model_config=hssm.ModelConfig(
            response=response,
            response_domains=response_domains,  # type: ignore[arg-type]
            list_params=["v"],
            default_priors={"v": {"name": "Normal", "mu": 0.0, "sigma": 1.0}},
            rv=_synthetic_simulator(
                simulator_obs_dim,
                scalar_output=configured_obs_dim == 1,
            ),  # type: ignore[arg-type]
        ),
        loglik=_synthetic_logp,
        loglik_kind="analytical",
        p_outlier=None,
        process_initvals=False,
    )


@pytest.mark.parametrize("obs_dim", [1, 2, 3, 4])
def test_generated_rv_preserves_width_and_seed_stream(obs_dim):
    """Generated RVs retain arbitrary widths and deterministic stream order."""
    is_choice_only = obs_dim == 1
    simulator = _synthetic_simulator(obs_dim, scalar_output=is_choice_only)
    rv = make_hssm_rv(
        simulator,
        ["v"],
        is_choice_only=is_choice_only,
        expected_obs_dim=obs_dim,
    )
    legacy_rv = make_hssm_rv(
        simulator,
        ["v"],
        is_choice_only=is_choice_only,
    )
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    legacy_rng = np.random.default_rng(42)

    first_a = rv.rng_fn(rng_a, 0.5, size=5)
    second_a = rv.rng_fn(rng_a, 0.5, size=5)
    first_b = rv.rng_fn(rng_b, 0.5, size=5)
    second_b = rv.rng_fn(rng_b, 0.5, size=5)
    legacy_first = legacy_rv.rng_fn(legacy_rng, 0.5, size=5)
    legacy_second = legacy_rv.rng_fn(legacy_rng, 0.5, size=5)

    expected_shape = (5,) if is_choice_only else (5, obs_dim)
    assert first_a.shape == expected_shape
    assert second_a.shape == expected_shape
    np.testing.assert_array_equal(first_a, first_b)
    np.testing.assert_array_equal(second_a, second_b)
    np.testing.assert_array_equal(first_a, legacy_first)
    np.testing.assert_array_equal(second_a, legacy_second)
    assert not np.array_equal(first_a, second_a)


def test_distribution_rejects_callable_width_mismatch():
    """Configured response width must agree with generated-RV metadata."""
    with pytest.raises(
        ValueError,
        match="simulator observation width 3.*configured response width 4",
    ):
        make_distribution(
            rv=_synthetic_simulator(3),
            loglik=_synthetic_logp,
            list_params=["v"],
            expected_obs_dim=4,
        )


def test_model_rejects_callable_width_mismatch_before_building_distribution():
    """A custom model fails clearly when its simulator width is inconsistent."""
    with pytest.raises(
        ValueError,
        match="simulator observation width 2.*configured response width 3",
    ):
        _synthetic_model(3, simulator_obs_dim=2)


@pytest.mark.parametrize("obs_dim", [1, 2, 3, 4])
def test_predictive_shapes_follow_configured_response_width(obs_dim):
    """Prior and posterior predictions follow the physical response width."""
    model = _synthetic_model(obs_dim)
    prior = model.sample_prior_predictive(
        draws=3,
        random_seed=np.random.default_rng(41),
    )
    response_name = model.response_str
    response_dim = f"{response_name}_dim"
    prior_values = prior["prior_predictive"][response_name]
    prior["posterior"] = prior["prior"]
    posterior = model.sample_posterior_predictive(
        prior,
        draws=2,
        safe_mode=False,
        inplace=False,
    )
    assert posterior is not None
    posterior_values = posterior["posterior_predictive"][response_name]

    if obs_dim == 1:
        assert not hasattr(model.family, "create_extra_pps_coord")
        assert prior_values.dims == ("chain", "draw", "__obs__")
        assert posterior_values.dims == ("chain", "draw", "__obs__")
    else:
        np.testing.assert_array_equal(
            model.family.create_extra_pps_coord(), np.arange(obs_dim)
        )
        assert prior_values.dims[-1] == response_dim
        assert posterior_values.dims[-1] == response_dim
        np.testing.assert_array_equal(
            posterior_values.coords[response_dim], np.arange(obs_dim)
        )


def test_safe_mode_preserves_width_four_draw_and_response_coordinates():
    """Chunked prediction concatenates arbitrary-width draws without drift."""
    model = _synthetic_model(4)
    traces = model.sample_prior_predictive(
        draws=12,
        random_seed=np.random.default_rng(43),
    )
    traces["posterior"] = traces["prior"]

    result = model.sample_posterior_predictive(
        traces,
        draws=12,
        safe_mode=True,
        inplace=False,
    )

    assert result is not None
    response_name = model.response_str
    response_dim = f"{response_name}_dim"
    values = result["posterior_predictive"][response_name]
    assert values.dims == ("chain", "draw", "__obs__", response_dim)
    np.testing.assert_array_equal(values.coords["draw"], np.arange(12))
    np.testing.assert_array_equal(values.coords[response_dim], np.arange(4))


def test_choice_only_callable_width_mismatch_is_not_masked():
    """Scalar support does not hide malformed callable width metadata."""
    with pytest.raises(
        ValueError,
        match="simulator observation width 2.*configured response width 1",
    ):
        make_hssm_rv(
            _synthetic_simulator(2, scalar_output=True),
            ["v"],
            is_choice_only=True,
            expected_obs_dim=1,
        )


def test_choice_only_string_keeps_scalar_legacy_width():
    """Legacy choice-only names retain scalar support despite wrapper metadata."""
    rv = make_hssm_rv(
        "choice_only_model",
        ["beta"],
        is_choice_only=True,
        expected_obs_dim=1,
    )

    assert rv.signature == "()->()"


@pytest.mark.parametrize("obs_dim", [1, 4])
def test_sample_do_dataframe_uses_physical_response_order(obs_dim):
    """Intervention samples expose each configured physical response column."""
    model = _synthetic_model(obs_dim)
    predictive = model.sample_do(params={"v": 0.5}, draws=3)
    frame = hssm.utils.predictive_dt_to_dataframe(
        predictive,
        predictive_group="prior_predictive",
        response_str=model.response_str,
        response_dim=f"{model.response_str}_dim",
    )

    assert list(frame.columns) == ["chain", "draw", "__obs__", *model.response]
    assert len(frame) == 3 * len(model.data)


def test_scalar_dataframe_ignores_deadline_technical_suffix():
    """Scalar predictions use the physical response name, not its suffix."""
    response_str = "response,deadline"
    predictive = xr.DataTree.from_dict(
        {
            "posterior_predictive": xr.Dataset(
                {
                    response_str: (
                        ("chain", "draw", "__obs__"),
                        np.arange(6).reshape(1, 2, 3),
                    )
                }
            )
        }
    )

    frame = hssm.utils.predictive_dt_to_dataframe(
        predictive,
        response_str=response_str,
        response_dim=f"{response_str}_dim",
    )

    assert list(frame.columns) == ["chain", "draw", "__obs__", "response"]
    np.testing.assert_array_equal(frame["response"], np.arange(6))


def test_dataframe_rejects_missing_vector_response_coordinate():
    """A differently named vector dimension is not mistaken for scalar output."""
    response_str = "rt,response"
    predictive = xr.DataTree.from_dict(
        {
            "posterior_predictive": xr.Dataset(
                {
                    response_str: (
                        ("chain", "draw", "__obs__", "unexpected_dim"),
                        np.arange(12).reshape(1, 2, 3, 2),
                    )
                }
            )
        }
    )

    with pytest.raises(
        ValueError,
        match="Predictive response coordinate 'rt,response_dim' is missing",
    ):
        hssm.utils.predictive_dt_to_dataframe(
            predictive,
            response_str=response_str,
            response_dim=f"{response_str}_dim",
        )


def test_predictive_cleanup_uses_custom_response_name():
    """Parent cleanup derives names from the physical response declaration."""
    model = _synthetic_model(4)
    response_mean = f"{model.response_str}_mean"
    traces = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {response_mean: (("chain", "draw"), np.array([[0.5]]))}
            )
        }
    )

    result = model._clean_predictive_datatree(traces)

    assert model._parent in result["posterior"].data_vars
    assert response_mean not in result["posterior"].data_vars


def test_width_four_cloudpickle_round_trip_preserves_predictive_contract():
    """Existing model reconstruction retains width, metadata, and seeded draws."""
    model = _synthetic_model(4)
    restored = cloudpickle.loads(cloudpickle.dumps(model))

    assert restored._obs_dim == model._obs_dim == 4
    assert restored.response == model.response
    assert restored.response_domains == model.response_domains

    expected = model.sample_prior_predictive(
        draws=2,
        random_seed=np.random.default_rng(91),
    )
    actual = restored.sample_prior_predictive(
        draws=2,
        random_seed=np.random.default_rng(91),
    )
    xr.testing.assert_equal(
        actual["prior_predictive"].ds,
        expected["prior_predictive"].ds,
    )

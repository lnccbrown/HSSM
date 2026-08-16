"""Predictive-path tests for the registered JEAM circular model."""

import numpy as np
import pandas as pd
import pytensor
import pytest
import xarray as xr

pytest.importorskip("jeam")

import hssm
from hssm.integrations.jeam import simulate_circular_diffusion

OBSERVED_NAME = "rt,response"
OBSERVATION_DIM = "__obs__"
RESPONSE_DIM = "rt,response_dim"


def _build_circular_model():
    """Build an ordinary HSSM model from deterministic JEAM observations."""
    observations = simulate_circular_diffusion(
        theta=np.array([0.70, -0.35, 0.40, 0.08]),
        random_state=1947,
        n_replicas=6,
    )
    return hssm.HSSM(
        data=pd.DataFrame(observations, columns=["rt", "response"]),
        model="circular_diffusion",
        p_outlier=None,
    )


@pytest.fixture(scope="module")
def circular_model():
    """Share one circular model across the predictive-path tests."""
    return _build_circular_model()


def _assert_predictive_contract(values: np.ndarray) -> None:
    """Check the common final-axis, support, and numeric predictive contract."""
    assert values.shape[-1] == 2
    assert values.dtype == np.dtype(pytensor.config.floatX)
    assert np.all(np.isfinite(values))
    assert np.all(values[..., 0] > 0.0)
    assert np.all(values[..., 1] >= -np.pi)
    assert np.all(values[..., 1] < np.pi)


def _fixed_posterior(n_draws: int = 12) -> xr.DataTree:
    """Return identical parameter draws to isolate simulator RNG behavior."""
    coords = {"chain": [0], "draw": np.arange(n_draws)}
    posterior = xr.Dataset(
        {
            name: (("chain", "draw"), np.full((1, n_draws), value))
            for name, value in {
                "v_x": 0.70,
                "v_y": -0.35,
                "a": 0.40,
                "t": 0.08,
            }.items()
        },
        coords=coords,
    )
    return xr.DataTree.from_dict({"posterior": posterior})


def test_prior_predictive_preserves_shape_support_and_seed(circular_model):
    """Prior draws should retain `[rt, response]` and repeat exactly by seed."""
    first = circular_model.sample_prior_predictive(draws=3, random_seed=519)
    second = circular_model.sample_prior_predictive(draws=3, random_seed=519)
    different = circular_model.sample_prior_predictive(draws=3, random_seed=520)

    first_values = first["prior_predictive"][OBSERVED_NAME].values
    second_values = second["prior_predictive"][OBSERVED_NAME].values
    different_values = different["prior_predictive"][OBSERVED_NAME].values

    np.testing.assert_array_equal(first_values, second_values)
    assert not np.array_equal(first_values, different_values)
    assert first["prior_predictive"][OBSERVED_NAME].dims == (
        "chain",
        "draw",
        OBSERVATION_DIM,
        RESPONSE_DIM,
    )
    assert first_values.shape == (1, 3, 6, 2)
    np.testing.assert_array_equal(
        first["prior_predictive"].coords[RESPONSE_DIM].values,
        np.array([0, 1]),
    )
    _assert_predictive_contract(first_values)


def test_posterior_predictive_advances_one_seeded_safe_mode_stream(circular_model):
    """Chunked posterior draws should repeat by seed without repeating chunks."""
    first = circular_model.sample_posterior_predictive(
        dt=_fixed_posterior(),
        inplace=False,
        safe_mode=True,
        random_seed=997,
    )
    second = circular_model.sample_posterior_predictive(
        dt=_fixed_posterior(),
        inplace=False,
        safe_mode=True,
        random_seed=997,
    )
    different = circular_model.sample_posterior_predictive(
        dt=_fixed_posterior(),
        inplace=False,
        safe_mode=True,
        random_seed=998,
    )

    assert first is not None
    assert second is not None
    assert different is not None
    first_values = first["posterior_predictive"][OBSERVED_NAME].values
    second_values = second["posterior_predictive"][OBSERVED_NAME].values
    different_values = different["posterior_predictive"][OBSERVED_NAME].values

    np.testing.assert_array_equal(first_values, second_values)
    assert not np.array_equal(first_values, different_values)
    assert not np.array_equal(first_values[:, 0], first_values[:, 10])
    assert first["posterior_predictive"][OBSERVED_NAME].dims == (
        "chain",
        "draw",
        OBSERVATION_DIM,
        RESPONSE_DIM,
    )
    assert first_values.shape == (1, 12, 6, 2)
    np.testing.assert_array_equal(
        first["posterior_predictive"].coords["draw"].values,
        np.arange(12),
    )
    _assert_predictive_contract(first_values)


def test_circular_model_round_trip_reconstructs_predictive_state(tmp_path):
    """Existing HSSM persistence should reconstruct the config and custom RV."""
    model = _build_circular_model()
    model.save_model(
        model_name="circular_round_trip",
        base_path=tmp_path,
        allow_absolute_base_path=True,
    )

    model_path = tmp_path / "circular_round_trip"
    restored = hssm.HSSM.load_model(model_path)

    assert isinstance(restored, hssm.HSSM)
    assert restored.model_name == "circular_diffusion"
    assert restored._init_args["model"] == "circular_diffusion"
    assert restored._init_args["p_outlier"] is None
    assert restored.model_config.rv is simulate_circular_diffusion
    assert restored.data.equals(model.data)
    assert (model_path / "model.pkl").is_file()
    assert not (model_path / "traces.nc").exists()

    original_draws = model.sample_prior_predictive(draws=2, random_seed=1623)[
        "prior_predictive"
    ][OBSERVED_NAME].values
    restored_draws = restored.sample_prior_predictive(draws=2, random_seed=1623)[
        "prior_predictive"
    ][OBSERVED_NAME].values
    np.testing.assert_array_equal(original_draws, restored_draws)
    _assert_predictive_contract(restored_draws)

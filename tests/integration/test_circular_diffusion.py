"""Exercise the public native circular model from simulation through inference."""

import jax
import numpy as np
import pandas as pd
import pytensor
import pytest

import hssm
from hssm.config import Config
from hssm.defaults import default_model_config
from hssm.modelconfig import get_default_model_config


def test_registered_circular_metadata_roundtrips():
    """A fresh config must resolve to the same simulator and response metadata."""
    Config.from_defaults("circular_diffusion", None)
    assert default_model_config["circular_diffusion"] == get_default_model_config(
        "circular_diffusion"
    )


def test_registered_simulation_uses_hssm_parameter_names():
    """Dictionary and positional forms preserve identical physical observations."""
    params = {"v_x": -0.8, "v_y": 1.2, "a": 1.0, "t": 0.15}
    sim = hssm.simulate_data("circular_diffusion", params, 32, random_state=42)
    positional = hssm.simulate_data(
        "circular_diffusion", list(params.values()), 32, random_state=42
    )
    pd.testing.assert_frame_equal(sim, positional)
    assert list(sim.columns) == ["rt", "response"]
    assert (sim.rt > params["t"]).all()
    assert ((-np.pi <= sim.response) & (sim.response < np.pi)).all()
    assert (sim.response != np.trunc(sim.response)).all()
    with pytest.raises(ValueError, match="Specify only one"):
        hssm.simulate_data("circular_diffusion", {**params, "vx": 0.0}, 1)


def test_named_model_gradient_and_predictive_draws():
    """Check signed drift against its exact score and draw native RT/angle pairs."""
    data = pd.DataFrame({"rt": [0.49, 0.6, 1.12], "response": [-2.7, 0.1, 1.6]})
    with pytensor.config.change_flags(floatX="float64"), jax.enable_x64(True):
        model = hssm.HSSM(
            data=data,
            model="circular_diffusion",
            p_outlier=None,
            v_y=0.0,
            a=1.0,
            t=0.2,
        )
        assert model.choices is None
        assert model.loglik_kind == "analytical"
        # The default prior for the one free parameter v_x is N(0,1).
        v = 0.7
        score = model.pymc_model.compile_dlogp()({"v_x": np.array(v)})
        expected = np.cos(data.response).sum() - v * ((data.rt - 0.2).sum() + 1)
        np.testing.assert_allclose(score, [expected])
        draws = model.sample_prior_predictive(draws=5, random_seed=7)
        pairs = draws.prior_predictive[model.response_str].values
        assert pairs.shape == (1, 5, len(data), 2)
        assert (pairs[..., 0] > 0.2).all()
        assert ((pairs[..., 1] >= -np.pi) & (pairs[..., 1] < np.pi)).all()


def test_out_of_range_angles_are_rejected():
    """The stored angle is a circular coordinate, not an integer choice code."""
    data = pd.DataFrame({"rt": [0.5, 0.6], "response": [0.4, np.pi]})
    with pytest.raises(ValueError, match="response"):
        hssm.HSSM(data=data, model="circular_diffusion", p_outlier=None)

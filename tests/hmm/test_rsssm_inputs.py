"""Inputs that RSSSM used to accept wrongly, or to reject with a misleading error."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import hssm
from hssm import RSSSM
from hssm.hmm.utils import pad_and_align_to_T_max

matplotlib.use("Agg")


def _ddm_data(size=60):
    """One participant's DDM trials; this module needs no shared fixture."""
    theta = {"v": 1.0, "a": 0.8, "z": 0.5, "t": 0.3}
    return hssm.simulate_data(model="ddm", theta=theta, size=size, random_state=1)


@pytest.mark.parametrize("dtype", ["int64", "Int64", "string", "object"])
def test_contiguity_guard_does_not_depend_on_the_id_dtype(dtype):
    """Nullable id columns: interleaved rows were accepted, sorted rows rejected."""

    def frame(ids):
        values = ids if dtype.lower() == "int64" else [f"p{i}" for i in ids]
        return pd.DataFrame(
            {
                "participant_id": pd.Series(values, dtype=dtype),
                "rt": [0.5, 0.6, 0.7],
                "response": [1.0, -1.0, 1.0],
            }
        )

    padded, *_ = pad_and_align_to_T_max(
        frame([0, 0, 1]), "participant_id", ["rt", "response"]
    )
    assert padded.shape == (2, 2, 2)
    with pytest.raises(ValueError, match="contiguous per participant"):
        pad_and_align_to_T_max(frame([0, 1, 0]), "participant_id", ["rt", "response"])


@pytest.mark.parametrize("param", ["z", "v"])  # a shared parameter, and the anchor
def test_bounded_prior_is_refused_not_replaced(param):
    """A bounded `hssm.Prior` has empty `args`; it was built as a standard Normal."""
    data = _ddm_data()
    bounded = hssm.Prior("Normal", mu=0.5, sigma=0.1, bounds=(0.4, 0.6))
    with pytest.raises(NotImplementedError, match="bounds or a custom `dist`"):
        RSSSM(data=data, model="ddm", K=2, switching_params=["v"], **{param: bounded})
    # The alternative the message names builds the bounded prior that was asked for.
    model = RSSSM(
        data=data,
        model="ddm",
        K=2,
        switching_params=["v"],
        z={
            "name": "TruncatedNormal",
            "mu": 0.5,
            "sigma": 0.1,
            "lower": 0.4,
            "upper": 0.6,
        },
    )
    draws = pm.draw(model.pymc_model["z"], 500, random_seed=1)
    assert 0.4 <= draws.min() and draws.max() <= 0.6


@pytest.mark.parametrize("switching", [["v"], ["t"]])
def test_start_values_lie_inside_narrower_bounds(switching):
    """`ddm_st`'s network has t >= 0.25; the shared default start for t is 0.025."""
    theta = {"v": 0.8, "a": 1.2, "z": 0.5, "t": 0.4, "st": 0.1}
    data = hssm.simulate_data(model="ddm_st", theta=theta, size=60, random_state=0)
    model = RSSSM(data=data, model="ddm_st", K=2, switching_params=switching)

    point = model.pymc_model.initial_point()
    assert all(np.all(np.isfinite(value)) for value in point.values())
    lower, upper = model.bounds["t"]
    t_start = np.atleast_1d(
        model.pymc_model.rvs_to_initial_values[model.pymc_model["t"]]
    )
    assert np.all((lower < t_start) & (t_start < upper))


def test_emission_without_categorical_choices_is_refused():
    """`circular_diffusion` failed with "Invalid responses found in your dataset"."""
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {"rt": rng.uniform(0.4, 1.5, 20), "response": rng.uniform(-3.0, 3.0, 20)}
    )
    with pytest.raises(NotImplementedError, match="categorical choices only"):
        RSSSM(data=data, model="circular_diffusion", K=2, switching_params=["v_x"])


def test_true_regimes_stay_on_the_probability_axis():
    """With K = 3, a true regime of 2 was drawn at y = 2, outside the [0, 1] axis."""
    model = RSSSM(data=_ddm_data(6), model="ddm", K=3, switching_params=["v"])
    frequency = np.full((1, 6, 3), 1.0 / 3.0)
    regimes = xr.DataTree.from_dict(
        {
            "posterior_regimes": xr.Dataset(
                {
                    "regime_sample_frequency": (
                        ("participant", "trial", "regime"),
                        frequency,
                    )
                }
            )
        }
    )
    ax = model.plot_regime_recovery(regimes, true_regimes=[0, 1, 2, 2, 1, 0])
    np.testing.assert_allclose(ax.lines[-1].get_ydata(), [0, 0.5, 1, 1, 0.5, 0])

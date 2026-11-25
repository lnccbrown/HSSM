"""Regression checks for the public RT and circular predictive plots."""

from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import hssm


@pytest.fixture(scope="module")
def model():
    """Build a native circular model with one observed report in each quadrant."""
    data = pd.DataFrame(
        {"rt": [0.3, 0.5, 1.1, 1.5], "response": [-3.0, -0.2, 0.3, 3.0]}
    )
    return hssm.HSSM(
        model="circular_diffusion", data=data, p_outlier=None, a=1.0, t=0.1
    )


def _predictives(binary=False, omissions=False):
    # Four trials in each of four replicate data sets (two chains, two draws).
    # Deliberately concentrated replicates distinguish per-draw bands from pooling.
    angles = np.array([[-np.pi, -np.pi + 0.1], [-0.7, np.pi - 0.1]])
    bins = np.array([[0, 0], [1, 3]])
    responses = np.sign(angles) if binary else angles
    pairs = np.stack((bins + 0.5, responses), axis=-1)
    pairs = np.repeat(pairs[:, :, None, :], 4, axis=2)
    if omissions:
        # Unequal valid sample sizes: pooling all trials now gives the WRONG mean.
        pairs[0, 0, 1:, :] = np.nan
    dims = ("chain", "draw", "__obs__", "rt,response_dim")
    return xr.DataTree.from_dict(
        {"prior_predictive": xr.Dataset({"rt,response": (dims, pairs)})}
    )


def test_circular_plots_require_choices_only_for_choice_diagnostics(model):
    """Choice-specific diagnostics still fail before sampling."""
    with pytest.raises(ValueError, match="require categorical responses"):
        model.plot_quantile_probability(cond="condition")
    with pytest.raises(ValueError, match="require categorical responses"):
        model.plot_model_cartoon()


def test_default_panels_share_predictive_sample(model, monkeypatch):
    """One predictive simulation supplies both panels, with matching replicate data."""
    sample = Mock(return_value=_predictives(omissions=True))
    monkeypatch.setattr(model, "sample_prior_predictive", sample)
    rt, angle = model.plot_predictive(
        dt=xr.DataTree(),
        predictive_group="prior_predictive",
        n_samples=2,
        bins=4,
        x_range=(0, 4),
        uncertainty=None,
    )
    try:
        sample.assert_called_once()
        assert rt.figure is angle.figure
        assert (rt.name, angle.name) == ("rectilinear", "polar")
        theta, prob = angle.lines[0].get_data()
        assert theta[-1] - theta[0] == pytest.approx(2 * np.pi)
        assert prob[0] == prob[-1]
        # Unit-width RT bins encode the same bin assignments as the polar panel.
        # The polar line repeats its first value at the seam, excluded here.
        np.testing.assert_allclose(rt.lines[0].get_ydata(), prob[:-1])
    finally:
        plt.close(rt.figure)


def test_angular_predictive_fractions_and_pointwise_band(model):
    """Compare masses and uncertainty with four manually constructed replicates."""
    ax = model.plot_predictive(
        dt=_predictives(omissions=True),
        predictive_group="prior_predictive",
        outcome="response",
        n_samples=None,
        bins=4,
        step=False,
        hdi=0.5,
    )
    try:
        assert ax.name == "polar"
        curves = {line.get_label(): line for line in ax.lines}
        theta, mean = curves["predicted"].get_data()
        # Full circle at the centers of four equal bins, closed at the seam.
        np.testing.assert_allclose(theta, np.arange(-3, 6, 2) * np.pi / 4)
        np.testing.assert_allclose(mean, [0.5, 0.25, 0, 0.25, 0.5])
        assert np.sum(mean[:-1]) == pytest.approx(1)
        np.testing.assert_allclose(curves["observed"].get_ydata(), 0.25)
        # Quantiles across the four trial-fraction vectors (not across trials).
        (ribbon,) = ax.collections
        vertices = ribbon.get_paths()[0].vertices
        for center, upper in zip(theta[:-1], [1, 0.25, 0, 0.25], strict=True):
            r = vertices[np.isclose(vertices[:, 0], center), 1]
            np.testing.assert_allclose([r.min(), r.max()], [0, upper])
    finally:
        plt.close(ax.figure)


def test_circular_rt_only_and_categorical_signed_rt(model):
    """Opt-in RT-only stays unsigned; two-choice calls retain a single signed axis."""
    common = {"predictive_group": "prior_predictive", "n_samples": None}
    # In a notebook a previously drawn circular panel can still be the current axes.
    previous, _ = plt.subplots(subplot_kw={"projection": "polar"})
    rt = model.plot_predictive(outcome="rt", dt=_predictives(), **common)
    try:
        assert rt.name == "rectilinear" and rt.lines[0].get_xdata().min() > 0
    finally:
        plt.close(previous)
        plt.close(rt.figure)

    binary = hssm.HSSM(data=model.data.assign(response=[-1, -1, 1, 1]), model="ddm")
    signed = binary.plot_predictive(dt=_predictives(binary=True), **common)
    try:
        assert signed.name == "rectilinear"
        assert signed.lines[0].get_xdata().min() < 0 < signed.lines[0].get_xdata().max()
    finally:
        plt.close(signed.figure)


def test_polar_histogram_uses_the_declared_response_column(model):
    """Both simulated and observed angles come from the configured physical column."""
    renamed = hssm.HSSM(
        data=model.data.rename(columns={"response": "bearing"}),
        model="circular_diffusion",
        model_config={
            "response": ["rt", "bearing"],
            "response_domains": {
                "bearing": {"kind": "circular", "bounds": (-np.pi, np.pi)}
            },
        },
        p_outlier=None,
        a=1.0,
        t=0.1,
    )
    prior = renamed.sample_prior_predictive(draws=3, random_seed=23)
    rt, angle = renamed.plot_predictive(
        dt=prior, predictive_group="prior_predictive", n_samples=None, step=False
    )
    try:
        assert angle.name == "polar"
        assert angle.lines[0].get_ydata()[:-1].sum() == pytest.approx(1)
    finally:
        plt.close(rt.figure)


def test_unsupported_facets_fail_before_sampling(model):
    """An angular facet request should explain its limitation without sampling."""
    with pytest.raises(ValueError, match="(?i)facet|outcome=.rt."):
        model.plot_predictive(row="condition")

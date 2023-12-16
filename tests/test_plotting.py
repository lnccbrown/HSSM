"""Test plotting module."""

import pytest

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import hssm
from hssm.plotting.utils import (
    _get_plotting_df,
    _xarray_to_df,
    _get_title,
    _subset_df,
    _row_mask_with_error,
    _process_df_for_qp_plot,
)
from hssm.plotting.posterior_predictive import (
    _plot_posterior_predictive_1D,
    _plot_posterior_predictive_2D,
    plot_posterior_predictive,
)
from hssm.plotting.quantile_probability import (
    _plot_quantile_probability_1D,
    _plot_quantile_probability_2D,
    plot_quantile_probability,
)

hssm.set_floatX("float32")


def test__get_title():
    assert _get_title(("a"), ("b")) == "a = b"
    assert _get_title(("a", "b"), ("c", "d")) == "a = c | b = d"


def test__subset_df(cavanagh_test):
    with pytest.raises(ValueError):
        _row_mask_with_error(cavanagh_test, "conf", "Bad value")
    cav_subset = cavanagh_test.loc[
        (cavanagh_test["participant_id"] == 1) & (cavanagh_test["conf"] == "LC"), :
    ]
    subset_from_func = _subset_df(cavanagh_test, ["participant_id", "conf"], [1, "LC"])
    assert cav_subset.equals(subset_from_func)


@pytest.mark.parametrize(
    ["n_samples", "expected"],
    [
        (0, "error"),
        (1, 1000),
        (2, 2000),
        (3, 2000),
        (1.0, 2000),
        (0.0, "error"),
        (0.5, 1000),
        (2.0, "error"),
        (None, 2000),
    ],
)
def test__xarray_to_df(caplog, posterior, n_samples, expected):
    """Test _get_posterior_samples."""
    if expected == "error":
        with pytest.raises(ValueError):
            df = _xarray_to_df(posterior, n_samples=n_samples)
    else:
        df = _xarray_to_df(posterior, n_samples=n_samples)
        if n_samples and n_samples > posterior.draw.size:
            assert "n_samples > n_draws" in caplog.text

        assert len(df) == expected
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["chain", "draw", "obs_n"]
        obs_n = df.index.get_level_values(2)
        assert obs_n[0] == 0
        assert obs_n[-1] == 499
        assert np.all(obs_n.value_counts() == expected // 500)
        assert df.columns[0] == "rt"


def test__get_plotting_df(posterior, cavanagh_test):
    """Test _get_plotting_df."""

    # Makes a mock InferenceData object
    posterior_dataset = xr.Dataset(data_vars={"rt,response": posterior})
    idata = az.InferenceData(posterior_predictive=posterior_dataset)

    df = _get_plotting_df(idata, cavanagh_test, extra_dims=["participant_id", "conf"])
    assert len(df) == 2500
    assert isinstance(df.index, pd.MultiIndex)
    print(df)
    assert df.columns.to_list() == [
        "observed",
        "rt",
        "response",
        "participant_id",
        "conf",
    ]
    assert df.isna().sum().sum() == 0
    np.testing.assert_array_equal(
        df.iloc[2000:, 1:].values,
        cavanagh_test.loc[:, ["rt", "response", "participant_id", "conf"]].values,
    )

    df_no_original = _get_plotting_df(idata, data=None)
    assert df_no_original.shape == (2000, 3)
    assert df_no_original.columns.to_list() == ["observed", "rt", "response"]

    with pytest.raises(ValueError):
        _get_plotting_df(idata, data=None, extra_dims=["participant_id", "conf"])


def test__plot_posterior_predictive_1D(cav_idata, cavanagh_test):
    df = _get_plotting_df(
        cav_idata, cavanagh_test, extra_dims=["participant_id", "conf"]
    )
    df["Response Time"] = df["rt"] * np.where(df["response"] == 0, -1, 1)

    _, ax1 = plt.subplots()
    ax1 = _plot_posterior_predictive_1D(df, ax=ax1)
    assert len(ax1.get_lines()) == 2

    _, ax2 = plt.subplots()
    ax2 = _plot_posterior_predictive_1D(df, plot_data=False, ax=ax2)
    assert len(ax2.get_lines()) == 1


def test__plot_posterior_predictive_2D(cav_idata, cavanagh_test):
    df = _get_plotting_df(
        cav_idata, cavanagh_test, extra_dims=["participant_id", "conf"]
    )
    df["Response Time"] = df["rt"] * np.where(df["response"] == 0, -1, 1)

    g1 = _plot_posterior_predictive_2D(
        df,
        row="participant_id",
        col="conf",
    )
    assert len(g1.fig.axes) == 5 * 2
    assert len(g1.fig.axes[0].get_lines()) == 2

    g2 = _plot_posterior_predictive_2D(
        df,
        plot_data=False,
        row="participant_id",
        col="conf",
    )
    assert len(g2.fig.axes) == 5 * 2
    assert len(g2.fig.axes[0].get_lines()) == 1


def test_plot_posterior_predictive(cav_idata, cavanagh_test):
    # Mock model object
    model = hssm.HSSM(
        data=cavanagh_test,
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                    "theta": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                },
                "formula": "v ~ (1|participant_id) + theta",
                "link": "identity",
            },
        ],
    )  # Doesn't matter what model or data we use here
    with pytest.raises(ValueError):
        plot_posterior_predictive(model)

    model._inference_obj = cav_idata.copy()
    _, ax1 = plt.subplots()
    ax1 = plot_posterior_predictive(model, ax=ax1)  # Should work directly
    assert len(ax1.get_lines()) == 2

    delattr(model.traces, "posterior_predictive")
    _, ax2 = plt.subplots()
    ax2 = plot_posterior_predictive(
        model, ax=ax2, n_samples=2
    )  # Should sample posterior predictive
    assert len(ax2.get_lines()) == 2
    assert "posterior_predictive" in model.traces
    assert model.traces.posterior_predictive.draw.size == 2

    with pytest.raises(ValueError):
        plot_posterior_predictive(model, groups="participant_id")
    with pytest.raises(ValueError):
        plot_posterior_predictive(model, groups_order=["5", "4"])

    plots = plot_posterior_predictive(
        model, row="stim", col="participant_id", groups="conf"
    )
    assert len(plots) == 2
    # Lengths might defer because of subsetting the data frame
    assert len(plots[0].fig.axes) == 5
    assert len(plots[1].fig.axes) == 5 * 2

    plots = plot_posterior_predictive(
        model,
        row="stim",
        plot_data=False,
        col="participant_id",
        groups="conf",
        groups_order=["LC"],
    )
    assert len(plots) == 1
    assert len(plots[0].fig.axes) == 5
    assert len(plots[0].fig.axes[0].get_lines()) == 1

    with pytest.raises(ValueError):
        plot_posterior_predictive(
            model,
            row="stim",
            plot_data=False,
            col="participant_id",
            groups=["conf", "dbs"],
            groups_order=["LC"],
        )

    plots = plot_posterior_predictive(
        model,
        row="stim",
        plot_data=False,
        col="participant_id",
        groups=["conf", "dbs"],
        groups_order={"conf": ["LC"]},
    )
    assert len(plots) == len(
        cavanagh_test[cavanagh_test["conf"] == "LC"].groupby(["conf", "dbs"])
    )


def test__process_df_for_qp_plot(cav_idata, cavanagh_test):
    df = _get_plotting_df(
        cav_idata, cavanagh_test, extra_dims=["participant_id", "conf"]
    )

    processed_df = _process_df_for_qp_plot(df, 6, "conf", None)

    assert "conf" in processed_df.columns
    assert "is_correct" in processed_df.columns
    assert processed_df["quantile"].nunique() == 4
    assert np.all(
        processed_df.groupby(["observed", "chain", "draw", "conf", "quantile"])[
            "proportion"
        ].sum()
        == 1
    )


def has_twin(ax):
    """Checks if an axes has a twin axes with the same bounds.

    Credit: https://stackoverflow.com/questions/36209575/how-to-detect-if-a-twin-axis-has-been-generated-for-a-matplotlib-axis
    """
    for other_ax in ax.figure.axes:
        if other_ax is ax:
            continue
        if other_ax.bbox.bounds == ax.bbox.bounds:
            return True
    return False


def test__plot_quantile_probability_1D(cav_idata, cavanagh_test):
    df = _get_plotting_df(cav_idata, cavanagh_test, extra_dims=["stim"])
    ax = _plot_quantile_probability_1D(df, cond="stim")

    assert has_twin(ax)
    assert ax.get_xlabel() == "Proportion"
    assert ax.get_ylabel() == "rt"
    assert ax.get_title() == "Quantile Probability Plot"


def test__plot_quantile_probability_2D(cav_idata, cavanagh_test):
    df = _get_plotting_df(
        cav_idata, cavanagh_test, extra_dims=["participant_id", "stim"]
    )
    g = _plot_quantile_probability_2D(df, cond="stim", col="participant_id", col_wrap=3)

    assert len(g.fig.axes) == 10

    df = _get_plotting_df(
        cav_idata, cavanagh_test, extra_dims=["participant_id", "stim", "conf"]
    )
    g = _plot_quantile_probability_2D(df, cond="stim", col="participant_id", row="conf")

    assert len(g.fig.axes) == 5 * 4


def test_plot_quantile_probability(cav_idata, cavanagh_test):
    # Mock model object
    model = hssm.HSSM(
        data=cavanagh_test,
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                    "theta": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                },
                "formula": "v ~ (1|participant_id) + theta",
                "link": "identity",
            },
        ],
    )  # Doesn't matter what model or data we use here
    with pytest.raises(ValueError):
        plot_quantile_probability(model, cond="stim")

    model._inference_obj = cav_idata.copy()
    ax1 = plot_quantile_probability(
        model, cond="stim", data=cavanagh_test
    )  # Should work directly
    assert len(ax1.get_lines()) == 9

    delattr(model.traces, "posterior_predictive")
    ax2 = plot_quantile_probability(
        model, cond="stim", data=cavanagh_test, n_samples=2
    )  # Should sample posterior predictive
    assert len(ax2.get_lines()) == 9
    assert "posterior_predictive" in model.traces
    assert model.traces.posterior_predictive.draw.size == 2

    with pytest.raises(ValueError):
        plot_quantile_probability(model, groups="participant_id", cond="stim")
    with pytest.raises(ValueError):
        plot_quantile_probability(model, groups_order=["5", "4"], cond="stim")

    plots = plot_quantile_probability(
        model, row="dbs", col="participant_id", cond="stim", groups="conf"
    )
    assert len(plots) == 2

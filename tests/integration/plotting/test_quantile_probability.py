"""Tests for quantile-probability plotting helpers and public API."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import hssm
from hssm.plotting.quantile_probability import (
    _plot_quantile_probability_1D,
    _plot_quantile_probability_2D,
    plot_quantile_probability,
)
from hssm.plotting.utils import _get_plotting_df

hssm.set_floatX("float32")


def has_twin(ax):
    """Check whether an axes has a twin axes with the same bounds."""
    for other_ax in ax.figure.axes:
        if other_ax is ax:
            continue
        if other_ax.bbox.bounds == ax.bbox.bounds:
            return True
    return False


def test__plot_quantile_probability_1D_uses_passed_ax():
    """A user-supplied ``ax`` is drawn on, not forwarded to seaborn twice."""
    rng = np.random.default_rng(0)
    groups = [("observed", -1, [-1]), ("predicted", 0, [0, 1])]
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "observed": observed,
                    "chain": chain,
                    "draw": draw,
                    "stim": stim,
                    "rt": rng.uniform(0.3, 2.0, 50),
                    "response": rng.choice([-1, 1], 50),
                }
            )
            for observed, chain, draws in groups
            for draw in draws
            for stim in ["LL", "WW"]
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots()

    assert _plot_quantile_probability_1D(df, cond="stim", ax=ax) is ax
    assert ax.lines
    plt.close(fig)


def test_plot_quantile_probability_forwards_marker_to_seaborn():
    """A top-level ``marker`` styles the lines instead of being read as a column."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "rt": rng.uniform(0.3, 2.0, 200),
            "response": rng.choice([-1.0, 1.0], 200),
            "stim": rng.choice(["LL", "WW"], 200),
        }
    )
    model = hssm.HSSM(data=df, model="ddm")
    fig, ax = plt.subplots()

    plot_quantile_probability(
        model, cond="stim", data=df, predictive_group=None, ax=ax, marker="s"
    )

    # "s" is not a default marker of either seaborn call, so it must come from kwargs.
    assert {line.get_marker() for line in ax.lines} == {"s"}
    plt.close(fig)


@pytest.mark.slow
class TestQuantileProbabilityPlotting:
    """Tests for functions in hssm.plotting.quantile_probability."""

    @pytest.mark.parametrize("predictive_style", ["points", "ellipse", "both"])
    def test__plot_quantile_probability_1D(
        self, cav_dt, cavanagh_test, predictive_style
    ):
        """Check 1D quantile-probability plotting attributes."""
        df = _get_plotting_df(cav_dt, cavanagh_test, extra_dims=["stim"])
        ax = _plot_quantile_probability_1D(
            df, cond="stim", predictive_style=predictive_style
        )

        assert has_twin(ax)
        assert ax.get_xlabel() == "Proportion"
        assert ax.get_ylabel() == "rt"
        assert ax.get_title() == "Quantile Probability Plot"

    @pytest.mark.parametrize("predictive_style", ["points", "ellipse", "both"])
    def test__plot_quantile_probability_2D(
        self, cav_dt, cavanagh_test, predictive_style
    ):
        """Check 2D quantile-probability plotting grid dimensions."""
        df = _get_plotting_df(
            cav_dt, cavanagh_test, extra_dims=["participant_id", "stim"]
        )
        g = _plot_quantile_probability_2D(
            df,
            cond="stim",
            col="participant_id",
            col_wrap=3,
            predictive_style=predictive_style,
        )
        assert len(g.figure.axes) == 10

        df = _get_plotting_df(
            cav_dt, cavanagh_test, extra_dims=["participant_id", "stim", "conf"]
        )
        g = _plot_quantile_probability_2D(
            df,
            cond="stim",
            col="participant_id",
            row="conf",
            predictive_style=predictive_style,
        )
        assert len(g.figure.axes) == 5 * 4

    @pytest.mark.parametrize("predictive_style", ["points", "ellipse", "both"])
    def test_plot_quantile_probability(self, cav_dt, cavanagh_test, predictive_style):
        """Check public quantile-probability plotting API behavior."""
        model = hssm.HSSM(
            data=cavanagh_test,
            include=[
                {
                    "name": "v",
                    "prior": {
                        "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                        "theta": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                    },
                    "formula": "v ~ theta + (1|participant_id)",
                    "link": "identity",
                },
            ],
        )
        with pytest.raises(ValueError):
            plot_quantile_probability(
                model, cond="stim", predictive_style=predictive_style
            )

        model._inference_obj = cav_dt.copy()
        ax1 = plot_quantile_probability(
            model, cond="stim", data=cavanagh_test, predictive_style=predictive_style
        )
        assert ax1 is not None

        del model.traces["posterior_predictive"]
        ax2 = plot_quantile_probability(
            model, cond="stim", data=cavanagh_test, n_samples=2
        )
        assert ax2 is not None
        assert "posterior_predictive" in model.traces
        assert model.traces["posterior_predictive"].draw.size == 2

        with pytest.raises(ValueError):
            plot_quantile_probability(
                model,
                groups="participant_id",
                cond="stim",
                predictive_style=predictive_style,
            )
        with pytest.raises(ValueError):
            plot_quantile_probability(
                model,
                groups_order=["5", "4"],
                cond="stim",
                predictive_style=predictive_style,
            )

        plots = plot_quantile_probability(
            model,
            row="dbs",
            col="participant_id",
            cond="stim",
            groups="conf",
            predictive_style=predictive_style,
        )
        assert len(plots) == 2

    def test_plot_quantile_probability_no_predictive(self, cavanagh_test):
        """Test plotting only observed data when predictive_group is None."""
        model = hssm.HSSM(
            data=cavanagh_test,
            include=[
                {
                    "name": "v",
                    "prior": {
                        "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                    },
                    "formula": "v ~ 1",
                    "link": "identity",
                },
            ],
        )

        ax = plot_quantile_probability(
            model,
            cond="stim",
            data=cavanagh_test,
            predictive_group=None,
        )

        assert ax is not None

    def test_plot_quantile_probability_with_quantile_by(self, cav_dt, cavanagh_test):
        """Test quantile_probability plotting with quantile_by enabled."""
        model = hssm.HSSM(
            data=cavanagh_test,
            include=[
                {
                    "name": "v",
                    "prior": {
                        "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                        "theta": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                    },
                    "formula": "v ~ theta + (1|participant_id)",
                    "link": "identity",
                },
            ],
        )
        model._inference_obj = cav_dt.copy()

        ax1 = plot_quantile_probability(
            model,
            cond="stim",
            data=cavanagh_test,
            quantile_by="participant_id",
            n_samples=10,
        )
        assert ax1 is not None
        assert has_twin(ax1)

        g1 = plot_quantile_probability(
            model,
            cond="stim",
            data=cavanagh_test,
            col="conf",
            quantile_by="participant_id",
            n_samples=10,
        )
        assert g1 is not None
        assert len(g1.figure.axes) > 0

        ax2 = plot_quantile_probability(
            model,
            cond="stim",
            data=cavanagh_test,
            quantile_by=["participant_id"],
            n_samples=10,
        )
        assert ax2 is not None

        ax_no_grouping = plot_quantile_probability(
            model, cond="stim", data=cavanagh_test, quantile_by=None, n_samples=10
        )
        ax_with_grouping = plot_quantile_probability(
            model,
            cond="stim",
            data=cavanagh_test,
            quantile_by="participant_id",
            n_samples=10,
        )

        assert ax_no_grouping is not None
        assert ax_with_grouping is not None

        for style in ["points", "ellipse", "both"]:
            ax = plot_quantile_probability(
                model,
                cond="stim",
                data=cavanagh_test,
                quantile_by="participant_id",
                predictive_style=style,
                n_samples=10,
            )
            assert ax is not None


def test_plot_quantile_probability_groups_plot_only_their_own_data():
    """Each per-group grid holds that group's rows, not the whole dataset."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "rt": rng.uniform(0.3, 2.0, n),
            "response": rng.choice([-1.0, 1.0], n),
            "stim": rng.choice(["LL", "WW"], n),
            "conf": rng.choice(["high", "low"], n),
            "block": rng.choice(["b1", "b2"], n),
            "participant_id": np.repeat(["p1", "p2"], n // 2),
        }
    )
    model = hssm.HSSM(data=df, model="ddm")

    plots = plot_quantile_probability(
        model,
        cond="stim",
        data=df,
        predictive_group=None,
        row="conf",
        col="block",
        groups="participant_id",
    )

    assert [sorted(g.data["participant_id"].unique()) for g in plots] == [
        ["p1"],
        ["p2"],
    ]
    plt.close("all")

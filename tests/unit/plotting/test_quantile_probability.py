"""Tests for quantile-probability plotting helpers and public API."""

import numpy as np
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


@pytest.mark.slow
# TODO: move this file to tests/integration/plotting/test_quantile_probability.py #1117
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

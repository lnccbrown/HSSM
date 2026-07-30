"""Tests for predictive plotting helpers and public API."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import hssm
from hssm.plotting.predictive import (
    _plot_predictive_1D,
    _plot_predictive_2D,
    _process_lines,
    plot_predictive,
)
from hssm.plotting.utils import _get_plotting_df

hssm.set_floatX("float32")


class TestPredictivePlottingUnit:
    """Unit tests for small predictive plotting helpers."""

    def test__process_lines(self):
        """Line style and width helpers normalize sequences and dictionaries."""
        assert _process_lines(["--"], mode="linestyles") == ["--", "--"]
        assert _process_lines(("--", ":"), mode="linestyles") == ["--", ":"]
        assert _process_lines([1.5], mode="linewidths") == [1.5, 1.5]
        assert _process_lines((1.0, 2.0), mode="linewidths") == [1.0, 2.0]
        assert _process_lines({"predicted": "--"}, mode="linestyles") == ["--", "-"]
        assert _process_lines({"observed": 2.0}, mode="linewidths") == [1.25, 2.0]

        with pytest.raises(ValueError, match="Invalid mode"):
            _process_lines("-", mode="colors")
        with pytest.raises(ValueError, match="must be a str or a list of strs"):
            _process_lines(["-", 1], mode="linestyles")


# TODO: move these to tests/integration/plotting/test_predictive.py #1105
@pytest.mark.slow
class TestPredictivePlotting:
    """Tests for functions in hssm.plotting.predictive."""

    def test__plot_predictive_1D(self, cav_dt, cavanagh_test):
        """Check one-dimensional predictive plotting line counts."""
        df = _get_plotting_df(
            cav_dt, cavanagh_test, extra_dims=["participant_id", "conf"]
        )
        df["Response Time"] = df["rt"] * np.where(df["response"] == 0, -1, 1)

        _, ax1 = plt.subplots()
        ax1 = _plot_predictive_1D(df, ax=ax1)
        assert len(ax1.get_lines()) == 2

        _, ax2 = plt.subplots()
        ax2 = _plot_predictive_1D(df, plot_data=False, ax=ax2)
        assert len(ax2.get_lines()) == 1

    def test__plot_predictive_2D(self, cav_dt, cavanagh_test):
        """Check two-dimensional predictive plotting facet and line counts."""
        df = _get_plotting_df(
            cav_dt, cavanagh_test, extra_dims=["participant_id", "conf"]
        )
        df["Response Time"] = df["rt"] * np.where(df["response"] == 0, -1, 1)

        g1 = _plot_predictive_2D(
            df,
            row="participant_id",
            col="conf",
        )
        assert len(g1.figure.axes) == 5 * 2
        assert len(g1.figure.axes[0].get_lines()) == 2

        g2 = _plot_predictive_2D(
            df,
            plot_data=False,
            row="participant_id",
            col="conf",
        )
        assert len(g2.figure.axes) == 5 * 2
        assert len(g2.figure.axes[0].get_lines()) == 1

    def test_plot_predictive(self, cav_dt, cavanagh_test):
        """Check public predictive plotting across direct and sampled inputs."""
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
            plot_predictive(model)

        model._inference_obj = cav_dt.copy()
        _, ax1 = plt.subplots()
        ax1 = plot_predictive(model, ax=ax1)
        assert len(ax1.get_lines()) == 2

        del model.traces["posterior_predictive"]
        _, ax2 = plt.subplots()
        ax2 = plot_predictive(model, ax=ax2, n_samples=2)
        assert len(ax2.get_lines()) == 2
        assert "posterior_predictive" in model.traces
        assert model.traces["posterior_predictive"].draw.size == 2

        with pytest.raises(ValueError):
            plot_predictive(model, groups="participant_id")
        with pytest.raises(ValueError):
            plot_predictive(model, groups_order=["5", "4"])

        plots = plot_predictive(model, row="stim", col="participant_id", groups="conf")
        assert len(plots) == 2
        assert len(plots[0].figure.axes) == 5
        assert len(plots[1].figure.axes) == 5 * 2

        plots = plot_predictive(
            model,
            row="stim",
            plot_data=False,
            col="participant_id",
            groups="conf",
            groups_order=["LC"],
        )
        assert len(plots) == 1
        assert len(plots[0].figure.axes) == 5
        assert len(plots[0].figure.axes[0].get_lines()) == 1

        with pytest.raises(ValueError):
            plot_predictive(
                model,
                row="stim",
                plot_data=False,
                col="participant_id",
                groups=["conf", "dbs"],
                groups_order=["LC"],
            )

        plots = plot_predictive(
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

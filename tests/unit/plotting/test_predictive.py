"""Tests for predictive plotting helpers and public API."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import hssm
from hssm.plotting.predictive import (
    _curve_xy,
    _kde_density,
    _legend_order,
    _plot_predictive_1D,
    _plot_predictive_2D,
    _process_colors,
    _process_lines,
    plot_predictive,
)
from hssm.plotting.utils import (
    DEFAULT_PREDICTIVE_COLORS,
    _get_plotting_df,
    _hdi_to_intervals,
)

hssm.set_floatX("float32")


def _synthetic_plotting_df(
    n_chains: int = 2, n_draws: int = 5, n_obs: int = 60
) -> pd.DataFrame:
    """Build a small long-format df matching _get_plotting_df's contract."""
    rng = np.random.default_rng(42)
    frames = []
    for chain in range(n_chains):
        for draw in range(n_draws):
            frames.append(
                pd.DataFrame(
                    {
                        "observed": "predicted",
                        "rt": rng.lognormal(0.0, 0.4, n_obs),
                        "response": rng.choice([-1, 1], n_obs),
                    },
                    index=pd.MultiIndex.from_product(
                        [[chain], [draw], range(n_obs)],
                        names=["chain", "draw", "obs_n"],
                    ),
                )
            )
    frames.append(
        pd.DataFrame(
            {
                "observed": "observed",
                "rt": rng.lognormal(0.0, 0.4, n_obs),
                "response": rng.choice([-1, 1], n_obs),
            },
            index=pd.MultiIndex.from_product(
                [[-1], [-1], range(n_obs)], names=["chain", "draw", "obs_n"]
            ),
        )
    )
    df = pd.concat(frames)
    df["rt"] = df["rt"] * df["response"]
    return df


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

    def test__hdi_to_intervals(self):
        """Interval specs normalize to quantile pairs, widest first."""
        # single mass
        lo, hi = _hdi_to_intervals(0.9)[0]
        assert (lo, hi) == pytest.approx((0.05, 0.95))
        # percentage string
        assert _hdi_to_intervals("90%")[0] == pytest.approx((0.05, 0.95))
        # legacy tuple stays a quantile pair, never fans out
        assert _hdi_to_intervals((0.05, 0.95)) == [(0.05, 0.95)]
        # a list fans out into graded bands, sorted widest first
        intervals = _hdi_to_intervals([0.5, 0.94])
        assert intervals[0] == pytest.approx((0.03, 0.97))
        assert intervals[1] == pytest.approx((0.25, 0.75))
        # mixed spec types in one list
        intervals = _hdi_to_intervals([(0.25, 0.75), "94%"])
        assert intervals[0] == pytest.approx((0.03, 0.97))
        with pytest.raises(ValueError, match="must not be empty"):
            _hdi_to_intervals([])

    def test__curve_xy(self):
        """Step curves anchor to edges; polygons anchor to bin centers."""
        edges = np.array([0.0, 1.0, 2.0])
        values = np.array([0.3, 0.7])
        x_step, y_step = _curve_xy(edges, values, step=True)
        np.testing.assert_allclose(x_step, [0.0, 1.0, 1.0, 2.0])
        np.testing.assert_allclose(y_step, [0.3, 0.3, 0.7, 0.7])
        x_poly, y_poly = _curve_xy(edges, values, step=False)
        np.testing.assert_allclose(x_poly, [0.5, 1.5])
        np.testing.assert_allclose(y_poly, [0.3, 0.7])

    def test__process_colors(self):
        """Color specs resolve to a [predicted, observed] pair."""
        assert _process_colors(None, plot_data=True) == DEFAULT_PREDICTIVE_COLORS
        assert _process_colors("black", plot_data=False)[0] == "black"
        assert _process_colors(["red", "blue"], plot_data=True) == ["red", "blue"]
        resolved = _process_colors({"observed": "black"}, plot_data=True)
        assert resolved == [DEFAULT_PREDICTIVE_COLORS[0], "black"]
        with pytest.raises(ValueError, match="must be a list or dict"):
            _process_colors("black", plot_data=True)
        with pytest.raises(ValueError, match="keys of the `colors` dictionary"):
            _process_colors({"pred": "red"}, plot_data=True)

    def test__legend_order(self):
        """Legend orders observed, predicted, bands (wide to narrow), draws."""
        labels = ["predictive draws", "50% interval", "94% interval", "predicted"]
        ordered = [labels[i] for i in _legend_order(labels)]
        assert ordered == [
            "predicted",
            "94% interval",
            "50% interval",
            "predictive draws",
        ]


class TestKDEDensityUnit:
    """Statistical guarantees of the per-side KDE used by kind='kde'."""

    def test_no_mass_bleeds_across_zero(self):
        """A draw with only positive samples puts zero density on the left."""
        rng = np.random.default_rng(0)
        samples = rng.lognormal(0.0, 0.3, 200)  # all positive
        grid = np.linspace(-3, 3, 101)
        out = _kde_density(samples, grid, split_at_zero=True)
        assert (out[grid < 0] == 0).all()
        assert out[grid >= 0].max() > 0

    def test_sides_weighted_by_response_proportion(self):
        """Two-sided density integrates to ~1 jointly, split by proportion."""
        rng = np.random.default_rng(1)
        pos = rng.lognormal(0.0, 0.25, 300)
        neg = -rng.lognormal(0.0, 0.25, 100)  # 25% of responses
        samples = np.concatenate([pos, neg])
        grid = np.linspace(-8, 8, 2001)
        out = _kde_density(samples, grid, split_at_zero=True)
        dx = grid[1] - grid[0]
        mass_neg = out[grid < 0].sum() * dx
        mass_pos = out[grid >= 0].sum() * dx
        assert mass_neg + mass_pos == pytest.approx(1.0, abs=0.02)
        assert mass_neg == pytest.approx(0.25, abs=0.03)

    def test_sparse_side_contributes_zero_instead_of_failing(self):
        """A single sample on one side must not crash gaussian_kde."""
        samples = np.array([-0.8, 1.0, 1.2, 1.4, 0.9, 1.1])
        grid = np.linspace(-3, 3, 51)
        out = _kde_density(samples, grid, split_at_zero=True)
        assert (out[grid < 0] == 0).all()
        assert np.isfinite(out).all()

    def test_single_sided_mode_uses_one_kde(self):
        rng = np.random.default_rng(2)
        samples = rng.lognormal(0.0, 0.3, 200)
        grid = np.linspace(0.01, 6, 501)
        out = _kde_density(samples, grid, split_at_zero=False)
        dx = grid[1] - grid[0]
        assert out.sum() * dx == pytest.approx(1.0, abs=0.05)

    def test_empty_input_returns_zeros(self):
        grid = np.linspace(-1, 1, 11)
        out = _kde_density(np.array([]), grid, split_at_zero=True)
        assert (out == 0).all()


class TestPredictiveUncertaintyUnit:
    """Fast rendering tests for the uncertainty display paths."""

    def test_default_bands_draw_polycollections_not_lines(self):
        """Graded bands add fills, keeping the two-line contract intact."""
        df = _synthetic_plotting_df()
        _, ax = plt.subplots()
        ax = _plot_predictive_1D(
            df, ax=ax, uncertainty="band", interval=[(0.25, 0.75), (0.03, 0.97)]
        )
        assert len(ax.get_lines()) == 2  # mean + observed only
        assert len(ax.collections) == 2  # one PolyCollection per band
        plt.close("all")

    def test_uncertainty_off_draws_nothing_extra(self):
        """`uncertainty=None` restores the bare mean + observed rendering."""
        df = _synthetic_plotting_df()
        _, ax = plt.subplots()
        ax = _plot_predictive_1D(df, ax=ax, uncertainty=None)
        assert len(ax.get_lines()) == 2
        assert len(ax.collections) == 0
        plt.close("all")

    def test_samples_mode_draws_per_draw_curves(self):
        """`uncertainty="samples"` draws one curve per retained draw."""
        df = _synthetic_plotting_df(n_chains=2, n_draws=5)
        _, ax = plt.subplots()
        ax = _plot_predictive_1D(df, ax=ax, uncertainty="samples")
        # 10 per-draw curves + mean + observed
        assert len(ax.get_lines()) == 2 * 5 + 2
        plt.close("all")

    def test_legend_present_on_single_axis(self):
        """The 1D path now labels its artists and shows a legend."""
        df = _synthetic_plotting_df()
        _, ax = plt.subplots()
        ax = _plot_predictive_1D(
            df, ax=ax, uncertainty="band", interval=[(0.03, 0.97)], legend=True
        )
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert labels == ["observed", "predicted", "94% interval"]
        plt.close("all")

    def test_kde_kind_draws_bands_and_no_extra_lines(self):
        """`kind="kde"` produces the same artist structure as the hist path."""
        df = _synthetic_plotting_df()
        _, ax = plt.subplots()
        ax = _plot_predictive_1D(
            df,
            ax=ax,
            kind="kde",
            uncertainty="band",
            interval=[(0.25, 0.75), (0.03, 0.97)],
        )
        assert len(ax.get_lines()) == 2
        assert len(ax.collections) == 2
        plt.close("all")

    def test_kde_samples_mode_line_count(self):
        df = _synthetic_plotting_df(n_chains=2, n_draws=4)
        _, ax = plt.subplots()
        ax = _plot_predictive_1D(df, ax=ax, kind="kde", uncertainty="samples")
        assert len(ax.get_lines()) == 2 * 4 + 2
        plt.close("all")

    def test_alpha_and_zorder_kwargs_do_not_collide(self):
        """User alpha/zorder override the internal layer values, not crash."""
        df = _synthetic_plotting_df()
        _, ax = plt.subplots()
        ax = _plot_predictive_1D(df, ax=ax, uncertainty=None, alpha=0.5, zorder=7)
        assert ax.get_lines()[0].get_alpha() == 0.5
        assert ax.get_lines()[0].get_zorder() == 7
        plt.close("all")

    def test_single_band_gets_full_base_alpha(self):
        df = _synthetic_plotting_df()
        _, ax = plt.subplots()
        ax = _plot_predictive_1D(df, ax=ax, uncertainty="band", interval=[(0.03, 0.97)])
        assert ax.collections[0].get_alpha() == pytest.approx(0.25)
        plt.close("all")

    def test_invalid_kind_and_uncertainty_raise(self):
        """Validation fires before the model is touched, so None suffices."""
        with pytest.raises(ValueError, match="`kind` must be"):
            plot_predictive(None, kind="histogram")
        with pytest.raises(ValueError, match="`uncertainty` must be"):
            plot_predictive(None, uncertainty="bands")

    def test_legend_can_be_disabled(self):
        df = _synthetic_plotting_df()
        _, ax = plt.subplots()
        ax = _plot_predictive_1D(
            df, ax=ax, uncertainty="band", interval=[(0.03, 0.97)], legend=False
        )
        assert ax.get_legend() is None
        plt.close("all")

    def test_overlong_hdi_tuple_raises_clearly(self):
        with pytest.raises(ValueError, match="exactly two floats"):
            _hdi_to_intervals((0.1, 0.5, 0.9))

    def test_label_kwarg_does_not_leak(self):
        """A user-passed label must not break band rendering or duplicate."""
        df = _synthetic_plotting_df()
        _, ax = plt.subplots()
        ax = _plot_predictive_1D(
            df, ax=ax, uncertainty="band", interval=[(0.03, 0.97)], label="mine"
        )
        assert len(ax.collections) == 1
        plt.close("all")


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

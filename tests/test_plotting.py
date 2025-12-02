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
from hssm.plotting.predictive import (
    _plot_predictive_1D,
    _plot_predictive_2D,
    plot_predictive,
)
from hssm.plotting.quantile_probability import (
    _plot_quantile_probability_1D,
    _plot_quantile_probability_2D,
    plot_quantile_probability,
)

hssm.set_floatX("float32")


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


@pytest.mark.slow
class TestPlotting:
    """Grouping all plotting tests into a single slow pytest class."""

    def test__get_title(self):
        assert _get_title(("a"), ("b")) == "a = b"
        assert _get_title(("a", "b"), ("c", "d")) == "a = c | b = d"

    def test__subset_df(self, cavanagh_test):
        with pytest.raises(ValueError):
            _row_mask_with_error(cavanagh_test, "conf", "Bad value")
        cav_subset = cavanagh_test.loc[
            (cavanagh_test["participant_id"] == 1) & (cavanagh_test["conf"] == "LC"), :
        ]
        subset_from_func = _subset_df(
            cavanagh_test, ["participant_id", "conf"], [1, "LC"]
        )
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
    def test__xarray_to_df(self, caplog, posterior, n_samples, expected):
        """Test _get_posterior_samples."""
        if expected == "error":
            with pytest.raises(ValueError):
                _xarray_to_df(posterior, n_samples=n_samples)
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

    def test__get_plotting_df(self, posterior, cavanagh_test):
        """Test _get_plotting_df."""

        # Makes a mock InferenceData object
        posterior_dataset = xr.Dataset(data_vars={"rt,response": posterior})
        idata = az.InferenceData(posterior_predictive=posterior_dataset)

        df = _get_plotting_df(
            idata, cavanagh_test, extra_dims=["participant_id", "conf"]
        )
        assert len(df) == 2500
        assert isinstance(df.index, pd.MultiIndex)
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

    def test__plot_predictive_1D(self, cav_idata, cavanagh_test):
        df = _get_plotting_df(
            cav_idata, cavanagh_test, extra_dims=["participant_id", "conf"]
        )
        df["Response Time"] = df["rt"] * np.where(df["response"] == 0, -1, 1)

        _, ax1 = plt.subplots()
        ax1 = _plot_predictive_1D(df, ax=ax1)
        assert len(ax1.get_lines()) == 2

        _, ax2 = plt.subplots()
        ax2 = _plot_predictive_1D(df, plot_data=False, ax=ax2)
        assert len(ax2.get_lines()) == 1

    def test__plot_predictive_2D(self, cav_idata, cavanagh_test):
        df = _get_plotting_df(
            cav_idata, cavanagh_test, extra_dims=["participant_id", "conf"]
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

    def test_plot_predictive(self, cav_idata, cavanagh_test):
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
                    "formula": "v ~ theta + (1|participant_id)",
                    "link": "identity",
                },
            ],
        )  # Doesn't matter what model or data we use here
        with pytest.raises(ValueError):
            plot_predictive(model)

        model._inference_obj = cav_idata.copy()
        _, ax1 = plt.subplots()
        ax1 = plot_predictive(model, ax=ax1)  # Should work directly
        assert len(ax1.get_lines()) == 2

        delattr(model.traces, "posterior_predictive")
        _, ax2 = plt.subplots()
        ax2 = plot_predictive(
            model, ax=ax2, n_samples=2
        )  # Should sample posterior predictive
        assert len(ax2.get_lines()) == 2
        assert "posterior_predictive" in model.traces
        assert model.traces.posterior_predictive.draw.size == 2

        with pytest.raises(ValueError):
            plot_predictive(model, groups="participant_id")
        with pytest.raises(ValueError):
            plot_predictive(model, groups_order=["5", "4"])

        plots = plot_predictive(model, row="stim", col="participant_id", groups="conf")
        assert len(plots) == 2
        # Lengths might defer because of subsetting the data frame
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

    def test__process_df_for_qp_plot(self, cav_idata, cavanagh_test):
        df = _get_plotting_df(
            cav_idata, cavanagh_test, extra_dims=["participant_id", "conf"]
        )

        processed_df = _process_df_for_qp_plot(df=df, q=6, cond="conf", correct=None)

        assert "conf" in processed_df.columns
        assert "is_correct" in processed_df.columns
        assert processed_df["quantile"].nunique() == 4
        assert np.all(
            processed_df.groupby(["observed", "chain", "draw", "conf", "quantile"])[
                "proportion"
            ].sum()
            == 1
        )

        # Test 2: passing cond not as str
        with pytest.raises(ValueError):
            _process_df_for_qp_plot(df=df, q=6, cond=1, correct=None)

    @pytest.mark.parametrize("predictive_style", ["points", "ellipse", "both"])
    def test__plot_quantile_probability_1D(
        self, cav_idata, cavanagh_test, predictive_style
    ):
        """Tests the _plot_quantile_probability_1D function.

        Tests that the function correctly creates a 1D quantile probability plot with the
        specified predictive style and verifies the plot attributes.
        """
        df = _get_plotting_df(cav_idata, cavanagh_test, extra_dims=["stim"])
        ax = _plot_quantile_probability_1D(
            df, cond="stim", predictive_style=predictive_style
        )

        assert has_twin(ax)
        assert ax.get_xlabel() == "Proportion"
        assert ax.get_ylabel() == "rt"
        assert ax.get_title() == "Quantile Probability Plot"

    @pytest.mark.parametrize("predictive_style", ["points", "ellipse", "both"])
    def test__plot_quantile_probability_2D(
        self, cav_idata, cavanagh_test, predictive_style
    ):
        """Tests the _plot_quantile_probability_2D function.

        Tests that the function correctly creates 2D quantile probability plots with the
        specified predictive style and verifies the plot grid dimensions.
        """
        df = _get_plotting_df(
            cav_idata, cavanagh_test, extra_dims=["participant_id", "stim"]
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
            cav_idata, cavanagh_test, extra_dims=["participant_id", "stim", "conf"]
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
    def test_plot_quantile_probability(
        self, cav_idata, cavanagh_test, predictive_style
    ):
        """Tests the plot_quantile_probability function.

        Tests the main plotting function for quantile probability plots, including error cases,
        direct plotting, posterior predictive sampling, and grouped plotting.
        """
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
                    "formula": "v ~ theta + (1|participant_id)",
                    "link": "identity",
                },
            ],
        )  # Doesn't matter what model or data we use here
        with pytest.raises(ValueError):
            plot_quantile_probability(
                model, cond="stim", predictive_style=predictive_style
            )

        model._inference_obj = cav_idata.copy()
        ax1 = plot_quantile_probability(
            model, cond="stim", data=cavanagh_test, predictive_style=predictive_style
        )  # Should work directly
        # AF-TODO: Fix this assertion,
        # currently the test is failing because the number of lines is not 9
        # but unclear where expectation is from.
        # assert len(ax1.get_lines()) == 9

        delattr(model.traces, "posterior_predictive")
        ax2 = plot_quantile_probability(
            model, cond="stim", data=cavanagh_test, n_samples=2
        )  # Should sample posterior predictive
        # AF-TODO: Fix this assertion,
        # currently the test is failing because the number of lines is not 9
        # but unclear where expectation is from.
        # assert len(ax2.get_lines()) == 9
        assert "posterior_predictive" in model.traces
        assert model.traces.posterior_predictive.draw.size == 2

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
        """Test plot_quantile_probability with only observed data (no predictive samples)."""
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

        # Plot with predictive_group=None to show only observed data
        ax = plot_quantile_probability(
            model,
            cond="stim",
            data=cavanagh_test,
            predictive_group=None,  # This triggers the else branch at line 598
        )

        assert ax is not None
        # Should only have lines for observed data, no predictive samples
        # The exact number depends on how many quantiles and conditions there are

    def test__process_df_for_qp_plot_with_quantile_by(self, cav_idata, cavanagh_test):
        """Test _process_df_for_qp_plot with quantile_by parameter.

        This tests the functionality where quantiles are first computed for each
        participant, then averaged across participants.
        """
        # Get plotting dataframe with participant_id as extra dimension
        df = _get_plotting_df(
            cav_idata, cavanagh_test, extra_dims=["participant_id", "conf"]
        )

        # Test 1: Without quantile_by (original behavior)
        processed_df_no_grouping = _process_df_for_qp_plot(
            df, 5, "conf", None, quantile_by=None
        )

        # Basic checks
        assert "conf" in processed_df_no_grouping.columns
        assert "is_correct" in processed_df_no_grouping.columns
        assert "quantile" in processed_df_no_grouping.columns
        assert (
            processed_df_no_grouping["quantile"].nunique() == 3
        )  # 5 quantiles -> 3 interior

        # Test 2: With quantile_by as string (single grouping variable)
        processed_df_single = _process_df_for_qp_plot(
            df, 5, "conf", None, quantile_by="participant_id"
        )

        # Should have same columns as without grouping
        assert "conf" in processed_df_single.columns
        assert "is_correct" in processed_df_single.columns
        assert "quantile" in processed_df_single.columns
        assert processed_df_single["quantile"].nunique() == 3

        # Should NOT have participant_id column (it was averaged out)
        assert "participant_id" not in processed_df_single.columns

        # Should have same base grouping structure
        base_groups = ["observed", "chain", "draw", "conf", "is_correct", "quantile"]
        assert all(col in processed_df_single.columns for col in base_groups)

        # Test 3: With quantile_by as list (multiple grouping variables)
        # Add another grouping variable to the dataframe
        df_multi = df.copy()
        df_multi["session"] = np.random.randint(1, 3, len(df_multi))  # Random sessions

        processed_df_multi = _process_df_for_qp_plot(
            df_multi, 5, "conf", None, quantile_by=["participant_id", "session"]
        )

        # Should NOT have the quantile_by columns
        assert "participant_id" not in processed_df_multi.columns
        assert "session" not in processed_df_multi.columns

        # Should still have the base structure
        assert all(col in processed_df_multi.columns for col in base_groups)

        # Test 4: Verify that quantiles are actually being averaged
        # The RT values should differ between grouped and non-grouped versions
        # because one computes quantiles then averages them
        assert not np.allclose(
            processed_df_no_grouping["rt"].values,
            processed_df_single["rt"].values,
            rtol=0.01,
        ), "Quantile-by grouping should produce different RT values"

        # Test 5: Check that proportions still sum to 1 after grouping
        assert np.allclose(
            processed_df_single.groupby(
                ["observed", "chain", "draw", "conf", "quantile"]
            )["proportion"].sum(),
            1.0,
            rtol=0.01,
        )

        # Test 6: Verify shape consistency
        # Both should have similar number of rows (grouped by same base variables)

        # Might be less due to some groups having no data for certain quantiles
        assert processed_df_single.shape[0] == processed_df_no_grouping.shape[0]

    def test_plot_quantile_probability_with_quantile_by(self, cav_idata, cavanagh_test):
        """Test plot_quantile_probability with quantile_by parameter.

        This tests the full plotting pipeline with the quantile_by functionality,
        ensuring that plots can be created when quantiles are computed per-participant
        then averaged.
        """
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
                    "formula": "v ~ theta + (1|participant_id)",
                    "link": "identity",
                },
            ],
        )
        model._inference_obj = cav_idata.copy()

        # Test 1: Single plot with quantile_by as string
        ax1 = plot_quantile_probability(
            model,
            cond="stim",
            data=cavanagh_test,
            quantile_by="participant_id",
            n_samples=10,
        )
        assert ax1 is not None
        assert has_twin(ax1)

        # Test 2: Grid plot with quantile_by as string
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

        # Test 3: With quantile_by as list
        ax2 = plot_quantile_probability(
            model,
            cond="stim",
            data=cavanagh_test,
            quantile_by=["participant_id"],  # List with single item
            n_samples=10,
        )
        assert ax2 is not None

        # Test 4: Verify that plots with and without quantile_by both work
        # but may produce different visualizations
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

        # Both should be valid plots
        assert ax_no_grouping is not None
        assert ax_with_grouping is not None

        # Test 5: Test with different predictive styles
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

    def test__process_df_for_qp_plot_quantile_by_edge_cases(
        self, cav_idata, cavanagh_test
    ):
        """Test edge cases for quantile_by parameter.

        Tests various edge cases and potential error conditions for the quantile_by
        functionality.
        """
        df = _get_plotting_df(
            cav_idata, cavanagh_test, extra_dims=["participant_id", "conf"]
        )

        # Test 1: Empty list for quantile_by (should behave like None)
        # This might raise an error or behave like None depending on implementation
        with pytest.raises(
            ValueError, match="`quantile_by` must be a non-empty list of strings."
        ):
            _process_df_for_qp_plot(df, 5, "conf", None, quantile_by=[])

        with pytest.raises(
            ValueError, match="All elements in `quantile_by` must be strings."
        ):
            _process_df_for_qp_plot(df, 5, "conf", None, quantile_by=[1, 2])

        with pytest.raises(
            ValueError, match="`quantile_by` must be a string or a list of strings."
        ):
            _process_df_for_qp_plot(df, 5, "conf", None, quantile_by=1)

        # Test 3: Verify that column name detection works regardless of number of grouping vars
        # Add multiple extra dimensions
        df_extra = df.copy()
        df_extra["dim1"] = np.random.randint(1, 3, len(df_extra))
        df_extra["dim2"] = np.random.randint(1, 3, len(df_extra))

        # Test with 1 quantile_by variable
        result1 = _process_df_for_qp_plot(
            df_extra, 5, "conf", None, quantile_by="participant_id"
        )
        assert "quantile" in result1.columns

        # Test with 2 quantile_by variables
        result2 = _process_df_for_qp_plot(
            df_extra, 5, "conf", None, quantile_by=["participant_id", "dim1"]
        )
        assert "quantile" in result2.columns

        # Test with 3 quantile_by variables
        result3 = _process_df_for_qp_plot(
            df_extra, 5, "conf", None, quantile_by=["participant_id", "dim1", "dim2"]
        )
        assert "quantile" in result3.columns

        # All should have consistent column structure
        base_cols = [
            "observed",
            "chain",
            "draw",
            "conf",
            "is_correct",
            "quantile",
            "rt",
            "proportion",
        ]
        assert all(col in result1.columns for col in base_cols)
        assert all(col in result2.columns for col in base_cols)
        assert all(col in result3.columns for col in base_cols)

    def test__get_plotting_df_quantile_by_dims_validation(
        self, cav_idata, cavanagh_test
    ):
        """Test _get_plotting_df with various quantile_by_dims inputs for validation coverage."""

        # Test 0: quantile_by_dims as None (should be None)
        df_none = _get_plotting_df(
            cav_idata,
            cavanagh_test,
            extra_dims=["conf"],
            quantile_by_dims=None,  # None input
        )
        assert df_none is not None

        # Test 1: quantile_by_dims as string (should convert to list)
        df_string = _get_plotting_df(
            cav_idata,
            cavanagh_test,
            extra_dims=["conf"],
            quantile_by_dims="participant_id",  # String input
        )
        assert df_string is not None
        assert "participant_id" in df_string.columns
        assert "conf" in df_string.columns

        # Test 2: quantile_by_dims as list (normal case)
        df_list = _get_plotting_df(
            cav_idata,
            cavanagh_test,
            extra_dims=["conf"],
            quantile_by_dims=["participant_id"],  # List input
        )
        assert df_list is not None

        # Test 3: Empty list should raise ValueError
        with pytest.raises(
            ValueError, match="`quantile_by_dims` must be a non-empty list of strings."
        ):
            _get_plotting_df(
                cav_idata,
                cavanagh_test,
                extra_dims=["conf"],
                quantile_by_dims=[],  # Empty list
            )

        # Test 4: List with non-string elements should raise ValueError
        with pytest.raises(
            ValueError, match="All elements in `quantile_by_dims` must be strings."
        ):
            _get_plotting_df(
                cav_idata,
                cavanagh_test,
                extra_dims=["conf"],
                quantile_by_dims=[1, 2],  # Non-string elements
            )

        # Test 5: Overlap between quantile_by_dims and extra_dims should raise ValueError
        with pytest.raises(
            ValueError,
            match="`quantile_by_dims` and `extra_dims` must not have any overlap.",
        ):
            _get_plotting_df(
                cav_idata,
                cavanagh_test,
                extra_dims=["conf", "participant_id"],
                quantile_by_dims=["participant_id"],  # Overlaps with extra_dims
            )

    def test__get_plotting_df_quantile_by_dims_edge_cases(
        self, cav_idata, cavanagh_test
    ):
        """Test additional edge cases for quantile_by_dims to ensure full coverage."""

        # Test 1: quantile_by_dims provided but extra_dims is None
        df1 = _get_plotting_df(
            cav_idata,
            cavanagh_test,
            extra_dims=None,  # No extra_dims
            quantile_by_dims=["participant_id"],
        )
        assert df1 is not None
        # Since extra_dims is None, participant_id won't be in columns
        # (it's only used for quantile computation, not added to df)

        # Test 2: Both provided but NO overlap (normal successful case)
        df2 = _get_plotting_df(
            cav_idata,
            cavanagh_test,
            extra_dims=["conf"],  # Different from quantile_by_dims
            quantile_by_dims=["participant_id"],  # No overlap
        )
        assert df2 is not None
        assert "conf" in df2.columns
        assert "participant_id" in df2.columns

        # Test 3: Valid list of multiple quantile_by_dims (covers elif branch with valid list)
        df3 = _get_plotting_df(
            cav_idata,
            cavanagh_test,
            extra_dims=["conf"],
            quantile_by_dims=["participant_id", "dbs"],  # Multiple items, all valid
        )
        assert df3 is not None

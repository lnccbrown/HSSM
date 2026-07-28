"""Tests for plotting utility helpers."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import hssm
from hssm.plotting.utils import (
    _get_plotting_df,
    _get_title,
    _process_df_for_qp_plot,
    _row_mask_with_error,
    _subset_df,
    _xarray_to_df,
)

hssm.set_floatX("float32")
TEST_RNG_SEED = 0


class TestPlottingUtilsUnit:
    """Unit tests for utilities in hssm.plotting.utils."""

    def test__get_title(self):
        """Check grouped title formatting."""
        assert _get_title(("conf",), ("LC",)) == "conf = LC"
        assert _get_title(("a", "b"), ("c", "d")) == "a = c | b = d"

    def test__subset_df(self, cavanagh_test):
        """Check dataframe subsetting and invalid group values."""
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
        """Test conversion from posterior xarray to plotting dataframe."""
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
        posterior_dataset = xr.Dataset(data_vars={"rt,response": posterior})
        idata = xr.DataTree.from_dict({"posterior_predictive": posterior_dataset})

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

        with pytest.raises(ValueError, match="Either dt or data must be provided"):
            _get_plotting_df(dt=None, data=None)

    def test__process_df_for_qp_plot(self, cav_dt, cavanagh_test):
        """Check quantile-probability dataframe preparation and errors."""
        df = _get_plotting_df(
            cav_dt, cavanagh_test, extra_dims=["participant_id", "conf"]
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

        with pytest.raises(ValueError):
            _process_df_for_qp_plot(df=df, q=6, cond=1, correct=None)

        iterable_quantiles = _process_df_for_qp_plot(
            df=df,
            q=[0.25, 0.5, 0.75],
            cond="conf",
            correct=None,
        )
        np.testing.assert_allclose(
            sorted(iterable_quantiles["quantile"].unique()),
            [0.25, 0.5, 0.75],
        )

    def test__process_df_for_qp_plot_with_quantile_by(self, cav_dt, cavanagh_test):
        """Test _process_df_for_qp_plot with quantile_by parameter."""
        df = _get_plotting_df(
            cav_dt, cavanagh_test, extra_dims=["participant_id", "conf"]
        )

        processed_df_no_grouping = _process_df_for_qp_plot(
            df, 5, "conf", None, quantile_by=None
        )

        assert "conf" in processed_df_no_grouping.columns
        assert "is_correct" in processed_df_no_grouping.columns
        assert "quantile" in processed_df_no_grouping.columns
        assert processed_df_no_grouping["quantile"].nunique() == 3

        processed_df_single = _process_df_for_qp_plot(
            df, 5, "conf", None, quantile_by="participant_id"
        )

        assert "conf" in processed_df_single.columns
        assert "is_correct" in processed_df_single.columns
        assert "quantile" in processed_df_single.columns
        assert processed_df_single["quantile"].nunique() == 3
        assert "participant_id" not in processed_df_single.columns

        base_groups = ["observed", "chain", "draw", "conf", "is_correct", "quantile"]
        assert all(col in processed_df_single.columns for col in base_groups)

        df_multi = df.copy()
        rng = np.random.default_rng(TEST_RNG_SEED)
        df_multi["session"] = rng.integers(1, 3, size=len(df_multi))

        processed_df_multi = _process_df_for_qp_plot(
            df_multi, 5, "conf", None, quantile_by=["participant_id", "session"]
        )

        assert "participant_id" not in processed_df_multi.columns
        assert "session" not in processed_df_multi.columns
        assert all(col in processed_df_multi.columns for col in base_groups)

        assert not np.allclose(
            processed_df_no_grouping["rt"].values,
            processed_df_single["rt"].values,
            rtol=0.01,
        ), "Quantile-by grouping should produce different RT values"

        assert np.allclose(
            processed_df_single.groupby(
                ["observed", "chain", "draw", "conf", "quantile"]
            )["proportion"].sum(),
            1.0,
            rtol=0.01,
        )

        assert processed_df_single.shape[0] == processed_df_no_grouping.shape[0]

    def test__process_df_for_qp_plot_quantile_by_edge_cases(
        self, cav_dt, cavanagh_test
    ):
        """Test edge cases for quantile_by parameter."""
        df = _get_plotting_df(
            cav_dt, cavanagh_test, extra_dims=["participant_id", "conf"]
        )

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

        df_extra = df.copy()
        rng = np.random.default_rng(TEST_RNG_SEED)
        df_extra["dim1"] = rng.integers(1, 3, size=len(df_extra))
        df_extra["dim2"] = rng.integers(1, 3, size=len(df_extra))

        result1 = _process_df_for_qp_plot(
            df_extra, 5, "conf", None, quantile_by="participant_id"
        )
        assert "quantile" in result1.columns

        result2 = _process_df_for_qp_plot(
            df_extra, 5, "conf", None, quantile_by=["participant_id", "dim1"]
        )
        assert "quantile" in result2.columns

        result3 = _process_df_for_qp_plot(
            df_extra,
            5,
            "conf",
            None,
            quantile_by=["participant_id", "dim1", "dim2"],
        )
        assert "quantile" in result3.columns

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

    def test__get_plotting_df_quantile_by_dims_validation(self, cav_dt, cavanagh_test):
        """Test _get_plotting_df with various quantile_by_dims inputs."""
        df_none = _get_plotting_df(
            cav_dt,
            cavanagh_test,
            extra_dims=["conf"],
            quantile_by_dims=None,
        )
        assert df_none is not None

        df_string = _get_plotting_df(
            cav_dt,
            cavanagh_test,
            extra_dims=["conf"],
            quantile_by_dims="participant_id",
        )
        assert df_string is not None
        assert "participant_id" in df_string.columns
        assert "conf" in df_string.columns

        df_list = _get_plotting_df(
            cav_dt,
            cavanagh_test,
            extra_dims=["conf"],
            quantile_by_dims=["participant_id"],
        )
        assert df_list is not None

        with pytest.raises(
            ValueError, match="`quantile_by_dims` must be a non-empty list of strings."
        ):
            _get_plotting_df(
                cav_dt,
                cavanagh_test,
                extra_dims=["conf"],
                quantile_by_dims=[],
            )

        with pytest.raises(
            ValueError, match="All elements in `quantile_by_dims` must be strings."
        ):
            _get_plotting_df(
                cav_dt,
                cavanagh_test,
                extra_dims=["conf"],
                quantile_by_dims=[1, 2],
            )

        with pytest.raises(
            ValueError,
            match="`quantile_by_dims` and `extra_dims` must not have any overlap.",
        ):
            _get_plotting_df(
                cav_dt,
                cavanagh_test,
                extra_dims=["conf", "participant_id"],
                quantile_by_dims=["participant_id"],
            )

    def test__get_plotting_df_quantile_by_dims_edge_cases(self, cav_dt, cavanagh_test):
        """Test edge cases for quantile_by_dims."""
        df1 = _get_plotting_df(
            cav_dt,
            cavanagh_test,
            extra_dims=None,
            quantile_by_dims=["participant_id"],
        )
        assert df1 is not None

        df2 = _get_plotting_df(
            cav_dt,
            cavanagh_test,
            extra_dims=["conf"],
            quantile_by_dims=["participant_id"],
        )
        assert df2 is not None
        assert "conf" in df2.columns
        assert "participant_id" in df2.columns

        df3 = _get_plotting_df(
            cav_dt,
            cavanagh_test,
            extra_dims=["conf"],
            quantile_by_dims=["participant_id", "dbs"],
        )
        assert df3 is not None

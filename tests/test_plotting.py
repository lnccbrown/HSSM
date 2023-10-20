"""Test plotting module."""

import pytest

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from hssm.plotting.utils import _get_plotting_df, _xarray_to_df


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
        assert not isinstance(df.index, pd.MultiIndex)
        assert df.index.name == "rt,response_obs"
        assert df.index[0] == 0
        assert df.index[-1] == 499
        assert np.all(df.index.value_counts() == expected // 500)
        assert df.columns[0] == "rt"


def test__get_plotting_df(posterior, cavanagh_test):
    """Test _get_plotting_df."""

    # Makes a mock InferenceData object
    posterior_dataset = xr.Dataset(data_vars={"rt,response": posterior})
    idata = az.InferenceData(posterior_predictive=posterior_dataset)

    df = _get_plotting_df(idata, cavanagh_test, extra_dims=["participant_id", "conf"])
    assert len(df) == 2500
    assert not isinstance(df.index, pd.MultiIndex)
    assert df.columns.to_list() == ["rt", "response", "participant_id", "conf"]
    assert df.isna().sum().sum() == 0
    np.testing.assert_array_equal(
        df.iloc[2000:, :].values,
        cavanagh_test.loc[:, ["rt", "response", "participant_id", "conf"]].values,
    )

"""Plotting utilities for HSSM."""

from typing import Any, Iterable

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from ..utils import _random_sample


def _xarray_to_df(
    posterior: xr.DataArray, n_samples: int | float | None
) -> pd.DataFrame:
    """Extract samples from posterior and converts it to a DataFrame.

    We make the following assumptions:
    1. The inference data always has a posterior predictive group with a `rt,response`
       variable.
    2. This variable always has four dimensions: `chain`, `draw`, `rt,response_obs`,
        and `rt,response_dim`.

    Parameters
    ----------
    posterior
        An InferenceData object.
    n_samples
        When an integer >= 1, the number of samples to be extracted from the draw
        dimension. When a float between 0 and 1, the proportion of samples to be
        extracted from the draw dimension. When None, all samples are extracted.

    Returns
    -------
    pd.DataFrame
        A dataframe with the posterior samples.
    """
    sampled_posterior = _random_sample(posterior, n_samples=n_samples)

    # Convert the posterior samples to a dataframe
    stacked = (
        sampled_posterior.stack({"obs": ["chain", "draw", "rt,response_obs"]})
        .transpose()
        .to_pandas()
        .sort_index(axis=0, level=["chain", "draw", "rt,response_obs"])
        .droplevel(["chain", "draw"], axis=0)
    )

    # Rename the columns
    stacked.columns = ["rt", "response"]

    return stacked


def _get_plotting_df(
    idata: az.InferenceData,
    data: pd.DataFrame | None = None,
    extra_dims: list[str] | None = None,
    n_samples: int | float | None = 20,
) -> pd.DataFrame:
    """Prepare a dataframe for plotting.

    Parameters
    ----------
    idata
        An InferenceData object.
    data: optional
        A dataframe with the original data. If not provided, the function will only
        return the posterior samples without appending the observed data.
    extra_dims, optional
        Extra dimensions to be added to the dataframe from `idata`, by default None
    n_samples, optional
        When an interger >= 1, the number of samples to be extracted from the draw
        dimension. When a float between 0 and 1, the proportion of samples to be
        extracted from the draw dimension. When None, all samples are extracted.

    Returns
    -------
    pd.DataFrame
        A dataframe with the original data and the extra dimensions.
    """
    idata_posterior = idata["posterior_predictive"]["rt,response"]

    # get the posterior samples
    posterior = _xarray_to_df(idata_posterior, n_samples=n_samples)

    if data is None:
        if extra_dims:
            raise ValueError(
                "You supplied additional dimensions to plot, but no data was provided."
                + " HSSM requires a dataset to determine the values of the covariates"
                + " to plot these additional dimensions."
            )
        posterior.insert(0, "observed", "predicted")
        return posterior

    if extra_dims and idata_posterior["rt,response_obs"].size != data.shape[0]:
        raise ValueError(
            "The number of observations in the data and the number of posterior "
            + "samples are not equal."
        )

    # reset the index of the data to ensure proper merging
    extra_dims = [] if extra_dims is None else extra_dims
    data = data.reset_index(drop=True).loc[:, ["rt", "response"] + extra_dims]

    # merge the posterior samples with the data
    if extra_dims:
        posterior = posterior.merge(
            data.loc[:, extra_dims], left_index=True, right_index=True, how="left"
        )

    # concatenate the posterior samples with the data
    plotting_df = (
        pd.concat(
            [posterior, data],
            keys=["predicted", "observed"],
            names=["observed", "obs_n"],
        )
        .droplevel("obs_n")
        .reset_index("observed")
    )

    return plotting_df


def _row_mask_with_error(df: pd.DataFrame, col: str, val: Any) -> pd.DataFrame:
    """Subset a dataframe based on the value of a column.

    Parameters
    ----------
    df
        A dataframe.
    col
        A column to be subset.
    val
        A value to be subset.

    Returns
    -------
    pd.Series
        Row masks for df[col] == val.
    """
    row_mask = df[col] == val
    if not row_mask.any():
        raise ValueError(f"No rows found where {col} = {val}.")

    return row_mask


def _subset_df(
    df: pd.DataFrame, cols: Iterable["str"], col_values: Iterable[Any]
) -> pd.DataFrame:
    """Subset a dataframe based on the values of columns.

    For example, when `cols = ["A", "B"]`, and `col_values = [1, 2]`, the function
    returns `df[(df["A"] == 1) & (df["B"] == 2)]`.

    Parameters
    ----------
    df
        A dataframe.
    cols
        An Iterable of columns to be subset.
    col_values
        An Iterable of values to be subset.

    Returns
    -------
    pd.DataFrame
        A subset dataframe.
    """
    row_mask = np.column_stack(
        [
            _row_mask_with_error(df, col, col_value)
            for col, col_value in zip(cols, col_values)
        ]
    ).all(axis=1)

    return df.loc[row_mask, :]


def _get_title(cols: Iterable["str"], col_values: Iterable[Any]) -> str:
    """Generate a title for the plot.

    Parameters
    ----------
    cols
        An Iterable of columns in the subset.
    col_values
        An Iterable of values in the subset.

    Returns
    -------
    str
        A title for the plot.
    """
    title = " | ".join(
        [f"{col} = {col_value}" for col, col_value in zip(cols, col_values)]
    )

    return title

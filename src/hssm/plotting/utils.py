"""Plotting utilities for HSSM."""

import logging
from typing import Any, Iterable, cast

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from ..utils import _random_sample

_logger = logging.getLogger("hssm")


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
        .rename_axis(index={"rt,response_obs": "obs_n"})
        .sort_index(axis=0, level=["chain", "draw", "obs_n"])
    )

    # Rename the columns
    stacked.columns = ["rt", "response"]

    return stacked


def _process_data(data: pd.DataFrame, extra_dims: list[str]) -> pd.DataFrame:
    """Extract the relevant columns from the data and apply the right index.

    Parameters
    ----------
    data
        A dataframe with the original data.
    extra_dims
        Extra dimensions to be extracted from the dataframe.

    Returns
    -------
    pd.DataFrame
        The processed dataframe.
    """
    # reset the index of the data to ensure proper merging
    data = data.reset_index(drop=True).loc[:, ["rt", "response"] + extra_dims]

    # Makes sure that data has similar index to posterior
    data.index = pd.MultiIndex.from_product(
        [[-1], [-1], data.index], names=["chain", "draw", "obs_n"]
    )

    return data


def _get_plotting_df(
    idata: az.InferenceData | None = None,
    data: pd.DataFrame | None = None,
    extra_dims: list[str] | None = None,
    n_samples: int | float | None = 20,
) -> pd.DataFrame:
    """Prepare a dataframe for plotting.

    Parameters
    ----------
    idata : optional
        An InferenceData object. If not provided, the function will only return the
        processed original data.
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
    if idata is None and data is None:
        raise ValueError("Either idata or data must be provided.")

    extra_dims = [] if extra_dims is None else extra_dims

    if idata is None:
        data = _process_data(data, extra_dims)

        data.insert(0, "observed", "observed")
        return data

    # get the posterior samples
    idata_posterior = idata["posterior_predictive"]["rt,response"]
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

    data = _process_data(data, extra_dims)

    # merge the posterior samples with the data
    if extra_dims:
        posterior_merged = posterior.merge(
            data.loc[:, extra_dims],
            left_on="obs_n",
            right_on="obs_n",
            how="left",
            sort=True,
        )
        posterior_merged.index = posterior.sort_index().index
        posterior = posterior_merged

    # concatenate the posterior samples with the data
    plotting_df = pd.concat(
        {"predicted": posterior, "observed": data},
        names=["observed", "chain", "draw", "obs_n"],
    ).reset_index("observed")

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


def _process_df_for_qp_plot(
    df: pd.DataFrame, q: int | Iterable[float], cond: str, correct: str | None = None
) -> pd.DataFrame:
    """Process the data frame fo the quantile probability plot.

    Parameters
    ----------
    df
        The DataFrame to process.
    q
        If an `int`, quantiles will be determined using np.linspace(0, 1, q) (0 and 1
        will be excluded. If an iterable, will generate quantiles according to this
        iterable.
    cond
        The variable for the conditions.
    correct
        The column for whether the answer is correct.

    Returns
    -------
    pd.DataFrame
        The dataframe with `quantile`, `rt`, and `cond` variables.
    """
    if isinstance(q, Iterable):
        if any(q_elem < 0 or q_elem > 1 for q_elem in q):
            raise ValueError("All elements in `q` must be between 0 and 1.")

    if isinstance(q, int):
        if q >= 10:
            _logger.warning(
                "The number of quantiles (%d) is high. Generally 4-5 quantiles are"
                + " ideal for visualizing the data.",
                q,
            )
        q = np.linspace(0, 1, q)[1:-1]

    # flip the rts
    df.loc[:, "rt"] = np.where(df["response"] > 0, df["rt"], -df["rt"])

    df = df.copy().reset_index()

    df["is_correct"] = df["response"] > 0 if correct is None else df[correct]

    quantiles = (
        df.groupby(["observed", "chain", "draw", cond, "is_correct"])["rt"]
        .quantile(q=q)
        .reset_index()
        .rename(columns={"level_5": "quantile"})
    )

    pcts = (
        df.groupby(["observed", "chain", "draw", cond])["is_correct"]
        .value_counts(normalize=True)
        .reset_index()
    )

    quantiles = quantiles.merge(
        pcts, on=["observed", "chain", "draw", cond, "is_correct"], how="left"
    )

    return quantiles


def _check_groups_and_groups_order(
    groups: str | Iterable[str] | None,
    groups_order: Iterable[str] | dict[str, Iterable[str]] | None,
    row: str | None,
    col: str | None,
) -> tuple[Iterable[str], dict[str, Iterable[str]]]:
    """Check the validity of `groups` and `groups_order`."""
    if groups is None:
        if groups_order is not None:
            raise ValueError("`group_order` is only valid when `group` is provided.")
    else:
        if row is None or col is None:
            raise ValueError(
                "When `group` is provided, both `row` and `col` must be provided."
            )
        if groups_order is not None:
            if isinstance(groups_order, Iterable) and not isinstance(
                groups_order, dict
            ):
                if not isinstance(groups, str):
                    raise ValueError(
                        "`groups_order` can be a List-like only when `groups` is a str."
                    )
                groups_order = {groups: groups_order}
            elif isinstance(groups_order, dict):
                if not set(groups_order.keys()).issubset(set(groups)):
                    raise ValueError(
                        "`groups_order` can only contain keys that are in `groups`."
                    )
        else:
            groups_order = {}
        if isinstance(groups, str):
            groups = [groups]

    # Cast to the right types to satisfy mypy
    groups = cast(Iterable, groups)
    groups_order = cast(dict, groups_order)

    return groups, groups_order


def _use_traces_or_sample(
    model,
    data: pd.DataFrame | None,
    idata: az.InferenceData | None,
    n_samples: int | float | None,
) -> tuple[az.InferenceData, bool]:
    """Check if idata is provided, otherwise use traces.

    Also, if posterior predictive samples are not contained in traces, sample from
    the model.
    """
    # First, determine whether posterior predictive samples are available
    # If not, we need to sample from the posterior
    if idata is None:
        if model.traces is None:
            raise ValueError(
                "No InferenceData object provided. Please provide an InferenceData "
                + "object or sample the model first using model.sample()."
            )
        idata = model.traces

    sampled = False

    if "posterior_predictive" not in idata:
        _logger.info(
            "No posterior predictive samples found. Generating posterior predictive "
            + "samples using the provided InferenceData object and the original data. "
            + "This will modify the provided InferenceData object, or if not provided, "
            + "the traces object stored inside the model."
        )
        model.sample_posterior_predictive(
            idata=idata,
            data=data,
            inplace=True,
            n_samples=n_samples,
        )
        idata = model.traces
        sampled = True

    return cast(az.InferenceData, idata), sampled


def _check_sample_size(plotting_df):
    """Check if the sample size is valid."""
    sample_size = (
        plotting_df.loc[plotting_df["observed"] == "predicted", :]
        .groupby(["chain", "draw"])
        .size()
        .mean()
    )
    if sample_size < 50:
        _logger.warning(
            "The number of posterior predictive samples is less than 50. "
            + "The uncertainty interval may not be accurate."
        )

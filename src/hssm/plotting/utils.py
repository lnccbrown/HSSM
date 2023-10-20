"""Plotting utilities for HSSM."""

import logging

import arviz as az
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
    idata
        An InferenceData object.
    n_samples
        When an interger >= 1, the number of samples to be extracted from the draw
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
    observed: pd.DataFrame,
    extra_dims: list[str] | None = None,
    n_samples: int | float | None = 20,
) -> pd.DataFrame:
    """Prepare a dataframe for plotting.

    Parameters
    ----------
    idata
        An InferenceData object.
    data
        A dataframe with the original data.
    extra_dims, optional
        Extra dimensions to be added to the dataframe from `idata`, by default None

    Returns
    -------
    pd.DataFrame
        A dataframe with the original data and the extra dimensions.
    """
    idata_posterior = idata["posterior_predictive"]["rt,response"]
    if extra_dims and idata_posterior["rt,response_obs"].size != observed.shape[0]:
        raise ValueError(
            "The number of observations in the data and the number of posterior "
            + "samples are not equal."
        )

    # reset the index of the data to ensure proper merging
    extra_dims = [] if extra_dims is None else extra_dims
    observed = observed.reset_index(drop=True).loc[:, ["rt", "response"] + extra_dims]

    # get the posterior samples
    posterior = _xarray_to_df(idata_posterior, n_samples=n_samples)

    # merge the posterior samples with the data
    if extra_dims:
        posterior = posterior.merge(
            observed.loc[:, extra_dims], left_index=True, right_index=True, how="left"
        )

    # concatenate the posterior samples with the data
    plotting_df = pd.concat(
        [posterior, observed],
        keys=["predicted", "observed"],
        names=["observed", "obs_n"],
    ).droplevel("obs_n")

    return plotting_df

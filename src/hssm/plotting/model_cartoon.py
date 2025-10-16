"""Plotting functionalities for HSSM."""

import logging
from copy import deepcopy
from itertools import product
from typing import Any, Iterable, Literal, Protocol, cast

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from ssms.basic_simulators.simulator import simulator

# Original model cartoon plot from gui
from ..defaults import SupportedModels, default_model_config
from .predictive import _process_lines
from .utils import (
    _check_groups_and_groups_order,
    _get_plotting_df,
    _get_title,
    # _hdi_to_interval, # TODO: We eventually want to add HDI plotting back in here
    _subset_df,
    _to_idata_group,
    _use_traces_or_sample,
)

_logger = logging.getLogger("hssm")

TRAJ_COLOR_DEFAULT_DICT = {
    -1: "black",
    0: "black",
    1: "green",
    2: "blue",
    3: "red",
    4: "orange",
    5: "purple",
    6: "brown",
}


class PlotFunctionProtocol(Protocol):
    """Protocol for plot functions."""

    def __call__(self, *args, **kwargs) -> Axes:
        """Plot function."""
        ...


def _plot_model_cartoon_1D(
    data: pd.DataFrame,
    model_name: str,
    plot_data: bool = True,
    plot_mean: bool = True,
    plot_samples: bool = False,
    predictive_group: Literal[
        "posterior_predictive", "prior_predictive"
    ] = "posterior_predictive",
    colors: str | list[str] | None = None,
    title: str | None = "Model Plots",
    xlabel: str | None = "Response Time",
    ylabel: str | None = "",
    **kwargs,
) -> mpl.axes.Axes:
    """Plot the posterior predictive distribution against the observed data.

    Check the `plot_model_cartoon` function below for docstring.

    Returns
    -------
    mpl.Axes
        A matplotlib Axes object containing the plot.
    """
    if not (plot_mean or plot_samples):
        raise ValueError("At least one of plot_mean or plot_samples must be True")

    if "color" in kwargs:
        del kwargs["color"]
    colors = colors or ["#ec205b", "#338fb8"]

    if plot_data and isinstance(colors, str):
        raise ValueError("When `plot_data=True`, `colors` must be a list or dict.")

    if "ax" in kwargs:
        ax = kwargs.pop("ax")
    else:
        ax = plt.gca()

    config_tmp = default_model_config[cast("SupportedModels", model_name)]
    model_params = config_tmp["list_params"]

    n_choices = len(config_tmp["choices"])

    is_predictive_mean = data.source == predictive_group + "_mean"
    is_predictive_samples = data.source == predictive_group
    is_observed = data.observed == "observed"
    is_predicted = data.observed == "predicted"

    data_predictive_mean = data.loc[
        is_predictive_mean & is_predicted,
        :,
    ]

    data_observed = None
    if plot_mean:
        data_observed = data.loc[is_predictive_mean & is_observed, :]
    elif plot_samples and (not plot_mean):
        data_observed = data.loc[is_predictive_samples & is_observed, :]

    if plot_data and data_observed is None:
        raise ValueError("No data to plot. Please set plot_data=False or provide data.")

    data_predictive = data.loc[is_predictive_samples & is_predicted, :]

    plot_function: PlotFunctionProtocol | None = None
    if n_choices == 2:
        plot_function = plot_func_model
    elif n_choices > 2:
        plot_function = plot_func_model_n

    if plot_function is None:
        raise NotImplementedError(
            "The model plot works only for >=2 choice models at the moment"
        )

    ax = plot_function(
        model_name=model_name,
        axis=ax,
        theta_mean=(
            data_predictive_mean.reset_index()[model_params] if plot_mean else None
        ),
        theta_samples=(data_predictive[model_params] if plot_samples else None),
        data=(
            None
            if not plot_data or data_observed is None
            else data_observed.reset_index()
        ),
        **kwargs,
    )

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    return ax


def _plot_model_cartoon_2D(
    data: pd.DataFrame,
    model_name: str,
    plot_data: bool = True,
    plot_mean: bool = True,
    plot_samples: bool = False,
    predictive_group: Literal[
        "posterior_predictive", "prior_predictive"
    ] = "posterior_predictive",
    row: str | None = None,
    col: str | None = None,
    col_wrap: int | None = None,
    bins: int | np.ndarray | str | None = 100,
    step: bool = False,
    # interval: tuple[float, float] | None = None,
    colors: str | list[str] | None = None,
    linestyles: str | list[str] = "-",
    linewidths: float | list[float] = 1.25,
    title: str | None = "Model Cartoon",
    xlabel: str | None = "Response Time",
    ylabel: str | None = "",
    grid_kwargs: dict | None = None,
    **kwargs,
) -> sns.FacetGrid:
    """Plot the posterior predictive distribution against the observed data.

    Check the function below for docstring.

    Returns
    -------
    sns.FacetGrid
        A seaborn FacetGrid object containing the plot.
    """
    g = sns.FacetGrid(
        data=data,
        col=col,
        row=row,
        col_wrap=col_wrap,
        legend_out=True,
        **(grid_kwargs or {}),
    )

    g.map_dataframe(
        _plot_model_cartoon_1D,
        model_name=model_name,
        plot_data=plot_data,
        plot_mean=plot_mean,
        plot_samples=plot_samples,
        predictive_group=predictive_group,
        bins=bins,
        step=step,
        # interval=interval,
        colors=colors,
        linestyles=linestyles,
        linewidths=linewidths,
        title=None,
        xlabel=xlabel,
        ylabel=ylabel,
        **kwargs,
    )

    if plot_data:
        custom_lines = [
            Line2D([0], [0], color="blue", linestyle="-", lw=1.5),
            Line2D([0], [0], color="black", linestyle="-", lw=1.5),
        ]

        custom_labels = ["observed", "mean_predictive"]

        g.add_legend(
            dict(zip(custom_labels, custom_lines)),
            title="",
            label_order=["observed", "mean_predictive"],
        )

    if title:
        g.figure.subplots_adjust(top=0.9)
        g.figure.suptitle(title)

    g.set_xlabels(xlabel)
    g.set_ylabels(ylabel)

    return g


def compute_merge_necessary_deterministics(
    model, idata, idata_group: Literal["posterior", "prior"] = "posterior"
):
    """Compute the necessary deterministic variables for the model."""
    # Get the list of deterministic variables
    necessary_params = default_model_config[model.model_name]["list_params"]
    deterministics_list = []
    idata_group_keys = list(idata[idata_group].keys())
    # Compute the deterministic variables
    for param in necessary_params:
        if param not in idata_group_keys:
            if param in [
                deterministic.name for deterministic in model.pymc_model.deterministics
            ]:
                deterministics_list.append(
                    pm.compute_deterministics(
                        idata[idata_group], model=model.pymc_model, var_names=[param]
                    )
                )

    deterministics_idata = xr.merge(deterministics_list)
    setattr(idata, idata_group, xr.merge([idata[idata_group], deterministics_idata]))
    return idata


def attach_trialwise_params_to_df(
    model, df, idata, idata_group: Literal["posterior", "prior"] = "posterior"
):
    """Attach the trial-wise parameters to the dataframe."""
    necessary_params = default_model_config[model.model_name]["list_params"]
    df[necessary_params] = 0.0

    for chain_tmp, draw_tmp in {(x[0], x[1]) for x in list(df.index) if x[0] != -1}:
        for param in necessary_params:
            df.loc[(chain_tmp, draw_tmp, slice(None)), param] = (
                idata[idata_group].sel(chain=chain_tmp, draw=draw_tmp)[param].values
            )
    return df


def _make_idata_mean_posterior(idata: az.InferenceData) -> az.InferenceData:
    """Create a new InferenceData object containing only the posterior mean.

    Takes an InferenceData object and computes the mean across chains and draws,
    then restructures it to have a single chain and draw. Removes posterior
    predictive samples if present.

    Parameters
    ----------
    idata : arviz.InferenceData
        The InferenceData object to process

    Returns
    -------
    arviz.InferenceData
        A new InferenceData object containing only the posterior mean
    """
    setattr(idata, "posterior", idata["posterior"].mean(dim=["chain", "draw"]))
    setattr(idata, "posterior", idata["posterior"].assign_coords(chain=[0], draw=[0]))
    for data_var in list(idata["posterior"].data_vars):
        idata["posterior"][data_var] = idata["posterior"][data_var].expand_dims(
            dim=["chain", "draw"], axis=[0, 1]
        )

    if "posterior_predictive" in idata:
        del idata["posterior_predictive"]
    return idata


def _make_idata_mean_prior(idata: az.InferenceData) -> az.InferenceData:
    """Create a new InferenceData object containing only the prior mean.

    Takes an InferenceData object and computes the mean across chains and draws,
    then restructures it to have a single chain and draw. Removes prior
    predictive samples if present.

    Parameters
    ----------
    idata : arviz.InferenceData
        The InferenceData object to process

    Returns
    -------
    arviz.InferenceData
        A new InferenceData object containing only the posterior mean
    """
    setattr(idata, "prior", idata["prior"].mean(dim=["chain", "draw"]))
    setattr(idata, "prior", idata["prior"].assign_coords(chain=[0], draw=[0]))
    for data_var in list(idata["prior"].data_vars):
        idata["prior"][data_var] = idata["prior"][data_var].expand_dims(
            dim=["chain", "draw"], axis=[0, 1]
        )

    if "prior_predictive" in idata:
        del idata["prior_predictive"]
    return idata


def plot_model_cartoon(
    model,
    idata: az.InferenceData | None = None,
    data: pd.DataFrame | None = None,
    predictive_group: Literal[
        "posterior_predictive", "prior_predictive"
    ] = "posterior_predictive",
    plot_data: bool = True,
    n_samples: int | float | None = 20,
    n_samples_prior: int = 500,
    row: str | None = None,
    col: str | None = None,
    col_wrap: int | None = None,
    groups: str | Iterable[str] | None = None,
    groups_order: Iterable[str] | dict[str, Iterable[str]] | None = None,
    bins: int | np.ndarray | str | None = 50,
    step: bool = False,
    plot_predictive_mean: bool = True,
    plot_predictive_samples: bool = False,
    colors: str | list[str] | None = None,
    linestyles: str | list[str] | tuple[str] | dict[str, str] = "-",
    linewidths: float | list[float] | tuple[float] | dict[str, float] = 1.25,
    title: str | None = "Posterior Predictive Distribution",
    xlabel: str | None = "Response Time",
    ylabel: str | None = "",
    grid_kwargs: dict | None = None,
    **kwargs,
) -> Axes | sns.FacetGrid | list[sns.FacetGrid]:
    """Plot the posterior predictive distribution against the observed data.

    Parameters
    ----------
    model : hssm.HSSM
        A fitted HSSM model.
    idata : optional
        The InferenceData object with posterior samples. If not provided, will use the
        traces object stored inside the model. If posterior predictive samples are not
        present in this object, will generate posterior predictive samples using the
        this InferenceData object and the original data.
    data : optional
        The observed data.

        - If `data` is provided and the idata object does not contain a
        `"posterior_predictive"` group, will generate posterior predictive samples using
        covariate provided in this object. If the group does exist, it is assumed that
        the posterior predictive samples are generated with the covariates provided in
        this DataFrame.
        - If `data` is not provided (i.e., `data=None`), the behavior depends on whether
        "plot_data" is true or not. If `plot_data=True`, the plotting function will use
        the data stored in the `model` object and proceed as the case above. If
        `plot_data=False`, if posterior predictive samples are not present in the
        `idata` object, the plotting function will generate posterior predictive samples
        using the data stored in the `model` object. If posterior predictive samples
        exist in the `idata` object, these samples will be used for plotting, but a
        ValueError will be thrown if any of `col` or `row` is not None.
    predictive_group : optional
        The type of predictive distribution to plot, by default "posterior_predictive".
        Can be "posterior_predictive" or "prior_predictive".
    plot_data : optional
        Whether to plot the observed data, by default True.
    n_samples : optional
        When idata is provided, the number or proportion of predictive samples
        randomly drawn to be used from each chain for plotting. When idata is not
        provided, the number or proportion of posterior/prior
        samples to be used to generate predictive samples.
        The number or proportion are defined as follows:

        - When an integer >= 1, the number of samples to be extracted from the draw
          dimension.
        - When a float between 0 and 1, the proportion of samples to be extracted from
          the draw dimension.
        - When None, all samples are extracted.
    n_samples_prior : int
        When predictive_group is "prior_predictive", the number or proportion of prior
        samples to be used to generate predictive samples. The number or proportion are
        defined as follows:
        - When an integer >= 1, the number of samples to be drawn from the prior and
          respectively from the prior predictive.
    row : optional
        Variables that define subsets of the data, which will be drawn on the row
        dimension of the facets in the grid. When both row and col are None, one single
        plot will be produced, by default None.
    col : optional
        Variables that define subsets of the data, which will be drawn on the column
        dimension of the facets in the grid. When both row and col are None, one single
        plot will be produced, by default None.
    col_wrap : optional
        “Wrap” the column variable at this width, so that the column facets span
        multiple rows. Incompatible with a row facet., by default None.
    groups : optional
        Additional dimensions along which to plot different groups. This is useful when
        there are 3 or more dimensions of covariates to plot against, by default None.
    groups_order : optional
        The order to plot the groups, by default None, in which case the order is the
        order in which the groups appear in the data. Only when `groups` is a string,
        this can be an iterable of strings. Otherwise, this is a dictionary mapping the
        dimension name to the order of the groups in that dimension.
    bins : optional
        Specification of hist bins, by default 100. There are three options:
        - A string describing the binning strategy (passed to `np.histogram_bin_edges`).
        - A list-like defining the bin edges.
        - An integer defining the number of bins to be used.
    step : optional
        Whether to plot the distributions as a step function or a smooth density plot,
        by default False.
    colors : optional
        Colors to use for the different levels of the hue variable. When a `str`, the
        color of posterior predictives, in which case an error will be thrown if
        `plot_data` is `True`. When a length-2 iterable, indicates the colors in the
        order of posterior predictives and observed data. The values must be
        interpretable by matplotlib. When None, use default color palette, by default
        None.
    linestyles : optional
        Linestyles to use for the different levels of the hue variable. When a `str`,
        the linestyle of both distributions. When a length-2 iterable, indicates the
        linestyles in the order of posterior predictives and observed data. The values
        must be interpretable by matplotlib. When None, use solid lines, by default "-".
        When dictionary, the keys must be 'predicted' and/or 'observed', and the values
        must be interpretable by matplotlib.
    linewidths : optional
        Linewidths to use for the different levels of the hue variable. When a `float`,
        the linewidth of both distributions. When a length-2 iterable, indicates the
        linewidths in the order of posterior predictives and observed data, by default
        1.25.
    title : optional
        The title of the plot, by default "Posterior Predictive Distribution". Ignored
        when `groups` is provided.
    xlabel : optional
        The label for the x-axis, by default "Response Time".
    ylabel : optional
        The label for the y-axis, by default "Density".
    grid_kwargs : optional
        Additional keyword arguments are passed to the [`sns.FacetGrid` constructor]
        (https://seaborn.pydata.org/generated/seaborn.FacetGrid.html#seaborn.FacetGrid.__init__)
        when any of row or col is provided. When producing a single plot, these
        arguments are ignored.
    kwargs : optional
        Additional keyword arguments passed to ax.plot() functions.

    Returns
    -------
    mpl.axes.Axes | sns.FacetGrid
        The matplotlib `axis` or seaborn `FacetGrid` object containing the plot.
    """
    if not (plot_predictive_mean or plot_predictive_samples):
        raise ValueError(
            "At least one of plot_predictive_mean or "
            "plot_predictive_samples must be True"
        )

    # Process linestyles
    linestyles_ = _process_lines(linestyles, mode="linestyles")
    # Process linewidths
    linewidths_ = _process_lines(linewidths, mode="linewidths")

    groups, groups_order = _check_groups_and_groups_order(
        groups, groups_order, row, col
    )

    extra_dims = [e for e in [row, col] if e is not None] or None
    if extra_dims is not None and groups is not None:
        extra_dims += list(groups)

    if data is None:
        if (
            (not extra_dims)
            and (not plot_data)
            and (idata is not None)
            and (predictive_group in idata)
        ):
            # Allows data to be None only when plot_data=False and no extra_dims
            # and posterior predictive samples are available
            data = None
        else:
            data = model.data

    # Mean version of plot
    plotting_df_mean = None
    if plot_predictive_mean:
        if predictive_group == "posterior_predictive":
            idata_mean = _make_idata_mean_posterior(
                deepcopy(model.traces if idata is None else idata)
            )
            idata_mean, _ = _use_traces_or_sample(
                model,
                data,
                idata_mean,
                n_samples=None,
                predictive_group="posterior_predictive",
            )
        else:
            # Need to sample from prior predictive here to get samples from the prior
            idata, _ = _use_traces_or_sample(
                model,
                data,
                idata,
                n_samples=n_samples_prior,
                predictive_group=predictive_group,
            )

            idata_mean = _make_idata_mean_prior(deepcopy(idata))
            # AF-COMMENT: This is a hack to get the prior predictive mean
            # we should find a better way to do this eventually.
            idata_mean_tmp = model.predict(
                **dict(
                    idata=az.InferenceData(posterior=idata_mean["prior"]),
                    kind="response",
                    inplace=False,
                )
            )

            if hasattr(idata_mean, "prior_predictive"):
                del idata_mean["prior_predictive"]

            idata_mean.add_groups(prior_predictive=idata_mean_tmp.posterior_predictive)

        # # Get the plotting dataframe by chain and sample
        plotting_df_mean = _get_plotting_df(
            idata_mean,
            data,
            extra_dims=extra_dims,
            n_samples=None,
            response_str=model.response_str,
            predictive_group=predictive_group,
        )

        # Get plotting dataframe for predictive mean
        # df by chain and sample
        idata_mean = compute_merge_necessary_deterministics(
            model, idata_mean, idata_group=_to_idata_group(predictive_group)
        )

        plotting_df_mean = attach_trialwise_params_to_df(
            model,
            plotting_df_mean,
            idata_mean,
            idata_group=_to_idata_group(predictive_group),
        )

        # Flip the rt values if necessary
        if np.any(plotting_df_mean["response"] == 0) and model.n_choices == 2:
            plotting_df_mean["response"] = np.where(
                plotting_df_mean["response"] == 0, -1, 1
            )
        if model.n_choices == 2:
            plotting_df_mean["rt"] = (
                plotting_df_mean["rt"] * plotting_df_mean["response"]
            )

        plotting_df_mean["source"] = predictive_group + "_mean"

    if plot_predictive_samples:
        idata, sampled = _use_traces_or_sample(
            model, data, idata, n_samples=n_samples, predictive_group=predictive_group
        )

        # Get the plotting dataframe by chain and sample
        plotting_df = _get_plotting_df(
            idata,
            data,
            extra_dims=extra_dims,
            n_samples=None if sampled else n_samples,
            response_str=model.response_str,
            predictive_group=predictive_group,
        )

        # Get plotting dataframe for posterior mean
        # df by chain and sample
        idata = compute_merge_necessary_deterministics(
            model,
            idata,
            idata_group=_to_idata_group(predictive_group),
        )

        plotting_df = attach_trialwise_params_to_df(
            model,
            plotting_df,
            idata,
            idata_group=_to_idata_group(predictive_group),
        )

        # Flip the rt values if necessary
        if np.any(plotting_df["response"] == 0) and model.n_choices == 2:
            plotting_df["response"] = np.where(plotting_df["response"] == 0, -1, 1)
        if model.n_choices == 2:
            plotting_df["rt"] = plotting_df["rt"] * plotting_df["response"]

        plotting_df["source"] = predictive_group
    else:
        plotting_df = None

    if (plotting_df is not None) and (plotting_df_mean is not None):
        plotting_df = pd.concat([plotting_df, plotting_df_mean])
    elif plotting_df_mean is not None:
        plotting_df = plotting_df_mean

    # return plotting_df

    # Then, plot the posterior predictive distribution against the observed data
    # Determine whether we are producing a single plot or a grid of plots

    if not extra_dims:
        ax = _plot_model_cartoon_1D(
            data=plotting_df,
            model_name=model.model_name,
            plot_data=plot_data,
            plot_mean=plot_predictive_mean,
            plot_samples=plot_predictive_samples,
            predictive_group=predictive_group,
            bins=bins,
            step=step,
            colors=colors,
            linestyles=linestyles_,
            linewidths=linewidths_,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            **kwargs,
        )

        custom_lines = [
            Line2D([0], [0], color="blue", linestyle="-", lw=1.5),
            Line2D([0], [0], color="black", linestyle="-", lw=1.5),
        ]

        custom_labels = ["observed", "mean_predictive"]
        ax.legend(custom_lines, custom_labels, title="", loc="upper right")
        return ax

    # The multiple dimensions case
    # If group is not provided, we are producing a grid of plots
    if groups is None:
        g = _plot_model_cartoon_2D(
            data=plotting_df,
            model_name=model.model_name,
            plot_data=plot_data,
            plot_mean=plot_predictive_mean,
            plot_samples=plot_predictive_samples,
            predictive_group=predictive_group,
            row=row,
            col=col,
            col_wrap=col_wrap,
            bins=bins,
            step=step,
            colors=colors,
            linestyles=linestyles_,
            linewidths=linewidths_,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            grid_kwargs=grid_kwargs,
            **kwargs,
        )
        return g

    # The group dimension case
    plots = []
    if not isinstance(groups_order, dict):
        raise ValueError("groups_order must be a dictionary")

    orders = product(
        *[groups_order.get(g, plotting_df[g].unique().tolist()) for g in groups]
    )

    for order in orders:
        df = _subset_df(plotting_df, groups, order)
        title = _get_title(groups, order)
        if len(df) == 0:
            _logger.warning(
                "No posterior predictive samples found for the group %s."
                + "Skipping this group.",
                title,
            )
            continue
        g = _plot_model_cartoon_2D(
            data=df,
            model_name=model.model_name,
            plot_data=plot_data,
            plot_mean=plot_predictive_mean,
            plot_samples=plot_predictive_samples,
            predictive_group=predictive_group,
            row=row,
            col=col,
            col_wrap=col_wrap,
            bins=bins,
            step=step,
            colors=colors,
            linestyles=linestyles_,
            linewidths=linewidths_,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            grid_kwargs=grid_kwargs,
            **kwargs,
        )

        plots.append(g)

    return plots


# Original model cartoon plot from gui
def plot_func_model(
    model_name: str,
    axis: Axes,
    theta_mean: pd.DataFrame | None = None,
    theta_samples: pd.DataFrame | None = None,
    data: pd.DataFrame | None = None,
    n_samples=10,
    bin_size: float = 0.05,
    n_trajectories: int = 0,
    delta_t_model: float = 0.01,
    random_state: int | None = None,
    keep_slope: bool = True,
    keep_boundary: bool = True,
    keep_ndt: bool = True,
    keep_starting_point: bool = True,
    markersize_starting_point: float | int = 50,
    markertype_starting_point: str = ">",
    markershift_starting_point: float | int = 0,
    linewidth_histogram: float | int = 1.5,
    linewidth_model: float | int = 1.5,
    color_data: str = "blue",
    color_predictive_mean: str = "black",
    color_predictive: str = "black",
    alpha_mean: float = 1,
    alpha_predictive: float = 0.05,
    alpha_trajectories: float = 0.5,
    **kwargs,
) -> Axes:
    """Plot model cartoon with posterior predictive samples.

    Parameters
    ----------
    model_name : str
        Name of the model to plot.
    axis : matplotlib.axes.Axes
        Axis to plot into.
    theta_mean : pd.DataFrame, optional
        DataFrame containing posterior mean parameter values.
    theta_samples : pd.DataFrame, optional
        DataFrame containing posterior samples of parameters.
    data : pd.DataFrame, optional
        DataFrame containing observed data to overlay.
    n_samples : int, optional
        Number of posterior samples to use. Defaults to 10.
    bin_size : float, optional
        Size of bins used for histograms. Defaults to 0.05.
    n_trajectories : int, optional
        Number of trajectories to plot. Defaults to 0.
    delta_t_model : float, optional
        Time step for model simulation. Defaults to 0.01.
    random_state : int, optional
        Random seed for reproducibility.
    keep_slope : bool, optional
        Whether to plot drift slopes. Defaults to True.
    keep_boundary : bool, optional
        Whether to plot decision boundaries. Defaults to True.
    keep_ndt : bool, optional
        Whether to plot non-decision time. Defaults to True.
    keep_starting_point : bool, optional
        Whether to plot starting point. Defaults to True.
    markersize_starting_point : float or int, optional
        Size of starting point marker. Defaults to 50.
    markertype_starting_point : int, optional
        Type of starting point marker. Defaults to 0.
    markershift_starting_point : float or int, optional
        Shift of starting point marker. Defaults to 0.
    linewidth_histogram : float or int, optional
        Width of histogram lines. Defaults to 0.5.
    linewidth_model : float or int, optional
        Width of model lines. Defaults to 0.5.
    color_data : str, optional
        Color for data histogram. Defaults to "blue".
    color_predictive_mean : str, optional
        Color for posterior mean. Defaults to "black".
    color_predictive : str, optional
        Color for posterior samples. Defaults to "black".
    alpha_mean : float, optional
        Transparency of posterior mean. Defaults to 1.
    alpha_predictive : float, optional
        Transparency of posterior samples. Defaults to 0.05.
    alpha_trajectories : float, optional
        Transparency of trajectories. Defaults to 0.5.
    **kwargs
        Additional arguments passed to plotting functions.

    Returns
    -------
    matplotlib.axes.Axes
        The axis with the model cartoon plot.
    """
    ylim_low, ylim_high = kwargs.get("ylims", (-3, 3))
    xlim_low, xlim_high = kwargs.get("xlims", (-0.05, 5))

    # Extract some parameters from kwargs
    bins = list(np.arange(xlim_low, xlim_high, bin_size))

    # RUN SIMULATIONS
    # -------------------------------

    # Simulator Data from posterior mean
    if random_state is not None:
        np.random.seed(random_state)

    rand_int = np.random.randint(0, 400000000)

    sim_out = None
    if theta_mean is not None:
        sim_out = simulator(
            model=model_name,
            theta=theta_mean.values,
            n_samples=n_samples,
            no_noise=False,
            delta_t=delta_t_model,
            random_state=rand_int,
        )

        # Simulate model without noise: posterior mean
        # (this allows to extract the time-dynamics of the drift e.g.)
        sim_out_no_noise = simulator(
            model=model_name,
            theta=theta_mean.loc[np.random.choice(theta_mean.shape[0], 1), :].values,
            n_samples=1,
            no_noise=True,
            delta_t=delta_t_model,
            smooth_unif=False,
        )

    # Simulate model without noise: posterior samples
    if theta_samples is not None:
        posterior_pred_no_noise = {}
        posterior_pred_sims = {}
        for i, (chain, draw) in enumerate(
            list(theta_samples.index.droplevel("obs_n").unique())
        ):
            posterior_pred_no_noise[i] = simulator(
                model=model_name,
                theta=theta_samples.loc[(chain, draw), :].values,
                n_samples=1,
                no_noise=True,
                delta_t=delta_t_model,
                smooth_unif=False,
            )

            # Simulate model: posterior samples
            posterior_pred_sims[i] = simulator(
                model=model_name,
                theta=theta_samples.loc[(chain, draw), :].values,
                n_samples=n_samples,
                no_noise=False,
                delta_t=delta_t_model,
                random_state=rand_int + i,
            )

    # Simulate trajectories
    sim_out_traj = {}
    for i in range(n_trajectories):
        if theta_mean is not None:
            tmp_theta = theta_mean.loc[
                (np.random.choice(theta_mean.shape[0], 1)), :
            ].values
        elif theta_samples is not None:
            # wrap in max statement here
            # because negative value are possible,
            # however refer to data instead of posterior samples
            random_index = tuple(
                [
                    np.random.choice(theta_samples.index.get_level_values(name_))
                    for name_ in ("chain", "draw", "obs_n")
                ]
            )
            tmp_theta = theta_samples.loc[random_index, :].values
        else:
            raise ValueError("No theta values provided but n_trajectories is > 0")

        sim_out_traj[i] = simulator(
            model=model_name,
            theta=tmp_theta,
            n_samples=1,
            no_noise=False,
            delta_t=delta_t_model,
            random_state=rand_int + i,
            smooth_unif=False,
        )

    # ADD DATA HISTOGRAMS
    hist_bottom_high, hist_bottom_low = calculate_histogram_bounds(
        theta_mean,
        theta_samples,
        sim_out if (theta_mean is not None) else None,
        posterior_pred_no_noise if (theta_samples is not None) else None,
        **kwargs,
    )

    hist_histtype = kwargs.get("hist_histtype", "step")
    axis.set_xlim(xlim_low, xlim_high)
    axis.set_ylim(ylim_low, ylim_high)
    axis_twin_up: Axes = cast("Axes", axis.twinx())
    axis_twin_down: Axes = cast("Axes", axis.twinx())
    axis_twin_up.set_ylim(ylim_low, ylim_high)
    axis_twin_up.set_yticks([])
    axis_twin_down.set_ylim(ylim_high, ylim_low)
    axis_twin_down.set_yticks([])
    axis_twin_down.set_axis_off()
    axis_twin_up.set_axis_off()

    # This ensures zorder across axis elements is correct
    # for the sequence in which they are invoked.
    axis.set_zorder(1)
    axis.set_facecolor("none")
    axis_twin_up.set_zorder(0)
    axis_twin_down.set_zorder(0)

    if theta_mean is not None:
        if sim_out is None:
            raise ValueError("No sim_out provided but theta_mean is not None")
        data_up = np.abs(
            sim_out["rts"][(sim_out["rts"] != -999) & (sim_out["choices"] == 1)]
        )
        data_down = np.abs(
            sim_out["rts"][(sim_out["rts"] != -999) & (sim_out["choices"] != 1)]
        )

        add_histograms_to_twin_axes(
            data_up=data_up,
            data_down=data_down,
            hist_bottom_high=hist_bottom_high,
            hist_bottom_low=hist_bottom_low,
            color_data=color_predictive_mean,
            linewidth_histogram=linewidth_histogram,
            bins=bins,
            alpha=alpha_mean,
            axis_twin_up=axis_twin_up,
            axis_twin_down=axis_twin_down,
            hist_histtype=hist_histtype,
            bin_size=bin_size,
            zorder=-1,
        )

    if theta_samples is not None:
        # Add histograms for posterior samples:
        for k, sim_out_tmp in posterior_pred_sims.items():
            data_up = np.abs(
                sim_out_tmp["rts"][
                    (sim_out_tmp["rts"] != -999) & (sim_out_tmp["choices"] == 1)
                ]
            )
            data_down = np.abs(
                sim_out_tmp["rts"][
                    (sim_out_tmp["rts"] != -999) & (sim_out_tmp["choices"] != 1)
                ]
            )

            add_histograms_to_twin_axes(
                data_up=data_up,
                data_down=data_down,
                hist_bottom_high=hist_bottom_high,
                hist_bottom_low=hist_bottom_low,
                color_data=color_predictive,
                linewidth_histogram=linewidth_histogram,
                bins=bins,
                alpha=alpha_predictive,
                axis_twin_up=axis_twin_up,
                axis_twin_down=axis_twin_down,
                hist_histtype=hist_histtype,
                bin_size=bin_size,
                zorder=-k - 1,
            )

    # Add histograms for real data
    if data is not None:
        data_up = data.query(f"rt != {-999} and response == {1}")["rt"].values
        data_down = data.query(f"rt != {-999} and response != {1}")["rt"].values
        add_histograms_to_twin_axes(
            data_up=data_up,
            data_down=data_down,
            hist_bottom_high=hist_bottom_high,
            hist_bottom_low=hist_bottom_low,
            color_data=color_data,
            linewidth_histogram=linewidth_histogram,
            bins=bins,
            alpha=1,
            axis_twin_up=axis_twin_up,
            axis_twin_down=axis_twin_down,
            hist_histtype=hist_histtype,
            bin_size=bin_size,
        )

    z_cnt = 0  # controlling the order of elements in plot

    if theta_samples is not None:
        # ADD MODEL CARTOONS:
        t_s = np.arange(
            0, posterior_pred_no_noise[0]["metadata"]["max_t"], delta_t_model
        )

        # Model cartoon for posterior samples
        for j, sim_out_tmp in posterior_pred_no_noise.items():
            _add_model_cartoon_to_ax(
                sample=sim_out_tmp,
                axis=axis,
                keep_slope=keep_slope,
                keep_boundary=keep_boundary,
                keep_ndt=keep_ndt,
                keep_starting_point=keep_starting_point,
                markersize_starting_point=markersize_starting_point,
                markertype_starting_point=markertype_starting_point,
                markershift_starting_point=markershift_starting_point,
                delta_t_graph=delta_t_model,
                alpha=alpha_predictive,
                lw_m=linewidth_model,
                ylim_low=ylim_low,
                ylim_high=ylim_high,
                t_s=t_s,
                color=color_predictive,
                zorder_cnt=z_cnt,
            )

            z_cnt += 1

    if theta_mean is not None:
        t_s = np.arange(0, sim_out_no_noise["metadata"]["max_t"], delta_t_model)
        # Model cartoon for posterior mean
        _add_model_cartoon_to_ax(
            sample=sim_out_no_noise,
            axis=axis,
            keep_slope=keep_slope,
            keep_boundary=keep_boundary,
            keep_ndt=keep_ndt,
            keep_starting_point=keep_starting_point,
            markersize_starting_point=markersize_starting_point,
            markertype_starting_point=markertype_starting_point,
            markershift_starting_point=markershift_starting_point,
            delta_t_graph=delta_t_model,
            alpha=alpha_mean,
            lw_m=linewidth_model,
            ylim_low=ylim_low,
            ylim_high=ylim_high,
            t_s=t_s,
            color=color_predictive,
            zorder_cnt=z_cnt + 1,
        )

    # Add in trajectories
    if n_trajectories > 0:
        _add_trajectories(
            axis=axis,
            sample=sim_out_traj,
            t_s=t_s,
            delta_t_graph=delta_t_model,
            n=n_trajectories,
            alpha=alpha_trajectories,
            **kwargs,
        )
    return axis


def calculate_histogram_bounds(
    theta_mean: pd.DataFrame | None,
    theta_samples: pd.DataFrame | None,
    sim_out: dict[str, Any] | None,
    posterior_pred_no_noise: dict[int, dict[str, Any]] | None,
    **kwargs: Any,
) -> tuple[float, float]:
    """Calculate the bounds for histograms in model cartoon plots.

    This function determines appropriate histogram bounds based on either
    the posterior mean or posterior samples. If neither
    is provided, it returns default values.

    Parameters
    ----------
    theta_mean : pd.DataFrame | None
        DataFrame containing posterior mean parameters. If None, will try to use
        theta_samples instead.
    theta_samples : pd.DataFrame | None
        DataFrame containing posterior samples. Only used if theta_mean is None.
    sim_out : dict[str, Any] | None
        Dictionary containing simulation output for posterior mean.
    posterior_pred_no_noise : dict[int, dict[str, Any]] | None
        Dictionary containing simulation output for posterior samples without noise.
    **kwargs : Any
        Additional keyword arguments. Can include:
        - hist_bottom_high: float - override for upper histogram bound
        - hist_bottom_low: float - override for lower histogram bound

    Returns
    -------
    tuple[float, float]
        Upper and lower bounds for the histograms (hist_bottom_high, hist_bottom_low).
    """
    theta_mean_is_none = theta_mean is None
    theta_samples_is_none = theta_samples is None
    b_high = 3
    b_low = 3
    if all([theta_mean_is_none, theta_samples_is_none]):
        _logger.warning(
            'No "theta_mean" provided. Using default values for histogram'
            " location. Likely highly suboptimal choice!"
        )
        return b_high, b_low

    if not theta_mean_is_none:
        if sim_out is None:
            raise ValueError("No sim_out provided but theta_mean is not None")
        else:
            boundary_data = sim_out["metadata"]["boundary"]
            b_high = np.maximum(boundary_data, 0)[0]
            b_low = np.minimum(-boundary_data, 0)[0]
    elif not theta_samples_is_none:
        if posterior_pred_no_noise is None:
            raise ValueError(
                "No posterior_pred_no_noise provided but theta_samples is not None"
            )
        else:
            all_boundaries = [
                posterior_pred_no_noise[key_]["metadata"]["boundary"]
                for key_ in posterior_pred_no_noise
            ]
            b_high = np.max([np.maximum(boundary, 0)[0] for boundary in all_boundaries])
            b_low = np.min([np.minimum(-boundary, 0)[0] for boundary in all_boundaries])
    hist_bottom_high = kwargs.get("hist_bottom_high", b_high)
    hist_bottom_low = kwargs.get("hist_bottom_low", -b_low)
    return hist_bottom_high, hist_bottom_low


# AF-TODO: Add documentation for this function
def _add_trajectories(
    axis: Axes,
    sample: dict[int, Any],
    t_s: np.ndarray,
    delta_t_graph: float = 0.01,
    n: int = 10,
    highlight_rt_choice: bool = True,
    markersize_rt_choice: float | int = 50,
    markertype_rt_choice: str = "*",
    markercolor_rt_choice: str | list[str] | dict[str, str] = "red",
    linewidth: float | int = 1,
    alpha: float | int = 0.5,
    colors: str | list[str] | dict[str, str] = "black",
    **kwargs,
):
    """Add simulated decision trajectories to a given matplotlib axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The axis to add the trajectories to.
    sample : dict
        Dictionary containing simulation data including trajectories and metadata.
    t_s : numpy.ndarray
        Array of timepoints for plotting.
    delta_t_graph : float, optional
        Time step size for plotting, by default 0.01.
    n : int, optional
        Number of trajectories to plot, by default 10.
    highlight_trajectory_rt_choice : bool, optional
        Whether to highlight the response time and choice with a marker, by default
        True.
    markersize_trajectory_rt_choice : float or int, optional
        Size of marker for response time/choice, by default 50.
    markertype_trajectory_rt_choice : str, optional
        Marker style for response time/choice, by default "*".
    markercolor_trajectory_rt_choice : str, int, list or dict, optional
        Color(s) for response time/choice markers. Can be a single color,
        list of colors, or dict mapping choices to colors. By default "red".
    linewidth : float or int, optional
        Line width for trajectories, by default 1.
    alpha : float or int, optional
        Opacity of trajectories, by default 0.5.
    colors : str, list or dict, optional
        Color(s) for trajectories. Can be a single color, list of colors,
        or dict mapping choices to colors. By default "black".
    **kwargs
        Additional keyword arguments passed to plotting functions.
    """
    # Check markercolor type
    if isinstance(markercolor_rt_choice, str):
        markercolor_rt_choice_dict = {
            value_: markercolor_rt_choice
            for value_ in sample[0]["metadata"]["possible_choices"]
        }
    elif isinstance(markercolor_rt_choice, list):
        markercolor_rt_choice_dict = {
            value_: markercolor_rt_choice[cnt]
            for cnt, value_ in enumerate(sample[0]["metadata"]["possible_choices"])
        }
    elif isinstance(markercolor_rt_choice, dict):
        markercolor_rt_choice_dict = markercolor_rt_choice
    else:
        raise ValueError(
            "The `markercolor_trajectory_rt_choice`"
            " argument must be a string, list, or dict."
        )

    # Check trajectory color type
    if isinstance(colors, str):
        colors_dict = {}
        for value_ in sample[0]["metadata"]["possible_choices"]:
            colors_dict[value_] = colors
    elif isinstance(colors, list):
        cnt = 0
        for value_ in sample[0]["metadata"]["possible_choices"]:
            colors_dict[value_] = colors[cnt]
            cnt += 1
    elif isinstance(colors, dict):
        colors_dict = colors
    else:
        raise ValueError(
            "The `color_trajectories` argument must be a string, list, or dict."
        )

    # Make bounds
    (b_high, b_low) = (
        np.maximum(sample[0]["metadata"]["boundary"], 0),
        np.minimum((-1) * sample[0]["metadata"]["boundary"], 0),
    )

    b_h_init = b_high[0]
    b_l_init = b_low[0]
    n_roll = int((sample[0]["metadata"]["t"][0] / delta_t_graph) + 1)
    b_high = np.roll(b_high, n_roll)
    b_high[:n_roll] = b_h_init
    b_low = np.roll(b_low, n_roll)
    b_low[:n_roll] = b_l_init

    # Trajectories
    for i in range(n):
        metadata = sample[i]["metadata"]
        tmp_traj = metadata["trajectory"]
        tmp_traj_choice = float(sample[i]["choices"].flatten())
        maxid = np.minimum(np.argmax(np.where(tmp_traj > -999)), t_s.shape[0])

        # Identify boundary value at timepoint of crossing
        b_tmp = b_high[maxid + n_roll] if tmp_traj_choice > 0 else b_low[maxid + n_roll]

        axis.plot(
            t_s[:maxid] + metadata["t"][0],  # sample.t.values[0],
            tmp_traj[:maxid],
            color=colors_dict[tmp_traj_choice],
            alpha=alpha,
            linewidth=linewidth,
            zorder=2000 + i,
        )
        if highlight_rt_choice:
            axis.scatter(
                t_s[maxid] + metadata["t"][0],
                b_tmp,
                markersize_rt_choice,
                color=markercolor_rt_choice_dict[tmp_traj_choice],
                alpha=1,
                marker=markertype_rt_choice,
                zorder=2000 + i,
            )


def add_histograms_to_twin_axes(
    data_up: np.ndarray,
    data_down: np.ndarray,
    hist_bottom_high: float,
    hist_bottom_low: float,
    color_data: str,
    linewidth_histogram: float,
    bins: list[float],
    alpha: float,
    axis_twin_up: Axes,
    axis_twin_down: Axes,
    hist_histtype: Literal["bar", "barstacked", "step", "stepfilled"],
    bin_size: float,
    zorder: int = -1,
):
    """Add histograms to upper and lower twin axes.

    Args:
        data_up: Array of data points for upper histogram.
        data_down: Array of data points for lower histogram.
        hist_bottom_high: Bottom position for upper histogram.
        hist_bottom_low: Bottom position for lower histogram.
        color_data: Color to use for histogram bars/lines.
        linewidth_histogram: Width of histogram lines.
        bins: List of bin edges for histograms.
        alpha: Transparency value for histograms.
        axis_twin_up: Upper twin axis to plot on.
        axis_twin_down: Lower twin axis to plot on.
        hist_histtype: Type of histogram ('bar', 'barstacked', 'step', or 'stepfilled').
        bin_size: Size of histogram bins.
        zorder: Z-order for plot elements. Defaults to -1.
    """
    # Compute weights
    weights_up_data = np.tile(
        (1 / bin_size) / (data_up.shape[0] + data_down.shape[0]),
        reps=data_up.shape[0],
    )
    weights_down_data = np.tile(
        (1 / bin_size) / (data_up.shape[0] + data_down.shape[0]),
        reps=data_down.shape[0],
    )

    # Add histograms for simulated data
    axis_twin_up.hist(
        np.abs(data_up),
        bins=bins,
        weights=weights_up_data,
        histtype=hist_histtype,
        bottom=hist_bottom_high,
        alpha=alpha,
        color=color_data,
        edgecolor=color_data,
        linewidth=linewidth_histogram,
        zorder=zorder,
    )

    axis_twin_down.hist(
        np.abs(data_down),
        bins=bins,
        weights=weights_down_data,
        histtype=hist_histtype,
        bottom=hist_bottom_low,
        alpha=alpha,
        color=color_data,
        edgecolor=color_data,
        linewidth=linewidth_histogram,
        zorder=zorder,
    )


def _add_model_cartoon_to_ax(
    sample: dict,
    axis: Axes,
    t_s: np.ndarray,
    ylim_low: float,
    ylim_high: float,
    keep_slope: bool = True,
    keep_boundary: bool = True,
    keep_ndt: bool = True,
    keep_starting_point: bool = True,
    markersize_starting_point: float | int = 80,
    markertype_starting_point: str = ">",
    markershift_starting_point: float | int = -0.05,
    delta_t_graph: float | None = None,
    alpha: float | None = None,
    lw_m: float | None = None,
    tmp_label: str | None = None,
    zorder_cnt: int = 1,
    color: str = "black",
):
    """Add a model cartoon visualization to a matplotlib axis.

    Parameters
    ----------
    sample : dict
        Dictionary containing model metadata including boundary,
        trajectory, time points, etc.
    axis : Axes
        Matplotlib axis to plot on.
    keep_slope : bool, default=True
        Whether to plot the trajectory slope.
    keep_boundary : bool, default=True
        Whether to plot decision boundaries.
    keep_ndt : bool, default=True
        Whether to plot non-decision time marker.
    keep_starting_point : bool, default=True
        Whether to plot starting point marker.
    markersize_starting_point : float or int, default=80
        Size of starting point marker.
    markertype_starting_point : float or int, default=1
        Marker type for starting point.
    markershift_starting_point : float or int, default=-0.05
        Horizontal shift of starting point marker.
    delta_t_graph : float, optional
        Time step size for plotting.
    alpha : float, optional
        Opacity of plot elements.
    lw_m : float, optional
        Line width for plot elements.
    tmp_label : str, optional
        Label for plot legend.
    ylim_low : float or int, optional
        Lower y-axis limit.
    ylim_high : float or int, optional
        Upper y-axis limit.
    t_s : numpy.ndarray, optional
        Time points array.
    zorder_cnt : int, default=1
        Base z-order for plot elements.
    color : str, default="black"
        Color for plot elements.
    """
    # Make bounds
    (b_high, b_low) = (
        np.maximum(sample["metadata"]["boundary"], 0),
        np.minimum((-1) * sample["metadata"]["boundary"], 0),
    )

    # Set initial boundary value
    b_h_init = b_high[0]
    b_l_init = b_low[0]

    # Push boundary forward to accomodate non-decision time
    # Initial boundary value applied from t = 0 to t = t_ndt
    n_roll = int((sample["metadata"]["t"][0] / delta_t_graph) + 1)
    b_high = np.roll(b_high, n_roll)
    b_high[:n_roll] = b_h_init
    b_low = np.roll(b_low, n_roll)
    b_low[:n_roll] = b_l_init

    tmp_traj = sample["metadata"]["trajectory"]
    maxid = np.minimum(np.argmax(np.where(tmp_traj > -999)), t_s.shape[0])

    if keep_boundary:
        # Upper bound
        axis.plot(
            t_s,
            b_high[: t_s.shape[0]],
            color=color,
            alpha=alpha,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
            label=tmp_label,
        )

        # Lower bound
        axis.plot(
            t_s,
            b_low[: t_s.shape[0]],
            color=color,
            alpha=alpha,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
        )

    # Slope
    if keep_slope:
        axis.plot(
            t_s[:maxid] + sample["metadata"]["t"][0],
            tmp_traj[:maxid],
            color=color,
            alpha=alpha,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
        )  # TOOK AWAY LABEL

    # Non-decision time
    if keep_ndt:
        axis.axvline(
            x=sample["metadata"]["t"][0],
            ymin=ylim_low,
            ymax=ylim_high,
            color=color,
            linestyle="--",
            linewidth=lw_m,
            zorder=1000 + zorder_cnt,
            alpha=alpha,
        )
    # Starting point
    if keep_starting_point:
        axis.scatter(
            sample["metadata"]["t"][0] + markershift_starting_point,
            b_low[0] + (sample["metadata"]["z"][0] * (b_high[0] - b_low[0])),
            markersize_starting_point,
            marker=markertype_starting_point,
            color=color,
            alpha=alpha,
            zorder=1000 + zorder_cnt,
        )


def plot_func_model_n(
    model_name: str,
    axis: Axes,
    theta_mean: pd.DataFrame | None = None,
    theta_samples: pd.DataFrame | None = None,
    data: pd.DataFrame | None = None,
    n_trajectories: int = 10,
    bin_size: float = 0.05,
    n_samples: int = 10,
    linewidth_histogram: float | int = 0.5,
    linewidth_model: float | int = 0.5,
    legend_fontsize: int = 7,
    legend_shadow: bool = True,
    legend_location: str = "upper right",
    delta_t_model: float = 0.01,
    add_legend: bool = True,
    alpha_mean: float = 1.0,
    alpha_predictive: float = 0.05,
    alpha_trajectories: float = 0.5,
    keep_frame: bool = False,
    random_state: int | None = None,
    **kwargs,
) -> Axes:
    """Calculate and plot posterior predictive for a model.

    Parameters
    ----------
    model_name : str
        Name of the model to simulate.
    axis : matplotlib.axes.Axes
        The axis to plot on.
    theta_mean : pandas.DataFrame, optional
        Mean parameter values for simulation.
    theta_samples : pandas.DataFrame, optional
        Posterior samples of parameters.
    data : pandas.DataFrame, optional
        Observed data to plot.
    n_trajectories : int, default=10
        Number of trajectories to plot.
    bin_size : float, default=0.05
        Size of bins for histograms.
    n_samples : int, default=10
        Number of posterior samples to use.
    linewidth_histogram : float or int, default=0.5
        Line width for histogram elements.
    linewidth_model : float or int, default=0.5
        Line width for model cartoon elements.
    legend_fontsize : int, default=7
        Font size for legend.
    legend_shadow : bool, default=True
        Whether to add shadow to legend box.
    legend_location : str, default="upper right"
        Location of legend.
    delta_t_model : float, default=0.01
        Time step size for model cartoon plotting.
    add_legend : bool, default=True
        Whether to add legend to plot.
    alpha_mean : float, default=1.0
        Transparency for main plot elements.
    alpha_predictive : float, default=0.05
        Transparency for posterior predictive samples.
    alpha_trajectories : float, default=0.5
        Transparency for trajectory paths.
    keep_frame : bool, default=False
        Whether to keep the frame around the plot.
    random_state : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional keyword arguments passed to plotting functions.

    Notes
    -----
    This function visualizes model predictions by plotting simulated trajectories,
    histograms of response times, and model cartoons. It can show both the mean
    prediction and uncertainty from posterior samples.
    """
    color_dict = {
        -1: "black",
        0: "black",
        1: "green",
        2: "blue",
        3: "red",
        4: "orange",
        5: "purple",
        6: "brown",
    }

    ylim_low, ylim_high = kwargs.get("ylims", (0, 5))
    xlim_low, xlim_high = kwargs.get("xlims", (0, 5))

    # # Extract some parameters from kwargs
    axis.set_xlim(xlim_low, xlim_high)
    axis.set_ylim(ylim_low, ylim_high)
    bins = list(np.arange(xlim_low, xlim_high, bin_size))

    # ADD MODEL:

    # RUN SIMULATIONS
    # -------------------------------

    # Simulator Data
    if random_state is not None:
        np.random.seed(random_state)

    rand_int = np.random.randint(0, 400000000)
    if theta_mean is not None:
        sim_out = simulator(
            model=model_name,
            theta=theta_mean.values,
            n_samples=n_samples,
            no_noise=False,
            delta_t=delta_t_model,
            random_state=rand_int,
        )

        sim_out_no_noise = simulator(
            model=model_name,
            theta=theta_mean.loc[np.random.choice(theta_mean.shape[0], 1), :].values,
            n_samples=1,
            no_noise=True,
            delta_t=delta_t_model,
            smooth_unif=False,
        )

    # Simulate model without noise: posterior samples
    if theta_samples is not None:
        posterior_pred_no_noise = {}
        posterior_pred_sims = {}
        for i, (chain, draw) in enumerate(
            list(theta_samples.index.droplevel("obs_n").unique())
        ):
            # Simulate model: no noise
            posterior_pred_no_noise[i] = simulator(
                model=model_name,
                theta=theta_samples.loc[(chain, draw), :].values,
                n_samples=1,
                no_noise=True,
                delta_t=delta_t_model,
                smooth_unif=False,
            )

            # Simulate model: posterior samples
            posterior_pred_sims[i] = simulator(
                model=model_name,
                theta=theta_samples.loc[(chain, draw), :].values,
                n_samples=1,
                no_noise=False,
                delta_t=delta_t_model,
                random_state=rand_int,
            )

    # Simulate trajectories
    sim_out_traj = {}
    for i in range(n_trajectories):
        if theta_mean is not None:
            tmp_theta = theta_mean.loc[
                (np.random.choice(theta_mean.shape[0], 1)), :
            ].values
        elif theta_samples is not None:
            # wrap in max statement here
            # because negative value are possible,
            # however refer to data instead of posterior samples
            random_index = tuple(
                [
                    np.random.choice(theta_samples.index.get_level_values(name_))
                    for name_ in ("chain", "draw", "obs_n")
                ]
            )

            tmp_theta = theta_samples.loc[random_index, :].values
        else:
            raise ValueError("No theta values provided but n_trajectories is > 0")

        sim_out_traj[i] = simulator(
            model=model_name,
            theta=tmp_theta,
            n_samples=1,
            no_noise=False,
            delta_t=delta_t_model,
            random_state=rand_int + i,
            smooth_unif=False,
        )

    # ADD HISTOGRAMS
    # -------------------------------
    choices = default_model_config[cast("SupportedModels", model_name)]["choices"]
    cnt_cumul = 0

    # POSTERIOR MEAN BASED HISTOGRAM
    if theta_mean is not None:
        b = np.maximum(sim_out["metadata"]["boundary"], 0)
        bottom = b[0]

        for i, choice in enumerate(choices):
            tmp_label = None

            if add_legend and i == 0:
                tmp_label = "PostPred"

            weights = np.tile(
                (1 / bin_size)
                / sim_out["rts"][sim_out["rts"] != -999].flatten().shape[0],
                reps=sim_out["rts"][
                    (sim_out["choices"] == choice) & (sim_out["rts"] != -999)
                ]
                .flatten()
                .shape[0],
            )

            axis.hist(
                np.abs(
                    sim_out["rts"][
                        (sim_out["choices"] == choice) & (sim_out["rts"] != -999)
                    ]
                ),
                bins=bins,
                bottom=bottom,
                weights=weights,
                histtype="step",
                alpha=alpha_mean,
                color=color_dict[choice],
                zorder=cnt_cumul,
                label=tmp_label,
                linewidth=linewidth_histogram,
                linestyle="-.",
            )
            cnt_cumul += 1

    # POSTERIOR SAMPLE BASED HISTOGRAM
    if theta_samples is not None:
        if theta_mean is None:
            bottom = np.max(
                [
                    np.maximum(
                        posterior_pred_no_noise[key_]["metadata"]["boundary"], 0
                    )[0]
                    for key_, _ in posterior_pred_no_noise.items()
                ]
            )

        for k, sim_out_tmp in posterior_pred_sims.items():
            for i, choice in enumerate(choices):
                tmp_label = None

                if add_legend and i == 0:
                    tmp_label = "PostPred"

                weights = np.tile(
                    (1 / bin_size)
                    / sim_out_tmp["rts"][sim_out_tmp["rts"] != -999].shape[0],
                    reps=sim_out_tmp["rts"][
                        (sim_out_tmp["choices"] == choice)
                        & (sim_out_tmp["rts"] != -999)
                    ].shape[0],
                )

                axis.hist(
                    np.abs(
                        sim_out_tmp["rts"][
                            (sim_out_tmp["choices"] == choice)
                            & (sim_out_tmp["rts"] != -999)
                        ]
                    ),
                    bins=bins,
                    bottom=bottom,
                    weights=weights,
                    histtype="step",
                    alpha=alpha_predictive,
                    color=color_dict[choice],
                    zorder=cnt_cumul,
                    label=tmp_label,
                    linewidth=linewidth_histogram,
                    linestyle="-.",
                )
                cnt_cumul += 1

    # DATA BASED HISTOGRAM
    if data is not None:
        for i, choice in enumerate(choices):
            tmp_label = None

            if add_legend and (i == 0):
                tmp_label = "Data"

            data_tmp = data.query(f"rt != {-999} and response == {choice}")["rt"].values
            weights = np.tile(
                (1 / bin_size) / data.shape[0],
                reps=data_tmp.shape[0],
            )

            axis.hist(
                np.abs(data_tmp),
                bins=bins,
                bottom=bottom,
                weights=weights,
                histtype="step",
                alpha=1,
                color=color_dict[choice],
                zorder=cnt_cumul,
                label="Data",
                linewidth=linewidth_histogram,
                linestyle="-",
            )
            cnt_cumul += 1

    # ADD MODEL CARTOONS:

    tmp_label = None
    z_cnt = 0
    if theta_samples is not None:
        for k, sim_out_tmp in posterior_pred_no_noise.items():
            t_s = np.arange(0, sim_out_tmp["metadata"]["max_t"], delta_t_model)
            _add_model_n_cartoon_to_ax(
                sample=sim_out_tmp,
                axis=axis,
                delta_t_graph=delta_t_model,
                alpha=alpha_predictive,
                lw_m=linewidth_model,
                tmp_label=tmp_label,
                linestyle="-",
                ylim=ylim_high,
                t_s=t_s,
                color_dict=color_dict,
                zorder_cnt=z_cnt,
            )
            z_cnt += 1

    if theta_mean is not None:
        t_s = np.arange(0, sim_out_no_noise["metadata"]["max_t"], delta_t_model)
        _add_model_n_cartoon_to_ax(
            sample=sim_out_no_noise,
            axis=axis,
            delta_t_graph=delta_t_model,
            alpha=alpha_mean,
            lw_m=linewidth_model,
            tmp_label=tmp_label,
            linestyle="-",
            ylim=ylim_high,
            t_s=t_s,
            color_dict=color_dict,
            zorder_cnt=z_cnt + 1,
        )

    if (n_trajectories > 0) and (
        (theta_mean is not None) or (theta_samples is not None)
    ):
        _add_trajectories_n(
            axis=axis,
            sample=sim_out_traj,
            t_s=t_s,
            delta_t_graph=delta_t_model,
            n=n_trajectories,
            alpha=alpha_trajectories,
            **kwargs,
        )
    elif n_trajectories > 0:
        raise ValueError("Trajectories are requested but no theta values are provided.")

    if add_legend:
        custom_elems = [
            Line2D([0], [0], color=color_dict[choice], lw=1) for choice in choices
        ]
        custom_titles = ["response: " + str(choice) for choice in choices]

        custom_elems.append(Line2D([0], [0], color="black", lw=1.0, linestyle="dashed"))

        axis.legend(
            custom_elems,
            custom_titles,
            fontsize=legend_fontsize,
            shadow=legend_shadow,
            loc=legend_location,
        )

    # FRAME
    if not keep_frame:
        axis.set_frame_on(False)

    return axis


def _add_trajectories_n(
    axis: Axes,
    sample: dict[Any, Any],
    t_s: np.ndarray,
    delta_t_graph: float = 0.01,
    n: int = 10,
    highlight_rt_choice: bool = True,
    marker_size_rt_choice: float = 50,
    marker_type_rt_choice: str = "*",
    linewidth: float = 1,
    alpha: float = 0.5,
    colors: str | list[str] | dict[str, str] | dict[int, str] = TRAJ_COLOR_DEFAULT_DICT,
    **kwargs,
):
    """Add simulated decision trajectories to a given matplotlib axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes, optional
        The axis to plot on
    sample : list of dict, optional
        List of dictionaries containing simulated trajectories and metadata
    t_s : numpy.ndarray, optional
        Array of time points for plotting
    delta_t_graph : float, default=0.01
        Time step size for plotting
    n : int, default=10
        Number of trajectories to plot
    highlight_rt_choice : bool, default=True
        Whether to highlight response time and choice with markers
    marker_size_rt_choice : float, default=50
        Size of markers for response time/choice
    marker_type_rt_choice : str, default="*"
        Marker style for chosen response
    linewidth: float, default=1
        Line width for trajectory paths
    alpha : float, default=0.5
        Transparency of trajectory paths
    colors : str or list or dict, default="black"
        Color(s) for trajectories. Can be:
        - str: Single color for all trajectories
        - list: List of colors mapped to possible choices
        - dict: Mapping of choices to colors
    **kwargs
        Additional keyword arguments passed to plotting functions

    Notes
    -----
    This function visualizes multiple simulated decision paths, optionally highlighting
    the response times and choices. Each trajectory shows the evidence accumulation
    process leading to a decision.
    """
    # Check trajectory color type
    if isinstance(colors, str):
        colors_dict = {
            value_: colors for value_ in sample[0]["metadata"]["possible_choices"]
        }
    elif isinstance(colors, list):
        colors_dict = {
            value_: colors[i]
            for i, value_ in enumerate(sample[0]["metadata"]["possible_choices"])
        }
    elif isinstance(colors, dict):
        colors_dict = colors
    else:
        raise ValueError(
            "The `color_trajectories` argument must be a string, list, or dict."
        )

    # Make bounds
    b = np.maximum(sample[0]["metadata"]["boundary"], 0)
    b_init = b[0]
    n_roll = int((sample[0]["metadata"]["t"][0] / delta_t_graph) + 1)
    b = np.roll(b, n_roll)
    b[:n_roll] = b_init

    # Trajectories
    for i in range(n):
        metadata = sample[i]["metadata"]
        tmp_traj = metadata["trajectory"]
        tmp_traj_choice = float(sample[i]["choices"].flatten())

        for j in range(len(metadata["possible_choices"])):
            tmp_maxid = np.minimum(
                np.argmax(np.where(tmp_traj[:, j] > -999)), t_s.shape[0]
            )

            # Identify boundary value at timepoint of crossing
            b_tmp = b[tmp_maxid + n_roll]

            axis.plot(
                t_s[:tmp_maxid] + metadata["t"][0],
                tmp_traj[:tmp_maxid, j],
                color=colors_dict[j],
                alpha=alpha,
                linewidth=linewidth,
                zorder=2000 + i,
            )

            if highlight_rt_choice and tmp_traj_choice == j:
                axis.scatter(
                    t_s[tmp_maxid] + metadata["t"][0],
                    b_tmp,
                    marker_size_rt_choice,
                    color=colors_dict[tmp_traj_choice],
                    alpha=1,
                    marker=marker_type_rt_choice,
                    zorder=2000 + i,
                )
            elif highlight_rt_choice and tmp_traj_choice != j:
                axis.scatter(
                    t_s[tmp_maxid] + metadata["t"][0],  #  + 0.05,
                    tmp_traj[tmp_maxid, j],
                    marker_size_rt_choice,
                    color=colors_dict[j],
                    alpha=1,
                    marker="|",
                    zorder=2000 + i,
                )


def _add_model_n_cartoon_to_ax(
    sample: dict,
    axis: Axes,
    delta_t_graph: float,
    lw_m: float,
    tmp_label: str | None,
    linestyle: str,
    ylim: float,
    t_s: np.ndarray,
    color_dict: dict,
    zorder_cnt: int,
    alpha: float | None = None,
    keep_boundary: bool = True,
    keep_slope: bool = True,
    keep_starting_point: bool = True,
) -> None:
    """Add model cartoon visualization to a matplotlib axis.

    Parameters
    ----------
    sample : dict
        Dictionary containing model metadata and trajectories
    axis : matplotlib.axes.Axes
        The axis to plot on
    delta_t_graph : float
        Time step size for plotting
    lw_m : float
        Line width for plotting model elements
    tmp_label : str | None
        Label for the plot legend
    linestyle : str
        Style of lines in plot
    ylim : float
        Y-axis limit value
    t_s : np.ndarray
        Array of time points
    color_dict : dict
        Dictionary mapping choice indices to colors
    zorder_cnt : int
        Base z-order for plot elements
    alpha : float | None, optional
        Transparency value, by default None
    keep_boundary : bool, optional
        Whether to plot decision boundaries, by default True
    keep_slope : bool, optional
        Whether to plot drift slopes, by default True
    keep_starting_point : bool, optional
        Whether to plot starting point marker, by default True

    Returns
    -------
    None
        Modifies the provided axis in place
    """
    b = np.maximum(sample["metadata"]["boundary"], 0)
    b_init = b[0]
    n_roll = int((sample["metadata"]["t"][0] / delta_t_graph) + 1)
    b = np.roll(b, n_roll)
    b[:n_roll] = b_init

    # Upper bound
    if keep_boundary:
        axis.plot(
            t_s,
            b[: t_s.shape[0]],
            color="black",
            alpha=alpha,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
            linestyle=linestyle,
            label=tmp_label,
        )

    # Starting point
    if keep_starting_point:
        axis.axvline(
            x=sample["metadata"]["t"][0],
            ymin=-ylim,
            ymax=ylim,
            color="black",
            linestyle=linestyle,
            linewidth=lw_m,
            alpha=alpha,
        )

    # Slopes
    if keep_slope:
        tmp_traj = sample["metadata"]["trajectory"]

        for i in range(len(sample["metadata"]["possible_choices"])):
            tmp_maxid = np.minimum(
                np.argmax(np.where(tmp_traj[:, i] > -999)), t_s.shape[0]
            )

            # Slope
            axis.plot(
                t_s[:tmp_maxid] + sample["metadata"]["t"][0],
                tmp_traj[:tmp_maxid, i],
                color=color_dict[i],
                linestyle=linestyle,
                alpha=alpha,
                zorder=1000 + zorder_cnt,
                linewidth=lw_m,
            )

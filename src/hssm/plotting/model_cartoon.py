"""Plotting functionalities for HSSM."""

import logging
from copy import deepcopy
from itertools import product
from typing import Dict, Iterable, List, Tuple, cast

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
from posterior_predictive import _process_linestyles_pp, _process_linewidths_pp
from ssms.basic_simulators.simulator import simulator

# Original model cartoon plot from gui
from ssms.config import model_config

from ..defaults import SupportedModels, default_model_config
from .utils import (
    _check_groups_and_groups_order,
    _check_sample_size,
    _get_plotting_df,
    _get_title,
    _hdi_to_interval,
    _subset_df,
    _use_traces_or_sample,
)

_logger = logging.getLogger("hssm")


def _histogram(a: np.ndarray, bins: int | np.ndarray | str | None = 100) -> np.ndarray:
    return pd.Series(
        np.histogram(a, bins=bins, density=True)[0],  # type: ignore
        name="bin_n",
        copy=False,
    )


def _plot_model_cartoon_1D(
    data: pd.DataFrame,
    model_name: str,
    plot_data: bool = True,
    plot_mean: bool = True,
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
    if "color" in kwargs:
        del kwargs["color"]
    colors = colors or ["#ec205b", "#338fb8"]

    if plot_data and isinstance(colors, str):
        raise ValueError("When `plot_data=True`, `colors` must be a list or dict.")

    if "ax" in kwargs:
        ax = kwargs.pop("ax")
    else:
        ax = plt.gca()

    model_params = default_model_config[cast(SupportedModels, model_name)][
        "list_params"
    ]

    data_posterior_predictive_mean = data.loc[
        (data.source == "posterior_predictive_mean") & (data.observed == "predicted"), :
    ]
    data_observed = data.loc[
        (data.source == "posterior_predictive_mean") & (data.observed == "observed"), :
    ]

    data_posterior_predictive = data.loc[
        (data.source == "posterior_predictive") & (data.observed == "predicted"), :
    ]

    if plot_mean:
        ax = plot_func_model(
            model_name=model_name,
            theta_mean=data_posterior_predictive_mean.reset_index()[model_params],
            theta_posterior=data_posterior_predictive[model_params],
            data=(data_observed.reset_index() if plot_data else None),
            axis=ax,
            value_range=kwargs.get("value_range", (-0.5, 5)),
            linewidth_histogram=kwargs.get("linewidth_histogram", 1.5),
            linewidth_model=kwargs.get("linewidth_model", 1.5),
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
    row: str | None = None,
    col: str | None = None,
    col_wrap: int | None = None,
    bins: int | np.ndarray | str | None = 100,
    range: tuple[float, float] | None = None,
    step: bool = False,
    interval: tuple[float, float] | None = None,
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
        bins=bins,
        range=range,
        step=step,
        interval=interval,
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

        custom_labels = ["observed", "mean_pp"]

        g.add_legend(
            dict(zip(custom_labels, custom_lines)),
            title="",
            label_order=["observed", "mean_pp"],
        )

    if title:
        g.figure.subplots_adjust(top=0.9)
        g.figure.suptitle(title)

    g.set_xlabels(xlabel)
    g.set_ylabels(ylabel)

    return g


def compute_merge_necessary_deterministics(model, idata, inplace=True):
    """Compute the necessary deterministic variables for the model."""
    # Get the list of deterministic variables
    necessary_params = default_model_config[model.model_name]["list_params"]

    deterministics_list = []
    # Compute the deterministic variables
    for param in necessary_params:
        if param not in idata["posterior"].keys():
            if param in [
                deterministic.name for deterministic in model.pymc_model.deterministics
            ]:
                deterministics_list.append(
                    pm.compute_deterministics(
                        idata.posterior, model=model.pymc_model, var_names=param
                    )
                )

    deterministics_idata = xr.merge(deterministics_list)
    setattr(idata, "posterior", xr.merge([idata.posterior, deterministics_idata]))
    return idata


def attach_trialwise_params_to_df(model, df, idata):
    """Attach the trial-wise parameters to the dataframe."""
    necessary_params = default_model_config[model.model_name]["list_params"]
    for param in necessary_params:
        df[param] = 0.0

    for chain_tmp, draw_tmp in {(x[0], x[1]) for x in list(df.index) if x[0] != -1}:
        for param in necessary_params:
            df.loc[(chain_tmp, draw_tmp, slice(None)), param] = (
                idata["posterior"].sel(chain=chain_tmp, draw=draw_tmp)[param].values
            )
    return df


def _make_idata_mean_posterior(idata):
    setattr(idata, "posterior", idata.posterior.mean(dim=["chain", "draw"]))

    idata.posterior = idata.posterior.assign_coords(chain=[0], draw=[0])
    for data_var in list(idata.posterior.data_vars):
        idata.posterior[data_var] = idata.posterior[data_var].expand_dims(
            dim=["chain", "draw"], axis=[0, 1]
        )

    if "posterior_predictive" in idata:
        del idata.posterior_predictive
    return idata


# AF-TODO: Implement process_colors_pp
# def process_colors_pp(colors):


def plot_model_cartoon(
    model,
    idata: az.InferenceData | None = None,
    data: pd.DataFrame | None = None,
    plot_data: bool = True,
    n_samples: int | float | None = 20,
    row: str | None = None,
    col: str | None = None,
    col_wrap: int | None = None,
    groups: str | Iterable[str] | None = None,
    groups_order: Iterable[str] | dict[str, Iterable[str]] | None = None,
    bins: int | np.ndarray | str | None = 50,
    range: tuple[float, float] | None = None,
    step: bool = False,
    hdi: float | str | tuple[float, float] | None = None,
    show_mean: bool = True,
    show_samples: bool = False,
    colors: str | list[str] | None = None,
    linestyles: str | list[str] | tuple[str] | Dict[str, str] = "-",
    linewidths: float | list[float] | tuple[float] | Dict[str, float] = 1.25,
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
    plot_data : optional
        Whether to plot the observed data, by default True.
    n_samples : optional
        When idata is provided, the number or proportion of posterior predictive samples
        randomly drawn to be used from each chain for plotting. When idata is not
        provided, the number or proportion of posterior samples to be used to generate
        posterior predictive samples. The number or proportion are defined as follows:

        - When an integer >= 1, the number of samples to be extracted from the draw
          dimension.
        - When a float between 0 and 1, the proportion of samples to be extracted from
          the draw dimension.
        - When None, all samples are extracted.
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
    range : optional
            The lower and upper range of the bins. Lower and upper outliers are ignored.
            If not provided, range is simply the minimum and the maximum of the data, by
            default None.
    step : optional
        Whether to plot the distributions as a step function or a smooth density plot,
        by default False.
    hdi : optional
        A two-tuple of floats indicating the hdi to plot, by default None.
        The values in the tuple should be between 0 and 1, indicating the percentiles
        used to compute the interval. For example, (0.05, 0.95) will compute the 90%
        interval. There should be at least 50 posterior predictive samples for each
        chain for this to work properly. A warning message will be displayed if there
        are fewer than 50 posterior samples. If None, no interval is plotted.
        If a float, the interval the computed interval will be
        ((1 - hdi) / 2, 1 - (1 - hdi) / 2). If a string, the format needs
        to be e.g. '10%'.
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
    # Process hdi
    if hdi is not None:
        interval = _hdi_to_interval(hdi=hdi)
    else:
        interval = None

    # Process linestyles
    linestyles_ = _process_linestyles_pp(linestyles)
    # Process linewidths
    linewidths_ = _process_linewidths_pp(linewidths)

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
            and ("posterior_predictive" in idata)
        ):
            # Allows data to be None only when plot_data=False and no extra_dims
            # and posterior predictive samples are available
            data = None
        else:
            data = model.data

    # Mean version of plot
    if show_mean:
        if idata is None:
            idata_mean = _make_idata_mean_posterior(deepcopy(model.traces))
        else:
            idata_mean = _make_idata_mean_posterior(deepcopy(idata))

        idata_mean, _ = _use_traces_or_sample(model, data, idata_mean, n_samples=None)

        # Get the plotting dataframe by chain and sample
        plotting_df_mean = _get_plotting_df(
            idata_mean,
            data,
            extra_dims=extra_dims,
            n_samples=None,
            response_str=model.response_str,
        )

        # Get plotting dataframe for posterior mean

        # df by chain and sample
        idata_mean = compute_merge_necessary_deterministics(model, idata_mean)
        plotting_df_mean = attach_trialwise_params_to_df(
            model, plotting_df_mean, idata_mean
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

        plotting_df_mean["source"] = "posterior_predictive_mean"
    else:
        plotting_df_mean = None

    if show_samples:
        idata, sampled = _use_traces_or_sample(model, data, idata, n_samples=n_samples)

        # Get the plotting dataframe by chain and sample
        plotting_df = _get_plotting_df(
            idata,
            data,
            extra_dims=extra_dims,
            n_samples=None if sampled else n_samples,
            response_str=model.response_str,
        )

        # Get plotting dataframe for posterior mean

        # df by chain and sample
        idata = compute_merge_necessary_deterministics(model, idata)
        plotting_df = attach_trialwise_params_to_df(model, plotting_df, idata)

        # Flip the rt values if necessary
        if np.any(plotting_df["response"] == 0) and model.n_choices == 2:
            plotting_df["response"] = np.where(plotting_df["response"] == 0, -1, 1)
        if model.n_choices == 2:
            plotting_df["rt"] = plotting_df["rt"] * plotting_df["response"]

        plotting_df["source"] = "posterior_predictive"

    else:
        plotting_df = None

    # return plotting_df, plotting_df_mean

    # # return plotting_df, plotting_df_mean
    if interval is not None:
        _check_sample_size(plotting_df)

    if (plotting_df is not None) and (plotting_df_mean is not None):
        plotting_df = pd.concat([plotting_df, plotting_df_mean])
    elif plotting_df_mean is not None:
        plotting_df = plotting_df_mean

        # Then, plot the posterior predictive distribution against the observed data
        # Determine whether we are producing a single plot or a grid of plots

    if not extra_dims:
        ax = _plot_model_cartoon_1D(
            data=plotting_df,
            model_name=model.model_name,
            plot_data=plot_data,
            plot_mean=True,
            bins=bins,
            range=range,
            step=step,
            interval=interval,
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

        custom_labels = ["observed", "mean_pp"]
        ax.legend(custom_lines, custom_labels, title="", loc="upper right")
        return ax

    # The multiple dimensions case

    # If group is not provided, we are producing a grid of plots
    if groups is None:
        g = _plot_model_cartoon_2D(
            data=plotting_df,
            model_name=model.model_name,
            plot_data=plot_data,
            plot_mean=True,
            row=row,
            col=col,
            col_wrap=col_wrap,
            bins=bins,
            range=range,
            step=step,
            interval=interval,
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
    assert isinstance(groups_order, dict)
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
            plot_data=plot_data,
            plot_mean=True,
            row=row,
            col=col,
            col_wrap=col_wrap,
            bins=bins,
            range=range,
            step=step,
            interval=interval,
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
    theta_mean: pd.DataFrame,
    theta_posterior: pd.DataFrame,
    axis: Axes | None = None,
    data: pd.DataFrame | None = None,
    value_range: Tuple[float, float] | np.array | List[float] = None,
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
    markertype_starting_point: int | str = 0,
    markershift_starting_point: float | int = 0,
    linewidth_histogram: float | int = 0.5,
    linewidth_model: float | int = 0.5,
    color_data: str = "blue",
    color_pp_mean: str = "black",
    color_pp: str = "black",
    alpha_pp: float = 0.05,
    alpha_trajectories: float = 0.5,
    **kwargs,
):
    """Calculate posterior predictive for a certain bottom node.

    Arguments:
        bottom_node: pymc.stochastic
            Bottom node to compute posterior over.

        axis: matplotlib.axis
            Axis to plot into.

        value_range: numpy.ndarray
            Range over which to evaluate the likelihood.

    Optional:
        samples: int <default=10>
            Number of posterior samples to use.

        bin_size: float <default=0.05>
            Size of bins used for histograms

        alpha: float <default=0.05>
            alpha (transparency) level for the sample-wise elements of the plot

        add_posterior_uncertainty_rts: bool <default=True>
            Add sample by sample histograms?

        add_posterior_mean_rts: bool <default=True>
            Add a mean posterior?

        add_model: bool <default=True>
            Whether to add model cartoons to the plot.

        linewidth_histogram: float <default=0.5>
            linewdith of histrogram plot elements.

        linewidth_model: float <default=0.5>
            linewidth of plot elements concerning the model cartoons.

        legend_location: str <default='upper right'>
            string defining legend position. Find the rest of the options in the
            matplotlib documentation.

        legend_shadow: bool <default=True>
            Add shadow to legend box?

        legend_fontsize: float <default=12>
            Fontsize of legend.

        color_data : str <default="blue">
            Color for the data part of the plot.

        posterior_mean_color : str <default="red">
            Color for the posterior mean part of the plot.

        color_pp : str <default="black">
            Color for the posterior uncertainty part of the plot.

        delta_t_model:
            specifies plotting intervals for model cartoon elements of the graphs.
    """
    if value_range is None:
        # Infer from data by finding the min and max from the nodes
        raise NotImplementedError("value_range keyword argument must be supplied.")

    if len(value_range) > 2:
        value_range = (value_range[0], value_range[-1])

    # Extract some parameters from kwargs
    bins = np.arange(value_range[0], value_range[-1], bin_size)

    if model_config[model_name]["nchoices"] > 2:
        raise ValueError("The model plot works only for 2 choice models at the moment")

    # RUN SIMULATIONS
    # -------------------------------

    # Simulator Data from posterior mean
    if random_state is not None:
        np.random.seed(random_state)
    rand_int = np.random.choice(400000000)
    sim_out = simulator(
        model=model_name,
        theta=theta_mean.values,
        n_samples=n_samples,
        no_noise=False,
        delta_t=delta_t_model,
        bin_dim=None,
        random_state=rand_int,
    )

    # Simulate Trajectories
    sim_out_traj = {}
    for i in range(n_trajectories):
        rand_int = np.random.choice(400000000)
        sim_out_traj[i] = simulator(
            model=model_name,
            theta=theta_mean.loc[np.random.choice(theta_mean.shape[0], 1), :].values,
            n_samples=1,
            no_noise=False,
            delta_t=delta_t_model,
            bin_dim=None,
            random_state=rand_int,
            smooth_unif=False,
        )

    # Simulate model without noise: posterior mean
    # (this allows to extract the time-dynamics of the drift e.g.)
    sim_out_no_noise = simulator(
        model=model_name,
        theta=theta_mean.loc[np.random.choice(theta_mean.shape[0], 1), :].values,
        n_samples=1,
        no_noise=True,
        delta_t=delta_t_model,
        bin_dim=None,
        smooth_unif=False,
    )

    # Simulate model without noise: posterior samples
    posterior_pred_no_noise = {}
    for i, (chain, draw) in enumerate(
        list(theta_posterior.index.droplevel("obs_n").unique())
    ):
        posterior_pred_no_noise[i] = simulator(
            model=model_name,
            theta=theta_posterior.loc[(chain, draw), :].values,
            n_samples=1,
            no_noise=True,
            delta_t=delta_t_model,
            bin_dim=None,
            smooth_unif=False,
        )

    # Simulate model: posterior samples
    posterior_pred_sims = {}
    for i, (chain, draw) in enumerate(
        list(theta_posterior.index.droplevel("obs_n").unique())
    ):
        posterior_pred_sims[i] = simulator(
            model=model_name,
            theta=theta_posterior.loc[(chain, draw), :].values,
            n_samples=n_samples,
            no_noise=False,
            delta_t=delta_t_model,
            bin_dim=None,
            random_state=rand_int,
        )

    # ADD DATA HISTOGRAMS
    weights_up = np.tile(
        (1 / bin_size) / sim_out["rts"][(sim_out["rts"] != -999)].shape[0],
        reps=sim_out["rts"][(sim_out["rts"] != -999) & (sim_out["choices"] == 1)].shape[
            0
        ],
    )
    weights_down = np.tile(
        (1 / bin_size) / sim_out["rts"][(sim_out["rts"] != -999)].shape[0],
        reps=sim_out["rts"][(sim_out["rts"] != -999) & (sim_out["choices"] != 1)].shape[
            0
        ],
    )

    (b_high, b_low) = (
        np.maximum(sim_out["metadata"]["boundary"], 0),
        np.minimum((-1) * sim_out["metadata"]["boundary"], 0),
    )

    # ADD HISTOGRAMS
    # -------------------------------

    ylim = kwargs.pop("ylim", 3)
    # hist_bottom = kwargs.pop("hist_bottom", 2)
    hist_histtype = kwargs.pop("hist_histtype", "step")

    if ("ylim_high" in kwargs) and ("ylim_low" in kwargs):
        ylim_high = kwargs["ylim_high"]
        ylim_low = kwargs["ylim_low"]
    else:
        ylim_high = ylim
        ylim_low = -ylim

    if ("hist_bottom_high" in kwargs) and ("hist_bottom_low" in kwargs):
        hist_bottom_high = kwargs["hist_bottom_high"]
        hist_bottom_low = kwargs["hist_bottom_low"]
    else:
        hist_bottom_high = b_high[0]  # hist_bottom
        hist_bottom_low = -b_low[0]  # hist_bottom

    axis.set_xlim(value_range[0], value_range[-1])
    axis.set_ylim(ylim_low, ylim_high)
    axis_twin_up = axis.twinx()
    axis_twin_down = axis.twinx()
    axis_twin_up.set_ylim(ylim_low, ylim_high)
    axis_twin_up.set_yticks([])
    axis_twin_down.set_ylim(ylim_high, ylim_low)
    axis_twin_down.set_yticks([])
    axis_twin_down.set_axis_off()
    axis_twin_up.set_axis_off()

    # Add histograms for posterior mean simulation
    axis_twin_up.hist(
        np.abs(sim_out["rts"][(sim_out["rts"] != -999) & (sim_out["choices"] == 1)]),
        bins=bins,
        weights=weights_up,
        histtype=hist_histtype,
        bottom=hist_bottom_high,
        alpha=1,
        color=color_pp_mean,
        edgecolor=color_pp_mean,
        linewidth=linewidth_histogram,
        zorder=-1,
    )

    axis_twin_down.hist(
        np.abs(sim_out["rts"][(sim_out["rts"] != -999) & (sim_out["choices"] != 1)]),
        bins=bins,
        weights=weights_down,
        histtype=hist_histtype,
        bottom=hist_bottom_low,
        alpha=1,
        color=color_pp_mean,
        edgecolor=color_pp_mean,
        linewidth=linewidth_histogram,
        zorder=-1,
    )

    # Add histograms for posterior samples:
    for k, sim_out_tmp in posterior_pred_sims.items():
        weights_up = np.tile(
            (1 / bin_size) / sim_out_tmp["rts"][(sim_out_tmp["rts"] != -999)].shape[0],
            reps=sim_out_tmp["rts"][
                (sim_out_tmp["rts"] != -999) & (sim_out_tmp["choices"] == 1)
            ].shape[0],
        )
        weights_down = np.tile(
            (1 / bin_size) / sim_out_tmp["rts"][(sim_out_tmp["rts"] != -999)].shape[0],
            reps=sim_out_tmp["rts"][
                (sim_out_tmp["rts"] != -999) & (sim_out_tmp["choices"] != 1)
            ].shape[0],
        )

        # Add histograms for posterior samples
        axis_twin_up.hist(
            np.abs(
                sim_out_tmp["rts"][
                    (sim_out_tmp["rts"] != -999) & (sim_out_tmp["choices"] == 1)
                ]
            ),
            bins=bins,
            weights=weights_up,
            histtype=hist_histtype,
            bottom=hist_bottom_high,
            alpha=alpha_pp,
            color=color_pp,
            edgecolor=color_pp,
            linewidth=linewidth_histogram,
            zorder=k,
        )

        axis_twin_down.hist(
            np.abs(
                sim_out_tmp["rts"][
                    (sim_out_tmp["rts"] != -999) & (sim_out_tmp["choices"] != 1)
                ]
            ),
            bins=bins,
            weights=weights_down,
            histtype=hist_histtype,
            bottom=hist_bottom_low,
            alpha=alpha_pp,
            color=color_pp,
            edgecolor=color_pp,
            linewidth=linewidth_histogram,
            zorder=k,
        )

    # Add histograms for real data
    if data is not None:
        data_up = data.query(f"rt != {-999} and response == {1}")["rt"].values
        data_down = data.query(f"rt != {-999} and response != {1}")["rt"].values
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
            np.abs(data.query(f"rt != {-999} and response == {1}")["rt"].values),
            bins=bins,
            weights=weights_up_data,
            histtype=hist_histtype,
            bottom=hist_bottom_high,
            alpha=1,
            color=color_data,
            edgecolor=color_data,
            linewidth=linewidth_histogram,
            zorder=-1,
        )

        axis_twin_down.hist(
            np.abs(data.query(f"rt != {-999} and response != {1}")["rt"].values),
            bins=bins,
            weights=weights_down_data,
            histtype=hist_histtype,
            bottom=hist_bottom_low,
            alpha=1,
            color=color_data,
            edgecolor=color_data,
            linewidth=linewidth_histogram,
            zorder=-1,
        )

    # ADD MODEL CARTOONS:
    t_s = np.arange(0, sim_out["metadata"]["max_t"], delta_t_model)
    z_cnt = 0  # controlling the order of elements in plot

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
            alpha=alpha_pp,
            lw_m=linewidth_model,
            ylim_low=ylim_low,
            ylim_high=ylim_high,
            t_s=t_s,
            color=color_pp,
            zorder_cnt=z_cnt,
        )

        z_cnt += 1

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
        alpha=1,
        lw_m=linewidth_model,
        ylim_low=ylim_low,
        ylim_high=ylim_high,
        t_s=t_s,
        color=color_pp,
        zorder_cnt=z_cnt,
    )

    # Add in trajectories
    if n_trajectories > 0:
        _add_trajectories(
            axis=axis,
            sample=sim_out_traj,
            t_s=t_s,
            delta_t_graph=delta_t_model,
            n_trajectories=n_trajectories,
            alpha_trajectories=alpha_trajectories,
            **kwargs,
        )
    return axis


# AF-TODO: Add documentation for this function
def _add_trajectories(
    axis=None,
    sample=None,
    t_s=None,
    delta_t_graph=0.01,
    n_trajectories=10,
    supplied_trajectory=None,
    maxid_supplied_trajectory=1,  # useful for gifs
    highlight_trajectory_rt_choice=True,
    markersize_trajectory_rt_choice=50,
    markertype_trajectory_rt_choice="*",
    markercolor_trajectory_rt_choice="red",
    linewidth_trajectories=1,
    alpha_trajectories=0.5,
    color_trajectories="black",
    **kwargs,
):
    """Add trajectories to a given axis."""
    # Check markercolor type
    if isinstance(markercolor_trajectory_rt_choice, str):
        markercolor_trajectory_rt_choice_dict = {}
        for value_ in sample[0]["metadata"]["possible_choices"]:
            markercolor_trajectory_rt_choice_dict[value_] = (
                markercolor_trajectory_rt_choice
            )
    elif isinstance(markercolor_trajectory_rt_choice, list):
        cnt = 0
        for value_ in sample[0]["metadata"]["possible_choices"]:
            markercolor_trajectory_rt_choice_dict[value_] = (
                markercolor_trajectory_rt_choice[cnt]
            )
            cnt += 1
    elif isinstance(markercolor_trajectory_rt_choice, dict):
        markercolor_trajectory_rt_choice_dict = markercolor_trajectory_rt_choice
    else:
        pass

    # Check trajectory color type
    if isinstance(color_trajectories, str):
        color_trajectories_dict = {}
        for value_ in sample[0]["metadata"]["possible_choices"]:
            color_trajectories_dict[value_] = color_trajectories
    elif isinstance(color_trajectories, list):
        cnt = 0
        for value_ in sample[0]["metadata"]["possible_choices"]:
            color_trajectories_dict[value_] = color_trajectories[cnt]
            cnt += 1
    elif isinstance(color_trajectories, dict):
        color_trajectories_dict = color_trajectories
    else:
        pass

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
    for i in range(n_trajectories):
        tmp_traj = sample[i]["metadata"]["trajectory"]
        tmp_traj_choice = float(sample[i]["choices"].flatten())
        maxid = np.minimum(np.argmax(np.where(tmp_traj > -999)), t_s.shape[0])

        # Identify boundary value at timepoint of crossing
        b_tmp = b_high[maxid + n_roll] if tmp_traj_choice > 0 else b_low[maxid + n_roll]

        axis.plot(
            t_s[:maxid] + sample[i]["metadata"]["t"][0],  # sample.t.values[0],
            tmp_traj[:maxid],
            color=color_trajectories_dict[tmp_traj_choice],
            alpha=alpha_trajectories,
            linewidth=linewidth_trajectories,
            zorder=2000 + i,
        )

        if highlight_trajectory_rt_choice:
            axis.scatter(
                t_s[maxid] + sample[i]["metadata"]["t"][0],  # sample.t.values[0],
                b_tmp,
                # tmp_traj[maxid],
                markersize_trajectory_rt_choice,
                color=markercolor_trajectory_rt_choice_dict[tmp_traj_choice],
                alpha=1,
                marker=markertype_trajectory_rt_choice,
                zorder=2000 + i,
            )


# AF-TODO: Add documentation to this function
def _add_model_cartoon_to_ax(
    sample=None,
    axis=None,
    keep_slope=True,
    keep_boundary=True,
    keep_ndt=True,
    keep_starting_point=True,
    markersize_starting_point=80,
    markertype_starting_point=1,
    markershift_starting_point=-0.05,
    delta_t_graph=None,
    alpha=None,
    lw_m=None,
    tmp_label=None,
    ylim_low=None,
    ylim_high=None,
    t_s=None,
    zorder_cnt=1,
    color="black",
):
    # Make bounds
    (b_high, b_low) = (
        np.maximum(sample["metadata"]["boundary"], 0),
        np.minimum((-1) * sample["metadata"]["boundary"], 0),
    )

    b_h_init = b_high[0]
    b_l_init = b_low[0]
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
            t_s,  # + sample.t.values[0],
            b_high[: t_s.shape[0]],
            color=color,
            alpha=alpha,
            zorder=1000 + zorder_cnt,
            linewidth=lw_m,
            label=tmp_label,
        )

        # Lower bound
        axis.plot(
            t_s,  # + sample.t.values[0],
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
    model_name,
    theta,
    axis,
    n_trajectories=10,
    value_range=None,
    bin_size=0.05,
    n_samples=10,
    linewidth_histogram=0.5,
    linewidth_model=0.5,
    legend_fontsize=7,
    legend_shadow=True,
    legend_location="upper right",
    delta_t_model=0.001,
    add_legend=True,
    alpha=1.0,
    keep_frame=False,
    random_state=None,
    **kwargs,
):
    """Calculate posterior predictive for a certain bottom node.

    Arguments:
        bottom_node: pymc.stochastic
            Bottom node to compute posterior over.

        axis: matplotlib.axis
            Axis to plot into.

        value_range: numpy.ndarray
            Range over which to evaluate the likelihood.

    Optional:
        samples: int <default=10>
            Number of posterior samples to use.

        bin_size: float <default=0.05>
            Size of bins used for histograms

        alpha: float <default=0.05>
            alpha (transparency) level for the sample-wise elements of the plot

        add_posterior_uncertainty_rts: bool <default=True>
            Add sample by sample histograms?

        add_posterior_mean_rts: bool <default=True>
            Add a mean posterior?

        add_model: bool <default=True>
            Whether to add model cartoons to the plot.

        linewidth_histogram: float <default=0.5>
            linewdith of histrogram plot elements.

        linewidth_model: float <default=0.5>
            linewidth of plot elements concerning the model cartoons.

        legend_loc: str <default='upper right'>
            string defining legend position. Find the rest of the options
            in the matplotlib documentation.

        legend_shadow: bool <default=True>
            Add shadow to legend box?

        legend_fontsize: float <default=12>
            Fontsize of legend.

        color_data : str <default="blue">
            Color for the data part of the plot.

        posterior_mean_color : str <default="red">
            Color for the posterior mean part of the plot.

        color_pp : str <default="black">
            Color for the posterior uncertainty part of the plot.


        delta_t_model:
            specifies plotting intervals for model cartoon elements of the graphs.
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

    # AF-TODO: Add a mean version of this !
    if value_range is None:
        # Infer from data by finding the min and max from the nodes
        raise NotImplementedError("value_range keyword argument must be supplied.")

    if len(value_range) > 2:
        value_range = (value_range[0], value_range[-1])

    # Extract some parameters from kwargs
    bins = np.arange(value_range[0], value_range[-1], bin_size)
    # ------------
    ylim = kwargs.pop("ylim", 4)

    axis.set_xlim(value_range[0], value_range[-1])
    axis.set_ylim(0, ylim)

    # ADD MODEL:

    # RUN SIMULATIONS
    # -------------------------------

    # Simulator Data
    if random_state is not None:
        np.random.seed(random_state)

    rand_int = np.random.choice(400000000)
    sim_out = simulator(
        model=model_name,
        theta=theta,
        n_samples=n_samples,
        no_noise=False,
        delta_t=0.001,
        bin_dim=None,
        random_state=rand_int,
    )
    choices = sim_out["metadata"]["possible_choices"]
    sim_out_traj = {}
    for i in range(n_trajectories):
        rand_int = np.random.choice(400000000)
        sim_out_traj[i] = simulator(
            model=model_name,
            theta=theta,
            n_samples=1,
            no_noise=False,
            delta_t=0.001,
            bin_dim=None,
            random_state=rand_int,
            smooth_unif=False,
        )

    sim_out_no_noise = simulator(
        model=model_name,
        theta=theta,
        n_samples=1,
        no_noise=True,
        delta_t=0.001,
        bin_dim=None,
        smooth_unif=False,
    )

    # ADD HISTOGRAMS
    # -------------------------------

    # POSTERIOR BASED HISTOGRAM
    j = 0
    b = np.maximum(sim_out["metadata"]["boundary"], 0)
    bottom = b[0]
    for choice in choices:
        tmp_label = None

        if add_legend and j == 0:
            tmp_label = "PostPred"

        weights = np.tile(
            (1 / bin_size) / sim_out["rts"].shape[0],
            reps=sim_out["rts"][
                (sim_out["choices"] == choice) & (sim_out["rts"] != -999)
            ].shape[0],
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
            alpha=alpha,
            color=color_dict[choice],
            zorder=-1,
            label=tmp_label,
            linewidth=linewidth_histogram,
        )
        j += 1

    # ADD MODEL:
    tmp_label = None
    j = 0
    t_s = np.arange(0, sim_out["metadata"]["max_t"], delta_t_model)

    if add_legend and (j == 0):
        tmp_label = "PostPred"

    _add_model_n_cartoon_to_ax(
        sample=sim_out_no_noise,
        axis=axis,
        delta_t_graph=delta_t_model,
        sample_hist_alpha=alpha,
        lw_m=linewidth_model,
        tmp_label=tmp_label,
        linestyle="-",
        ylim=ylim,
        t_s=t_s,
        color_dict=color_dict,
        zorder_cnt=j,
    )

    if n_trajectories > 0:
        _add_trajectories_n(
            axis=axis,
            sample=sim_out_traj,
            t_s=t_s,
            delta_t_graph=delta_t_model,
            n_trajectories=n_trajectories,
            **kwargs,
        )

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
    axis=None,
    sample=None,
    t_s=None,
    delta_t_graph=0.01,
    n_trajectories=10,
    highlight_trajectory_rt_choice=True,
    markersize_trajectory_rt_choice=50,
    markertype_trajectory_rt_choice="*",
    markercolor_trajectory_rt_choice="black",
    linewidth_trajectories=1,
    alpha_trajectories=0.5,
    color_trajectories="black",
    **kwargs,
):
    """Add trajectories to a given axis."""
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

    # Check trajectory color type
    if isinstance(color_trajectories, str):
        color_trajectories_dict = {}
        for value_ in sample[0]["metadata"]["possible_choices"]:
            color_trajectories_dict[value_] = color_trajectories
    elif isinstance(color_trajectories, list):
        cnt = 0
        for value_ in sample[0]["metadata"]["possible_choices"]:
            color_trajectories_dict[value_] = color_trajectories[cnt]
            cnt += 1
    elif isinstance(color_trajectories, dict):
        color_trajectories_dict = color_trajectories
    else:
        pass

    # Make bounds
    b = np.maximum(sample[0]["metadata"]["boundary"], 0)
    b_init = b[0]
    n_roll = int((sample[0]["metadata"]["t"][0] / delta_t_graph) + 1)
    b = np.roll(b, n_roll)
    b[:n_roll] = b_init

    # Trajectories
    for i in range(n_trajectories):
        tmp_traj = sample[i]["metadata"]["trajectory"]
        tmp_traj_choice = float(sample[i]["choices"].flatten())

        for j in range(len(sample[i]["metadata"]["possible_choices"])):
            tmp_maxid = np.minimum(
                np.argmax(np.where(tmp_traj[:, j] > -999)), t_s.shape[0]
            )

            # Identify boundary value at timepoint of crossing
            b_tmp = b[tmp_maxid + n_roll]

            axis.plot(
                t_s[:tmp_maxid] + sample[i]["metadata"]["t"][0],
                tmp_traj[:tmp_maxid, j],
                color=color_dict[j],
                alpha=alpha_trajectories,
                linewidth=linewidth_trajectories,
                zorder=2000 + i,
            )

            if highlight_trajectory_rt_choice and tmp_traj_choice == j:
                axis.scatter(
                    t_s[tmp_maxid] + sample[i]["metadata"]["t"][0],
                    b_tmp,
                    markersize_trajectory_rt_choice,
                    color=color_dict[tmp_traj_choice],
                    alpha=1,
                    marker=markertype_trajectory_rt_choice,
                    zorder=2000 + i,
                )
            elif highlight_trajectory_rt_choice and tmp_traj_choice != j:
                axis.scatter(
                    t_s[tmp_maxid] + sample[i]["metadata"]["t"][0] + 0.05,
                    tmp_traj[tmp_maxid, j],
                    markersize_trajectory_rt_choice,
                    color=color_dict[j],
                    alpha=1,
                    marker=5,
                    zorder=2000 + i,
                )


def _add_model_n_cartoon_to_ax(
    sample=None,
    axis=None,
    delta_t_graph=None,
    sample_hist_alpha=None,
    keep_boundary=True,
    keep_ndt=True,
    keep_slope=True,
    keep_starting_point=True,
    lw_m=None,
    linestyle="-",
    tmp_label=None,
    ylim=None,
    t_s=None,
    zorder_cnt=1,
    color_dict=None,
):
    """Add model cartoon to axis."""
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
            alpha=sample_hist_alpha,
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
            alpha=sample_hist_alpha,
        )

    # # MAKE SLOPES (VIA TRAJECTORIES HERE --> RUN NOISE FREE SIMULATIONS)!
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
                alpha=sample_hist_alpha,
                zorder=1000 + zorder_cnt,
                linewidth=lw_m,
            )

    return b[0]

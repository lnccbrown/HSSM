"""Plotting functionalities for HSSM."""

import logging
from itertools import product
from typing import Iterable

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import (
    _check_groups_and_groups_order,
    _check_sample_size,
    _get_plotting_df,
    _get_title,
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


def _plot_posterior_predictive_1D(
    data: pd.DataFrame,
    plot_data: bool = True,
    bins: int | np.ndarray | str | None = 100,
    range: tuple[float, float] | None = None,
    step: bool = False,
    interval: tuple[float, float] | None = None,
    colors: str | list[str] | None = None,
    linestyles: str | list[str] = "-",
    linewidths: float | list[float] = 1.25,
    title: str | None = "Posterior Predictive Distribution",
    xlabel: str | None = "Response Time",
    ylabel: str | None = "Density",
    **kwargs,
) -> mpl.axes.Axes:
    """Plot the posterior predictive distribution against the observed data.

    Check the `plot_posterior_predictive` function below for docstring.

    Returns
    -------
    mpl.Axes
        A matplotlib Axes object containing the plot.
    """
    if "color" in kwargs:
        del kwargs["color"]
    colors = colors or ["#ec205b", "#338fb8"]

    styles: dict[str, str | float] = {}

    if plot_data and isinstance(colors, str):
        raise ValueError("When `plot_data=True`, `colors` must be a list or dict.")

    styles["color"] = colors[0] if isinstance(colors, list) else colors
    styles["linestyle"] = linestyles[0] if isinstance(linestyles, list) else linestyles
    styles["linewidth"] = linewidths[0] if isinstance(linewidths, list) else linewidths

    predicted = data.loc[data["observed"] == "predicted", "rt"]
    bin_edges = np.histogram_bin_edges(predicted, bins=bins, range=range)  # type: ignore

    if "ax" in kwargs:
        ax = kwargs.pop("ax")
    else:
        ax = plt.gca()

    hists = (
        predicted.groupby(["chain", "draw"])
        .apply(_histogram, bins=bin_edges)
        .reset_index(level=2, name="rt")
        .rename(columns={"level_2": "bin_n"})
    )
    hists_groupby = hists.groupby("bin_n")["rt"]
    hists_mean = hists_groupby.mean()

    ax.plot(
        bin_edges[:-1],
        hists_mean,
        drawstyle="steps" if step else "default",
        **styles,
        **kwargs,
    )

    if interval is not None:
        hists_lower = hists_groupby.quantile(interval[0])
        hists_upper = hists_groupby.quantile(interval[1])
        ax.fill_between(
            bin_edges[:-1],
            hists_lower,
            hists_upper,
            color=styles["color"],
            alpha=0.1,
            **kwargs,
        )

    if plot_data:
        styles["color"] = colors[1]
        styles["linestyle"] = (
            linestyles[1] if isinstance(linestyles, list) else linestyles
        )
        styles["linewidth"] = (
            linewidths[1] if isinstance(linewidths, list) else linewidths
        )

        observed = data.loc[data["observed"] == "observed", "rt"]
        data_hist = _histogram(observed.values, bins=bin_edges)
        ax.plot(
            bin_edges[:-1],
            data_hist,
            drawstyle="steps" if step else "default",
            **styles,
            **kwargs,
        )

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    return ax


def _plot_posterior_predictive_2D(
    data: pd.DataFrame,
    plot_data: bool = True,
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
    title: str | None = "Posterior Predictive Distribution",
    xlabel: str | None = "Response Time",
    ylabel: str | None = "Density",
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
        _plot_posterior_predictive_1D,
        plot_data=plot_data,
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
        g.add_legend(
            dict(zip(["predicted", "observed"], g.fig.axes[0].get_lines())),
            title="",
            label_order=["observed", "predicted"],
        )

    if title:
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(title)

    g.set_xlabels(xlabel)
    g.set_ylabels(ylabel)

    return g


def plot_posterior_predictive(
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
    bins: int | np.ndarray | str | None = 100,
    range: tuple[float, float] | None = None,
    step: bool = False,
    interval: tuple[float, float] | None = None,
    colors: str | list[str] | None = None,
    linestyles: str | list[str] = "-",
    linewidths: float | list[float] = 1.25,
    title: str | None = "Posterior Predictive Distribution",
    xlabel: str | None = "Response Time",
    ylabel: str | None = "Density",
    grid_kwargs: dict | None = None,
    **kwargs,
) -> mpl.axes.Axes | sns.FacetGrid | list[sns.FacetGrid]:
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
        The lower and upper range of the bins. Lower and upper outliers are ignored. If
        not provided, range is simply the minimum and the maximum of the data, by
        default None.
    step : optional
        Whether to plot the distributions as a step function or a smooth density plot,
        by default False.
    interval : optional
        A two-tuple of floats indicating the interval of uncertainty, by default None.
        The values in the tuple should be between 0 and 1, indicating the percentiles
        used to compute the interval. For example, (0.05, 0.95) will compute the 90%
        interval. There should be at least 50 posterior predictive samples for each
        chain for this to work properly. A warning message will be displayed if there
        are fewer than 50 posterior samples. If None, no interval is plotted.
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

    idata, sampled = _use_traces_or_sample(model, data, idata, n_samples=n_samples)

    plotting_df = _get_plotting_df(
        idata, data, extra_dims=extra_dims, n_samples=None if sampled else n_samples
    )

    if interval is not None:
        _check_sample_size(plotting_df)

    # Flip the rt values if necessary
    if np.any(plotting_df["response"] == 0):
        plotting_df["response"] = np.where(plotting_df["response"] == 0, -1, 1)
    if model.n_responses == 2:
        plotting_df["rt"] = plotting_df["rt"] * plotting_df["response"]

    # Then, plot the posterior predictive distribution against the observed data
    # Determine whether we are producing a single plot or a grid of plots

    if not extra_dims:
        ax = _plot_posterior_predictive_1D(
            data=plotting_df,
            plot_data=plot_data,
            bins=bins,
            range=range,
            step=step,
            interval=interval,
            colors=colors,
            linestyles=linestyles,
            linewidths=linewidths,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            **kwargs,
        )

        return ax

    # The multiple dimensions case

    # If group is not provided, we are producing a grid of plots
    if groups is None:
        g = _plot_posterior_predictive_2D(
            data=plotting_df,
            plot_data=plot_data,
            row=row,
            col=col,
            col_wrap=col_wrap,
            bins=bins,
            range=range,
            step=step,
            interval=interval,
            colors=colors,
            linestyles=linestyles,
            linewidths=linewidths,
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
        g = _plot_posterior_predictive_2D(
            data=df,
            plot_data=plot_data,
            row=row,
            col=col,
            col_wrap=col_wrap,
            bins=bins,
            range=range,
            step=step,
            interval=interval,
            colors=colors,
            linestyles=linestyles,
            linewidths=linewidths,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            grid_kwargs=grid_kwargs,
            **kwargs,
        )

        plots.append(g)

    return plots

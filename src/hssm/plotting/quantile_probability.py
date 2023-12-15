"""Code for producing quantile probability plots."""

import logging
from itertools import product
from typing import Any, Iterable

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import (
    _check_groups_and_groups_order,
    _get_plotting_df,
    _get_title,
    _process_df_for_qp_plot,
    _subset_df,
    _use_traces_or_sample,
)

_logger = logging.getLogger("hssm")


def _plot_quantile_probability_1D(
    data: pd.DataFrame,
    cond: str,
    x: str = "proportion",
    y: str = "rt",
    hue: str = "quantile",
    plot_posterior: bool = True,
    correct: str | None = None,
    q: int | Iterable[float] = 5,
    title: str | None = "Quantile Probability Plot",
    xlabel: str | None = "Proportion",
    ylabel: str | None = None,
    xticklabels: Iterable["str"] | None = None,
    data_kwargs: dict[str, Any] | None = None,
    pps_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> mpl.axes.Axes:
    """Produce one quantile probability plot.

    Used internally by the functions below to produce the plot.
    See the functions below for docstrings.
    """
    plot_data = _process_df_for_qp_plot(data, q, cond, correct)
    df_data = plot_data.loc[plot_data["observed"] == "observed", :]

    ax = kwargs.get("ax", plt.gca())

    if data_kwargs is None:
        data_kwargs = kwargs.copy()

    data_kwargs_default = {
        "marker": "X",
    }

    data_kwargs = data_kwargs_default | data_kwargs

    ax = sns.lineplot(
        data=df_data,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        **data_kwargs,
    )

    if plot_posterior:
        df_posterior = plot_data.loc[plot_data["observed"] == "predicted", :]

        if pps_kwargs is None:
            pps_kwargs = kwargs.copy()

        pps_kwargs_default = {
            "marker": "o",
            "alpha": 0.3,
        }

        pps_kwargs = pps_kwargs_default | pps_kwargs
        ax = sns.scatterplot(
            data=df_posterior,
            x=x,
            y=y,
            hue="quantile",
            ax=ax,
            **pps_kwargs,
        )

    ticks_and_labels = (
        df_data.groupby(x, observed=True)[cond].first().reset_index(x, drop=False)
    )
    xticks = ticks_and_labels[x]
    xticklabels = xticklabels or ticks_and_labels[cond]
    secax = ax.twiny()
    secax.set_xticks(xticks)
    secax.set_xticklabels(xticklabels)
    secax.set_xlim(*ax.get_xlim())
    secax.set_label(cond)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.legend(ncol=2)

    return ax


def _plot_quantile_probability_2D(
    data: pd.DataFrame,
    cond: str,
    x: str = "proportion",
    y: str = "rt",
    hue: str = "quantile",
    row: str | None = None,
    col: str | None = None,
    col_wrap: int | None = None,
    plot_posterior: bool = True,
    correct: str | None = None,
    q: int | Iterable[float] = 5,
    title: str | None = "Quantile Probability Plot",
    xlabel: str | None = "Proportion",
    ylabel: str | None = None,
    xticklabels: Iterable["str"] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    data_kwargs: dict[str, Any] | None = None,
    pps_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> sns.FacetGrid:
    """Plot the quantile probabilities against the observed data.

    Check the function below for docstring.

    Returns
    -------
    sns.FacetGrid
        A seaborn FacetGrid object containing the plot.
    """
    g = sns.FacetGrid(
        data=data,
        row=row,
        col=col,
        col_wrap=col_wrap,
        legend_out=True,
        **(grid_kwargs or {}),
    )
    g.map_dataframe(
        _plot_quantile_probability_1D,
        x=x,
        y=y,
        cond=cond,
        hue=hue,
        plot_posterior=plot_posterior,
        correct=correct,
        q=q,
        title=None,
        xlabel=xlabel,
        ylabel=ylabel,
        xticklabels=xticklabels,
        data_kwargs=data_kwargs,
        pps_kwargs=pps_kwargs,
        **kwargs,
    )

    g.add_legend()
    if title:
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle(title)

    g.set_xlabels(xlabel)
    g.set_ylabels(ylabel)

    # Ensures that the x-limits for the axes on top are correct.
    first_ax = g.fig.axes[0]
    for ax in g.fig.get_axes():
        if ax.get_label() == cond:
            ax.set_xlim(first_ax.get_xlim())

    return g


def plot_quantile_probability(
    model,
    cond: str,
    data: pd.DataFrame | None = None,
    idata: az.InferenceData | None = None,
    n_samples: int = 20,
    x: str = "proportion",
    y: str = "rt",
    hue: str = "quantile",
    row: str | None = None,
    col: str | None = None,
    col_wrap: int | None = None,
    groups: str | Iterable[str] | None = None,
    groups_order: Iterable[str] | dict[str, Iterable[str]] | None = None,
    plot_posterior: bool = True,
    correct: str | None = None,
    q: int | Iterable[float] = 5,
    title: str | None = "Quantile Probability Plot",
    xlabel: str | None = "Proportion",
    ylabel: str | None = None,
    xticklabels: Iterable["str"] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    data_kwargs: dict[str, Any] | None = None,
    pps_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> sns.FacetGrid:
    """Plot the quantile probabilities against the observed data.

    Parameters
    ----------
    model
        A model object that has a `plot_quantile_probability` method.
    cond
        The column in `data` that indicates the conditions.
    data : optional
        A pandas DataFrame containing the observed data. If None, the data from
        `idata.observed_data` will be used.
    idata : optional
        An arviz InferenceData object. If None, the model's trace will be used.
        If the model's trace does not contain posterior predictive samples, and
        "plot_posterior" is True, will use the model and `data` to produce posterior
        predictive samples.
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
    x : optional
        The column in `data` that indicates the x-axis variable. By default, this is
        "proportion", which is the proportion of (in)correct responses in each group in
        `cond`.
    y : optional
        The column in `data` that indicates the y-axis variable. By default, this is
        "rt", which is the response time.
    hue : optional
        The column in `data` that indicates the hue variable. By default, this is
        "quantile", which is the quantile of the response time.
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
    plot_posterior : optional
        Whether to plot the posterior predictive distribution. By default True.
    correct : optional
        The column in `data` that indicates the correct responses. If None, `response`
        column from `data` indicates whether the response is correct or not. By default
        None.
    q : optional
        If an `int`, quantiles will be determined using np.linspace(0, 1, q) (0 and 1
        will be excluded. If an iterable, will generate quantiles according to this
        iterable.
    title : optional
        The title of the plot, by default "Quantile Predictive Plot". Ignored
        when `groups` is provided.
    xlabel : optional
        The label for the x-axis, by default "Proportion".
    ylabel : optional
        The label for the y-axis, by default None.
    xticklabels : optional
        The labels for groups on the top x-axis, by default None, which will be inferred
        from the data.
    grid_kwargs : optional
        Keyword arguments passed to seaborn.FacetGrid.
    data_kwargs : optional
        Keyword arguments passed to seaborn.lineplot.
    pps_kwargs : optional
        Keyword arguments passed to seaborn.scatterplot.
    kwargs : optional
        Keyword arguments passed to both seaborn.lineplot and seaborn.scatterplot.

    Returns
    -------
    mpl.axes.Axes | sns.FacetGrid | list[sns.FacetGrid]
        A seaborn FacetGrid object containing the plot.
    """
    if data is None:
        data = model.data

    groups, groups_order = _check_groups_and_groups_order(
        groups, groups_order, row, col
    )

    extra_dims = (
        list(
            set(
                [
                    e
                    for e in [cond, correct, row, col, kwargs.get("marker", None)]
                    if e is not None
                ]
            )
        )
        or None
    )
    if extra_dims is not None and groups is not None:
        extra_dims += list(groups)

    if plot_posterior:
        # Use the model's trace if idata is None
        idata, sampled = _use_traces_or_sample(model, data, idata, n_samples)

        plotting_df = _get_plotting_df(
            idata, data, extra_dims=extra_dims, n_samples=None if sampled else n_samples
        )
    else:
        plotting_df = _get_plotting_df(
            None, data, extra_dims=extra_dims, n_samples=None
        )

    # Flip the rt values if necessary
    if np.any(plotting_df["response"] == 0):
        plotting_df["response"] = np.where(plotting_df["response"] == 0, -1, 1)
    if model.n_responses == 2:
        plotting_df["rt"] = plotting_df["rt"] * plotting_df["response"]

    # If group is not provided, we are producing a single plot
    if row is None and col is None and groups is None:
        ax = _plot_quantile_probability_1D(
            plotting_df,
            cond=cond,
            x=x,
            y=y,
            hue=hue,
            plot_posterior=plot_posterior,
            correct=correct,
            q=q,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xticklabels=xticklabels,
            data_kwargs=data_kwargs,
            pps_kwargs=pps_kwargs,
            **kwargs,
        )

        return ax

    # If group is not provided, we are producing a grid of plots
    if groups is None:
        g = _plot_quantile_probability_2D(
            plotting_df,
            cond=cond,
            x=x,
            y=y,
            hue=hue,
            row=row,
            col=col,
            col_wrap=col_wrap,
            plot_posterior=plot_posterior,
            correct=correct,
            q=q,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xticklabels=xticklabels,
            grid_kwargs=grid_kwargs,
            data_kwargs=data_kwargs,
            pps_kwargs=pps_kwargs,
            **kwargs,
        )

        return g

    # If group is provided, we are producing a list of plots
    plots = []
    assert isinstance(groups_order, dict)
    orders = product(
        *[groups_order.get(g, plotting_df[g].unique().tolist()) for g in groups]
    )

    for order in orders:
        df = _subset_df(plotting_df, groups, order)
        title = _get_title(groups, order)
        if len(df) == 0:
            _logger.warning("No data for group %s. Skipping this group", title)
            continue
        g = _plot_quantile_probability_2D(
            plotting_df,
            cond=cond,
            x=x,
            y=y,
            hue=hue,
            row=row,
            col=col,
            col_wrap=col_wrap,
            plot_posterior=plot_posterior,
            correct=correct,
            q=q,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xticklabels=xticklabels,
            grid_kwargs=grid_kwargs,
            data_kwargs=data_kwargs,
            pps_kwargs=pps_kwargs,
            **kwargs,
        )

        plots.append(g)

    return plots

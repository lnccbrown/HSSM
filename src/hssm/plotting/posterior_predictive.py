"""Plotting functionalities for HSSM."""

import logging
from itertools import product
from typing import Iterable, Mapping

import arviz as az
import matplotlib as mpl

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import _get_plotting_df, _get_title, _subset_df

_logger = logging.getLogger("hssm")


def _plot_posterior_predictive_1D(
    data: pd.DataFrame,
    plot_data: bool = True,
    palette: str | list | dict | None = None,
    stat: str = "density",
    binwidth: float | tuple[float, float] | None = 0.1,
    element: str | bool = "step",
    title: str | None = None,
    xlabel: str | None = "Response Time",
    ylabel: str | None = None,
    **kwargs,
) -> sns.FacetGrid:
    """Plot the posterior predictive distribution against the observed data.

    Check the `plot_posterior_predictive` function below for docstring.

    Returns
    -------
    mpl.Axes
        A matplotlib Axes object containing the plot.
    """
    data = data if plot_data else data.loc[data["observed"] == "predicted", :]
    ax = sns.histplot(
        data=data,
        hue="observed" if plot_data else None,
        stat=stat,
        x="Response Time",
        fill=False,
        binwidth=binwidth,
        element=element,
        palette=palette or ["#ec205b", "#338fb8"],
        common_bins=False,
        common_norm=False,
        **kwargs,
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def _plot_posterior_predictive_2D(
    data: pd.DataFrame,
    plot_data: bool = True,
    row: str | None = None,
    col: str | None = None,
    col_wrap: int | None = None,
    palette: str | list | dict | None = None,
    stat: str = "density",
    binwidth: float | tuple[float, float] | None = 0.1,
    element: str | bool = "step",
    grid_kwargs: dict | None = None,
    title: str | None = None,
    xlabel: str | None = "Response Time",
    ylabel: str | None = None,
    **kwargs,
) -> sns.FacetGrid:
    """Plot the posterior predictive distribution against the observed data.

    Check the function below for docstring.

    Returns
    -------
    sns.FacetGrid
        A seaborn FacetGrid object containing the plot.
    """
    data = data if plot_data else data.loc[data["observed"] == "predicted", :]
    g = sns.FacetGrid(
        data=data,
        col=col,
        row=row,
        col_wrap=col_wrap,
        legend_out=True,
        hue="observed" if plot_data else None,
        palette=palette or ["#ec205b", "#338fb8"],
        **(grid_kwargs or {}),
    )

    g.map_dataframe(
        sns.histplot,
        x="Response Time",
        stat=stat,
        fill=False,
        binwidth=binwidth,
        element=element,
        common_bins=False,
        common_norm=False,
        **kwargs,
    )

    g.add_legend(title="", label_order=["observed", "predicted"])

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
    palette: str | list | dict | None = None,
    stat: str = "density",
    binwidth: float | tuple[float, float] | None = 0.1,
    element: str | bool = "step",
    title: str | None = None,
    xlabel: str | None = "Response Time",
    ylabel: str | None = None,
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
    palette : optional
        Colors to use for the different levels of the hue variable. Should be something
        that can be interpreted by color_palette(), or a dictionary mapping hue levels
        to matplotlib colors., by default None.
    stat : optional
        The stats to be plotted. By default `"density"`. Other options include `"kde"`,
        etc. Please see the documentation for the [`sns.histplot` function]
        (https://seaborn.pydata.org/generated/seaborn.histplot.html) for more options.
    binwidth : optional
        A number or a pair of numbers. Width of each bin. There are two other binning
        options: `bins` and `binrange`. Please see the documentation for the
        [`sns.histplot` function]
        (https://seaborn.pydata.org/generated/seaborn.histplot.html) for how to use
        them. in combination to control the bins. This argument overrides bins but can
        be used with `binrange.`
    element : optional
        Visual representation of the histogram statistic. Only relevant with univariate
        data. Default is `"step"`. Other options include `"bars"` and `"poly"`.
    grid_kwargs : optional
        Additional keyword arguments are passed to the [`sns.FacetGrid` constructor]
        (https://seaborn.pydata.org/generated/seaborn.FacetGrid.html#seaborn.FacetGrid.__init__)
        when any of row or col is provided. When producing a single plot, these
        arguments are ignored.
    kwargs : optional
        Additional keyword arguments passed to the [sns.`histplot` function]
        (https://seaborn.pydata.org/generated/seaborn.histplot.html).

    Returns
    -------
    mpl.axes.Axes | sns.FacetGrid
        The matplotlib `axis` or seaborn `FacetGrid` object containing the plot.
    """
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
                groups_order, Mapping
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

    # First, determine whether posterior predictive samples are available
    # If not, we need to sample from the posterior
    if idata is None:
        if model.traces is None:
            raise ValueError(
                "No InferenceData object provided. Please provide an InferenceData "
                + "object or sample the model first using model.sample()."
            )
        idata = model.traces

    extra_dims = [e for e in [row, col] if e is not None] or None
    if extra_dims is not None and groups is not None:
        extra_dims += list(groups)

    if data is None:
        if not extra_dims and not plot_data and "posterior_predictive" in idata:
            # Allows data to be None only when plot_data=False and no extra_dims
            # and posterior predictive samples are available
            data = None
        else:
            data = model.data

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
        plotting_df = _get_plotting_df(
            idata, data, extra_dims=extra_dims, n_samples=None
        )
    else:
        plotting_df = _get_plotting_df(
            idata, data, extra_dims=extra_dims, n_samples=n_samples
        )

    # Flip the rt values if necessary
    if np.any(plotting_df["rt"] == 0):
        plotting_df["rt"] = np.where(plotting_df["rt"] == 0, -1, 1)
    if model.n_responses == 2:
        plotting_df["Response Time"] = plotting_df["rt"] * plotting_df["response"]
    else:
        plotting_df["Response Time"] = plotting_df["rt"]

    # Then, plot the posterior predictive distribution against the observed data
    # Determine whether we are producing a single plot or a grid of plots

    if not extra_dims:
        ax = _plot_posterior_predictive_1D(
            data=plotting_df,
            plot_data=plot_data,
            stat=stat,
            binwidth=binwidth,
            element=element,
            palette=palette,
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
            palette=palette,
            stat=stat,
            binwidth=binwidth,
            element=element,
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
            palette=palette,
            stat=stat,
            binwidth=binwidth,
            element=element,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            grid_kwargs=grid_kwargs,
            **kwargs,
        )

        plots.append(g)

    return plots

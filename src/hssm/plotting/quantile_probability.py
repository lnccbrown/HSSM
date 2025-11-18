"""Code for producing quantile probability plots."""

import logging
from itertools import product
from typing import Any, Iterable, Literal

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from .utils import (
    _check_groups_and_groups_order,
    _get_plotting_df,
    _get_title,
    _process_df_for_qp_plot,
    _subset_df,
    _use_traces_or_sample,
)

_logger = logging.getLogger("hssm")


def _confidence_to_n_std(confidence: float, n_dim: int = 2) -> float:
    """
    Convert confidence level to number of standard deviations for an ellipse.

    For a bivariate Gaussian, the ellipse at n_std standard deviations
    contains a certain percentage of the probability mass.

    Parameters
    ----------
    confidence : float
        Confidence level between 0 and 1 (e.g., 0.95 for 95%)
    n_dim : int, default=2
        Number of dimensions (default: 2 for bivariate)

    Returns
    -------
    float
        Number of standard deviations

    Notes
    -----
    The relationship is n_std = sqrt(chi2.ppf(confidence, df=n_dim))
    """
    if not 0 < confidence < 1:
        raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")

    chi2_val = chi2.ppf(confidence, df=n_dim)
    n_std = np.sqrt(chi2_val)

    return n_std


def _compute_ellipse_params(
    df_group: pd.DataFrame,
    x_col: str,
    y_col: str,
    confidence: float = 0.95,
) -> dict[str, Any] | None:
    """
    Compute ellipse parameters from a group of points.

    Parameters
    ----------
    df_group : pd.DataFrame
        Group of points (single quantile + condition combo)
    x_col : str
        Column name for x values (proportion)
    y_col : str
        Column name for y values (rt)
    confidence : float, default=0.95
        Confidence level for ellipse

    Returns
    -------
    dict or None
        Dictionary with 'center', 'width', 'height', 'angle', 'n_points'
        or None if insufficient data
    """
    # Need at least 3 points to compute covariance
    if len(df_group) < 3:
        return None

    # Extract points
    points = df_group[[x_col, y_col]].values

    # Compute mean and covariance
    mean = points.mean(axis=0)
    cov = np.cov(points.T)

    # Stabilize covariance matrix by adding a small constant to the diagonal
    # and symmetrizing the matrix
    cov = (cov + cov.T) / 2 + 1e-10 * np.eye(cov.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # if np.any(eigenvalues <= 0):
    #     return None

    # Compute ellipse parameters
    n_std = _confidence_to_n_std(confidence)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    _logger.debug(
        "Ellipse parameters: len(df_group)=%d, "
        "mean=%s, width=%f, height=%f, angle=%f, n_points=%d",
        len(df_group),
        mean,
        width,
        height,
        angle,
        len(df_group),
    )

    return {
        "center": mean,
        "width": width,
        "height": height,
        "angle": angle,
        "n_points": len(df_group),
    }


def _plot_quantile_probability_1D(
    data: pd.DataFrame,
    cond: str,
    x: str = "proportion",
    y: str = "rt",
    hue: str = "quantile",
    plot_predictive: bool = True,
    predictive_style: Literal["points", "ellipse", "both"] = "points",
    ellipse_confidence: float = 0.95,
    ellipse_min_points: int = 5,
    correct: str | None = None,
    q: int | Iterable[float] = 5,
    quantile_by: list[str] | str | None = None,
    title: str | None = "Quantile Probability Plot",
    xlabel: str | None = "Proportion",
    ylabel: str | None = None,
    xticklabels: Iterable["str"] | None = None,
    data_kwargs: dict[str, Any] | None = None,
    predictive_samples_kwargs: dict[str, Any] | None = None,
    ellipse_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> mpl.axes.Axes:
    """Produce one quantile probability plot.

    Used internally by the functions below to produce the plot.
    See the functions below for docstrings.

    Parameters
    ----------
    predictive_style : {'points', 'ellipse', 'both'}
        How to plot posterior predictive samples:
        - 'points': Traditional scatter plot (default)
        - 'ellipse': Confidence ellipses for each quantile+condition group
        - 'both': Both points and ellipses
    ellipse_confidence : float, default=0.95
        Confidence level for ellipses (0 to 1)
    ellipse_min_points : int, default=5
        Minimum number of points required to draw an ellipse
    ellipse_kwargs : dict, optional
        Additional kwargs for ellipse patches (facecolor, edgecolor, alpha, etc.)

    Notes
    -----
    Ellipses show the bivariate confidence region for the (proportion, rt)
    pairs within each quantile+condition group. This is useful for:
    - Visualizing uncertainty more compactly than point clouds
    - Showing correlation structure in the predictions
    - Reducing visual clutter with many posterior samples
    """
    plot_data = _process_df_for_qp_plot(data, q, cond, correct, quantile_by)
    df_data = plot_data.loc[plot_data["observed"] == "observed", :]

    ax = kwargs.get("ax", plt.gca())

    if data_kwargs is None:
        data_kwargs = kwargs.copy()

    data_kwargs_default = {
        "marker": "X",
    }

    data_kwargs = data_kwargs_default | data_kwargs

    # Plot observed data (always as line plot)
    ax = sns.lineplot(
        data=df_data,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        **data_kwargs,
    )

    if plot_predictive:
        df_predictive = plot_data.loc[plot_data["observed"] == "predicted", :]

        # Determine if we need to plot points, ellipses, or both
        plot_points = predictive_style in ["points", "both"]
        plot_ellipses = predictive_style in ["ellipse", "both"]

        # Plot points if requested
        if plot_points:
            if predictive_samples_kwargs is None:
                predictive_samples_kwargs = kwargs.copy()

            predictive_samples_kwargs_default = {
                "marker": "o",
                "alpha": 0.3,
            }

            predictive_samples_kwargs = (
                predictive_samples_kwargs_default | predictive_samples_kwargs
            )
            ax = sns.scatterplot(
                data=df_predictive,
                x=x,
                y=y,
                hue=hue,
                ax=ax,
                **predictive_samples_kwargs,
            )

        # Plot ellipses if requested
        if plot_ellipses:
            if ellipse_kwargs is None:
                ellipse_kwargs = {}

            ellipse_kwargs_default = {
                "facecolor": "none",
                "linewidth": 1.5,
                "alpha": 0.6,
            }
            ellipse_kwargs = ellipse_kwargs_default | ellipse_kwargs

            # Get unique quantiles, conditions, and x-values
            quantiles = df_predictive[hue].unique()
            conditions = df_predictive[cond].unique()
            correct_vals = df_predictive["is_correct"].unique()

            # Get the color mapping from the current plot
            # The line plot has already been created, so we can extract colors from it
            handles, labels = ax.get_legend_handles_labels()
            color_map = {
                label: handle.get_color() for handle, label in zip(handles, labels)
            }

            # Group by quantile, condition, and x-value
            for quantile in quantiles:
                # Get color for this quantile from the existing plot
                quantile_color = color_map.get(str(quantile), None)
                if quantile_color is None:  # pragma: no cover
                    _logger.warning(
                        "Could not find color for quantile=%s in legend", quantile
                    )
                    continue

                for cond_val in conditions:
                    for correct_val in correct_vals:
                        # Filter data for this specific point
                        # (quantile, condition, x-value)
                        mask = (
                            (df_predictive[hue] == quantile)
                            & (df_predictive[cond] == cond_val)
                            & (df_predictive["is_correct"] == correct_val)
                        )

                        df_group = df_predictive[mask]

                        # Skip if insufficient points
                        if len(df_group) < ellipse_min_points:
                            _logger.debug(
                                "Skipping ellipse for quantile=%s, %s=%s, %s=%s: "
                                "only %d points (need %d)",
                                quantile,
                                cond,
                                cond_val,
                                x,
                                correct_val,
                                len(df_group),
                                ellipse_min_points,
                            )
                            continue

                        # Compute ellipse parameters
                        ellipse_params = _compute_ellipse_params(
                            df_group, x, y, confidence=ellipse_confidence
                        )

                        if ellipse_params is None:  # pragma: no cover
                            _logger.warning(
                                "Could not compute ellipse for quantile=%s, "
                                "%s=%s, %s=%s (singular covariance)",
                                quantile,
                                cond,
                                cond_val,
                                x,
                                correct_val,
                            )
                            continue

                        # Create ellipse patch with quantile color
                        ellipse = Ellipse(
                            xy=ellipse_params["center"],
                            width=ellipse_params["width"],
                            height=ellipse_params["height"],
                            angle=ellipse_params["angle"],
                            edgecolor=quantile_color,
                            **ellipse_kwargs,
                        )
                        ax.add_patch(ellipse)

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
    plot_predictive: bool = True,
    predictive_style: Literal["points", "ellipse", "both"] = "points",
    ellipse_confidence: float = 0.95,
    ellipse_min_points: int = 5,
    correct: str | None = None,
    q: int | Iterable[float] = 5,
    quantile_by: list[str] | str | None = None,
    title: str | None = "Quantile Probability Plot",
    xlabel: str | None = "Proportion",
    ylabel: str | None = None,
    xticklabels: Iterable["str"] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    data_kwargs: dict[str, Any] | None = None,
    predictive_samples_kwargs: dict[str, Any] | None = None,
    ellipse_kwargs: dict[str, Any] | None = None,
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
        plot_predictive=plot_predictive,
        predictive_style=predictive_style,
        ellipse_confidence=ellipse_confidence,
        ellipse_min_points=ellipse_min_points,
        correct=correct,
        q=q,
        quantile_by=quantile_by,
        title=None,
        xlabel=xlabel,
        ylabel=ylabel,
        xticklabels=xticklabels,
        data_kwargs=data_kwargs,
        predictive_samples_kwargs=predictive_samples_kwargs,
        ellipse_kwargs=ellipse_kwargs,
        **kwargs,
    )

    g.add_legend()
    if title:
        g.figure.subplots_adjust(top=0.85)
        g.figure.suptitle(title)

    g.set_xlabels(xlabel)
    g.set_ylabels(ylabel)

    # Ensures that the x-limits for the axes on top are correct.
    first_ax = g.figure.axes[0]
    for ax in g.figure.get_axes():
        if ax.get_label() == cond:
            ax.set_xlim(first_ax.get_xlim())

    return g


def plot_quantile_probability(
    model,
    cond: str,
    data: pd.DataFrame | None = None,
    idata: az.InferenceData | None = None,
    predictive_group: Literal["posterior_predictive", "prior_predictive"]
    | None = "posterior_predictive",
    n_samples: int = 20,
    x: str = "proportion",
    y: str = "rt",
    hue: str = "quantile",
    row: str | None = None,
    col: str | None = None,
    col_wrap: int | None = None,
    groups: str | Iterable[str] | None = None,
    groups_order: Iterable[str] | dict[str, Iterable[str]] | None = None,
    predictive_style: Literal["points", "ellipse", "both"] = "points",
    ellipse_confidence: float = 0.95,
    ellipse_min_points: int = 5,
    correct: str | None = None,
    q: int | Iterable[float] = 5,
    quantile_by: list[str] | str | None = None,
    title: str | None = "Quantile Probability Plot",
    xlabel: str | None = "Proportion",
    ylabel: str | None = None,
    xticklabels: Iterable["str"] | None = None,
    grid_kwargs: dict[str, Any] | None = None,
    data_kwargs: dict[str, Any] | None = None,
    predictive_samples_kwargs: dict[str, Any] | None = None,
    ellipse_kwargs: dict[str, Any] | None = None,
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
        "plot_predictive" is True, will use the model and `data` to produce posterior
        predictive samples.
    predictive_group : optional
        The type of predictive distribution to plot, by default "posterior_predictive".
        Can be "posterior_predictive" or "prior_predictive".
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
    predictive_style : {'points', 'ellipse', 'both'}, default='points'
        How to plot posterior predictive samples:
        - 'points': Traditional scatter plot (default)
        - 'ellipse': Confidence ellipses for each quantile+condition group
        - 'both': Both points and ellipses
    ellipse_confidence : float, default=0.95
        Confidence level for ellipses (0 to 1). Only used when predictive_style is
        'ellipse' or 'both'. For example, 0.95 creates ellipses that
        contain approximately 95% of the probability mass for
        each quantile+condition group.
    ellipse_min_points : int, default=5
        Minimum number of points required to draw an ellipse. Groups with fewer points
        will be skipped with a debug warning.
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
    predictive_samples_kwargs : optional
        Keyword arguments passed to seaborn.scatterplot.
    ellipse_kwargs : optional
        Keyword arguments passed to matplotlib.patches.Ellipse. Useful for customizing
        ellipse appearance (e.g., facecolor, edgecolor, alpha, linewidth).
    kwargs : optional
        Keyword arguments passed to both seaborn.lineplot and seaborn.scatterplot.

    Returns
    -------
    mpl.axes.Axes | sns.FacetGrid | list[sns.FacetGrid]
        A seaborn FacetGrid object containing the plot.
    """
    # AF-TODO: Should provide a few more safeguards to ensure
    # 1. quantile_by dimension is a column(s) of strings
    # 2. there is no overlap between quantile_by and extra_dims

    # Location of those safeguards doesn't have to be directly
    # in this function, but the logic pertains how this
    # function is supposed to be used.

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

    if predictive_group is not None:
        # Use the model's trace if idata is None
        idata, sampled = _use_traces_or_sample(
            model, data, idata, n_samples, predictive_group
        )

        plotting_df = _get_plotting_df(
            idata,
            data,
            extra_dims=extra_dims,
            quantile_by_dims=quantile_by,
            n_samples=None if sampled else n_samples,
            response_str=model.response_str,
            predictive_group=predictive_group,
        )
    else:
        # Note if idata is passed as None,
        # predictive_group is ignored in _get_plotting_df
        plotting_df = _get_plotting_df(
            None,
            data,
            extra_dims=extra_dims,
            quantile_by_dims=quantile_by,
            n_samples=None,
            response_str=model.response_str,
        )

    # Flip the rt values if necessary
    if np.any(plotting_df["response"] == 0) and model.n_choices == 2:
        plotting_df["response"] = np.where(plotting_df["response"] == 0, -1, 1)
    if model.n_choices == 2:
        plotting_df["rt"] = plotting_df["rt"] * plotting_df["response"]

    # If group is not provided, we are producing a single plot
    if row is None and col is None and groups is None:
        ax = _plot_quantile_probability_1D(
            plotting_df,
            cond=cond,
            x=x,
            y=y,
            hue=hue,
            plot_predictive=predictive_group is not None,
            predictive_style=predictive_style,
            ellipse_confidence=ellipse_confidence,
            ellipse_min_points=ellipse_min_points,
            correct=correct,
            q=q,
            quantile_by=quantile_by,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xticklabels=xticklabels,
            data_kwargs=data_kwargs,
            predictive_samples_kwargs=predictive_samples_kwargs,
            ellipse_kwargs=ellipse_kwargs,
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
            plot_predictive=predictive_group is not None,
            predictive_style=predictive_style,
            ellipse_confidence=ellipse_confidence,
            ellipse_min_points=ellipse_min_points,
            correct=correct,
            q=q,
            quantile_by=quantile_by,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xticklabels=xticklabels,
            grid_kwargs=grid_kwargs,
            data_kwargs=data_kwargs,
            predictive_samples_kwargs=predictive_samples_kwargs,
            ellipse_kwargs=ellipse_kwargs,
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
            plot_predictive=predictive_group is not None,
            predictive_style=predictive_style,
            ellipse_confidence=ellipse_confidence,
            ellipse_min_points=ellipse_min_points,
            correct=correct,
            q=q,
            quantile_by=quantile_by,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xticklabels=xticklabels,
            grid_kwargs=grid_kwargs,
            data_kwargs=data_kwargs,
            predictive_samples_kwargs=predictive_samples_kwargs,
            ellipse_kwargs=ellipse_kwargs,
            **kwargs,
        )

        plots.append(g)

    return plots

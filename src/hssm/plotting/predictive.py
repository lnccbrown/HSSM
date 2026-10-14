"""Plotting functionalities for HSSM."""

import logging
from itertools import product
from typing import Iterable, Literal, cast, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from .utils import (
    DEFAULT_PREDICTIVE_COLORS,
    _band_alphas,
    _check_groups_and_groups_order,
    _check_sample_size,
    _curve_xy,
    _get_plotting_df,
    _get_title,
    _hdi_to_intervals,
    _subset_df,
    _use_traces_or_sample,
)

_logger = logging.getLogger("hssm")

_DEFAULT_TITLE = "Posterior Predictive Distribution"
_DEFAULT_XLABEL = "Response Time"


def _histogram(a: np.ndarray, bins: int | np.ndarray | str | None = 50) -> pd.Series:
    return pd.Series(
        np.histogram(a, bins=bins, density=True)[0],  # type: ignore
        name="bin_n",
        copy=False,
    )


def _kde_density(
    samples: np.ndarray,
    grid: np.ndarray,
    split_at_zero: bool,
    bw_method: str | float | None = None,
) -> np.ndarray:
    """Evaluate a kernel density estimate for RT samples on a shared grid.

    On the signed-RT axis (2-choice models) there is a hard density gap around
    zero — no response is faster than the non-decision time — and a single KDE
    would smooth mass straight across it. When `split_at_zero` is True (the
    grid spans zero), a separate KDE is therefore fit per choice side, each
    weighted by its response proportion and evaluated only on its own
    half-axis, so the two lobes jointly integrate to one and no mass bleeds
    across zero. Sides with fewer than two distinct samples (e.g. a draw with
    a single error trial) contribute zero density rather than failing.
    """
    out = np.zeros_like(grid, dtype=float)
    n_total = samples.size
    if n_total == 0:
        return out
    if split_at_zero:
        sides = [(samples[samples < 0], grid < 0), (samples[samples >= 0], grid >= 0)]
    else:
        sides = [(samples, np.ones_like(grid, dtype=bool))]
    for side_samples, mask in sides:
        if side_samples.size >= 2 and np.ptp(side_samples) > 0:
            kde = gaussian_kde(side_samples, bw_method=bw_method)
            out[mask] = kde(grid[mask]) * (side_samples.size / n_total)
    return out


def _kde_series(
    a: pd.Series,
    grid: np.ndarray,
    split_at_zero: bool,
    bw_method: str | float | None = None,
) -> pd.Series:
    return pd.Series(
        _kde_density(a.to_numpy(), grid, split_at_zero, bw_method),
        name="bin_n",
        copy=False,
    )


def _process_colors(
    colors: str | list[str] | dict[str, str] | None, plot_data: bool
) -> list[str]:
    """Resolve the `colors` argument to a [predicted, observed] pair."""
    if colors is None:
        return list(DEFAULT_PREDICTIVE_COLORS)
    if isinstance(colors, str):
        if plot_data:
            raise ValueError("When `plot_data=True`, `colors` must be a list or dict.")
        return [colors, DEFAULT_PREDICTIVE_COLORS[1]]
    if isinstance(colors, dict):
        if not set(colors.keys()).issubset({"predicted", "observed"}):
            raise ValueError(
                "The keys of the `colors` dictionary must be 'predicted' and/or "
                "'observed'."
            )
        return [
            colors.get("predicted", DEFAULT_PREDICTIVE_COLORS[0]),
            colors.get("observed", DEFAULT_PREDICTIVE_COLORS[1]),
        ]
    return list(colors)


def _plot_predictive_1D(
    data: pd.DataFrame,
    plot_data: bool = True,
    kind: Literal["hist", "kde"] = "hist",
    bins: int | np.ndarray | str | None = 50,
    grid_points: int = 256,
    bw_method: str | float | None = None,
    x_range: tuple[float, float] | None = None,
    step: bool = True,
    interval: list[tuple[float, float]] | tuple[float, float] | None = None,
    uncertainty: Literal["band", "samples", "both"] | None = None,
    alpha_mean: float = 1.0,
    alpha_uncertainty: float | None = None,
    colors: str | list[str] | dict[str, str] | None = None,
    linestyles: str | list[str] = "-",
    linewidths: float | list[float] = 1.25,
    title: str | None = _DEFAULT_TITLE,
    xlabel: str | None = _DEFAULT_XLABEL,
    ylabel: str | None = "Density",
    legend: bool = False,
    **kwargs,
) -> Axes:
    """Plot the posterior predictive distribution against the observed data.

    Check the `plot_predictive` function below for docstring.

    Returns
    -------
    Axes
        A matplotlib Axes object containing the plot.
    """
    if "color" in kwargs:
        del kwargs["color"]
    kwargs.pop("label", None)  # labels are managed internally for the legend
    # alpha/zorder are set internally per layer; a user-passed value overrides
    # them on the mean and observed lines instead of colliding.
    user_alpha = kwargs.pop("alpha", None)
    user_zorder = kwargs.pop("zorder", None)
    if user_alpha is not None:
        alpha_mean = user_alpha
    colors = _process_colors(colors, plot_data)
    intervals = [interval] if isinstance(interval, tuple) else interval

    styles: dict[str, str | float] = {}
    styles["color"] = colors[0]
    styles["linestyle"] = linestyles[0] if isinstance(linestyles, list) else linestyles
    styles["linewidth"] = linewidths[0] if isinstance(linewidths, list) else linewidths

    predicted = data.loc[data["observed"] == "predicted", "rt"]

    if "ax" in kwargs:
        ax = kwargs.pop("ax")
    else:
        ax = plt.gca()

    # Build the per-(chain, draw) curve matrix on a shared x support. Both
    # backends produce the same long shape, so the mean / band / sample
    # machinery below is backend-agnostic.
    if kind == "kde":
        x_lo, x_hi = x_range or (float(predicted.min()), float(predicted.max()))
        grid = np.linspace(x_lo, x_hi, grid_points)
        split_at_zero = x_lo < 0.0 < x_hi
        hists = (
            predicted.groupby(["chain", "draw"])
            .apply(
                _kde_series,
                grid=grid,
                split_at_zero=split_at_zero,
                bw_method=bw_method,
            )
            .reset_index(level=2, name="rt")
            .rename(columns={"level_2": "bin_n"})
        )

        def to_xy(values: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
            return grid, np.asarray(values)

    else:
        bin_edges = np.histogram_bin_edges(predicted, bins=bins, range=x_range)  # type: ignore
        hists = (
            predicted.groupby(["chain", "draw"])
            .apply(_histogram, bins=bin_edges)
            .reset_index(level=2, name="rt")
            .rename(columns={"level_2": "bin_n"})
        )

        def to_xy(values: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
            return _curve_xy(bin_edges, values, step)

    hists_groupby = hists.groupby("bin_n")["rt"]
    hists_mean = hists_groupby.mean()

    # Layer 1 — per-draw sample curves (the model-cartoon uncertainty idiom).
    if uncertainty in ("samples", "both"):
        n_draws = hists.groupby(level=["chain", "draw"]).ngroups
        alpha_samples = (
            alpha_uncertainty
            if alpha_uncertainty is not None
            else float(np.clip(4.0 / n_draws, 0.02, 0.25))
        )
        first = True
        for _, draw_hist in hists.groupby(level=["chain", "draw"]):
            x_s, y_s = to_xy(draw_hist.sort_values("bin_n")["rt"].to_numpy())
            ax.plot(
                x_s,
                y_s,
                color=styles["color"],
                linestyle=styles["linestyle"],
                linewidth=0.6 * float(styles["linewidth"]),
                alpha=alpha_samples,
                zorder=1,
                label="predictive draws" if first else "_nolegend_",
            )
            first = False

    # Layer 2 — graded uncertainty bands, widest first so narrower bands
    # print on top with higher opacity.
    if uncertainty in ("band", "both") and intervals:
        base_alpha = alpha_uncertainty if alpha_uncertainty is not None else 0.25
        band_alphas = _band_alphas(base_alpha, len(intervals))
        for (lo, hi), band_alpha in zip(intervals, band_alphas):
            x_b, y_lo = to_xy(hists_groupby.quantile(lo))
            _, y_hi = to_xy(hists_groupby.quantile(hi))
            ax.fill_between(
                x_b,
                y_lo,
                y_hi,
                color=styles["color"],
                alpha=float(band_alpha),
                linewidth=0,
                zorder=2,
                label=f"{hi - lo:.0%} interval",
            )

    # Layer 3 — the predictive mean.
    x_m, y_m = to_xy(hists_mean)
    ax.plot(
        x_m,
        y_m,
        color=styles["color"],
        linestyle=styles["linestyle"],
        linewidth=styles["linewidth"],
        alpha=alpha_mean,
        zorder=3 if user_zorder is None else user_zorder,
        label="predicted",
        **kwargs,
    )

    # Layer 4 — the observed data, always on top.
    if plot_data:
        styles["color"] = colors[1]
        styles["linestyle"] = (
            linestyles[1] if isinstance(linestyles, list) else linestyles
        )
        styles["linewidth"] = (
            linewidths[1] if isinstance(linewidths, list) else linewidths
        )

        observed = data.loc[data["observed"] == "observed", "rt"]
        if kind == "kde":
            data_hist = _kde_density(
                observed.to_numpy(), grid, split_at_zero, bw_method
            )
        else:
            data_hist = _histogram(observed.to_numpy(), bins=bin_edges)
        x_o, y_o = to_xy(data_hist)
        ax.plot(
            x_o,
            y_o,
            color=styles["color"],
            linestyle=styles["linestyle"],
            linewidth=styles["linewidth"],
            alpha=user_alpha,
            zorder=4 if user_zorder is None else user_zorder,
            label="observed",
            **kwargs,
        )

    sns.despine(ax=ax)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend and (plot_data or uncertainty is not None):
        handles, labels = ax.get_legend_handles_labels()
        order = _legend_order(labels)
        ax.legend(
            [handles[i] for i in order], [labels[i] for i in order], frameon=False
        )

    return ax


def _legend_order(labels: list[str]) -> list[int]:
    """Order legend entries: observed, predicted, bands (wide→narrow), draws."""

    def rank(label: str) -> tuple[int, float, str]:
        if label == "observed":
            return (0, 0.0, "")
        if label == "predicted":
            return (1, 0.0, "")
        if label.endswith("interval"):
            # widest band first: sort by parsed mass, descending
            try:
                mass = float(label.split("%")[0])
            except ValueError:
                mass = 0.0
            return (2, -mass, label)
        return (3, 0.0, label)

    return sorted(range(len(labels)), key=lambda i: rank(labels[i]))


def _plot_predictive_2D(
    data: pd.DataFrame,
    plot_data: bool = True,
    row: str | None = None,
    col: str | None = None,
    col_wrap: int | None = None,
    kind: Literal["hist", "kde"] = "hist",
    bins: int | np.ndarray | str | None = 50,
    grid_points: int = 256,
    bw_method: str | float | None = None,
    x_range: tuple[float, float] | None = None,
    step: bool = True,
    interval: list[tuple[float, float]] | tuple[float, float] | None = None,
    uncertainty: Literal["band", "samples", "both"] | None = None,
    alpha_mean: float = 1.0,
    alpha_uncertainty: float | None = None,
    colors: str | list[str] | dict[str, str] | None = None,
    linestyles: str | list[str] = "-",
    linewidths: float | list[float] = 1.25,
    title: str | None = _DEFAULT_TITLE,
    xlabel: str | None = _DEFAULT_XLABEL,
    ylabel: str | None = "Density",
    legend: bool = True,
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
        _plot_predictive_1D,
        plot_data=plot_data,
        kind=kind,
        bins=bins,
        grid_points=grid_points,
        bw_method=bw_method,
        x_range=x_range,
        step=step,
        interval=interval,
        uncertainty=uncertainty,
        alpha_mean=alpha_mean,
        alpha_uncertainty=alpha_uncertainty,
        colors=colors,
        linestyles=linestyles,
        linewidths=linewidths,
        title=None,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=False,
        **kwargs,
    )

    if legend and (plot_data or uncertainty is not None):
        handles, labels = g.figure.axes[0].get_legend_handles_labels()
        if handles:
            order = _legend_order(labels)
            g.add_legend(
                {labels[i]: handles[i] for i in order},
                title="",
            )

    if title:
        g.figure.subplots_adjust(top=0.9)
        g.figure.suptitle(title)

    g.set_xlabels(xlabel)
    g.set_ylabels(ylabel)

    return g


@overload
def _process_lines(
    line_attrs: str | Iterable[str] | tuple[str] | dict[str, str],
    mode: Literal["linestyles"],
) -> list[str]: ...


@overload
def _process_lines(
    line_attrs: float | Iterable[float] | tuple[float] | dict[str, float],
    mode: Literal["linewidths"],
) -> list[float]: ...


def _process_lines(
    line_attrs: (
        str
        | Iterable[str]
        | Iterable[float]
        | float
        | tuple[str]
        | tuple[float]
        | dict[str, str]
        | dict[str, float]
    ),
    mode: Literal["linestyles", "linewidths"],
) -> list[str] | list[float]:
    check_type: type[str | float]
    if mode == "linestyles":
        dict_defaults_ls: dict[str, str] = {"predicted": "-", "observed": "-"}
        check_type = str
    elif mode == "linewidths":
        dict_defaults_lw: dict[str, float] = {"predicted": 1.25, "observed": 1.25}
        check_type = float
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if isinstance(line_attrs, check_type):
        if check_type is str:
            return [cast("str", line_attrs)] * 2
        elif check_type is float:
            return [cast("float", line_attrs)] * 2
        else:
            raise ValueError(f"Invalid type: {check_type}")
    elif isinstance(line_attrs, (list, tuple)):
        line_attrs_list = list(line_attrs)
        if not all(isinstance(la, check_type) for la in line_attrs_list):
            raise ValueError(
                f"The `{mode}` argument must be a {check_type.__name__} or a list "
                f"of {check_type.__name__}s."
            )
        elif len(line_attrs_list) in {1, 2}:
            if check_type is str:
                str_list = cast("list[str]", line_attrs_list)
                return str_list * 2 if len(str_list) == 1 else str_list

            float_list = cast("list[float]", line_attrs_list)
            return float_list * 2 if len(float_list) == 1 else float_list
        else:
            raise ValueError(
                f"The `{mode}` argument must be a {check_type.__name__} or a list "
                f"of 1 or 2 {check_type.__name__}s."
            )
    elif isinstance(line_attrs, dict):
        if not set(line_attrs.keys()).issubset({"predicted", "observed"}):
            raise ValueError(
                f"The keys of the `{mode}` dictionary must be 'predicted' and/or "
                "'observed'."
            )
        else:
            if mode == "linestyles":
                line_attrs_str = cast("dict[str, str]", line_attrs)
                return [
                    line_attrs_str.get("predicted", dict_defaults_ls["predicted"]),
                    line_attrs_str.get("observed", dict_defaults_ls["observed"]),
                ]
            elif mode == "linewidths":
                line_attrs_float = cast("dict[str, float]", line_attrs)
                return [
                    line_attrs_float.get("predicted", dict_defaults_lw["predicted"]),
                    line_attrs_float.get("observed", dict_defaults_lw["observed"]),
                ]
    else:
        raise ValueError(
            f"The `{mode}` argument must be a string, a list of strings,"
            " or a dictionary."
        )


def plot_predictive(
    model,
    dt: xr.DataTree | None = None,
    data: pd.DataFrame | None = None,
    predictive_group: Literal[
        "posterior_predictive", "prior_predictive"
    ] = "posterior_predictive",
    plot_data: bool = True,
    n_samples: int | float | None = 20,
    row: str | None = None,
    col: str | None = None,
    col_wrap: int | None = None,
    groups: str | Iterable[str] | None = None,
    groups_order: Iterable[str] | dict[str, Iterable[str]] | None = None,
    kind: Literal["hist", "kde"] = "hist",
    bins: int | np.ndarray | str | None = 50,
    grid_points: int = 256,
    bw_method: str | float | None = None,
    x_range: tuple[float, float] | None = None,
    step: bool = True,
    hdi: float | str | tuple[float, float] | list | None = None,
    uncertainty: Literal["band", "samples", "both"] | None = "band",
    alpha_mean: float = 1.0,
    alpha_uncertainty: float | None = None,
    colors: str | list[str] | dict[str, str] | None = None,
    linestyles: str | list[str] | tuple[str] | dict[str, str] = "-",
    linewidths: float | list[float] | tuple[float] | dict[str, float] = 1.25,
    title: str | None = _DEFAULT_TITLE,
    xlabel: str | None = _DEFAULT_XLABEL,
    ylabel: str | None = "Density",
    legend: bool = True,
    grid_kwargs: dict | None = None,
    **kwargs,
) -> Axes | sns.FacetGrid | list[sns.FacetGrid]:
    """Plot the posterior predictive distribution against the observed data.

    Parameters
    ----------
    model : hssm.HSSM
        A fitted HSSM model.
    dt : optional
        The DataTree object with posterior samples. If not provided, will use the
        traces object stored inside the model. If posterior predictive samples are not
        present in this object, will generate posterior predictive samples using
        this DataTree object and the original data.
    data : optional
        The observed data.

        - If `data` is provided and the `dt` object does not contain a
        `"posterior_predictive"` group, will generate posterior predictive samples using
        covariate provided in this object. If the group does exist, it is assumed that
        the posterior predictive samples are generated with the covariates provided in
        this DataFrame.
        - If `data` is not provided (i.e., `data=None`), the behavior depends on whether
        "plot_data" is true or not. If `plot_data=True`, the plotting function will use
        the data stored in the `model` object and proceed as the case above. If
        `plot_data=False`, if posterior predictive samples are not present in the
        `dt` object, the plotting function will generate posterior predictive samples
        using the data stored in the `model` object. If posterior predictive samples
        exist in the `dt` object, these samples will be used for plotting, but a
        ValueError will be thrown if any of `col` or `row` is not None.
    predictive_group : optional
        The type of predictive distribution to plot, by default "posterior_predictive".
        Can be "posterior_predictive" or "prior_predictive".
    plot_data : optional
        Whether to plot the observed data, by default True.
    n_samples : optional
        When `dt` is provided, the number or proportion of posterior predictive samples
        randomly drawn to be used from each chain for plotting. When `dt` is not
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
    kind : optional
        The density estimator drawn for every curve, by default "hist":
        - `"hist"`: binned histogram densities (see `bins`, `step`).
        - `"kde"`: Gaussian kernel density estimates evaluated on a shared
          grid (see `grid_points`, `bw_method`). On the signed-RT axis the
          KDE is fit separately per choice side, weighted by response
          proportion, so no density is smoothed across the gap at zero.
    bins : optional
        Specification of hist bins (`kind="hist"` only), by default 50.
        There are three options:
        - A string describing the binning strategy (passed to `np.histogram_bin_edges`).
        - A list-like defining the bin edges.
        - An integer defining the number of bins to be used.
    grid_points : optional
        Number of grid points the KDE is evaluated on (`kind="kde"` only),
        by default 256.
    bw_method : optional
        Bandwidth selection passed to `scipy.stats.gaussian_kde`
        (`kind="kde"` only), by default None (Scott's rule).
    x_range : optional
        The lower and upper range of the bins. If not provided, the range spans the
        1%-99% quantiles of the predictive samples, widened to include all
        observed data (so observations are never silently clipped), by default None.
    step : optional
        Whether to plot the distributions as edge-anchored step histograms (True,
        the default) or as frequency polygons through the bin centers (False).
    hdi : optional
        The uncertainty interval(s) to display when `uncertainty` includes bands,
        by default None, which plots graded 50% and 94% bands. Accepts:
        - A float mass, e.g. `0.9` for a symmetric 90% interval.
        - A percentage string, e.g. `"90%"`.
        - A (lo, hi) tuple of quantiles, e.g. `(0.05, 0.95)`.
        - A list of any of the above, one entry per graded band.
        Intervals are equal-tailed percentile intervals across posterior
        predictive draws (not highest-density intervals). There should be at
        least ~50 predicted trials per draw for the bands to be reliable; a
        warning is logged when there are fewer.
    uncertainty : optional
        How to display uncertainty across posterior predictive draws, by default
        "band":
        - `"band"`: graded quantile ribbons around the predictive mean.
        - `"samples"`: each retained posterior draw as a translucent curve under
          an opaque mean, in the style of `plot_model_cartoon`.
        - `"both"`: sample curves with band ribbons on top.
        - None: only the predictive mean is drawn (the pre-0.5 behavior).
    alpha_mean : optional
        Opacity of the predictive mean curve, by default 1.0.
    alpha_uncertainty : optional
        Opacity of the uncertainty display, by default None, which resolves to
        0.25 for the innermost band (outer bands are scaled down) and an
        automatic per-curve alpha for `"samples"` based on the number of draws.
    colors : optional
        Colors to use for the different levels of the hue variable. When a `str`, the
        color of posterior predictives, in which case an error will be thrown if
        `plot_data` is `True`. When a length-2 iterable, indicates the colors in the
        order of posterior predictives and observed data. When a dict, the keys must
        be 'predicted' and/or 'observed'. The values must be interpretable by
        matplotlib. When None, use the default palette, by default None.
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
        The label for the x-axis, by default "Response Time" — automatically
        annotated as "Response Time (sign = choice)" on 2-choice models, where
        the negative half-axis carries the other choice.
    ylabel : optional
        The label for the y-axis, by default "Density".
    legend : optional
        Whether to draw a legend, by default True. No legend is drawn when the
        plot contains only the predictive mean (`plot_data=False` with
        `uncertainty=None`) — a single-series legend adds no information.
    grid_kwargs : optional
        Additional keyword arguments are passed to the [`sns.FacetGrid` constructor]
        (https://seaborn.pydata.org/generated/seaborn.FacetGrid.html#seaborn.FacetGrid.__init__)
        when any of row or col is provided. When producing a single plot, these
        arguments are ignored.
    kwargs : optional
        Additional keyword arguments passed to ax.plot() functions.

    Returns
    -------
    Axes | sns.FacetGrid | list[sns.FacetGrid]
        The matplotlib `axis` or seaborn `FacetGrid` object containing the plot.
    """
    if kind not in ("hist", "kde"):
        raise ValueError(
            f"`kind` must be 'hist' or 'kde', got {kind!r}. (Did you pass a "
            "positional argument in the wrong slot?)"
        )
    if uncertainty not in ("band", "samples", "both", None):
        raise ValueError(
            "`uncertainty` must be one of 'band', 'samples', 'both', or None, "
            f"got {uncertainty!r}."
        )

    # Process hdi into a list of quantile intervals. `hdi=None` with band-style
    # uncertainty falls back to the default graded 50% + 94% bands; passing
    # `uncertainty=None` switches uncertainty display off entirely.
    intervals: list[tuple[float, float]] | None = None
    if uncertainty in ("band", "both"):
        intervals = _hdi_to_intervals([0.5, 0.94] if hdi is None else hdi)

    # Adapt default text to what is actually being plotted.
    if title == _DEFAULT_TITLE and predictive_group == "prior_predictive":
        title = "Prior Predictive Distribution"

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
            and (dt is not None)
            and ("posterior_predictive" in dt)
        ):
            # Allows data to be None only when plot_data=False and no extra_dims
            # and posterior predictive samples are available
            data = None
        else:
            data = model.data

    dt, sampled = _use_traces_or_sample(
        model, data, dt, n_samples=n_samples, predictive_group=predictive_group
    )

    plotting_df = _get_plotting_df(
        dt,
        data,
        extra_dims=extra_dims,
        n_samples=None if sampled else n_samples,
        response_str=model.response_str,
        predictive_group=predictive_group,
    )

    if uncertainty is not None:
        _check_sample_size(plotting_df)

    # Flip the rt values if necessary
    if np.any(plotting_df["response"] == 0) and model.n_choices == 2:
        plotting_df["response"] = np.where(plotting_df["response"] == 0, -1, 1)
    if model.n_choices == 2:
        plotting_df["rt"] = plotting_df["rt"] * plotting_df["response"]
        # Annotate the signed-RT convention instead of leaving readers to
        # infer what the negative half-axis means.
        if xlabel == _DEFAULT_XLABEL:
            xlabel = "Response Time (sign = choice)"

    # Default x-range: trim extreme predictive tails, but never clip observed
    # data. The old default (min/max of predicted alone) wasted most of the
    # axis on empty tails and silently dropped out-of-range observations.
    if x_range is None:
        predicted_rt = plotting_df.loc[plotting_df["observed"] == "predicted", "rt"]
        lo, hi = predicted_rt.quantile([0.01, 0.99])
        observed_rt = plotting_df.loc[plotting_df["observed"] == "observed", "rt"]
        if plot_data and len(observed_rt) > 0:
            lo = min(lo, observed_rt.min())
            hi = max(hi, observed_rt.max())
        x_range = (float(lo), float(hi))

    # Then, plot the posterior predictive distribution against the observed data
    # Determine whether we are producing a single plot or a grid of plots
    if not extra_dims:
        ax = _plot_predictive_1D(
            data=plotting_df,
            plot_data=plot_data,
            kind=kind,
            bins=bins,
            grid_points=grid_points,
            bw_method=bw_method,
            x_range=x_range,
            step=step,
            interval=intervals,
            uncertainty=uncertainty,
            alpha_mean=alpha_mean,
            alpha_uncertainty=alpha_uncertainty,
            colors=colors,
            linestyles=linestyles_,
            linewidths=linewidths_,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=legend,
            **kwargs,
        )

        return ax

    # The multiple dimensions case
    # If group is not provided, we are producing a grid of plots
    if groups is None:
        g = _plot_predictive_2D(
            data=plotting_df,
            plot_data=plot_data,
            row=row,
            col=col,
            col_wrap=col_wrap,
            kind=kind,
            bins=bins,
            grid_points=grid_points,
            bw_method=bw_method,
            x_range=x_range,
            step=step,
            interval=intervals,
            uncertainty=uncertainty,
            alpha_mean=alpha_mean,
            alpha_uncertainty=alpha_uncertainty,
            colors=colors,
            linestyles=linestyles_,
            linewidths=linewidths_,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=legend,
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
        g = _plot_predictive_2D(
            data=df,
            plot_data=plot_data,
            row=row,
            col=col,
            col_wrap=col_wrap,
            kind=kind,
            bins=bins,
            grid_points=grid_points,
            bw_method=bw_method,
            x_range=x_range,
            step=step,
            interval=intervals,
            uncertainty=uncertainty,
            alpha_mean=alpha_mean,
            alpha_uncertainty=alpha_uncertainty,
            colors=colors,
            linestyles=linestyles_,
            linewidths=linewidths_,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=legend,
            grid_kwargs=grid_kwargs,
            **kwargs,
        )

        plots.append(g)

    return plots

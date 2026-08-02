"""Plotting functionalities for HSSM."""

import logging
from copy import deepcopy
from itertools import product
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from seaborn import FacetGrid
from ssms.basic_simulators.simulator import simulator

# Original model cartoon plot from gui
from ..defaults import SupportedModels, default_model_config
from .predictive import _legend_order, _process_lines
from .utils import (
    DEFAULT_PREDICTIVE_COLORS,
    _band_alphas,
    _check_groups_and_groups_order,
    _curve_xy,
    _get_plotting_df,
    _get_title,
    _subset_df,
    _to_dt_group,
    _use_traces_or_sample,
)

_logger = logging.getLogger("hssm")

TRAJ_COLOR_DEFAULT_DICT: Mapping[Any, str] = MappingProxyType(
    {
        -1: "black",
        0: "black",
        1: "green",
        2: "blue",
        3: "red",
        4: "orange",
        5: "purple",
        6: "brown",
    }
)


class PlotFunctionProtocol(Protocol):
    """Protocol for plot functions."""

    def __call__(self, *args, **kwargs) -> Axes:
        """Plot function."""
        ...


def _cartoon_model_meta(model) -> tuple[list[str], list[int]]:
    """Return ``(list_params, choices)`` for a fitted model.

    Built-in models live in ``default_model_config`` (dict, keyed by the
    ``SupportedModels`` literal). ``HSSMBase`` subclasses (aDDM, RLSSM) are absent
    from that registry — they carry their config on ``model.model_config`` (an
    ``aDDMConfig``/``RLSSMConfig`` dataclass with ``list_params`` + ``choices``), so
    fall back to it. Generalizing the registry to hold subclass models is tracked
    in lnccbrown/HSSM#1031; until then this is the single source of truth.
    """
    cfg = default_model_config.get(model.model_name)
    if cfg is not None:
        return list(cfg["list_params"]), list(cfg["choices"])
    mc = model.model_config
    return list(mc.list_params), list(mc.choices)


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
    list_params: list[str] | None = None,
    choices: list[int] | None = None,
    **kwargs,
) -> Axes:
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
    colors = colors or list(DEFAULT_PREDICTIVE_COLORS)

    if plot_data and isinstance(colors, str):
        raise ValueError("When `plot_data=True`, `colors` must be a list or dict.")

    if "ax" in kwargs:
        ax = kwargs.pop("ax")
    else:
        ax = plt.gca()

    # list_params/choices are resolved by plot_model_cartoon via _cartoon_model_meta
    # (model.model_config) so HSSMBase subclasses (aDDM/RLSSM) — absent from
    # default_model_config — work. Fall back to the registry for built-in models
    # called without them.
    if list_params is None or choices is None:
        config_tmp = default_model_config[cast("SupportedModels", model_name)]
        list_params = list_params or config_tmp["list_params"]
        choices = choices or config_tmp["choices"]
    model_params = list_params

    n_choices = len(choices)

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

    # plot_func_model (2-choice) derives choices from sim metadata; only the
    # >2-choice renderer needs choices threaded in (avoids its registry lookup).
    plot_function_kwargs: dict[str, Any] = (
        {} if n_choices == 2 else {"choices": choices}
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
        **plot_function_kwargs,
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
    list_params: list[str] | None = None,
    choices: list[int] | None = None,
    **kwargs,
) -> FacetGrid:
    """Plot the posterior predictive distribution against the observed data.

    Check the function below for docstring.

    Returns
    -------
    FacetGrid
        A seaborn FacetGrid object containing the plot.
    """
    g = FacetGrid(
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
        list_params=list_params,
        choices=choices,
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
    model, dt, dt_group: Literal["posterior", "prior"] = "posterior"
):
    """Compute the necessary deterministic variables for the model."""
    # Get the list of deterministic variables
    necessary_params, _ = _cartoon_model_meta(model)
    deterministics_list = []
    group_ds = dt[dt_group].to_dataset()
    dt_group_keys = list(group_ds.data_vars)
    # Compute the deterministic variables
    for param in necessary_params:
        if param not in dt_group_keys:
            if param in [
                deterministic.name for deterministic in model.pymc_model.deterministics
            ]:
                deterministics_list.append(
                    pm.compute_deterministics(
                        group_ds, model=model.pymc_model, var_names=[param]
                    )
                )

    deterministics_dt = xr.merge(deterministics_list)
    dt[dt_group] = xr.merge([group_ds, deterministics_dt])
    return dt


def attach_trialwise_params_to_df(
    model, df, dt, dt_group: Literal["posterior", "prior"] = "posterior"
):
    """Attach the trial-wise parameters to the dataframe.

    Handles both trial-wise parameters (shape ``(n_obs,)`` after selecting
    chain/draw, e.g. from regression models) and scalar parameters
    (shape ``(1,)`` after selecting chain/draw, e.g. from intercept-only
    models where Bambi >= 0.17 no longer broadcasts the intercept to all
    observations).
    """
    necessary_params, _ = _cartoon_model_meta(model)
    df[necessary_params] = 0.0

    group_ds = dt[dt_group].to_dataset()
    for chain_tmp, draw_tmp in {(x[0], x[1]) for x in list(df.index) if x[0] != -1}:
        n_rows = len(df.loc[(chain_tmp, draw_tmp, slice(None))])
        for param in necessary_params:
            values = group_ds.sel(chain=chain_tmp, draw=draw_tmp)[param].values
            # Handle scalar / intercept-only params that don't have the
            # full observation dimension. Bambi >= 0.17 returns shape (1,)
            # for intercept-only deterministics; directly-sampled scalars
            # come back as 0-dim arrays.
            if values.ndim == 0 or values.shape[0] != n_rows:
                values = np.broadcast_to(values.flat[0], (n_rows,))
            df.loc[(chain_tmp, draw_tmp, slice(None)), param] = values
    return df


def _make_dt_mean_group(
    dt: xr.DataTree, group: Literal["posterior", "prior"]
) -> xr.DataTree:
    """Create a new DataTree containing only the mean of a given group.

    Takes a DataTree object and computes the mean across chains and draws of the
    requested group (``"posterior"`` or ``"prior"``), then restructures it to have
    a single chain and draw. Removes the corresponding predictive samples if present.

    Parameters
    ----------
    dt : xarray.DataTree
        The DataTree object to process.
    group : {"posterior", "prior"}
        The group to reduce to its mean.

    Returns
    -------
    xarray.DataTree
        A new DataTree object containing only the mean of the requested group.
    """
    group_ds = dt[group].to_dataset().mean(dim=["chain", "draw"])
    group_ds = group_ds.assign_coords(chain=[0], draw=[0])
    for data_var in list(group_ds.data_vars):
        group_ds[data_var] = group_ds[data_var].expand_dims(
            dim=["chain", "draw"], axis=[0, 1]
        )
    dt[group] = group_ds

    predictive_group = group + "_predictive"
    if predictive_group in dt:
        del dt[predictive_group]
    return dt


def _make_dt_mean_posterior(dt: xr.DataTree) -> xr.DataTree:
    """Create a new DataTree object containing only the posterior mean."""
    return _make_dt_mean_group(dt, "posterior")


def _make_dt_mean_prior(dt: xr.DataTree) -> xr.DataTree:
    """Create a new DataTree object containing only the prior mean."""
    return _make_dt_mean_group(dt, "prior")


def plot_model_cartoon(
    model,
    dt: xr.DataTree | None = None,
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
) -> Axes | FacetGrid | list[FacetGrid]:
    """Plot the posterior predictive distribution against the observed data.

    Note (aDDM / covariate models): this cartoon re-simulates at the posterior-mean
    parameters with the simulator self-sampling its own fixation sequence (Mode 1)
    and regenerates its predictive with the model's default continuation policy. It
    does NOT condition on the observed fixations, and a ``continuation_mode`` passed
    to ``sample_posterior_predictive`` does not reach it, so it is a schematic of the
    fitted drift/boundary geometry, not the fixation-conditioned predictive check.
    Tracked in lnccbrown/HSSM#1039.

    Parameters
    ----------
    model : hssm.HSSM
        A fitted HSSM model.
    dt : optional
        The DataTree object with posterior samples. If not provided, will use the
        traces object stored inside the model. If posterior predictive samples are not
        present in this object, will generate posterior predictive samples using the
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
        When dt is provided, the number or proportion of predictive samples
        randomly drawn to be used from each chain for plotting. When dt is not
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
        Additional keyword arguments are passed to the [`FacetGrid` constructor]
        (https://seaborn.pydata.org/generated/seaborn.FacetGrid.html#seaborn.FacetGrid.__init__)
        when any of row or col is provided. When producing a single plot, these
        arguments are ignored.
    kwargs : optional
        Additional keyword arguments passed to ax.plot() functions.

    Returns
    -------
    Axes | FacetGrid | list[FacetGrid]
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
            and (dt is not None)
            and (predictive_group in dt)
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
            dt_mean = _make_dt_mean_posterior(
                deepcopy(model.traces if dt is None else dt)
            )
            dt_mean, _ = _use_traces_or_sample(
                model,
                data,
                dt_mean,
                n_samples=None,
                predictive_group="posterior_predictive",
            )
        else:
            # Need to sample from prior predictive here to get samples from the prior
            dt, _ = _use_traces_or_sample(
                model,
                data,
                dt,
                n_samples=n_samples_prior,
                predictive_group=predictive_group,
            )

            dt_mean = _make_dt_mean_prior(deepcopy(dt))
            # AF-COMMENT: This is a hack to get the prior predictive mean
            # we should find a better way to do this eventually.
            mean_dt = xr.DataTree.from_dict(
                {"posterior": dt_mean["prior"].to_dataset()}
            )
            dt_mean_tmp = model.predict(
                idata=mean_dt,  # bambi.Model.predict still uses idata
                kind="response",
                inplace=False,
            )

            dt_mean["prior_predictive"] = dt_mean_tmp[
                "posterior_predictive"
            ].to_dataset()

        # # Get the plotting dataframe by chain and sample
        plotting_df_mean = _get_plotting_df(
            dt_mean,
            data,
            extra_dims=extra_dims,
            n_samples=None,
            response_str=model.response_str,
            predictive_group=predictive_group,
        )

        # Get plotting dataframe for predictive mean
        # df by chain and sample
        dt_mean = compute_merge_necessary_deterministics(
            model, dt_mean, dt_group=_to_dt_group(predictive_group)
        )

        plotting_df_mean = attach_trialwise_params_to_df(
            model,
            plotting_df_mean,
            dt_mean,
            dt_group=_to_dt_group(predictive_group),
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
        dt, sampled = _use_traces_or_sample(
            model, data, dt, n_samples=n_samples, predictive_group=predictive_group
        )

        # Get the plotting dataframe by chain and sample
        plotting_df = _get_plotting_df(
            dt,
            data,
            extra_dims=extra_dims,
            n_samples=None if sampled else n_samples,
            response_str=model.response_str,
            predictive_group=predictive_group,
        )

        # Get plotting dataframe for posterior mean
        # df by chain and sample
        dt = compute_merge_necessary_deterministics(
            model,
            dt,
            dt_group=_to_dt_group(predictive_group),
        )

        plotting_df = attach_trialwise_params_to_df(
            model,
            plotting_df,
            dt,
            dt_group=_to_dt_group(predictive_group),
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

    if plotting_df is None:
        raise ValueError(
            "At least one of `plot_predictive_samples` or `plot_predictive_mean` "
            "must be True."
        )

    # return plotting_df

    # Then, plot the posterior predictive distribution against the observed data
    # Determine whether we are producing a single plot or a grid of plots

    # Resolve (list_params, choices) once from the model so subclass models
    # (aDDM/RLSSM, absent from default_model_config) render like built-ins.
    _list_params, _choices = _cartoon_model_meta(model)

    if not extra_dims:
        ax = _plot_model_cartoon_1D(
            data=plotting_df,
            model_name=model.model_name,
            list_params=_list_params,
            choices=_choices,
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
            list_params=_list_params,
            choices=_choices,
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
            list_params=_list_params,
            choices=_choices,
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
    n_reps: int = 10,
    bin_size: float = 0.05,
    bins: int | np.ndarray | str | None = None,
    step: bool = True,
    uncertainty: Literal["band", "samples", "both"] | None = None,
    intervals: list[tuple[float, float]] | None = None,
    hist_height: float | None = None,
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
    linestyle_histogram: str = "-",
    linestyle_histogram_data: str | None = None,
    linewidth_histogram_data: float | None = None,
    color_data: str = "blue",
    color_predictive_mean: str = "black",
    color_predictive: str = "black",
    alpha_mean: float = 1,
    alpha_predictive: float | None = None,
    alpha_uncertainty: float | None = None,
    alpha_trajectories: float = 0.5,
    legend: bool = True,
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
    n_reps : int, optional
        Number of simulation repetitions per parameter row (previously named
        ``n_samples``, which shadowed the public draw-count parameter of
        ``plot_model_cartoon``). Defaults to 10.
    bin_size : float, optional
        Spacing in seconds of the legacy histogram grid, used when `bins` is
        None. Defaults to 0.05.
    bins : int, array-like, or str, optional
        Histogram bins for the RT histograms, as accepted by
        ``np.histogram_bin_edges``. When provided, wins over `bin_size` and is
        resolved on the pooled simulated + observed RTs. Defaults to None
        (legacy `bin_size` grid).
    step : bool, optional
        If True (default), draw histogram curves as edge-anchored step
        outlines; if False, as frequency polygons through the bin centers.
    uncertainty : {"band", "samples", "both"} or None, optional
        How posterior predictive uncertainty is displayed (same vocabulary as
        ``plot_predictive``). ``"band"`` draws graded pointwise quantile
        bands, ``"samples"`` per-draw curves, ``"both"`` combines them. None
        (default here) keeps the legacy rendering: a plug-in mean when
        `theta_mean` is given and per-draw overlays at `alpha_predictive`
        when `theta_samples` is given.
    intervals : list of (float, float), optional
        Pre-resolved quantile pairs for the bands, sorted widest-first (the
        output of ``_hdi_to_intervals``). Required when `uncertainty` is
        "band" or "both".
    hist_height : float, optional
        If given, rescale all histogram curves by one common factor so the
        tallest drawn curve has this height (in y-data units above its
        bound). Defaults to None (raw defective-density units).
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
        Width of predictive histogram lines. Defaults to 1.5.
    linewidth_model : float or int, optional
        Width of model lines. Defaults to 1.5.
    linestyle_histogram : str, optional
        Line style of predictive histogram curves. Defaults to "-".
    linestyle_histogram_data : str, optional
        Line style of the observed-data histogram. Defaults to None (same as
        `linestyle_histogram`).
    linewidth_histogram_data : float, optional
        Width of the observed-data histogram lines. Defaults to None (same as
        `linewidth_histogram`).
    color_data : str, optional
        Color for data histogram. Defaults to "blue".
    color_predictive_mean : str, optional
        Color for posterior mean. Defaults to "black".
    color_predictive : str, optional
        Color for posterior samples. Defaults to "black".
    alpha_mean : float, optional
        Transparency of posterior mean. Defaults to 1.
    alpha_predictive : float, optional
        Deprecated alias for `alpha_uncertainty`, kept for GUI-heritage
        callers. In the legacy rendering (``uncertainty=None``) it sets the
        per-draw overlay opacity (default 0.05).
    alpha_uncertainty : float, optional
        Opacity of the uncertainty layer. None (default) resolves to 0.25 for
        bands and an automatic per-curve value for sample curves; an explicit
        0.0 is honored.
    alpha_trajectories : float, optional
        Transparency of trajectories. Defaults to 0.5.
    legend : bool, optional
        Whether to assemble a legend from the labeled artists on the main
        axis and the upper twin axis. Defaults to True.
    **kwargs
        Additional arguments passed to plotting functions.

    Returns
    -------
    matplotlib.axes.Axes
        The axis with the model cartoon plot.
    """
    if "n_samples" in kwargs:
        _logger.warning(
            "plot_func_model(n_samples=...) was renamed to n_reps= (simulation "
            "repetitions per parameter row); the old name will stop working in "
            "a future release."
        )
        n_reps = kwargs.pop("n_samples")
    if "hist_histtype" in kwargs:
        _logger.warning(
            "hist_histtype is no longer used: histogram curves are drawn from "
            "precomputed densities (see the `step` parameter) and the argument "
            "is ignored."
        )
        kwargs.pop("hist_histtype")

    # `alpha_predictive` is the deprecated spelling; map it onto the new knob
    # when the caller has not set `alpha_uncertainty` directly.
    if alpha_predictive is not None and alpha_uncertainty is None:
        alpha_uncertainty = alpha_predictive
    legacy_alpha = alpha_predictive if alpha_predictive is not None else 0.05
    legacy = uncertainty is None

    ylim_low, ylim_high = kwargs.get("ylims", (-3, 3))
    xlim_low, xlim_high = kwargs.get("xlims", (-0.05, 5))

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
            n_samples=n_reps,
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
                n_samples=n_reps,
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
        else:  # pragma: no cover
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

    axis.set_xlim(xlim_low, xlim_high)
    axis.set_ylim(ylim_low, ylim_high)
    axis_twin_up: Axes = axis.twinx()
    axis_twin_down: Axes = axis.twinx()
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

    # Shared bin edges for every histogram in this facet. `bins=None` keeps
    # the legacy bin_size grid bit-for-bit; an explicit `bins` wins and is
    # resolved on the pooled RTs that will actually be drawn.
    if bins is None:
        bin_edges = np.arange(xlim_low, xlim_high, bin_size)
    else:
        if bin_size != 0.05:
            _logger.warning("Both `bins` and `bin_size` were provided; `bins` wins.")
        pooled: list[np.ndarray] = []
        if theta_mean is not None and sim_out is not None:
            rts = np.asarray(sim_out["rts"]).ravel()
            pooled.append(np.abs(rts[rts != -999]))
        if theta_samples is not None:
            for sim_out_tmp in posterior_pred_sims.values():
                rts = np.asarray(sim_out_tmp["rts"]).ravel()
                pooled.append(np.abs(rts[rts != -999]))
        if data is not None:
            obs_rts = data["rt"].to_numpy()
            pooled.append(np.abs(obs_rts[obs_rts != -999]))
        bin_edges = np.histogram_bin_edges(
            np.concatenate(pooled) if pooled else np.array([0.0, 1.0]),
            bins=bins,
            range=(max(xlim_low, 0.0), xlim_high),
        )

    # Per-draw density matrices (and the observed densities) in raw defective-
    # density units. `scale` maps them all with one common factor so up/down
    # and predicted/observed stay comparable.
    densities_up: np.ndarray | None = None
    densities_down: np.ndarray | None = None
    if not legacy and theta_samples is not None:
        densities_up, densities_down = _density_matrices(posterior_pred_sims, bin_edges)
        if densities_up.shape[0] < 10 and uncertainty in ("band", "both"):
            _logger.warning(
                "Quantile bands over fewer than 10 posterior draws are noisy; "
                "consider a larger `n_samples`."
            )
    plugin_up: np.ndarray | None = None
    plugin_down: np.ndarray | None = None
    if theta_mean is not None and sim_out is not None:
        plugin_up, plugin_down = _density_matrices({0: sim_out}, bin_edges)

    obs_up: np.ndarray | None = None
    obs_down: np.ndarray | None = None
    if data is not None:
        obs_up, obs_down = _defective_densities(
            data["rt"].to_numpy(), data["response"].to_numpy(), bin_edges
        )

    if hist_height is not None:
        peak = max(
            (
                m.max()
                for m in (densities_up, densities_down, plugin_up, plugin_down)
                if m is not None
            ),
            default=0.0,
        )
        if obs_up is not None and obs_down is not None:
            peak = max(peak, obs_up.max(), obs_down.max())
        hist_scale = hist_height / peak if peak > 0 else 1.0
    else:
        hist_scale = 1.0
    axis_twin_up.set_zorder(0)
    axis_twin_down.set_zorder(0)

    # Predictive histogram layers. The two twins get identical calls — the
    # lower twin's inverted ylim renders its curves downward from the bound.
    twin_sides = (
        (axis_twin_up, hist_bottom_high, True),
        (axis_twin_down, hist_bottom_low, False),
    )
    if not legacy:
        band_matrices = (
            (densities_up, densities_down)
            if densities_up is not None and densities_down is not None
            else (plugin_up, plugin_down)
        )
        if band_matrices[0] is not None and band_matrices[1] is not None:
            for ax_twin, bottom, is_up in twin_sides:
                _add_uncertain_histograms(
                    ax_twin,
                    bin_edges,
                    bottom,
                    band_matrices[0] if is_up else band_matrices[1],
                    uncertainty=uncertainty,
                    intervals=intervals,
                    step=step,
                    scale=hist_scale,
                    color=color_predictive,
                    linestyle=linestyle_histogram,
                    linewidth=linewidth_histogram,
                    alpha_mean=alpha_mean,
                    alpha_uncertainty=alpha_uncertainty,
                    add_labels=is_up,
                )
    else:
        # Legacy rendering: plug-in mean curve, plus per-draw overlays at the
        # old fixed alpha when samples are given.
        if plugin_up is not None and plugin_down is not None:
            for ax_twin, bottom, is_up in twin_sides:
                _add_uncertain_histograms(
                    ax_twin,
                    bin_edges,
                    bottom,
                    plugin_up if is_up else plugin_down,
                    uncertainty=None,
                    intervals=None,
                    step=step,
                    scale=hist_scale,
                    color=color_predictive_mean,
                    linestyle=linestyle_histogram,
                    linewidth=linewidth_histogram,
                    alpha_mean=alpha_mean,
                    alpha_uncertainty=None,
                    add_labels=is_up,
                )
        if theta_samples is not None:
            legacy_up, legacy_down = _density_matrices(posterior_pred_sims, bin_edges)
            for ax_twin, bottom, is_up in twin_sides:
                _add_uncertain_histograms(
                    ax_twin,
                    bin_edges,
                    bottom,
                    legacy_up if is_up else legacy_down,
                    uncertainty="samples",
                    intervals=None,
                    step=step,
                    scale=hist_scale,
                    color=color_predictive,
                    linestyle=linestyle_histogram,
                    linewidth=linewidth_histogram,
                    alpha_mean=alpha_mean,
                    alpha_uncertainty=legacy_alpha,
                    add_labels=is_up,
                    draw_mean=False,
                )

    # Observed data — always a plain line on top of the stack, never banded.
    if obs_up is not None and obs_down is not None:
        ls_data = (
            linestyle_histogram_data
            if linestyle_histogram_data is not None
            else linestyle_histogram
        )
        lw_data = (
            linewidth_histogram_data
            if linewidth_histogram_data is not None
            else linewidth_histogram
        )
        for ax_twin, bottom, is_up in twin_sides:
            x_o, y_o = _curve_xy(bin_edges, obs_up if is_up else obs_down, step)
            ax_twin.plot(
                x_o,
                bottom + hist_scale * y_o,
                color=color_data,
                linestyle=ls_data,
                linewidth=lw_data,
                zorder=4,
                label="observed" if is_up else "_nolegend_",
            )

    z_cnt = 0  # controlling the order of elements in plot

    if theta_samples is not None:
        # ADD MODEL CARTOONS:
        t_s = np.arange(
            0, posterior_pred_no_noise[0]["metadata"]["max_t"], delta_t_model
        )

        if legacy:
            # Legacy rendering: one full five-artist cartoon per draw,
            # including the per-draw ndt axvlines.
            for _, sim_out_tmp in posterior_pred_no_noise.items():
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
                    alpha=legacy_alpha,
                    lw_m=linewidth_model,
                    ylim_low=ylim_low,
                    ylim_high=ylim_high,
                    t_s=t_s,
                    color=color_predictive,
                    zorder_cnt=z_cnt,
                )
                z_cnt += 1
        else:
            b_high_m, b_low_m, drifts_m, ndts, z_abs = _geometry_arrays(
                posterior_pred_no_noise, t_s, delta_t_model
            )
            n_geom_draws = b_high_m.shape[0]
            geom_spaghetti_alpha = (
                alpha_uncertainty
                if alpha_uncertainty is not None
                else float(np.clip(4.0 / n_geom_draws, 0.02, 0.25))
            )
            geom_base_alpha = (
                alpha_uncertainty if alpha_uncertainty is not None else 0.25
            )
            ndt_ref = (
                float(np.asarray(sim_out_no_noise["metadata"]["t"]).ravel()[0])
                if theta_mean is not None
                else float(np.mean(ndts))
            )

            if uncertainty in ("samples", "both"):
                # Per-draw geometry spaghetti. The ndt axvlines are replaced
                # by the rug below — N near-vertical dashed lines were the
                # single worst clutter contributor of the old rendering.
                for _, sim_out_tmp in posterior_pred_no_noise.items():
                    _add_model_cartoon_to_ax(
                        sample=sim_out_tmp,
                        axis=axis,
                        keep_slope=keep_slope,
                        keep_boundary=keep_boundary,
                        keep_ndt=False,
                        keep_starting_point=keep_starting_point,
                        markersize_starting_point=markersize_starting_point,
                        markertype_starting_point=markertype_starting_point,
                        markershift_starting_point=markershift_starting_point,
                        delta_t_graph=delta_t_model,
                        alpha=geom_spaghetti_alpha,
                        lw_m=0.6 * linewidth_model,
                        ylim_low=ylim_low,
                        ylim_high=ylim_high,
                        t_s=t_s,
                        color=color_predictive,
                        zorder_cnt=0,
                    )

            if uncertainty in ("band", "both") and intervals:
                if keep_boundary:
                    _render_boundary_bands(
                        axis,
                        t_s,
                        b_high_m,
                        b_low_m,
                        intervals,
                        geom_base_alpha,
                        color_predictive,
                    )
                if keep_slope:
                    _render_drift_band(
                        axis,
                        t_s,
                        drifts_m,
                        ndt_ref,
                        intervals,
                        geom_base_alpha,
                        color_predictive,
                        linewidth_model,
                    )
                if keep_starting_point:
                    _render_start_uncertainty(
                        axis,
                        ndt_ref,
                        z_abs,
                        intervals,
                        uncertainty,
                        geom_base_alpha,
                        markershift_starting_point,
                        linewidth_model,
                        color_predictive,
                    )
            if keep_ndt:
                _render_ndt_uncertainty(
                    axis,
                    ndts,
                    intervals,
                    uncertainty,
                    geom_base_alpha,
                    geom_spaghetti_alpha,
                    color_predictive,
                    ylim_low,
                    ylim_high,
                )

    if theta_mean is not None:
        t_s = np.arange(0, sim_out_no_noise["metadata"]["max_t"], delta_t_model)
        # Model cartoon for the reference geometry. In uncertainty modes it
        # sits above every band/spaghetti layer (zorder 1030); in legacy mode
        # it keeps its historical slot just above the per-draw cartoons.
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
            zorder_cnt=(z_cnt + 1) if legacy else 30,
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

    # Legend: harvest labeled artists from the main axis and the upper twin
    # (the lower twin labels everything "_nolegend_"), order them with the
    # shared predictive-family ranking, and dedupe by label keeping the first.
    if legend:
        handles: list = []
        labels: list[str] = []
        for ax_ in (axis, axis_twin_up):
            h_, l_ = ax_.get_legend_handles_labels()
            handles.extend(h_)
            labels.extend(l_)
        if labels:
            seen: set[str] = set()
            ordered = [
                i
                for i in _legend_order(labels)
                if not (labels[i] in seen or seen.add(labels[i]))
            ]
            axis.legend(
                [handles[i] for i in ordered],
                [labels[i] for i in ordered],
                frameon=False,
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
    markercolor_rt_choice: str | list[str] | Mapping[Any, str] = "red",
    linewidth: float | int = 1,
    alpha: float | int = 0.5,
    colors: str | list[str] | Mapping[Any, str] | None = None,
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
    colors : str, list, dict, or None, optional
        Color(s) for trajectories. Can be a single color, list of colors,
        or dict mapping choices to colors. When ``None`` (default), uses
        ``TRAJ_COLOR_DEFAULT_DICT``.
    **kwargs
        Additional keyword arguments passed to plotting functions.
    """
    # Check markercolor type
    possible_choices = sample[0]["metadata"]["possible_choices"]
    if isinstance(markercolor_rt_choice, str):
        markercolor_rt_choice_dict: Mapping[Any, str] = {
            value_: markercolor_rt_choice for value_ in possible_choices
        }
    elif isinstance(markercolor_rt_choice, list):
        if len(markercolor_rt_choice) < len(possible_choices):
            raise ValueError(
                "The `markercolor_rt_choice` list must have at least as many "
                f"entries as possible choices ({len(possible_choices)}), "
                f"but got {len(markercolor_rt_choice)}."
            )
        markercolor_rt_choice_dict = {
            value_: markercolor_rt_choice[cnt]
            for cnt, value_ in enumerate(possible_choices)
        }
    elif isinstance(markercolor_rt_choice, Mapping):
        markercolor_rt_choice_dict = markercolor_rt_choice
    else:
        raise ValueError(
            "The `markercolor_rt_choice` argument must be a string, list, or mapping."
        )

    # Check trajectory color type
    if colors is None:
        colors_dict: Mapping[Any, str] = dict(TRAJ_COLOR_DEFAULT_DICT)
    elif isinstance(colors, str):
        colors_dict = {value_: colors for value_ in possible_choices}
    elif isinstance(colors, list):
        if len(colors) < len(possible_choices):
            raise ValueError(
                "The `colors` list must have at least as many entries as "
                f"possible choices ({len(possible_choices)}), "
                f"but got {len(colors)}."
            )
        colors_dict = {
            value_: colors[cnt] for cnt, value_ in enumerate(possible_choices)
        }
    elif isinstance(colors, Mapping):
        colors_dict = colors
    else:
        raise ValueError("The `colors` argument must be a string, list, or mapping.")

    # Make bounds
    (b_high, b_low) = (
        np.maximum(sample[0]["metadata"]["boundary"], 0),
        np.minimum((-1) * sample[0]["metadata"]["boundary"], 0),
    )

    b_h_init = b_high[0]
    b_l_init = b_low[0]
    # .item() extracts a Python scalar; required since NumPy >= 1.25 deprecated
    # implicit ndim>0-to-scalar conversion inside int()/float().
    n_roll = int((sample[0]["metadata"]["t"][0] / delta_t_graph + 1).item())
    b_high = np.roll(b_high, n_roll)
    b_high[:n_roll] = b_h_init
    b_low = np.roll(b_low, n_roll)
    b_low[:n_roll] = b_l_init

    # Trajectories
    for i in range(n):
        metadata = sample[i]["metadata"]
        tmp_traj = metadata["trajectory"]
        tmp_traj_choice = float(sample[i]["choices"].flatten().item())
        maxid = np.minimum(np.argmax(np.where(tmp_traj > -999)), t_s.shape[0])

        axis.plot(
            t_s[:maxid] + metadata["t"][0],  # sample.t.values[0],
            tmp_traj[:maxid],
            color=colors_dict[tmp_traj_choice],
            alpha=alpha,
            linewidth=linewidth,
            zorder=2000 + i,
        )

        # Identify boundary value at timepoint of crossing
        if (maxid + n_roll < len(b_high)) and (maxid + n_roll < len(b_low)):
            b_tmp = (
                b_high[maxid + n_roll] if tmp_traj_choice > 0 else b_low[maxid + n_roll]
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
        else:
            _logger.warning(
                "Timepoint of crossing is out of bounds for boundary values. "
                "Skipping boundary value highlighting."
            )


def _defective_densities(
    rts: np.ndarray,
    choices: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-side defective RT densities for one set of simulated trials.

    The upper density integrates to P(choice == 1 | valid) and the lower to
    the complement, so the two sides jointly integrate to 1 — the invariant
    the old per-point ``ax.hist`` weights encoded. Deadline misses (``-999``)
    are excluded from both the numerator and the denominator, matching the
    previous behavior. ``np.diff(bin_edges)`` generalizes the uniform
    ``bin_size`` assumption so user-supplied edge arrays work.
    """
    rts = np.asarray(rts).ravel()
    choices = np.asarray(choices).ravel()
    valid = rts != -999
    up = np.abs(rts[valid & (choices == 1)])
    down = np.abs(rts[valid & (choices != 1)])
    n_total = up.size + down.size
    n_bins = len(bin_edges) - 1
    if n_total == 0:
        return np.zeros(n_bins), np.zeros(n_bins)
    widths = np.diff(bin_edges)
    dens_up = np.histogram(up, bins=bin_edges)[0] / (n_total * widths)
    dens_down = np.histogram(down, bins=bin_edges)[0] / (n_total * widths)
    return dens_up, dens_down


def _density_matrices(
    sim_outs: Mapping[int, dict],
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-draw defective densities into ``(n_draws, n_bins)`` matrices.

    One row per posterior draw's simulation output, on the shared bin edges —
    the substrate for pointwise mean and quantile-band computation.
    """
    ups, downs = zip(
        *(
            _defective_densities(s["rts"], s["choices"], bin_edges)
            for s in sim_outs.values()
        )
    )
    return np.vstack(ups), np.vstack(downs)


def _add_uncertain_histograms(
    ax: Axes,
    bin_edges: np.ndarray,
    bottom: float,
    densities: np.ndarray,
    *,
    uncertainty: Literal["band", "samples", "both"] | None,
    intervals: list[tuple[float, float]] | None,
    step: bool,
    scale: float,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha_mean: float,
    alpha_uncertainty: float | None,
    add_labels: bool,
    draw_mean: bool = True,
) -> None:
    """Draw one side's RT histogram layers onto a twin axis.

    Renders the #1123 layer stack — per-draw sample curves (zorder 1), graded
    quantile bands (2), and the mean curve (3) — from a precomputed
    ``(n_draws, n_bins)`` density matrix. All y values are ``bottom +
    scale * density``; the inverted ylim on the lower twin handles mirroring,
    so this function is identical for both sides. Called exactly twice per
    facet; ``add_labels`` is True only for the upper twin so each layer gets
    one legend entry.
    """
    n_draws = densities.shape[0]

    def label(text: str) -> str:
        return text if add_labels else "_nolegend_"

    # Layer 1 — per-draw sample curves ("spaghetti").
    if uncertainty in ("samples", "both"):
        alpha_samples = (
            alpha_uncertainty
            if alpha_uncertainty is not None
            else float(np.clip(4.0 / n_draws, 0.02, 0.25))
        )
        first = True
        for row in densities:
            x, y = _curve_xy(bin_edges, row, step)
            ax.plot(
                x,
                bottom + scale * y,
                color=color,
                linestyle=linestyle,
                linewidth=0.6 * linewidth,
                alpha=alpha_samples,
                zorder=1,
                label=label("predictive draws") if first else "_nolegend_",
            )
            first = False

    # Layer 2 — graded uncertainty bands, widest first so narrower bands
    # print on top with higher opacity.
    if uncertainty in ("band", "both") and intervals:
        base_alpha = alpha_uncertainty if alpha_uncertainty is not None else 0.25
        band_alphas = _band_alphas(base_alpha, len(intervals))
        for (lo, hi), band_alpha in zip(intervals, band_alphas):
            x_b, y_lo = _curve_xy(bin_edges, np.quantile(densities, lo, axis=0), step)
            _, y_hi = _curve_xy(bin_edges, np.quantile(densities, hi, axis=0), step)
            ax.fill_between(
                x_b,
                bottom + scale * y_lo,
                bottom + scale * y_hi,
                color=color,
                alpha=float(band_alpha),
                linewidth=0,
                zorder=2,
                label=label(f"{hi - lo:.0%} interval"),
            )

    # Layer 3 — the predictive mean. With uncertainty on this is the pointwise
    # mean across draws (it cannot exit its own bands); with uncertainty off
    # the caller passes the single plug-in density as the only row. The legacy
    # per-draw overlay path sets `draw_mean=False` because its mean curve, if
    # any, comes from a separate plug-in call.
    if draw_mean:
        x_m, y_m = _curve_xy(bin_edges, densities.mean(axis=0), step)
        ax.plot(
            x_m,
            bottom + scale * y_m,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha_mean,
            zorder=3,
            label=label("predicted"),
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
    # .item(): NumPy >= 1.25 deprecated implicit ndim>0-to-scalar conversion.
    n_roll = int((sample["metadata"]["t"][0] / delta_t_graph + 1).item())
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


def _geometry_arrays(
    sims: Mapping[int, dict],
    t_s: np.ndarray,
    delta_t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-draw geometry into matrices on the shared time grid.

    Returns ``(b_high, b_low, drifts, ndts, z_abs)``. Boundaries are rolled by
    each draw's own non-decision time (the same roll `_add_model_cartoon_to_ax`
    applies), so pointwise quantiles across rows describe "where the bound can
    be at absolute time t". Drift paths stay in decision time (their natural
    frame) and are NaN beyond each draw's absorption point so
    ``np.nanquantile`` sees only live draws.
    """
    n_t = t_s.shape[0]
    n_draws = len(sims)
    b_high_m = np.empty((n_draws, n_t))
    b_low_m = np.empty((n_draws, n_t))
    drifts = np.full((n_draws, n_t), np.nan)
    ndts = np.empty(n_draws)
    z_abs = np.empty(n_draws)

    for i, sample in sims.items():
        meta = sample["metadata"]
        b_high = np.maximum(meta["boundary"], 0)
        b_low = np.minimum((-1) * meta["boundary"], 0)
        b_h_init = b_high[0]
        b_l_init = b_low[0]
        n_roll = int((meta["t"][0] / delta_t + 1).item())
        b_high = np.roll(b_high, n_roll)
        b_high[:n_roll] = b_h_init
        b_low = np.roll(b_low, n_roll)
        b_low[:n_roll] = b_l_init
        b_high_m[i] = b_high[:n_t]
        b_low_m[i] = b_low[:n_t]

        traj = np.asarray(meta["trajectory"]).ravel()
        maxid = int(np.minimum(np.argmax(np.where(traj > -999)), n_t))
        drifts[i, :maxid] = traj[:maxid]

        ndts[i] = float(np.asarray(meta["t"]).ravel()[0])
        z_val = float(np.asarray(meta["z"]).ravel()[0])
        z_abs[i] = b_l_init + z_val * (b_h_init - b_l_init)

    return b_high_m, b_low_m, drifts, ndts, z_abs


def _render_boundary_bands(
    axis: Axes,
    t_s: np.ndarray,
    b_high: np.ndarray,
    b_low: np.ndarray,
    intervals: list[tuple[float, float]],
    base_alpha: float,
    color: str,
) -> None:
    """Graded fan-chart ribbons around the upper and lower decision bounds.

    One ``fill_between`` ribbon per interval per bound — a band *around* the
    boundary curve, not a fill to zero — drawn widest-first so narrower bands
    print on top with higher opacity.
    """
    band_alphas = _band_alphas(base_alpha, len(intervals))
    for (lo, hi), band_alpha in zip(intervals, band_alphas):
        axis.fill_between(
            t_s,
            np.quantile(b_high, lo, axis=0),
            np.quantile(b_high, hi, axis=0),
            color=color,
            alpha=float(band_alpha),
            linewidth=0,
            zorder=1010,
            label=f"boundary {hi - lo:.0%} interval",
        )
        axis.fill_between(
            t_s,
            np.quantile(b_low, lo, axis=0),
            np.quantile(b_low, hi, axis=0),
            color=color,
            alpha=float(band_alpha),
            linewidth=0,
            zorder=1010,
            label="_nolegend_",
        )


def _render_drift_band(
    axis: Axes,
    t_s: np.ndarray,
    drifts: np.ndarray,
    ndt_ref: float,
    intervals: list[tuple[float, float]],
    base_alpha: float,
    color: str,
    linewidth: float,
) -> None:
    """Open drift cone: dashed pointwise quantile paths of the widest interval.

    Two thin dashed lines instead of a filled wedge — a fill in the middle of
    the axis would stack its alpha over the boundary ribbons, trajectories,
    and histograms, recreating exactly the occlusion this redesign removes.
    Quantiles are taken in decision time (each draw's path starts at its own
    t=0) and the cone is truncated once fewer than half the draws are still
    unabsorbed, so the tail is never computed from a vanishing subsample. The
    cone is plotted shifted by the reference geometry's non-decision time.
    """
    alive = ~np.isnan(drifts)
    keep = alive.mean(axis=0) >= 0.5
    if not keep.any():
        return
    last = int(keep.size if keep.all() else np.argmin(keep))
    if last < 2:
        return
    lo, hi = intervals[0]  # widest-first ordering
    q_lo = np.nanquantile(drifts[:, :last], lo, axis=0)
    q_hi = np.nanquantile(drifts[:, :last], hi, axis=0)
    tt = t_s[:last] + ndt_ref
    line_alpha = min(1.0, 2.0 * base_alpha)
    for values, label in ((q_lo, "drift interval"), (q_hi, "_nolegend_")):
        axis.plot(
            tt,
            values,
            color=color,
            linestyle="--",
            linewidth=0.75 * linewidth,
            alpha=line_alpha,
            zorder=1012,
            label=label,
        )


def _render_ndt_uncertainty(
    axis: Axes,
    ndts: np.ndarray,
    intervals: list[tuple[float, float]] | None,
    mode: Literal["band", "samples", "both"],
    base_alpha: float,
    spaghetti_alpha: float,
    color: str,
    ylim_low: float,
    ylim_high: float,
) -> None:
    """Non-decision-time uncertainty: graded axvspans and/or a rug of ticks.

    Replaces the per-draw ``axvline`` smear (N near-identical dashed verticals
    was the single worst clutter contributor). In band mode the spans collapse
    onto the reference line when ndt uncertainty is tiny — a slight visual
    thickening, which is the honest rendering of "almost no uncertainty". The
    rug is a single ``Line2D`` of tick markers near the bottom axis edge.
    """
    if mode in ("band", "both") and intervals:
        band_alphas = _band_alphas(base_alpha, len(intervals))
        for (lo, hi), band_alpha in zip(intervals, band_alphas):
            axis.axvspan(
                float(np.quantile(ndts, lo)),
                float(np.quantile(ndts, hi)),
                color=color,
                alpha=float(band_alpha),
                linewidth=0,
                zorder=1002,
                label="_nolegend_",
            )
    if mode in ("samples", "both"):
        rug_y = ylim_low + 0.03 * (ylim_high - ylim_low)
        axis.plot(
            ndts,
            np.full_like(ndts, rug_y),
            linestyle="none",
            marker="|",
            markersize=8,
            color=color,
            alpha=max(spaghetti_alpha, 0.25),
            zorder=1000,
            label="ndt draws",
        )


def _render_start_uncertainty(
    axis: Axes,
    ndt_ref: float,
    z_abs: np.ndarray,
    intervals: list[tuple[float, float]] | None,
    mode: Literal["band", "samples", "both"],
    base_alpha: float,
    markershift: float,
    linewidth: float,
    color: str,
) -> None:
    """Starting-point uncertainty: a vertical interval whisker.

    The starting point is a scalar per draw, so the minimal honest device is
    a capped whisker spanning the widest interval's quantiles at the reference
    non-decision time. The per-draw scatter cloud in samples mode is emitted
    by the per-draw cartoon calls themselves.
    """
    if mode in ("band", "both") and intervals:
        lo, hi = intervals[0]
        x = ndt_ref + markershift
        axis.plot(
            [x, x],
            [float(np.quantile(z_abs, lo)), float(np.quantile(z_abs, hi))],
            color=color,
            linewidth=linewidth,
            alpha=min(1.0, 2.0 * base_alpha),
            marker="_",
            markersize=8,
            zorder=1031,
            label="_nolegend_",
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
    choices: list[int] | None = None,
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
    # choices is threaded in by _plot_model_cartoon_1D (from model.model_config);
    # fall back to the registry for built-in models called directly.
    if choices is None:
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
                color=TRAJ_COLOR_DEFAULT_DICT[choice],
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
                    color=TRAJ_COLOR_DEFAULT_DICT[choice],
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
                color=TRAJ_COLOR_DEFAULT_DICT[choice],
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
                color_dict=TRAJ_COLOR_DEFAULT_DICT,
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
            color_dict=TRAJ_COLOR_DEFAULT_DICT,
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
            Line2D([0], [0], color=TRAJ_COLOR_DEFAULT_DICT[choice], lw=1)
            for choice in choices
        ]
        custom_titles = ["response: " + str(choice) for choice in choices]

        custom_elems.append(Line2D([0], [0], color="black", lw=1.0, linestyle="dashed"))

        axis.legend(
            custom_elems,
            custom_titles,
            fontsize=legend_fontsize,
            shadow=legend_shadow,
            loc=cast("Any", legend_location),
        )  # type: ignore

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
    **kwargs
        Additional keyword arguments passed to plotting functions

    Notes
    -----
    This function visualizes multiple simulated decision paths, optionally highlighting
    the response times and choices. Each trajectory shows the evidence accumulation
    process leading to a decision. Trajectory colors are taken from
    ``TRAJ_COLOR_DEFAULT_DICT``.
    """
    colors_dict = TRAJ_COLOR_DEFAULT_DICT

    # Make bounds
    b = np.maximum(sample[0]["metadata"]["boundary"], 0)
    b_init = b[0]
    # .item(): NumPy >= 1.25 deprecated implicit ndim>0-to-scalar conversion.
    n_roll = int((sample[0]["metadata"]["t"][0] / delta_t_graph + 1).item())
    b = np.roll(b, n_roll)
    b[:n_roll] = b_init

    # Trajectories
    for i in range(n):
        metadata = sample[i]["metadata"]
        tmp_traj = metadata["trajectory"]
        tmp_traj_choice = float(sample[i]["choices"].flatten().item())

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
    color_dict: Mapping[Any, str],
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
    # .item(): NumPy >= 1.25 deprecated implicit ndim>0-to-scalar conversion.
    n_roll = int((sample["metadata"]["t"][0] / delta_t_graph + 1).item())
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

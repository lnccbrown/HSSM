"""Plotting functionalities for HSSM."""

import logging

import arviz as az
import matplotlib as mpl

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import _get_plotting_df

_logger = logging.getLogger("hssm")


def plot_posterior_predictive(
    model,
    idata: az.InferenceData | None = None,
    data: pd.DataFrame | None = None,
    n_samples: int | float | None = 20,
    row: str | None = None,
    col: str | None = None,
    col_wrap: int | None = None,
    palette: str | list | dict | None = None,
    binwidth: float | tuple[float, float] | None = 0.1,
    grid_kwargs: dict | None = None,
    title: str | None = None,
    xlabel: str | None = "Response Time",
    ylabel: str | None = None,
    **kwargs,
) -> mpl.axes.Axes | sns.FacetGrid:
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
        The original data. If not provided, will use the data object stored inside the
        model. If posterior predictive samples are not present in this object, will
        generate posterior predictive samples using the idata object and the original
        data in this DataFrame.
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
    palette : optional
        Colors to use for the different levels of the hue variable. Should be something
        that can be interpreted by color_palette(), or a dictionary mapping hue levels
        to matplotlib colors., by default None.
    binwidth : optional
        A number or a pair of numbers. Width of each bin. There are two other binning
        options: `bins` and `binrange`. Please see the documentation for the
        [`sns.histplot` function]
        (https://seaborn.pydata.org/generated/seaborn.histplot.html) for how to use
        them. in combination to control the bins. This argument overrides bins but can
        be used with `binrange.`
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
        The matplotlib axis or seaborn FacetGrid object containing the plot.
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

    if data is None:
        data = model.data

    extra_dims = [e for e in [row, col] if e is not None] or None

    if "posterior_predictive" not in idata:
        _logger.info(
            "No posterior predictive samples found. Generating posterior predictive "
            + "samples using the provided InferenceData object and the original data. "
            + "This will modify the provided InferenceData object, or if not provided, "
            + "the traces object stored inside the model."
        )
        model.sample_posterior_predictive(
            idata=idata, data=data, inplace=True, n_samples=n_samples
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

    if extra_dims is None:
        ax = sns.histplot(
            data=plotting_df,
            hue="observed",
            stat="density",
            x="Response Time",
            fill=False,
            binwidth=binwidth,
            element="step",
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

    # The multiple dimensions case

    # For some reason `FacetGrid` does not recognize index levels as columns
    # Convert it to a column
    plotting_df = plotting_df.reset_index("observed")

    g = sns.FacetGrid(
        data=plotting_df,
        col=col,
        row=row,
        col_wrap=col_wrap,
        legend_out=True,
        hue="observed",
        palette=palette or ["#ec205b", "#338fb8"],
        **(grid_kwargs or {}),
    )

    g.map_dataframe(
        sns.histplot,
        x="Response Time",
        stat="density",
        fill=False,
        binwidth=binwidth,
        element="step",
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

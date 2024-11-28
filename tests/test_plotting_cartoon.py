"""Test plotting module."""

import pytest

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import hssm
from hssm.plotting.utils import (
    _get_plotting_df,
    _xarray_to_df,
    _get_title,
    _subset_df,
    _row_mask_with_error,
    _process_df_for_qp_plot,
)
from hssm.plotting.posterior_predictive import (
    _plot_posterior_predictive_1D,
    _plot_posterior_predictive_2D,
    plot_posterior_predictive,
)
from hssm.plotting.quantile_probability import (
    _plot_quantile_probability_1D,
    _plot_quantile_probability_2D,
    plot_quantile_probability,
)

hssm.set_floatX("float32")


# I want to parameter
@pytest.mark.parametrize(
    ["n_trajectories", "groups", "plot_pp_mean", "plot_pp_samples", "row", "col"],
    [
        (2, None, False, False, "participant_id", "stim"),
        (2, None, True, True, "participant_id", "stim"),
        (2, None, False, True, "participant_id", "stim"),
        (0, None, False, True, "participant_id", "stim"),
        (0, None, True, False, "participant_id", "stim"),
        (2, ["dbs"], True, True, "participant_id", "stim"),
        (2, None, True, False, "participant_id", None),
        (2, None, True, False, "participant_id", "stim"),
        (2, None, True, False, None, None),
    ],
)
def test_plot_model_cartoon_2_choice(
    cav_model_cartoon, n_trajectories, groups, plot_pp_mean, plot_pp_samples, row, col
):
    """Test plot_model_cartoon for 2-choice data."""
    if (not plot_pp_mean) and (not plot_pp_samples):
        with pytest.raises(ValueError):
            ax = hssm.plotting.plot_model_cartoon(
                cav_model_cartoon,
                n_samples=10,
                bins=30,
                col=col,
                row=row,
                groups=groups,
                plot_pp_mean=plot_pp_mean,
                plot_pp_samples=plot_pp_samples,
                alpha=0.025,
                **{"n_trajectories": n_trajectories},
            )
    else:
        ax = hssm.plotting.plot_model_cartoon(
            cav_model_cartoon,
            n_samples=10,
            bins=30,
            col=col,
            row=row,
            groups=groups,
            plot_pp_mean=plot_pp_mean,
            plot_pp_samples=plot_pp_samples,
            alpha=0.025,
            **{"n_trajectories": n_trajectories},
        )

        if groups is None:
            if row is not None:
                assert np.all(ax.row_names == cav_model_cartoon.data[row].unique())
            if col is not None:
                assert np.all(ax.col_names == cav_model_cartoon.data[col].unique())
        elif groups is ["dbs"]:
            assert isinstance(ax, list)
            assert len(ax) == len(cav_model_cartoon.data[groups[0]].unique())


@pytest.mark.parametrize(
    ["n_trajectories", "groups", "plot_pp_mean", "plot_pp_samples", "row", "col"],
    [
        (2, None, False, False, "participant_id", "stim"),
        (2, None, True, True, "participant_id", "stim"),
        (2, None, False, True, "participant_id", "stim"),
        (0, None, False, True, "participant_id", "stim"),
        (0, None, True, False, "participant_id", "stim"),
        (2, None, True, False, "participant_id", None),
        (2, None, True, False, "participant_id", "stim"),
        (2, None, True, False, None, None),
    ],
)
def test_plot_model_cartoon_3_choice(
    race_model_cartoon, n_trajectories, groups, plot_pp_mean, plot_pp_samples, row, col
):
    """Test plot_model_cartoon for 3-choice data."""

    if (not plot_pp_mean) and (not plot_pp_samples):
        with pytest.raises(ValueError):
            ax = hssm.plotting.plot_model_cartoon(
                race_model_cartoon,
                n_samples=10,
                bins=30,
                col=col,
                row=row,
                groups=groups,
                plot_pp_mean=plot_pp_mean,
                plot_pp_samples=plot_pp_samples,
                alpha=0.025,
                **{"n_trajectories": n_trajectories},
            )
    else:
        ax = hssm.plotting.plot_model_cartoon(
            race_model_cartoon,
            n_samples=10,
            bins=30,
            col=col,
            row=row,
            groups=groups,
            plot_pp_mean=plot_pp_mean,
            plot_pp_samples=plot_pp_samples,
            alpha=0.025,
            **{"n_trajectories": n_trajectories},
        )

        if groups is None:
            if row is not None:
                assert np.all(ax.row_names == race_model_cartoon.data[row].unique())
            if col is not None:
                assert np.all(ax.col_names == race_model_cartoon.data[col].unique())
        else:
            assert isinstance(ax, list)
            assert len(ax) == len(race_model_cartoon.data[groups].unique())

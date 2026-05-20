"""Test plotting module."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

import hssm
from hssm.plotting.model_cartoon import (
    _to_idata_group,
    attach_trialwise_params_to_df,
    compute_merge_necessary_deterministics,
)
from hssm.plotting.utils import _get_plotting_df

matplotlib.use("Agg")

hssm.set_floatX("float32")


def test_attach_trialwise_params_to_df_handles_fixed_params():
    """Fixed constructor params must not crash the cartoon plot helper.

    When a parameter is fixed via ``hssm.HSSM(..., z=0.5)`` it never
    enters ``idata[idata_group]``. The helper should fall back to
    ``model.params[param].prior`` instead of raising ``KeyError``.

    Such params don't appear in ``idata[idata_group]`` because they
    weren't sampled. The helper should read the fixed value from
    ``model.params[param].prior`` instead and broadcast it to every row.
    Before the fix, the function unconditionally indexed ``idata`` and
    raised ``KeyError`` on the first missing param.
    """
    rng = np.random.default_rng(0)
    n = 60
    data = pd.DataFrame(
        {
            "rt": rng.uniform(0.5, 1.5, size=n),
            "response": rng.integers(0, 4, size=n),
            "stim": rng.choice(["low", "medium", "high"], size=n),
            "participant_id": "0",
        }
    )

    # `z` and `theta` are passed as scalars -> registered as fixed params
    # and excluded from the sampled posterior.
    race_model = hssm.HSSM(
        model="race_no_bias_angle_4",
        data=data,
        noncentered=False,
        include=[
            {
                "name": "v0",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.5},
                },
                "formula": "v0 ~ 1 + C(stim)",
                "link": "identity",
            },
        ],
        theta=0.1,
        z=0.5,
        p_outlier=0.00,
    )

    # Sanity: the helper's preconditions hold.
    assert race_model.params["z"].is_fixed
    assert race_model.params["z"].prior == 0.5
    assert race_model.params["theta"].is_fixed
    assert race_model.params["theta"].prior == 0.1

    # Use prior_predictive to skip MCMC and keep the test fast.
    idata = race_model.sample_prior_predictive(draws=3)
    assert "z" not in idata.prior.data_vars
    assert "theta" not in idata.prior.data_vars

    plotting_df = _get_plotting_df(
        idata,
        data,
        extra_dims=["stim"],
        n_samples=None,
        response_str=race_model.response_str,
        predictive_group="prior_predictive",
    )
    idata = compute_merge_necessary_deterministics(
        race_model, idata, idata_group=_to_idata_group("prior_predictive")
    )

    # Before the fix this raised `KeyError: "No variable named 'z'..."`.
    plotting_df = attach_trialwise_params_to_df(
        race_model, plotting_df, idata, idata_group=_to_idata_group("prior_predictive")
    )

    # The helper only fills predictive rows (chain != -1); observed rows
    # keep the initialized 0.0. Check that the fixed values landed on
    # every predictive row.
    pred_rows = plotting_df[plotting_df.index.get_level_values("chain") != -1]
    assert (pred_rows["z"] == 0.5).all(), (
        "fixed z=0.5 not broadcast to every predictive row"
    )
    assert (pred_rows["theta"] == 0.1).all(), (
        "fixed theta=0.1 not broadcast to every predictive row"
    )


# I want to parameter
@pytest.mark.slow
@pytest.mark.parametrize(
    [
        "n_trajectories",
        "groups",
        "plot_predictive_mean",
        "plot_predictive_samples",
        "predictive_group",
        "row",
        "col",
    ],
    [
        (2, None, False, False, "posterior_predictive", "participant_id", "stim"),
        (2, None, False, False, "prior_predictive", "participant_id", "stim"),
        (2, None, True, True, "posterior_predictive", "participant_id", "stim"),
        (2, None, True, True, "prior_predictive", "participant_id", "stim"),
        (2, None, False, True, "posterior_predictive", "participant_id", "stim"),
        (2, None, False, True, "prior_predictive", "participant_id", "stim"),
        (0, None, False, True, "posterior_predictive", "participant_id", "stim"),
        (0, None, False, True, "prior_predictive", "participant_id", "stim"),
        (0, None, True, False, "posterior_predictive", "participant_id", "stim"),
        (0, None, True, False, "prior_predictive", "participant_id", "stim"),
        (2, ["dbs"], True, True, "posterior_predictive", "participant_id", "stim"),
        (2, ["dbs"], True, True, "prior_predictive", "participant_id", "stim"),
        (2, None, True, False, "posterior_predictive", "participant_id", None),
        (2, None, True, False, "prior_predictive", "participant_id", None),
        (2, None, True, False, "posterior_predictive", "participant_id", "stim"),
        (2, None, True, False, "prior_predictive", "participant_id", "stim"),
        (2, None, True, False, "posterior_predictive", None, None),
        (2, None, True, False, "prior_predictive", None, None),
    ],
)
def test_plot_model_cartoon_2_choice(
    cav_model_cartoon,
    n_trajectories,
    groups,
    plot_predictive_mean,
    plot_predictive_samples,
    predictive_group,
    row,
    col,
):
    """Test plot_model_cartoon for 2-choice data."""
    if (not plot_predictive_mean) and (not plot_predictive_samples):
        with pytest.raises(ValueError):
            ax = hssm.plotting.plot_model_cartoon(
                cav_model_cartoon,
                n_samples=10,
                n_samples_prior=10,  # AF-TODO: Low number of samples fails
                bins=30,
                col=col,
                row=row,
                groups=groups,
                predictive_group=predictive_group,
                plot_predictive_mean=plot_predictive_mean,
                plot_predictive_samples=plot_predictive_samples,
                alpha_mean=0.025,
                n_trajectories=n_trajectories,
            )
    else:
        ax = hssm.plotting.plot_model_cartoon(
            cav_model_cartoon,
            n_samples=10,
            n_samples_prior=100,  # AF-TODO: Low number of samples fails
            bins=30,
            col=col,
            row=row,
            groups=groups,
            predictive_group=predictive_group,
            plot_predictive_mean=plot_predictive_mean,
            plot_predictive_samples=plot_predictive_samples,
            alpha_mean=0.025,
            n_trajectories=n_trajectories,
        )

        if groups is None:
            if row is not None:
                assert np.all(ax.row_names == cav_model_cartoon.data[row].unique())
            if col is not None:
                assert np.all(ax.col_names == cav_model_cartoon.data[col].unique())
        elif groups == ["dbs"]:
            assert isinstance(ax, list)
            assert len(ax) == len(cav_model_cartoon.data[groups[0]].unique())


@pytest.mark.slow
def test_plot_model_cartoon_intercept_only(intercept_only_ddm_cartoon):
    """Test plot_model_cartoon with intercept-only DDM (no regression).

    Bambi >= 0.17 returns scalar deterministics with shape (1,) for
    intercept-only models. This test ensures attach_trialwise_params_to_df
    correctly broadcasts these to all observation rows.
    """
    ax = hssm.plotting.plot_model_cartoon(
        intercept_only_ddm_cartoon,
        n_samples_prior=100,
        bins=20,
        plot_predictive_mean=True,
        plot_predictive_samples=False,
        predictive_group="prior_predictive",
        n_trajectories=2,
    )
    assert ax is not None


@pytest.mark.slow
@pytest.mark.parametrize(
    [
        "n_trajectories",
        "groups",
        "plot_predictive_mean",
        "plot_predictive_samples",
        "predictive_group",
        "row",
        "col",
    ],
    [
        (2, None, False, False, "posterior_predictive", "participant_id", "stim"),
        (2, None, False, False, "prior_predictive", "participant_id", "stim"),
        (2, None, True, True, "posterior_predictive", "participant_id", "stim"),
        (2, None, True, True, "prior_predictive", "participant_id", "stim"),
        (2, None, False, True, "posterior_predictive", "participant_id", "stim"),
        (2, None, False, True, "prior_predictive", "participant_id", "stim"),
        (0, None, False, True, "posterior_predictive", "participant_id", "stim"),
        (0, None, False, True, "prior_predictive", "participant_id", "stim"),
        (0, None, True, False, "posterior_predictive", "participant_id", "stim"),
        (0, None, True, False, "prior_predictive", "participant_id", "stim"),
        (2, None, True, False, "posterior_predictive", "participant_id", None),
        (2, None, True, False, "prior_predictive", "participant_id", None),
        (2, None, True, False, "posterior_predictive", "participant_id", "stim"),
        (2, None, True, False, "prior_predictive", "participant_id", "stim"),
        (2, None, True, False, "posterior_predictive", None, None),
        (2, None, True, False, "prior_predictive", None, None),
    ],
)
def test_plot_model_cartoon_3_choice(
    race_model_cartoon,
    n_trajectories,
    groups,
    plot_predictive_mean,
    plot_predictive_samples,
    predictive_group,
    row,
    col,
):
    """Test plot_model_cartoon for 3-choice data."""
    if (not plot_predictive_mean) and (not plot_predictive_samples):
        with pytest.raises(ValueError):
            ax = hssm.plotting.plot_model_cartoon(
                race_model_cartoon,
                n_samples=10,
                n_samples_prior=10,
                bins=30,
                col=col,
                row=row,
                groups=groups,
                plot_predictive_mean=plot_predictive_mean,
                plot_predictive_samples=plot_predictive_samples,
                predictive_group=predictive_group,
                alpha_mean=0.025,
                n_trajectories=n_trajectories,
            )
    else:
        ax = hssm.plotting.plot_model_cartoon(
            race_model_cartoon,
            n_samples=10,
            n_samples_prior=10,
            bins=30,
            col=col,
            row=row,
            groups=groups,
            plot_predictive_mean=plot_predictive_mean,
            plot_predictive_samples=plot_predictive_samples,
            predictive_group=predictive_group,
            alpha_mean=0.025,
            n_trajectories=n_trajectories,
        )

        if groups is None:
            if row is not None:
                assert np.all(ax.row_names == race_model_cartoon.data[row].unique())
            if col is not None:
                assert np.all(ax.col_names == race_model_cartoon.data[col].unique())
        else:
            assert isinstance(ax, list)
            assert len(ax) == len(race_model_cartoon.data[groups].unique())

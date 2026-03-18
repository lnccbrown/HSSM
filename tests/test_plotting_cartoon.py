"""Test plotting module."""

import pytest
import numpy as np
import hssm

hssm.set_floatX("float32")


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
        elif groups is ["dbs"]:
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

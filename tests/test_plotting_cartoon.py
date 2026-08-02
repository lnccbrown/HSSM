"""Test plotting module."""

import numpy as np
import pytest
from matplotlib.collections import PolyCollection

import hssm

hssm.set_floatX("float32")


@pytest.mark.slow
@pytest.mark.parametrize(
    [
        "n_trajectories",
        "groups",
        "uncertainty",
        "predictive_group",
        "row",
        "col",
    ],
    [
        (2, None, "both", "posterior_predictive", "participant_id", "stim"),
        (2, None, "both", "prior_predictive", "participant_id", "stim"),
        (2, None, "samples", "posterior_predictive", "participant_id", "stim"),
        (2, None, "samples", "prior_predictive", "participant_id", "stim"),
        (0, None, "band", "posterior_predictive", "participant_id", "stim"),
        (0, None, "band", "prior_predictive", "participant_id", "stim"),
        (0, None, None, "posterior_predictive", "participant_id", "stim"),
        (0, None, None, "prior_predictive", "participant_id", "stim"),
        (2, ["dbs"], "band", "posterior_predictive", "participant_id", "stim"),
        (2, ["dbs"], "band", "prior_predictive", "participant_id", "stim"),
        (2, None, None, "posterior_predictive", "participant_id", None),
        (2, None, None, "prior_predictive", "participant_id", None),
        (2, None, "band", "posterior_predictive", None, None),
        (2, None, "band", "prior_predictive", None, None),
    ],
)
def test_plot_model_cartoon_2_choice(
    cav_model_cartoon,
    n_trajectories,
    groups,
    uncertainty,
    predictive_group,
    row,
    col,
):
    """Test plot_model_cartoon for 2-choice data."""
    ax = hssm.plotting.plot_model_cartoon(
        cav_model_cartoon,
        n_samples=10,
        n_samples_prior=100,  # AF-TODO: Low number of samples fails
        col=col,
        row=row,
        groups=groups,
        predictive_group=predictive_group,
        uncertainty=uncertainty,
        n_trajectories=n_trajectories,
        random_state=42,
    )

    if groups is None:
        if row is not None:
            assert np.all(ax.row_names == cav_model_cartoon.data[row].unique())
        if col is not None:
            assert np.all(ax.col_names == cav_model_cartoon.data[col].unique())
        if row is not None and uncertainty in ("band", "both"):
            # Band mode must place quantile-band PolyCollections in each
            # facet's twin axes (twins register on the figure).
            fig = ax.figure
            assert any(
                isinstance(coll, PolyCollection)
                for axes in fig.axes
                for coll in axes.collections
            )
    elif groups == ["dbs"]:
        assert isinstance(ax, list)
        assert len(ax) == len(cav_model_cartoon.data[groups[0]].unique())


@pytest.mark.slow
def test_plot_model_cartoon_legacy_booleans(cav_model_cartoon):
    """The deprecated boolean spellings still work, with a FutureWarning."""
    with pytest.warns(FutureWarning):
        ax = hssm.plotting.plot_model_cartoon(
            cav_model_cartoon,
            n_samples=10,
            plot_predictive_mean=True,
            plot_predictive_samples=True,
        )
    assert ax is not None

    with pytest.raises(ValueError):
        hssm.plotting.plot_model_cartoon(
            cav_model_cartoon,
            n_samples=10,
            plot_predictive_mean=False,
            plot_predictive_samples=False,
        )


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
        uncertainty=None,
        predictive_group="prior_predictive",
        n_trajectories=2,
    )
    assert ax is not None


@pytest.mark.slow
def test_plot_model_cartoon_random_state_end_to_end(cav_model_cartoon):
    """Same random_state => identical figure through the full public path
    (draw selection, simulations, trajectories)."""
    import matplotlib.pyplot as plt

    # Warm-up materializes the posterior-predictive group once, so both
    # seeded renders below take the identical code path.
    hssm.plotting.plot_model_cartoon(cav_model_cartoon, n_samples=10)
    plt.close("all")

    def snapshot():
        ax = hssm.plotting.plot_model_cartoon(
            cav_model_cartoon,
            n_samples=10,
            uncertainty="band",
            n_trajectories=2,
            random_state=7,
        )
        fig = ax.get_figure()
        data = [
            line.get_ydata().copy() for axes in fig.axes for line in axes.get_lines()
        ]
        plt.close("all")
        return data

    first, second = snapshot(), snapshot()
    assert first and len(first) == len(second)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


@pytest.mark.slow
def test_plot_model_cartoon_obs_conditioning(cav_model_cartoon):
    """obs= conditions the cartoon on one trial of the regression model."""
    ax = hssm.plotting.plot_model_cartoon(
        cav_model_cartoon,
        n_samples=10,
        uncertainty="band",
        obs=0,
        random_state=3,
    )
    assert ax is not None

    with pytest.raises(ValueError, match="obs="):
        hssm.plotting.plot_model_cartoon(
            cav_model_cartoon, n_samples=10, obs=10_000_000
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    [
        "n_trajectories",
        "groups",
        "uncertainty",
        "predictive_group",
        "row",
        "col",
    ],
    [
        (2, None, "both", "posterior_predictive", "participant_id", "stim"),
        (2, None, "both", "prior_predictive", "participant_id", "stim"),
        (2, None, "samples", "posterior_predictive", "participant_id", "stim"),
        (2, None, "samples", "prior_predictive", "participant_id", "stim"),
        (0, None, "band", "posterior_predictive", "participant_id", "stim"),
        (0, None, "band", "prior_predictive", "participant_id", "stim"),
        (0, None, None, "posterior_predictive", "participant_id", "stim"),
        (0, None, None, "prior_predictive", "participant_id", "stim"),
        (2, None, None, "posterior_predictive", "participant_id", None),
        (2, None, None, "prior_predictive", "participant_id", None),
        (2, None, "band", "posterior_predictive", None, None),
        (2, None, "band", "prior_predictive", None, None),
    ],
)
def test_plot_model_cartoon_3_choice(
    race_model_cartoon,
    n_trajectories,
    groups,
    uncertainty,
    predictive_group,
    row,
    col,
):
    """Test plot_model_cartoon for 3-choice data."""
    ax = hssm.plotting.plot_model_cartoon(
        race_model_cartoon,
        n_samples=10,
        n_samples_prior=10,
        col=col,
        row=row,
        groups=groups,
        uncertainty=uncertainty,
        predictive_group=predictive_group,
        n_trajectories=n_trajectories,
        random_state=42,
    )

    if groups is None:
        if row is not None:
            assert np.all(ax.row_names == race_model_cartoon.data[row].unique())
        if col is not None:
            assert np.all(ax.col_names == race_model_cartoon.data[col].unique())
    else:
        assert isinstance(ax, list)
        assert len(ax) == len(race_model_cartoon.data[groups].unique())

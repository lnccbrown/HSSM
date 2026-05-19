"""Test plotting module."""

import matplotlib
import pandas as pd
import pytest
import numpy as np
import hssm
from hssm.plotting import model_cartoon as cartoon_module
from hssm.plotting.model_cartoon import plot_func_model

matplotlib.use("Agg")

hssm.set_floatX("float32")


def test_plot_func_model_uses_facet_average_for_drift(monkeypatch):
    """The no-noise drift visualisation must use facet-averaged params.

    Regression test for a bug where the mean-drift cartoon line was
    drawn from a randomly-picked trial's parameters while the
    posterior-sample drift cartoons were pinned (by the simulator) to
    trial 0 of each (chain, draw). For trial-wise drift regressions
    (`v ~ covariate`) the two referenced different trials with
    different drifts, producing a visually misaligned black mean line.
    With the fix, both pass facet-averaged parameters as a single-row
    theta so the recorded drift trajectory represents the facet mean.
    """
    import matplotlib.pyplot as plt

    # Trial-varying v (positions 0..4) so a random pick would obviously
    # differ from the average.
    v_per_trial = np.array([0.2, 0.6, 1.0, 1.4, 1.8])
    a, z, t = 1.3, 0.5, 0.3
    n_trials = len(v_per_trial)
    expected_avg_v = float(v_per_trial.mean())

    # theta_mean: indexed (chain, draw, obs_n) — same layout that
    # attach_trialwise_params_to_df produces.
    theta_mean = pd.DataFrame(
        {
            "v": v_per_trial,
            "a": np.full(n_trials, a),
            "z": np.full(n_trials, z),
            "t": np.full(n_trials, t),
        },
        index=pd.MultiIndex.from_tuples(
            [(0, 0, i) for i in range(n_trials)], names=["chain", "draw", "obs_n"]
        ),
    )

    captured_no_noise_thetas: list[np.ndarray] = []
    real_simulator = cartoon_module.simulator

    def recording_simulator(*args, **kwargs):
        if kwargs.get("no_noise", False):
            captured_no_noise_thetas.append(np.asarray(kwargs["theta"]).copy())
        return real_simulator(*args, **kwargs)

    monkeypatch.setattr(cartoon_module, "simulator", recording_simulator)

    fig, ax = plt.subplots()
    plot_func_model(
        model_name="ddm",
        axis=ax,
        theta_mean=theta_mean,
        theta_samples=None,
        n_samples=1,
        n_trajectories=0,  # avoid trajectory-side simulator calls
    )

    # First no_noise=True call inside plot_func_model is the posterior-mean
    # drift visualization — that's the one we care about for this test.
    assert captured_no_noise_thetas, (
        "expected at least one no_noise=True simulator call"
    )
    mean_theta = captured_no_noise_thetas[0]
    assert mean_theta.shape == (1, 4), (
        f"mean drift theta should be a single averaged row, "
        f"got shape {mean_theta.shape}"
    )
    assert mean_theta[0, 0] == pytest.approx(expected_avg_v, abs=1e-6), (
        f"facet-averaged drift expected {expected_avg_v}, "
        f"got {mean_theta[0, 0]} — random-trial selection bug regressed"
    )
    plt.close(fig)


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

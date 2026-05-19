"""Test plotting module."""

import matplotlib
import pandas as pd
import pytest
import numpy as np
import hssm
from hssm.plotting import model_cartoon as cartoon_module
from hssm.plotting.model_cartoon import plot_func_model, plot_func_model_n

matplotlib.use("Agg")

hssm.set_floatX("float32")


# (model_name, plot_func, drift_param_names, fixed_params, n_drift_params)
# - drift_param_names varies per trial so random pick vs average disagrees
# - fixed_params are constant across trials (a, z, t, theta, etc.)
# - n_drift_params is the number of leading drift columns in theta
_CARTOON_CASES = [
    pytest.param(
        "ddm",
        plot_func_model,
        ["v"],
        {"a": 1.3, "z": 0.5, "t": 0.3},
        id="2choice-ddm",
    ),
    pytest.param(
        "race_no_bias_angle_4",
        plot_func_model_n,
        ["v0", "v1", "v2", "v3"],
        {"a": 1.3, "z": 0.5, "t": 0.3, "theta": 0.1},
        id="nchoice-race4",
    ),
]


def _ensure_model_config_registered(model_name):
    """Lazy-load a model's config into the global registry if absent.

    `default_model_config` ships pre-populated with ``"ddm"`` only; other
    models are loaded on first use through `hssm.HSSM`. Tests that exercise
    the cartoon helpers directly need to populate the registry themselves.
    """
    from hssm.defaults import default_model_config
    from hssm.modelconfig import get_default_model_config

    if model_name not in default_model_config:
        default_model_config[model_name] = get_default_model_config(model_name)


@pytest.mark.parametrize(
    "model_name, plot_func, drift_param_names, fixed_params", _CARTOON_CASES
)
def test_cartoon_uses_facet_average_for_drift(
    monkeypatch, model_name, plot_func, drift_param_names, fixed_params
):
    """The no-noise drift visualisation must use facet-averaged params.

    Regression test for a bug where the mean-drift cartoon line was
    drawn from a randomly-picked trial's parameters while the
    posterior-sample drift cartoons were pinned (by the simulator) to
    trial 0 of each (chain, draw). For trial-wise drift regressions
    (``v ~ covariate`` for DDM, ``v0 ~ covariate`` for race models) the
    two referenced different trials with different drifts, producing a
    visually misaligned black mean line. With the fix, both pass
    facet-averaged parameters as a single-row theta so the recorded
    drift trajectory represents the facet mean.

    Parameterised across the 2-choice path (``plot_func_model``) and
    the n-choice path (``plot_func_model_n``) since they have the same
    bug structure.
    """
    import matplotlib.pyplot as plt

    _ensure_model_config_registered(model_name)
    n_trials = 5
    # Trial-varying drift columns so a random pick obviously differs
    # from the average.
    drift_columns = {
        name: np.linspace(0.2 + 0.3 * idx, 1.8 + 0.3 * idx, n_trials)
        for idx, name in enumerate(drift_param_names)
    }
    fixed_columns = {k: np.full(n_trials, v) for k, v in fixed_params.items()}

    theta_mean = pd.DataFrame(
        {**drift_columns, **fixed_columns},
        index=pd.MultiIndex.from_tuples(
            [(0, 0, i) for i in range(n_trials)], names=["chain", "draw", "obs_n"]
        ),
    )
    expected_avg = theta_mean.mean(axis=0).values

    captured_no_noise_thetas: list[np.ndarray] = []
    real_simulator = cartoon_module.simulator

    def recording_simulator(*args, **kwargs):
        if kwargs.get("no_noise", False):
            captured_no_noise_thetas.append(np.asarray(kwargs["theta"]).copy())
        return real_simulator(*args, **kwargs)

    monkeypatch.setattr(cartoon_module, "simulator", recording_simulator)

    fig, ax = plt.subplots()
    plot_func(
        model_name=model_name,
        axis=ax,
        theta_mean=theta_mean,
        theta_samples=None,
        n_samples=1,
        n_trajectories=0,  # avoid trajectory-side simulator calls
    )

    # First no_noise=True call is the posterior-mean drift simulation —
    # that's the one we care about for this test.
    assert captured_no_noise_thetas, (
        "expected at least one no_noise=True simulator call"
    )
    mean_theta = captured_no_noise_thetas[0]
    expected_shape = (1, theta_mean.shape[1])
    assert mean_theta.shape == expected_shape, (
        f"mean drift theta should be a single averaged row, "
        f"got shape {mean_theta.shape}, expected {expected_shape}"
    )
    np.testing.assert_allclose(
        mean_theta[0],
        expected_avg,
        atol=1e-6,
        err_msg=(
            f"facet-averaged drift params expected {expected_avg}, "
            f"got {mean_theta[0]} — random-trial selection bug regressed"
        ),
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

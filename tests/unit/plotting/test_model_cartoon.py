"""Unit tests for the model-cartoon uncertainty rendering (#1124).

Fast by construction: hand-built θ DataFrames plus the real ssm-simulators
simulator at tiny scale (single-row no-noise calls are milliseconds), no HSSM
model and no netcdf fixtures. Mirrors the artist-structure assertion style
established for plot_predictive in test_predictive.py.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection

from hssm.plotting.model_cartoon import (
    _add_uncertain_histograms,
    _defective_densities,
    _defective_densities_n,
    _density_matrices,
    _geometry_arrays,
    _reduce_theta,
    _resolve_uncertainty_from_legacy,
    plot_func_model,
    plot_func_model_n,
)
from hssm.plotting.utils import _band_alphas, _curve_xy, _hdi_to_intervals

INTERVALS = _hdi_to_intervals([0.5, 0.94])


def _theta_frames(n_chains=2, n_draws=3, n_obs=6, jitter=0.05):
    """Hand-built θ frames matching the shapes plot_func_model receives."""
    idx = pd.MultiIndex.from_product(
        [range(n_chains), range(n_draws), range(n_obs)],
        names=["chain", "draw", "obs_n"],
    )
    rng = np.random.default_rng(7)
    theta_samples = pd.DataFrame({"v": 0.5, "a": 1.2, "z": 0.5, "t": 0.3}, index=idx)
    theta_samples["v"] += rng.normal(0, jitter, len(theta_samples))
    theta_samples["a"] += rng.normal(0, jitter, len(theta_samples))
    theta_mean = pd.DataFrame({"v": [0.5] * n_obs, "a": 1.2, "z": 0.5, "t": 0.3})
    return theta_mean, theta_samples


def _observed():
    return pd.DataFrame(
        {"rt": [0.5, 0.7, -0.6, 0.9, -0.8, 0.55], "response": [1, 1, -1, 1, -1, 1]}
    )


def _render(uncertainty, **overrides):
    theta_mean, theta_samples = _theta_frames()
    fig, ax = plt.subplots()
    kwargs = dict(
        model_name="ddm",
        axis=ax,
        theta_mean=theta_mean,
        theta_samples=None if uncertainty is None else theta_samples,
        data=_observed(),
        n_reps=2,
        random_state=0,
        uncertainty=uncertainty,
        intervals=INTERVALS if uncertainty is not None else None,
    )
    kwargs.update(overrides)
    plot_func_model(**kwargs)
    return fig, ax, fig.axes[1], fig.axes[2]  # main, twin up, twin down


class TestDefectiveDensities:
    def test_mass_identities(self):
        rts = np.array([0.5, 0.7, 0.6, 0.9, 0.8, -999.0])
        choices = np.array([1, 1, -1, 1, -1, 1])
        edges = np.linspace(0, 2, 21)
        up, down = _defective_densities(rts, choices, edges)
        widths = np.diff(edges)
        total = (up * widths).sum() + (down * widths).sum()
        assert total == pytest.approx(1.0)
        # -999 excluded from numerator AND denominator: 3 of 5 valid are up
        assert (up * widths).sum() == pytest.approx(3 / 5)

    def test_nonuniform_edges(self):
        rts = np.array([0.5, 0.7, 0.6, 1.4])
        choices = np.array([1, 1, -1, -1])
        edges = np.array([0.0, 0.55, 0.65, 1.0, 2.0])
        up, down = _defective_densities(rts, choices, edges)
        widths = np.diff(edges)
        assert (up * widths).sum() + (down * widths).sum() == pytest.approx(1.0)

    def test_all_deadline_returns_zeros(self):
        rts = np.full(4, -999.0)
        choices = np.array([1, -1, 1, -1])
        edges = np.linspace(0, 1, 5)
        up, down = _defective_densities(rts, choices, edges)
        assert not up.any() and not down.any()

    def test_simulator_shape_is_handled(self):
        # simulator returns (n_reps, n_trials, 1); ravel must cope
        rts = np.full((2, 3, 1), 0.5)
        choices = np.ones((2, 3, 1))
        edges = np.linspace(0, 1, 5)
        up, down = _defective_densities(rts, choices, edges)
        assert (up * np.diff(edges)).sum() == pytest.approx(1.0)

    def test_n_choice_masses(self):
        rts = np.array([0.5, 0.7, 0.6, 0.9, -999.0])
        choices_arr = np.array([0, 1, 2, 1, 0])
        edges = np.linspace(0, 2, 11)
        dens = _defective_densities_n(rts, choices_arr, edges, [0, 1, 2])
        widths = np.diff(edges)
        masses = {c: (d * widths).sum() for c, d in dens.items()}
        assert sum(masses.values()) == pytest.approx(1.0)
        assert masses[1] == pytest.approx(2 / 4)

    def test_density_matrices_shape_and_mean(self):
        edges = np.linspace(0, 2, 9)
        sims = {
            i: {"rts": np.array([0.5, 0.9]), "choices": np.array([1, -1])}
            for i in range(3)
        }
        d_up, d_down = _density_matrices(sims, edges)
        assert d_up.shape == (3, 8) and d_down.shape == (3, 8)
        manual_up, _ = _defective_densities(sims[0]["rts"], sims[0]["choices"], edges)
        np.testing.assert_allclose(d_up.mean(axis=0), manual_up)


class TestLegacyShim:
    def test_untouched_when_booleans_absent(self):
        assert _resolve_uncertainty_from_legacy(None, None, "band", 1.0) == (
            "band",
            1.0,
        )

    @pytest.mark.parametrize(
        "mean,samples,expected_mode,expected_alpha",
        [
            (True, False, None, 1.0),
            (True, True, "samples", 1.0),
            (False, True, "samples", 0.0),
        ],
    )
    def test_boolean_mappings(self, mean, samples, expected_mode, expected_alpha):
        with pytest.warns(FutureWarning):
            mode, alpha = _resolve_uncertainty_from_legacy(mean, samples, "band", 1.0)
        assert mode == expected_mode
        assert alpha == expected_alpha

    def test_both_false_raises(self):
        with pytest.raises(ValueError, match="At least one of"):
            _resolve_uncertainty_from_legacy(False, False, "band", 1.0)

    def test_conflict_with_explicit_uncertainty_raises(self):
        with pytest.raises(ValueError, match="not both"):
            _resolve_uncertainty_from_legacy(None, True, "samples", 1.0)


class TestSharedHelpers:
    def test_band_alphas_ladder(self):
        np.testing.assert_allclose(_band_alphas(0.25, 2), [0.1125, 0.25])
        # a single band gets the full base alpha
        np.testing.assert_allclose(_band_alphas(0.25, 1), [0.25])

    def test_curve_xy_shared_location(self):
        from hssm.plotting import predictive, utils

        assert predictive._curve_xy is utils._curve_xy

    def test_geometry_arrays_shapes(self):
        t_s = np.arange(0, 2, 0.01)
        n_t = t_s.shape[0]
        boundary = np.ones(n_t + 50)
        traj = np.concatenate([np.linspace(0, 1, 60), np.full(n_t, -999.0)])
        sims = {
            i: {
                "metadata": {
                    "boundary": boundary,
                    "trajectory": traj,
                    "t": np.array([0.3]),
                    "z": np.array([0.5]),
                }
            }
            for i in range(4)
        }
        b_high, b_low, drifts, ndts, z_abs = _geometry_arrays(sims, t_s, 0.01)
        assert b_high.shape == (4, n_t)
        assert np.isnan(drifts[:, -1]).all()  # absorbed well before the end
        np.testing.assert_allclose(ndts, 0.3)
        # z=0.5 on a symmetric bound: starting point at 0
        np.testing.assert_allclose(z_abs, 0.0, atol=1e-12)

    def test_geometry_arrays_short_boundary_constant_extension(self):
        t_s = np.arange(0, 2, 0.01)
        n_t = t_s.shape[0]
        boundary = np.linspace(1.0, 0.5, n_t - 30)
        traj = np.concatenate([np.linspace(0, 1, 60), np.full(n_t, -999.0)])
        sims = {
            0: {
                "metadata": {
                    "boundary": boundary,
                    "trajectory": traj,
                    "t": np.array([0.0]),
                    "z": np.array([0.5]),
                }
            }
        }
        b_high, b_low, *_ = _geometry_arrays(sims, t_s, 0.01)
        assert b_high.shape == (1, n_t)
        # beyond the simulated grid: constant extension of the last value
        np.testing.assert_allclose(b_high[0, n_t - 30 :], b_high[0, n_t - 31])
        np.testing.assert_allclose(b_low[0, n_t - 30 :], b_low[0, n_t - 31])

    def test_geometry_arrays_ignores_dict_keys(self):
        t_s = np.arange(0, 1, 0.01)
        n_t = t_s.shape[0]
        traj = np.concatenate([np.linspace(0, 1, 20), np.full(n_t, -999.0)])

        def sim(level):
            return {
                "metadata": {
                    "boundary": np.full(n_t, float(level)),
                    "trajectory": traj,
                    "t": np.array([0.0]),
                    "z": np.array([0.5]),
                }
            }

        # non-contiguous keys must not be used as row indices
        sims = {3: sim(1.0), 7: sim(2.0)}
        b_high, *_ = _geometry_arrays(sims, t_s, 0.01)
        np.testing.assert_allclose(b_high[0], 1.0)
        np.testing.assert_allclose(b_high[1], 2.0)


class TestThetaReduction:
    def test_trial_mean_and_obs_selection(self):
        theta_mean, theta_samples = _theta_frames()
        reduced = _reduce_theta(theta_samples)
        manual = theta_samples.groupby(level=["chain", "draw"]).mean()
        pd.testing.assert_frame_equal(reduced, manual)

        picked = _reduce_theta(theta_samples, obs=3)
        pd.testing.assert_frame_equal(picked, theta_samples.xs(3, level="obs_n"))

        flat = _reduce_theta(theta_mean)
        assert flat.shape == (1, theta_mean.shape[1])
        np.testing.assert_allclose(flat.values[0], theta_mean.mean().values)
        np.testing.assert_allclose(
            _reduce_theta(theta_mean, obs=1).values, theta_mean.loc[[1]].values
        )

    def test_missing_obs_label_raises(self):
        _, theta_samples = _theta_frames()
        with pytest.raises(ValueError, match="obs=99"):
            _reduce_theta(theta_samples, obs=99)

    def test_reference_geometry_is_trial_mean(self):
        # Trial 0 carries an outlier ndt; the drawn reference ndt must be the
        # trial mean, not trial 0's value (the pre-#1125 convention).
        theta_mean, _ = _theta_frames()
        theta_mean["t"] = [1.5] + [0.3] * (len(theta_mean) - 1)
        fig, ax = plt.subplots()
        plot_func_model(
            model_name="ddm",
            axis=ax,
            theta_mean=theta_mean,
            theta_samples=None,
            data=_observed(),
            n_reps=2,
            random_state=0,
            uncertainty=None,
            intervals=None,
        )
        vlines = [
            ln
            for ln in ax.get_lines()
            if ln.get_linestyle() == "--" and len(set(ln.get_xdata())) == 1
        ]
        assert len(vlines) == 1
        assert float(vlines[0].get_xdata()[0]) == pytest.approx(
            theta_mean["t"].mean(), abs=0.02
        )
        plt.close("all")


class TestSimulationRouting:
    def test_obs_conditions_every_simulated_layer(self, monkeypatch):
        import hssm.plotting.model_cartoon as mc

        theta_mean, theta_samples = _theta_frames()
        calls: list[dict] = []
        real_simulator = mc.simulator

        def recording(*args, **kw):
            calls.append(kw)
            return real_simulator(*args, **kw)

        monkeypatch.setattr(mc, "simulator", recording)
        fig, ax = plt.subplots()
        plot_func_model(
            model_name="ddm",
            axis=ax,
            theta_mean=theta_mean,
            theta_samples=theta_samples,
            data=_observed(),
            n_reps=2,
            random_state=0,
            uncertainty="band",
            intervals=INTERVALS,
            obs=2,
        )
        plt.close("all")

        # every simulated layer conditions on exactly one trial's θ
        for kw in calls:
            assert np.asarray(kw["theta"]).shape[0] == 1
        no_noise = [kw for kw in calls if kw.get("no_noise")]
        np.testing.assert_allclose(no_noise[0]["theta"], theta_mean.loc[[2]].values)
        expected_draws = theta_samples.xs(2, level="obs_n")
        np.testing.assert_allclose(
            no_noise[1]["theta"], expected_draws.iloc[[0]].values
        )

    def test_default_histograms_stay_marginal(self, monkeypatch):
        import hssm.plotting.model_cartoon as mc

        theta_mean, _ = _theta_frames()
        calls: list[dict] = []
        real_simulator = mc.simulator

        def recording(*args, **kw):
            calls.append(kw)
            return real_simulator(*args, **kw)

        monkeypatch.setattr(mc, "simulator", recording)
        _render("band")
        plt.close("all")

        n_obs = theta_mean.shape[0]
        no_noise = [kw for kw in calls if kw.get("no_noise")]
        noisy = [kw for kw in calls if not kw.get("no_noise")]
        # geometry: one reduced θ per call; histograms: all trial rows
        for kw in no_noise:
            assert np.asarray(kw["theta"]).shape[0] == 1
        for kw in noisy:
            assert np.asarray(kw["theta"]).shape[0] == n_obs
        # reference geometry is the across-trials mean of theta_mean
        np.testing.assert_allclose(no_noise[0]["theta"][0], theta_mean.mean().values)

    def test_trajectories_realize_reference_theta(self, monkeypatch):
        """Trajectories are noisy realizations of the SAME reduced θ that
        draws the reference geometry — their crossing markers land on the
        drawn boundary by construction (was: a random trial's θ each)."""
        import hssm.plotting.model_cartoon as mc

        theta_mean, _ = _theta_frames()
        calls: list[dict] = []
        real_simulator = mc.simulator

        def recording(*args, **kw):
            calls.append(kw)
            return real_simulator(*args, **kw)

        monkeypatch.setattr(mc, "simulator", recording)
        fig, ax, *_ = _render("band", n_trajectories=3)

        traj = [kw for kw in calls if not kw.get("no_noise") and "smooth_unif" in kw]
        assert len(traj) == 3
        expected = theta_mean.mean().to_frame().T.values
        for kw in traj:
            np.testing.assert_allclose(kw["theta"], expected)

        # crossing markers sit exactly on the (constant, a=1.2) drawn bound
        markers = [c for c in ax.collections if c.get_zorder() >= 2000]
        assert markers
        for m in markers:
            assert abs(m.get_offsets()[0][1]) == pytest.approx(1.2, rel=1e-6)
        plt.close("all")

    def test_max_t_derived_from_xlims(self, monkeypatch):
        import hssm.plotting.model_cartoon as mc

        calls: list[dict] = []
        real_simulator = mc.simulator

        def recording(*args, **kw):
            calls.append(kw)
            return real_simulator(*args, **kw)

        monkeypatch.setattr(mc, "simulator", recording)
        _render("band", n_trajectories=2, xlims=(0, 2))
        plt.close("all")

        no_noise = [c for c in calls if c.get("no_noise")]
        noisy = [c for c in calls if not c.get("no_noise")]
        traj = [c for c in noisy if c.get("smooth_unif") is False]
        rt_hist = [c for c in noisy if "smooth_unif" not in c]
        assert no_noise and traj and rt_hist

        # geometry + trajectories: horizon derived from xlims
        for c in no_noise + traj:
            assert c["max_t"] == pytest.approx(2.5)
        # noisy RT sims keep the simulator's long default horizon (shrinking
        # it would censor slow RTs and re-normalize the defective densities)
        for c in rt_hist:
            assert "max_t" not in c


class TestUncertainHistogramLayers:
    def _densities(self, n_draws=6, n_bins=10):
        rng = np.random.default_rng(0)
        return np.abs(rng.normal(1.0, 0.1, size=(n_draws, n_bins)))

    def test_band_mode_artists(self):
        fig, ax = plt.subplots()
        edges = np.linspace(0, 1, 11)
        _add_uncertain_histograms(
            ax,
            edges,
            bottom=1.5,
            densities=self._densities(),
            uncertainty="band",
            intervals=INTERVALS,
            step=True,
            scale=1.0,
            color="red",
            linestyle="-",
            linewidth=1.25,
            alpha_mean=1.0,
            alpha_uncertainty=None,
            add_labels=True,
        )
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(polys) == 2
        # ladder: widest (94%) drawn first at 0.45*base, narrowest at base
        assert polys[0].get_alpha() == pytest.approx(0.1125)
        assert polys[1].get_alpha() == pytest.approx(0.25)
        assert all(p.get_zorder() == 2 for p in polys)
        (mean_line,) = ax.get_lines()
        assert mean_line.get_zorder() == 3
        # everything sits on the bottom offset
        assert mean_line.get_ydata().min() >= 1.5
        plt.close("all")

    def test_samples_mode_artists_and_auto_alpha(self):
        fig, ax = plt.subplots()
        n_draws = 6
        _add_uncertain_histograms(
            ax,
            np.linspace(0, 1, 11),
            bottom=0.0,
            densities=self._densities(n_draws),
            uncertainty="samples",
            intervals=None,
            step=True,
            scale=1.0,
            color="red",
            linestyle="-",
            linewidth=1.25,
            alpha_mean=1.0,
            alpha_uncertainty=None,
            add_labels=True,
        )
        lines = ax.get_lines()
        assert len(lines) == n_draws + 1  # spaghetti + mean
        expected = float(np.clip(4.0 / n_draws, 0.02, 0.25))
        assert lines[0].get_alpha() == pytest.approx(expected)
        assert lines[0].get_linewidth() == pytest.approx(0.6 * 1.25)
        labels = [ln.get_label() for ln in lines]
        assert labels.count("predictive draws") == 1
        assert labels.count("predicted") == 1
        assert not ax.collections
        plt.close("all")

    def test_alpha_uncertainty_zero_is_respected(self):
        fig, ax = plt.subplots()
        _add_uncertain_histograms(
            ax,
            np.linspace(0, 1, 6),
            bottom=0.0,
            densities=self._densities(4, 5),
            uncertainty="band",
            intervals=[(0.03, 0.97)],
            step=True,
            scale=1.0,
            color="red",
            linestyle="-",
            linewidth=1.25,
            alpha_mean=1.0,
            alpha_uncertainty=0.0,
            add_labels=True,
        )
        assert ax.collections[0].get_alpha() == 0.0
        plt.close("all")

    def test_step_and_polygon_point_counts(self):
        edges = np.linspace(0, 1, 18)  # 17 bins
        for step, n_expected in ((True, 34), (False, 17)):
            fig, ax = plt.subplots()
            _add_uncertain_histograms(
                ax,
                edges,
                bottom=0.0,
                densities=self._densities(2, 17),
                uncertainty=None,
                intervals=None,
                step=step,
                scale=1.0,
                color="k",
                linestyle="-",
                linewidth=1.0,
                alpha_mean=1.0,
                alpha_uncertainty=None,
                add_labels=True,
            )
            assert len(ax.get_lines()[0].get_xdata()) == n_expected
            plt.close("all")


class TestPlotFuncModel:
    def test_band_mode_full_census(self):
        fig, ax, up, down = _render("band")
        for twin in (up, down):
            polys = [c for c in twin.collections if isinstance(c, PolyCollection)]
            assert len(polys) == 2
            assert all(p.get_linewidth() == 0 for p in polys)
            # mean (zorder 3) + observed (zorder 4)
            zorders = sorted(ln.get_zorder() for ln in twin.get_lines())
            assert zorders == [3, 4]
        # geometry: 4 boundary ribbons (zorder 1010) + one drift band per
        # interval (zorder 1012) on the main axis — all filled, none dashed
        polys_main = [c for c in ax.collections if isinstance(c, PolyCollection)]
        ribbons = [c for c in polys_main if c.get_zorder() == 1010]
        drift_bands = [c for c in polys_main if c.get_zorder() == 1012]
        assert len(ribbons) == 4
        assert len(drift_bands) == len(INTERVALS)
        # the dashed linestyle stays reserved for the ndt reference line:
        # no dashed non-vertical lines exist in band mode
        dashed_nonvertical = [
            ln
            for ln in ax.get_lines()
            if ln.get_linestyle() == "--" and len(set(ln.get_xdata())) > 1
        ]
        assert not dashed_nonvertical
        # ndt spans land in patches, one per interval
        assert len(ax.patches) == len(INTERVALS)
        # exactly ONE dashed vertical line (the reference ndt), none per draw
        dashed_vlines = [
            ln
            for ln in ax.get_lines()
            if ln.get_linestyle() == "--" and len(set(ln.get_xdata())) == 1
        ]
        assert len(dashed_vlines) == 1
        plt.close("all")

    def test_samples_mode_census(self):
        fig, ax, up, down = _render("samples")
        n_draws = 6  # 2 chains x 3 draws
        # per twin: n_draws spaghetti + mean + observed
        assert len(up.get_lines()) == n_draws + 2
        expected_alpha = float(np.clip(4.0 / n_draws, 0.02, 0.25))
        assert up.get_lines()[0].get_alpha() == pytest.approx(expected_alpha)
        # no histogram or boundary bands anywhere ...
        assert not [c for c in up.collections if isinstance(c, PolyCollection)]
        assert not [c for c in ax.collections if isinstance(c, PolyCollection)]
        # ... but the ndt envelope (graded axvspans) is drawn in every mode
        assert len(ax.patches) == len(INTERVALS)
        # and no per-draw rug/axvline artifacts survive
        assert not [ln for ln in ax.get_lines() if ln.get_label() == "ndt draws"]
        plt.close("all")

    def test_uncertainty_none_reproduces_legacy_census(self):
        fig, ax, up, down = _render(None)
        # twins: plug-in mean + observed lines only, no collections
        for twin in (up, down):
            assert len(twin.get_lines()) == 2
            assert not twin.collections
        # main axis: reference geometry only — 2 bounds + drift + ndt line...
        assert len(ax.get_lines()) == 4
        # ...and the start-point scatter
        assert len(ax.collections) == 1
        assert not ax.patches
        plt.close("all")

    def test_legacy_value_pin_plugin_mean(self):
        """The None-mode mean curve equals the independently computed plug-in."""
        from ssms.basic_simulators.simulator import simulator

        theta_mean, _ = _theta_frames()
        # reproduce the renderer's seeding protocol: one Generator, simulator
        # seeds drawn positionally (the plug-in sim's seed is the first draw)
        rng = np.random.default_rng(0)
        sim_out = simulator(
            model="ddm",
            theta=theta_mean.values,
            n_samples=2,
            no_noise=False,
            delta_t=0.01,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        edges = np.arange(-0.05, 5, 0.05)
        expected_up, _ = _defective_densities(sim_out["rts"], sim_out["choices"], edges)
        fig, ax, up, down = _render(None, data=None)
        x, y = _curve_xy(edges, expected_up, True)
        mean_line = up.get_lines()[0]
        bottom = mean_line.get_ydata().min() if expected_up.min() == 0 else None
        np.testing.assert_allclose(
            mean_line.get_ydata() - (mean_line.get_ydata() - y).min(), y, atol=1e-8
        )
        plt.close("all")

    def test_zorder_semantic_stack(self):
        fig, ax, up, down = _render("both")
        ribbons = [c for c in ax.collections if isinstance(c, PolyCollection)]
        ref_lines = [ln for ln in ax.get_lines() if ln.get_zorder() >= 1030]
        spaghetti = [ln for ln in ax.get_lines() if ln.get_zorder() == 1000]
        assert ribbons and ref_lines and spaghetti
        assert max(ln.get_zorder() for ln in spaghetti) < min(
            r.get_zorder() for r in ribbons
        )
        assert max(r.get_zorder() for r in ribbons) < min(
            ln.get_zorder() for ln in ref_lines
        )
        plt.close("all")

    def test_legend_labels_and_off(self):
        fig, ax, up, down = _render("both")
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert "observed" in labels and "predicted" in labels
        assert "94% interval" in labels and "50% interval" in labels
        assert "predictive draws" in labels
        assert "_nolegend_" not in labels
        # ordered: observed first, then predicted (the shared ranking)
        assert labels.index("observed") < labels.index("predicted")
        plt.close("all")

        fig, ax, up, down = _render("band", legend=False)
        assert ax.get_legend() is None
        plt.close("all")

    def test_bins_and_hist_height_live(self):
        fig, ax, up, down = _render("band", bins=17, hist_height=0.8)
        mean_line = [ln for ln in up.get_lines() if ln.get_zorder() == 3][0]
        assert len(mean_line.get_xdata()) == 34  # 17 bins, step geometry

        def tops(twins):
            out = []
            for twin in twins:
                for ln in twin.get_lines():
                    y = ln.get_ydata()
                    out.append(np.max(np.abs(y - y.min())))
                for coll in twin.collections:
                    for path in coll.get_paths():
                        y = path.vertices[:, 1]
                        out.append(np.max(np.abs(y - y.min())))
            return out

        # The scale normalizes the per-draw density MATRIX maximum, so band
        # mode (which draws quantiles, all <= the matrix max) stays under h...
        assert max(tops((up, down))) <= 0.8 + 1e-9
        plt.close("all")

        # ...and samples mode, which draws every row, hits h exactly.
        fig, ax, up, down = _render("samples", bins=17, hist_height=0.8)
        assert max(tops((up, down))) == pytest.approx(0.8, rel=1e-6)
        plt.close("all")

    def test_styles_are_live(self):
        fig, ax, up, down = _render(
            "band",
            color_predictive="red",
            color_data="green",
            linestyle_histogram="--",
        )
        mean_line = [ln for ln in up.get_lines() if ln.get_zorder() == 3][0]
        obs_line = [ln for ln in up.get_lines() if ln.get_zorder() == 4][0]
        assert mean_line.get_color() == "red"
        assert mean_line.get_linestyle() == "--"
        assert obs_line.get_color() == "green"
        plt.close("all")

    def test_reproducible_with_seed(self):
        """Same random_state => identical figure, including every random layer."""

        def snapshot():
            fig, ax, up, down = _render("band", n_trajectories=2)
            lines = [ln.get_ydata().copy() for ln in ax.get_lines()]
            traj = [
                ln.get_ydata().copy()
                for ln in ax.get_lines()
                if 2000 <= ln.get_zorder() < 3000
            ]
            polys = [
                c.get_paths()[0].vertices.copy()
                for c in ax.collections
                if isinstance(c, PolyCollection)
            ]
            spans = [p.get_extents().bounds for p in ax.patches]
            bands = [
                c.get_paths()[0].vertices.copy()
                for c in up.collections
                if isinstance(c, PolyCollection)
            ]
            plt.close("all")
            return lines, traj, polys, spans, bands

        first, second = snapshot(), snapshot()
        assert len(first[1]) == 2  # the trajectories are actually asserted
        for a, b in zip(first, second):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)

    def test_generator_instance_accepted_and_stateful(self):
        """A Generator renders fine; two sequential renders sharing ONE
        Generator consume successive stream segments (the facet mechanism)."""

        def traj_data(rng):
            fig, ax, *_ = _render("band", n_trajectories=2, random_state=rng)
            traj = [
                ln.get_ydata().copy()
                for ln in ax.get_lines()
                if 2000 <= ln.get_zorder() < 3000
            ]
            plt.close("all")
            return traj

        shared = np.random.default_rng(5)
        a, b = traj_data(shared), traj_data(shared)
        assert any(x.shape != y.shape or not np.array_equal(x, y) for x, y in zip(a, b))
        # ...while re-seeding reproduces the first segment exactly
        c = traj_data(np.random.default_rng(5))
        for x, y in zip(a, c):
            np.testing.assert_array_equal(x, y)


class TestPlotFuncModelN:
    def _render_n(self, uncertainty, **overrides):
        idx = pd.MultiIndex.from_product(
            [range(2), range(3), range(4)], names=["chain", "draw", "obs_n"]
        )
        rng = np.random.default_rng(3)
        cols = {"v0": 1.0, "v1": 1.2, "v2": 0.8, "a": 2.0, "z": 0.4, "t": 0.2}
        theta_samples = pd.DataFrame(cols, index=idx)
        theta_samples["a"] += rng.normal(0, 0.08, len(theta_samples))
        theta_mean = pd.DataFrame({k: [v] * 4 for k, v in cols.items()})
        data = pd.DataFrame({"rt": [0.5, 0.7, 0.6, 0.9], "response": [0, 1, 2, 1]})
        fig, ax = plt.subplots()
        kwargs = dict(
            model_name="race_no_bias_3",
            axis=ax,
            theta_mean=theta_mean,
            theta_samples=None if uncertainty is None else theta_samples,
            data=data,
            n_reps=2,
            random_state=0,
            choices=[0, 1, 2],
            n_trajectories=0,
            uncertainty=uncertainty,
            intervals=INTERVALS if uncertainty in ("band", "both") else None,
        )
        kwargs.update(overrides)
        plot_func_model_n(**kwargs)
        return fig, ax

    def test_band_mode_per_choice_collections(self):
        fig, ax = self._render_n("band")
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        # 3 choices x 2 intervals (histograms) + 2 boundary ribbons
        assert len(polys) == 3 * len(INTERVALS) + len(INTERVALS)
        assert len(ax.patches) == len(INTERVALS)  # ndt spans
        colors = {ln.get_color() for ln in ax.get_lines() if ln.get_zorder() == 3}
        assert colors == {"black", "green", "blue"}  # per-choice mean curves
        plt.close("all")

    def test_per_draw_sims_respect_n_reps_and_distinct_seeds(self, monkeypatch):
        """The per-choice bands must summarize real, independent histograms:
        n_reps was silently ignored (n_samples=1) and every draw shared one
        seed, so the bands described single-rep, noise-correlated sims."""
        import hssm.plotting.model_cartoon as mc

        calls: list[dict] = []
        real_simulator = mc.simulator

        def recording(*args, **kw):
            calls.append(kw)
            return real_simulator(*args, **kw)

        monkeypatch.setattr(mc, "simulator", recording)
        self._render_n("band", n_reps=3)
        plt.close("all")

        noisy = [kw for kw in calls if not kw.get("no_noise")]
        per_draw = noisy[1:]  # the first noisy call is the plug-in sim
        assert len(per_draw) == 6  # 2 chains x 3 draws
        for kw in per_draw:
            assert kw["n_samples"] == 3
        seeds = [kw["random_state"] for kw in noisy]
        assert len(set(seeds)) == len(seeds)

    def test_none_mode_data_only_no_nameerror(self):
        """Regression: bottom was unbound when only data was passed."""
        data = pd.DataFrame({"rt": [0.5, 0.7, 0.6], "response": [0, 1, 2]})
        fig, ax = plt.subplots()
        plot_func_model_n(
            model_name="race_no_bias_3",
            axis=ax,
            data=data,
            choices=[0, 1, 2],
            n_trajectories=0,
        )
        assert len(ax.get_lines()) == 3  # one observed curve per choice
        plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

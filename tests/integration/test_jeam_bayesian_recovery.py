"""Bayesian recovery protocol and marked-slow gate for the JEAM handshake."""

import json
from dataclasses import asdict, replace
from unittest.mock import Mock

import numpy as np
import pytensor
import pytest

import hssm

pytest.importorskip("jeam")

from scripts.benchmark_jeam_bayesian_recovery import (
    DEFAULT_PRIOR_DRAWS,
    DEFAULT_PRIOR_SEED,
    PriorPredictiveCheck,
    _resolved_initial_point,
    _sample_with_resolved_init,
    run_recovery,
)
from scripts.benchmark_jeam_objective_parity import HSSM_BOUNDS, PARAMETER_ORDER

PINNED_JEAM_REVISION = "a9f547b3630ae8ff31ccec1b904e0c02fdba6d99"
MEAN_ERROR_LIMITS = {"a": 0.10, "t": 0.04, "v_x": 0.15, "v_y": 0.15}
MAX_RHAT = 1.01
MIN_ESS = 1_000.0
MAX_MCSE_MEAN = 0.005
MAX_RT_QUANTILE_ERROR = 0.12
MAX_MEAN_ANGLE_ERROR = 0.10
MAX_RESULTANT_LENGTH_ERROR = 0.08


@pytest.fixture(scope="module")
def recovery_result():
    """Run one default-float64 recovery benchmark and restore global precision."""
    original_floatx = pytensor.config.floatX
    hssm.set_floatX("float64")
    try:
        result = run_recovery()
        print(json.dumps(asdict(result), sort_keys=True))
        return result
    finally:
        hssm.set_floatX(original_floatx)


def _valid_prior_predictive() -> PriorPredictiveCheck:
    """Return a supported prior-predictive summary without drawing samples."""
    return PriorPredictiveCheck(
        shape=(1, DEFAULT_PRIOR_DRAWS, 300, 2),
        all_finite=True,
        minimum_rt=np.nextafter(0.0, 1.0),
        maximum_rt=2.0,
        minimum_angle=-np.pi,
        maximum_angle=np.nextafter(np.pi, -np.inf),
        rt_probabilities=(0.5, 0.9),
        observed_rt_quantiles=(1.0, 1.0),
        prior_rt_quantiles=(0.1, 20.0),
        prior_to_observed_rt_ratios=(0.1, 20.0),
    )


def test_supported_recovery_preflight_passes_initial_point_to_sampling():
    """Pass the frozen supported point explicitly to every Slice chain."""
    minimum_rt = 0.15
    initial_point = _resolved_initial_point(minimum_rt)
    prior = _valid_prior_predictive()
    model = Mock()
    sampling = {
        "minimum_observed_rt": minimum_rt,
        "initial_logp": -10.0,
        "chains": 4,
        "tune": 10,
        "draws": 10,
        "chain_seeds": (1, 2, 3, 4),
    }

    _sample_with_resolved_init(
        model, [], initial_point, prior_predictive=prior, **sampling
    )

    assert initial_point == (1.0, 0.075, 0.0, 0.0)
    assert all(
        HSSM_BOUNDS[name][0] < value < HSSM_BOUNDS[name][1]
        for name, value in zip(PARAMETER_ORDER, initial_point, strict=True)
    )
    assert model.sample.call_args.kwargs["initvals"] == dict(
        zip(PARAMETER_ORDER, initial_point, strict=True)
    )


@pytest.mark.parametrize(
    ("initial_point", "initial_logp", "prior_overrides", "expected_failure"),
    [
        (
            (1.0, 0.15, 0.0, 0.0),
            -10.0,
            {},
            "resolved initial point support",
        ),
        ((1.0, 0.075, 0.0, 0.0), np.nan, {}, "initial logp"),
        (
            (1.0, 0.075, 0.0, 0.0),
            -10.0,
            {"all_finite": False},
            "prior predictive finiteness",
        ),
        (
            (1.0, 0.075, 0.0, 0.0),
            -10.0,
            {"shape": (100, 300, 2)},
            "prior predictive shape",
        ),
        (
            (1.0, 0.075, 0.0, 0.0),
            -10.0,
            {
                "minimum_rt": 0.0,
                "prior_to_observed_rt_ratios": (np.nextafter(0.1, 0.0), 1.0),
            },
            "prior predictive RT support; prior predictive RT scale",
        ),
        (
            (1.0, 0.075, 0.0, 0.0),
            -10.0,
            {"rt_probabilities": (0.5,)},
            "prior predictive RT quantiles",
        ),
        (
            (1.0, 0.075, 0.0, 0.0),
            -10.0,
            {"maximum_angle": np.pi},
            "prior predictive angular support",
        ),
    ],
)
def test_invalid_recovery_preflight_stops_before_sampling(
    initial_point, initial_logp, prior_overrides, expected_failure
):
    """Reject every preflight failure family before MCMC starts."""
    model = Mock()

    with pytest.raises(RuntimeError, match=expected_failure):
        _sample_with_resolved_init(
            model,
            [],
            initial_point,
            minimum_observed_rt=0.15,
            initial_logp=initial_logp,
            prior_predictive=replace(_valid_prior_predictive(), **prior_overrides),
            chains=4,
            tune=10,
            draws=10,
            chain_seeds=(1, 2, 3, 4),
        )

    model.sample.assert_not_called()


def test_recovery_rejects_truth_outside_the_declared_parameter_order():
    """Custom truth must provide one value for every ordered parameter."""
    with pytest.raises(ValueError, match="truth must follow"):
        run_recovery(truth=(1.0, 0.1, 0.2))


@pytest.mark.slow
def test_truth_is_recovered_with_converged_slice_chains(recovery_result):
    """Every truth should be covered with small bias and reliable diagnostics."""
    for parameter in recovery_result.parameters:
        assert abs(parameter.mean_error) <= MEAN_ERROR_LIMITS[parameter.name]
        assert parameter.interval_lower <= parameter.truth <= parameter.interval_upper
        assert parameter.rhat <= MAX_RHAT
        assert parameter.ess_bulk >= MIN_ESS
        assert parameter.ess_tail >= MIN_ESS
        assert parameter.mcse_mean <= MAX_MCSE_MEAN
        assert np.isfinite(parameter.posterior_sd)
        assert 0.0 < parameter.posterior_sd
        assert parameter.mcse_mean / parameter.posterior_sd <= 0.05
        assert np.isfinite(parameter.ess_bulk_per_second)
        assert parameter.ess_bulk_per_second > 0.0


@pytest.mark.slow
def test_recovery_uses_the_declared_gradient_free_sampler(recovery_result):
    """The integration must use PyMC Slice and expose no NUTS statistics."""
    diagnostics = recovery_result.slice_diagnostics

    assert recovery_result.sampler == "pymc.Slice[a,t,v_x,v_y]"
    assert diagnostics.sample_stats == ("nstep_in", "nstep_out")
    assert diagnostics.mean_steps_in > 0.0
    assert diagnostics.mean_steps_out > 0.0
    assert recovery_result.chains == 4
    assert recovery_result.tune == recovery_result.draws == 1_500
    assert recovery_result.hdi_probability == 0.94


@pytest.mark.slow
def test_prior_predictive_and_initial_logp_are_valid(recovery_result):
    """The black-box path must initialize and respect its response support."""
    prior = recovery_result.prior_predictive

    assert recovery_result.initial_point == (
        1.0,
        min(0.1, recovery_result.minimum_observed_rt / 2.0),
        0.0,
        0.0,
    )
    assert 0.0 < recovery_result.initial_point[1] < recovery_result.minimum_observed_rt
    assert np.isfinite(recovery_result.initial_logp)
    assert recovery_result.prior_draws == DEFAULT_PRIOR_DRAWS
    assert recovery_result.prior_seed == DEFAULT_PRIOR_SEED
    assert recovery_result.prior_predictive_seconds > 0.0
    assert prior.shape == (1, DEFAULT_PRIOR_DRAWS, 300, 2)
    assert prior.all_finite
    assert 0.0 < prior.minimum_rt <= prior.maximum_rt
    assert -np.pi <= prior.minimum_angle <= prior.maximum_angle < np.pi
    assert prior.rt_probabilities == (0.5, 0.9)
    assert len(prior.observed_rt_quantiles) == len(prior.prior_rt_quantiles) == 2
    assert all(0.1 <= ratio <= 20.0 for ratio in prior.prior_to_observed_rt_ratios)


@pytest.mark.slow
def test_posterior_predictive_recovers_rt_and_circular_summaries(recovery_result):
    """Predictive RT quantiles and circular moments should reproduce the data."""
    predictive = recovery_result.predictive
    np.testing.assert_allclose(
        predictive.predictive_rt_quantiles,
        predictive.observed_rt_quantiles,
        rtol=0.0,
        atol=MAX_RT_QUANTILE_ERROR,
    )
    assert predictive.mean_angle_distance <= MAX_MEAN_ANGLE_ERROR
    assert (
        abs(
            predictive.predictive_resultant_length
            - predictive.observed_resultant_length
        )
        <= MAX_RESULTANT_LENGTH_ERROR
    )


@pytest.mark.slow
def test_recovery_report_records_provenance_and_efficiency(recovery_result):
    """The scientific gate should remain reproducible and machine-readable."""
    assert recovery_result.jeam_revision == PINNED_JEAM_REVISION
    assert recovery_result.pytensor_floatx == "float64"
    assert recovery_result.trials == 300
    assert recovery_result.prior_draws == DEFAULT_PRIOR_DRAWS
    assert recovery_result.prior_seed == DEFAULT_PRIOR_SEED
    assert recovery_result.predictive_draws == 100
    assert recovery_result.predictive_seed == 7291
    assert recovery_result.sampling_seconds > 0.0
    assert recovery_result.predictive_seconds > 0.0

    payload = json.loads(json.dumps(asdict(recovery_result)))
    assert payload["sampler"] == "pymc.Slice[a,t,v_x,v_y]"
    assert payload["jeam_revision"] == PINNED_JEAM_REVISION
    assert payload["predictive_draws"] == 100
    assert payload["predictive_seed"] == 7291

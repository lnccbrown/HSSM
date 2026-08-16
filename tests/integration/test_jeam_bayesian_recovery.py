"""Marked-slow Bayesian recovery gate for the circular JEAM handshake."""

import json
from dataclasses import asdict

import numpy as np
import pytensor
import pytest

import hssm

pytest.importorskip("jeam")
pytestmark = pytest.mark.slow

from scripts.benchmark_jeam_bayesian_recovery import run_recovery

PINNED_JEAM_REVISION = "0c0ef8b834dd062ad8aea5ff8e7a09dfb55492ce"
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


def test_truth_is_recovered_with_converged_slice_chains(recovery_result):
    """Every truth should be covered with small bias and reliable diagnostics."""
    for parameter in recovery_result.parameters:
        assert abs(parameter.mean_error) <= MEAN_ERROR_LIMITS[parameter.name]
        assert parameter.interval_lower <= parameter.truth <= parameter.interval_upper
        assert parameter.rhat <= MAX_RHAT
        assert parameter.ess_bulk >= MIN_ESS
        assert parameter.ess_tail >= MIN_ESS
        assert parameter.mcse_mean <= MAX_MCSE_MEAN
        assert np.isfinite(parameter.ess_bulk_per_second)
        assert parameter.ess_bulk_per_second > 0.0


def test_recovery_uses_the_declared_gradient_free_sampler(recovery_result):
    """The integration must use PyMC Slice and expose no NUTS statistics."""
    diagnostics = recovery_result.slice_diagnostics

    assert recovery_result.sampler == "pymc.Slice[a,t,v_x,v_y]"
    assert diagnostics.sample_stats == ("nstep_in", "nstep_out")
    assert diagnostics.mean_steps_in > 0.0
    assert diagnostics.mean_steps_out > 0.0
    assert recovery_result.chains == 4
    assert recovery_result.tune == recovery_result.draws == 1_500


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


def test_recovery_report_records_provenance_and_efficiency(recovery_result):
    """The scientific gate should remain reproducible and machine-readable."""
    assert recovery_result.jeam_revision == PINNED_JEAM_REVISION
    assert recovery_result.pytensor_floatx == "float64"
    assert recovery_result.trials == 300
    assert recovery_result.sampling_seconds > 0.0
    assert recovery_result.predictive_seconds > 0.0

    payload = json.loads(json.dumps(asdict(recovery_result)))
    assert payload["sampler"] == "pymc.Slice[a,t,v_x,v_y]"
    assert payload["jeam_revision"] == PINNED_JEAM_REVISION

"""Fast deterministic objective and optimizer parity for the JEAM handshake."""

import json
from dataclasses import asdict

import numpy as np
import pytest

pytest.importorskip("jeam")

from scripts.benchmark_jeam_objective_parity import (
    DEFAULT_MAXITER,
    DEFAULT_POPSIZE,
    HSSM_BOUNDS,
    PARAMETER_ORDER,
    optimization_bounds,
    run_benchmark,
    simulate_dataset,
)

PINNED_JEAM_REVISION = "a9f547b3630ae8ff31ccec1b904e0c02fdba6d99"
OBJECTIVE_RTOL = 1e-6
OBJECTIVE_ATOL = 5e-5


@pytest.fixture(scope="module")
def parity_result():
    """Run the fixed 300-trial benchmark once for all parity assertions."""
    return run_benchmark(timing_repeats=3)


def test_objective_grid_matches_direct_jeam(parity_result):
    """Compiled HSSM should preserve direct JEAM objectives through the bounds."""
    np.testing.assert_allclose(
        parity_result.compiled_objectives,
        parity_result.direct_objectives,
        rtol=OBJECTIVE_RTOL,
        atol=OBJECTIVE_ATOL,
    )
    assert len(parity_result.candidate_parameters) == 4
    assert parity_result.parameter_order == PARAMETER_ORDER

    data = simulate_dataset()
    ndt_upper = optimization_bounds(data)[1][1]
    assert np.min(data[:, 0]) - ndt_upper > 1e-15
    assert parity_result.candidate_parameters[-1][1] == ndt_upper


def test_fixed_budget_optimizer_follows_the_same_path(parity_result):
    """Equal seeds and objectives should produce the same bounded DE estimate."""
    direct = parity_result.direct_fit
    compiled = parity_result.compiled_fit
    expected_evaluations = (
        (DEFAULT_MAXITER + 1) * DEFAULT_POPSIZE * len(PARAMETER_ORDER)
    )

    np.testing.assert_allclose(
        compiled.parameters,
        direct.parameters,
        rtol=0.0,
        atol=1e-12,
    )
    assert compiled.objective == pytest.approx(
        direct.objective,
        rel=OBJECTIVE_RTOL,
        abs=OBJECTIVE_ATOL,
    )
    assert direct.evaluations == compiled.evaluations == expected_evaluations
    assert direct.iterations == compiled.iterations == DEFAULT_MAXITER
    assert direct.objective < parity_result.direct_objectives[0]


def test_optimization_bounds_respect_hssm_ndt_upper_bound():
    """High-RT datasets must not expand the optimizer beyond HSSM support."""
    data = np.array([[2.5, 0.0], [3.0, 0.5]], dtype=np.float64)
    ndt_upper = optimization_bounds(data)[1][1]

    assert ndt_upper == HSSM_BOUNDS["t"][1]
    assert ndt_upper < np.min(data[:, 0])


def test_benchmark_records_provenance_and_machine_readable_output(parity_result):
    """The benchmark output should identify its producer and serialize as JSON."""
    assert parity_result.jeam_revision == PINNED_JEAM_REVISION
    assert parity_result.trials == 300
    assert parity_result.runtime.direct_jeam_ms > 0.0
    assert parity_result.runtime.compiled_hssm_ms > 0.0

    payload = json.loads(json.dumps(asdict(parity_result)))
    assert payload["jeam_revision"] == PINNED_JEAM_REVISION
    assert payload["parameter_order"] == list(PARAMETER_ORDER)

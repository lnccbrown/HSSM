"""Fast contracts for durable repeated-recovery evidence capture."""

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("jeam")

import scripts.benchmark_jeam_repeated_recovery as repeated
from scripts.benchmark_jeam_objective_parity import FitSummary

DATA = np.array([[0.2, -0.5], [0.3, 0.75]], dtype=np.float64)


def _recovery_result() -> SimpleNamespace:
    return SimpleNamespace(
        sampler="pymc.Slice[a,t,v_x,v_y]",
        hdi_probability=0.94,
        minimum_observed_rt=0.2,
        initial_point=(1.0, 0.05, 0.0, 0.0),
        initial_logp=-10.0,
        slice_diagnostics=SimpleNamespace(),
        prior_predictive=SimpleNamespace(),
        predictive=SimpleNamespace(),
        prior_predictive_seconds=0.25,
        sampling_seconds=2.0,
        predictive_seconds=0.5,
    )


def _stub_scenario(monkeypatch: pytest.MonkeyPatch) -> Mock:
    scenario = repeated.DEFAULT_SCENARIOS[0]
    monkeypatch.setattr(repeated.objective, "simulate_dataset", lambda **_: DATA)

    def objective(received):
        assert received is DATA
        return lambda _: 4.0

    monkeypatch.setattr(repeated.objective, "make_direct_objective", objective)
    monkeypatch.setattr(repeated.objective, "make_compiled_hssm_objective", objective)
    monkeypatch.setattr(
        repeated.objective, "optimization_bounds", lambda data: ((0.1, 2.0),) * 4
    )
    monkeypatch.setattr(
        repeated.objective,
        "_fit",
        lambda *_, **__: FitSummary(scenario.truth, 4.0, 10, 2),
    )
    recovery = Mock(return_value=_recovery_result())
    monkeypatch.setattr(repeated.bayesian, "run_recovery", recovery)
    monkeypatch.setattr(repeated, "_scenario_parameters", lambda *_: ())
    return recovery


def _run_scenario(writer=None):
    return repeated.run_scenario(
        repeated.DEFAULT_SCENARIOS[0],
        trials=2,
        tune=3,
        draws=4,
        prior_draws=2,
        predictive_draws=2,
        optimizer_maxiter=2,
        optimizer_popsize=3,
        evidence_writer=writer,
    )


def test_scenario_reuses_one_raw_dataset_and_records_only_raw_measurements(
    monkeypatch,
):
    """Reuse one dataset and exclude derived gate values from measurements."""
    scenario = repeated.DEFAULT_SCENARIOS[0]
    recovery = _stub_scenario(monkeypatch)
    writer = Mock()
    result = _run_scenario(writer)

    recorded = writer.record_dataset.call_args.args[0]
    assert recorded is DATA
    assert recovery.call_args.kwargs["data"] is recorded
    assert recovery.call_args.kwargs["evidence_writer"] is writer
    measurements = writer.write_measurements.call_args.args[0]
    expected_scenario = asdict(scenario) | {
        "truth": list(scenario.truth),
        "chain_seeds": list(scenario.chain_seeds),
        "trials": 2,
        "tune": 3,
        "draws": 4,
        "prior_draws": 2,
        "predictive_draws": 2,
        "optimizer_maxiter": 2,
        "optimizer_popsize": 3,
    }
    assert measurements["scenario"] == expected_scenario
    assert measurements["objective"]["candidates"] == [
        list(candidate) for candidate in result.objective_candidates
    ]
    assert measurements["objective"]["direct_values"] == [4.0] * 3
    assert measurements["initialization"] == {
        "minimum_observed_rt": 0.2,
        "point": [1.0, 0.05, 0.0, 0.0],
        "logp": -10.0,
    }
    assert {"gate", "aggregate"}.isdisjoint(measurements)

    _run_scenario()
    assert recovery.call_args.kwargs["data"] is DATA
    assert recovery.call_args.kwargs["evidence_writer"] is None


def test_evidence_run_preflights_scenarios_then_finalizes_protocol(
    tmp_path, monkeypatch
):
    """Preflight first, finalize last, and freeze the resolved protocol."""
    root, events, finalized = tmp_path / "evidence", [], {}

    def prepare(directory, **kwargs):
        events.append("prepare")
        assert (directory, kwargs["protocol_base_revision"]) == (
            root,
            repeated.PROTOCOL_BASE_REVISION,
        )
        assert kwargs["source_paths"] == repeated.EVIDENCE_SOURCE_PATHS
        return {"producer_revision": "revision"}

    class Writer:
        def __init__(self, directory):
            events.append(f"writer:{directory.name}")

    def scenario(item, **kwargs):
        assert isinstance(kwargs["evidence_writer"], Writer)
        events.append(f"scenario:{item.name}")
        return item.name

    def finalize(directory, **kwargs):
        events.append("finalize")
        assert directory == root
        finalized.update(kwargs)

    monkeypatch.setattr(repeated, "prepare_evidence_bundle", prepare)
    monkeypatch.setattr(repeated, "ScenarioEvidenceWriter", Writer)
    monkeypatch.setattr(repeated, "run_scenario", scenario)
    monkeypatch.setattr(repeated, "aggregate_results", lambda _: ())
    monkeypatch.setattr(
        repeated, "evaluate_gate", lambda *_, **__: repeated.GateResult(True, ())
    )
    monkeypatch.setattr(repeated, "finalize_evidence_bundle", finalize)
    monkeypatch.setattr(repeated.objective, "_installed_jeam_revision", lambda: "jeam")

    result = repeated.run_evidence_benchmark(root)

    names = [item.name for item in repeated.DEFAULT_SCENARIOS]
    assert events == [
        "prepare",
        *(event for name in names for event in (f"writer:{name}", f"scenario:{name}")),
        "finalize",
    ]
    assert result.scenarios == tuple(names)
    assert finalized["provenance"] == {"producer_revision": "revision"}
    protocol = finalized["protocol"]
    assert (
        protocol
        | {
            "schema_version": 1,
            "result_schema_version": 2,
            "trials_per_scenario": repeated.DEFAULT_TRIALS,
        }
        == protocol
    )
    assert protocol["scenarios"] == [
        asdict(item) for item in repeated.DEFAULT_SCENARIOS
    ]
    assert finalized["result"]["schema_version"] == 2

"""Fast contracts for durable JEAM Bayesian recovery capture."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("jeam")

from scripts import benchmark_jeam_bayesian_recovery as recovery

DATA = np.array([[0.8, -0.4], [1.2, 0.7]], dtype=np.float64)
RESPONSE_DIMS = ("chain", "draw", "observation", "response_dim")


class _ModelReached(Exception):
    pass


def _stop_model(events):
    def constructor(*, data, **_):
        events.append(("model", data))
        raise _ModelReached

    return constructor


def _response_tree(group: str) -> xr.DataTree:
    return xr.DataTree.from_dict(
        {group: xr.Dataset({"rt,response": (RESPONSE_DIMS, DATA.reshape(1, 1, 2, 2))})}
    )


def test_supplied_data_is_copied_without_simulation_and_default_path_still_simulates(
    monkeypatch,
):
    """Use a frozen supplied copy or the unchanged simulation default."""
    supplied = DATA.copy()
    events = []
    writer = Mock()
    writer.record_dataset.side_effect = lambda value: events.append(("dataset", value))
    simulator = Mock(side_effect=AssertionError)
    monkeypatch.setattr(recovery, "simulate_dataset", simulator)
    monkeypatch.setattr(recovery.hssm, "HSSM", _stop_model(events))

    with pytest.raises(_ModelReached):
        recovery.run_recovery(data=supplied, trials=2, evidence_writer=writer)

    assert [name for name, _ in events] == ["dataset", "model"]
    recorded, modeled = events[0][1], events[1][1]
    assert recorded is not supplied
    assert (recorded.dtype, recorded.flags.c_contiguous, recorded.flags.writeable) == (
        np.dtype("float64"),
        True,
        False,
    )
    assert isinstance(modeled, pd.DataFrame)
    np.testing.assert_array_equal(modeled.to_numpy(), recorded)
    supplied[0, 0] = 99
    assert recorded[0, 0] == DATA[0, 0]

    simulator.side_effect = None
    simulator.return_value = DATA
    with pytest.raises(_ModelReached):
        recovery.run_recovery(trials=2)
    simulator.assert_called_once()
    assert [name for name, _ in events] == ["dataset", "model", "model"]


@pytest.mark.parametrize(
    ("data", "trials", "message"),
    [
        ([[0.8, 0.0]], 1, "numpy.ndarray"),
        (np.array([[0.8, 0.0]], dtype=np.float32), 1, "float64"),
        (np.array([0.8, 0.0]), 1, "shape"),
        (np.array([[0.8, 0.0]]), 2, "shape"),
        (np.array([[np.nan, 0.0]]), 1, "finite"),
        (np.array([[0.0, 0.0]]), 1, "strictly positive"),
        (np.array([[0.8, -np.pi - 0.01]]), 1, "angle"),
        (np.array([[0.8, np.pi]]), 1, "angle"),
    ],
)
def test_invalid_supplied_data_has_no_side_effects(data, trials, message, monkeypatch):
    """Fail every invalid data class before simulation, recording, or modeling."""
    simulator, model, writer = Mock(), Mock(), Mock()
    monkeypatch.setattr(recovery, "simulate_dataset", simulator)
    monkeypatch.setattr(recovery.hssm, "HSSM", model)

    with pytest.raises((TypeError, ValueError), match=message):
        recovery.run_recovery(data=data, trials=trials, evidence_writer=writer)

    simulator.assert_not_called()
    model.assert_not_called()
    assert not writer.method_calls


def test_writer_records_raw_stages_with_posterior_before_summary(monkeypatch):
    """Record each raw object immediately, with posterior before reduction."""
    prior = _response_tree("prior_predictive")
    traces = xr.DataTree.from_dict(
        {
            "sample_stats": xr.Dataset(
                {
                    name: (("chain", "draw"), [[1.0, 1.0]])
                    for name in ("nstep_in", "nstep_out")
                }
            )
        }
    )
    predictive = _response_tree("posterior_predictive")
    model = Mock(pymc_model={name: object() for name in recovery.PARAMETER_ORDER})
    model.compile_logp.return_value = lambda _: -12.0
    model.sample_prior_predictive.return_value = prior
    model.sample_posterior_predictive.return_value = predictive
    events = []
    writer = Mock()
    for method, name in (
        (writer.record_dataset, "dataset"),
        (writer.record_prior, "prior"),
        (writer.record_posterior, "posterior"),
        (writer.record_predictive, "predictive"),
    ):
        method.side_effect = lambda value, name=name: events.append((name, value))

    def summarize(value, **_):
        events.append(("summary", value))
        return pd.DataFrame(
            {
                "mean": [1.0, 0.1, 0.0, 0.0],
                "sd": [0.1] * 4,
                "hdi_3%_lb": [0.8, 0.08, -0.2, -0.2],
                "hdi_97%_ub": [1.2, 0.12, 0.2, 0.2],
                "r_hat": [1.0] * 4,
                "ess_bulk": [100.0] * 4,
                "ess_tail": [100.0] * 4,
                "mcse_mean": [0.001] * 4,
            },
            index=recovery.PARAMETER_ORDER,
        )

    monkeypatch.setattr(recovery, "simulate_dataset", Mock(side_effect=AssertionError))
    monkeypatch.setattr(recovery.hssm, "HSSM", Mock(return_value=model))
    monkeypatch.setattr(recovery.pm, "Slice", lambda **_: object())
    monkeypatch.setattr(recovery, "_sample_with_resolved_init", lambda *_, **__: traces)
    monkeypatch.setattr(recovery.az, "summary", summarize)
    monkeypatch.setattr(recovery, "_installed_jeam_revision", lambda: "revision")

    recovery.run_recovery(
        data=DATA,
        trials=2,
        chains=1,
        tune=2,
        draws=2,
        chain_seeds=(1,),
        prior_draws=1,
        predictive_draws=1,
        evidence_writer=writer,
    )

    assert [name for name, _ in events] == [
        "dataset",
        "prior",
        "posterior",
        "summary",
        "predictive",
    ]
    assert [value for _, value in events[1:]] == [prior, traces, traces, predictive]

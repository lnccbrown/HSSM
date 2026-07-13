"""Tests for HSSM posterior-predictive sampling."""

import numpy as np
import pytest
import xarray as xr

import hssm

hssm.set_floatX("float32")

PARAMETER_NAMES = "draws,safe_mode,inplace"
PARAMETER_GRID = [
    (1, False, False),
    (1, True, False),
    # (None, False, False), # slow test...
    (None, True, False),
    # (50, False, False),
    (50, True, False),
    # (np.arange(500), False, False), # very slow to test
    (np.arange(500), True, False),
    (1, False, True),
    (1, True, True),
    # (None, False, True), # very slow to test
    (None, True, True),
    (50, False, True),
    (50, True, True),
    (np.arange(50), False, True),
    (np.arange(500), True, True),
    ([1, 2, 3, 4, 5], False, False),
    ([1, 2, 3, 4, 5], True, False),
    ([1, 2, 3, 4, 5], False, True),
    ([1, 2, 3, 4, 5], True, True),
]


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_sample_posterior_predictive(cav_dt, cavanagh_test, draws, safe_mode, inplace):
    """Test sample_posterior_predictive method."""
    model = hssm.HSSM(
        data=cavanagh_test,
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                    "theta": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                },
                "formula": "v ~ theta + (1|participant_id)",
                "link": "identity",
            },
        ],
    )  # Doesn't matter what model or data we use here
    if "posterior_predictive" in cav_dt:
        del cav_dt["posterior_predictive"]
    cav_dt_copy = cav_dt.copy()

    posterior_predictive = model.sample_posterior_predictive(
        dt=cav_dt_copy, draws=draws, safe_mode=safe_mode, inplace=inplace
    )

    if draws is None:
        size = 500
    elif isinstance(draws, int):
        size = draws
    elif isinstance(draws, np.ndarray):
        size = draws.size
    elif isinstance(draws, list):
        size = len(draws)
    else:
        raise ValueError("draws must be int, None, np.ndarray, or list")

    try:
        if inplace:
            assert "posterior_predictive" in cav_dt_copy
            assert cav_dt_copy.posterior_predictive.draw.size == size
        else:
            assert posterior_predictive is not None
            assert "posterior_predictive" not in cav_dt_copy
            assert posterior_predictive.posterior_predictive.draw.size == size
    except AssertionError:
        raise


def test_sample_posterior_predictive_uses_attached_traces_for_response_params(
    data_ddm, minimal_posterior_datatree, monkeypatch
):
    """Response-parameter prediction delegates with attached traces by default."""
    model = hssm.HSSM(data=data_ddm)
    traces = minimal_posterior_datatree()
    expected = traces.copy(deep=True)
    model._inference_obj = traces
    calls = []

    def fake_predict(dt, kind, data, inplace, include_group_specific):
        calls.append((dt, kind, data, inplace, include_group_specific))
        return expected

    monkeypatch.setattr(model.model, "predict", fake_predict)

    result = model.sample_posterior_predictive(
        kind="response_params",
        inplace=False,
        include_group_specific=False,
    )

    assert result is expected
    assert len(calls) == 1
    assert calls[0][0] is traces
    assert calls[0][1:] == ("response_params", None, False, False)


def test_sample_posterior_predictive_replaces_existing_group_inplace(
    caplog, data_ddm, minimal_posterior_datatree, monkeypatch
):
    """An explicit in-place prediction removes stale draws before replacement."""
    model = hssm.HSSM(data=data_ddm)
    traces = minimal_posterior_datatree(include_posterior_predictive=True)
    replacement = xr.Dataset(
        {"prediction": (("chain", "draw"), np.array([[1.0, 2.0]]))},
        coords={"chain": [0], "draw": [0, 1]},
    )
    calls = []

    def fake_predict(dt, kind, data, inplace, include_group_specific):
        calls.append((dt, kind, data, inplace, include_group_specific))
        assert "posterior_predictive" not in dt
        dt["posterior_predictive"] = replacement

    monkeypatch.setattr(model.model, "predict", fake_predict)

    result = model.sample_posterior_predictive(
        dt=traces,
        kind="response_params",
        inplace=True,
    )

    assert result is None
    assert len(calls) == 1
    assert calls[0][0] is traces
    assert calls[0][1:] == ("response_params", None, True, True)
    np.testing.assert_array_equal(
        traces["posterior_predictive"]["prediction"].values,
        np.array([[1.0, 2.0]]),
    )
    assert "pre-existing posterior_predictive group deleted" in caplog.text

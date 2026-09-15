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


@pytest.mark.xfail(
    reason="bambi 0.20 migration (#1305): R13 bambi 0.20 keeps constant parameters and `*_Intercept_centered` RVs in `posterior` (#1330)",
    strict=False,
)
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


@pytest.fixture(scope="module")
def fitted_ddm_reg(data_ddm_reg):
    """Return a regression DDM together with a short trace it was fitted on."""
    model = hssm.HSSM(data=data_ddm_reg, include=[{"name": "v", "formula": "v ~ x"}])
    dt = model.sample(draws=10, tune=10, chains=1, cores=1, progressbar=False)
    return model, dt


@pytest.fixture
def ddm_reg_dt(fitted_ddm_reg):
    """Return a fresh copy of the fitted trace without a predictive group."""
    _, dt = fitted_ddm_reg
    dt = dt.copy()
    if "posterior_predictive" in dt:
        del dt["posterior_predictive"]
    return dt


N_NEW_OBS = 40


@pytest.mark.slow
@pytest.mark.parametrize("safe_mode", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
def test_out_of_sample_prediction_lands_in_posterior_predictive(
    fitted_ddm_reg, ddm_reg_dt, data_ddm_reg, safe_mode, inplace
):
    """Out-of-sample draws are exposed as ``posterior_predictive``, not ``predictions``.

    bambi 0.20 routes ``Model.predict(data=...)`` to a ``predictions`` group (and
    ``predictions_constant_data``). HSSM keeps its documented contract: the response
    draws live in ``posterior_predictive`` with one ``__obs__`` per row of ``data``.
    """
    model, _ = fitted_ddm_reg
    new_data = data_ddm_reg.iloc[:N_NEW_OBS].reset_index(drop=True)
    posterior_vars = set(ddm_reg_dt["posterior"].to_dataset().data_vars)

    result = model.sample_posterior_predictive(
        dt=ddm_reg_dt, data=new_data, draws=5, safe_mode=safe_mode, inplace=inplace
    )

    if inplace:
        assert result is None
        out = ddm_reg_dt
    else:
        assert result is not None
        assert "posterior_predictive" not in ddm_reg_dt
        out = result

    assert "predictions" not in out
    assert "predictions_constant_data" not in out
    pps = out["posterior_predictive"].to_dataset()
    assert list(pps.data_vars) == ["rt,response"]
    assert pps.sizes["__obs__"] == N_NEW_OBS
    assert pps.sizes["draw"] == 5
    # the posterior is left untouched: no trial-wise variables sized to `data`
    assert set(out["posterior"].to_dataset().data_vars) == posterior_vars
    assert out["posterior"].to_dataset().sizes["draw"] == 10


@pytest.mark.slow
def test_out_of_sample_prediction_replaces_stale_predictions_groups(
    fitted_ddm_reg, ddm_reg_dt, data_ddm_reg
):
    """Left-over bambi ``predictions*`` groups on ``dt`` do not survive a new call."""
    model, _ = fitted_ddm_reg
    new_data = data_ddm_reg.iloc[:N_NEW_OBS].reset_index(drop=True)
    stale = xr.Dataset({"stale": (("chain", "draw"), np.zeros((1, 1)))})
    ddm_reg_dt["predictions"] = stale
    ddm_reg_dt["predictions_constant_data"] = stale

    model.sample_posterior_predictive(
        dt=ddm_reg_dt, data=new_data, draws=2, safe_mode=False, inplace=True
    )

    assert "predictions" not in ddm_reg_dt
    assert "predictions_constant_data" not in ddm_reg_dt
    pps = ddm_reg_dt["posterior_predictive"].to_dataset()
    assert pps.sizes["__obs__"] == N_NEW_OBS


@pytest.mark.slow
def test_in_sample_prediction_matches_out_of_sample_layout(
    fitted_ddm_reg, ddm_reg_dt, data_ddm_reg
):
    """In- and out-of-sample results share the same group and variable layout."""
    model, _ = fitted_ddm_reg
    in_sample = model.sample_posterior_predictive(
        dt=ddm_reg_dt, draws=2, safe_mode=False, inplace=False
    )
    out_of_sample = model.sample_posterior_predictive(
        dt=ddm_reg_dt, data=data_ddm_reg, draws=2, safe_mode=False, inplace=False
    )

    a = in_sample["posterior_predictive"].to_dataset()
    b = out_of_sample["posterior_predictive"].to_dataset()
    assert list(a.data_vars) == list(b.data_vars)
    assert a["rt,response"].dims == b["rt,response"].dims
    assert a.sizes == b.sizes


@pytest.mark.slow
@pytest.mark.parametrize("safe_mode", [True, False])
def test_in_sample_prediction_ignores_stale_predictions_group(
    fitted_ddm_reg, ddm_reg_dt, safe_mode
):
    """A stale bambi ``predictions`` group without the response is not mistaken for draws.

    A prior out-of-sample ``kind="response_params"`` call leaves a ``predictions``
    group holding only likelihood parameters. A following in-sample
    ``kind="response"`` call writes to ``posterior_predictive`` and must not try to
    read the response from the stale group.
    """
    model, _ = fitted_ddm_reg
    stale = xr.Dataset({"v": (("chain", "draw", "__obs__"), np.zeros((1, 1, 3)))})
    ddm_reg_dt["predictions"] = stale
    ddm_reg_dt["predictions_constant_data"] = stale

    out = model.sample_posterior_predictive(
        dt=ddm_reg_dt, draws=2, safe_mode=safe_mode, inplace=False
    )

    assert "predictions" not in out
    assert "predictions_constant_data" not in out
    pps = out["posterior_predictive"].to_dataset()
    assert list(pps.data_vars) == ["rt,response"]
    assert pps.sizes["draw"] == 2

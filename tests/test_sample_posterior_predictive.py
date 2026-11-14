"""Tests for HSSM posterior-predictive sampling."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import hssm

hssm.set_floatX("float32")

# Full-posterior cases run on a thinned copy of the cavanagh fixture: predicting
# over all 500 draws dominated this file's runtime, and the assertions only check
# draw counts. 20 draws still gives safe_mode two chunks of 10.
THIN_DRAWS = 20

PARAMETER_NAMES = "fixture,draws,safe_mode,inplace"
PARAMETER_GRID = [
    ("cav_dt", 1, False, False),
    ("cav_dt", 1, True, False),
    # ("cav_dt_thin", None, False, False), # slow test...
    ("cav_dt_thin", None, True, False),
    # ("cav_dt", 50, False, False),
    ("cav_dt", 50, True, False),
    # ("cav_dt_thin", np.arange(THIN_DRAWS), False, False), # very slow to test
    ("cav_dt_thin", np.arange(THIN_DRAWS), True, False),
    ("cav_dt", 1, False, True),
    ("cav_dt", 1, True, True),
    # ("cav_dt_thin", None, False, True), # very slow to test
    ("cav_dt_thin", None, True, True),
    ("cav_dt", 50, False, True),
    ("cav_dt", 50, True, True),
    ("cav_dt", np.arange(50), False, True),
    ("cav_dt_thin", np.arange(THIN_DRAWS), True, True),
    ("cav_dt", [1, 2, 3, 4, 5], False, False),
    ("cav_dt", [1, 2, 3, 4, 5], True, False),
    ("cav_dt", [1, 2, 3, 4, 5], False, True),
    ("cav_dt", [1, 2, 3, 4, 5], True, True),
]


@pytest.fixture
def cav_dt_thin(cav_dt):
    """Return the Cavanagh idata thinned to its first ``THIN_DRAWS`` draws."""
    return cav_dt.isel(
        draw=slice(0, THIN_DRAWS),  # Take only the first THIN_DRAWS draws
        missing_dims="ignore",  # Ignore any dimensions that are missing in the selection
    )


@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_sample_posterior_predictive(
    request, cavanagh_test, fixture, draws, safe_mode, inplace
):
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
    cav_dt = request.getfixturevalue(fixture)
    if "posterior_predictive" in cav_dt:
        del cav_dt["posterior_predictive"]
    cav_dt_copy = cav_dt.copy()

    posterior_predictive = model.sample_posterior_predictive(
        dt=cav_dt_copy, draws=draws, safe_mode=safe_mode, inplace=inplace
    )

    if draws is None:
        size = cav_dt.posterior.draw.size
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


@pytest.mark.parametrize("model", ["ddm", "softmax_inv_temperature_2"])
def test_posterior_predictive_with_p_outlier_regression(model):
    """A trial-wise ``p_outlier`` alone still gives one draw per trial.

    A regression on any other parameter hides a missing broadcast (#1416), so
    every other parameter stays intercept-only.
    """
    rng = np.random.default_rng(0)
    if model == "ddm":
        theta = {"v": 0.5, "a": 1.5, "z": 0.5, "t": 0.3}
        data = hssm.simulate_data(model="ddm", theta=theta, size=100, random_state=1)
    else:
        data = pd.DataFrame({"response": rng.choice([-1, 1], size=100)})
    hssm_model = hssm.HSSM(
        data=data.assign(x=rng.normal(size=100)),
        model=model,
        p_outlier={"formula": "p_outlier ~ 1 + x", "link": "logit"},
    )
    hssm_model.sample(draws=10, tune=10, chains=1, cores=1, progressbar=False)

    hssm_model.sample_posterior_predictive(draws=5)

    assert hssm_model.traces["posterior_predictive"].sizes["__obs__"] == 100


# Non-decision time fixed to a vector: 0.2 on even rows, 1.5 on odd rows.
FIXED_T = np.where(np.arange(40) % 2 == 0, 0.2, 1.5)


def _fixed_t_model():
    """Return a DDM whose ``t`` is fixed to ``FIXED_T``, without a lapse."""
    n_obs = FIXED_T.size
    theta = {
        "v": np.full(n_obs, 0.5),
        "a": np.full(n_obs, 1.0),
        "z": np.full(n_obs, 0.5),
        "t": FIXED_T,
    }
    data = hssm.simulate_data(model="ddm", theta=theta, size=1, random_state=0)
    return hssm.HSSM(data=data, model="ddm", t=FIXED_T, p_outlier=None)


@pytest.mark.parametrize("predictive", ["posterior", "prior"])
def test_predictive_simulates_fixed_vector_params(predictive):
    """Predictive draws use each row's value of a fixed-vector parameter (#1453).

    A DDM response time is ``t`` plus a positive decision time, so every
    simulated rt exceeds its own row's ``t``. The simulator used to receive
    Bambi's ``0.0`` placeholder for ``t``, so rows with ``t = 1.5`` were
    simulated with response times below 1.5.
    """
    model = _fixed_t_model()
    if predictive == "posterior":
        model.sample(
            draws=10, tune=10, chains=1, cores=1, random_seed=0, progressbar=False
        )
        # The plotting functions pass the model's own data.
        all_sims = [
            model.sample_posterior_predictive(draws=5, inplace=False, data=data)[
                "posterior_predictive"
            ]["rt,response"].values
            for data in (None, model.data)
        ]
    else:
        dt = model.sample_prior_predictive(draws=5, random_seed=0)
        all_sims = [dt["prior_predictive"]["rt,response"].values]

    for sims in all_sims:
        assert sims.shape[-2:] == (FIXED_T.size, 2)
        assert np.all(sims[..., 0] > FIXED_T)


def test_posterior_predictive_rejects_new_data_with_fixed_vector_params(
    minimal_posterior_datatree,
):
    """New rows have no value of a fixed vector, so simulating them raises."""
    model = _fixed_t_model()

    with pytest.raises(ValueError, match="fixed to a vector"):
        model.sample_posterior_predictive(
            dt=minimal_posterior_datatree(), data=model.data.iloc[:10]
        )

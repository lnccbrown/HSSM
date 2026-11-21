"""Tests for HSSM utility helpers."""

from types import SimpleNamespace

import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest
import xarray as xr
from jax import config
from scipy import stats
from ssms.basic_simulators.simulator import simulator

import hssm
from hssm.utils import (
    _compute_log_likelihood,
    _generate_random_indices,
    _random_sample,
    log_likelihood,
    set_floatX,
)

hssm.set_floatX("float32")


@pytest.mark.slow
def test_get_alias_dict():
    """Build aliases for default and regression parameterizations."""
    # Simulate some data:
    v_true, a_true, z_true, t_true = [0.5, 1.5, 0.5, 0.5]
    obs_ddm = simulator([v_true, a_true, z_true, t_true], model="ddm", n_samples=1000)
    obs_ddm = np.column_stack([obs_ddm["rts"][:, 0], obs_ddm["choices"][:, 0]])

    dataset = pd.DataFrame(obs_ddm, columns=["rt", "response"])
    dataset["x"] = dataset["rt"] * 0.1
    dataset["y"] = dataset["rt"] * 0.5
    dataset["group"] = np.random.randint(0, 10, len(dataset))

    alias_default = hssm.HSSM(data=dataset)._aliases
    alias_regression = hssm.HSSM(
        data=dataset,
        include=[
            {
                "name": "v",  # change to name
                "prior": {
                    "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
                    "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
                    "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
                },
                "formula": "v ~ (1|group) + x + y",
                "link": "identity",
            }
        ],
    )._aliases

    alias_regression_a = hssm.HSSM(
        data=dataset,
        include=[
            {
                "name": "a",  # change to name
                "prior": {
                    "Intercept": {
                        "name": "Uniform",
                        "lower": 0.0,
                        "upper": 1.0,
                        "initval": 0.5,
                    },
                    "x": {"name": "Uniform", "lower": -0.5, "upper": 0.5, "initval": 0},
                },
                "formula": "a ~ (1|group) + x",
            }
        ],
    )._aliases

    assert alias_default["c(rt, response)"] == "rt,response"
    assert alias_default["Intercept"] == "v"
    assert alias_default["v"] == "v_mean"
    assert "a" not in alias_default

    assert alias_regression["c(rt, response)"] == "rt,response"
    assert alias_regression["Intercept"] == "v_Intercept"
    assert alias_regression["x"] == "v_x"
    assert alias_regression["1|group"] == "v_1|group"

    assert alias_regression_a["c(rt, response)"] == "rt,response"
    assert alias_regression_a["Intercept"] == "a_Intercept"
    assert alias_regression_a["x"] == "a_x"
    assert alias_regression_a["1|group"] == "a_1|group"


def test_set_floatX():
    """Synchronize floating-point precision across computational backends."""
    # Should raise error when wrong value is passed.
    with pytest.raises(ValueError):
        set_floatX("bad_value")

    set_floatX("float32")
    assert pytensor.config.floatX == "float32"
    assert not config.read("jax_enable_x64")

    set_floatX("float64")
    assert pytensor.config.floatX == "float64"
    assert config.read("jax_enable_x64")


@pytest.fixture
def minimal_log_likelihood_datatree(minimal_posterior_datatree):
    """Return a factory for compact log-likelihood DataTrees."""

    def _make(include_log_likelihood=False):
        traces = minimal_posterior_datatree(
            posterior_var="theta",
            posterior_values=np.array([[0.25, 0.75]]),
        )
        if include_log_likelihood:
            traces["log_likelihood"] = xr.Dataset(
                {
                    "rt,response": (
                        ("chain", "draw", "__obs__"),
                        np.full((1, 2, 1), -99.0),
                    )
                },
                coords={"chain": [0], "draw": [0, 1], "__obs__": [0]},
            )
        return traces

    return _make


def _log_likelihood_model(family):
    def compute_likelihood_params(
        dt,
        data,
        include_group_specific,
        sample_new_groups,
    ):
        assert data is None
        assert include_group_specific is True
        assert sample_new_groups is False
        return dt

    return SimpleNamespace(
        response_component=SimpleNamespace(
            term=SimpleNamespace(alias="rt,response", name="c(rt, response)")
        ),
        family=family,
        _compute_likelihood_params=compute_likelihood_params,
    )


def _fake_log_likelihood(family, **kwargs):
    """Return deterministic likelihood values for DataTree plumbing tests."""
    assert family is kwargs["model"].family
    assert kwargs["data"] is None
    assert kwargs["posterior"].name == "posterior"
    assert kwargs["compile_mode"] is None
    return xr.DataArray(
        np.array([[[-1.0], [-2.0]]]),
        dims=("chain", "draw", "__obs__"),
        coords={"chain": [0], "draw": [0, 1], "__obs__": [0]},
    )


def test_compute_log_likelihood_returns_deep_copy(
    minimal_log_likelihood_datatree, monkeypatch
):
    """Non-in-place computation leaves the supplied posterior untouched."""
    model = _log_likelihood_model(family=object())
    traces = minimal_log_likelihood_datatree()
    monkeypatch.setattr("hssm.utils.log_likelihood", _fake_log_likelihood)

    result = _compute_log_likelihood(model, traces, data=None, inplace=False)

    assert result is not traces
    assert isinstance(result, xr.DataTree)
    assert "log_likelihood" in result
    assert "log_likelihood" not in traces
    result["posterior"]["theta"].values[0, 0] = 10.0
    assert traces["posterior"]["theta"].values[0, 0] == 0.25


def test_compute_log_likelihood_rejects_missing_family(
    minimal_log_likelihood_datatree,
):
    """Log-likelihood computation requires a configured model family."""
    model = _log_likelihood_model(family=None)
    traces = minimal_log_likelihood_datatree()

    with pytest.raises(
        ValueError,
        match="Model family is not defined. Cannot compute log-likelihood.",
    ):
        _compute_log_likelihood(model, traces, data=None)


def test_compute_log_likelihood_warns_and_replaces_existing_group(
    caplog, minimal_log_likelihood_datatree, monkeypatch
):
    """Recomputation replaces stale log-likelihood values in-place."""
    model = _log_likelihood_model(family=object())
    traces = minimal_log_likelihood_datatree(include_log_likelihood=True)
    monkeypatch.setattr("hssm.utils.log_likelihood", _fake_log_likelihood)

    result = _compute_log_likelihood(model, traces, data=None, inplace=True)

    assert result is traces
    np.testing.assert_array_equal(
        traces["log_likelihood"]["rt,response"].values,
        np.array([[[-1.0], [-2.0]]]),
    )
    assert "Replacing existing log_likelihood group in dt." in caplog.text


@pytest.mark.parametrize("bounds", [{"lb": -2.0, "ub": 2.0}, {"lb": -2.0}, {"ub": 2.0}])
@pytest.mark.parametrize("response", ["model", "explicit_y", "new_data"])
def test_log_likelihood_constrained_response(bounds, response):
    """A Bambi `constrained` response is evaluated as a truncated Normal at each draw.

    The response is read from the first column of the term's data, not from the
    bound columns, and every draw uses its own parameters rather than the first
    draw's. scipy's truncnorm is the reference.
    """
    rng = np.random.default_rng(1452)
    y = rng.uniform(-1.5, 1.5, 6)
    mu = rng.normal(0.0, 1.0, (2, 3))
    sigma = rng.uniform(0.5, 2.0, (2, 3))
    bound_args = ", ".join(f"{key}={value}" for key, value in bounds.items())
    model = bmb.Model(
        f"constrained(y, {bound_args}) ~ 1", pd.DataFrame({"y": y}), family="gaussian"
    )
    model.build()
    posterior = xr.Dataset(
        {
            "Intercept": (("chain", "draw"), mu),
            "sigma": (("chain", "draw"), sigma),
        },
        coords={"chain": [0, 1], "draw": [0, 1, 2]},
    )

    kwargs = {}
    data = None
    y_eval = y
    if response == "explicit_y":
        kwargs["y"] = y
    elif response == "new_data":
        y_eval = rng.uniform(-1.5, 1.5, 4)
        data = pd.DataFrame({"y": y_eval})
    dt = model._compute_likelihood_params(
        xr.DataTree.from_dict({"posterior": posterior}), data, True, False
    )
    result = log_likelihood(
        model.family, model, dt.children["posterior"], data=data, **kwargs
    )

    loc, scale = mu[..., None], sigma[..., None]
    lower, upper = bounds.get("lb", -np.inf), bounds.get("ub", np.inf)
    expected = stats.truncnorm.logpdf(
        y_eval, (lower - loc) / scale, (upper - loc) / scale, loc=loc, scale=scale
    )
    np.testing.assert_allclose(result.values, expected, rtol=1e-5)


def _gaussian_posterior(rng):
    """Two chains and three draws of an intercept-only Gaussian model."""
    mu = rng.normal(0.0, 1.0, (2, 3))
    sigma = rng.uniform(0.5, 2.0, (2, 3))
    posterior = xr.Dataset(
        {
            "Intercept": (("chain", "draw"), mu),
            "sigma": (("chain", "draw"), sigma),
        },
        coords={"chain": [0, 1], "draw": [0, 1, 2]},
    )
    return posterior, mu[..., None], sigma[..., None]


def _evaluate_log_likelihood(model, posterior, data=None):
    dt = model._compute_likelihood_params(
        xr.DataTree.from_dict({"posterior": posterior}), data, True, False
    )
    return log_likelihood(model.family, model, dt.children["posterior"], data=data)


@pytest.mark.parametrize("kind", ["constrained", "truncated"])
@pytest.mark.parametrize("n_new", [None, 1, 4])
def test_log_likelihood_truncated_response(kind, n_new):
    """`truncated` and `constrained` responses are truncated Normals.

    `n_new` is the number of rows of new data (None: the model's data). A single
    row must give one log-likelihood per draw, not one per column of the
    response. scipy's truncnorm is the reference.
    """
    rng = np.random.default_rng(1452)
    df = pd.DataFrame({"y": rng.uniform(-1.5, 1.5, 5)})
    model = bmb.Model(f"{kind}(y, lb=-2.0) ~ 1", df, family="gaussian")
    model.build()
    posterior, loc, scale = _gaussian_posterior(rng)

    data = None if n_new is None else pd.DataFrame({"y": rng.uniform(-1.5, 1.5, n_new)})
    n_obs = len(df if data is None else data)
    y_eval = (df if data is None else data)["y"].to_numpy()
    result = _evaluate_log_likelihood(model, posterior, data)

    expected = stats.truncnorm.logpdf(
        y_eval, (-2.0 - loc) / scale, np.inf, loc=loc, scale=scale
    )
    assert result.shape == (2, 3, n_obs)
    np.testing.assert_allclose(result.values, expected, rtol=1e-5)


def test_log_likelihood_truncated_response_row_bounds():
    """A `truncated` response with one lower bound per row uses each row's bound."""
    rng = np.random.default_rng(14521)
    lb = np.array([-2.0, -1.0, -0.5, -2.0, -1.0])
    df = pd.DataFrame({"y": lb + rng.uniform(0.1, 1.5, 5), "lb": lb})
    model = bmb.Model("truncated(y, lb=lb) ~ 1", df, family="gaussian")
    model.build()
    posterior, loc, scale = _gaussian_posterior(rng)

    result = _evaluate_log_likelihood(model, posterior)

    expected = stats.truncnorm.logpdf(
        df["y"].to_numpy(), (lb - loc) / scale, np.inf, loc=loc, scale=scale
    )
    np.testing.assert_allclose(result.values, expected, rtol=1e-5)


@pytest.mark.parametrize("new_data", [False, True])
def test_log_likelihood_censored_response(new_data):
    """A `censored` response uses the CDF, survival function or density per row."""
    rng = np.random.default_rng(14522)
    status = ["none", "left", "right", "none", "right", "left"]
    df = pd.DataFrame({"y": rng.uniform(-1.5, 1.5, 6), "status": status})
    model = bmb.Model("censored(y, status) ~ 1", df, family="gaussian")
    model.build()
    posterior, loc, scale = _gaussian_posterior(rng)

    data = None
    if new_data:
        data = pd.DataFrame(
            {"y": rng.uniform(-1.5, 1.5, 3), "status": ["right", "none", "left"]}
        )
    eval_df = df if data is None else data
    result = _evaluate_log_likelihood(model, posterior, data)

    y_eval = eval_df["y"].to_numpy()
    by_status = {
        "none": stats.norm.logpdf(y_eval, loc, scale),
        "left": stats.norm.logcdf(y_eval, loc, scale),
        "right": stats.norm.logsf(y_eval, loc, scale),
    }
    expected = np.stack(
        [by_status[s][..., i] for i, s in enumerate(eval_df["status"])], axis=-1
    )
    np.testing.assert_allclose(result.values, expected, rtol=1e-5)


def _log_likelihood_with_y(model, posterior, data, y):
    dt = model._compute_likelihood_params(
        xr.DataTree.from_dict({"posterior": posterior}), data, True, False
    )
    return log_likelihood(model.family, model, dt.children["posterior"], data=data, y=y)


def test_log_likelihood_constrained_explicit_y_with_predictor_only_data():
    """`constrained` bounds are scalars, so `data` needs no response column with `y`."""
    rng = np.random.default_rng(14523)
    df = pd.DataFrame({"y": rng.uniform(-1.5, 1.5, 5), "other": 0.0})
    model = bmb.Model("constrained(y, lb=-2.0) ~ 1", df, family="gaussian")
    model.build()
    posterior, loc, scale = _gaussian_posterior(rng)
    y_new = rng.uniform(-1.5, 1.5, 4)

    result = _log_likelihood_with_y(
        model, posterior, pd.DataFrame({"other": np.zeros(4)}), y_new
    )

    expected = stats.truncnorm.logpdf(
        y_new, (-2.0 - loc) / scale, np.inf, loc=loc, scale=scale
    )
    np.testing.assert_allclose(result.values, expected, rtol=1e-5)


def test_log_likelihood_truncated_explicit_y_needs_response_columns():
    """Per-row bounds come from `data`, so predictor-only `data` raises clearly."""
    rng = np.random.default_rng(14524)
    df = pd.DataFrame({"y": rng.uniform(-1.5, 1.5, 5), "lb": -2.0, "other": 0.0})
    model = bmb.Model("truncated(y, lb=lb) ~ 1", df, family="gaussian")
    model.build()
    posterior, _, _ = _gaussian_posterior(rng)

    with pytest.raises(ValueError, match="every column of the response term"):
        _log_likelihood_with_y(
            model, posterior, pd.DataFrame({"other": np.zeros(4)}), np.zeros(4)
        )


def test_log_likelihood_censored_explicit_y_is_the_censoring_point():
    """With `y` passed, a censored row is censored at its value of `y`."""
    rng = np.random.default_rng(14525)
    status = ["none", "left", "right", "left"]
    df = pd.DataFrame({"y": rng.uniform(-1.5, 1.5, 4), "status": status})
    model = bmb.Model("censored(y, status) ~ 1", df, family="gaussian")
    model.build()
    posterior, loc, scale = _gaussian_posterior(rng)
    y_new = df["y"].to_numpy() + 0.25

    result = _log_likelihood_with_y(model, posterior, df, y_new)

    by_status = {
        "none": stats.norm.logpdf(y_new, loc, scale),
        "left": stats.norm.logcdf(y_new, loc, scale),
        "right": stats.norm.logsf(y_new, loc, scale),
    }
    expected = np.stack([by_status[s][..., i] for i, s in enumerate(status)], axis=-1)
    np.testing.assert_allclose(result.values, expected, rtol=1e-5)


def test_log_likelihood_truncated_response_unbounded_new_rows():
    """New rows that are all unbounded get the untruncated Normal density."""
    rng = np.random.default_rng(14526)
    lb = np.array([-2.0, -np.inf, -1.0, -np.inf, -2.0])
    df = pd.DataFrame({"y": rng.uniform(-0.5, 1.5, 5), "lb": lb})
    model = bmb.Model("truncated(y, lb=lb) ~ 1", df, family="gaussian")
    model.build()
    posterior, loc, scale = _gaussian_posterior(rng)
    data = pd.DataFrame({"y": rng.uniform(-1.5, 1.5, 3), "lb": -np.inf})

    result = _evaluate_log_likelihood(model, posterior, data)

    expected = stats.norm.logpdf(data["y"].to_numpy(), loc, scale)
    np.testing.assert_allclose(result.values, expected, rtol=1e-5)


def test_log_likelihood_passes_jax_mode_to_pymc_compile(monkeypatch):
    """Post-sampling likelihoods honor the requested JAX compiler."""
    compile_calls = []

    def fake_compile(inputs, outputs, **kwargs):
        """Capture the compiler mode passed to PyMC."""
        compile_calls.append((inputs, outputs, kwargs))
        return lambda **values: np.full(2, -1.0)

    family = SimpleNamespace(
        likelihood=SimpleNamespace(dist=pm.Normal, name="Normal"),
        _make_dist_kwargs_and_coords=lambda model, posterior, **kwargs: (
            {
                "mu": np.zeros((1, 1, 2)),
                "sigma": np.ones((1, 1, 2)),
            },
            {
                "chain": pd.Index([0]),
                "draw": pd.Index([0]),
                "__obs__": pd.Index([0, 1]),
            },
        ),
    )
    model = SimpleNamespace(
        response_component=SimpleNamespace(
            term=SimpleNamespace(
                data=np.array([0.0, 1.0]),
                is_constrained=False,
                is_truncated=False,
                is_censored=False,
            )
        ),
        family=family,
    )
    monkeypatch.setattr("hssm.utils.pm.compile", fake_compile)

    result = log_likelihood(
        family,
        model,
        posterior=xr.DataArray(),
        compile_mode="JAX",
    )

    assert result.shape == (1, 1, 2)
    assert len(compile_calls) == 1
    assert compile_calls[0][2]["mode"] == "JAX"


@pytest.mark.parametrize(
    ["n_samples", "n_draws", "expected"],
    [
        (0, 100, "error"),
        (1, 100, 1),
        (100, 100, 100),
        (101, 100, None),
        (1.0, 100, 100),
        (0.0, 100, "error"),
        (0.5, 100, 50),
        (2.0, 100, "error"),
        (None, 100, None),
        (0.5, 0, "error"),
    ],
)
def test__generate_random_indice(caplog, n_samples, n_draws, expected):
    """Test _generate_random_indices."""
    if expected == "error":
        with pytest.raises(ValueError):
            indices = _generate_random_indices(n_samples, n_draws)
    else:
        indices = _generate_random_indices(n_samples, n_draws)
        if expected is None:
            assert indices is None
        else:
            assert indices.dtype == np.int64
            assert len(indices) == expected
            assert np.all(indices < n_draws)
            assert np.all(indices >= 0)
            if n_samples > n_draws:
                assert "n_samples > n_draws" in caplog.text


def test__generate_random_indices_rng():
    """A seeded rng makes the selection reproducible; None stays legacy."""
    a = _generate_random_indices(10, 100, rng=np.random.default_rng(42))
    b = _generate_random_indices(10, 100, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    # the legacy global-RNG path still works and honors the same contract
    c = _generate_random_indices(10, 100)
    assert len(c) == 10 and np.all(c < 100)


def assertions(caplog, obj, n_samples, expected):
    """Assert random-sampling behavior for one xarray object."""
    if expected == "error":
        with pytest.raises(ValueError):
            sampled_obj = _random_sample(obj, n_samples=n_samples)
    else:
        sampled_obj = _random_sample(obj, n_samples=n_samples)
        assert sampled_obj.draw.size == expected
        assert sampled_obj.chain.size == 2
        assert type(sampled_obj) is type(obj)
        if n_samples and n_samples > obj.draw.size:
            assert "n_samples > n_draws" in caplog.text


@pytest.mark.parametrize(
    ["n_samples", "expected"],
    [
        (0, "error"),
        (1, 1),
        (100, 100),
        (501, 500),
        (0.0, "error"),
        (0.5, 250),
        (1.0, 500),
        (2.0, "error"),
        (None, 500),
    ],
)
def test__random_sample(
    caplog,
    cav_dt,
    n_samples,
    expected,
):
    """Subsample posterior groups for supported sample specifications."""
    posterior = cav_dt["posterior"]
    posterior_predictive = cav_dt["posterior_predictive"]

    assertions(caplog, posterior, n_samples, expected)
    assertions(caplog, posterior_predictive, n_samples, expected)


def test_check_data_for_rl():
    """Validate and sort reinforcement-learning trial data."""
    # Valid DataFrame
    data_valid = pd.DataFrame(
        {
            "participant_id": [2, 2, 1, 1],
            "trial_id": [1, 2, 1, 2],
            "value": [10, 20, 30, 40],
        }
    )
    sorted_data, n_participants, n_trials = hssm.utils.check_data_for_rl(data_valid)
    # Check sorting
    assert list(sorted_data["participant_id"]) == [1, 1, 2, 2]
    assert list(sorted_data["trial_id"]) == [1, 2, 1, 2]
    assert n_participants == 2
    assert n_trials == 2

    # Missing participant_id column
    data_missing_participant = pd.DataFrame(
        {
            "trial_id": [1, 2, 1, 2],
            "value": [10, 20, 30, 40],
        }
    )
    with pytest.raises(ValueError, match="participant_id"):
        hssm.utils.check_data_for_rl(data_missing_participant)

    # Missing trial_id column
    data_missing_trial = pd.DataFrame(
        {
            "participant_id": [1, 1, 2, 2],
            "value": [10, 20, 30, 40],
        }
    )
    with pytest.raises(ValueError, match="trial_id"):
        hssm.utils.check_data_for_rl(data_missing_trial)

    # Unequal number of trials per participant
    data_unequal_trials = pd.DataFrame(
        {
            "participant_id": [1, 1, 2],
            "trial_id": [1, 2, 1],
            "value": [10, 20, 30],
        }
    )
    with pytest.raises(ValueError, match="same number of trials"):
        hssm.utils.check_data_for_rl(data_unequal_trials)

    # Custom column names
    data_custom_cols = pd.DataFrame(
        {
            "subj": [1, 1, 2, 2],
            "trial": [1, 2, 1, 2],
            "value": [10, 20, 30, 40],
        }
    )
    sorted_data, n_participants, n_trials = hssm.utils.check_data_for_rl(
        data_custom_cols, participant_id_col="subj", trial_id_col="trial"
    )
    assert list(sorted_data["subj"]) == [1, 1, 2, 2]
    assert list(sorted_data["trial"]) == [1, 2, 1, 2]
    assert n_participants == 2
    assert n_trials == 2


def test_predictive_idata_to_dataframe(data_ddm):
    """Convert prior-predictive draws to a tidy DataFrame."""
    model = hssm.HSSM(data=data_ddm)
    sample_do = model.sample_do(params={"v": 1.0}, draws=10)
    assert isinstance(sample_do, xr.DataTree)
    df = hssm.utils.predictive_dt_to_dataframe(
        sample_do, predictive_group="prior_predictive"
    )
    assert df is not None
    assert df.shape == (10 * data_ddm.shape[0], 5)
    assert set(df.columns) == {"chain", "draw", "rt", "response", "__obs__"}

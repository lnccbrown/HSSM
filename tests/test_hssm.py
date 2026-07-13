"""Tests for the HSSM public model interface."""

from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from pymc.variational import Approximation

import hssm
from hssm import HSSM
from hssm.likelihoods import DDM, logp_ddm

hssm.set_floatX("float32", update_jax=True)

param_v = {
    "name": "v",
    "prior": {
        "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
        "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
        "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
    },
    "formula": "v ~ 1 + x + y",
}

param_a = param_v | dict(name="a", formula="a ~ 1 + x + y")


@pytest.mark.slow
@pytest.mark.parametrize(
    "include, expected_exception",
    [
        (
            [param_v],
            None,
        ),
        (
            [
                param_v,
                param_a,
            ],
            None,
        ),
        (
            [{"name": "invalid_param", "prior": "invalid_param"}],
            ValueError,
        ),
        (
            [
                {
                    "name": "v",
                    "prior": {
                        "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0}
                    },
                    "formula": "v ~ 1",
                    "invalid_key": "identity",
                }
            ],
            TypeError,
        ),
        (
            [
                {
                    "name": "v",
                    "prior": {
                        "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0}
                    },
                    "formula": "invalid_formula",
                }
            ],
            IndexError,
        ),
    ],
)
def test_transform_params_general(data_ddm_reg, include, expected_exception):
    """Validate include specifications and reject malformed entries."""
    if expected_exception is not None:
        with pytest.raises(expected_exception):
            HSSM(data=data_ddm_reg, include=include)
    else:
        model = HSSM(data=data_ddm_reg, include=include)
        # Check model properties using a loop
        param_names = ["v", "a", "z", "t", "p_outlier"]
        model_param_names = list(model.params.keys())
        assert model_param_names == param_names
        assert len(model.params) == 5


@pytest.mark.slow
def test_custom_model(data_ddm):
    """Validate custom-model configuration requirements."""
    with pytest.raises(
        ValueError, match="When using a custom model, please provide a `loglik_kind.`"
    ):
        HSSM(data=data_ddm, model="custom")

    with pytest.raises(ValueError, match=r"^Please provide `list_params`"):
        HSSM(data=data_ddm, model="custom", loglik_kind="analytical")

    with pytest.raises(ValueError, match=r"^Please provide `list_params`"):
        HSSM(data=data_ddm, model="custom", loglik=DDM, loglik_kind="analytical")

    with pytest.raises(
        ValueError,
        match=r"^Please provide `list_params`",
    ):
        HSSM(
            data=data_ddm,
            model="custom",
            loglik=DDM,
            loglik_kind="analytical",
            model_config={},
        )

    model = HSSM(
        data=data_ddm,
        model="custom",
        model_config={
            "list_params": ["v", "a", "z", "t"],
            "choices": [-1, 1],
            "bounds": {
                "v": (-3.0, 3.0),
                "a": (0.3, 2.5),
                "z": (0.1, 0.9),
                "t": (0.0, 2.0),
            },
        },
        loglik=logp_ddm,
        loglik_kind="analytical",
    )

    assert model.model_config.model_name == "custom"
    assert model.model_config.loglik_kind == "analytical"
    assert model.model_config.list_params == ["v", "a", "z", "t", "p_outlier"]


@pytest.mark.slow
def test_model_definition_outside_include(data_ddm):
    """Accept parameter definitions outside include and reject duplicates."""
    model_with_one_param_fixed = HSSM(data_ddm, a=0.5)

    assert "a" in model_with_one_param_fixed.params
    assert model_with_one_param_fixed.params["a"].prior == 0.5

    model_with_one_param = HSSM(
        data_ddm, a={"prior": {"name": "Normal", "mu": 0.5, "sigma": 0.1}}
    )

    assert "a" in model_with_one_param.params
    assert model_with_one_param.params["a"].prior.name == "Normal"

    with pytest.raises(
        ValueError, match="Parameter `a` specified in both `include` and `kwargs`."
    ):
        HSSM(data_ddm, include=[{"name": "a", "prior": 0.5}], a=0.5)


@pytest.mark.slow
def test_sample_prior_predictive(data_ddm_reg):
    """Generate prior-predictive DataTrees across regression structures."""
    data_ddm_reg = data_ddm_reg.iloc[:10, :]

    model_no_regression = HSSM(data=data_ddm_reg)
    rng = np.random.default_rng()

    prior_predictive_1 = model_no_regression.sample_prior_predictive(draws=10)
    prior_predictive_2 = model_no_regression.sample_prior_predictive(
        draws=10, random_seed=rng
    )
    for prior_predictive in (prior_predictive_1, prior_predictive_2):
        assert isinstance(prior_predictive, xr.DataTree)
        assert {"prior", "prior_predictive", "observed_data"} <= set(
            prior_predictive.children
        )
        assert prior_predictive["prior_predictive"].sizes["draw"] == 10

    model_regression = HSSM(
        data=data_ddm_reg, include=[dict(name="v", formula="v ~ 1 + x")]
    )
    model_regression.sample_prior_predictive(draws=10)

    model_regression_a = HSSM(
        data=data_ddm_reg, include=[dict(name="a", formula="a ~ 1 + x")]
    )
    model_regression_a.sample_prior_predictive(draws=10)

    model_regression_multi = HSSM(
        data=data_ddm_reg,
        include=[
            dict(name="v", formula="v ~ 1 + x"),
            dict(name="a", formula="a ~ 1 + y"),
        ],
    )
    model_regression_multi.sample_prior_predictive(draws=10)

    data_ddm_reg.loc[:, "subject_id"] = np.arange(10)

    model_regression_random_effect = HSSM(
        data=data_ddm_reg,
        include=[
            dict(name="v", formula="v ~ (1|subject_id) + x"),
            dict(name="a", formula="a ~ (1|subject_id) + y"),
        ],
    )
    model_regression_random_effect.sample_prior_predictive(draws=10)


@pytest.mark.slow
def test_override_default_link(caplog, data_ddm_reg):
    """Honor custom links while warning about unusual bounds."""
    param_v = {
        "name": "v",
        "prior": {
            "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
            "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
        },
        "formula": "v ~ 1 + x + y",
    }
    param_v = param_v | dict(bounds=(-np.inf, np.inf))
    param_a = param_v | dict(name="a", formula="a ~ 1 + x + y", bounds=(0, np.inf))
    param_z = param_v | dict(name="z", formula="z ~ 1 + x + y", bounds=(0, 1))
    param_t = param_v | dict(name="t", formula="t ~ 1 + x + y", bounds=(0.1, np.inf))

    model = HSSM(
        data=data_ddm_reg,
        include=[param_v, param_a, param_z, param_t],
        link_settings="log_logit",
    )

    assert model.params["v"].link == "identity"
    assert model.params["a"].link == "log"
    assert model.params["z"].link.name == "gen_logit"
    assert model.params["t"].link == "identity"

    assert "t" in caplog.records[0].message
    assert "strange" in caplog.records[0].message


@pytest.mark.slow
def test_resampling(data_ddm):
    """Replace attached traces when a model is sampled again."""
    model = HSSM(data=data_ddm)
    sample_1 = model.sample(draws=10, chains=1, tune=0, progressbar=False)
    assert sample_1 is model.traces

    sample_2 = model.sample(draws=10, chains=1, tune=0, progressbar=False)
    assert sample_2 is model.traces

    assert sample_1 is not sample_2


@pytest.mark.slow
def test_add_likelihood_parameters_to_data(data_ddm):
    """Test if the likelihood parameters are added to the DataTree object."""
    model = HSSM(data=data_ddm)
    sample_1 = model.sample(draws=10, chains=1, tune=10, progressbar=False)
    sample_1_copy = deepcopy(sample_1)
    model.add_likelihood_parameters_to_datatree(inplace=True)

    # Get distributional components (make sure to take the right aliases)
    distributional_component_names = [
        key_ if key_ not in model._aliases else model._aliases[key_]
        for key_ in model.model.distributional_components.keys()
    ]

    # Check that after computing the likelihood parameters
    # all respective parameters appear in the DataTree object
    assert np.all(
        [
            component_ in model.traces.posterior.data_vars
            for component_ in distributional_component_names
        ]
    )

    # Check that before computing the likelihood parameters
    # at least one parameter is missing (in the simplest case
    # this is the {parent}_mean parameter if nothing received a regression)

    assert not np.all(
        [
            component_ in sample_1_copy.posterior.data_vars
            for component_ in distributional_component_names
        ]
    )


def test_log_likelihood_uses_attached_traces_and_returns_copy(data_ddm, monkeypatch):
    """Attached traces are the default input without being mutated."""
    model = HSSM(data=data_ddm)
    traces = xr.DataTree.from_dict(
        {"posterior": xr.Dataset({"v": (("chain", "draw"), np.array([[0.5]]))})}
    )
    model._inference_obj = traces
    calls = []

    def fake_compute_log_likelihood(bambi_model, dt, data, inplace):
        calls.append((bambi_model, dt, data, inplace))
        result = dt.copy(deep=True)
        result["log_likelihood"] = xr.Dataset(
            {"rt,response": (("chain", "draw"), np.array([[-1.0]]))}
        )
        return result

    monkeypatch.setattr(
        "hssm.base._compute_log_likelihood", fake_compute_log_likelihood
    )

    result = model.log_likelihood(
        inplace=False,
        keep_likelihood_params=True,
    )

    assert len(calls) == 1
    assert calls[0][0] is model.model
    assert calls[0][1] is traces
    assert calls[0][2] is None
    assert calls[0][3] is False
    assert result is not traces
    assert isinstance(result, xr.DataTree)
    assert "log_likelihood" in result
    assert "log_likelihood" not in traces


def test_add_likelihood_parameters_requires_traces(data_ddm):
    """Likelihood parameters need supplied or previously attached traces."""
    model = HSSM(data=data_ddm)

    with pytest.raises(
        ValueError,
        match="No datatree provided and model not yet sampled!",
    ):
        model.add_likelihood_parameters_to_datatree()


def test_add_likelihood_parameters_accepts_explicit_datatree(data_ddm, monkeypatch):
    """An explicit DataTree is copied before likelihood parameters are added."""
    model = HSSM(data=data_ddm)
    traces = xr.DataTree.from_dict(
        {"posterior": xr.Dataset({"v": (("chain", "draw"), np.array([[0.5]]))})}
    )
    received = []

    def fake_compute_likelihood_params(dt):
        received.append(dt)
        dt["posterior"] = dt["posterior"].ds.assign(
            v_mean=(("chain", "draw"), np.array([[0.5]]))
        )
        return dt

    monkeypatch.setattr(
        model.model,
        "_compute_likelihood_params",
        fake_compute_likelihood_params,
    )

    result = model.add_likelihood_parameters_to_datatree(dt=traces, inplace=False)

    assert len(received) == 1
    assert received[0] is not traces
    assert result is received[0]
    assert isinstance(result, xr.DataTree)
    assert "v_mean" in result["posterior"].data_vars
    assert "v_mean" not in traces["posterior"].data_vars


# Setting any parameter to a fixed value should work:
@pytest.mark.slow
def test_model_creation_constant_parameter(data_ddm):
    """Allow any non-parent parameter to be fixed."""
    for param_name in ["v", "a", "z", "t"]:
        model = HSSM(data=data_ddm, **{param_name: 1.0})
        assert model._parent != param_name
        assert model.params[param_name].prior == 1.0


# Setting any single parameter to a regression should respect the default bounds:
@pytest.mark.slow
@pytest.mark.parametrize(
    "param_name, dist_name",
    [("v", "Normal"), ("a", "Gamma"), ("z", "Beta"), ("t", "Gamma")],
)
def test_model_creation_single_regression(data_ddm_reg, param_name, dist_name):
    """Use bounded default priors for one regression parameter."""
    model = HSSM(
        data=data_ddm_reg,
        include=[{"name": param_name, "formula": f"{param_name} ~ 1 + x"}],
    )
    assert model.params[param_name].prior["Intercept"].name == dist_name
    assert model.params[param_name].prior["x"].name == "Normal"


# Setting all parameters to fixed values should throw an error:
def test_model_creation_all_parameters_constant(data_ddm):
    """Reject models without any free parameters."""
    with pytest.raises(ValueError):
        HSSM(data=data_ddm, v=1.0, a=1.0, z=1.0, t=1.0)


# Prior settings
@pytest.mark.slow
def test_prior_settings_basic(cavanagh_test):
    """Apply requested prior-setting modes."""
    model_1 = HSSM(
        data=cavanagh_test,
        global_formula="y ~ 1 + (1|participant_id)",
        prior_settings=None,
    )

    assert model_1.params["v"].prior is None, (
        "Default prior doesn't yield Nonetype for 'v'!"
    )

    model_2 = HSSM(
        data=cavanagh_test,
        global_formula="y ~ 1 + (1|participant_id)",
        prior_settings="safe",
    )

    assert isinstance(model_2.params[model_2._parent].prior, dict), (
        "Prior assigned to parent is not a dict!"
    )


@pytest.mark.slow
def test_compile_logp(cavanagh_test):
    """Compile log probability at the model's initial point."""
    model_1 = HSSM(
        data=cavanagh_test,
        global_formula="y ~ 1 + (1|participant_id)",
        prior_settings=None,
    )

    out = model_1.compile_logp(model_1.initial_point(transformed=False))
    assert out is not None


@pytest.mark.slow
class TestFixedVectorParams:
    """Tests for passing np.ndarray as a fixed vector parameter."""

    def test_fixed_vector_non_parent(self, data_ddm):
        """A fixed vector for v (non-parent) should build successfully."""
        n_obs = len(data_ddm)
        v_fixed = np.random.uniform(0.3, 0.7, size=n_obs)
        model = HSSM(
            data=data_ddm,
            model="ddm",
            include=[{"name": "v", "prior": v_fixed}],
        )

        # v should not be parent; another param takes that role
        assert model.params.parent != "v"
        # v is fixed (scalar constant to Bambi, vector substituted in logp)
        assert model.params["v"].is_fixed
        assert model.params["v"].is_trialwise

    def test_fixed_vector_length_mismatch(self, data_ddm):
        """Vector length != n_obs should raise ValueError."""
        wrong_length = np.random.uniform(0.3, 0.7, size=len(data_ddm) + 5)
        with pytest.raises(ValueError, match="has length"):
            HSSM(
                data=data_ddm,
                model="ddm",
                include=[{"name": "v", "prior": wrong_length}],
            )

    def test_fixed_vector_bounds_validation(self, data_ddm):
        """Vector values outside bounds should raise ValueError."""
        n_obs = len(data_ddm)
        out_of_bounds = np.full(n_obs, 2.0)  # z bounds are (0, 1)
        with pytest.raises(ValueError):
            HSSM(
                data=data_ddm,
                model="ddm",
                include=[{"name": "z", "prior": out_of_bounds}],
            )

    def test_fixed_vector_sampling(self, data_ddm):
        """Sampling with a fixed vector should succeed and exclude it from posterior."""
        n_obs = len(data_ddm)
        v_fixed = np.random.uniform(0.3, 0.7, size=n_obs)
        model = HSSM(
            data=data_ddm,
            model="ddm",
            include=[{"name": "v", "prior": v_fixed}],
        )
        idata = model.sample(draws=10, chains=1, tune=10, progressbar=False)

        # v must not appear in the posterior (it's a constant, not sampled)
        assert "v" not in idata.posterior.data_vars
        # The other params must be present
        for param in ["a", "z", "t"]:
            assert param in idata.posterior.data_vars

    def test_fixed_vector_multiple_params(self, data_ddm):
        """Fixing multiple parameters to vectors should work."""
        n_obs = len(data_ddm)
        v_fixed = np.random.uniform(0.3, 0.7, size=n_obs)
        t_fixed = np.random.uniform(0.1, 0.4, size=n_obs)
        model = HSSM(
            data=data_ddm,
            model="ddm",
            include=[
                {"name": "v", "prior": v_fixed},
                {"name": "t", "prior": t_fixed},
            ],
        )
        idata = model.sample(draws=10, chains=1, tune=10, progressbar=False)

        # Both fixed params excluded from posterior
        assert "v" not in idata.posterior.data_vars
        assert "t" not in idata.posterior.data_vars
        # Remaining params present
        assert "a" in idata.posterior.data_vars
        assert "z" in idata.posterior.data_vars


@pytest.mark.slow
def test_sample_do(data_ddm):
    """Return intervention samples and the intervened PyMC model."""
    model = HSSM(data=data_ddm)
    sample_do, do_model = model.sample_do(
        params={"v": 1.0}, draws=10, return_model=True
    )
    assert isinstance(do_model, pm.Model)
    assert sample_do is not None
    assert "v_mean" in sample_do.prior.data_vars
    assert set(sample_do.prior_predictive.dims) == {
        "chain",
        "draw",
        "__obs__",
        "rt,response_dim",
    }
    assert set(sample_do.prior_predictive.coords) == {
        "chain",
        "draw",
        "__obs__",
        "rt,response_dim",
    }
    assert np.unique(sample_do.prior["v_mean"].values) == [1.0]


@pytest.mark.parametrize(
    ("method_name", "expected_message"),
    [
        (
            "plot_trace",
            "HSSM.plot_trace has been deprecated. Please use az.plot_trace_dist from "
            "ArviZ directly. For example: "
            "`az.plot_trace_dist(model.traces, var_names=[...])`.",
        ),
        (
            "summary",
            "HSSM.summary has been deprecated. Please use az.summary from ArviZ "
            "directly. For example: `az.summary(model.traces, var_names=[...])`.",
        ),
    ],
)
def test_deprecated_inference_helpers_raise_documented_error(
    data_ddm, method_name, expected_message
):
    """Removed inference helpers direct users to their ArviZ replacements."""
    model = HSSM(data=data_ddm)

    with pytest.raises(NotImplementedError) as error:
        getattr(model, method_name)()

    assert str(error.value) == expected_message


@pytest.mark.parametrize(
    "config_backend, backend_arg, expected_backend",
    [
        (None, None, "numba"),
        ("jax", None, "jax"),
        ("pytensor", None, "c"),
        ("jax", "numba", "numba"),
        ("pytensor", "numba", "numba"),
        ("pytensor", "jax", "jax"),
    ],
)
def test_vi_passes_backend_to_pm_fit(
    data_ddm, monkeypatch, config_backend, backend_arg, expected_backend
):
    """The resolved backend is forwarded to `pm.fit`."""
    model = HSSM(data=data_ddm, model="ddm")
    model.model_config.backend = config_backend

    captured = {}

    def fake_fit(*args, **kwargs):
        captured["backend"] = kwargs.get("backend")

    monkeypatch.setattr("hssm.base.pm.fit", fake_fit)
    # Skip post-processing since fake_fit returns no approximation object.
    monkeypatch.setattr(model, "_clean_posterior_group", lambda dt=None: None)

    model.vi(niter=1, draws=1, backend=backend_arg)

    assert captured["backend"] == expected_backend


def test_vi_idata_rejects_attached_approximation(data_ddm):
    """VI approximations must be sampled before they can be exposed as DataTrees."""
    model = HSSM(data=data_ddm)
    model._inference_obj_vi = Mock(spec=Approximation)

    with pytest.raises(ValueError, match="attached variational inference object"):
        _ = model.vi_idata


def test_drop_parent_str_requires_datatree(data_ddm):
    """Posterior cleanup rejects a missing trace object."""
    model = HSSM(data=data_ddm)

    with pytest.raises(
        ValueError,
        match=r"Please provide a DataTree \(traces\) object\.",
    ):
        model._drop_parent_str_from_datatree(None)


def test_drop_parent_str_renames_response_mean(data_ddm):
    """Posterior cleanup restores the model's response-parameter name."""
    model = HSSM(data=data_ddm)
    traces = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {
                    "rt,response_mean": (
                        ("chain", "draw"),
                        np.array([[0.5]]),
                    )
                }
            )
        }
    )

    result = model._drop_parent_str_from_datatree(traces)

    assert result is traces
    assert model._parent in result["posterior"].data_vars
    assert "rt,response_mean" not in result["posterior"].data_vars
    np.testing.assert_array_equal(
        result["posterior"][model._parent].values,
        np.array([[0.5]]),
    )


def test_is_choice_only_and_deadline(data_ddm):
    """Expose choice-only and deadline response metadata."""
    config_choice_only = {"response": ["response"]}

    model = HSSM(data=data_ddm, model="ddm", model_config=config_choice_only)
    assert model.model_config.is_choice_only
    assert model.response_c == "response"
    assert model.response_str == "response"

    data_deadline = data_ddm.copy()
    data_deadline["deadline"] = 0

    model_with_deadline = HSSM(
        data=data_deadline, model="ddm", model_config=config_choice_only, deadline=True
    )

    assert model_with_deadline.is_choice_only
    assert model_with_deadline.deadline
    assert model_with_deadline.response is not None
    assert len(model_with_deadline.response) == 2
    assert model_with_deadline.response_c == "c(response, deadline)"
    assert model_with_deadline.response_str == "response,deadline"

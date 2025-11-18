import bambi as bmb
import numpy as np
import pytest

import hssm
from hssm import HSSM
from hssm.likelihoods import DDM, logp_ddm
from copy import deepcopy

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
    "include, should_raise_exception",
    [
        (
            [param_v],
            False,
        ),
        (
            [
                param_v,
                param_a,
            ],
            False,
        ),
        (
            [{"name": "invalid_param", "prior": "invalid_param"}],
            True,
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
            True,
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
            True,
        ),
    ],
)
def test_transform_params_general(data_ddm_reg, include, should_raise_exception):
    if should_raise_exception:
        with pytest.raises(Exception):
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
    with pytest.raises(
        ValueError, match="When using a custom model, please provide a `loglik_kind.`"
    ):
        model = HSSM(data=data_ddm, model="custom")

    with pytest.raises(
        ValueError, match="Please provide `list_params` via `model_config`."
    ):
        model = HSSM(data=data_ddm, model="custom", loglik_kind="analytical")

    with pytest.raises(
        ValueError, match="Please provide `list_params` via `model_config`."
    ):
        model = HSSM(
            data=data_ddm, model="custom", loglik=DDM, loglik_kind="analytical"
        )

    with pytest.raises(
        ValueError,
        match="Please provide `list_params` via `model_config`.",
    ):
        model = HSSM(
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

    assert model.model_name == "custom"
    assert model.loglik_kind == "analytical"
    assert model.list_params == ["v", "a", "z", "t", "p_outlier"]


@pytest.mark.slow
def test_model_definition_outside_include(data_ddm):
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
    data_ddm_reg = data_ddm_reg.iloc[:10, :]

    model_no_regression = HSSM(data=data_ddm_reg)
    rng = np.random.default_rng()

    prior_predictive_1 = model_no_regression.sample_prior_predictive(draws=10)
    prior_predictive_2 = model_no_regression.sample_prior_predictive(
        draws=10, random_seed=rng
    )

    model_regression = HSSM(
        data=data_ddm_reg, include=[dict(name="v", formula="v ~ 1 + x")]
    )
    prior_predictive_3 = model_regression.sample_prior_predictive(draws=10)

    model_regression_a = HSSM(
        data=data_ddm_reg, include=[dict(name="a", formula="a ~ 1 + x")]
    )
    prior_predictive_4 = model_regression_a.sample_prior_predictive(draws=10)

    model_regression_multi = HSSM(
        data=data_ddm_reg,
        include=[
            dict(name="v", formula="v ~ 1 + x"),
            dict(name="a", formula="a ~ 1 + y"),
        ],
    )
    prior_predictive_5 = model_regression_multi.sample_prior_predictive(draws=10)

    data_ddm_reg.loc[:, "subject_id"] = np.arange(10)

    model_regression_random_effect = HSSM(
        data=data_ddm_reg,
        include=[
            dict(name="v", formula="v ~ (1|subject_id) + x"),
            dict(name="a", formula="a ~ (1|subject_id) + y"),
        ],
    )
    prior_predictive_6 = model_regression_random_effect.sample_prior_predictive(
        draws=10
    )


@pytest.mark.slow
def test_override_default_link(caplog, data_ddm_reg):
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
    model = HSSM(data=data_ddm)
    sample_1 = model.sample(draws=10, chains=1, tune=0)
    assert sample_1 is model.traces

    sample_2 = model.sample(draws=10, chains=1, tune=0)
    assert sample_2 is model.traces

    assert sample_1 is not sample_2


@pytest.mark.slow
def test_add_likelihood_parameters_to_data(data_ddm):
    """Test if the likelihood parameters are added to the InferenceData object."""
    model = HSSM(data=data_ddm)
    sample_1 = model.sample(draws=10, chains=1, tune=10)
    sample_1_copy = deepcopy(sample_1)
    model.add_likelihood_parameters_to_idata(inplace=True)

    # Get distributional components (make sure to take the right aliases)
    distributional_component_names = [
        key_ if key_ not in model._aliases else model._aliases[key_]
        for key_ in model.model.distributional_components.keys()
    ]

    # Check that after computing the likelihood parameters
    # all respective parameters appear in the InferenceData object
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


# Setting any parameter to a fixed value should work:
@pytest.mark.slow
def test_model_creation_constant_parameter(data_ddm):
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
    model = HSSM(
        data=data_ddm_reg,
        include=[{"name": param_name, "formula": f"{param_name} ~ 1 + x"}],
    )
    assert model.params[param_name].prior["Intercept"].name == dist_name
    assert model.params[param_name].prior["x"].name == "Normal"


# Setting all parameters to fixed values should throw an error:
def test_model_creation_all_parameters_constant(data_ddm):
    with pytest.raises(ValueError):
        HSSM(data=data_ddm, v=1.0, a=1.0, z=1.0, t=1.0)


# Prior settings
@pytest.mark.slow
def test_prior_settings_basic(cavanagh_test):
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
    model_1 = HSSM(
        data=cavanagh_test,
        global_formula="y ~ 1 + (1|participant_id)",
        prior_settings=None,
    )

    out = model_1.compile_logp(model_1.initial_point(transformed=False))
    assert out is not None


@pytest.mark.slow
def test_sample_do(data_ddm):
    model = HSSM(data=data_ddm)
    sample_do = model.sample_do(params={"v": 1.0}, draws=10)
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

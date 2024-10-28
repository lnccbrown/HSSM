from unittest.mock import Mock

import bambi as bmb
import pytest

from hssm import HSSM, Link, Prior
from hssm.config import Config
from hssm.defaults import SupportedModels, default_model_config
from hssm.param.param import Param
from hssm.param.params import (
    Params,
    collect_user_params,
    make_params,
    make_param_from_user_param,
    make_param_from_defaults,
)
from hssm.param.regression_param import RegressionParam
from hssm.param.simple_param import DefaultParam, SimpleParam
from hssm.param.user_param import UserParam


def create_mock_model(
    model_name: SupportedModels = "ddm",
    loglik_kind="analytical",
    global_formula=None,
    link_settings=None,
    prior_settings=None,
):
    def mock_config(model: SupportedModels, loglik_kind="analytical"):
        if model not in default_model_config:
            return lambda param: (None, None)

        defaults = default_model_config[model]["likelihoods"][loglik_kind]

        priors = defaults["default_priors"]
        bounds = defaults["bounds"]

        return lambda param: (priors.get(param), bounds.get(param))

    # The Mock objects here are used to create the HSSM model object without
    # actually having to create the object itself. This is because when the tested
    # functions are called, they are called within the constructor of the HSSM object.
    # At this point, the model object is only partially constructed.
    # It is not possible to create the HSSM object before the tested functions are
    # called.
    config = Mock(
        spec=Config,
        get_defaults=mock_config(model_name, loglik_kind),
    )

    model = Mock(
        model_name=model_name,
        spec=HSSM,
        list_params=default_model_config[model_name]["list_params"] + ["p_outlier"],
        has_lapse=True,
        loglik_kind=loglik_kind,
        global_formula=global_formula,
        link_settings=link_settings,
        prior_settings=prior_settings,
        model_config=config,
        response_c="c(rt, response)",
        additional_namespace={},
    )
    return model


include = [
    {"name": "v", "prior": {"name": "Normal", "mu": 0, "sigma": 1}},
    {"name": "a", "prior": bmb.Prior("HalfNormal", sigma=1)},
]


def kw():
    return {
        "z": UserParam(prior=bmb.Prior("Normal", mu=0, sigma=1)),
        "other_param": 1,
    }


p_outlier = 0.5


def test_collect_user_params():
    kwargs = kw()
    model = create_mock_model("ddm")
    print(model.list_params)
    params = collect_user_params(model, include, kwargs, p_outlier)

    assert isinstance(params, dict)
    assert "t" not in params
    for param_name, param in params.items():
        assert param_name in ["v", "a", "z", "t", "p_outlier"]
        assert isinstance(param, UserParam)

    assert params["v"].prior == {"name": "Normal", "mu": 0, "sigma": 1}
    assert params["a"].prior == bmb.Prior("HalfNormal", sigma=1)
    assert params["z"].prior == bmb.Prior("Normal", mu=0, sigma=1)
    assert params["p_outlier"] == UserParam("p_outlier", prior=0.5)

    # kwargs is modified in place
    assert "other_param" in kwargs
    assert "z" not in kwargs


def test_make_param_from_user_param():
    # 1. Test the case with no global formula or link settings
    # We test the following three cases:
    # 1.1. UserParam that is definitely a regression parameter
    # 1.2. UserParam that is definitely a simple parameter
    # 1.3. UserParam that is ambiguous
    model = create_mock_model("ddm")

    # 1.1. UserParam that is definitely a regression parameter
    reg_user_param = UserParam(
        name="a",
        formula="a ~ x",
    )

    assert reg_user_param.is_regression

    param = make_param_from_user_param(model, "a", reg_user_param)
    assert isinstance(param, RegressionParam)
    assert param.name == "a"
    assert param.formula == "a ~ x"
    assert param.prior is None
    assert (
        param.bounds
        == default_model_config["ddm"]["likelihoods"]["analytical"]["bounds"]["a"]
    )
    assert param.link is None
    assert param.user_param is reg_user_param

    # 1.2. UserParam that is definitely a simple parameter
    simple_user_param = UserParam(
        name="v",
        prior={"name": "Normal", "mu": 0, "sigma": 1},
    )
    assert simple_user_param.is_simple

    param = make_param_from_user_param(model, "v", simple_user_param)
    assert isinstance(param, SimpleParam)
    assert param.name == "v"
    assert param.prior == dict(name="Normal", mu=0, sigma=1)

    # 1.3. UserParam that is ambiguous
    ambiguous_user_param = UserParam(
        name="z",
        bounds=(0, 1),
    )
    assert not ambiguous_user_param.is_simple and not ambiguous_user_param.is_regression

    param = make_param_from_user_param(model, "z", ambiguous_user_param)
    assert isinstance(param, SimpleParam)
    assert param.name == "z"
    assert param.prior is not None
    assert param.prior.name == "Uniform"
    assert param.bounds == (0, 1)
    assert param.link is None

    # 2. Test the case of log-logit link settings
    # We test the following three cases:
    # 2.1. UserParam that is definitely a regression parameter
    model = create_mock_model("ddm", link_settings="log_logit")
    param = make_param_from_user_param(model, "a", reg_user_param)
    assert isinstance(param, RegressionParam)
    assert param.name == "a"
    assert param.formula == "a ~ x"
    assert param.prior is None
    assert (
        param.bounds
        == default_model_config["ddm"]["likelihoods"]["analytical"]["bounds"]["a"]
    )
    assert param.link == "log"
    assert param.user_param is reg_user_param

    # 3. Test the case of global formula
    # We test the following three cases:
    # 3.1. UserParam that is ambiguous
    model = create_mock_model("ddm", global_formula="t ~ x + z")
    param = make_param_from_user_param(model, "z", ambiguous_user_param)
    assert isinstance(param, RegressionParam)
    assert param.name == "z"
    assert param.formula == "t ~ x + z"  # By this time, the formula is not yet updated
    assert param.prior is None
    assert param.bounds == (0, 1)

    # 4. Test the case of global formula and link settings
    # We test the following three cases:
    # 4.1. UserParam that is ambiguous
    model = create_mock_model(
        "ddm", global_formula="t ~ x + z", link_settings="log_logit"
    )
    param = make_param_from_user_param(model, "z", ambiguous_user_param)
    assert isinstance(param, RegressionParam)
    assert param.name == "z"
    assert param.formula == "t ~ x + z"  # By this time, the formula is not yet updated
    assert param.prior is None
    assert param.bounds == (0, 1)
    assert isinstance(param.link, Link)


def test_make_param_from_defaults():
    # 1. Test the case with no global formula or link settings
    model = create_mock_model("ddm")

    param = make_param_from_defaults(model, "v")
    assert isinstance(param, SimpleParam)
    assert param.name == "v"
    assert isinstance(param.prior, bmb.Prior)
    assert param.prior.name == "Normal"
    assert param.bounds is not None
    assert param.link is None

    # 2. Test the case of log-logit link settings
    model = create_mock_model("ddm", link_settings="log_logit")
    param = make_param_from_defaults(model, "z")
    assert isinstance(param, SimpleParam)
    assert param.name == "z"
    assert isinstance(param.prior, bmb.Prior)
    assert param.prior.name == "Uniform"
    assert param.bounds is not None
    assert param.link is None

    # 3. Test the case of global formula
    model = create_mock_model("ddm", global_formula="t ~ x + z")
    param = make_param_from_defaults(model, "z")
    assert isinstance(param, RegressionParam)
    assert param.name == "z"
    assert param.formula == "t ~ x + z"  # By this time, the formula is not yet updated
    assert param.prior is None
    assert param.bounds is not None
    assert param.link is None

    # 4. Test the case of global formula and link settings
    model = create_mock_model(
        "ddm", global_formula="t ~ x + z", link_settings="log_logit"
    )
    param = make_param_from_defaults(model, "z")
    assert isinstance(param, RegressionParam)
    assert param.name == "z"
    assert param.formula == "t ~ x + z"  # By this time, the formula is not yet updated
    assert param.prior is None
    assert param.bounds is not None
    assert isinstance(param.link, Link)


def test_make_params():
    model = create_mock_model("ddm")
    kwargs = kw()
    user_params_dict = collect_user_params(model, include, kwargs, p_outlier)

    params_dict = make_params(model, user_params_dict)
    assert all(param_name in params_dict for param_name in model.list_params)
    assert all(isinstance(param, SimpleParam) for param in params_dict.values())

    assert params_dict["v"].name == "v"
    assert isinstance(params_dict["v"].prior, Prior)
    assert params_dict["v"].prior.name == "Normal"
    assert params_dict["v"].prior.args["mu"] == 0
    assert params_dict["v"].prior.args["sigma"] == 1
    assert not params_dict["v"].prior.is_truncated
    assert params_dict["v"].bounds is not None
    assert not isinstance(params_dict["v"], DefaultParam)

    assert params_dict["a"].name == "a"
    assert isinstance(params_dict["a"].prior, bmb.Prior)
    assert not isinstance(params_dict["a"].prior, Prior)
    assert params_dict["a"].prior.name == "HalfNormal"
    assert params_dict["a"].prior.args["sigma"] == 1
    assert params_dict["a"].bounds is not None
    assert not isinstance(params_dict["a"], DefaultParam)

    assert params_dict["z"].name == "z"
    assert isinstance(params_dict["z"].prior, bmb.Prior)
    assert not isinstance(params_dict["z"].prior, Prior)
    assert params_dict["z"].prior.name == "Normal"
    assert params_dict["z"].prior.args["mu"] == 0
    assert params_dict["z"].prior.args["sigma"] == 1
    assert params_dict["z"].bounds is not None
    assert not isinstance(params_dict["z"], DefaultParam)

    assert params_dict["t"].name == "t"
    assert isinstance(params_dict["t"].prior, bmb.Prior)
    assert not isinstance(params_dict["t"].prior, Prior)
    assert params_dict["t"].prior.name == "HalfNormal"
    assert params_dict["t"].prior.args["sigma"] == 2.0
    assert isinstance(params_dict["t"], DefaultParam)

    model = create_mock_model("ddm", global_formula="t ~ x + z")

    params_dict = make_params(model, user_params_dict)
    assert all(param_name in params_dict for param_name in model.list_params)
    assert all(isinstance(params_dict[param], SimpleParam) for param in ["v", "a", "z"])

    assert params_dict["v"].name == "v"
    assert isinstance(params_dict["v"].prior, Prior)
    assert params_dict["v"].prior.name == "Normal"
    assert params_dict["v"].prior.args["mu"] == 0
    assert params_dict["v"].prior.args["sigma"] == 1
    assert not params_dict["v"].prior.is_truncated
    assert params_dict["v"].bounds is not None
    assert not isinstance(params_dict["v"], DefaultParam)

    assert params_dict["a"].name == "a"
    assert isinstance(params_dict["a"].prior, bmb.Prior)
    assert not isinstance(params_dict["a"].prior, Prior)
    assert params_dict["a"].prior.name == "HalfNormal"
    assert params_dict["a"].prior.args["sigma"] == 1
    assert params_dict["a"].bounds is not None
    assert not isinstance(params_dict["a"], DefaultParam)

    assert params_dict["z"].name == "z"
    assert isinstance(params_dict["z"].prior, bmb.Prior)
    assert not isinstance(params_dict["z"].prior, Prior)
    assert params_dict["z"].prior.name == "Normal"
    assert params_dict["z"].prior.args["mu"] == 0
    assert params_dict["z"].prior.args["sigma"] == 1
    assert params_dict["z"].bounds is not None
    assert not isinstance(params_dict["z"], DefaultParam)

    assert params_dict["t"].name == "t"
    assert isinstance(params_dict["t"], RegressionParam)
    assert params_dict["t"].formula == "t ~ x + z"
    assert params_dict["t"].prior is None
    assert params_dict["t"].bounds is not None


def test_from_user_specs():
    model = create_mock_model("ddm")
    kwargs = kw()

    params = Params.from_user_specs(model, include, kwargs, p_outlier)
    assert all(param_name in params for param_name in model.list_params)
    assert params.parent == "v"
    assert params["v"].is_parent
    assert params.parent_param is params["v"]
    assert all(not params[param].is_parent for param in ["a", "z", "t", "p_outlier"])

    model = create_mock_model("ddm", global_formula="t ~ x + z")

    params = Params.from_user_specs(model, include, kwargs, p_outlier)
    assert all(param_name in params for param_name in model.list_params)
    print(params.parent)
    assert params["z"].is_parent
    assert params.parent_param is params["z"]
    assert all(not params[param].is_parent for param in ["v", "a", "t", "p_outlier"])


def test_parse_bambi(data_ddm_reg):
    # Simple case
    kwargs = kw()
    model = create_mock_model("ddm")
    params = Params.from_user_specs(model, include, kwargs, p_outlier)

    formula, priors, links = params.parse_bambi(model=model)

    assert isinstance(formula, bmb.Formula)
    formulas = formula.get_all_formulas()
    assert len(formulas) == 1
    assert "c(rt, response) ~ 1" in formulas

    assert priors["v"] == {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
    }
    assert priors["a"] == bmb.Prior("HalfNormal", sigma=1)
    assert priors["z"] == bmb.Prior("Normal", mu=0, sigma=1)
    assert priors["t"] == bmb.Prior("HalfNormal", sigma=2.0)
    assert priors["p_outlier"] == 0.5

    assert links == {"v": "identity"}

    # Has global formula
    kwargs = kw()
    model = create_mock_model("ddm", global_formula="t ~ x + y")
    params = Params.from_user_specs(model, include, kwargs, p_outlier)

    formula, priors, links = params.parse_bambi(model=model)

    assert isinstance(formula, bmb.Formula)
    formulas = formula.get_all_formulas()
    assert len(formulas) == 1
    assert "c(rt, response) ~ x + y" in formulas

    assert priors["v"] == bmb.Prior("Normal", mu=0, sigma=1)
    assert priors["a"] == bmb.Prior("HalfNormal", sigma=1)
    assert priors["z"] == bmb.Prior("Normal", mu=0, sigma=1)
    assert priors["p_outlier"] == 0.5
    assert "t" not in priors

    assert links == {"t": "identity"}

    # Has log-logit link settings and global formula
    kwargs = kw()
    model = create_mock_model("ddm", global_formula="t ~ x + y", prior_settings="safe")
    setattr(model, "data", data_ddm_reg)
    params = Params.from_user_specs(model, include, kwargs, p_outlier)

    formula, priors, links = params.parse_bambi(model=model)

    assert isinstance(formula, bmb.Formula)
    formulas = formula.get_all_formulas()
    assert len(formulas) == 1
    assert "c(rt, response) ~ x + y" in formulas

    assert priors["v"] == bmb.Prior("Normal", mu=0, sigma=1)
    assert priors["a"] == bmb.Prior("HalfNormal", sigma=1)
    assert priors["z"] == bmb.Prior("Normal", mu=0, sigma=1)
    assert priors["p_outlier"] == 0.5
    for key in ["Intercept", "x", "y"]:
        assert isinstance(priors["t"][key], bmb.Prior)

    assert links == {"t": "identity"}

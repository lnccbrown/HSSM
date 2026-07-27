import bambi as bmb
import numpy as np
import pytest

import hssm
from hssm import Prior
from hssm.modelconfig import get_default_model_config
from hssm.link import Link
from hssm.param import UserParam
from hssm.param.regression_param import RegressionParam, _make_priors_recursive
from hssm.prior import HSSM_SETTINGS_DISTRIBUTIONS, HDDM_SETTINGS_GROUP


v_reg = UserParam(
    name="v",
    formula="v ~ 1 + x + y",
    prior={
        "Intercept": {"name": "Uniform", "lower": 0.0, "upper": 0.5},
        "x": dict(name="Uniform", lower=0.0, upper=1.0),
        "y": bmb.Prior("Uniform", lower=0.0, upper=1.0),
        "z": 0.1,
    },
)

v_reg_1 = UserParam(
    name="v",
    formula="v ~ 1 + x + y",
    prior={
        "Intercept": {"name": "Uniform", "lower": 0.0, "upper": 0.5},
        "x": dict(name="Uniform", lower=0.0, upper=1.0),
        "y": bmb.Prior("Uniform", lower=0.0, upper=1.0),
        "z": 0.1,
    },
    bounds=(0.0, 1.0),
)


def test_from_user_param():
    v = RegressionParam.from_user_param(v_reg)

    assert v.name == "v"
    assert v.formula == "v ~ 1 + x + y"
    assert isinstance(v.prior, dict)
    assert v.prior["Intercept"]["name"] == "Uniform"
    assert v.prior["Intercept"]["lower"] == 0.0
    assert v.prior["Intercept"]["upper"] == 0.5
    x = v.prior["x"]
    assert isinstance(x, dict)
    assert x["name"] == "Uniform"
    assert x["lower"] == 0.0
    assert x["upper"] == 1.0
    assert isinstance(v.prior["y"], bmb.Prior)
    assert isinstance(v.prior["z"], float)

    assert v.bounds is None
    assert v.user_param is v_reg


def test_from_defaults(caplog):
    param = RegressionParam.from_defaults(
        name="v", formula="v ~ 1 + x + y", bounds=(0.0, 1.0)
    )

    assert param.name == "v"
    assert param.formula == "v ~ 1 + x + y"
    assert param.prior is None
    assert param.bounds == (0.0, 1.0)
    assert param.user_param is None
    assert param.link is None

    param = RegressionParam.from_defaults(
        name="v",
        formula="v ~ 1 + x + y",
        bounds=(0.0, 1.0),
        link_settings="log_logit",
    )
    assert param.name == "v"
    assert param.formula == "v ~ 1 + x + y"
    assert param.prior is None
    assert param.bounds == (0.0, 1.0)
    assert param.user_param is None
    assert isinstance(param.link, Link)
    assert param.link.name == "gen_logit"
    assert param.link.bounds == (0.0, 1.0)

    param = RegressionParam.from_defaults(
        name="v",
        formula="v ~ 1 + x + y",
        bounds=(0.0, np.inf),
        link_settings="log_logit",
    )
    assert isinstance(param.link, str)
    assert param.link == "log"

    param = RegressionParam.from_defaults(
        name="v",
        formula="v ~ 1 + x + y",
        bounds=(-np.inf, np.inf),
        link_settings="log_logit",
    )
    assert isinstance(param.link, str)
    assert param.link == "identity"

    param = RegressionParam.from_defaults(
        name="v",
        formula="v ~ 1 + x + y",
        bounds=(-np.inf, 2.0),
        link_settings="log_logit",
    )

    assert caplog.records[-1].levelname == "WARNING"
    assert caplog.records[-1].message == (
        "The bounds for parameter v (-inf, 2.000000) seem "
        + "strange. Nothing is done to the link function. "
        + "Please check if they are correct."
    )


def test_fill_defaults():
    v = RegressionParam.from_user_param(v_reg)
    v.fill_defaults(bounds=(0.0, 2.0))

    assert v.name == "v"
    assert v.formula == "v ~ 1 + x + y"
    assert isinstance(v.prior, dict)
    assert v.prior["Intercept"]["name"] == "Uniform"
    assert v.prior["Intercept"]["lower"] == 0.0
    assert v.prior["Intercept"]["upper"] == 0.5
    assert isinstance(v.prior["x"], dict)
    assert v.prior["x"]["name"] == "Uniform"
    assert v.prior["x"]["lower"] == 0.0
    assert v.prior["x"]["upper"] == 1.0
    assert isinstance(v.prior["y"], bmb.Prior)
    assert isinstance(v.prior["z"], float)

    assert v.bounds == (0.0, 2.0)
    assert v.user_param is v_reg

    v = RegressionParam.from_user_param(v_reg_1)
    v.fill_defaults(bounds=(0.0, 2.0))
    assert v.bounds == (0.0, 1.0)

    with pytest.raises(
        ValueError,
        match="v is a regression parameter. It should not have a default prior.",
    ):
        v = RegressionParam.from_user_param(v_reg)
        v.fill_defaults(prior=0, bounds=(2.0, 0.0))

    v = RegressionParam(name="v", formula=None, prior=None, bounds=(0.0, 1.0))
    assert v.formula is None
    v.fill_defaults(formula="v ~ 1 + x + y")
    assert v.formula == "v ~ 1 + x + y"

    v = RegressionParam(name="v", formula="v ~ 1 + x + y", bounds=(0.0, 1.0))
    v.fill_defaults(formula="v ~ 1 + x")
    assert v.formula == "v ~ 1 + x + y"

    with pytest.raises(
        ValueError,
        match="Formula not specified for parameter v.",
    ):
        v = RegressionParam(name="v", formula=None, bounds=(0.0, 1.0))
        v.fill_defaults(bounds=(0.0, 1.0))

    v = RegressionParam(name="v", formula="v ~ 1 + x + y", bounds=(0.0, 1.0))
    v.fill_defaults(bounds=(0.0, 2.0), link_settings="log_logit")

    assert isinstance(v.link, Link)
    assert v.link.name == "gen_logit"
    assert v.link.bounds == (0.0, 1.0)


def test_validate():
    with pytest.raises(
        ValueError,
        match="Formula not specified for parameter v.",
    ):
        v = RegressionParam(name="v", formula=None, bounds=(0.0, 1.0))
        v.validate()

    v = RegressionParam(name="v", formula="v ~1 + x + y", bounds=(0.0, 1.0))
    v.validate()
    assert v.formula == "v ~ 1 + x + y"
    assert v.link == "identity"

    v = RegressionParam(
        name="v", formula="1 + x + y", bounds=(0.0, 1.0), link="log_logit"
    )
    v.validate()
    assert v.formula == "v ~ 1 + x + y"
    assert isinstance(v.link, Link)
    assert v.link.name == "gen_logit"
    assert v.link.bounds == (0.0, 1.0)


angle_config = get_default_model_config("angle")
angle_params = angle_config["list_params"]
angle_bounds = angle_config["likelihoods"]["approx_differentiable"]["bounds"].values()
param_and_bounds_angle = list(
    zip(angle_params, angle_bounds, [False] * len(angle_params))
)

ddm_config = get_default_model_config("full_ddm")
ddm_params = ddm_config["list_params"]
ddm_bounds = ddm_config["likelihoods"]["blackbox"]["bounds"].values()
param_and_bounds_ddm = list(zip(ddm_params, ddm_bounds, [True] * len(ddm_params)))


@pytest.mark.parametrize(
    ("param_name", "bounds", "is_ddm"),
    param_and_bounds_angle,
)
def test_make_safe_priors(cavanagh_test, caplog, param_name, bounds, is_ddm):
    # Necessary for verifying the values of certain parameters of the priors
    hssm.set_floatX("float64")
    # The basic regression case, no group-specific terms
    param = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 1 + theta",
        bounds=bounds,
    )

    param.make_safe_priors(data=cavanagh_test, eval_env={}, is_ddm=is_ddm)

    assert param.prior is not None
    assert (intercept_prior := param.prior["Intercept"]) is not None
    assert (slope_prior := param.prior["theta"]) is not None

    assert isinstance(intercept_prior, Prior)
    assert intercept_prior.is_truncated
    assert intercept_prior.bounds == bounds
    assert intercept_prior.dist is not None
    lower, upper = intercept_prior.bounds
    _mu = intercept_prior._args["mu"]
    if isinstance(_mu, np.ndarray):
        assert _mu.item() == (lower + upper) / 2
    else:
        assert _mu == (lower + upper) / 2
    assert intercept_prior._args["sigma"] == 0.25

    assert isinstance(slope_prior, bmb.Prior)
    assert slope_prior.dist is None
    assert slope_prior.args["mu"] == 0.0
    assert slope_prior.args["sigma"] == 0.25

    unif_prior = {"name": "Uniform", "lower": 0.0, "upper": 1.0}
    set_prior = {
        "Intercept": unif_prior,
        "theta": unif_prior,
    }

    # Test that nothing is overwritten if the prior is already set
    param_with_prior = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 1 + theta",
        prior=set_prior,
        bounds=bounds,
    )

    param_with_prior.make_safe_priors(data=cavanagh_test, eval_env={}, is_ddm=False)
    assert param_with_prior.prior == set_prior

    # The regression case, with group-specific terms
    param_group = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 1 + (1 + theta | participant_id)",
        bounds=bounds,
    )

    param_group.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    assert all(
        param in param_group.prior
        for param in ["Intercept", "1|participant_id", "theta|participant_id"]
    )

    assert param_group.prior["Intercept"].is_truncated

    group_intercept_prior = param_group.prior["1|participant_id"]
    group_slope_prior = param_group.prior["theta|participant_id"]

    _check_group_prior_with_common(group_intercept_prior)
    _check_group_prior(group_slope_prior)

    param_no_common_intercept = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 0 + (1 + theta | participant_id)",
        bounds=bounds,
    )

    param_no_common_intercept.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    assert "limitation" in caplog.records[-1].msg
    assert "Intercept" not in param_no_common_intercept.prior
    group_intercept_prior = param_no_common_intercept.prior["1|participant_id"]
    group_slope_prior = param_no_common_intercept.prior["theta|participant_id"]

    _check_group_prior(group_intercept_prior)
    _check_group_prior(group_slope_prior)

    # Change back after testing
    hssm.set_floatX("float32")


def _check_group_prior(group_prior):
    assert isinstance(group_prior, bmb.Prior)
    assert group_prior.dist is None
    assert group_prior.name == "Normal"

    mu = group_prior.args["mu"]
    sigma = group_prior.args["sigma"]

    assert isinstance(group_prior, bmb.Prior)
    assert mu.name == "Normal"
    assert mu.args["mu"] == 0.0
    assert mu.args["sigma"] == 0.25

    assert isinstance(group_prior, bmb.Prior)
    assert sigma.name == "Weibull"
    assert sigma.args["alpha"] == 1.5
    assert sigma.args["beta"] == 0.3


def _check_group_prior_with_common(group_prior):
    assert isinstance(group_prior, bmb.Prior)
    assert group_prior.dist is None
    assert group_prior.name == "Normal"

    mu = group_prior.args["mu"]
    sigma = group_prior.args["sigma"]

    assert mu == 0.0

    assert isinstance(group_prior, bmb.Prior)
    assert sigma.name == "Weibull"
    assert sigma.args["alpha"] == 1.5
    assert sigma.args["beta"] == 0.3


v_mu = {"name": "Normal", "mu": 2.0, "sigma": 3.0}
v_sigma = {"name": "HalfNormal", "sigma": 2.0}
v_prior = {"name": "Normal", "mu": v_mu, "sigma": v_sigma}

a_mu = {"name": "Gamma", "mu": 1.5, "sigma": 0.75}
a_sigma = {"name": "HalfNormal", "sigma": 0.1}
a_prior = {"name": "Gamma", "mu": a_mu, "sigma": a_sigma}

# AF-TODO: Test below tests for equality between priors name
# and mu name .... z is a special case for this
# These tests probably need to be rewritten following a different
# approach that relies on default dictionaries from prior.py?

# Skipping z for now because I couldn't come up with an immediate
# solution
# z_mu = {"name": "Gamma", "mu": 10.0, "sigma": 10.0}
# z_sigma = {"name": "Gamma", "mu": 10.0, "sigma": 10.0}
# z_prior = {"name": "Beta", "alpha": z_mu, "beta": z_sigma}

t_mu = {"name": "Gamma", "mu": 0.2, "sigma": 0.2}
t_sigma = {"name": "HalfNormal", "sigma": 0.2}
t_prior = {"name": "Gamma", "mu": t_mu, "sigma": t_sigma}


@pytest.mark.parametrize(
    ("param_name", "mu", "prior"),
    [
        ("v", v_mu, v_prior),
        ("a", a_mu, a_prior),
        # ("z", z_mu, z_prior),
        ("t", t_mu, t_prior),
    ],
)
def test_make_safe_priors_ddm(cavanagh_test, caplog, param_name, mu, prior):
    # Necessary for verifying the values of certain parameters of the priors
    hssm.set_floatX("float64")

    bounds = (-10, 10)

    # The basic regression case, no group-specific terms
    param = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 1 + theta",
        bounds=bounds,  # invalid, just for testing
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=True)

    intercept_prior = param.prior["Intercept"]
    slope_prior = param.prior["theta"]

    assert isinstance(intercept_prior, Prior)
    assert intercept_prior.bounds == bounds
    assert intercept_prior.dist is not None
    mu1 = mu.copy()
    print(f"{intercept_prior}=")
    print(f"{mu1}=")

    assert intercept_prior.name == mu1.pop("name")
    for key, val in mu1.items():
        val1 = intercept_prior._args[key]
        np.testing.assert_almost_equal(val1, val)

    assert isinstance(slope_prior, bmb.Prior)
    assert slope_prior.dist is None
    assert slope_prior.args["mu"] == 0.0
    assert slope_prior.args["sigma"] == 0.25

    # If prior is set, do not override
    unif_prior = {"name": "Uniform", "lower": 0.0, "upper": 1.0}
    set_prior = {
        "Intercept": unif_prior,
        "theta": unif_prior,
    }

    param_with_prior = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 1 + theta",
        bounds=bounds,
        prior=set_prior,
    )

    param_with_prior.make_safe_priors(cavanagh_test, {}, is_ddm=True)
    assert param_with_prior.prior == set_prior

    # The regression case, with group-specific terms
    param_group = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 1 + (1 + theta | participant_id)",
        bounds=bounds,
    )

    param_group.make_safe_priors(cavanagh_test, {}, is_ddm=True)

    assert all(
        param in param_group.prior
        for param in ["Intercept", "1|participant_id", "theta|participant_id"]
    )

    assert param_group.prior["Intercept"].is_truncated

    group_intercept_prior = param_group.prior["1|participant_id"]
    group_slope_prior = param_group.prior["theta|participant_id"]

    def _check_group_prior_intercept_ddm(group_prior, prior):
        assert isinstance(group_prior, bmb.Prior)
        assert group_prior.dist is None
        prior1 = prior.copy()
        assert group_prior.name == prior1.pop("name")
        for key, val in prior1.items():
            hyperprior = group_prior.args[key]
            val1 = val.copy()
            assert hyperprior.name == val1.pop("name")
            for key2, val2 in val1.items():
                assert hyperprior.args[key2] == val2

    _check_group_prior_with_common(group_intercept_prior)
    _check_group_prior(group_slope_prior)

    param_no_common_intercept = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 0 + (1 + theta | participant_id)",
        bounds=bounds,
    )

    param_no_common_intercept.make_safe_priors(cavanagh_test, {}, is_ddm=True)
    assert "limitation" in caplog.records[-1].msg

    assert "Intercept" not in param_no_common_intercept.prior
    group_intercept_prior = param_no_common_intercept.prior["1|participant_id"]
    group_slope_prior = param_no_common_intercept.prior["theta|participant_id"]

    _check_group_prior_intercept_ddm(group_intercept_prior, prior)
    _check_group_prior(group_slope_prior)

    # Change back after testing
    hssm.set_floatX("float32")


def test__make_priors_recursive():
    test_dict = {
        "name": "Uniform",
        "lower": 0.1,
        "upper": {"name": "Normal", "mu": 0.5, "sigma": 0.1},
    }

    result_prior = _make_priors_recursive(test_dict)
    assert isinstance(result_prior, bmb.Prior)
    assert isinstance(result_prior.args["upper"], bmb.Prior)
    assert result_prior.args["upper"].name == "Normal"


def test_process_prior():
    prior1 = {
        "name": "Normal",
        "mu": {"name": "Normal", "mu": 0.0, "sigma": 1},
        "sigma": {"name": "HalfNormal", "sigma": 1},
    }
    prior2 = 0.4
    prior3 = bmb.Prior("Normal", mu=0.0, sigma=1.0)

    v = RegressionParam(
        name="v",
        formula="v ~ 1 + x + y",
        prior={
            "Intercept": prior1,
            "x": prior2,
            "y": prior3,
        },
    )

    v.process_prior()

    assert isinstance(v.prior["y"], bmb.Prior)

    assert isinstance(v.prior["Intercept"], bmb.Prior)
    assert v.prior["Intercept"].name == "Normal"
    assert isinstance(v.prior["Intercept"].args["mu"], bmb.Prior)
    assert v.prior["Intercept"].args["mu"].name == "Normal"
    assert v.prior["Intercept"].args["mu"].args["mu"] == 0.0
    assert v.prior["Intercept"].args["mu"].args["sigma"] == 1.0
    assert v.prior["Intercept"].args["sigma"].name == "HalfNormal"
    assert v.prior["Intercept"].args["sigma"].args["sigma"] == 1.0

    assert isinstance(v.prior["x"], float)
    assert v.prior["x"] == prior2

    assert isinstance(v.prior["y"], bmb.Prior)
    assert v.prior["y"] is prior3


def test_repr():
    prior1 = {
        "name": "Normal",
        "mu": {"name": "Normal", "mu": 0.0, "sigma": 1},
        "sigma": {"name": "HalfNormal", "sigma": 1},
    }
    prior2 = 0.4
    prior3 = bmb.Prior("Normal", mu=0.0, sigma=1.0)

    v = RegressionParam(
        name="v",
        formula="v ~ 1 + x + y",
        prior={
            "Intercept": prior1,
            "x": prior2,
            "y": prior3,
        },
    )

    v.process_prior()

    assert (
        repr(v) == "v:\n"
        "    Formula: v ~ 1 + x + y\n"
        "    Priors:\n"
        "        Intercept ~ Normal(mu: Normal(mu: 0.0, sigma: 1.0), "
        "sigma: HalfNormal(sigma: 1.0))\n"
        "        x: 0.4\n"
        "        y ~ Normal(mu: 0.0, sigma: 1.0)\n"
        "    Link: identity"
    )

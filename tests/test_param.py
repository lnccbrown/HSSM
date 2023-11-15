import pytest
import bambi as bmb
import numpy as np

import hssm
from hssm.param import (
    Param,
    _make_default_prior,
    _make_priors_recursive,
    _make_bounded_prior,
)


def test_param_creation_non_regression():
    # Test different param creation strategies
    v = {
        "name": "v",
        "prior": {
            "name": "Normal",
            "mu": 0.0,
            "sigma": 2.0,
        },
    }

    param_v = Param(**v)
    param_v.override_default_link()
    param_v.convert()

    with pytest.raises(
        ValueError,
        match="Cannot override the default link function for parameter v."
        + " The object has already been processed.",
    ):
        param_v.override_default_link()

    assert param_v.name == "v"
    assert isinstance(param_v.prior, bmb.Prior)
    assert param_v.prior.args["mu"] == 0.0
    assert not param_v.is_regression
    assert not param_v.is_truncated
    assert not param_v.is_fixed

    a = {
        "name": "a",
        "prior": dict(
            name="Uniform",
            lower=0.01,
            upper=5,
        ),
        "bounds": (0, 1e5),
    }

    param_a = Param(**a)
    param_a.override_default_link()
    param_a.convert()

    assert param_a.is_truncated
    assert param_a.link is None
    assert not param_a.is_fixed
    assert param_a.prior.is_truncated
    param_a_output = param_a.__str__().split("\r\n")[1].split("Prior: ")[1]
    assert param_a_output == str(param_a.prior)

    # A Uniform prior for `z` over (0, 1) set using bmb.Prior.
    # bounds are not set, existing default bounds will be used
    z = {"name": "z", "prior": bmb.Prior("Uniform", lower=0.0, upper=1.0)}

    param_z = Param(**z)
    param_z.convert()
    assert not param_z.is_truncated
    assert not param_z.is_regression
    assert not param_z.is_fixed

    z1 = {
        "name": "z1",
        "prior": bmb.Prior("Uniform", lower=0.0, upper=1.0),
        "bounds": (0, 1),
    }

    param_z1 = Param(**z1)
    param_z1.convert()
    assert param_z1.is_truncated
    assert not param_z1.is_fixed
    assert param_z1.prior.is_truncated
    param_z1_output = param_z1.__str__().split("\r\n")[1].split("Prior: ")[1]
    assert param_z1_output == str(param_z1.prior)

    # A fixed value for t
    t = {"name": "t", "prior": 0.5, "bounds": (0, 1)}
    param_t = Param(**t)
    param_t.convert()
    assert not param_t.is_truncated
    assert param_t.is_fixed
    param_t_output = str(param_t).split("\r\n")[1].split("Value: ")[1]
    assert param_t_output == str(param_t.prior)

    model = hssm.HSSM(
        model="angle",
        data=hssm.simulate_data(
            model="angle", theta=[0.5, 1.5, 0.5, 0.5, 0.3], size=10
        ),
        include=[v, a, z, t],
    )

    pv, pa, pz, pt, ptheta, _ = model.params.values()
    assert pv.is_truncated
    assert pa.is_truncated
    assert pz.is_truncated
    assert not pt.is_truncated
    assert pt.is_fixed
    assert not ptheta.is_truncated

    model_1 = hssm.HSSM(
        model="angle",
        data=hssm.simulate_data(
            model="angle", theta=[0.5, 1.5, 0.5, 0.5, 0.3], size=10
        ),
        include=[v, a, z, t],
        link_settings="log_logit",
    )

    for param in model_1.params.values():
        assert param.link is None


def test_param_creation_regression():
    v_reg = {
        "name": "v",
        "formula": "v ~ 1 + x + y",
        "prior": {
            "Intercept": {"name": "Uniform", "lower": 0.0, "upper": 0.5},
            "x": dict(name="Uniform", lower=0.0, upper=1.0),
            "y": bmb.Prior("Uniform", lower=0.0, upper=1.0),
            "z": 0.1,
        },
    }

    v_reg_param = Param(**v_reg)
    with pytest.raises(
        ValueError,
        match="Cannot override the default link function. Bounds are not"
        + " specified for parameter v.",
    ):
        v_reg_param.override_default_link()
    v_reg_param.convert()

    assert v_reg_param.is_regression
    assert not v_reg_param.is_fixed
    assert not v_reg_param.is_truncated
    assert v_reg_param.formula == v_reg["formula"]
    with pytest.raises(
        ValueError,
        match="Cannot override the default link function for parameter v."
        + " The object has already been processed.",
    ):
        v_reg_param.override_default_link()

    # Generate some fake simulation data
    intercept = 0.3
    x = np.random.uniform(0.5, 0.2, size=10)
    y = np.random.uniform(0.4, 0.1, size=10)
    z = np.random.uniform(0.2, 0.5, size=10)

    v = intercept + 0.8 * x + 0.3 * y + 0.1 * z

    true_values = np.column_stack([v, np.repeat([[1.5, 0.5, 0.5]], axis=0, repeats=10)])

    dataset_reg_v = hssm.simulate_data(
        model="ddm",
        theta=true_values,
        size=1,  # Generate one data point for each of the 1000 set of true values
    )

    dataset_reg_v["x"] = x
    dataset_reg_v["y"] = y

    model_reg_v = hssm.HSSM(
        data=dataset_reg_v,
        model="ddm",
        include=[v_reg],
    )

    v_reg_param = model_reg_v.params["v"]

    assert v_reg_param.is_regression
    assert not v_reg_param.is_fixed
    assert not v_reg_param.is_truncated
    assert v_reg_param.formula == v_reg["formula"]

    model_reg_v = hssm.HSSM(
        data=dataset_reg_v,
        model="ddm",
        include=[v_reg],
        link_settings="log_logit",
    )

    assert model_reg_v.params["v"].link == "identity"


def test__make_default_prior():
    prior1 = _make_default_prior((-10.0, 10.0))
    assert prior1.name == "Uniform"
    assert prior1.args["lower"] == -10.0
    assert prior1.args["upper"] == 10.0

    prior2 = _make_default_prior((0.0, np.inf))
    assert prior2.name == "HalfNormal"
    assert prior2.args["sigma"] == 2.0

    prior3 = _make_default_prior((1.0, np.inf))
    assert prior3.name == "TruncatedNormal"
    assert prior3.args["mu"] == 1.0
    assert prior3.args["sigma"] == 2.0
    assert prior3.args["lower"] == 1.0

    prior3 = _make_default_prior((-np.inf, 1.0))
    assert prior3.name == "TruncatedNormal"
    assert prior3.args["mu"] == 1.0
    assert prior3.args["sigma"] == 2.0
    assert prior3.args["upper"] == 1.0


def test_param_non_regression():
    param_value = Param("a", prior=0.5)
    param_value.convert()

    param_dict = Param(
        "z",
        prior={
            "name": "Uniform",
            "upper": 0.5,
            "lower": 0.8,
        },
    )
    param_dict.convert()

    param_prior = Param("t", prior=bmb.Prior("Uniform", upper=0.5, lower=0.8))
    param_prior.convert()

    assert param_value.name == "a"
    assert isinstance(param_dict.prior, bmb.Prior)
    assert param_dict.prior == param_prior.prior
    assert not param_value.is_regression

    assert param_value.link is None
    assert param_prior.formula is None

    formula1, d1, link1 = param_dict.parse_bambi()  # pylint: disable=W0212
    formula2, d2, link2 = param_prior.parse_bambi()  # pylint: disable=W0212

    assert formula1 is None and formula2 is None
    assert isinstance(d1["z"], bmb.Prior)
    assert d1["z"] == d2["t"]
    assert link1 is None and link2 is None

    assert repr(param_value) == "a:\r\n    Value: 0.5"
    assert (
        repr(param_dict)
        == f"z:\r\n    Prior: {param_dict.prior}\r\n    Explicit bounds: None"
    )

    with pytest.raises(
        ValueError, match="`link` should be None if no regression is specified."
    ):
        Param("t", 0.5, link="identity").convert()

    with pytest.raises(ValueError, match="Please specify the prior or bounds for a."):
        Param("a").convert()


def test_param_regression():
    def fake_func(x):
        return x * 2  # pylint: disable=C3001

    fake_link = bmb.Link(
        "Fake", link=fake_func, linkinv=fake_func, linkinv_backend=fake_func
    )

    priors_dict = {
        "Intercept": {
            "name": "Normal",
            "mu": 0,
            "sigma": 0.5,
        },
        "x1": bmb.Prior("Normal", mu=0, sigma=0.5),
    }

    param_reg_formula1 = Param("a", formula="1 + x1", prior=priors_dict)
    param_reg_formula2 = Param(
        "a", formula="a ~ 1 + x1", prior=priors_dict, link=fake_link
    )

    param_reg_parent = Param("a", formula="a ~ 1 + x1")

    param_reg_formula1.convert()
    param_reg_formula2.convert()
    param_reg_parent.set_parent()
    param_reg_parent.convert()

    assert param_reg_formula1.formula == "a ~ 1 + x1"
    assert isinstance(param_reg_formula2.link, bmb.Link)

    dep_priors2 = param_reg_formula2.prior

    assert isinstance(dep_priors2["Intercept"], bmb.Prior)
    assert dep_priors2["Intercept"] == dep_priors2["x1"]

    formula1, d1, link1 = param_reg_formula1.parse_bambi()  # pylint: disable=W0212
    formula2, d2, link2 = param_reg_formula2.parse_bambi()  # pylint: disable=W0212

    assert formula1 == formula2
    assert d1 == d2
    assert link1["a"] == "identity"
    assert link2["a"].name == "Fake"

    formula3, d3, _ = param_reg_parent.parse_bambi()  # pylint: disable=W0212

    assert formula3 == "c(rt, response) ~ 1 + x1"
    assert param_reg_parent.formula == "a ~ 1 + x1"

    assert d3 is None

    rep = repr(param_reg_parent)
    lines = rep.split("\r\n")
    assert lines[3] == "        Unspecified. Using defaults"


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


def test__make_bounded_prior(caplog):
    name = "v"
    bounds = (0, 1)
    prior0 = 0.01
    prior1 = -10.0
    prior2 = dict(name="Uniform", lower=-10.0, upper=10.0)
    prior3 = bmb.Prior(**prior2)
    prior4 = hssm.Prior(**prior2)
    prior5 = hssm.Prior(bounds=bounds, **prior2)
    prior6 = bmb.Prior("Uniform", dist=lambda x: x)
    prior7 = hssm.Prior("Uniform", dist=lambda x: x)

    assert _make_bounded_prior(name, prior0, bounds) == 0.01

    # pass floats that are out of bounds should raise error
    with pytest.raises(ValueError):
        _make_bounded_prior(name, prior1, bounds)

    bounded_prior2 = _make_bounded_prior(name, prior2, bounds)
    assert isinstance(bounded_prior2, hssm.Prior)
    assert bounded_prior2.is_truncated
    assert bounded_prior2.dist is not None
    assert bounded_prior2.bounds == bounds

    bounded_prior3 = _make_bounded_prior(name, prior3, bounds)
    bounded_prior4 = _make_bounded_prior(name, prior4, bounds)

    assert bounded_prior2._args == bounded_prior3._args
    assert bounded_prior3._args == bounded_prior4._args
    assert bounded_prior2._args == bounded_prior4._args

    # pass priors that are already bounded should raise error
    with pytest.raises(ValueError):
        _make_bounded_prior(name, prior5, bounds)

    # pass priors that are defined with `dist` should raise warnings.
    _make_bounded_prior(name, prior6, bounds)
    _make_bounded_prior(name, prior7, bounds)

    assert caplog.records[0].msg == caplog.records[1].msg


some_forumla = "1 + x + y"


@pytest.mark.parametrize(
    ("formula", "link", "bounds", "result"),
    [
        (None, None, (0, 1), None),
        (some_forumla, None, None, "Error"),
        (some_forumla, None, (0, 1), "gen_logit"),
        (some_forumla, None, (0, np.inf), "log"),
        (some_forumla, None, (-np.inf, 1), "warning"),
        (some_forumla, None, (-np.inf, np.inf), "identity"),
        (some_forumla, "logit", None, "logit"),
        (some_forumla, "gen_logit", None, "Error"),
    ],
)
def test_param_override_default_link(caplog, formula, link, bounds, result):
    if result == "Error":
        with pytest.raises(ValueError):
            param = Param("a", formula=formula, link=link, bounds=bounds)
            param.override_default_link()
    else:
        param = Param("a", formula=formula, link=link, bounds=bounds)
        param.override_default_link()
        param.convert()
        if result == "warning":
            assert "strange" in caplog.records[0].msg
        else:
            if result == "gen_logit":
                assert isinstance(param.link, hssm.Link)
            elif result is None:
                assert param.link is None
            else:
                assert param.link == result

        with pytest.raises(ValueError):
            param.override_default_link()

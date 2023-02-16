import bambi as bmb
import pytest

from hssm.utils import Param


def test_param_non_regression():

    param_value = Param("a", prior=0.5)

    param_dict = Param(
        "z",
        prior={
            "name": "Uniform",
            "upper": 0.5,
            "lower": 0.8,
        },
    )

    param_prior = Param("t", prior=bmb.Prior("Uniform", upper=0.5, lower=0.8))

    assert param_value.name == "a"
    assert isinstance(param_dict.prior, bmb.Prior)
    assert param_dict.prior == param_prior.prior
    assert not param_value.is_regression()

    formula1, d1, link1 = param_dict._parse_bambi()  # pylint: disable=W0212
    formula2, d2, link2 = param_prior._parse_bambi()  # pylint: disable=W0212

    assert formula1 is None and formula2 is None
    assert isinstance(d1["z"], bmb.Prior)
    assert d1["z"] == d2["t"]
    assert link1 is None and link2 is None

    assert repr(param_value) == "a = 0.5"
    assert repr(param_dict) == f"z ~ {param_dict.prior}"

    with pytest.raises(
        ValueError,
        match="A regression model is specified for parameter t."
        + " `prior` parameter should not be specified.",
    ):
        Param("t", prior=0.1, formula="1 + x1 + x2")

    with pytest.raises(
        ValueError, match="Please specify a value or a prior for parameter z."
    ):
        Param("z")

    with pytest.raises(
        ValueError,
        match="dep_priors should not be specified for z if a formula "
        + "is not specified",
    ):
        Param("z", prior=0.1, dep_priors={})


def test_param_regression():

    fake_func = lambda x: x * 2  # pylint: disable=C3001
    fake_link = bmb.Link(
        "Fake", link=fake_func, linkinv=fake_func, linkinv_backend=fake_func
    )

    priors_dict = {
        "intercept": {
            "name": "Normal",
            "mu": 0,
            "sigma": 0.5,
        },
        "x1": bmb.Prior("Normal", mu=0, sigma=0.5),
    }

    param_reg_formula1 = Param("a", formula="1 + x1", dep_priors=priors_dict)
    param_reg_formula2 = Param(
        "a", formula="a ~ 1 + x1", dep_priors=priors_dict, link=fake_link
    )

    assert param_reg_formula1.formula == "a ~ 1 + x1"
    assert isinstance(param_reg_formula2.link, bmb.Link)

    dep_priors2 = param_reg_formula2.dep_priors

    assert isinstance(dep_priors2["intercept"], bmb.Prior)
    assert dep_priors2["intercept"] == dep_priors2["x1"]

    formula1, d1, link1 = param_reg_formula1._parse_bambi()  # pylint: disable=W0212
    formula2, d2, link2 = param_reg_formula2._parse_bambi()  # pylint: disable=W0212

    assert formula1 == formula2
    assert d1 == d2
    assert link1["a"] == "identity"
    assert link2["a"].name == "Fake"

    with pytest.raises(
        ValueError,
        match="A regression model is specified for parameter z."
        + " `prior` parameter should not be specified.",
    ):
        Param("z", formula="1 + x1", dep_priors=priors_dict, prior=0.5)

    with pytest.raises(
        ValueError,
        match="Priors for the variables that z is regressed on " + "are not specified.",
    ):
        Param("z", formula="1 + x1")

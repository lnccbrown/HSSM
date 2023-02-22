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

    assert param_value.link is None
    assert param_prior.formula is None

    formula1, d1, link1 = param_dict._parse_bambi()  # pylint: disable=W0212
    formula2, d2, link2 = param_prior._parse_bambi()  # pylint: disable=W0212

    assert formula1 is None and formula2 is None
    assert isinstance(d1["z"], bmb.Prior)
    assert d1["z"] == d2["t"]
    assert link1 is None and link2 is None

    assert repr(param_value) == "a = 0.5"
    assert repr(param_dict) == f"z ~ {param_dict.prior}"

    with pytest.raises(
        ValueError, match="`link` should be None if no regression is specified."
    ):
        Param("t", 0.5, link="identity")


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

    param_reg_formula1 = Param("a", formula="1 + x1", prior=priors_dict)
    param_reg_formula2 = Param(
        "a", formula="a ~ 1 + x1", prior=priors_dict, link=fake_link
    )

    param_reg_parent = Param(
        "a", formula="a ~ 1 + x1", prior=priors_dict, is_parent=True
    )

    assert param_reg_formula1.formula == "a ~ 1 + x1"
    assert isinstance(param_reg_formula2.link, bmb.Link)

    dep_priors2 = param_reg_formula2.prior

    assert isinstance(dep_priors2["intercept"], bmb.Prior)
    assert dep_priors2["intercept"] == dep_priors2["x1"]

    formula1, d1, link1 = param_reg_formula1._parse_bambi()  # pylint: disable=W0212
    formula2, d2, link2 = param_reg_formula2._parse_bambi()  # pylint: disable=W0212

    assert formula1 == formula2
    assert d1 == d2
    assert link1["a"] == "identity"
    assert link2["a"].name == "Fake"

    assert param_reg_parent.formula == "c(rt, response) ~ 1 + x1"

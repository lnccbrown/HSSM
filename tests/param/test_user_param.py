import bambi as bmb
from hssm.param_refactor.user_param import UserParam


def test_user_param_initialization():
    param = UserParam(
        name="param1", prior=0.5, formula="y ~ x", link="identity", bounds=(-1, 1)
    )
    assert param.name == "param1"
    assert param.prior == 0.5
    assert param.formula == "y ~ x"
    assert param.link == "identity"
    assert param.bounds == (-1, 1)


def test_user_param_from_dict():
    param_dict = {
        "name": "param2",
        "prior": {"mu": 0, "sigma": 1},
        "formula": "y ~ z",
        "link": "logit",
        "bounds": (0, 1),
    }
    param = UserParam.from_dict(param_dict)
    assert param.name == "param2"
    assert param.prior == {"mu": 0, "sigma": 1}
    assert param.formula == "y ~ z"
    assert param.link == "logit"
    assert param.bounds == (0, 1)


def test_user_param_from_kwargs_with_dict():
    param = UserParam.from_kwargs(
        name="param3",
        param={
            "prior": {"mu": 0, "sigma": 1},
            "formula": "y ~ w",
            "link": "probit",
            "bounds": (0, 1),
        },
    )
    assert param.name == "param3"
    assert param.prior == {"mu": 0, "sigma": 1}
    assert param.formula == "y ~ w"
    assert param.link == "probit"
    assert param.bounds == (0, 1)


def test_user_param_from_kwargs_with_user_param():
    existing_param = UserParam(
        name="param4", prior=1.0, formula="y ~ v", link="log", bounds=(-2, 2)
    )
    param = UserParam.from_kwargs(name="param5", param=existing_param)
    assert param.name == "param5"
    assert param.prior == 1.0
    assert param.formula == "y ~ v"
    assert param.link == "log"
    assert param.bounds == (-2, 2)


def test_user_param_from_kwargs_with_prior():
    bambi_prior = bmb.Prior("Normal", mu=0, sigma=1)
    param = UserParam.from_kwargs(name="param6", param=bambi_prior)
    assert param.name == "param6"
    assert param.prior == bambi_prior
    assert param.formula is None
    assert param.link is None
    assert param.bounds is None


def test_user_param_from_kwargs_constant():
    constant = 0.0
    param = UserParam.from_kwargs(name="param6", param=constant)
    assert param.name == "param6"
    assert param.prior == constant
    assert param.formula is None
    assert param.link is None
    assert param.bounds is None

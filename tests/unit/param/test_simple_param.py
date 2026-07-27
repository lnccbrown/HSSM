import bambi as bmb
import numpy as np
import pytest
import re

from hssm import Prior
from hssm.param import UserParam
from hssm.param.simple_param import SimpleParam, DefaultParam


def test_from_user_param():
    """Test that the from_user_param class method works correctly."""
    user_param = UserParam(name="test", prior=0, formula="x", bounds=(0, 1))
    with pytest.raises(
        ValueError, match="Regression specified for simple parameter test"
    ):
        SimpleParam.from_user_param(user_param)

    user_param = UserParam(name="test", prior=0, link="identity", bounds=(0, 1))
    with pytest.raises(ValueError, match="Link specified for simple parameter test"):
        SimpleParam.from_user_param(user_param)

    user_param = UserParam(name="test", prior=0, bounds=(0, 1))
    param = SimpleParam.from_user_param(user_param)
    assert isinstance(param, SimpleParam)
    assert param.name == "test"
    assert param.prior == 0
    assert param.bounds == (0, 1)
    assert param.user_param is user_param

    user_param = UserParam(name="test", bounds=(0, 1))
    param = SimpleParam.from_user_param(user_param)
    assert isinstance(param, DefaultParam)
    assert param.name == "test"
    assert param.prior is None
    assert param.bounds == (0, 1)
    assert param.user_param is None


def test_validate():
    """Test that the validate method works correctly."""
    param = SimpleParam(name="test", bounds=(0, 1))
    with pytest.raises(ValueError, match="Prior not specified for parameter test"):
        param.validate()

    param.prior = -0.1
    with pytest.raises(
        ValueError,
        match=re.escape("Fixed Value -0.1 not in bounds (0, 1) for parameter test"),
    ):
        param.validate()

    param.prior = np.array([-0.1, 0.5])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Fixed Value [-0.1  0.5] not in bounds (0, 1) for parameter test"
        ),
    ):
        param.validate()

    param.prior = {"name": "Normal", "mu": -0.1, "sigma": 0.5}
    param.validate()


def test_fill_defaults():
    """Test that the fill_defaults method works correctly."""
    param = SimpleParam(name="test", prior=0)
    param.fill_defaults(prior=100, bounds=(0, 1))
    assert param.prior == 0
    assert param.bounds == (0, 1)

    with pytest.raises(ValueError, match="Formula specified for simple parameter test"):
        param.fill_defaults(prior=0, formula="x")
    with pytest.raises(ValueError, match="Link specified for simple parameter test"):
        param.fill_defaults(prior=0, link="identity")


def test_process_prior():
    """Test that the process_prior method works correctly."""
    param = SimpleParam(name="test", prior={"name": "Normal", "mu": 0, "sigma": 1})
    param.process_prior()
    assert isinstance(param.prior, bmb.Prior) and not isinstance(param.prior, Prior)
    assert param.prior.name == "Normal"
    assert param.prior.args["mu"] == 0
    assert param.prior.args["sigma"] == 1

    param = SimpleParam(
        name="test", prior={"name": "Normal", "mu": 0, "sigma": 1}, bounds=(0, 1)
    )
    param.process_prior()
    assert isinstance(param.prior, Prior)
    assert param.prior.name == "Normal"
    assert param.prior._args["mu"] == 0
    assert param.prior._args["sigma"] == 1
    assert param.prior.bounds == (0, 1)
    assert param.prior.is_truncated


def test_repr():
    """Test the __repr__ method."""
    param = SimpleParam(name="test", prior={"name": "Normal", "mu": 0, "sigma": 1})
    param.process_prior()
    assert repr(param) == "test:\n    Prior: Normal(mu: 0.0, sigma: 1.0)"
    param = SimpleParam(
        name="test", prior={"name": "Normal", "mu": 0, "sigma": 1}, bounds=(0, 1)
    )
    param.process_prior()
    print(repr(param))
    assert (
        repr(param) == "test:\n"
        "    Prior: Normal(mu: 0.0, sigma: 1.0)\n"
        "    Explicit bounds: (0, 1)"
    )
    param = SimpleParam(name="test", prior=0.5, bounds=(0, 1))
    param.process_prior()
    assert repr(param) == "test:\n    Value: 0.5\n    Explicit bounds: (0, 1)"
    prior = np.random.uniform(size=3)
    param = SimpleParam(name="test", prior=prior, bounds=(0, 1))
    param.process_prior()
    assert repr(param) == f"test:\n    Value: {prior}\n    Explicit bounds: (0, 1)"
    prior = np.random.uniform(size=10)
    param = SimpleParam(name="test", prior=prior, bounds=(0, 1))
    param.process_prior()
    assert (
        repr(param) == f"test:\n    Value: {prior[:5]}...\n    Explicit bounds: (0, 1)"
    )

import numpy as np
import pytest

from hssm.param import UserParam
from hssm.param.param import Param


def test_bounds_validation():
    """Test that the bounds are validated correctly."""
    with pytest.raises(ValueError, match=r"Invalid bounds: \(0, 0, 1\)"):
        Param(name="test", prior=0, bounds=(0, 0, 1))
    with pytest.raises(ValueError, match=r"Invalid bounds: \(1, 0\)"):
        Param(name="test", prior=0, bounds=(1, 0))


def test_from_user_param():
    """Test that the from_user_param class method works correctly."""
    user_param = UserParam(
        name="test", prior=0, formula="x", link="identity", bounds=(0, 1)
    )
    param = Param.from_user_param(user_param)
    assert param.name == "test"
    assert param.prior == 0
    assert param.formula == "x"
    assert param.link == "identity"
    assert param.bounds == (0, 1)
    assert param.user_param is user_param


def test_is_regression():
    """Test that the is_regression property works correctly."""
    assert not Param(name="test", prior=0).is_regression
    assert Param(name="test", prior=0, formula="x").is_regression


def test_is_fixed():
    """Test that the is_fixed property works correctly."""
    assert Param(name="test", prior=0).is_fixed
    assert Param(name="test", prior=np.array([0, 1])).is_fixed
    assert not Param(name="test", prior={"mean": 0, "std": 1}).is_fixed


def test_is_parent():
    """Test that the is_parent property works correctly."""
    param = Param(name="test", prior=0)
    assert not param.is_parent
    param.is_parent = True
    assert param.is_parent


def test_is_vector():
    """Test that the is_vector property works correctly."""
    assert not Param(name="test", prior=0).is_vector
    assert Param(name="test", prior=np.array([0, 1])).is_vector
    assert not Param(name="test", prior={"mean": 0, "std": 1}).is_vector


def test_fill_defaults():
    """Test that the fill_defaults method works correctly."""
    param = Param(name="test", prior=0)
    param.fill_defaults()
    assert param.formula is None
    assert param.link is None
    assert param.bounds is None

    param = Param(name="test", prior=0, formula="x", link="identity", bounds=(0, 1))
    param.fill_defaults(
        name="test_1", prior=1, formula="y", link="logit", bounds=(1, 2)
    )
    assert param.formula == "x"
    assert param.link == "identity"
    assert param.bounds == (0, 1)
    assert param.user_param is None
    assert not param.is_parent
    assert param.is_fixed
    assert param.is_vector
    assert param.is_regression

    param = Param(name="test", prior=np.array([0, 1]))
    param.fill_defaults(prior=0, formula="x", link="identity", bounds=(0, 1))
    assert param.formula == "x"
    assert np.all(param.prior == np.array([0, 1]))
    assert param.link == "identity"
    assert param.bounds == (0, 1)
    assert param.user_param is None


def test_unimplemented_methods():
    """Test that the unimplemented methods raise NotImplementedError."""
    param = Param(name="test", prior=0)
    with pytest.raises(NotImplementedError):
        param.validate()
    with pytest.raises(NotImplementedError):
        param.process_prior()

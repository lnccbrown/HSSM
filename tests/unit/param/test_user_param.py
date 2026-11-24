import pytest

from hssm.param import UserParam


def test_bounds_validation():
    """Test that the bounds are validated correctly."""
    with pytest.raises(ValueError, match=r"Invalid bounds: \(0, 0, 1\)"):
        UserParam(name="test", prior=0, bounds=(0, 0, 1))
    with pytest.raises(ValueError, match=r"Invalid bounds: \(1, 0\)"):
        UserParam(name="test", prior=0, bounds=(1, 0))


def test_is_regression():
    """Test that the is_regression property works correctly."""
    assert not UserParam(name="test", prior=0).is_regression
    assert UserParam(name="test", prior=0, formula="x").is_regression


def test_from_dict():
    """Test that the from_dict class method works correctly."""
    param_dict = {
        "name": "test",
        "prior": 0,
        "formula": "x",
        "link": "identity",
        "bounds": (0, 1),
    }
    param = UserParam.from_dict(param_dict)
    assert param.name == "test"
    assert param.prior == 0
    assert param.formula == "x"
    assert param.link == "identity"
    assert param.bounds == (0, 1)


def test_from_kwargs():
    """Test that the from_kwargs class method works correctly."""
    param = UserParam.from_kwargs("test", 0)
    assert param.name == "test"
    assert param.prior == 0

    param = UserParam.from_kwargs("test", {"prior": 0, "formula": "x"})
    assert param.name == "test"
    assert param.prior == 0
    assert param.formula == "x"

    user_param = UserParam(name="test", prior=0)
    param = UserParam.from_kwargs("test", user_param)
    assert param.name == "test"
    assert param.prior == 0
    assert param is user_param


def test_to_dict():
    """Test that the to_dict method works correctly."""
    param = UserParam(name="test", prior=0, formula="x", link="identity", bounds=(0, 1))
    param_dict = param.to_dict()
    assert param_dict == {
        "name": "test",
        "prior": 0,
        "formula": "x",
        "link": "identity",
        "bounds": (0, 1),
    }

    param = UserParam(name="test", prior=0, bounds=(0, 1))
    param_dict = param.to_dict()
    assert param_dict == {
        "name": "test",
        "prior": 0,
        "bounds": (0, 1),
    }

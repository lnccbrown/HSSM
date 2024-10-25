import re
import bambi as bmb
import numpy as np
import pytest

from hssm.param.user_param import UserParam
from hssm.param.utils import deserialize_prior

from tests.param.test_utils import test_cases_deserialize, test_cases_serialize


def test_bounds_validation():
    """Test that the bounds are validated correctly."""
    with pytest.raises(ValueError, match=re.escape("Invalid bounds: (0, 0, 1)")):
        UserParam(name="test", prior=0, bounds=(0, 0, 1))
    with pytest.raises(ValueError, match=re.escape("Invalid bounds: (1, 0)")):
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


def _check_equivalence(user_param, d):
    """Check that the user_param and dictionary are equivalent."""
    for attr in ["name", "formula", "link", "bounds"]:
        if attr in d:
            assert getattr(user_param, attr) == d[attr]
        else:
            assert getattr(user_param, attr) is None

    if "prior" in d:
        if isinstance(user_param.prior, np.ndarray):
            assert np.allclose(user_param.prior, deserialize_prior(d["prior"]))
        else:
            assert user_param.prior == deserialize_prior(d["prior"])


@pytest.mark.parametrize("prior", test_cases_serialize)
def test_serialize(prior):
    """Test that the UserParam object can be serialized and deserialized."""
    user_param = UserParam(name="test", prior=prior, link="identity", bounds=(0, 1))
    d = user_param.serialize()
    _check_equivalence(user_param, d)


def test_serialize_link_obj():
    """Test that the UserParam object can be serialized and deserialized."""
    user_param = UserParam(name="test", prior=0, link=bmb.Link("identity"))
    with pytest.raises(
        ValueError, match="Cannot serialize link object for parameter test"
    ):
        user_param.serialize()


@pytest.mark.parametrize("prior", test_cases_deserialize)
def test_deserialize(prior):
    """Test that the UserParam object can be serialized and deserialized."""
    d = {
        "name": "test",
        "prior": prior,
        "link": "identity",
        "bounds": (0, 1),
    }
    user_param = UserParam.deserialize(d.copy())

    _check_equivalence(user_param, d)

import bambi as bmb
import numpy as np
import pytest

from hssm import Prior
from hssm.param.utils import (
    deserialize_prior,
    serialize_prior,
    str_to_array,
    array_to_str,
)

normal_prior = bmb.Prior("Normal", mu=0.0, sigma=1.0)
half_normal_prior = bmb.Prior("HalfNormal", sigma=1.0)
nested_prior = bmb.Prior(
    "Normal",
    mu=bmb.Prior("Normal", mu=0.0, sigma=1.0),
    sigma=bmb.Prior("HalfNormal", sigma=1.0),
)

normal_prior_dict = {
    "name": "Normal",
    "mu": 0.0,
    "sigma": 1.0,
    "auto_scale": True,
}
half_normal_prior_dict = {
    "name": "HalfNormal",
    "sigma": 1.0,
    "auto_scale": True,
}
nested_prior_dict = {
    "name": "Normal",
    "mu": normal_prior_dict,
    "sigma": half_normal_prior_dict,
    "auto_scale": True,
}

test_cases_serialize = [
    1.0,  # constant
    np.array([1.0, 2.0]),  # array
    normal_prior_dict,  # dictionary
    Prior.from_bambi(normal_prior, bounds=(0.0, 1.0)),  # hssm.Prior
    nested_prior,  # Nested Prior
    {
        "Intercept": normal_prior_dict,
        "x": normal_prior_dict,
    },  # nested dictionary
    {
        "Intercept": normal_prior,
        "x": half_normal_prior,
    },  # dictionary with bmb.Prior
]

test_cases_deserialize = [
    {"type": "constant", "value": 1.0},  # constant
    {"type": "array", "value": "1.000000,2.000000"},  # array
    {
        "type": "simple",
        "value": normal_prior_dict,
    },  # dictionary
    {
        "type": "prior",
        "value": normal_prior_dict | {"bounds": (0.0, 1.0)},
    },  # hssm.Prior
    {
        "type": "prior",
        "value": nested_prior_dict,
    },
    {
        "type": "regression",
        "value": {
            "Intercept": {
                "type": "simple",
                "value": normal_prior_dict,
            },
            "x": {
                "type": "simple",
                "value": normal_prior_dict,
            },
        },
    },
    {
        "type": "regression",
        "value": {
            "Intercept": {
                "type": "prior",
                "value": normal_prior_dict,
            },
            "x": {
                "type": "prior",
                "value": half_normal_prior_dict,
            },
        },
    },
]


def test_array_to_str():
    """Test that the array_to_str function works correctly."""
    assert array_to_str(np.array([1.0, 2.0])) == "1.000000,2.000000"
    assert array_to_str(np.array([1.0])) == "1.000000"
    assert array_to_str(np.array([])) == ""


def test_str_to_array():
    """Test that the str_to_array function works correctly."""
    assert np.allclose(str_to_array("1.0,2.0"), np.array([1.0, 2.0]))
    assert np.allclose(str_to_array("1.0"), np.array([1.0]))
    assert np.allclose(str_to_array(""), np.array([]))


@pytest.mark.parametrize(
    "prior,expected", zip(test_cases_serialize, test_cases_deserialize)
)
def test_serialize_prior(prior, expected):
    """Test that the serialize_prior function works correctly."""
    serialized_prior = serialize_prior(prior)

    assert serialized_prior == expected


@pytest.mark.parametrize(
    "serialized_prior,expected", zip(test_cases_deserialize, test_cases_serialize)
)
def test_deserialize_prior(serialized_prior, expected):
    """Test that the deserialize_prior function works correctly."""
    deserialized_prior = deserialize_prior(serialized_prior)

    prior_type = serialized_prior["type"]

    match prior_type:
        case "constant":
            assert deserialized_prior == expected
        case "array":
            assert np.allclose(deserialized_prior, expected)
        case _:
            assert deserialized_prior == expected


@pytest.mark.parametrize("prior", test_cases_serialize)
def test_equivalence_before_and_after_serialization(prior):
    """Test that the serialized and deserialized priors are equivalent."""
    serialized_prior = serialize_prior(prior)
    deserialized_prior = deserialize_prior(serialized_prior)

    if isinstance(prior, np.ndarray):
        assert np.allclose(prior, deserialized_prior)
    else:
        assert prior == deserialized_prior

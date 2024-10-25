from typing import Any

import pytest

import bambi as bmb
import numpy as np

import hssm
from hssm.prior import Prior, serialize_prior_obj, deserialize_prior_obj

hssm.set_floatX("float32")


def _check_prior_dict_equivalence(prior: bmb.Prior, prior_dict: dict[str, Any]):
    prior_dict = prior_dict.copy()
    assert prior.name == prior_dict.pop("name")
    if auto_scale := prior_dict.pop("auto_scale", None):
        assert prior.auto_scale == auto_scale
    if bounds := prior_dict.pop("bounds", None):
        assert isinstance(prior, Prior)
        assert prior.bounds == bounds
        assert prior.is_truncated

    args = (
        prior._args if isinstance(prior, Prior) and prior.is_truncated else prior.args
    )

    for key, value in args.items():
        if isinstance(value, bmb.Prior):
            _check_prior_dict_equivalence(value, prior_dict.pop(key))
        else:
            assert value == prior_dict.pop(key)


@pytest.fixture(scope="function")
def flat_prior_dict():
    return {
        "name": "Normal",
        "bounds": (-1.0, 1.0),
        "auto_scale": True,
        "mu": 0,
        "sigma": 1,
    }


@pytest.fixture(scope="function")
def flat_prior():
    return Prior(
        "Normal", auto_scale=True, dist=None, bounds=(-1.0, 1.0), mu=0, sigma=1
    )


@pytest.fixture(scope="function")
def nested_prior_dict():
    return {
        "name": "Normal",
        "bounds": (-1.0, 1.0),
        "auto_scale": True,
        "mu": {
            "name": "Normal",
            "auto_scale": False,
            "mu": 0,
            "sigma": 1,
        },
        "sigma": {
            "name": "Normal",
            "auto_scale": False,
            "mu": 0,
            "sigma": 1,
        },
    }


@pytest.fixture(scope="function")
def nested_prior():
    return Prior(
        "Normal",
        auto_scale=True,
        dist=None,
        bounds=(-1.0, 1.0),
        mu=Prior("Normal", auto_scale=False, mu=0, sigma=1),
        sigma=bmb.Prior("Normal", auto_scale=False, mu=0, sigma=1),
    )


def test_truncation():
    hssm_prior = Prior("Uniform", lower=0.0, upper=1.0)
    bmb_prior = bmb.Prior("Uniform", lower=0.0, upper=1.0)
    assert hssm_prior.args == bmb_prior.args

    bounded_prior1 = Prior("Uniform", lower=0.0, upper=1.0, bounds=(0.2, 0.8))
    assert bounded_prior1.is_truncated
    assert bounded_prior1._args == bmb_prior.args
    assert callable(bounded_prior1.dist)
    assert not bounded_prior1.args

    prior2 = Prior("Uniform", lower=0.0, upper=1.0, bounds=(-np.inf, np.inf))
    assert not prior2.is_truncated
    assert prior2.dist is None

    with pytest.raises(ValueError):
        bounded_prior_err = Prior(
            "Uniform", lower=0.0, upper=1.0, bounds=(0.2, 0.8), dist=lambda x: x
        )


def test_str():
    hssm_prior = Prior("Uniform", lower=0.3, upper=1.0)
    bmb_prior = bmb.Prior("Uniform", lower=0.3, upper=1.0)

    assert str(hssm_prior) == str(bmb_prior)

    bounded_prior = Prior("Uniform", lower=0.3, upper=1.0, bounds=(0.4, 0.8))
    assert str(bounded_prior) == str(bmb_prior)


def test_eq():
    hssm_prior = Prior("Uniform", lower=0.3, upper=1.0)
    bmb_prior = bmb.Prior("Uniform", lower=0.3, upper=1.0)

    bounded_prior = Prior("Uniform", lower=0.3, upper=1.0, bounds=(0.4, 0.8))
    bounded_prior1 = Prior("Uniform", lower=0.3, upper=1.0, bounds=(0.4, 0.8))
    bounded_prior2 = Prior("Uniform", lower=0.3, upper=1.0, bounds=(-np.inf, np.inf))

    dist = lambda x: x

    dist_hssm_prior = Prior("Uniform", dist=dist)
    dist_bmb_prior = bmb.Prior("Uniform", dist=dist)

    assert hssm_prior == bmb_prior
    assert bounded_prior != bmb_prior

    assert bounded_prior == bounded_prior1
    assert hssm_prior == bounded_prior2

    assert dist_hssm_prior == dist_bmb_prior


def test_serialize_prior_obj(flat_prior, nested_prior):
    _check_prior_dict_equivalence(flat_prior, serialize_prior_obj(flat_prior))
    _check_prior_dict_equivalence(nested_prior, serialize_prior_obj(nested_prior))


def test_deserialize_prior_obj(flat_prior_dict, nested_prior_dict):
    _check_prior_dict_equivalence(
        deserialize_prior_obj(flat_prior_dict), flat_prior_dict
    )
    _check_prior_dict_equivalence(
        deserialize_prior_obj(nested_prior_dict), nested_prior_dict
    )


def test_to_dict(flat_prior, nested_prior):
    _check_prior_dict_equivalence(flat_prior, flat_prior.to_dict())
    _check_prior_dict_equivalence(nested_prior, nested_prior.to_dict())


def test_from_dict(flat_prior_dict, nested_prior_dict):
    _check_prior_dict_equivalence(Prior.from_dict(flat_prior_dict), flat_prior_dict)
    _check_prior_dict_equivalence(Prior.from_dict(nested_prior_dict), nested_prior_dict)


def test_equivalence_before_and_after_serialization(flat_prior, nested_prior):
    assert flat_prior == deserialize_prior_obj(serialize_prior_obj(flat_prior))
    assert nested_prior == deserialize_prior_obj(serialize_prior_obj(nested_prior))


def test_from_bambi():
    bambi_prior = bmb.Prior("Normal", auto_scale=True, dist=None, mu=0, sigma=1)
    prior_from_bambi_unbounded = Prior.from_bambi(bambi_prior)

    _check_prior_dict_equivalence(bambi_prior, prior_from_bambi_unbounded.to_dict())

    prior_from_bambi_bounded = Prior.from_bambi(bambi_prior, bounds=(-1.0, 1.0))
    assert prior_from_bambi_bounded.is_truncated
    assert prior_from_bambi_bounded.bounds == (-1.0, 1.0)
    assert prior_from_bambi_bounded.dist is not None
    assert prior_from_bambi_bounded._args == prior_from_bambi_unbounded.args

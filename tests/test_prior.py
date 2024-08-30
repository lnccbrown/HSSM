import pytest

import bambi as bmb
import numpy as np

import hssm
from hssm import Prior

hssm.set_floatX("float32")


@pytest.fixture
def prior():
    name = "Normal"
    auto_scale = True
    dist = None
    bounds = (-1.0, 1.0)
    args = {"mu": 0, "sigma": 1}
    return Prior(name, auto_scale, dist, bounds, **args)


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


def test_to_dict(prior):
    expected_dict = {
        "name": prior.name,
        "bounds": prior.bounds,
        "auto_scale": prior.auto_scale,
        "mu": 0,
        "sigma": 1,
    }
    assert prior.to_dict() == expected_dict


def test_from_dict():
    prior_dict = {
        "name": "Normal",
        "bounds": (-1.0, 1.0),
        "auto_scale": True,
        "mu": 0,
        "sigma": 1,
    }
    prior_from_dict = Prior.from_dict(prior_dict)
    assert prior_from_dict.name == "Normal"
    assert prior_from_dict.bounds == (-1.0, 1.0)
    assert prior_from_dict.auto_scale == True
    assert prior_from_dict.dist is not None
    assert prior_from_dict.args == {}
    assert prior_from_dict._args == {"mu": 0, "sigma": 1}
    assert prior_from_dict.is_truncated


def test_from_bambi():
    bambi_prior = bmb.Prior("Normal", auto_scale=True, dist=None, mu=0, sigma=1)
    prior_from_bambi_unbounded = Prior.from_bambi(bambi_prior)

    assert prior_from_bambi_unbounded.name == "Normal"
    assert prior_from_bambi_unbounded.bounds is None
    assert prior_from_bambi_unbounded.auto_scale == True
    assert prior_from_bambi_unbounded.dist is None
    assert prior_from_bambi_unbounded.args["mu"] == 0
    assert prior_from_bambi_unbounded.args["sigma"] == 1
    assert prior_from_bambi_unbounded._args["mu"] == 0
    assert prior_from_bambi_unbounded._args["sigma"] == 1
    assert not prior_from_bambi_unbounded.is_truncated

    prior_from_bambi_bounded = Prior.from_bambi(bambi_prior, bounds=(-1.0, 1.0))
    assert prior_from_bambi_bounded.name == "Normal"
    assert prior_from_bambi_bounded.bounds == (-1.0, 1.0)
    assert prior_from_bambi_bounded.auto_scale == True
    assert prior_from_bambi_bounded.dist is not None
    assert prior_from_bambi_bounded.args == {}
    assert prior_from_bambi_bounded._args["mu"] == 0
    assert prior_from_bambi_bounded._args["sigma"] == 1
    assert prior_from_bambi_bounded.is_truncated

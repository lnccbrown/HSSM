import pytest

import bambi as bmb
import numpy as np

import hssm
from hssm import Prior

hssm.set_floatX("float32")


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

    with pytest.raises(AssertionError):
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

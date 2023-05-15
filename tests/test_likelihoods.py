"""
Unit testing for WFPT likelihood function.

This code compares WFPT likelihood function with
old implementation of WFPT from (https://github.com/hddm-devs/hddm)
"""
import math
import random

import numpy as np
import pytest
import ssms.basic_simulators
from hddm_wfpt import wfpt
from numpy.random import rand

# pylint: disable=C0413
from hssm.wfpt.base import decision_func, log_pdf_sv


@pytest.fixture
def data_fixture():
    random_integer = random.randint(0, 9)
    v_true, a_true, z_true, t_true = [0.5, 1.5, 0.5, 0.5]
    obs_ddm = ssms.basic_simulators.simulator(
        [v_true, a_true, z_true, t_true],
        model="ddm",
        n_samples=1000,
        random_state=random_integer,
    )
    return np.column_stack([obs_ddm["rts"][:, 0], obs_ddm["choices"][:, 0]])


def test_kterm(data_fixture):
    """
    This function defines a range of kterms and tests results to
     makes sure they are not equal to infinity or unknown values
    """
    for k_term in range(7, 12):
        v = (rand() - 0.5) * 1.5
        sv = 0
        a = (1.5 + rand()) / 2
        z = 0.5 * rand()
        t = rand() * 0.5
        err = 1e-7
        logp = log_pdf_sv(data_fixture, v, sv, a, z, t, err, k_terms=k_term)
        logp = sum(logp.eval())
        assert not math.isinf(logp)
        assert not math.isnan(logp)


def test_decision(data_fixture):
    """
    This function tests output of decision function
    """
    decision = decision_func()
    err = 1e-7
    data = data_fixture[:, 0] * data_fixture[:, 1]
    lambda_rt = decision(np.abs(data), err)
    assert all(not v for v in lambda_rt.eval())
    assert data_fixture.shape[0] == lambda_rt.eval().shape[0]


def test_logp(data_fixture):
    """
    This function compares new and old implementation of logp calculation
    """
    for _ in range(10):
        v = (rand() - 0.5) * 1.5
        sv = 0
        a = 1.5 + rand()
        z = 0.5 * rand()
        t = rand() * min(abs(data_fixture[:, 0]))
        err = 1e-7

        # We have to pass a / 2 to ensure that the log-likelihood will return the
        # same value as the cython version.
        pytensor_log = log_pdf_sv(data_fixture, v, sv, a / 2, z, t, err=err)
        data = data_fixture[:, 0] * data_fixture[:, 1]
        cython_log = wfpt.pdf_array(data, v, sv, a, z, 0, t, 0, err, 1)
        np.testing.assert_array_almost_equal(pytensor_log.eval(), cython_log, 0)


@pytest.fixture
def test_params():
    return {
        "v": 1,
        "sv": 0,
        "z": 0.5,
        "t": 0.5,
        "err": 1e-7,
    }


def test_no_inf_values(data_fixture, test_params):
    """
    This test checks if the output does not include inf values even
    when parameters are out of range, when small_number is set to 1e-15.
    """
    for a in np.arange(2.5, 5.1, 0.1):  # a ranges from 2.5 to 5 with step size 0.1
        logp = log_pdf_sv(data_fixture, a=a, small_number=1e-15, **test_params)

        assert np.all(
            np.isfinite(logp.eval())
        ), f"log_pdf_sv() returned non-finite values for a = {a}."


@pytest.fixture
def test_params_with_a():
    return {
        "v": 1,
        "sv": 0,
        "z": 0.5,
        "a": 0.5,
        "err": 1e-7,
    }


def test_no_inf_values_t(data_fixture, test_params_with_a):
    """
    This test checks if the output does not include inf values even
    when parameters are out of range, when small_number is set to 1e-15.
    """
    for t in np.arange(3.0, 5.1, 0.1):  # a ranges from 2.5 to 5 with step size 0.1
        logp = log_pdf_sv(data_fixture, t=t, small_number=1e-15, **test_params_with_a)

        assert np.all(
            np.isfinite(logp.eval())
        ), f"log_pdf_sv() returned non-finite values for t = {t}."

"""
Unit testing for WFPT likelihood function.

This code compares WFPT likelihood function with
old implementation of WFPT from (https://github.com/hddm-devs/hddm)
"""
import math
import os
import sys

import numpy as np
import pytest
import ssms.basic_simulators
from hddm_wfpt import wfpt
from numpy.random import rand

sys.path.insert(0, os.path.abspath("src"))
# pylint: disable=C0413
from hssm.wfpt.base import decision_func, log_pdf_sv


@pytest.fixture
def data_fixture():
    v_true, a_true, z_true, t_true, theta_true = [0.5, 1.5, 0.5, 0.5, 0.3]
    obs_angle = ssms.basic_simulators.simulator(
        [v_true, a_true, z_true, t_true, theta_true], model="angle", n_samples=1000
    )
    return obs_angle["rts"][:, 0] * obs_angle["choices"][:, 0]


def test_kterm(data_fixture):
    """
    This function defines a range of kterms and tests results to
     makes sure they are not equal to infinity or unknown values
    """
    for k_term in range(1, 11):
        v = (rand() - 0.5) * 1.5
        sv = 0
        a = 1.5 + rand()
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
    lambda_rt = decision(data_fixture, err)
    assert all(not v for v in lambda_rt.eval())
    assert len(data_fixture) == len(lambda_rt.eval())


def test_logp(data_fixture):
    """
    This function compares new and old implementation of logp calculation
    """
    for _ in range(20):
        v = (rand() - 0.5) * 1.5
        sv = 0
        a = 1.5 + rand()
        z = 0.5 * rand()
        t = rand() * 0.5
        err = 1e-7
        aesara_log = log_pdf_sv(data_fixture, v, sv, a, z, t, err=err)
        cython_log = wfpt.pdf_array(data_fixture, v, sv, a, z, 0, t, 0, err, 1)
        np.testing.assert_array_almost_equal(aesara_log.eval(), cython_log, 2)
"""Unit testing for WFPT likelihood function.

This code compares WFPT likelihood function with
old implementation of WFPT from (https://github.com/hddm-devs/hddm)
"""
import math

import numpy as np
import pytest
from numpy.random import rand

import hssm

# pylint: disable=C0413
from hssm.likelihoods.analytical import compare_k, logp_ddm, logp_ddm_sdv
from hssm.likelihoods.blackbox import logp_ddm_bbox, logp_ddm_sdv_bbox

hssm.set_floatX("float32")


def test_kterm(data_ddm):
    """This function defines a range of kterms and tests results to
    makes sure they are not equal to infinity or unknown values.
    """
    for k_term in range(7, 12):
        v = (rand() - 0.5) * 1.5
        sv = 0
        a = (1.5 + rand()) / 2
        z = 0.5 * rand()
        t = rand() * 0.5
        err = 1e-7
        logp = logp_ddm_sdv(data_ddm, v, a, z, t, sv, err, k_terms=k_term)
        logp = sum(logp.eval())
        assert not math.isinf(logp)
        assert not math.isnan(logp)


def test_compare_k(data_ddm):
    """This function tests output of decision function."""
    err = 1e-7
    data = data_ddm["rt"] * data_ddm["response"]
    lambda_rt = compare_k(np.abs(data.values), err)
    assert all(not v for v in lambda_rt.eval())
    assert data_ddm.shape[0] == lambda_rt.eval().shape[0]


# def test_logp(data_fixture):
#     """
#     This function compares new and old implementation of logp calculation
#     """
#     for _ in range(10):
#         v = (rand() - 0.5) * 1.5
#         sv = 0
#         a = 1.5 + rand()
#         z = 0.5 * rand()
#         t = rand() * min(abs(data_fixture[:, 0]))
#         err = 1e-7
#
#         # We have to pass a / 2 to ensure that the log-likelihood will return the
#         # same value as the cython version.
#         pytensor_log = log_pdf_sv(data_fixture, v, sv, a / 2, z, t, err=err)
#         data = data_fixture[:, 0] * data_fixture[:, 1]
#         cython_log = wfpt.pdf_array(data, v, sv, a, z, 0, t, 0, err, 1)
#         np.testing.assert_array_almost_equal(pytensor_log.eval(), cython_log, 0)


@pytest.fixture
def shared_params():
    return {
        "v": 1,
        "sv": 0,
        "a": 0.5,
        "z": 0.5,
        "t": 0.5,
        "err": 1e-7,
        "epsilon": 1e-15,
    }


def test_no_inf_values_a(data_ddm, shared_params):
    for a in np.arange(2.5, 5.1, 0.5):
        params = {**shared_params, "a": a}
        logp = logp_ddm_sdv(data_ddm, **params)
        assert np.all(
            np.isfinite(logp.eval())
        ), f"log_pdf_sv() returned non-finite values for a = {a}."


def test_no_inf_values_t(data_ddm, shared_params):
    for t in np.arange(3.0, 5.1, 0.5):
        params = {**shared_params, "t": t}
        logp = logp_ddm_sdv(data_ddm, **params)
        assert np.all(
            np.isfinite(logp.eval())
        ), f"log_pdf_sv() returned non-finite values for t = {t}."


def test_no_inf_values_v(data_ddm, shared_params):
    for v in np.arange(3.0, 5.1, 0.5):
        params = {**shared_params, "v": v}
        logp = logp_ddm_sdv(data_ddm, **params)
        assert np.all(
            np.isfinite(logp.eval())
        ), f"log_pdf_sv() returned non-finite values for v = {v}."


def test_bbox(data_ddm):
    true_values = (0.5, 1.5, 0.5, 0.5)
    true_values_sdv = (0.5, 1.5, 0.5, 0.5, 0)
    data = data_ddm.values

    np.testing.assert_almost_equal(
        logp_ddm(data, *true_values).eval(),
        logp_ddm_bbox(data, *true_values),
        decimal=4,
    )

    np.testing.assert_almost_equal(
        logp_ddm_sdv(data, *true_values_sdv).eval(),
        logp_ddm_sdv_bbox(data, *true_values_sdv),
        decimal=4,
    )

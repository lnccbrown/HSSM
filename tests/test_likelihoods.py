"""Unit testing for WFPT likelihood function.

This code compares WFPT likelihood function with
old implementation of WFPT from (https://github.com/hddm-devs/hddm)
"""

import math
from pathlib import Path
from itertools import product

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
from numpy.random import rand

import hssm

# pylint: disable=C0413
from hssm.likelihoods.analytical import compare_k, logp_ddm, logp_ddm_sdv
from hssm.likelihoods.blackbox import logp_ddm_bbox, logp_ddm_sdv_bbox
from hssm.distribution_utils import make_likelihood_callable

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


cav_data = hssm.load_data("cavanagh_theta")
param_matrix = product(
    (0.0, 0.01, 0.05, 0.5), ("analytical", "approx_differentiable", "blackbox")
)


@pytest.mark.parametrize("p_outlier, loglik_kind", param_matrix)
def test_lapse_distribution_cav(p_outlier, loglik_kind):
    true_values = (1.5, 0.5, 0.5)
    a, z, t = true_values
    # v is set to vector, because
    # as parent it will be a regression by default
    # 'approx_differentiable' likelihoods need to take
    # into account.
    v = 0.5 * np.ones(cav_data.shape[0])

    model = hssm.HSSM(
        model="ddm",
        data=cav_data,
        p_outlier=p_outlier,
        loglik_kind=loglik_kind,
        loglik=(
            Path(__file__).parent / "fixtures" / "ddm.onnx"
            if loglik_kind == "approx_differentiable"
            else None
        ),
        prior_settings=None,  # Avoid unnecessary computation
        lapse=(
            None if p_outlier == 0.0 else hssm.Prior("Uniform", lower=0.0, upper=10.0)
        ),
    )

    #
    distribution = (
        model.model_distribution.dist(v=v, a=a, z=z, t=t, p_outlier=p_outlier)
        if p_outlier > 0
        else model.model_distribution.dist(v=v, a=a, z=z, t=t)
    )

    # We could do this outside of the function,
    # but for some reason mypy complaints with:
    # error: Item "str" of "Any | str" has no attribute "values"
    # while here it allows it.
    cav_data_numpy = cav_data[["rt", "response"]].values

    # Convert to float32 if blackbox loglik is used
    # This is necessary because the blackbox likelihood function logp_ddm_bbox is
    # does not go through any PyTensor function standalone so does not respect the
    # floatX setting

    # This step is not necessary for HSSM as a whole because the likelihood function
    # will be part of a PyTensor graph so the floatX setting will be respected
    cav_data_inp = (
        cav_data_numpy.astype("float32")
        if loglik_kind == "blackbox"
        else cav_data_numpy
    )

    model_logp = pm.logp(distribution, cav_data_inp).eval()

    if loglik_kind == "analytical":
        logp_func = logp_ddm
    elif loglik_kind == "approx_differentiable":
        logp_func = make_likelihood_callable(
            loglik=Path(__file__).parent / "fixtures" / "ddm.onnx",
            loglik_kind="approx_differentiable",
            backend="pytensor",
            params_is_reg=[False] * 4,
        )
    else:
        logp_func = logp_ddm_bbox

    manual_logp = logp_func(cav_data_inp, *(v, a, z, t))
    if p_outlier == 0.0:
        manual_logp = pt.where(
            pt.sub(cav_data_inp[:, 0], t) <= 1e-15, -66.1, manual_logp
        ).eval()
        np.testing.assert_almost_equal(model_logp, manual_logp, decimal=4)
        return

    manual_logp = pt.where(
        pt.sub(cav_data_inp[:, 0], t) <= 1e-15,
        -66.1,
        manual_logp,
    )
    manual_logp = pt.log(
        (1 - p_outlier) * pt.exp(manual_logp)
        + p_outlier
        * pt.exp(pm.logp(pm.Uniform.dist(lower=0.0, upper=10.0), cav_data_inp[:, 0]))
    ).eval()

    np.testing.assert_almost_equal(model_logp, manual_logp, decimal=4)

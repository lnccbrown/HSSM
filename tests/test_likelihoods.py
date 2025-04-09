"""Unit testing for WFPT likelihood function.

This code compares WFPT likelihood function with
old implementation of WFPT from (https://github.com/hddm-devs/hddm)
"""

from pathlib import Path
from itertools import product
from hssm.utils import SuppressOutput

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.compile.nanguardmode import NanGuardMode

import hssm

# pylint: disable=C0413
from hssm.likelihoods.analytical import logp_ddm, logp_ddm_sdv
from hssm.likelihoods.blackbox import logp_ddm_bbox, logp_ddm_sdv_bbox
from hssm.distribution_utils import make_likelihood_callable

hssm.set_floatX("float32")

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


names = ["a", "t", "v"]
values = [2.5, 3.0, 3.0]
parameters = [(name, np.arange(value, 5.1, 0.5)) for name, value in zip(names, values)]


@pytest.mark.slow
@pytest.mark.parametrize("param_name, param_values", parameters)
def test_no_inf_values(data_ddm, shared_params, param_name, param_values):
    for value in param_values:
        params = shared_params | {param_name: value}
        logp = logp_ddm_sdv(data_ddm, **params).eval()
        assert logp.ndim == 1, "logp_ddm_sdv() returned wrong number of dimensions."
        assert np.all(np.isfinite(logp)), (
            f"log_pdf_sv() returned non-finite values for {param_name} = {value}."
        )


true_values = (0.5, 1.5, 0.5, 0.5)
true_values_sdv = true_values + (0,)
standard = (logp_ddm, logp_ddm_bbox, true_values)
sdv = (logp_ddm_sdv, logp_ddm_sdv_bbox, true_values_sdv)
parameters = [standard, sdv]  # type: ignore


@pytest.mark.slow
@pytest.mark.parametrize("logp_func, logp_bbox_func, true_values", parameters)
def test_bbox(data_ddm, logp_func, logp_bbox_func, true_values):
    data = data_ddm.values

    np.testing.assert_almost_equal(
        logp_func(data, *true_values).eval(),
        logp_bbox_func(data, *true_values),
        decimal=4,
    )


cav_data: pd.DataFrame = hssm.load_data("cavanagh_theta")
cav_data_numpy = cav_data[["rt", "response"]].values
param_matrix = product(
    (0.0, 0.01, 0.05, 0.5), ("analytical", "approx_differentiable", "blackbox")
)
nan_guard_mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False)


@pytest.mark.slow
def test_analytical_gradient():
    v = pt.dvector()
    a = pt.dvector()
    z = pt.dvector()
    t = pt.dvector()
    sv = pt.dvector()
    size = cav_data_numpy.shape[0]
    logp = logp_ddm(cav_data_numpy, v, a, z, t).sum()
    grad = pt.grad(logp, wrt=[v, a, z, t])

    # Temporary measure to suppress output from pytensor.function
    # See issues #594 in hssm and #1037 in pymc-devs/pytensor repos
    with SuppressOutput():
        grad_func = pytensor.function(
            [v, a, z, t],
            grad,
            mode=nan_guard_mode,
        )
    v_test = np.random.normal(size=size)
    a_test = np.random.uniform(0.0001, 2, size=size)
    z_test = np.random.uniform(0.1, 1.0, size=size)
    t_test = np.random.uniform(0, 2, size=size)
    sv_test = np.random.uniform(0.001, 1.0, size=size)
    grad = np.array(grad_func(v_test, a_test, z_test, t_test))

    assert np.all(np.isfinite(grad), axis=None), "Gradient contains non-finite values."

    # Also temporary
    with SuppressOutput():
        grad_func_sdv = pytensor.function(
            [v, a, z, t, sv],
            pt.grad(
                logp_ddm_sdv(cav_data_numpy, v, a, z, t, sv).sum(), wrt=[v, a, z, t, sv]
            ),
            mode=nan_guard_mode,
        )

    grad_sdv = np.array(grad_func_sdv(v_test, a_test, z_test, t_test, sv_test))

    assert np.all(np.isfinite(grad_sdv), axis=None), (
        "Gradient contains non-finite values."
    )


@pytest.mark.slow
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

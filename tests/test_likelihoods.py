"""Unit testing for WFPT likelihood function.

This code compares WFPT likelihood function with
old implementation of WFPT from (https://github.com/hddm-devs/hddm)
"""

from pathlib import Path
from itertools import product
from hssm.utils import SuppressOutput

import jax
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.compile.debug.nanguardmode import NanGuardMode

import hssm

# pylint: disable=C0413
from hssm.likelihoods.analytical import (
    logp_ddm,
    logp_ddm_sdv,
    logp_lba2,
    logp_lba3,
    logp_lba4,
)
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


true_values = (0.5, 1.5, 0.5, 0.5)
true_values_sdv = true_values + (0,)
standard = (logp_ddm, logp_ddm_bbox, true_values)
sdv = (logp_ddm_sdv, logp_ddm_sdv_bbox, true_values_sdv)
parameters = [standard, sdv]  # type: ignore

CLOSE_TOLERANCE = 1e-4


class TestLbaLikelihood:
    theta_lba2 = dict(A=0.2, b=0.5, v0=1.0, v1=1.0)
    theta_lba3 = theta_lba2 | {"v2": 1.0}
    theta_lba4 = theta_lba3 | {"v3": 1.0}
    lba_parameters = [
        (logp_lba2, theta_lba2, 2),
        (logp_lba3, theta_lba3, 3),
    ]

    @staticmethod
    def _filter_theta(theta, exclude_keys=("A", "b")):
        return {k: v for k, v in theta.items() if k not in exclude_keys}

    @staticmethod
    def _vectorize_param(theta, param, size):
        return {k: (np.full(size, v) if k == param else v) for k, v in theta.items()}

    @staticmethod
    def _make_lba_data(n_choices):
        rows = [[0.35, choice] for choice in range(n_choices)] + [
            [0.5, choice] for choice in range(n_choices)
        ]
        return np.array(rows, dtype=np.float32)

    @pytest.mark.parametrize(
        "logp_func, theta, n_choices",
        lba_parameters,
    )
    def test_lba_synthetic_data(self, logp_func, theta, n_choices):
        lba_data = self._make_lba_data(n_choices)
        size = lba_data.shape[0]

        out_base = logp_func(lba_data, **theta).eval()
        assert out_base.shape == (size,)
        assert np.isfinite(out_base).all()

        for param in theta:
            param_vec = self._vectorize_param(theta, param, size)
            out_vec = logp_func(lba_data, **param_vec).eval()
            assert np.allclose(out_vec, out_base, atol=CLOSE_TOLERANCE)

        for A, b in product([np.full(size, 0.6), 0.6], [np.full(size, 0.5), 0.5]):
            with pytest.raises(pm.logprob.utils.ParameterValueError):
                logp_func(
                    lba_data,
                    A=A,
                    b=b,
                    **self._filter_theta(theta, ("A", "b")),
                ).eval()

    def test_lba4_jax_likelihood_supports_jit_vectors_and_gradients(self):
        """LBA4 should be a native JAX likelihood with differentiable parameters."""
        lba_data = self._make_lba_data(4)
        expected = np.array(
            [
                0.65798534,
                0.65798534,
                0.65798534,
                0.65798534,
                -6.08265287,
                -6.08265287,
                -6.08265287,
                -6.08265287,
            ],
            dtype=np.float32,
        )

        output = np.asarray(jax.jit(logp_lba4)(lba_data, **self.theta_lba4))
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)

        for param in self.theta_lba4:
            vectorized = np.asarray(
                logp_lba4(
                    lba_data,
                    **self._vectorize_param(self.theta_lba4, param, len(lba_data)),
                )
            )
            np.testing.assert_allclose(vectorized, expected, rtol=1e-5, atol=1e-5)

            def summed_logp(value):
                params = self.theta_lba4 | {param: value}
                return logp_lba4(lba_data, **params).sum()

            gradient = np.asarray(jax.grad(summed_logp)(self.theta_lba4[param]))
            assert np.isfinite(gradient).all()

        invalid = np.asarray(
            logp_lba4(
                lba_data,
                A=0.6,
                b=0.5,
                v0=1.0,
                v1=1.0,
                v2=1.0,
                v3=1.0,
            )
        )
        assert np.isneginf(invalid).all()


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

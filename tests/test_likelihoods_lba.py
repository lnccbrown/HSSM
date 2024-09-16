"""Unit testing for LBA likelihood functions."""

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import arviz as az
from pytensor.compile.nanguardmode import NanGuardMode

import hssm

# pylint: disable=C0413
from hssm.likelihoods.analytical import logp_lba2, logp_lba3
from hssm.likelihoods.blackbox import logp_ddm_bbox, logp_ddm_sdv_bbox
from hssm.distribution_utils import make_likelihood_callable

hssm.set_floatX("float32")

CLOSE_TOLERANCE = 1e-4


def test_lba2_basic():
    size = 1000

    lba_data_out = hssm.simulate_data(
        model="lba2", theta=dict(A=0.2, b=0.5, v0=1.0, v1=1.0), size=size
    )

    # Test if vectorization ok across parameters
    out_A_vec = logp_lba2(
        lba_data_out.values, A=np.array([0.2] * size), b=0.5, v0=1.0, v1=1.0
    ).eval()
    out_base = logp_lba2(lba_data_out.values, A=0.2, b=0.5, v0=1.0, v1=1.0).eval()
    assert np.allclose(out_A_vec, out_base, atol=CLOSE_TOLERANCE)

    out_b_vec = logp_lba2(
        lba_data_out.values,
        A=np.array([0.2] * size),
        b=np.array([0.5] * size),
        v0=1.0,
        v1=1.0,
    ).eval()
    assert np.allclose(out_b_vec, out_base, atol=CLOSE_TOLERANCE)

    out_v_vec = logp_lba2(
        lba_data_out.values,
        A=np.array([0.2] * size),
        b=np.array([0.5] * size),
        v0=np.array([1.0] * size),
        v1=np.array([1.0] * size),
    ).eval()
    assert np.allclose(out_v_vec, out_base, atol=CLOSE_TOLERANCE)

    # Test A > b leads to error
    with pytest.raises(pm.logprob.utils.ParameterValueError):
        logp_lba2(
            lba_data_out.values, A=np.array([0.6] * 1000), b=0.5, v0=1.0, v1=1.0
        ).eval()

    with pytest.raises(pm.logprob.utils.ParameterValueError):
        logp_lba2(lba_data_out.values, A=0.6, b=0.5, v0=1.0, v1=1.0).eval()

    with pytest.raises(pm.logprob.utils.ParameterValueError):
        logp_lba2(
            lba_data_out.values, A=0.6, b=np.array([0.5] * 1000), v0=1.0, v1=1.0
        ).eval()

    with pytest.raises(pm.logprob.utils.ParameterValueError):
        logp_lba2(
            lba_data_out.values,
            A=np.array([0.6] * 1000),
            b=np.array([0.5] * 1000),
            v0=1.0,
            v1=1.0,
        ).eval()


def test_lba3_basic():
    size = 1000

    lba_data_out = hssm.simulate_data(
        model="lba3", theta=dict(A=0.2, b=0.5, v0=1.0, v1=1.0, v2=1.0), size=size
    )

    # Test if vectorization ok across parameters
    out_A_vec = logp_lba3(
        lba_data_out.values, A=np.array([0.2] * size), b=0.5, v0=1.0, v1=1.0, v2=1.0
    ).eval()

    out_base = logp_lba3(
        lba_data_out.values, A=0.2, b=0.5, v0=1.0, v1=1.0, v2=1.0
    ).eval()

    assert np.allclose(out_A_vec, out_base, atol=CLOSE_TOLERANCE)

    out_b_vec = logp_lba3(
        lba_data_out.values,
        A=np.array([0.2] * size),
        b=np.array([0.5] * size),
        v0=1.0,
        v1=1.0,
        v2=1.0,
    ).eval()
    assert np.allclose(out_b_vec, out_base, atol=CLOSE_TOLERANCE)

    out_v_vec = logp_lba3(
        lba_data_out.values,
        A=np.array([0.2] * size),
        b=np.array([0.5] * size),
        v0=np.array([1.0] * size),
        v1=np.array([1.0] * size),
        v2=np.array([1.0] * size),
    ).eval()
    assert np.allclose(out_v_vec, out_base, atol=CLOSE_TOLERANCE)

    # Test A > b leads to error
    with pytest.raises(pm.logprob.utils.ParameterValueError):
        logp_lba3(
            lba_data_out.values, A=np.array([0.6] * 1000), b=0.5, v0=1.0, v1=1.0, v2=1.0
        ).eval()

    with pytest.raises(pm.logprob.utils.ParameterValueError):
        logp_lba3(lba_data_out.values, b=0.5, A=0.6, v0=1.0, v1=1.0, v2=1.0).eval()

    with pytest.raises(pm.logprob.utils.ParameterValueError):
        logp_lba3(
            lba_data_out.values, A=0.6, b=np.array([0.5] * 1000), v0=1.0, v1=1.0, v2=1.0
        ).eval()

    with pytest.raises(pm.logprob.utils.ParameterValueError):
        logp_lba3(
            lba_data_out.values,
            A=np.array([0.6] * 1000),
            b=np.array([0.5] * 1000),
            v0=1.0,
            v1=1.0,
            v2=1.0,
        ).eval()


test_lba3_basic()

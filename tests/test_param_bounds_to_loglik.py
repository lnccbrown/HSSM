from pathlib import Path

import numpy as np
import pytensor.tensor as pt
import pytest
import ssms

from hssm.wfpt import lan
from hssm.wfpt.base import log_pdf_sv
from hssm.wfpt.config import default_model_config
from hssm.wfpt.wfpt import apply_param_bounds_to_loglik


@pytest.fixture
def data():
    v_true, a_true, z_true, t_true = [0.5, 1.5, 0.5, 0.5]
    obs_ddm = ssms.basic_simulators.simulator(
        [v_true, a_true, z_true, t_true], model="ddm", n_samples=1000
    )
    return np.column_stack([obs_ddm["rts"][:, 0], obs_ddm["choices"][:, 0]])


@pytest.fixture(scope="module")
def fixture_path():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def data_angle():
    v_true, a_true, z_true, t_true, theta_true = [0.5, 1.5, 0.5, 0.5, 0.3]
    obs_angle = ssms.basic_simulators.simulator(
        [v_true, a_true, z_true, t_true, theta_true], model="angle", n_samples=1000
    )
    return np.column_stack([obs_angle["rts"][:, 0], obs_angle["choices"][:, 0]])


@pytest.mark.parametrize("a", [0.5, -4.0])
def test_adjust_logp_with_analytical(data, a):
    v = 1
    sv = 0
    z = 0.5
    t = 0.5
    err = 1e-7
    logp = log_pdf_sv(data, v, sv, a, z, t, err, k_terms=7)
    adjusted_logp = apply_param_bounds_to_loglik(
        logp,
        ["v", "sv", "a", "z", "t"],
        v,
        sv,
        a,
        z,
        t,
        err,
        default_boundaries=default_model_config["ddm"]["default_boundaries"],
    )
    assert pt.all(pt.eq(adjusted_logp, logp)).eval() == True
    assert (
        pt.all(pt.eq(adjusted_logp, pt.full_like(adjusted_logp, -66.1))).eval() == True
    )


@pytest.mark.parametrize("theta", [0.5, -4.0])
def test_adjust_logp_with_angle(data_angle, fixture_path, theta):
    v = 1
    a = 0.5
    z = 0.5
    t = 0.5

    logp_pytensor = lan.make_pytensor_logp(fixture_path / "test.onnx")
    logp_angle = logp_pytensor(data_angle, v, a, z, t, theta)

    adjusted_logp = apply_param_bounds_to_loglik(
        logp_angle,
        ["v", "a", "z", "t", "theta"],
        v,
        a,
        z,
        t,
        theta,
        default_boundaries=default_model_config["angle"]["default_boundaries"],
    )
    assert pt.all(pt.eq(adjusted_logp, logp_angle)).eval() == True
    assert (
        pt.all(pt.eq(adjusted_logp, pt.full_like(adjusted_logp, -66.1))).eval() == True
    )

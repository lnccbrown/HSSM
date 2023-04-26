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


vector_length = 1000
lower_bound = 0.3
upper_bound = 2.5
vector_a = np.random.rand(vector_length) * (upper_bound - lower_bound) + lower_bound


@pytest.mark.parametrize(
    "a, expected_result",
    [
        (0.5, "equal"),
        (0.1, "not_equal"),
        (vector_a, "equal"),
        (np.random.rand(1000), "not_equal_no_inf"),
    ],
)
def test_adjust_logp_with_analytical(
    data,
    a,
    expected_result,
):
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
        bounds=default_model_config["ddm"]["default_boundaries"],
    )
    if expected_result == "equal":
        assert pt.all(pt.eq(adjusted_logp, logp)).eval()
    elif expected_result == "not_equal":
        assert pt.all(pt.eq(adjusted_logp, -66.1)).eval()
    elif expected_result == "not_equal_no_inf":
        assert not pt.all(pt.eq(adjusted_logp, logp)).eval() and not np.any(
            np.isinf(adjusted_logp.eval())
        )


vector_length = 1000
lower_bound = -0.1
upper_bound = 1.3
vector_theta = np.random.rand(vector_length) * (upper_bound - lower_bound) + lower_bound


@pytest.mark.parametrize(
    "theta, expected_result",
    [
        (0.5, "equal"),
        (-4.0, "not_equal"),
        (vector_theta, "equal"),
        (np.random.rand(1000), "not_equal_no_inf"),
    ],
)
def test_adjust_logp_with_angle(data_angle, fixture_path, theta, expected_result):
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
        bounds=default_model_config["angle"]["default_boundaries"],
    )

    if expected_result == "equal":
        assert pt.all(pt.eq(adjusted_logp, logp_angle)).eval()
    elif expected_result == "not_equal":
        assert pt.all(pt.eq(adjusted_logp, -66.1)).eval()
    elif expected_result == "not_equal_no_inf":
        assert not pt.all(pt.eq(adjusted_logp, logp_angle)).eval() and not np.any(
            np.isinf(adjusted_logp.eval())
        )

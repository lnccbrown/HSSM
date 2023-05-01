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


def generate_random_vector(vector_length, lower_bound, upper_bound):
    return np.random.rand(vector_length) * (upper_bound - lower_bound) + lower_bound


def generate_samples_outside_boundary(samples_needed, boundary):
    samples = []

    while len(samples) < samples_needed:
        random_sample = np.random.uniform(
            -10, 10
        )  # Generate a random number between -10 and 10
        if random_sample < boundary[0] or random_sample > boundary[1]:
            samples.append(random_sample)

    outside_boundary_array = np.array(samples)
    return outside_boundary_array


vector_length = 1000
vector_a = generate_random_vector(vector_length, 0.3, 2.5)
vector_a_2 = generate_samples_outside_boundary(vector_length, (0.3, 3.0))


@pytest.mark.parametrize(
    "a, expected_result",
    [
        (0.5, "equal"),
        (0.0, "not_equal"),
        (vector_a, "equal"),
        (vector_a_2, "not_equal_no_inf"),
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
        bounds=default_model_config["ddm"]["bounds"],
    )
    if expected_result == "equal":
        assert pt.allclose(adjusted_logp, logp, atol=1e-5).eval()
    elif expected_result == "not_equal":
        assert pt.all(pt.eq(adjusted_logp, -66.1)).eval()
    elif expected_result == "not_equal_no_inf":
        assert not pt.all(pt.eq(adjusted_logp, logp)).eval()


vector_theta = generate_random_vector(vector_length, -0.1, 1.3)
vector_theta_2 = generate_samples_outside_boundary(vector_length, (-0.1, 1.3))


@pytest.mark.parametrize(
    "theta, expected_result",
    [
        (0.5, "equal"),
        (-4.0, "not_equal"),
        (vector_theta, "equal"),
        (vector_theta_2, "not_equal_no_inf"),
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
        bounds=default_model_config["angle"]["bounds"],
    )

    if expected_result == "equal":
        assert pt.allclose(adjusted_logp, logp_angle, atol=1e-5).eval()
    elif expected_result == "not_equal":
        assert pt.all(pt.eq(adjusted_logp, -66.1)).eval()
    elif expected_result == "not_equal_no_inf":
        assert not pt.all(pt.eq(adjusted_logp, logp_angle)).eval() and not np.any(
            np.isinf(adjusted_logp.eval())
        )

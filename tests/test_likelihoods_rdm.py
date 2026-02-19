"""Unit testing for Racing Diffusion likelihood functions."""

from itertools import product

import numpy as np
import pymc as pm
import pytest

import hssm

# pylint: disable=C0413
from hssm.likelihoods.analytical import logp_rdm3

hssm.set_floatX("float32")

CLOSE_TOLERANCE = 1e-4


def filter_theta(theta, exclude_keys=["A", "b"]):
    """Filter out specific keys from the theta dictionary."""
    return {k: v for k, v in theta.items() if k not in exclude_keys}


def assert_parameter_value_error(logp_func, data_out, A, b, theta):
    """Helper function to assert ParameterValueError for given parameters."""
    with pytest.raises(pm.logprob.utils.ParameterValueError):
        logp_func(
            data_out.values,
            A=A,
            b=b,
            **filter_theta(theta, ["A", "b"]),
        )


def vectorize_param(theta, param, size):
    """
    Vectorize a specific parameter in the theta dictionary.

    Parameters:
    theta (dict): Dictionary of parameters.
    param (str): The parameter to vectorize.
    size (int): The size of the vector.

    Returns:
    dict: A new dictionary with the specified parameter vectorized.
    """
    return {k: (np.full(size, v) if k == param else v) for k, v in theta.items()}


# Parameters for racing diffusion: drifts (v0, v1, v2), boundary (b), starting point (A), non-decision time (t)
theta_rd3 = dict(v0=1.0, v1=1.2, v2=1.4, b=2.0, A=1.0, t=0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "logp_func, model, theta",
    [
        (logp_rdm3, "racing_diffusion_3", theta_rd3),
    ],
)
def test_racing_diffusion(logp_func, model, theta):
    size = 1000
    data_out = hssm.simulate_data(model=model, theta=theta, size=size)

    # Test if vectorization is ok across parameters
    for param in theta:
        param_vec = vectorize_param(theta, param, size)
        out_vec = logp_func(data_out.values, **param_vec)
        out_base = logp_func(data_out.values, **theta)
        assert np.allclose(out_vec, out_base, atol=CLOSE_TOLERANCE)


    # Test A > b leads to -inf
    A_values = [np.full(size, 0.6), 0.6]
    b_values = [np.full(size, 0.5), 0.5]

    for A, b in product(A_values, b_values):
        # We wrap in try-except because depending on PyMC version or context,
        # it might raise ParameterValueError or return -inf.
        try:
            res = logp_func(
                data_out.values,
                A=A,
                b=b,
                **filter_theta(theta, ["A", "b"]),
            )
            # If no error is raised, we expect -inf
            assert np.isneginf(res).all()
        except pm.logprob.utils.ParameterValueError:
            pass
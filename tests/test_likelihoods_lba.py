"""Unit testing for LBA likelihood functions."""

from itertools import product

import jax
import numpy as np
import pymc as pm
import pytest

import hssm

# pylint: disable=C0413
from hssm.likelihoods.analytical import logp_lba2, logp_lba3, logp_lba4

hssm.set_floatX("float32")

CLOSE_TOLERANCE = 1e-4


def filter_theta(theta, exclude_keys=["A", "b"]):
    """Filter out specific keys from the theta dictionary."""
    return {k: v for k, v in theta.items() if k not in exclude_keys}


def assert_parameter_value_error(logp_func, lba_data_out, A, b, theta):
    """Helper function to assert ParameterValueError for given parameters."""
    with pytest.raises(pm.logprob.utils.ParameterValueError):
        logp_func(
            lba_data_out.values,
            A=A,
            b=b,
            **filter_theta(theta, ["A", "b"]),
        ).eval()


def vectorize_param(theta, param, size):
    """
    Vectorize a specific parameter in the theta dictionary.

    Parameters:
    theta (dict): Dictionary of parameters.
    param (str): The parameter to vectorize.
    size (int): The size of the vector.

    Returns:
    dict: A new dictionary with the specified parameter vectorized.

    Examples:
    >>> theta = {"A": 0.2, "b": 0.5, "v0": 1.0, "v1": 1.0}
    >>> vectorize_param(theta, "A", 3)
    {'A': array([0.2, 0.2, 0.2]), 'b': 0.5, 'v0': 1.0, 'v1': 1.0}

    >>> vectorize_param(theta, "v0", 2)
    {'A': 0.2, 'b': 0.5, 'v0': array([1., 1.]), 'v1': 1.0}
    """
    return {k: (np.full(size, v) if k == param else v) for k, v in theta.items()}


theta_lba2 = dict(A=0.2, b=0.5, v0=1.0, v1=1.0)
theta_lba3 = theta_lba2 | {"v2": 1.0}
theta_lba4 = theta_lba3 | {"v3": 1.0}


def make_lba_data(n_choices):
    """Make deterministic LBA-like data with all choices represented."""
    rows = [[0.35, choice] for choice in range(n_choices)] + [
        [0.5, choice] for choice in range(n_choices)
    ]
    return np.array(rows, dtype=np.float32)


@pytest.mark.parametrize(
    "logp_func, theta, n_choices",
    [
        (logp_lba2, theta_lba2, 2),
        (logp_lba3, theta_lba3, 3),
    ],
)
def test_lba_synthetic_data(logp_func, theta, n_choices):
    lba_data = make_lba_data(n_choices)
    size = lba_data.shape[0]

    out_base = logp_func(lba_data, **theta).eval()
    assert out_base.shape == (size,)
    assert np.isfinite(out_base).all()

    for param in theta:
        param_vec = vectorize_param(theta, param, size)
        out_vec = logp_func(lba_data, **param_vec).eval()
        assert np.allclose(out_vec, out_base, atol=CLOSE_TOLERANCE)

    for A, b in product([np.full(size, 0.6), 0.6], [np.full(size, 0.5), 0.5]):
        with pytest.raises(pm.logprob.utils.ParameterValueError):
            logp_func(lba_data, A=A, b=b, **filter_theta(theta, ["A", "b"])).eval()


def test_lba4_jax_likelihood_supports_jit_vectors_and_gradients():
    """LBA4 should be a native JAX likelihood with differentiable parameters."""
    lba_data = make_lba_data(4)
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

    output = np.asarray(jax.jit(logp_lba4)(lba_data, **theta_lba4))
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)

    for param in theta_lba4:
        vectorized = np.asarray(
            logp_lba4(lba_data, **vectorize_param(theta_lba4, param, len(lba_data)))
        )
        np.testing.assert_allclose(vectorized, expected, rtol=1e-5, atol=1e-5)

        def summed_logp(value):
            params = theta_lba4 | {param: value}
            return logp_lba4(lba_data, **params).sum()

        gradient = np.asarray(jax.grad(summed_logp)(theta_lba4[param]))
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
@pytest.mark.parametrize(
    "logp_func, model, theta",
    [(logp_lba2, "lba2", theta_lba2), (logp_lba3, "lba3", theta_lba3)],
)
def test_lba(logp_func, model, theta):
    size = 1000
    lba_data_out = hssm.simulate_data(model=model, theta=theta, size=size)

    # Test if vectorization is ok across parameters
    for param in theta:
        param_vec = vectorize_param(theta, param, size)
        out_vec = logp_func(lba_data_out.values, **param_vec).eval()
        out_base = logp_func(lba_data_out.values, **theta).eval()
        assert np.allclose(out_vec, out_base, atol=CLOSE_TOLERANCE)

    # Test A > b leads to error
    A_values = [np.full(size, 0.6), 0.6]
    b_values = [np.full(size, 0.5), 0.5]

    for A, b in product(A_values, b_values):
        assert_parameter_value_error(logp_func, lba_data_out, A, b, theta)

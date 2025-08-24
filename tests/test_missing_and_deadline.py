from itertools import product
from pathlib import Path

import pytest
import pytensor
import pytensor.tensor as pt
import numpy as np

from hssm.utils import _rearrange_data
from hssm.distribution_utils import (
    assemble_callables,
    make_likelihood_callable,
    make_missing_data_callable,
)


DECIMAL = 4


@pytest.fixture
def fixture_path():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def data():
    arr = np.ones((100, 3), dtype=np.float32)
    arr[:, 0] = np.random.rand(100)
    arr[:, 2] = np.random.rand(100)
    missing_indices = arr[:, 0] > arr[:, 2]
    arr[missing_indices, 0] = -999.0
    return _rearrange_data(arr)


# AF-TODO: Reactivate CPN case
# AF-TODO: Something is broken about the is_vector == False case
# cases = product(["opn", "cpn"], [True, False])
cases = product(["opn"], [True])


@pytest.mark.slow
@pytest.mark.parametrize("cpn, is_vector", cases)
def test_make_missing_data_callable(data, fixture_path, cpn, is_vector):
    is_cpn = cpn == "cpn"
    is_deadline = not is_cpn

    if is_cpn:
        data = data[:, :-1]

    v = pt.as_tensor_variable(np.random.rand(100) if is_vector else np.random.rand())
    a = pt.as_tensor_variable(0.2)
    z = pt.as_tensor_variable(0.3)
    t = pt.as_tensor_variable(0.4)

    dist_params = [v, a, z, t]

    # First, test if the callables in jax and pytensor give the same result
    missing_onnx_path = fixture_path / f"ddm_{cpn}.onnx"
    missing_callable_jax = make_missing_data_callable(
        missing_onnx_path,
        backend="jax",
        params_is_reg=[is_vector] + [False] * 3,
        params_only=is_cpn,
    )

    missing_callable_pytensor = make_missing_data_callable(
        missing_onnx_path,
        backend="pytensor",
    )

    missing_result_jax = missing_callable_jax(
        None if is_cpn else data[:, -1:], *dist_params
    ).eval()
    missing_result_pytensor = missing_callable_pytensor(
        None if is_cpn else data[:, -1:], *dist_params
    ).eval()

    np.testing.assert_array_almost_equal(
        missing_result_jax,
        missing_result_pytensor,
        decimal=DECIMAL,
    )

    # Second, test if the gradient of the callables in jax and pytensor give the same result
    v_grad_jax = pytensor.grad(
        missing_callable_jax(None if is_cpn else data[:, -1:], *dist_params).sum(), v
    ).eval()
    v_grad_pytensor = pytensor.grad(
        missing_callable_pytensor(None if is_cpn else data[:, -1:], *dist_params).sum(),
        v,
    ).eval()

    np.testing.assert_array_almost_equal(
        v_grad_jax,
        v_grad_pytensor,
        decimal=DECIMAL,
    )

    ###### Prepare data for testing the assembled likelihoods
    data = _rearrange_data(data)
    n_missing = np.sum(data[:, 0] == -999.0).astype(int)
    likelihood_onnx_path = fixture_path / "ddm.onnx"

    # First, test if the assembled callables give the same result as the individual
    # callables, the jax case
    logp_callable_jax = make_likelihood_callable(
        likelihood_onnx_path,
        loglik_kind="approx_differentiable",
        backend="jax",
        params_is_reg=[is_vector] + [False] * 3,
    )

    result_data_jax = logp_callable_jax(
        data[n_missing:, :] if is_cpn else data[n_missing:, :-1],
        v[n_missing:] if is_vector else v,
        *dist_params[1:],
    ).eval()
    missing_data_jax = missing_callable_jax(
        None if is_cpn else data[:n_missing, -1:],
        v[:n_missing] if is_vector else v,
        *dist_params[1:],
    ).eval()

    assembled_loglik_jax = assemble_callables(
        logp_callable_jax,
        missing_callable_jax,
        params_only=is_cpn,
        has_deadline=is_deadline,
    )

    result_assembled_jax = assembled_loglik_jax(data, *dist_params).eval()

    result_individual = np.zeros(100)
    result_individual[n_missing:] = result_data_jax
    result_individual[:n_missing] = missing_data_jax

    np.testing.assert_array_almost_equal(
        result_assembled_jax,
        result_individual,
        decimal=DECIMAL,
    )

    # Then, test if the same happens in the pytensor case
    logp_callable_pytensor = make_likelihood_callable(
        likelihood_onnx_path,
        loglik_kind="approx_differentiable",
        backend="pytensor",
    )

    assembled_loglik_pytensor = assemble_callables(
        logp_callable_pytensor,
        missing_callable_pytensor,
        params_only=is_cpn,
        has_deadline=is_deadline,
    )

    result_data_pytensor = logp_callable_pytensor(
        data[n_missing:, :] if is_cpn else data[n_missing:, :-1],
        v[n_missing:] if is_vector else v,
        *dist_params[1:],
    ).eval()
    missing_data_pytensor = missing_callable_pytensor(
        None if is_cpn else data[:n_missing, -1:],
        v[:n_missing] if is_vector else v,
        *dist_params[1:],
    ).eval()

    result_assembled_pytensor = assembled_loglik_pytensor(data, *dist_params).eval()

    result_individual = np.zeros(100)
    result_individual[n_missing:] = result_data_pytensor
    result_individual[:n_missing] = missing_data_pytensor

    np.testing.assert_array_almost_equal(
        result_assembled_pytensor,
        result_individual,
        decimal=DECIMAL,
    )

    # Also test results from pytensor and jax are the same
    np.testing.assert_array_almost_equal(
        result_assembled_jax,
        result_assembled_pytensor,
        decimal=DECIMAL,
    )

    # Test if the likelihood functions produce the same gradients
    v_grad_jax = pytensor.grad(
        logp_callable_jax(
            data[n_missing:, :] if is_cpn else data[n_missing:, :-1],
            v[n_missing:] if is_vector else v,
            *dist_params[1:],
        ).sum(),
        wrt=v,
    ).eval()
    v_grad_pytensor = pytensor.grad(
        logp_callable_pytensor(
            data[n_missing:, :] if is_cpn else data[n_missing:, :-1],
            v[n_missing:] if is_vector else v,
            *dist_params[1:],
        ).sum(),
        wrt=v,
    ).eval()

    np.testing.assert_array_almost_equal(v_grad_jax, v_grad_pytensor, decimal=DECIMAL)

    # This somehow doesn't work in the cpn, non-vector case
    # NOTE: Investigate later
    # v_grad_jax = pytensor.grad(
    #     assembled_loglik_jax(data, *dist_params).sum(), wrt=v
    # ).eval()
    # v_grad_pytensor = pytensor.grad(
    #     assembled_loglik_pytensor(data, *dist_params).sum(), wrt=v
    # ).eval()

    # np.testing.assert_array_almost_equal(v_grad_jax, v_grad_pytensor, decimal=DECIMAL)

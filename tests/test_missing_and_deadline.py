from pathlib import Path

import pytest
import numpy as np
import pytensor


DECIMAL = 4


@pytest.fixture
def fixture_path():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def ddm(fixture_path):
    return fixture_path / "ddm.onnx"


@pytest.fixture
def cpn(fixture_path):
    return fixture_path / "ddm_cpn.onnx"


@pytest.fixture
def opn(fixture_path):
    return fixture_path / "ddm_opn.onnx"


@pytest.fixture
def data():
    data = np.ones((100, 3), dtype=np.float32)
    data[:, 0] = np.random.rand(100)
    data[:, 2] = np.random.rand(100)
    missing_indices = data[:, 0] > data[:, 2]
    data[missing_indices, 0] = -999.0

    return data


from hssm.distribution_utils import (
    assemble_callables,
    make_likelihood_callable,
    make_missing_data_callable,
)


def test_make_missing_data_callable_cpn(data, ddm, cpn):
    # Test corner case where data_dim == 0 (CPN case)
    # Also needs to be careful when all parameters are scalar
    # In which case the cpn callable should return a scalar
    data = data[:, :-1]

    dist_params = [0.1, 0.2, 0.4, 0.3]
    dist_params_vector = dist_params.copy()
    param_vec = np.ones(100) * 0.1
    dist_params_vector[0] = param_vec

    # Test cpn when all inputs are scalars
    cpn_callable_jax = make_missing_data_callable(
        cpn, is_cpn_only=True, backend="jax", params_is_reg=[False] * 4
    )

    cpn_callable_pytensor = make_missing_data_callable(
        cpn, is_cpn_only=True, backend="pytensor"
    )

    result_jax = cpn_callable_jax(None, *dist_params).eval()
    result_pytensor = cpn_callable_pytensor(None, *dist_params).eval()

    np.testing.assert_array_almost_equal(result_jax, result_pytensor, decimal=DECIMAL)

    # Test cpn when some inputs are vectors
    cpn_callable_jax_vector = make_missing_data_callable(
        cpn, is_cpn_only=True, backend="jax", params_is_reg=[True] + [False] * 3
    )

    result_jax = cpn_callable_jax_vector(None, *dist_params_vector).eval()
    result_pytensor = cpn_callable_pytensor(None, *dist_params_vector).eval()

    np.testing.assert_array_almost_equal(result_jax, result_pytensor, decimal=DECIMAL)

    # Test assembling two callables, the all-scalar case
    logp_callable_jax = make_likelihood_callable(
        ddm,
        loglik_kind="approx_differentiable",
        backend="jax",
        params_is_reg=[False] * 4,
        data_dim=2,
    )
    missing_mask = data[:, 0] == -999.0

    result_data_jax = logp_callable_jax(data[~missing_mask, :], *dist_params).eval()
    missing_eval = cpn_callable_jax(None, *dist_params).eval()

    assembled_loglik = assemble_callables(
        logp_callable_jax, cpn_callable_jax, is_cpn_only=True, has_deadline=False
    )

    result_assembled = assembled_loglik(data, *dist_params).eval()

    result = np.zeros(100)
    result[~missing_mask] = result_data_jax
    result[missing_mask] = missing_eval

    np.testing.assert_array_almost_equal(
        result_assembled,
        result,
        decimal=DECIMAL,
    )

    # Test assembling two callables, the vector case
    logp_callable_jax_vector = make_likelihood_callable(
        ddm,
        loglik_kind="approx_differentiable",
        backend="jax",
        params_is_reg=[True] + [False] * 3,
        data_dim=2,
    )

    result_data_jax = logp_callable_jax_vector(
        data[~missing_mask, :], param_vec[~missing_mask], *dist_params[1:]
    ).eval()
    missing_eval = cpn_callable_jax_vector(
        None, param_vec[missing_mask], *dist_params[1:]
    ).eval()

    assembled_loglik = assemble_callables(
        logp_callable_jax_vector,
        cpn_callable_jax_vector,
        is_cpn_only=True,
        has_deadline=False,
    )

    result_assembled = assembled_loglik(data, *dist_params_vector).eval()

    result = np.zeros(100)
    result[~missing_mask] = result_data_jax
    result[missing_mask] = missing_eval

    np.testing.assert_array_almost_equal(
        result_assembled,
        result,
        decimal=DECIMAL,
    )

    # Assembling two callables, the pytensor case, all scalar
    logp_callable_pytensor = make_likelihood_callable(
        ddm,
        loglik_kind="approx_differentiable",
        backend="pytensor",
        data_dim=2,
    )

    assembled_loglik = assemble_callables(
        logp_callable_pytensor,
        cpn_callable_pytensor,
        is_cpn_only=True,
        has_deadline=False,
    )

    result_assembled = assembled_loglik(data, *dist_params).eval()

    np.testing.assert_array_almost_equal(
        result_assembled,
        result,
        decimal=DECIMAL,
    )

    # Assembling two callables, the pytensor case, the vector case
    assembled_loglik = assemble_callables(
        logp_callable_pytensor,
        cpn_callable_pytensor,
        is_cpn_only=True,
        has_deadline=False,
    )

    result_assembled = assembled_loglik(data, *dist_params_vector).eval()

    np.testing.assert_array_almost_equal(
        result_assembled,
        result,
        decimal=DECIMAL,
    )


def test_make_missing_data_callable_opn(data, ddm, opn):
    # Test edge case where data_dim == 0 (OPN case)
    # Also needs to be careful when all parameters are scalar
    # In which case the cpn callable should return a scalar

    dist_params = [0.1, 0.2, 0.4, 0.3]
    dist_params_vector = dist_params.copy()
    param_vec = np.ones(100) * 0.1
    dist_params_vector[0] = param_vec

    # Test cpn when all inputs are scalars
    opn_callable_jax = make_missing_data_callable(
        opn, is_cpn_only=False, backend="jax", params_is_reg=[False] * 4
    )

    opn_callable_pytensor = make_missing_data_callable(
        opn, is_cpn_only=False, backend="pytensor"
    )

    result_jax = opn_callable_jax(data[:, -1].reshape((100, 1)), *dist_params).eval()
    result_pytensor = opn_callable_pytensor(data[:, [-1]], *dist_params).eval()

    np.testing.assert_array_almost_equal(result_jax, result_pytensor, decimal=DECIMAL)

    # Test opn when some inputs are vectors
    opn_callable_jax_vector = make_missing_data_callable(
        opn, is_cpn_only=False, backend="jax", params_is_reg=[True] + [False] * 3
    )

    result_jax = opn_callable_jax_vector(data[:, [-1]], *dist_params_vector).eval()
    result_pytensor = opn_callable_pytensor(data[:, [-1]], *dist_params_vector).eval()

    np.testing.assert_array_almost_equal(result_jax, result_pytensor, decimal=DECIMAL)

    # Test assembling two callables, the all-scalar case
    logp_callable_jax = make_likelihood_callable(
        ddm,
        loglik_kind="approx_differentiable",
        backend="jax",
        params_is_reg=[False] * 4,
        data_dim=2,
    )

    missing_mask = data[:, 0] == -999.0

    result_data_jax = logp_callable_jax(data[~missing_mask, :-1], *dist_params).eval()
    missing_eval = opn_callable_jax(data[missing_mask, -1:], *dist_params).eval()

    assembled_loglik = assemble_callables(
        logp_callable_jax, opn_callable_jax, is_cpn_only=False, has_deadline=True
    )

    result_assembled = assembled_loglik(data, *dist_params).eval()

    result = np.zeros(100)
    result[~missing_mask] = result_data_jax
    result[missing_mask] = missing_eval

    np.testing.assert_array_almost_equal(
        result_assembled,
        result,
        decimal=DECIMAL,
    )

    # Test assembling two callables, the vector case
    logp_callable_jax_vector = make_likelihood_callable(
        ddm,
        loglik_kind="approx_differentiable",
        backend="jax",
        params_is_reg=[True] + [False] * 3,
        data_dim=2,
    )

    result_data_jax = logp_callable_jax_vector(
        data[~missing_mask, :-1], param_vec[~missing_mask], *dist_params[1:]
    ).eval()
    missing_eval = opn_callable_jax_vector(
        data[missing_mask, -1:], param_vec[missing_mask], *dist_params[1:]
    ).eval()

    assembled_loglik = assemble_callables(
        logp_callable_jax_vector,
        opn_callable_jax_vector,
        is_cpn_only=False,
        has_deadline=True,
    )

    result_assembled = assembled_loglik(data, *dist_params_vector).eval()

    result = np.zeros(100)
    result[~missing_mask] = result_data_jax
    result[missing_mask] = missing_eval

    np.testing.assert_array_almost_equal(
        result_assembled,
        result,
        decimal=DECIMAL,
    )

    # Assembling two callables, the pytensor case, all scalar
    logp_callable_pytensor = make_likelihood_callable(
        ddm,
        loglik_kind="approx_differentiable",
        backend="pytensor",
        data_dim=2,
    )

    assembled_loglik = assemble_callables(
        logp_callable_pytensor,
        opn_callable_pytensor,
        is_cpn_only=False,
        has_deadline=True,
    )

    result_assembled = assembled_loglik(data, *dist_params).eval()

    np.testing.assert_array_almost_equal(
        result_assembled,
        result,
        decimal=DECIMAL,
    )

    # Assembling two callables, the pytensor case, the vector case
    assembled_loglik = assemble_callables(
        logp_callable_pytensor,
        opn_callable_pytensor,
        is_cpn_only=False,
        has_deadline=True,
    )

    result_assembled = assembled_loglik(data, *dist_params_vector).eval()

    np.testing.assert_array_almost_equal(
        result_assembled,
        result,
        decimal=DECIMAL,
    )

import bambi as bmb
import numpy as np
import pymc as pm
import pytest
import pytensor.tensor as pt

import hssm
from hssm import distribution_utils
from hssm.distribution_utils.dist import (
    apply_param_bounds_to_loglik,
    make_distribution,
    ensure_positive_ndt,
)
from hssm.likelihoods.analytical import logp_ddm, DDM

hssm.set_floatX("float32")


def test_make_ssm_rv():
    params = ["v", "a", "z", "t"]
    seed = 42

    # The order of true values, however, is
    # v, a, z, t
    true_values = [0.5, 0.5, 0.5, 0.3]

    wfpt_rv = distribution_utils.make_ssm_rv("ddm", params)
    rng = np.random.default_rng()

    random_sample = wfpt_rv.rng_fn(rng, *true_values, size=500)

    assert random_sample.shape == (500, 2)

    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    sequential_sample_1 = np.array(
        [wfpt_rv.rng_fn(rng1, *true_values, size=500) for _ in range(5)]
    )

    sequential_sample_2 = np.array(
        [wfpt_rv.rng_fn(rng2, *true_values, size=500) for _ in range(5)]
    )

    np.testing.assert_array_equal(sequential_sample_1, sequential_sample_2)

    true_values[0] = np.ones(100) * 0.5

    random_sample = wfpt_rv.rng_fn(rng, *true_values, size=100)

    assert random_sample.shape[0] == 100

    with pytest.raises(ValueError):
        wfpt_rv.rng_fn(rng, *true_values, size=499)


def test_lapse_distribution():
    lapse_dist = bmb.Prior("Uniform", lower=0.0, upper=1.0)
    rv = distribution_utils.make_ssm_rv("ddm", ["v", "a", "z", "t"], lapse=lapse_dist)
    random_sample = rv.rng_fn(np.random.default_rng(), *[0.5, 0.5, 0.5, 0.3], 0.05, 10)
    assert random_sample.shape == (10, 2)

    random_sample_1 = rv.rng_fn(
        np.random.default_rng(), np.random.uniform(size=10), *[0.5, 0.5, 0.3], 0.05, 10
    )

    assert random_sample_1.shape == (10, 2)
    assert -1.0 in random_sample_1[:, 1]
    assert 1.0 in random_sample_1[:, 1]
    assert 0 not in random_sample_1[:, 1]

    rng1 = np.random.default_rng(10)
    rng2 = np.random.default_rng(10)

    random_sample_a = rv.rng_fn(rng1, *[0.5, 0.5, 0.5, 0.3], 0.05, 10)
    random_sample_b = rv.rng_fn(rng2, *[0.5, 0.5, 0.5, 0.3], 0.05, 10)

    np.testing.assert_array_equal(random_sample_a, random_sample_b)

    # Test reproducibility
    random_sample_a = rv.rng_fn(rng1, *[0.5, 0.5, 0.5, 0.3], 0.05, 10)
    random_sample_b = rv.rng_fn(rng2, *[0.5, 0.5, 0.5, 0.3], 0.05, 10)

    np.testing.assert_array_equal(random_sample_a, random_sample_b)


def test_apply_param_bounds_to_loglik():
    """Tests the function in separation."""
    logp = np.random.normal(size=1000)

    list_params = ["param1", "param2"]
    bounds = {"param1": [-1.0, 1.0], "param2": [-1.0, 1.0]}

    scalar_in_bound = -0.5
    scalar_out_of_bound = 2.0

    random_vector = np.random.uniform(-3, 3, size=1000)
    out_of_bound_indices = np.logical_or(random_vector <= -1.0, random_vector >= 1.0)

    np.testing.assert_array_equal(
        apply_param_bounds_to_loglik(
            logp, list_params, scalar_in_bound, scalar_in_bound, bounds=bounds
        ).eval(),
        logp,
    )

    np.testing.assert_array_equal(
        apply_param_bounds_to_loglik(
            logp, list_params, scalar_in_bound, scalar_out_of_bound, bounds=bounds
        ).eval(),
        -66.1,
    )

    results_vector = np.asarray(
        apply_param_bounds_to_loglik(
            logp, list_params, scalar_in_bound, random_vector, bounds=bounds
        ).eval(),
    )

    np.testing.assert_array_equal(results_vector[out_of_bound_indices], -66.1)

    np.testing.assert_array_equal(
        results_vector[~out_of_bound_indices], logp[~out_of_bound_indices]
    )


def test_make_distribution():
    def fake_logp_function(data, param1, param2):
        """Make up a fake log likelihood function for this test only."""
        return data[:, 0] * param1 * param2

    data = np.zeros((1000, 2))
    data[:, 0] = np.random.normal(size=1000)
    bounds = {"param1": [-1.0, 1.0], "param2": [-1.0, 1.0]}

    Dist = make_distribution(
        "fake",
        loglik=fake_logp_function,
        list_params=["param1", "param2"],
        bounds=bounds,
    )

    scalar_in_bound = -0.5
    scalar_out_of_bound = 2.0

    random_vector = np.random.uniform(-3, 3, size=1000)
    out_of_bound_indices = np.logical_or(random_vector <= -1.0, random_vector >= 1.0)

    np.testing.assert_array_equal(
        Dist.logp(data, scalar_in_bound, scalar_in_bound).eval(),
        data[:, 0] * scalar_in_bound * scalar_in_bound,
    )

    np.testing.assert_array_equal(
        Dist.logp(data, scalar_in_bound, scalar_out_of_bound).eval(),
        -66.1,
    )

    results_vector = np.asarray(Dist.logp(data, scalar_in_bound, random_vector).eval())

    np.testing.assert_array_equal(results_vector[out_of_bound_indices], -66.1)

    np.testing.assert_array_equal(
        results_vector[~out_of_bound_indices],
        data[:, 0][~out_of_bound_indices]
        * scalar_in_bound
        * random_vector[~out_of_bound_indices],
    )


def test_extra_fields(data_ddm):
    ones = np.ones(data_ddm.shape[0])
    x = ones * 0.5
    y = ones * 4.0

    def logp_ddm_extra_fields(data, v, a, z, t, x, y):
        return logp_ddm(data, v, a, z, t) * x * y

    DDM_WITH_XY = make_distribution(
        rv="ddm",
        loglik=logp_ddm_extra_fields,
        list_params=["v", "a", "z", "t"],
        extra_fields=[x, y],
    )

    true_values = dict(v=0.5, a=1.5, z=0.5, t=0.5)

    np.testing.assert_almost_equal(
        pm.logp(DDM.dist(**true_values), data_ddm).eval(),
        pm.logp(DDM_WITH_XY.dist(**true_values), data_ddm).eval() / 2.0,
    )

    data_ddm_copy = data_ddm.copy()
    data_ddm_copy["x"] = x
    data_ddm_copy["y"] = y

    ddm_model_xy = hssm.HSSM(
        data=data_ddm_copy,
        model_config=dict(extra_fields=["x", "y"]),
        loglik=logp_ddm_extra_fields,
        p_outlier=None,
        lapse=None,
    )

    np.testing.assert_almost_equal(
        pm.logp(DDM.dist(**true_values), data_ddm).eval(),
        pm.logp(ddm_model_xy.model_distribution.dist(**true_values), data_ddm).eval()
        / 2.0,
    )

    ddm_model = hssm.HSSM(data=data_ddm)
    ddm_model_p = hssm.HSSM(
        data=data_ddm_copy,
        model_config=dict(extra_fields=["x", "y"]),
        loglik=logp_ddm_extra_fields,
    )
    ddm_model_p_logp_without_lapse = (
        pm.logp(
            ddm_model_p.model_distribution.dist(**true_values, p_outlier=0),
            data_ddm,
        )
        / 2
    )
    ddm_model_p_logp_lapse = pt.log(
        0.95 * pt.exp(ddm_model_p_logp_without_lapse)
        + 0.05
        * pt.exp(pm.logp(pm.Uniform.dist(lower=0.0, upper=10.0), data_ddm["rt"].values))
    )
    np.testing.assert_almost_equal(
        pm.logp(
            ddm_model.model_distribution.dist(**true_values, p_outlier=0.05), data_ddm
        ).eval(),
        ddm_model_p_logp_lapse.eval(),
    )


def test_ensure_positive_ndt():
    data = np.zeros((1000, 2))
    data[:, 0] = np.random.uniform(size=1000)

    logp = np.random.uniform(size=1000)

    list_params = ["v", "a", "z", "t"]
    dist_params = [0.5] * 4

    after_replacement = ensure_positive_ndt(data, logp, list_params, dist_params).eval()
    mask = data[:, 0] - 0.5 <= 1e-15

    assert np.all(after_replacement[mask] == np.array(-66.1))
    assert np.all(after_replacement[~mask] == logp[~mask])

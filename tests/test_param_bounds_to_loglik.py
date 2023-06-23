import numpy as np

from hssm.wfpt.wfpt import apply_param_bounds_to_loglik, make_distribution


def test_apply_param_bouds_to_loglik():
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
        return data * param1 * param2

    data = np.random.normal(size=1000)
    bounds = {"param1": [-1.0, 1.0], "param2": [-1.0, 1.0]}

    Dist = make_distribution(
        model_name="custom",
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
        data * scalar_in_bound * scalar_in_bound,
    )

    np.testing.assert_array_equal(
        Dist.logp(data, scalar_in_bound, scalar_out_of_bound).eval(),
        -66.1,
    )

    results_vector = np.asarray(Dist.logp(data, scalar_in_bound, random_vector).eval())

    np.testing.assert_array_equal(results_vector[out_of_bound_indices], -66.1)

    np.testing.assert_array_equal(
        results_vector[~out_of_bound_indices],
        data[~out_of_bound_indices]
        * scalar_in_bound
        * random_vector[~out_of_bound_indices],
    )

import bambi as bmb
import numpy as np
import pytest

from hssm import distribution_utils


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

    rng1 = np.random.default_rng(10)
    rng2 = np.random.default_rng(10)

    random_sample_a = rv.rng_fn(rng1, *[0.5, 0.5, 0.5, 0.3], 0.05, 10)
    random_sample_b = rv.rng_fn(rng2, *[0.5, 0.5, 0.5, 0.3], 0.05, 10)

    np.testing.assert_array_equal(random_sample_a, random_sample_b)

    # Test reproducibility
    random_sample_a = rv.rng_fn(rng1, *[0.5, 0.5, 0.5, 0.3], 0.05, 10)
    random_sample_b = rv.rng_fn(rng2, *[0.5, 0.5, 0.5, 0.3], 0.05, 10)

    np.testing.assert_array_equal(random_sample_a, random_sample_b)

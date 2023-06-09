import numpy as np
import pytest

from hssm import wfpt


def test_make_model_rv():
    params = ["v", "a", "z", "t"]
    seed = 42

    # The order of true values, however, is
    # v, a, z, t
    true_values = [0.5, 0.5, 0.5, 0.3]

    wfpt_rv = wfpt.make_model_rv("ddm", params)
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
        wfpt_rv.rng_fn(rng, *true_values, size=500)

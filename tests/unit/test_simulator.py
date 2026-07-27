import numpy as np
import pandas as pd
import pytest

import hssm
from hssm.simulator import simulate_data

hssm.set_floatX("float32")


def test_simulator():
    theta = [0.5, 1.5, 0.5, 0.5]

    # Should throw error if model is not supported:
    with pytest.raises(ValueError):
        simulate_data("custom", theta=theta, size=10)

    # Should return DataFrame by default
    df = simulate_data("ddm", theta=theta, size=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10

    # Should return array if output_df is False
    arr = simulate_data("ddm", theta=theta, size=10, output_df=False)
    assert isinstance(arr, np.ndarray)

    # Should return same values if random_state is set to the same value
    arr1 = simulate_data("ddm", theta=theta, size=10, random_state=1, output_df=False)
    arr2 = simulate_data("ddm", theta=theta, size=10, random_state=1, output_df=False)
    np.testing.assert_array_equal(arr1, arr2)

    # Should return `size` for each subject
    arr = simulate_data("ddm", theta=[theta, theta], size=10, output_df=False)
    assert arr.shape == (2 * 10, 2)

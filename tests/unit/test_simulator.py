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


def test_simulator_registered_model(registered_ssms_model):
    """A model registered in ssm-simulators after import simulates by name (#1467)."""
    name = registered_ssms_model["name"]
    theta = dict(v=0.5, a=1.5, z=0.5, t=0.3)

    df = simulate_data(name, theta=theta, size=10, random_state=1)
    ddm_df = simulate_data("ddm", theta=theta, size=10, random_state=1)

    # The registered model is a copy of the DDM, so it reproduces its draws.
    pd.testing.assert_frame_equal(df, ddm_df)


def test_simulator_unknown_model_lists_registered(registered_ssms_model):
    """The error for an unknown name lists the models ssm-simulators knows."""
    with pytest.raises(ValueError, match=registered_ssms_model["name"]):
        simulate_data("not_a_model", theta=[0.5, 1.5, 0.5, 0.5], size=10)

"""
Unit testing for WFPT likelihood function.

This code compares WFPT likelihood function with
old implementation of WFPT from (https://github.com/hddm-devs/hddm)
"""
import os
import sys

import numpy as np
import pytest
import ssms.basic_simulators

sys.path.insert(0, os.path.abspath("src"))

import pandas as pd

sys.path.insert(0, os.path.abspath("src"))


@pytest.fixture
def data_fixture():
    v_true, a_true, z_true, t_true, sv_true = [0.5, 1.5, 0.5, 0.5, 0.3]
    obs_angle = ssms.basic_simulators.simulator(
        [v_true, a_true, z_true, t_true, sv_true], model="ddm", n_samples=1000
    )
    obs = np.column_stack([obs_angle["rts"][:, 0], obs_angle["choices"][:, 0]])
    obs = pd.DataFrame(obs, columns=["rt", "response"])
    return obs


def test_kterm(data_fixture):
    """
    This function defines a range of kterms and tests results to
     makes sure they are not equal to infinity or unknown values
    """

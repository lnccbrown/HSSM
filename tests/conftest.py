import pytest

import arviz as az
import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
from ssms.basic_simulators.simulator import simulator

import hssm

mpl.use("Agg")


@pytest.fixture(scope="module")
def data_ddm():
    v_true, a_true, z_true, t_true = [0.5, 1.5, 0.5, 0.5]
    obs_ddm = simulator([v_true, a_true, z_true, t_true], model="ddm", n_samples=100)
    obs_ddm = np.column_stack([obs_ddm["rts"][:, 0], obs_ddm["choices"][:, 0]])
    data = pd.DataFrame(obs_ddm, columns=["rt", "response"])

    return data


@pytest.fixture(scope="module")
def data_angle():
    v_true, a_true, z_true, t_true, theta_true = [0.5, 1.5, 0.5, 0.5, 0.3]
    obs_angle = simulator(
        [v_true, a_true, z_true, t_true, theta_true], model="angle", n_samples=100
    )
    obs_angle = np.column_stack([obs_angle["rts"][:, 0], obs_angle["choices"][:, 0]])
    data = pd.DataFrame(obs_angle, columns=["rt", "response"])
    return data


@pytest.fixture(scope="module")
def data_ddm_reg():
    # Generate some fake simulation data
    intercept = 1.5
    x = np.random.uniform(-5.0, 5.0, size=1000)
    y = np.random.uniform(-5.0, 5.0, size=1000)

    v = intercept + 0.8 * x + 0.3 * y
    true_values = np.column_stack(
        [v, np.repeat([[1.5, 0.5, 0.5]], axis=0, repeats=1000)]
    )

    dataset_reg_v = hssm.simulate_data(
        model="ddm",
        theta=true_values,
        size=1,  # Generate one data point for each of the 1000 set of true values
    )

    dataset_reg_v["x"] = x
    dataset_reg_v["y"] = y

    return dataset_reg_v


@pytest.fixture
def cav_idata():
    return az.from_netcdf("tests/fixtures/cavanagh_idata.nc")


@pytest.fixture
def posterior():
    return xr.open_dataarray("tests/fixtures/cavanagh_idata_pps.nc")


@pytest.fixture
def cavanagh_test():
    return pd.read_csv("tests/fixtures/cavanagh_theta_test.csv", index_col=None)

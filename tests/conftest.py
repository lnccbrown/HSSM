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
    x = np.random.uniform(-0.5, 0.5, size=1000)
    y = np.random.uniform(-0.5, 0.5, size=1000)

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


@pytest.fixture(scope="module")
def data_ddm_reg_va():
    # Generate some fake simulation data
    intercept = 1.5
    intercept_a = 1.0
    x = np.random.uniform(-0.5, 0.5, size=100)
    y = np.random.uniform(-0.5, 0.5, size=100)

    m = np.random.uniform(-0.5, 0.5, size=100)
    n = np.random.uniform(-0.5, 0.5, size=100)

    v = intercept + 0.8 * x + 0.3 * y
    a = intercept_a + 0.1 * m + 0.1 * n
    true_values = np.column_stack([v, a, np.repeat([[0.5, 0.5]], axis=0, repeats=100)])

    dataset_reg_va = hssm.simulate_data(
        model="ddm",
        theta=true_values,
        size=1,  # Generate one data point for each of the 1000 set of true values
    )

    dataset_reg_va["x"] = x
    dataset_reg_va["y"] = y
    dataset_reg_va["m"] = m
    dataset_reg_va["n"] = n

    return dataset_reg_va


@pytest.fixture
def cav_idata():
    return az.from_netcdf("tests/fixtures/cavanagh_idata.nc")


@pytest.fixture
def posterior():
    return xr.open_dataarray("tests/fixtures/cavanagh_idata_pps.nc")


@pytest.fixture
def cavanagh_test():
    return pd.read_csv("tests/fixtures/cavanagh_theta_test.csv", index_col=None)


# @pytest.fixture
# def cavanagh_data():
#     return hssm.load_data("cavanagh_theta")


@pytest.fixture
def basic_hssm_model():
    cav_data = hssm.load_data("cavanagh_theta")
    basic_hssm_model = hssm.HSSM(
        data=cav_data,
        process_initvals=True,
        link_settings="log_logit",
        model="angle",
        include=[
            {
                "name": "v",
                "formula": "v ~ 1 + C(stim)",
            }
        ],
    )
    return basic_hssm_model


# Cartoon plot fixtures
@pytest.fixture
def cav_model_cartoon(cavanagh_test):
    cav_model = hssm.HSSM(
        model="ddm",
        data=cavanagh_test,
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.5},
                },
                "formula": "v ~ 1 + stim",
                "link": "identity",
            },
            {
                "name": "a",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 1.5, "sigma": 0.5},
                },
                "formula": "a ~ 1 + (1|participant_id)",
                "link": "identity",
            },
        ],
        p_outlier=0.00,
    )

    # Attach trace
    idata_cav = az.from_netcdf("tests/fixtures/idata_cavanagh_cartoon.nc")
    cav_model._inference_obj = idata_cav
    return cav_model


@pytest.fixture
def race_model_cartoon():
    my_race_data = pd.read_csv("tests/fixtures/data_race.csv")
    race_model = hssm.HSSM(
        model="race_no_bias_angle_4",
        data=my_race_data,
        include=[
            {
                "name": "v0",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.5},
                },
                "formula": "v0 ~ 1 + stim",
                "link": "identity",
            },
            {
                "name": "a",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 1.5, "sigma": 0.5},
                },
                "formula": "a ~ 1 + (1|participant_id)",
                "link": "identity",
            },
        ],
        p_outlier=0.00,
    )
    # Attach trace
    idata_race = az.from_netcdf("tests/fixtures/test_idata_race.nc")
    race_model._inference_obj = idata_race
    return race_model


# Only useful if running tests serially
def pytest_collection_modifyitems(config, items):
    slow_tests = [item for item in items if "slow" in item.keywords]
    fast_tests = [item for item in items if "slow" not in item.keywords]

    # Reorder items so fast tests run first, then slow tests
    items[:] = fast_tests + slow_tests

import bambi as bmb
import numpy as np
import pandas as pd
import pytensor
import pytest
import ssms

from hssm import hssm
from hssm.wfpt.config import default_model_config


@pytest.fixture
def data():
    v_true, a_true, z_true, t_true, sv_true = [0.5, 1.5, 0.5, 0.5, 0.3]
    obs_ddm = ssms.basic_simulators.simulator(
        [v_true, a_true, z_true, t_true, sv_true], model="ddm", n_samples=1000
    )
    obs_ddm = np.column_stack([obs_ddm["rts"][:, 0], obs_ddm["choices"][:, 0]])
    dataset = pd.DataFrame(obs_ddm, columns=["rt", "response"])
    dataset["x"] = dataset["rt"] * 0.1
    dataset["y"] = dataset["rt"] * 0.5
    return dataset


@pytest.fixture
def data_angle():
    v_true, a_true, z_true, t_true, theta_true = [0.5, 1.5, 0.5, 0.5, 0.3]
    obs_angle = ssms.basic_simulators.simulator(
        [v_true, a_true, z_true, t_true, theta_true], model="angle", n_samples=1000
    )
    obs_angle = np.column_stack([obs_angle["rts"][:, 0], obs_angle["choices"][:, 0]])
    data = pd.DataFrame(obs_angle, columns=["rt", "response"])
    return data


def test_ddm(data):
    pytensor.config.floatX = "float32"
    model = hssm.HSSM(data=data)
    assert model.list_params == ["v", "sv", "a", "z", "t"]
    assert isinstance(model.formula, bmb.formula.Formula)
    assert model.link == {"v": "identity"}
    assert model.model_config == default_model_config["ddm"]
    assert list(model.model_config.keys()) == [
        "loglik_kind",
        "list_params",
        "backend",
        "formula",
        "prior",
    ]
    assert list(model.model_config["prior"].keys()) == model.list_params


def test_hssm_init_model_names(data, data_angle):
    with pytest.raises(ValueError):
        hssm.HSSM(data, model="invalid_model")

    hssm_ddm = hssm.HSSM(data)
    assert hssm_ddm.model_distribution.__name__ == "WFPTDistribution"


def test_hssm_sample(data):
    model = hssm.HSSM(data)
    posterior_samples = model.sample()

    assert posterior_samples is not None
    assert "posterior" in posterior_samples
    assert "sample_stats" in posterior_samples
    assert "observed_data" in posterior_samples


# def test_angle(data_angle):
#     pytensor.config.floatX = "float32"
#     model = hssm.HSSM(data=data_angle, model="angle")
#     assert isinstance(model.model, bmb.models.Model)
#     assert model.list_params == ["v", "a", "z", "t", "theta"]
#     assert isinstance(model.formula, bmb.formula.Formula)
#     assert model.link == {"v": "identity"}
#     assert model.model_config == default_model_config["angle"]


def test_transform_params(data):
    include = [
        {
            "name": "v",
            "prior": {
                "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
                "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
                "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
            },
            "formula": "v ~ 1 + x + y",
            "link": "identity",
        }
    ]
    model = hssm.HSSM(data=data, include=include)

    # Check model properties using a loop
    param_names = ["v", "sv", "a", "z", "t"]
    for idx, param in enumerate(model.params):
        assert param.name == param_names[idx]

    # Check the specific include properties
    assert model.params[0].prior.keys() == include[0]["prior"].keys()
    assert model.params[0].formula == include[0]["formula"]

    assert len(model.params) == 5


def test_transform_params_two(data):
    include = [
        {
            "name": "v",
            "prior": {
                "Intercept": {"name": "Uniform", "lower": -2.0, "upper": 3.0},
                "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
                "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
            },
            "formula": "v ~ 1 + x + y",
        },
        {
            "name": "a",
            "prior": {
                "Intercept": {"name": "Uniform", "lower": -2.0, "upper": 3.0},
                "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
                "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
            },
            "formula": "a ~ 1 + x + y",
        },
    ]
    model = hssm.HSSM(data=data, include=include)
    assert model.params[0].prior.keys() == include[0]["prior"].keys()
    assert model.params[1].prior.keys() == include[1]["prior"].keys()
    assert model.params[0].formula == include[0]["formula"]
    assert model.params[1].formula == include[1]["formula"]
    assert len(model.params) == 5


def test_transform_params_three(data):
    include = [
        {
            "name": "v",  # change to name
            "prior": {
                "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
                "x": {"name": "Uniform", "lower": -2.0, "upper": 1.0},
                "y": {"name": "Uniform", "lower": -2.0, "upper": 1.0},
            },
            "formula": "v ~ 1 + x + y",
        },
        {"name": "a", "prior": 0.5},
    ]
    model = hssm.HSSM(data=data, include=include)
    assert model.params[1].prior == include[1]["prior"]
    assert model.params[0].prior.keys() == include[0]["prior"].keys()
    assert model.params[1].name == include[1]["name"]
    assert model.params[0].formula == include[0]["formula"]
    assert len(model.params) == 5


def test_transform_params_four(data):
    include = [
        {
            "name": "a",  # change to name
            "prior": {
                "Intercept": {
                    "name": "Uniform",
                    "lower": 0.0,
                    "upper": 1.0,
                    "initval": 0.5,
                },
                "x": {"name": "Uniform", "lower": -0.5, "upper": 0.5, "initval": 0},
            },
            "formula": "a ~ 1 + x",
        }
    ]
    model = hssm.HSSM(data=data, include=include)
    assert model.model_config["formula"] == default_model_config["angle"]["formula"]
    assert model.params[0].prior.keys() == include[0]["prior"].keys()
    assert model.params[0].formula == include[0]["formula"]
    assert len(model.params) == 5


def test_invalid_include_key(data):
    include = [
        {
            "name": "v",
            "prior": {"Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0}},
            "formula": "v ~ 1",
            "invalid_key": "identity",
        }
    ]
    with pytest.raises(Exception):
        hssm.HSSM(data=data, include=include)


def test_invalid_param_name(data):
    include = [
        {
            "name": "invalid_param",
            "prior": {"Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0}},
            "formula": "v ~ 1",
        }
    ]
    with pytest.raises(Exception):
        hssm.HSSM(data=data, include=include)


def test_invalid_formula(data):
    include = [
        {
            "name": "v",
            "prior": {"Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0}},
            "formula": "invalid_formula",
        }
    ]
    with pytest.raises(Exception):
        hssm.HSSM(data=data, include=include)

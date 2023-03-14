import arviz
import bambi as bmb
import numpy as np
import pandas as pd
import pytensor
import pytest
import ssms.basic_simulators

from hssm import hssm


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
def data_lan():
    v_true, a_true, z_true, t_true, theta_true = [0.5, 1.5, 0.5, 0.5, 0.3]
    obs_angle = ssms.basic_simulators.simulator(
        [v_true, a_true, z_true, t_true, theta_true], model="angle", n_samples=1000
    )
    obs_angle = np.column_stack([obs_angle["rts"][:, 0], obs_angle["choices"][:, 0]])
    return pd.DataFrame(obs_angle, columns=["rt", "response"])


def test_base(data):
    pytensor.config.floatX = "float32"
    model = hssm.HSSM(data=data)
    assert isinstance(model.model, bmb.models.Model)
    assert model.list_params == ["v", "sv", "a", "z", "t"]
    assert isinstance(model.formula, bmb.formula.Formula)
    assert model.link == {"v": "identity"}
    samples = model.sample()
    assert isinstance(samples, arviz.data.inference_data.InferenceData)


def test_lan(data_lan):
    pytensor.config.floatX = "float32"
    model = hssm.HSSM(data=data_lan, model="angle")
    assert isinstance(model.model, bmb.models.Model)
    assert model.list_params == ["v", "a", "z", "t", "theta"]
    assert isinstance(model.formula, bmb.formula.Formula)
    assert model.link == {"v": "identity"}
    samples = model.sample()
    assert isinstance(samples, arviz.data.inference_data.InferenceData)


def test_transform_params(data):
    include = [
        {
            "name": "v",  # change to name
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
    assert model.params[0].prior.keys() == include[0]["prior"].keys()
    assert model.params[0].formula == include[0]["formula"]
    assert model.params[0].name == "v"
    assert model.params[1].name == "a"
    assert model.params[2].name == "sv"
    assert model.params[3].name == "z"
    assert model.params[4].name == "t"
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

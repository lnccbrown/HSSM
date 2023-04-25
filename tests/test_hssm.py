from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import ssms

from hssm import hssm
from hssm.wfpt import WFPT


@pytest.fixture
def data():
    v_true, a_true, z_true, t_true = [0.5, 1.5, 0.5, 0.5]
    obs_ddm = ssms.basic_simulators.simulator(
        [v_true, a_true, z_true, t_true], model="ddm", n_samples=1000
    )
    obs_ddm = np.column_stack([obs_ddm["rts"][:, 0], obs_ddm["choices"][:, 0]])
    dataset = pd.DataFrame(obs_ddm, columns=["rt", "response"])
    dataset["x"] = dataset["rt"] * 0.1
    dataset["y"] = dataset["rt"] * 0.5
    return dataset


@pytest.fixture(scope="module")
def fixture_path():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def data_angle():
    v_true, a_true, z_true, t_true, theta_true = [0.5, 1.5, 0.5, 0.5, 0.3]
    obs_angle = ssms.basic_simulators.simulator(
        [v_true, a_true, z_true, t_true, theta_true], model="angle", n_samples=1000
    )
    obs_angle = np.column_stack([obs_angle["rts"][:, 0], obs_angle["choices"][:, 0]])
    data = pd.DataFrame(obs_angle, columns=["rt", "response"])
    return data


@pytest.fixture
def example_model_config():
    return {
        "loglik_kind": "example",
        "list_params": ["v", "sv", "a", "z", "t"],
        "default_prior": {
            "v": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "sv": {"name": "Uniform", "lower": 0.0, "upper": 1.0},
            "a": {"name": "Uniform", "lower": 0.30, "upper": 2.5},
            "z": {"name": "Uniform", "lower": 0.10, "upper": 0.9},
            "t": {"name": "Uniform", "lower": 0.0, "upper": 2.0},
        },
        "default_boundaries": {
            "v": (-3.0, 3.0),
            "sv": (0.0, 1.0),
            "a": (0.3, 2.5),
            "z": (0.1, 0.9),
            "t": (0.0, 2.0),
        },
    }


@pytest.mark.parametrize(
    "include, should_raise_exception",
    [
        (
            [
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
            ],
            False,
        ),
        (
            [
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
            ],
            False,
        ),
        (
            [{"name": "invalid_param", "prior": "invalid_param"}],
            True,
        ),
        (
            [
                {
                    "name": "v",
                    "prior": {
                        "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0}
                    },
                    "formula": "v ~ 1",
                    "invalid_key": "identity",
                }
            ],
            True,
        ),
        (
            [
                {
                    "name": "v",
                    "prior": {
                        "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0}
                    },
                    "formula": "invalid_formula",
                }
            ],
            True,
        ),
    ],
)
def test_transform_params_general(data, include, should_raise_exception):
    if should_raise_exception:
        with pytest.raises(Exception):
            hssm.HSSM(data=data, include=include)
    else:
        model = hssm.HSSM(data=data, include=include)
        # Check model properties using a loop
        param_names = ["v", "sv", "a", "z", "t"]
        model_param_names = sorted([param.name for param in model.params])
        assert model_param_names == sorted(param_names)
        assert len(model.params) == 5


def test_model_config_and_loglik_path_update(data_angle, fixture_path):
    my_hssm = hssm.HSSM(
        data=data_angle,
        model="angle",
        model_config={
            "loglik_kind": "approx_differentiable",
            "loglik_path": fixture_path / "new_path.onnx",
        },
    )
    assert my_hssm.model_config["loglik_path"] == fixture_path / "new_path.onnx"
    assert my_hssm.model_config["loglik_kind"] == "approx_differentiable"


def test_custom_model(data, example_model_config):

    with pytest.raises(ValueError):
        model = hssm.HSSM(data=data, model="custom", model_config=example_model_config)

    example_model_config["loglik_kind"] = "approx_differentiable"

    with pytest.raises(ValueError):
        model = hssm.HSSM(data=data, model="custom", model_config=example_model_config)

    example_model_config["loglik"] = WFPT

    model = hssm.HSSM(data=data, model="custom", model_config=example_model_config)

    assert model.model_name == "custom"
    assert model.model_config == example_model_config

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import ssms

from hssm import hssm
from hssm.wfpt import WFPT
from hssm.wfpt.config import download_hf


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
        "loglik_kind": "blackbox",
        "list_params": ["v", "sv", "a", "z", "t"],
        "bounds": {
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

    with pytest.raises(KeyError):
        model = hssm.HSSM(data=data, model="custom", model_config=example_model_config)

    example_model_config["loglik"] = WFPT
    example_model_config["loglik_kind"] = "analytical"

    model = hssm.HSSM(data=data, model="custom", model_config=example_model_config)

    assert model.model_name == "custom"
    assert model.model_config == example_model_config


def test_model_definition_outside_include(data):
    model_with_one_param_fixed = hssm.HSSM(data, a=0.5)

    assert "a" in model_with_one_param_fixed.priors
    assert model_with_one_param_fixed.priors["a"] == 0.5

    model_with_one_param = hssm.HSSM(
        data, a={"prior": {"name": "Normal", "mu": 0.5, "sigma": 0.1}}
    )

    assert "a" in model_with_one_param.priors
    assert model_with_one_param.priors["a"].name == "Normal"

    with pytest.raises(
        ValueError, match='Parameter "a" is already specified in `include`'
    ):
        hssm.HSSM(data, include=[{"name": "a", "prior": 0.5}], a=0.5)


def test_custom_model_without_model_config_and_loglik_raises_error(data):
    with pytest.raises(
        ValueError,
        match="For custom models, both `likelihood_kind` and `loglik` must be provided.",
    ):
        hssm.HSSM(data=data, model="custom")


#
def test_custom_model_with_analytical_likelihood_type(data):
    likelihood_kind = "analytical"
    loglik = WFPT
    model = hssm.HSSM(
        data=data, model="ddm", likelihood_kind=likelihood_kind, loglik=loglik
    )
    assert model.model_config["loglik"] == loglik


#
def test_custom_model_with_approx_differentiable_likelihood_type(data_angle):
    likelihood_kind = "approx_differentiable"
    loglik = "angle.onnx"
    model = hssm.HSSM(
        data=data_angle, model="angle", likelihood_kind=likelihood_kind, loglik=loglik
    )
    assert model.model_config["loglik"] == download_hf(loglik)

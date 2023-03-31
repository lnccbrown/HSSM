import numpy as np
import pandas as pd
import pytest
import ssms

from hssm import hssm


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

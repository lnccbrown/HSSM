import numpy as np
import pandas as pd
import pytensor
import pytest
import ssms.basic_simulators
from jax.config import config

import hssm
from hssm.utils import set_floatX


def test_get_alias_dict():
    # Simulate some data:
    v_true, a_true, z_true, t_true, sv_true = [0.5, 1.5, 0.5, 0.5, 0.3]
    obs_ddm = ssms.basic_simulators.simulator(
        [v_true, a_true, z_true, t_true, sv_true], model="ddm", n_samples=1000
    )
    obs_ddm = np.column_stack([obs_ddm["rts"][:, 0], obs_ddm["choices"][:, 0]])

    dataset = pd.DataFrame(obs_ddm, columns=["rt", "response"])
    dataset["x"] = dataset["rt"] * 0.1
    dataset["y"] = dataset["rt"] * 0.5
    dataset["group"] = np.random.randint(0, 10, len(dataset))

    alias_default = hssm.HSSM(data=dataset)._aliases
    alias_regression = hssm.HSSM(
        data=dataset,
        include=[
            {
                "name": "v",  # change to name
                "prior": {
                    "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
                    "x": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
                    "y": {"name": "Uniform", "lower": -0.50, "upper": 0.50},
                },
                "formula": "v ~ (1|group) + x + y",
                "link": "identity",
            }
        ],
    )._aliases

    alias_regression_a = hssm.HSSM(
        data=dataset,
        include=[
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
                "formula": "a ~ (1|group) + x",
            }
        ],
    )._aliases

    assert alias_default["c(rt, response)"] == "rt,response"
    assert alias_default["Intercept"] == "v"
    assert alias_default["a"] == "a"

    assert alias_regression["c(rt, response)"] == "rt,response"
    assert alias_regression["Intercept"] == "v_Intercept"
    assert alias_regression["x"] == "v_x"
    assert alias_regression["1|group"] == "v_1|group"

    assert alias_regression_a["c(rt, response)"] == "rt,response"
    assert alias_regression_a["Intercept"] == "a_Intercept"
    assert alias_regression_a["x"] == "a_x"
    assert alias_regression_a["1|group"] == "a_1|group"


def test_set_floatX():
    # Should raise error when wrong value is passed.
    with pytest.raises(ValueError):
        set_floatX("bad_value")

    set_floatX("float32")
    assert pytensor.config.floatX == "float32"
    assert not config.read("jax_enable_x64")

    set_floatX("float64")
    assert pytensor.config.floatX == "float64"
    assert config.read("jax_enable_x64")

import numpy as np
import pandas as pd
import pytensor
import pytest
from ssms.basic_simulators.simulator import simulator
from jax import config

import hssm
from hssm.utils import (
    set_floatX,
    _generate_random_indices,
    _random_sample,
)

hssm.set_floatX("float32")


@pytest.mark.slow
def test_get_alias_dict():
    # Simulate some data:
    v_true, a_true, z_true, t_true = [0.5, 1.5, 0.5, 0.5]
    obs_ddm = simulator([v_true, a_true, z_true, t_true], model="ddm", n_samples=1000)
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
    assert alias_default["v"] == "v_mean"
    assert "a" not in alias_default

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


@pytest.mark.parametrize(
    ["n_samples", "n_draws", "expected"],
    [
        (0, 100, "error"),
        (1, 100, 1),
        (100, 100, 100),
        (101, 100, None),
        (1.0, 100, 100),
        (0.0, 100, "error"),
        (0.5, 100, 50),
        (2.0, 100, "error"),
        (None, 100, None),
        (0.5, 0, "error"),
    ],
)
def test__generate_random_indice(caplog, n_samples, n_draws, expected):
    """Test _generate_random_indices."""
    if expected == "error":
        with pytest.raises(ValueError):
            indices = _generate_random_indices(n_samples, n_draws)
    else:
        indices = _generate_random_indices(n_samples, n_draws)
        if expected is None:
            assert indices is None
        else:
            assert indices.dtype == np.int64
            assert len(indices) == expected
            assert np.all(indices < n_draws)
            assert np.all(indices >= 0)
            if n_samples > n_draws:
                assert "n_samples > n_draws" in caplog.text


def assertions(caplog, obj, n_samples, expected):
    if expected == "error":
        with pytest.raises(ValueError):
            sampled_obj = _random_sample(obj, n_samples=n_samples)
    else:
        sampled_obj = _random_sample(obj, n_samples=n_samples)
        assert sampled_obj.draw.size == expected
        assert sampled_obj.chain.size == 2
        assert type(sampled_obj) == type(obj)
        if n_samples and n_samples > obj.draw.size:
            assert "n_samples > n_draws" in caplog.text


@pytest.mark.parametrize(
    ["n_samples", "expected"],
    [
        (0, "error"),
        (1, 1),
        (100, 100),
        (501, 500),
        (0.0, "error"),
        (0.5, 250),
        (1.0, 500),
        (2.0, "error"),
        (None, 500),
    ],
)
def test__random_sample(
    caplog,
    cav_idata,
    n_samples,
    expected,
):
    posterior = cav_idata.posterior
    posterior_predictive = cav_idata.posterior_predictive

    assertions(caplog, posterior, n_samples, expected)
    assertions(caplog, posterior_predictive, n_samples, expected)


def test_check_data_for_rl():
    # Valid DataFrame
    data_valid = pd.DataFrame(
        {
            "participant_id": [2, 2, 1, 1],
            "trial_id": [1, 2, 1, 2],
            "value": [10, 20, 30, 40],
        }
    )
    sorted_data, n_participants, n_trials = hssm.utils.check_data_for_rl(data_valid)
    # Check sorting
    assert list(sorted_data["participant_id"]) == [1, 1, 2, 2]
    assert list(sorted_data["trial_id"]) == [1, 2, 1, 2]
    assert n_participants == 2
    assert n_trials == 2

    # Missing participant_id column
    data_missing_participant = pd.DataFrame(
        {
            "trial_id": [1, 2, 1, 2],
            "value": [10, 20, 30, 40],
        }
    )
    with pytest.raises(ValueError, match="participant_id"):
        hssm.utils.check_data_for_rl(data_missing_participant)

    # Missing trial_id column
    data_missing_trial = pd.DataFrame(
        {
            "participant_id": [1, 1, 2, 2],
            "value": [10, 20, 30, 40],
        }
    )
    with pytest.raises(ValueError, match="trial_id"):
        hssm.utils.check_data_for_rl(data_missing_trial)

    # Unequal number of trials per participant
    data_unequal_trials = pd.DataFrame(
        {
            "participant_id": [1, 1, 2],
            "trial_id": [1, 2, 1],
            "value": [10, 20, 30],
        }
    )
    with pytest.raises(ValueError, match="same number of trials"):
        hssm.utils.check_data_for_rl(data_unequal_trials)

    # Custom column names
    data_custom_cols = pd.DataFrame(
        {
            "subj": [1, 1, 2, 2],
            "trial": [1, 2, 1, 2],
            "value": [10, 20, 30, 40],
        }
    )
    sorted_data, n_participants, n_trials = hssm.utils.check_data_for_rl(
        data_custom_cols, participant_id_col="subj", trial_id_col="trial"
    )
    assert list(sorted_data["subj"]) == [1, 1, 2, 2]
    assert list(sorted_data["trial"]) == [1, 2, 1, 2]
    assert n_participants == 2
    assert n_trials == 2


def test_predictive_idata_to_dataframe(data_ddm):
    model = hssm.HSSM(data=data_ddm)
    sample_do = model.sample_do(params={"v": 1.0}, draws=10)
    df = hssm.utils.predictive_idata_to_dataframe(
        sample_do, predictive_group="prior_predictive"
    )
    assert df is not None
    assert df.shape == (10 * data_ddm.shape[0], 5)
    assert set(df.columns) == {"chain", "draw", "rt", "response", "__obs__"}

from pathlib import Path

import pytest
import numpy as np
import pandas as pd

import hssm


@pytest.fixture
def cpn():
    return Path(__file__).parent / "fixtures" / "ddm_cpn.onnx"


pattern = r"Field\(s\) `.*` not found in data\."

# The DataValidatorMixin is tested in the test_data_validator.py file, so this file
# can probably be removed in the future. CP


def test_data_sanity_check(data_ddm):
    # Case 1: raise error if there are missing fields in data
    with pytest.raises(ValueError, match=pattern):
        hssm.HSSM(data=data_ddm.loc[:, ["response"]], model_name="ddm")


def test_extra_fields_not_found(data_ddm):
    # Case 2: raise error if fields in "extra_fields" are not found in data
    with pytest.raises(ValueError, match=pattern):
        hssm.HSSM(
            data=data_ddm,
            model_name="ddm",
            model_config={"extra_fields": ["extra_field"]},
        )


def test_deadline_no_deadline_field(data_ddm):
    with pytest.raises(
        ValueError,
        match="You have specified that your data has deadline, but "
        + f"`deadline` is not found in your dataset.",
    ):
        hssm.HSSM(data=data_ddm, model="ddm", deadline=True)


def test_invalid_responses(data_ddm):
    data_ddm_miscoded = data_ddm.copy()
    data_ddm_miscoded["response"] = data_ddm_miscoded["response"].replace({-1.0: 0.0})
    with pytest.raises(
        ValueError,
        match=r"Invalid responses found in your dataset: \[0\]",
    ):
        hssm.HSSM(data=data_ddm_miscoded, model="ddm")

    with pytest.raises(
        ValueError,
        match=r"Invalid responses found in your dataset: \[0, 2\]",
    ):
        data_ddm_miscoded = data_ddm.copy()
        data_ddm_miscoded["response"] = np.random.choice([0, 1, 2], data_ddm.shape[0])

        hssm.HSSM(data=data_ddm_miscoded, model="ddm", choices=[1, 2, 3])


def test_circular_response_validation_reaches_distribution_construction(monkeypatch):
    data = np.array([[0.4, -2.75], [0.8, -0.125], [1.2, 1.875]], dtype=np.float64)
    data = pd.DataFrame(data, columns=["rt", "response"])
    observed = {}

    def stop_after_validation(model):
        observed["response"] = model.data["response"].to_numpy(copy=True)
        observed["choices"] = model.choices
        observed["n_choices"] = model.n_choices
        raise RuntimeError("distribution construction reached")

    monkeypatch.setattr(hssm.HSSM, "_make_model_distribution", stop_after_validation)

    def circular_logp(data, v):
        return np.zeros(len(data), dtype=np.float64)

    model_config = {
        "response_kind": "circular",
        "response_bounds": {"response": (-np.pi, np.pi)},
        "list_params": ["v"],
        "bounds": {"v": (-np.inf, np.inf)},
    }
    with pytest.raises(RuntimeError, match="distribution construction reached"):
        hssm.HSSM(
            data=data,
            model="custom",
            model_config=model_config,
            loglik=circular_logp,
            loglik_kind="blackbox",
            p_outlier=None,
            v={"prior": {"name": "Normal", "mu": 0.0, "sigma": 1.0}},
        )

    np.testing.assert_array_equal(observed["response"], data["response"].to_numpy())
    assert observed["choices"] is None
    assert observed["n_choices"] is None


def test_circular_response_validation_precedes_distribution_construction(monkeypatch):
    data = pd.DataFrame({"rt": [0.4, 0.8], "response": [0.0, np.pi]})
    distribution_was_built = False

    def record_distribution_construction(model):
        nonlocal distribution_was_built
        distribution_was_built = True

    monkeypatch.setattr(
        hssm.HSSM, "_make_model_distribution", record_distribution_construction
    )

    with pytest.raises(ValueError, match=r"Circular responses.*must lie in .*\)"):
        hssm.HSSM(
            data=data,
            model="custom",
            model_config={
                "response_kind": "circular",
                "response_bounds": {"response": (-np.pi, np.pi)},
                "list_params": ["v"],
                "bounds": {"v": (-np.inf, np.inf)},
            },
            loglik=lambda data, v: np.zeros(len(data)),
            loglik_kind="blackbox",
            p_outlier=None,
            v={"prior": {"name": "Normal", "mu": 0.0, "sigma": 1.0}},
        )

    assert not distribution_was_built


def test_continuous_choice_only_lapse_requires_categorical_choices():
    data = pd.DataFrame({"response": [-0.5, 0.25]})

    with pytest.raises(ValueError, match="choice-only lapse requires categorical"):
        hssm.HSSM(
            data=data,
            model="custom",
            model_config={
                "response": ("response",),
                "response_kind": "continuous",
                "response_bounds": {"response": (-1.0, 1.0)},
                "list_params": ["v"],
                "bounds": {"v": (-np.inf, np.inf)},
            },
            loglik=lambda data, v: np.zeros(len(data)),
            loglik_kind="blackbox",
            v={"prior": {"name": "Normal", "mu": 0.0, "sigma": 1.0}},
        )


@pytest.mark.slow  # as model needs to be built
def test_missing_responses(data_ddm, caplog):
    data_ddm_miscoded = data_ddm.copy()
    data_ddm_miscoded["response"] = np.random.choice([1], data_ddm.shape[0])

    hssm.HSSM(data=data_ddm_miscoded, model="ddm")

    with pytest.warns(
        UserWarning,
        match=r"You set choices to be \(-1, 1\), but \[-1\] are missing from your dataset\.",
    ):
        hssm.HSSM(data=data_ddm_miscoded, model="ddm")


def test_deadline_missing_data_true(data_ddm, cpn):
    # Case 6: if deadline or missing_data is True, data should contain missing values
    with pytest.raises(
        ValueError,
        match=r"missing_data argument is provided as True, "
        " so RTs of -999.0 are treated as missing. \n"
        "However, you have no RTs of -999.0 in your dataset!",
    ):
        hssm.HSSM(
            data=data_ddm, model="ddm", missing_data=True, loglik_missing_data=cpn
        )


@pytest.mark.slow  # as model needs to be built
def test_deadline_missing_data_false(data_ddm, cpn):
    # AF-TODO: Let's take the save route and let
    # the user delete the data, re below:
    # Case 7: if missing_data and deadline are set to False,
    # then missing data values should be dropped
    missing_size = 10
    missing_indices = np.random.choice(data_ddm.shape[0], missing_size, replace=False)
    data_ddm_missing = data_ddm.copy()
    data_ddm_missing.loc[missing_indices, "rt"] = -999.0

    with pytest.raises(
        ValueError,
        match="Missing data provided as False. \n"
        "However, you have RTs of -999.0 in your dataset!",
    ):
        hssm.HSSM(
            data=data_ddm_missing,
            model="ddm",
            loglik_missing_data=cpn,  # Once on huggingface, this is not needd
        )

    # assert model.data.shape[0] == data_ddm.shape[0] - missing_size

    # AF-TODO: Previously this test had a subset of the missing data
    # set to nan.
    # But the check for `no missing data` and `some rts ==-999.0`
    # flags this with ValueError before the `nan` check is applied.
    data_ddm_missing.loc[missing_indices, "rt"] = np.nan
    with pytest.raises(
        ValueError,
        match="You have NaN response times in your dataset, which is not allowed.",
    ):
        hssm.HSSM(
            data=data_ddm_missing,
            model="ddm",
        )

    # Same logic, here overwriting the nans ALL with -10
    data_ddm_missing.loc[missing_indices, "rt"] = -999.0
    data_ddm_missing.iloc[missing_indices[0], 0] = -10

    with pytest.raises(
        ValueError,
        match="You have negative response times in your dataset, which is not allowed.",
    ):
        hssm.HSSM(
            data=data_ddm_missing,
            model="ddm",
            missing_data=True,
            loglik_missing_data=cpn,
        )

    with pytest.raises(
        ValueError,
        match="You have specified a loglik_missing_data function, but you have not "
        + "set the missing_data or deadline flag to True.",
    ):
        hssm.HSSM(
            data=data_ddm,
            model="ddm",
            loglik_missing_data=cpn,
        )

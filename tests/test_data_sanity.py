from pathlib import Path

import pytest
import numpy as np

import hssm


@pytest.fixture
def cpn():
    return Path(__file__).parent / "fixtures" / "ddm_cpn.onnx"


pattern = r"Field\(s\) `.*` not found in data\."

# The DataValidator class is tested in the test_data_validator.py file, so this file
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


@pytest.mark.slow  # as model needs to be built
def test_missing_responses(data_ddm, caplog):
    data_ddm_miscoded = data_ddm.copy()
    data_ddm_miscoded["response"] = np.random.choice([1], data_ddm.shape[0])

    hssm.HSSM(data=data_ddm_miscoded, model="ddm")

    with pytest.warns(
        UserWarning,
        match=r"You set choices to be \[-1, 1\], but \[-1\] are missing from your dataset\.",
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

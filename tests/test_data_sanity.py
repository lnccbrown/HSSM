from pathlib import Path

import pytest
import numpy as np

import hssm


@pytest.fixture
def cpn():
    return Path(__file__).parent / "fixtures" / "ddm_cpn.onnx"


def test_data_sanity_check(data_ddm, cpn, caplog):
    # Case 1: raise error if there are missing fields in data
    with pytest.raises(ValueError, match="Field rt not found in data."):
        hssm.HSSM(data=data_ddm.loc[:, ["response"]], model_name="ddm")

    # Case 2: raise error if fields in "extra_fields" are not found in data
    with pytest.raises(ValueError, match="Field extra_field not found in data."):
        hssm.HSSM(
            data=data_ddm,
            model_name="ddm",
            model_config={"extra_fields": ["extra_field"]},
        )

    # Case 3: raise error if deadline is set to True, but there is no deadline field
    with pytest.raises(
        ValueError,
        match="You have specified that your data has deadline, but "
        + f"`deadline` is not found in your dataset.",
    ):
        hssm.HSSM(data=data_ddm, model="ddm", deadline=True)

    # Case 4: raise error if hierarchical model is set to True, but there is no
    # participant_id field.
    with pytest.raises(
        ValueError,
        match="You have specified that your model is hierarchical, but "
        + "`participant_id` is not found in your dataset.",
    ):
        hssm.HSSM(data=data_ddm, model="ddm", hierarchical=True)

    # Case 5: raise error if there are invalid responses in data
    with pytest.raises(
        ValueError,
        match=r"Invalid responses found in your dataset: \[0\]",
    ):
        data_ddm_miscoded = data_ddm.copy()
        data_ddm_miscoded["response"] = data_ddm_miscoded["response"].replace(
            {-1.0: 0.0}
        )

        hssm.HSSM(data=data_ddm_miscoded, model="ddm")

    with pytest.raises(
        ValueError,
        match=r"Invalid responses found in your dataset: \[0\]",
    ):
        data_ddm_miscoded = data_ddm.copy()
        data_ddm_miscoded["response"] = np.random.choice([0, 1, 2], data_ddm.shape[0])

        hssm.HSSM(data=data_ddm_miscoded, model="ddm", choices=[1, 2, 3])

    # Case 6: raise warning if there are missing responses in data
    data_ddm_miscoded = data_ddm.copy()
    data_ddm_miscoded["response"] = np.random.choice([1, 2], data_ddm.shape[0])

    hssm.HSSM(data=data_ddm_miscoded, model="ddm", choices=[1, 2, 3])

    assert (
        caplog.records[-1].msg
        == "You set choices to be [1, 2, 3], but [3] are missing from your dataset."
    )

    # Case 7: if deadline or missing_data is True, data should contain missing values
    with pytest.raises(
        ValueError,
        match="You have no missing data in your dataset, "
        + "which is not allowed when `missing_data` or `deadline` is set to "
        + "True.",
    ):
        hssm.HSSM(
            data=data_ddm, model="ddm", missing_data=True, loglik_missing_data=cpn
        )

    # Case 7: if missing_data and deadline are set to False, then missing data values should be dropped
    missing_size = 10
    missing_indices = np.random.choice(data_ddm.shape[0], missing_size, replace=False)
    data_ddm_missing = data_ddm.copy()
    data_ddm_missing.loc[missing_indices, "rt"] = -999.0

    model = hssm.HSSM(
        data=data_ddm_missing,
        model="ddm",
    )

    assert (
        caplog.records[-1].msg
        == "`missing_data` is set to False, but you have missing data in your dataset. Missing data will be dropped."
    )

    assert model.data.shape[0] == data_ddm.shape[0] - missing_size

    data_ddm_missing.iloc[missing_indices[0], 0] = np.nan

    with pytest.raises(
        ValueError,
        match="You have NaN response times in your dataset, which is not allowed.",
    ):
        hssm.HSSM(
            data=data_ddm_missing,
            model="ddm",
        )

        assert (
            caplog.records[-1].msg
            == "`missing_data` is set to False, but you have missing data in your dataset. Missing data will be dropped."
        )

    data_ddm_missing.loc[missing_indices[0], "rt"] = -10

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
            data=data_ddm_missing,
            model="ddm",
            loglik_missing_data=cpn,
        )

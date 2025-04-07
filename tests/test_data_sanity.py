from pathlib import Path

import pytest
import numpy as np

import hssm


@pytest.fixture
def cpn():
    return Path(__file__).parent / "fixtures" / "ddm_cpn.onnx"


@pytest.mark.slow
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
        match=r"Invalid responses found in your dataset: \[0, 2\]",
    ):
        data_ddm_miscoded = data_ddm.copy()
        data_ddm_miscoded["response"] = np.random.choice([0, 1, 2], data_ddm.shape[0])

        hssm.HSSM(data=data_ddm_miscoded, model="ddm", choices=[1, 2, 3])

    # Case 6: raise warning if there are missing responses in data
    data_ddm_miscoded = data_ddm.copy()
    data_ddm_miscoded["response"] = np.random.choice([1], data_ddm.shape[0])

    hssm.HSSM(data=data_ddm_miscoded, model="ddm")

    print("THE CAPLOG RECORDS")
    print([record.msg for record in caplog.records])

    assert "You set choices to be [-1, 1], but [-1] are missing from your dataset." in [
        record.msg % record.args if record.args else record.msg
        for record in caplog.records
    ]

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
        "`missing_data` is set to False, but you have missing data"
        " in your dataset. Missing data will be dropped."
        in [
            record.msg % record.args if record.args else record.msg
            for record in caplog.records
        ]
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
            "`missing_data` is set to False, but you have missing data in your dataset."
            " Missing data will be dropped."
            in [
                record.msg % record.args if record.args else record.msg
                for record in caplog.records
            ]
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

import pytest
import pandas as pd
import numpy as np
from hssm.data_validator import DataValidator


@pytest.fixture
def base_data():
    return pd.DataFrame(
        {
            "rt": [0.5, 0.7, 0.9, 1.1],
            "response": [1, 0, 1, 0],
            "deadline": [1.0, 1.0, 1.0, 1.0],
            "extra": [10, 20, 30, 40],
        }
    )


@pytest.fixture
def dv_instance(base_data):
    return DataValidator(
        data=base_data.copy(),
        response=["rt", "response"],
        choices=[0, 1],
        n_choices=2,
        extra_fields=["extra"],
        deadline=True,
        deadline_name="deadline",
        missing_data=False,
        missing_data_value=-999.0,
    )
    assert isinstance(dv, DataValidator)
    assert dv.data.equals(base_data)
    assert dv.response == ["rt", "response"]
    assert dv.choices == [0, 1]
    assert dv.n_choices == 2
    assert dv.extra_fields == ["extra"]
    assert dv.deadline is True
    assert dv.deadline_name == "deadline"
    assert dv.missing_data is False
    assert dv.missing_data_value == -999.0

import pandas as pd
from hssm import load_data


def test_load_data_valid_dataset():
    dataset_name = "cavanagh_theta"
    loaded_data = load_data(dataset_name)
    assert isinstance(
        loaded_data, pd.DataFrame
    ), f"load_data should return a DataFrame, got {type(loaded_data)}"

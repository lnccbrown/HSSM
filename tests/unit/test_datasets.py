"""Tests for hssm.load_data and hssm.list_data."""

import pandas as pd
import pytest

import hssm
from hssm.datasets import DATASETS


def test_list_data_returns_dataset_names():
    names = hssm.list_data()

    assert isinstance(names, tuple)
    assert set(names) == set(DATASETS)
    assert "cavanagh_theta" in names


def test_load_data_returns_dataframe():
    data = hssm.load_data("cavanagh_theta")

    assert isinstance(data, pd.DataFrame)
    assert not data.empty


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_load_data_loads_every_listed_dataset(dataset):
    assert isinstance(hssm.load_data(dataset), pd.DataFrame)


def test_load_data_raises_on_unknown_dataset():
    with pytest.raises(ValueError, match="not found"):
        hssm.load_data("not_a_dataset")


def test_load_data_requires_an_argument():
    with pytest.raises(TypeError):
        hssm.load_data()

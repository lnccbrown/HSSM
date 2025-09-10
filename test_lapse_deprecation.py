"""Tests for lapse configuration and deprecation handling in the HSSM package.

This module uses pytest to verify correct behavior of LapseConfig,
deprecated interfaces, and related error handling in hssm.
"""

import bambi as bmb
import numpy as np
import pandas as pd
import pytest

import hssm
from hssm import HSSM


@pytest.fixture
def example_data():
    """Return a DataFrame with random RT and response columns for testing."""
    np.random.seed(42)
    n_trials = 100
    return pd.DataFrame(
        {
            "rt": np.abs(np.random.normal(1.0, 0.3, n_trials)),
            "response": np.random.choice([-1, 1], n_trials),
        }
    )


def test_lapseconfig_fixed_probability():
    """Test creating LapseConfig with a fixed lapse probability."""
    config = hssm.LapseConfig(p_outlier=0.05)
    assert config.p_outlier == 0.05
    assert config.lapse_dist is not None


def test_lapseconfig_with_prior():
    """Test creating LapseConfig with bambi Prior objects for both fields."""
    prior = bmb.Prior("Beta", alpha=1, beta=19)
    lapse_dist = bmb.Prior("Uniform", lower=0.0, upper=10.0)
    config = hssm.LapseConfig(p_outlier=prior, lapse_dist=lapse_dist)
    assert config.p_outlier == prior
    assert config.lapse_dist == lapse_dist


def test_lapseconfig_none():
    """Test creating LapseConfig with p_outlier=0 disables lapse modeling."""
    config = hssm.LapseConfig(p_outlier=0.0)
    assert config.p_outlier == 0.0


def test_deprecated_interface_warns():
    """Test that using the deprecated p_outlier parameter triggers a warning."""
    hssm_instance = HSSM.__new__(HSSM)
    with pytest.warns(DeprecationWarning):
        processed_config = hssm_instance._process_lapse_configuration(
            lapse_config=None,
            p_outlier=0.1,
            lapse=bmb.Prior("Uniform", lower=0.0, upper=20.0),
        )
        assert isinstance(processed_config, hssm.LapseConfig)
        assert processed_config.p_outlier == 0.1


def test_lapseconfig_invalid_values():
    """Test that invalid p_outlier values raise ValueError."""
    with pytest.raises(ValueError):
        hssm.LapseConfig(p_outlier=1.5)
    with pytest.raises(ValueError):
        hssm.LapseConfig(p_outlier=-0.1)


def test_lapseconfig_dictionary_interface():
    """Test a dictionary can be used as a lapse_config and is converted properly."""
    hssm_instance = HSSM.__new__(HSSM)
    dict_config = {
        "p_outlier": 0.03,
        "lapse_dist": bmb.Prior("Uniform", lower=0, upper=15),
    }
    processed_config = hssm_instance._process_lapse_configuration(
        lapse_config=dict_config,
        p_outlier=None,
        lapse=None,
    )
    assert isinstance(processed_config, hssm.LapseConfig)
    assert processed_config.p_outlier == 0.03

"""
Test suite for HSSM lapse configuration interface.

Tests both the new LapseConfig dataclass interface and backward compatibility
with deprecated p_outlier/lapse parameters.
"""

import pytest
import warnings
import pandas as pd
import numpy as np
import bambi as bmb
from hssm import LapseConfig
from hssm.hssm import HSSM


class TestLapseConfig:
    """Test the new LapseConfig dataclass interface."""

    def test_create_lapse_config_with_fixed_probability(self) -> None:
        """Test creating LapseConfig with a fixed lapse probability."""
        config = LapseConfig(p_outlier=0.05)
        assert config.p_outlier == 0.05
        # lapse_dist gets a default value in __post_init__
        assert config.lapse_dist is not None
        assert "p_outlier=0.05" in str(config)

    def test_create_lapse_config_with_prior(self) -> None:
        """Test creating LapseConfig with bambi Prior objects."""
        p_outlier_prior = bmb.Prior("Beta", alpha=1, beta=19)
        lapse_dist_prior = bmb.Prior("Uniform", lower=0.0, upper=10.0)

        config = LapseConfig(p_outlier=p_outlier_prior, lapse_dist=lapse_dist_prior)
        assert config.p_outlier == p_outlier_prior
        assert config.lapse_dist == lapse_dist_prior

    def test_create_lapse_config_with_only_p_outlier(self) -> None:
        """Test creating LapseConfig with only p_outlier (lapse_dist gets default)."""
        config = LapseConfig(p_outlier=0.02)
        assert config.p_outlier == 0.02
        # lapse_dist gets a default value in __post_init__
        assert config.lapse_dist is not None

    def test_create_lapse_config_with_explicit_none_lapse_dist(self) -> None:
        """Test that explicitly passing None for lapse_dist still gets default."""
        config = LapseConfig(p_outlier=0.03, lapse_dist=None)
        assert config.p_outlier == 0.03
        # Even when explicitly set to None, gets default in __post_init__
        assert config.lapse_dist is not None

    def test_validate_p_outlier_range(self) -> None:
        """Test that p_outlier validation works correctly."""
        # Valid values should work
        LapseConfig(p_outlier=0.0)  # Boundary case
        LapseConfig(p_outlier=0.5)  # Middle value
        LapseConfig(p_outlier=1.0)  # Boundary case

        # Invalid values should raise ValueError
        with pytest.raises(ValueError, match="p_outlier must be between 0 and 1"):
            LapseConfig(p_outlier=1.5)

        with pytest.raises(ValueError, match="p_outlier must be between 0 and 1"):
            LapseConfig(p_outlier=-0.1)

    def test_validate_p_outlier_type(self) -> None:
        """Test that p_outlier type validation works correctly."""
        # Valid types should work
        LapseConfig(p_outlier=0.05)  # float
        LapseConfig(p_outlier=bmb.Prior("Beta", alpha=1, beta=9))  # Prior

        # Invalid types should raise TypeError
        with pytest.raises(TypeError, match="p_outlier must be a float or bambi.Prior"):
            LapseConfig(p_outlier="0.05")  # string

        with pytest.raises(TypeError, match="p_outlier must be a float or bambi.Prior"):
            LapseConfig(p_outlier=[0.05])  # list


class TestLapseConfigBackwardCompatibility:
    """Test backward compatibility with deprecated interface."""

    def test_deprecated_interface_warns(self) -> None:
        """Test that using deprecated p_outlier parameter triggers warning."""
        hssm_instance = HSSM.__new__(HSSM)  # Create instance without calling __init__

        with pytest.warns(
            DeprecationWarning, match="p_outlier.*lapse.*parameters are deprecated"
        ):
            processed_config = hssm_instance._process_lapse_configuration(
                lapse_config=None, p_outlier=0.1, lapse=None
            )

        assert isinstance(processed_config, LapseConfig)
        assert processed_config.p_outlier == 0.1

    def test_deprecated_interface_with_both_params(self) -> None:
        """Test deprecated interface with both p_outlier and lapse."""
        hssm_instance = HSSM.__new__(HSSM)
        lapse_prior = bmb.Prior("Uniform", lower=0.0, upper=20.0)

        with pytest.warns(DeprecationWarning):
            processed_config = hssm_instance._process_lapse_configuration(
                lapse_config=None, p_outlier=0.1, lapse=lapse_prior
            )

        assert isinstance(processed_config, LapseConfig)
        assert processed_config.p_outlier == 0.1
        assert processed_config.lapse_dist == lapse_prior

    def test_new_interface_no_warning(self) -> None:
        """Test that new interface does not trigger warnings."""
        hssm_instance = HSSM.__new__(HSSM)
        lapse_config = LapseConfig(p_outlier=0.05)

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors

            processed_config = hssm_instance._process_lapse_configuration(
                lapse_config=lapse_config, p_outlier=None, lapse=None
            )

        assert processed_config == lapse_config

    def test_conflicting_interfaces_raises_error(self) -> None:
        """Test that using both old and new interfaces raises an error."""
        hssm_instance = HSSM.__new__(HSSM)
        lapse_config = LapseConfig(p_outlier=0.05)

        with pytest.raises(
            ValueError, match="Cannot specify both lapse_config and deprecated"
        ):
            hssm_instance._process_lapse_configuration(
                lapse_config=lapse_config, p_outlier=0.1, lapse=None
            )

    def test_dictionary_interface_support(self) -> None:
        """Test that dictionary interface is supported."""
        hssm_instance = HSSM.__new__(HSSM)
        dict_config = {
            "p_outlier": 0.03,
            "lapse_dist": bmb.Prior("Uniform", lower=0, upper=15),
        }

        processed_config = hssm_instance._process_lapse_configuration(
            lapse_config=dict_config, p_outlier=None, lapse=None
        )

        assert isinstance(processed_config, LapseConfig)
        assert processed_config.p_outlier == 0.03
        assert processed_config.lapse_dist is not None

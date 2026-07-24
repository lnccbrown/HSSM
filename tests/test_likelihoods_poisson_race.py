"""Integration-style tests for poisson_race simulator behavior overrides."""

import numpy as np
import pandas as pd
import pytest
from ssms.config import model_config

import hssm
from hssm.simulator import simulate_data

hssm.set_floatX("float32")


@pytest.mark.skipif(
    "poisson_race" not in model_config,
    reason="poisson_race not available in installed ssms",
)
class TestPoissonRaceSimulator:
    """Tests for smooth_unif override behavior in simulate_data."""

    theta_poisson_race = dict(r1=2.5, r2=3.5, k1=1.3, k2=1.6, t=0.05)

    def test_smooth_unif_defaults_to_false(self):
        """Default poisson_race simulation should match smooth_unif=False."""
        df_default = simulate_data(
            "poisson_race",
            theta=self.theta_poisson_race,
            size=20,
            random_state=42,
        )
        df_explicit_false = simulate_data(
            "poisson_race",
            theta=self.theta_poisson_race,
            size=20,
            random_state=42,
            smooth_unif=False,
        )
        df_explicit_true = simulate_data(
            "poisson_race",
            theta=self.theta_poisson_race,
            size=20,
            random_state=42,
            smooth_unif=True,
        )

        assert isinstance(df_default, pd.DataFrame)
        assert len(df_default) == 20
        assert set(df_default.columns) == {"rt", "response"}
        np.testing.assert_allclose(
            df_default["rt"].to_numpy(),
            df_explicit_false["rt"].to_numpy(),
        )
        assert not np.allclose(
            df_default["rt"].to_numpy(),
            df_explicit_true["rt"].to_numpy(),
        )

    def test_smooth_unif_explicit_override_respected(self):
        """An explicit smooth_unif=True should alter RTs for the same seed."""
        df_default = simulate_data(
            "poisson_race",
            theta=self.theta_poisson_race,
            size=20,
            random_state=42,
            smooth_unif=False,
        )
        df_explicit_true = simulate_data(
            "poisson_race",
            theta=self.theta_poisson_race,
            size=20,
            random_state=42,
            smooth_unif=True,
        )

        assert isinstance(df_explicit_true, pd.DataFrame)
        assert len(df_explicit_true) == 20
        assert set(df_explicit_true.columns) == {"rt", "response"}
        assert not np.allclose(
            df_default["rt"].to_numpy(),
            df_explicit_true["rt"].to_numpy(),
        )

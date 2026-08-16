"""End-to-end construction tests for the JEAM model registration."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jeam")

import hssm
from hssm.integrations.jeam import simulate_circular_diffusion


def test_simulated_circular_data_builds_an_ordinary_hssm_model():
    """The fixed CDM should use HSSM's standard config-driven lifecycle."""
    observations = simulate_circular_diffusion(
        theta=np.array([0.70, -0.35, 0.40, 0.08]),
        random_state=1947,
        n_replicas=12,
    )
    data = pd.DataFrame(observations, columns=["rt", "response"])

    model = hssm.HSSM(
        data=data,
        model="circular_diffusion",
        p_outlier=None,
    )

    assert type(model) is hssm.HSSM
    assert model.model_name == "circular_diffusion"
    assert model.response_kind == "circular"
    assert model.response_bounds == {"response": (-np.pi, np.pi)}
    assert model.choices is None
    assert model.n_choices is None
    assert model.list_params == ["v_x", "v_y", "a", "t"]
    assert model.model_config.rv is simulate_circular_diffusion

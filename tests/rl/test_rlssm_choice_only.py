"""Tests for the choice-only (softmax) RLSSM model.

Covers the 2AB_RescorlaWagner_Softmax model: config validation,
instantiation without an RT column, and PyMC model construction.
"""

from collections.abc import Generator

import numpy as np
import pandas as pd
import pytensor
import pytest

import hssm
from hssm.rl import RLSSM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _set_floatx_float32() -> Generator[None, None, None]:
    """Ensure float32 is used for this module's tests, then restore previous setting."""
    prev_floatx = pytensor.config.floatX
    hssm.set_floatX("float32", update_jax=True)
    try:
        yield
    finally:
        hssm.set_floatX(prev_floatx, update_jax=True)


@pytest.fixture(scope="module")
def choice_only_data() -> pd.DataFrame:
    """Synthetic balanced-panel choice-only dataset (no RT column).

    5 participants × 20 trials each, choices ∈ {0, 1}, feedback ∈ {0.0, 1.0}.
    """
    rng = np.random.default_rng(42)
    n_participants, n_trials = 5, 20
    n_total = n_participants * n_trials
    return pd.DataFrame(
        {
            "participant_id": np.repeat(np.arange(n_participants), n_trials),
            "response": rng.integers(0, 2, size=n_total).astype(float),
            "feedback": rng.integers(0, 2, size=n_total).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# Choice-only (softmax) model tests
# ---------------------------------------------------------------------------


def test_rlssm_softmax_config() -> None:
    """2AB_RescorlaWagner_Softmax config should be choice-only with correct params."""
    from hssm.rl.registry import get_rlssm_model_config

    cfg = get_rlssm_model_config("2AB_RescorlaWagner_Softmax")
    assert cfg.is_choice_only
    assert cfg.response == ["response"]
    assert cfg.choices == [0, 1]
    assert cfg.list_params == ["rl_alpha", "beta"]
    assert "rl_alpha" in cfg.bounds
    assert "beta" in cfg.bounds
    assert cfg.extra_fields == ["feedback"]


def test_rlssm_softmax_instantiate(choice_only_data) -> None:
    """2AB_RescorlaWagner_Softmax should instantiate without error."""
    model = RLSSM(data=choice_only_data, model="2AB_RescorlaWagner_Softmax")
    assert isinstance(model, RLSSM)
    assert model.model_config.is_choice_only
    assert "rl_alpha" in model.params
    assert "beta" in model.params
    assert "q_diff" not in model.params  # computed inside Op, not a free param


def test_rlssm_softmax_no_rt_column(choice_only_data) -> None:
    """Choice-only RLSSM should not require an 'rt' column in the data."""
    assert "rt" not in choice_only_data.columns
    model = RLSSM(data=choice_only_data, model="2AB_RescorlaWagner_Softmax")
    assert model.model is not None


def test_rlssm_softmax_pymc_model(choice_only_data) -> None:
    """pymc_model should be accessible after softmax model construction."""
    model = RLSSM(data=choice_only_data, model="2AB_RescorlaWagner_Softmax")
    assert model.pymc_model is not None

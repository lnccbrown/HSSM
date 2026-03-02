"""Tests for the RLSSM class.

Mirrors the structure of tests/test_hssm.py, covering initialisation,
config validation, param keys, balanced-panel enforcement, the no-lapse
variant, bambi / PyMC model construction, and a sampling smoke test.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import hssm
from hssm import RLSSM, RLSSMConfig
from hssm.rl.likelihoods.two_armed_bandit import compute_v_subject_wise
from hssm.utils import annotate_function

hssm.set_floatX("float32", update_jax=True)

# ---------------------------------------------------------------------------
# Module-level annotated helpers (shared by all tests)
# ---------------------------------------------------------------------------

# Annotate the RL learning function: maps
#   (rl_alpha, scaler, response, feedback) -> v
_compute_v_annotated = annotate_function(
    inputs=["rl_alpha", "scaler", "response", "feedback"],
    outputs=["v"],
)(compute_v_subject_wise)


# Annotated SSM log-likelihood function (simplified for testing).
# It receives a 2-D lan_matrix whose columns correspond to
#   [v, a, z, t, theta, rt, response]
# and returns per-trial log-probabilities of shape (n_total_trials, 1).
@annotate_function(
    inputs=["v", "a", "z", "t", "theta", "rt", "response"],
    outputs=["logp"],
    computed={"v": _compute_v_annotated},
)
def _dummy_ssm_logp(lan_matrix: jnp.ndarray) -> jnp.ndarray:
    """Return per-trial log-probabilities (column-sum); structural tests only."""
    # Return 1D (N,) — PyTensor declares the Op output as pt.vector(), so
    # gradients arrive as (N,). A (N,1) return causes a VJP shape mismatch.
    return jnp.sum(lan_matrix, axis=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rldm_data() -> pd.DataFrame:
    """Load the RLDM fixture dataset (balanced panel)."""
    raw = np.load("tests/fixtures/rldm_data.npy", allow_pickle=True).item()
    return pd.DataFrame(raw["data"])


@pytest.fixture(scope="module")
def rlssm_config() -> RLSSMConfig:
    """Minimal but valid RLSSMConfig for the RLDM fixture dataset."""
    return RLSSMConfig(
        model_name="rldm_test",
        loglik_kind="approx_differentiable",
        decision_process="angle",
        decision_process_loglik_kind="approx_differentiable",
        learning_process_loglik_kind="blackbox",
        list_params=["rl_alpha", "scaler", "a", "theta", "t", "z"],
        params_default=[0.1, 1.0, 1.0, 0.0, 0.3, 0.5],
        bounds={
            "rl_alpha": (0.0, 1.0),
            "scaler": (0.0, 10.0),
            "a": (0.1, 3.0),
            "theta": (-0.1, 0.1),
            "t": (0.001, 1.0),
            "z": (0.1, 0.9),
        },
        learning_process={"v": _compute_v_annotated},
        response=["rt", "response"],
        choices=[0, 1],
        extra_fields=["feedback"],
        ssm_logp_func=_dummy_ssm_logp,
    )


# ---------------------------------------------------------------------------
# Initialisation & config-validation tests
# ---------------------------------------------------------------------------


def test_rlssm_init(rldm_data: pd.DataFrame, rlssm_config: RLSSMConfig) -> None:
    """Basic RLSSM initialisation should succeed and return an RLSSM instance."""
    model = RLSSM(data=rldm_data, rlssm_config=rlssm_config)
    assert isinstance(model, RLSSM)
    assert model.model_name == "rldm_test"


def test_rlssm_panel_attrs(rldm_data: pd.DataFrame, rlssm_config: RLSSMConfig) -> None:
    """_n_participants and _n_trials should match the fixture data structure."""
    model = RLSSM(data=rldm_data, rlssm_config=rlssm_config)

    n_participants = rldm_data["participant_id"].nunique()
    n_trials = len(rldm_data) // n_participants

    assert model._n_participants == n_participants
    assert model._n_trials == n_trials


def test_rlssm_params_keys(rldm_data: pd.DataFrame, rlssm_config: RLSSMConfig) -> None:
    """model.params should contain exactly list_params + p_outlier."""
    model = RLSSM(data=rldm_data, rlssm_config=rlssm_config)
    expected = set(rlssm_config.list_params) | {"p_outlier"}
    assert set(model.params.keys()) == expected


def test_rlssm_unbalanced_raises(
    rldm_data: pd.DataFrame, rlssm_config: RLSSMConfig
) -> None:
    """Dropping one row should make the panel unbalanced → ValueError."""
    unbalanced = rldm_data.iloc[:-1].copy()
    with pytest.raises(ValueError, match="balanced panels"):
        RLSSM(data=unbalanced, rlssm_config=rlssm_config)


def test_rlssm_nan_participant_id_raises(
    rldm_data: pd.DataFrame, rlssm_config: RLSSMConfig
) -> None:
    """NaN in participant_id column should raise ValueError before groupby silently drops rows."""
    nan_data = rldm_data.copy()
    nan_data.loc[nan_data.index[0], "participant_id"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        RLSSM(data=nan_data, rlssm_config=rlssm_config)


def test_rlssm_missing_ssm_logp_func_raises(
    rldm_data: pd.DataFrame, rlssm_config: RLSSMConfig
) -> None:
    """RLSSMConfig without ssm_logp_func should raise ValueError on init."""
    bad_config = RLSSMConfig(
        model_name="rldm_bad",
        loglik_kind="approx_differentiable",
        decision_process="angle",
        decision_process_loglik_kind="approx_differentiable",
        learning_process_loglik_kind="blackbox",
        list_params=rlssm_config.list_params,
        params_default=rlssm_config.params_default,
        bounds=rlssm_config.bounds,
        learning_process=rlssm_config.learning_process,
        response=list(rlssm_config.response),
        choices=list(rlssm_config.choices),
        extra_fields=list(rlssm_config.extra_fields),
        # ssm_logp_func intentionally omitted → defaults to None
    )
    with pytest.raises(ValueError, match="ssm_logp_func"):
        RLSSM(data=rldm_data, rlssm_config=bad_config)


def test_rlssm_unannotated_ssm_logp_func_raises(
    rldm_data: pd.DataFrame, rlssm_config: RLSSMConfig
) -> None:
    """A plain callable without @annotate_function attrs should raise ValueError."""
    bad_config = RLSSMConfig(
        model_name="rldm_bad",
        loglik_kind="approx_differentiable",
        decision_process="angle",
        decision_process_loglik_kind="approx_differentiable",
        learning_process_loglik_kind="blackbox",
        list_params=rlssm_config.list_params,
        params_default=rlssm_config.params_default,
        bounds=rlssm_config.bounds,
        learning_process=rlssm_config.learning_process,
        response=list(rlssm_config.response),
        choices=list(rlssm_config.choices),
        extra_fields=list(rlssm_config.extra_fields),
        ssm_logp_func=lambda x: x,  # callable but no .inputs/.outputs/.computed
    )
    with pytest.raises(ValueError, match="annotate_function"):
        RLSSM(data=rldm_data, rlssm_config=bad_config)


def test_rlssm_missing_data_raises(
    rldm_data: pd.DataFrame, rlssm_config: RLSSMConfig
) -> None:
    """Passing missing_data!=False should raise ValueError with 'missing_data' in msg."""
    with pytest.raises(ValueError, match="missing_data"):
        RLSSM(data=rldm_data, rlssm_config=rlssm_config, missing_data=True)


def test_rlssm_deadline_raises(
    rldm_data: pd.DataFrame, rlssm_config: RLSSMConfig
) -> None:
    """Passing deadline!=False should raise ValueError with 'deadline' in msg."""
    with pytest.raises(ValueError, match="deadline"):
        RLSSM(data=rldm_data, rlssm_config=rlssm_config, deadline=True)


# ---------------------------------------------------------------------------
# Model-structure tests
# ---------------------------------------------------------------------------


def test_rlssm_no_lapse(rldm_data: pd.DataFrame, rlssm_config: RLSSMConfig) -> None:
    """Setting p_outlier=None should remove p_outlier from params."""
    model = RLSSM(data=rldm_data, rlssm_config=rlssm_config, p_outlier=None)
    assert "p_outlier" not in model.params


def test_rlssm_model_built(rldm_data: pd.DataFrame, rlssm_config: RLSSMConfig) -> None:
    """The bambi model should be built and the computed param 'v' absent from params."""
    model = RLSSM(data=rldm_data, rlssm_config=rlssm_config)
    assert model.model is not None
    # rl_alpha is a free (sampled) parameter
    assert "rl_alpha" in model.params
    # v is computed inside the Op; it must NOT appear as a free parameter
    assert "v" not in model.params


def test_rlssm_pymc_model(rldm_data: pd.DataFrame, rlssm_config: RLSSMConfig) -> None:
    """pymc_model should be accessible after model construction."""
    model = RLSSM(data=rldm_data, rlssm_config=rlssm_config)
    assert model.pymc_model is not None


# ---------------------------------------------------------------------------
# Slow sampling smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_rlssm_sample_smoke(rldm_data: pd.DataFrame, rlssm_config: RLSSMConfig) -> None:
    """Minimal sampling run should return an InferenceData object."""
    model = RLSSM(data=rldm_data, rlssm_config=rlssm_config)
    trace = model.sample(draws=2, tune=2, chains=1, cores=1)
    assert trace is not None

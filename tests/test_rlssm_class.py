"""Tests for RLSSM class construction and metadata requirements."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from hssm.config import RLSSMConfig
from hssm.rlssm import RLSSM
from hssm.utils import annotate_function
from hssm.rl.likelihoods.two_armed_bandit import compute_v_subject_wise as bandit_v


compute_v = annotate_function(
    inputs=["rl_alpha", "scaler", "response", "feedback"], outputs=["v"]
)(bandit_v)


@annotate_function(inputs=["v", "a", "rt", "response"])
def decision_logp(lan_matrix: jnp.ndarray) -> jnp.ndarray:
    """Simple zero logp; relies on builder to feed computed `v` and `a` plus data."""
    return jnp.zeros(lan_matrix.shape[0])


@pytest.fixture
def dummy_data() -> pd.DataFrame:
    n_participants = 2
    n_trials = 3
    rts = np.linspace(0.3, 0.9, n_participants * n_trials)
    responses = np.tile([0, 1, 0], n_participants)
    participant_ids = np.repeat(np.arange(n_participants), n_trials)
    feedback = np.ones_like(responses)
    trial_id = np.tile(np.arange(n_trials), n_participants)
    rl_alpha = np.full_like(responses, 0.5, dtype=float)
    scaler = np.full_like(responses, 1.2, dtype=float)
    a = np.full_like(responses, 1.0, dtype=float)
    return pd.DataFrame(
        {
            "rt": rts,
            "response": responses,
            "participant_id": participant_ids,
            "feedback": feedback,
            "trial_id": trial_id,
            "rl_alpha": rl_alpha,
            "scaler": scaler,
            "a": a,
        }
    )


@pytest.fixture
def rlssm_config() -> RLSSMConfig:
    return RLSSMConfig(
        model_name="dummy_rlssm",
        description="Dummy RLSSM for testing",
        list_params=["rl_alpha", "scaler", "a"],
        params_default=[0.5, 1.2, {"name": "Normal", "mu": 1.0, "sigma": 0.5}],
        bounds={"rl_alpha": (0.0, 1.0), "scaler": (0.1, 3.0), "a": (0.1, 2.0)},
        response=["rt", "response"],
        choices=(0, 1),
        extra_fields=["participant_id", "feedback", "trial_id"],
        decision_process="dummy_decision",
        learning_process={"v": compute_v},
        decision_process_loglik_kind="approx_differentiable",
        learning_process_loglik_kind="analytical",
    )


class TestRLSSMClassInstantiation:
    def test_rlssm_instantiation_smoke(
        self, dummy_data: pd.DataFrame, rlssm_config: RLSSMConfig
    ):
        """RLSSM builds with annotated decision logp and learning_process compute func."""
        model = RLSSM(
            data=dummy_data,
            model="dummy_rlssm",
            model_config=rlssm_config,
            loglik=decision_logp,
            loglik_kind="approx_differentiable",
        )

        assert hasattr(model, "model_distribution")
        assert model.model_distribution is not None

    def test_rlssm_requires_inputs_metadata(
        self, dummy_data: pd.DataFrame, rlssm_config: RLSSMConfig
    ):
        """Unannotated decision logp should raise a clear metadata error."""

        def unannotated_logp(lan_matrix):  # pragma: no cover - simple stub
            return jnp.zeros(lan_matrix.shape[0])

        with pytest.raises(
            ValueError,
            match="requires the decision-process log-likelihood to declare `inputs`",
        ):
            RLSSM(
                data=dummy_data,
                model="dummy_rlssm",
                model_config=rlssm_config,
                loglik=unannotated_logp,
                loglik_kind="approx_differentiable",
            )

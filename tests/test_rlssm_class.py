"""Tests for RLSSM class construction and metadata requirements."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from hssm.config import RLSSMConfig
from hssm.rl.likelihoods.two_armed_bandit import (
    compute_v_subject_wise as bandit_v,
)
from hssm.rlssm import RLSSM
from hssm.utils import annotate_function

compute_v = annotate_function(
    inputs=["rl_alpha", "scaler", "response", "feedback"], outputs=["v"]
)(bandit_v)


@annotate_function(inputs=["v", "a", "rt", "response"])
def decision_logp(lan_matrix: jnp.ndarray) -> jnp.ndarray:
    """Return zero logp; builder feeds computed `v` and `a` plus data."""
    return jnp.zeros(lan_matrix.shape[0])


@pytest.fixture
def dummy_data() -> pd.DataFrame:
    """Provide small dummy RLSSM dataset."""
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
    """Provide baseline RLSSMConfig used across tests."""
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


class TestRLSSMInstantiation:
    """Instantiation smoke coverage for RLSSM."""

    def test_rlssm_instantiation_smoke(
        self, dummy_data: pd.DataFrame, rlssm_config: RLSSMConfig
    ):
        """Build with annotated decision logp and learning compute func."""
        data_with_v = dummy_data.copy()
        data_with_v["v"] = 0.1

        config_pt = replace(
            rlssm_config,
            backend="pytensor",
            loglik_kind="analytical",
            learning_process={},
            extra_fields=["participant_id", "feedback", "trial_id", "v"],
        )

        model = RLSSM(
            data=data_with_v,
            model="dummy_rlssm",
            model_config=config_pt,
            loglik=decision_logp,
            loglik_kind="analytical",
        )

        assert hasattr(model, "model_distribution")
        assert model.model_distribution is not None


class TestRLSSMMetadataValidation:
    """Validate metadata requirements for RLSSM construction."""

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

    def test_computed_param_must_be_in_loglik_inputs(
        self, dummy_data: pd.DataFrame, rlssm_config: RLSSMConfig
    ):
        """Require decision logp inputs to include computed parameters."""

        @annotate_function(inputs=["a", "rt", "response"])
        def logp_missing_v(lan_matrix: jnp.ndarray) -> jnp.ndarray:
            return jnp.zeros(lan_matrix.shape[0])

        with pytest.raises(ValueError, match="Computed parameters must be included"):
            RLSSM(
                data=dummy_data,
                model="dummy_rlssm",
                model_config=rlssm_config,
                loglik=logp_missing_v,
                loglik_kind="approx_differentiable",
            )

    def test_learning_process_requires_inputs(
        self, dummy_data: pd.DataFrame, rlssm_config: RLSSMConfig
    ):
        """Learning-process callables must declare inputs metadata."""

        def compute_v_unannotated(
            rl_alpha: jnp.ndarray,
            scaler: jnp.ndarray,
            response: jnp.ndarray,
            feedback: jnp.ndarray,
        ) -> jnp.ndarray:
            return compute_v(rl_alpha, scaler, response, feedback)

        config_missing_inputs = replace(
            rlssm_config, learning_process={"v": compute_v_unannotated}
        )

        with pytest.raises(ValueError, match="must be annotated with `inputs`"):
            RLSSM(
                data=dummy_data,
                model="dummy_rlssm",
                model_config=config_missing_inputs,
                loglik=decision_logp,
                loglik_kind="approx_differentiable",
            )


class TestRLSSMDataValidation:
    """Validate data-shape expectations enforced by the RL builder."""

    def test_requires_participant_id_extra_field(
        self, dummy_data: pd.DataFrame, rlssm_config: RLSSMConfig
    ):
        """extra_fields must include participant_id for RL builder."""
        config_no_pid = replace(rlssm_config, extra_fields=["feedback", "trial_id"])

        with pytest.raises(ValueError, match="participant_id.*extra_fields"):
            RLSSM(
                data=dummy_data,
                model="dummy_rlssm",
                model_config=config_no_pid,
                loglik=decision_logp,
                loglik_kind="approx_differentiable",
            )

    def test_requires_uniform_trial_counts(
        self, dummy_data: pd.DataFrame, rlssm_config: RLSSMConfig
    ):
        """All participants must share a uniform trial count."""
        # Drop one trial for the last participant to break uniformity
        unbalanced = dummy_data.drop(
            dummy_data[dummy_data["participant_id"] == 1].index[-1]
        )

        with pytest.raises(ValueError, match="same number of trials"):
            RLSSM(
                data=unbalanced,
                model="dummy_rlssm",
                model_config=rlssm_config,
                loglik=decision_logp,
                loglik_kind="approx_differentiable",
            )


class TestRLSSMSampling:
    """Integration smoke coverage for RLSSM sampling."""

    @pytest.mark.slow
    def test_rlssm_sampling_smoke(
        self, dummy_data: pd.DataFrame, rlssm_config: RLSSMConfig
    ):
        """Integration smoke test: sampling succeeds with RLSSM."""
        data_with_v = dummy_data.copy()
        data_with_v["v"] = 0.1

        config_pt = replace(
            rlssm_config,
            backend="pytensor",
            learning_process={},
            extra_fields=["participant_id", "feedback", "trial_id", "v"],
        )

        model = RLSSM(
            data=data_with_v,
            model="dummy_rlssm",
            model_config=config_pt,
            loglik=decision_logp,
            loglik_kind="analytical",
        )

        idata = model.sample(draws=5, tune=0, chains=1, cores=1)

        assert idata.posterior.data_vars
        assert "a" in idata.posterior.data_vars

    @pytest.mark.slow
    def test_rlssm_sampling_jax_backend(
        self, dummy_data: pd.DataFrame, rlssm_config: RLSSMConfig
    ):
        """Sampling works with the default (JAX) backend path."""
        data_with_v = dummy_data.copy()
        data_with_v["v"] = 0.1

        config_jax = replace(
            rlssm_config,
            backend=None,
            learning_process={},
            extra_fields=["participant_id", "feedback", "trial_id", "v"],
            loglik_kind="analytical",
        )

        model = RLSSM(
            data=data_with_v,
            model="dummy_rlssm",
            model_config=config_jax,
            loglik=decision_logp,
            loglik_kind="analytical",
        )

        idata = model.sample(draws=5, tune=0, chains=1, cores=1)

        assert idata.posterior.data_vars
        assert "a" in idata.posterior.data_vars

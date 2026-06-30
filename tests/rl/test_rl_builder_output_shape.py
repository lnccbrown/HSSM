import numpy as np
import jax.numpy as jnp
import pytest

from hssm.rl.likelihoods.builder import make_rl_logp_func
from hssm.utils import annotate_function


@annotate_function(
    inputs=["v", "a", "z", "t", "rt", "response"], outputs=["logp"], computed={}
)
def dummy_ssm_logp(lan_matrix: jnp.ndarray) -> np.ndarray:
    summed = jnp.sum(lan_matrix, axis=1)
    return np.asarray(summed).reshape((-1, 1))


@annotate_function(inputs=["rl.alpha", "scaler", "response", "feedback"], outputs=["v"])  # type: ignore[arg-type]
def compute_v_subject_wise(subj_trials: jnp.ndarray) -> jnp.ndarray:
    rl_alpha_col, scaler_col, response_col, feedback_col = (
        subj_trials[:, 0],
        subj_trials[:, 1],
        subj_trials[:, 2],
        subj_trials[:, 3],
    )
    return (feedback_col - response_col) * scaler_col


@annotate_function(
    inputs=["v", "a", "z", "t", "rt", "response"],
    outputs=["logp"],
    computed={"v": compute_v_subject_wise},
)
def ssm_logp_with_v(lan_matrix: jnp.ndarray) -> np.ndarray:
    summed = jnp.sum(lan_matrix, axis=1)
    return np.asarray(summed).reshape((-1, 1))


@annotate_function(inputs=["bonus", "response"], outputs=["theta"])  # type: ignore[arg-type]
def compute_theta_subject_wise(subj_trials: jnp.ndarray) -> jnp.ndarray:
    bonus_col, response_col = subj_trials[:, 0], subj_trials[:, 1]
    return jnp.abs(bonus_col * (response_col + 1)) + 0.01


@annotate_function(
    inputs=["v", "theta", "a", "z", "t", "rt", "response"],
    outputs=["logp"],
    computed={"v": compute_v_subject_wise, "theta": compute_theta_subject_wise},
)
def ssm_logp_with_v_and_theta(lan_matrix: jnp.ndarray) -> np.ndarray:
    summed = jnp.sum(lan_matrix, axis=1)
    return np.asarray(summed).reshape((-1, 1))


class TestRLBuilderOutputShape:
    @staticmethod
    def _gen_data(n_participants: int, n_trials: int, cols: list[str]) -> np.ndarray:
        total = n_participants * n_trials
        col_map = {
            "rt": np.random.rand(total),
            "response": np.random.randint(0, 2, size=total),
            "feedback": np.random.randint(0, 2, size=total),
        }
        return np.column_stack([col_map[c] for c in cols])

    @staticmethod
    def _gen_params(
        n_participants: int, n_trials: int, names: list[str]
    ) -> list[np.ndarray]:
        total = n_participants * n_trials
        param_map = {
            "v": np.random.randn(total),
            "a": np.abs(np.random.randn(total)) + 1.0,
            "z": np.random.rand(total),
            "t": np.abs(np.random.randn(total)) + 0.1,
            "rl.alpha": np.full(total, 0.2),
            "scaler": np.full(total, 1.0),
            "bonus": np.full(total, 0.25),
        }
        return [param_map[p] for p in names]

    @staticmethod
    def _assert_2d_shape(out: np.ndarray, n_participants: int, n_trials: int) -> None:
        assert isinstance(out, np.ndarray)
        assert out.ndim == 2
        assert out.shape == (n_participants * n_trials, 1)

    def test_make_rl_logp_func_output_is_2d_array(self):
        n_participants, n_trials = 3, 5
        data = self._gen_data(n_participants, n_trials, ["rt", "response"])
        v, a, z, t = self._gen_params(n_participants, n_trials, ["v", "a", "z", "t"])
        logp_fn = make_rl_logp_func(
            ssm_logp_func=dummy_ssm_logp,
            n_participants=n_participants,
            n_trials=n_trials,
            data_cols=["rt", "response"],
            list_params=["v", "a", "z", "t"],
            extra_fields=[],
        )
        out = logp_fn(data, v, a, z, t)
        self._assert_2d_shape(out, n_participants, n_trials)

    def test_make_rl_logp_func_with_computed_param_output_is_2d_array(self):
        n_participants = 2
        n_trials = 4
        data = self._gen_data(n_participants, n_trials, ["rt", "response", "feedback"])
        a, z, t, rl_alpha, scaler = self._gen_params(
            n_participants, n_trials, ["a", "z", "t", "rl.alpha", "scaler"]
        )
        logp_fn = make_rl_logp_func(
            ssm_logp_func=ssm_logp_with_v,
            n_participants=n_participants,
            n_trials=n_trials,
            data_cols=["rt", "response", "feedback"],
            list_params=["a", "z", "t"],
            extra_fields=["rl.alpha", "scaler"],
        )
        out = logp_fn(data, a, z, t, rl_alpha, scaler)
        self._assert_2d_shape(out, n_participants, n_trials)

    def test_make_rl_logp_func_with_two_computed_params_output_is_2d_array(self):
        n_participants = 2
        n_trials = 3
        data = self._gen_data(n_participants, n_trials, ["rt", "response", "feedback"])
        a, z, t = self._gen_params(n_participants, n_trials, ["a", "z", "t"])
        rl_alpha = np.full(n_participants * n_trials, 0.3)
        scaler = np.full(n_participants * n_trials, 1.5)
        bonus = self._gen_params(n_participants, n_trials, ["bonus"])[0]
        logp_fn = make_rl_logp_func(
            ssm_logp_func=ssm_logp_with_v_and_theta,
            n_participants=n_participants,
            n_trials=n_trials,
            data_cols=["rt", "response", "feedback"],
            list_params=["a", "z", "t"],
            extra_fields=["rl.alpha", "scaler", "bonus"],
        )
        out = logp_fn(data, a, z, t, rl_alpha, scaler, bonus)
        self._assert_2d_shape(out, n_participants, n_trials)

    @pytest.mark.parametrize(
        "n_participants,n_trials", [(1, 1), (1, 7), (4, 3), (5, 10)]
    )
    def test_parametrized_no_computed_output_shape(self, n_participants, n_trials):
        data = self._gen_data(n_participants, n_trials, ["rt", "response"])
        v, a, z, t = self._gen_params(n_participants, n_trials, ["v", "a", "z", "t"])
        logp_fn = make_rl_logp_func(
            ssm_logp_func=dummy_ssm_logp,
            n_participants=n_participants,
            n_trials=n_trials,
            data_cols=["rt", "response"],
            list_params=["v", "a", "z", "t"],
            extra_fields=[],
        )
        out = logp_fn(data, v, a, z, t)
        self._assert_2d_shape(out, n_participants, n_trials)

    @pytest.mark.parametrize("n_participants,n_trials", [(1, 2), (2, 1), (3, 5)])
    def test_parametrized_one_computed_output_shape(self, n_participants, n_trials):
        data = self._gen_data(n_participants, n_trials, ["rt", "response", "feedback"])
        a, z, t, rl_alpha, scaler = self._gen_params(
            n_participants, n_trials, ["a", "z", "t", "rl.alpha", "scaler"]
        )
        logp_fn = make_rl_logp_func(
            ssm_logp_func=ssm_logp_with_v,
            n_participants=n_participants,
            n_trials=n_trials,
            data_cols=["rt", "response", "feedback"],
            list_params=["a", "z", "t"],
            extra_fields=["rl.alpha", "scaler"],
        )
        out = logp_fn(data, a, z, t, rl_alpha, scaler)
        self._assert_2d_shape(out, n_participants, n_trials)

    @pytest.mark.parametrize("n_participants,n_trials", [(1, 3), (2, 2), (4, 4)])
    def test_parametrized_two_computed_output_shape(self, n_participants, n_trials):
        data = self._gen_data(n_participants, n_trials, ["rt", "response", "feedback"])
        a, z, t = self._gen_params(n_participants, n_trials, ["a", "z", "t"])
        rl_alpha = np.full(n_participants * n_trials, 0.3)
        scaler = np.full(n_participants * n_trials, 1.5)
        bonus = self._gen_params(n_participants, n_trials, ["bonus"])[0]
        logp_fn = make_rl_logp_func(
            ssm_logp_func=ssm_logp_with_v_and_theta,
            n_participants=n_participants,
            n_trials=n_trials,
            data_cols=["rt", "response", "feedback"],
            list_params=["a", "z", "t"],
            extra_fields=["rl.alpha", "scaler", "bonus"],
        )
        out = logp_fn(data, a, z, t, rl_alpha, scaler, bonus)
        self._assert_2d_shape(out, n_participants, n_trials)

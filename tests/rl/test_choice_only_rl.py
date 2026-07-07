"""Choice-only RLSSM likelihood and construction smoke tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import hssm
from hssm.rl import RLSSMConfig, registry
from hssm.rl.rlssm import _RLSSM
from hssm.utils import annotate_function


@pytest.mark.parametrize(
    ("decision_process", "inputs", "matrix", "expected"),
    [
        (
            "inv_temp_softmax_2",
            ["beta", "q0", "q1", "response"],
            jnp.asarray(
                [
                    [1.0, 0.2, 1.2, 1.0],
                    [1.0, 0.2, 1.2, 0.0],
                    [3.0, 0.2, 1.2, 1.0],
                ]
            ),
            jnp.asarray(
                [
                    1.2 - jnp.logaddexp(0.2, 1.2),
                    0.2 - jnp.logaddexp(0.2, 1.2),
                    3.6 - jnp.logaddexp(0.6, 3.6),
                ]
            ),
        ),
        (
            "inv_temp_softmax_3",
            ["beta", "q0", "q1", "q2", "response"],
            jnp.asarray(
                [
                    [1.0, 0.2, 1.2, 2.2, 2.0],
                    [1.0, 0.2, 1.2, 2.2, 0.0],
                    [3.0, 0.2, 1.2, 2.2, 2.0],
                ]
            ),
            jnp.asarray(
                [
                    2.2 - jnp.log(jnp.exp(0.2) + jnp.exp(1.2) + jnp.exp(2.2)),
                    0.2 - jnp.log(jnp.exp(0.2) + jnp.exp(1.2) + jnp.exp(2.2)),
                    6.6 - jnp.log(jnp.exp(0.6) + jnp.exp(3.6) + jnp.exp(6.6)),
                ]
            ),
        ),
    ],
)
def test_inv_temp_softmax_base_logp_values_and_metadata(
    decision_process, inputs, matrix, expected
):
    """Choice-only inverse-temperature softmax logp matches expected values."""
    base_logp = registry._get_ssm_logp(decision_process)

    result = base_logp(matrix)

    assert base_logp.inputs == inputs
    assert base_logp.outputs == ["logp"]
    assert result.shape == (matrix.shape[0],)
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-6)
    assert result[-1] > result[0]


@pytest.mark.parametrize(
    ("decision_process", "matrix"),
    [
        (
            "inv_temp_softmax_2",
            jnp.asarray(
                [
                    [1.0, 0.2, 1.2, -1.0],
                    [1.0, 0.2, 1.2, 2.0],
                    [1.0, 0.2, 1.2, 1.5],
                ]
            ),
        ),
        (
            "inv_temp_softmax_3",
            jnp.asarray(
                [
                    [1.0, 0.2, 1.2, 2.2, -1.0],
                    [1.0, 0.2, 1.2, 2.2, 3.0],
                    [1.0, 0.2, 1.2, 2.2, 1.5],
                ]
            ),
        ),
    ],
)
def test_inv_temp_softmax_rejects_invalid_response_labels(decision_process, matrix):
    """Invalid choice labels produce deterministic impossible logp values."""
    base_logp = registry._get_ssm_logp(decision_process)

    result = base_logp(matrix)

    assert jnp.all(jnp.isneginf(result))


def test_choice_only_rlssm_passes_choice_only_flag_to_distribution(monkeypatch):
    """The RLSSM distribution builder receives the choice-only config flag."""

    @annotate_function(inputs=["beta", "q0", "q1", "response"], outputs=["logp"])
    def _fake_base_logp(lan_matrix):
        return jnp.sum(lan_matrix, axis=1)

    config = RLSSMConfig(
        model_name="choice_only_test",
        loglik_kind="approx_differentiable",
        decision_process="inv_temp_softmax_2",
        decision_process_loglik_kind="approx_differentiable",
        learning_process_kind="approx_differentiable",
        list_params=["rl_alpha", "beta"],
        bounds={"rl_alpha": (0.0, 1.0), "beta": (0.0, 20.0)},
        params_default=[0.2, 2.0],
        learning_process={},
        response=["response"],
        choices=[0, 1],
        extra_fields=[],
        ssm_logp_func=_fake_base_logp,
        loglik=lambda *args: jnp.zeros(1),
    )
    model = _RLSSM.__new__(_RLSSM)
    model.list_params = list(config.list_params)
    model.model_config = config
    model.bounds = dict(config.bounds)
    model.lapse = None
    model.data = pd.DataFrame({"response": [0, 1]})
    captured = {}

    def _capture_make_distribution(**kwargs):
        captured.update(kwargs)
        return object

    monkeypatch.setattr("hssm.rl.rlssm.make_distribution", _capture_make_distribution)

    model._make_model_distribution()

    assert captured["is_choice_only"] is True


class TestChoiceOnlyRealSSMSSmoke:
    """Smoke tests against a real ssms install when the new presets are present."""

    @staticmethod
    def _require_choice_only_ssms_model():
        ssms_rl = pytest.importorskip(
            "ssms.rl", reason="installed ssm-simulators has no ssms.rl module"
        )
        if not hasattr(ssms_rl, "resolve_model"):
            pytest.skip("ssms.rl does not expose resolve_model")
        try:
            ssms_rl.resolve_model("2AB_RW_InvTempSoftmax")
        except Exception:
            pytest.skip("installed ssms.rl does not expose 2AB_RW_InvTempSoftmax")

    @pytest.mark.slow
    def test_real_ssms_choice_only_rlssm_compiles_finite_no_lapse_logp(self):
        """A real ssms choice-only preset can compile a finite no-lapse logp."""
        self._require_choice_only_ssms_model()
        hssm.set_floatX("float32", update_jax=True)

        config = RLSSMConfig.from_ssms_model("2AB_RW_InvTempSoftmax")
        data = pd.DataFrame(
            {
                "participant_id": [0, 0, 0, 1, 1, 1],
                "response": [0, 1, 1, 1, 0, 1],
                "feedback": [1, 0, 1, 0, 1, 1],
            }
        )

        model = hssm.RLSSM(
            data=data,
            model_config=config,
            p_outlier=None,
            prior_settings=None,
        )
        logp_fn = model.compile_logp()
        logp = logp_fn(model.initial_point(transformed=False))

        assert np.isfinite(logp)

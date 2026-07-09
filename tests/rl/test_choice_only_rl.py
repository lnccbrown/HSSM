"""Choice-only RLSSM likelihood and construction smoke tests."""

from __future__ import annotations

import bambi as bmb
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest

import hssm
from hssm.rl import RLSSMConfig, registry
from hssm.rl.rlssm import _RLSSM
from hssm.utils import annotate_function


@annotate_function(inputs=["beta", "response"], outputs=["logp"], computed={})
def _fake_choice_only_logp(lan_matrix):
    """Small response-only logp for constructor validation tests."""
    return -jnp.zeros(lan_matrix.shape[0])


def _fake_choice_only_config(choices=None):
    """Build a minimal choice-only RLSSMConfig without depending on ssms."""
    return RLSSMConfig(
        model_name="ddm",
        loglik_kind="approx_differentiable",
        decision_process="inv_temp_softmax_2",
        decision_process_loglik_kind="approx_differentiable",
        learning_process_kind="approx_differentiable",
        list_params=["beta"],
        bounds={"beta": (0.0, 20.0)},
        params_default=[2.0],
        learning_process={},
        response=["response"],
        choices=[0, 1] if choices is None else choices,
        extra_fields=[],
        ssm_logp_func=_fake_choice_only_logp,
    )


def _fake_choice_only_data(responses):
    """Return a balanced response-only panel for validation tests."""
    return pd.DataFrame(
        {
            "participant_id": np.repeat([0, 1], len(responses) // 2),
            "response": responses,
        }
    )


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
        (
            "inv_temp_softmax_4",
            ["beta", "q0", "q1", "q2", "q3", "response"],
            jnp.asarray(
                [
                    [1.0, 0.2, 1.2, 2.2, 3.2, 3.0],
                    [1.0, 0.2, 1.2, 2.2, 3.2, 0.0],
                    [3.0, 0.2, 1.2, 2.2, 3.2, 3.0],
                ]
            ),
            jnp.asarray(
                [
                    3.2
                    - jnp.log(
                        jnp.exp(0.2) + jnp.exp(1.2) + jnp.exp(2.2) + jnp.exp(3.2)
                    ),
                    0.2
                    - jnp.log(
                        jnp.exp(0.2) + jnp.exp(1.2) + jnp.exp(2.2) + jnp.exp(3.2)
                    ),
                    9.6
                    - jnp.log(
                        jnp.exp(0.6) + jnp.exp(3.6) + jnp.exp(6.6) + jnp.exp(9.6)
                    ),
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
    np.testing.assert_allclose(
        np.asarray(result), np.asarray(expected), rtol=1e-5, atol=1e-6
    )
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
        (
            "inv_temp_softmax_4",
            jnp.asarray(
                [
                    [1.0, 0.2, 1.2, 2.2, 3.2, -1.0],
                    [1.0, 0.2, 1.2, 2.2, 3.2, 4.0],
                    [1.0, 0.2, 1.2, 2.2, 3.2, 1.5],
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


def test_inv_temp_softmax_preserves_lan_matrix_float_dtype():
    """Choice-only softmax logp should not downcast incoming floating values."""
    prev_floatx = pytensor.config.floatX
    hssm.set_floatX("float64", update_jax=True)
    try:
        matrix = jnp.asarray(
            [
                [1.0, 0.2, 1.2, 1.0],
                [1.0, 0.2, 1.2, 0.0],
            ],
            dtype=jnp.float64,
        )
        base_logp = registry._get_ssm_logp("inv_temp_softmax_2")

        result = base_logp(matrix)

        assert result.dtype == matrix.dtype
    finally:
        hssm.set_floatX(prev_floatx, update_jax=True)


@pytest.mark.parametrize(
    ("responses", "match"),
    [
        ([0, 1.5, 0, 1], "integral"),
        ([0, "left", 0, 1], "numeric"),
        ([0, np.inf, 0, 1], "finite"),
        ([0, 2, 0, 1], "Invalid responses"),
    ],
)
def test_choice_only_rlssm_validates_response_labels(responses, match):
    """Choice-only RLSSM response labels are checked before logp evaluation."""
    with pytest.raises(ValueError, match=match):
        hssm.RLSSM(
            data=_fake_choice_only_data(responses),
            model_config=_fake_choice_only_config(),
            p_outlier=None,
            prior_settings=None,
        )


def test_choice_only_rlssm_warns_for_missing_declared_choices():
    """Declared choice labels may be absent, but users should be warned."""
    with pytest.warns(UserWarning, match="missing from your dataset"):
        hssm.RLSSM(
            data=_fake_choice_only_data([0, 0, 0, 0]),
            model_config=_fake_choice_only_config(),
            p_outlier=None,
            prior_settings=None,
        )


def _make_shell_rlssm(config, lapse):
    """Build an _RLSSM shell for distribution-construction unit tests."""
    model = _RLSSM.__new__(_RLSSM)
    model.list_params = [*config.list_params, "p_outlier"]
    model.model_config = config
    model.bounds = dict(config.bounds)
    model.lapse = lapse
    model.data = pd.DataFrame({"response": [0, 1], "rt": [0.5, 0.6]})
    return model


def test_choice_only_rlssm_converts_scalar_lapse_for_distribution(monkeypatch):
    """Choice-only RLSSM passes scalar lapse probabilities as log probabilities."""

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
    model = _make_shell_rlssm(config, lapse=0.5)
    captured = {}

    def _capture_make_distribution(**kwargs):
        captured.update(kwargs)
        return object

    monkeypatch.setattr("hssm.rl.rlssm.make_distribution", _capture_make_distribution)

    model._make_model_distribution()

    assert captured["is_choice_only"] is True
    assert captured["lapse"] == pytest.approx(np.log(0.5))


def test_rt_choice_rlssm_passes_lapse_prior_through(monkeypatch):
    """RT+choice RLSSM does not convert Bambi lapse priors."""
    config = RLSSMConfig(
        model_name="rt_choice_test",
        loglik_kind="approx_differentiable",
        decision_process="angle",
        decision_process_loglik_kind="approx_differentiable",
        learning_process_kind="approx_differentiable",
        list_params=["v", "a"],
        bounds={"v": (-5.0, 5.0), "a": (0.0, 5.0)},
        params_default=[0.0, 1.0],
        learning_process={},
        response=["rt", "response"],
        choices=[0, 1],
        extra_fields=[],
        ssm_logp_func=lambda *args: jnp.zeros(1),
        loglik=lambda *args: jnp.zeros(1),
    )
    lapse_prior = bmb.Prior("Uniform", lower=0.0, upper=20.0)
    model = _make_shell_rlssm(config, lapse=lapse_prior)
    captured = {}

    def _capture_make_distribution(**kwargs):
        captured.update(kwargs)
        return object

    monkeypatch.setattr("hssm.rl.rlssm.make_distribution", _capture_make_distribution)

    model._make_model_distribution()

    assert captured["is_choice_only"] is False
    assert captured["lapse"] is lapse_prior


class TestChoiceOnlyRealSSMSSmoke:
    """Smoke tests against a real ssms install when the new presets are present."""

    @staticmethod
    def _require_choice_only_ssms_model(model_name):
        ssms_rl = pytest.importorskip(
            "ssms.rl", reason="installed ssm-simulators has no ssms.rl module"
        )
        if not hasattr(ssms_rl, "resolve_model"):
            pytest.skip("ssms.rl does not expose resolve_model")
        try:
            ssms_rl.resolve_model(model_name)
        except Exception:
            pytest.skip(f"installed ssms.rl does not expose {model_name}")

    @pytest.mark.parametrize(
        ("model_name", "n_choices"),
        [
            ("2AB_RW_InvTempSoftmax", 2),
            ("3AB_RW_InvTempSoftmax", 3),
        ],
    )
    @pytest.mark.slow
    def test_real_ssms_choice_only_rlssm_compiles_finite_active_lapse_logp(
        self, model_name, n_choices
    ):
        """A real ssms choice-only preset can compile a finite active-lapse logp."""
        self._require_choice_only_ssms_model(model_name)
        prev_floatx = pytensor.config.floatX
        hssm.set_floatX("float32", update_jax=True)
        try:
            config = RLSSMConfig.from_ssms_model(model_name)
            expected_qs = {f"q{i}" for i in range(n_choices)}
            assert config.response == ["response"]
            assert config.decision_process == f"inv_temp_softmax_{n_choices}"
            assert config.list_params == ["rl_alpha", "beta"]
            assert set(config.ssm_logp_func.computed) == expected_qs
            data = pd.DataFrame(
                {
                    "participant_id": np.repeat([0, 1], n_choices * 2),
                    "response": np.tile(np.arange(n_choices), 4),
                    "feedback": np.tile([1, 0], n_choices * 2),
                }
            )

            model = hssm.RLSSM(
                data=data,
                model_config=config,
                prior_settings=None,
            )
            logp_fn = model.compile_logp()
            logp = logp_fn(model.initial_point(transformed=False))
            dist_logp = pm.logp(
                model.model_distribution.dist(
                    rl_alpha=0.5,
                    beta=2.0,
                    p_outlier=0.05,
                ),
                data["response"].to_numpy(),
            ).eval()

            assert model.lapse == pytest.approx(1 / n_choices)
            assert "p_outlier" in model.params
            assert np.isfinite(logp)
            assert np.all(np.isfinite(dist_logp))
            assert np.all(dist_logp <= 0.0)
        finally:
            hssm.set_floatX(prev_floatx, update_jax=True)

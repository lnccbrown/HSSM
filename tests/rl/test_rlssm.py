"""Tests for the RLSSM class.

Mirrors the structure of tests/test_hssm.py, covering initialisation,
config validation, param keys, balanced-panel enforcement, the no-lapse
variant, bambi / PyMC model construction, and a sampling smoke test.
"""

import logging
from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cloudpickle
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytensor
import pytest

import hssm
from hssm.rl import registry
from hssm.distribution_utils import make_distribution as real_make_distribution
from hssm.rl import RLSSM, RLSSMConfig, register_rlssm_model
from hssm.rl.likelihoods.two_armed_bandit import compute_v_subject_wise
from hssm.rl.rlssm import _RLSSM
from hssm.utils import annotate_function

# Annotate the RL learning function: maps
#   (rl_alpha, scaler, response, feedback) -> v
_compute_v_annotated = annotate_function(
    inputs=["rl_alpha", "scaler", "response", "feedback"],
    outputs=["v"],
)(compute_v_subject_wise)


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


@pytest.fixture(scope="module", autouse=True)
def _set_floatx_float32() -> Generator[None, None, None]:
    """Ensure float32 is used for this module's tests, then restore previous setting."""
    prev_floatx = pytensor.config.floatX
    hssm.set_floatX("float32", update_jax=True)
    try:
        yield
    finally:
        hssm.set_floatX(prev_floatx, update_jax=True)


# runs before every test function to isolate the RLSSM registry state, preventing test bleed-through
@pytest.fixture(autouse=True)
def isolated_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate RL registries so simplified-interface tests do not leak state."""
    monkeypatch.setattr(registry, "_SSM_REGISTRY", deepcopy(registry._SSM_REGISTRY))
    monkeypatch.setattr(registry, "_RLSSM_REGISTRY", deepcopy(registry._RLSSM_REGISTRY))
    monkeypatch.setattr(registry, "_SSM_LOGP_CACHE", dict(registry._SSM_LOGP_CACHE))


@pytest.fixture(scope="module")
def rldm_data() -> pd.DataFrame:
    """Load the RLDM fixture dataset (balanced panel)."""
    raw = np.load(
        Path(__file__).parent.parent / "fixtures" / "rldm_data.npy", allow_pickle=True
    ).item()
    return pd.DataFrame(raw["data"])


@pytest.fixture(scope="module")
def rlssm_config() -> RLSSMConfig:
    """Minimal but valid RLSSMConfig for the RLDM fixture dataset."""
    return RLSSMConfig(
        model_name="rldm_test",
        loglik_kind="approx_differentiable",
        decision_process="angle",
        decision_process_loglik_kind="approx_differentiable",
        learning_process_kind="blackbox",
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


class TestRLSSMInit:
    """Basic construction, attribute checks, and invalid-input guards at construction time."""

    def test_rlssm_init(self, rldm_data, rlssm_config) -> None:
        """Basic RLSSM initialisation should succeed and return an RLSSM instance."""
        model = RLSSM(data=rldm_data, model_config=rlssm_config)
        assert isinstance(model, RLSSM)
        assert model.model_config.model_name == "rldm_test"

    def test_rlssm_panel_attrs(self, rldm_data, rlssm_config) -> None:
        """n_participants and n_trials should match the fixture data structure."""
        model = RLSSM(data=rldm_data, model_config=rlssm_config)

        n_participants = rldm_data["participant_id"].nunique()
        n_trials = len(rldm_data) // n_participants

        assert model.n_participants == n_participants
        assert model.n_trials == n_trials

    def test_rlssm_params_keys(self, rldm_data, rlssm_config) -> None:
        """model.params should contain exactly list_params + p_outlier."""
        model = RLSSM(data=rldm_data, model_config=rlssm_config)
        expected = set(rlssm_config.list_params) | {"p_outlier"}
        assert set(model.params.keys()) == expected

    def test_rlssm_unbalanced_raises(self, rldm_data, rlssm_config) -> None:
        """Dropping one row should make the panel unbalanced → ValueError."""
        unbalanced = rldm_data.iloc[:-1].copy()
        with pytest.raises(ValueError, match="balanced panels"):
            RLSSM(data=unbalanced, model_config=rlssm_config)

    def test_rlssm_nan_participant_id_raises(self, rldm_data, rlssm_config) -> None:
        """NaN in participant_id column should raise ValueError before groupby silently drops rows."""
        nan_data = rldm_data.copy()
        nan_data.loc[nan_data.index[0], "participant_id"] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            RLSSM(data=nan_data, model_config=rlssm_config)

    def test_rlssm_missing_ssm_logp_func_raises(self, rldm_data, rlssm_config) -> None:
        """RLSSMConfig without ssm_logp_func should raise ValueError on init."""
        bad_config = RLSSMConfig(
            model_name="rldm_bad",
            loglik_kind="approx_differentiable",
            decision_process="angle",
            decision_process_loglik_kind="approx_differentiable",
            learning_process_kind="blackbox",
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
            RLSSM(data=rldm_data, model_config=bad_config)

    def test_rlssm_unannotated_ssm_logp_func_raises(
        self, rldm_data, rlssm_config
    ) -> None:
        """A plain callable without @annotate_function attrs should raise ValueError."""
        bad_config = RLSSMConfig(
            model_name="rldm_bad",
            loglik_kind="approx_differentiable",
            decision_process="angle",
            decision_process_loglik_kind="approx_differentiable",
            learning_process_kind="blackbox",
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
            RLSSM(data=rldm_data, model_config=bad_config)

    def test_rlssm_missing_data_raises(self, rldm_data, rlssm_config) -> None:
        """Passing missing_data!=False should raise NotImplementedError with 'missing_data' in msg."""
        with pytest.raises(NotImplementedError, match="missing_data"):
            RLSSM(data=rldm_data, model_config=rlssm_config, missing_data=True)

    def test_rlssm_deadline_raises(self, rldm_data, rlssm_config) -> None:
        """Passing deadline!=False should raise NotImplementedError with 'deadline' in msg."""
        with pytest.raises(NotImplementedError, match="deadline"):
            RLSSM(data=rldm_data, model_config=rlssm_config, deadline=True)


class TestRLSSMModelStructure:
    """Internal model anatomy after construction: params, prefix, lapse, bambi/pymc, extra_fields."""

    def test_rlssm_params_is_trialwise_aligned(self, rldm_data, rlssm_config) -> None:
        """params_is_trialwise must align with list_params (same length, p_outlier=False)."""
        model = RLSSM(data=rldm_data, model_config=rlssm_config)
        assert model.model_config.list_params is not None
        params_is_trialwise = [
            name != "p_outlier" for name in model.model_config.list_params
        ]
        assert len(params_is_trialwise) == len(model.model_config.list_params)
        for name, is_tw in zip(model.model_config.list_params, params_is_trialwise):
            if name == "p_outlier":
                assert not is_tw, "p_outlier must be non-trialwise"
            else:
                assert is_tw, f"{name} must be trialwise"

    def test_rlssm_get_prefix(self, rldm_data, rlssm_config) -> None:
        """_get_prefix must use token-based matching, not substring search.

        - 'rl_alpha_Intercept' → 'rl_alpha'  (underscore-containing RL param)
        - 'p_outlier_log__'   → 'p_outlier'  (lapse param via token loop, not substring)
        - 'a_Intercept'       → 'a'           (single-token standard param)
        """
        model = RLSSM(data=rldm_data, model_config=rlssm_config)
        assert model._get_prefix("rl_alpha_Intercept") == "rl_alpha"
        assert model._get_prefix("p_outlier_log__") == "p_outlier"
        assert model._get_prefix("p_outlier") == "p_outlier"
        assert model._get_prefix("a_Intercept") == "a"
        # Fallback: not in params
        assert model._get_prefix("unknown_param") == "unknown_param"

    def test_rlssm_no_lapse(self, rldm_data, rlssm_config) -> None:
        """Setting p_outlier=None should remove p_outlier from params."""
        model = RLSSM(data=rldm_data, model_config=rlssm_config, p_outlier=None)
        assert "p_outlier" not in model.params

    def test_rlssm_model_built(self, rldm_data, rlssm_config) -> None:
        """The bambi model should be built and the computed param 'v' absent from params."""
        model = RLSSM(data=rldm_data, model_config=rlssm_config)
        assert model.model is not None
        # rl_alpha is a free (sampled) parameter
        assert "rl_alpha" in model.params
        # v is computed inside the Op; it must NOT appear as a free parameter
        assert "v" not in model.params

    def test_rlssm_extra_fields_are_copies(self, rldm_data, rlssm_config) -> None:
        """extra_fields passed to make_distribution must be independent numpy copies.

        to_numpy(copy=True) should return a new buffer; if it returned a view,
        in-place mutations of the DataFrame would silently corrupt the distribution.
        """
        model = RLSSM(data=rldm_data, model_config=rlssm_config)
        captured: dict = {}

        def capturing_make_distribution(*args, **kwargs):
            captured["extra_fields"] = kwargs.get("extra_fields")
            return real_make_distribution(*args, **kwargs)

        with patch(
            "hssm.rl.rlssm.make_distribution", side_effect=capturing_make_distribution
        ):
            model._make_model_distribution()

        assert captured.get("extra_fields") is not None
        for field_name, arr in zip(rlssm_config.extra_fields, captured["extra_fields"]):
            original = model.data[field_name].to_numpy()
            assert not np.shares_memory(arr, original), (
                f"extra_fields['{field_name}'] shares memory with the DataFrame — "
                "it is a view, not a copy"
            )

    def test_rlssm_pymc_model(self, rldm_data, rlssm_config) -> None:
        """pymc_model should be accessible after model construction."""
        model = RLSSM(data=rldm_data, model_config=rlssm_config)
        assert model.pymc_model is not None

    def test_rlssm_no_extra_fields_none_passed_to_make_distribution(
        self, rldm_data, rlssm_config
    ) -> None:
        """When extra_fields is empty, make_distribution receives extra_fields=None."""
        config_no_extra = RLSSMConfig(
            model_name="rldm_no_extra",
            loglik_kind="approx_differentiable",
            decision_process="angle",
            decision_process_loglik_kind="approx_differentiable",
            learning_process_kind="blackbox",
            list_params=rlssm_config.list_params,
            params_default=rlssm_config.params_default,
            bounds=rlssm_config.bounds,
            learning_process=rlssm_config.learning_process,
            response=list(rlssm_config.response),
            choices=list(rlssm_config.choices),
            extra_fields=[],  # empty → should pass None to make_distribution
            ssm_logp_func=_dummy_ssm_logp,
        )
        model = RLSSM(data=rldm_data, model_config=config_no_extra)
        captured: dict = {}

        def capturing_make_distribution(*args, **kwargs):
            captured["extra_fields"] = kwargs.get("extra_fields")
            return real_make_distribution(*args, **kwargs)

        with patch(
            "hssm.rl.rlssm.make_distribution", side_effect=capturing_make_distribution
        ):
            model._make_model_distribution()

        assert captured.get("extra_fields") is None


class TestRLSSMSerialization:
    """Cloudpickle serialisation and deserialisation."""

    def test_rlssm_pickle_round_trip(
        self, rldm_data: pd.DataFrame, rlssm_config: RLSSMConfig
    ) -> None:
        """Cloudpickle round-trip must reconstruct an equivalent RLSSM.

        Verifies that __getstate__ / __setstate__ survive serialisation:
        - The reconstructed object is a fresh RLSSM (not the same instance).
        - n_participants and n_trials are preserved.
        - list_params (including p_outlier) are preserved.
        - model_config.model_name is preserved.
        - model.model (bambi model) is rebuilt, confirming full re-initialisation.
        """
        model = RLSSM(data=rldm_data, model_config=rlssm_config)
        blob = cloudpickle.dumps(model)
        restored = cloudpickle.loads(blob)

        assert restored is not model
        assert isinstance(restored, RLSSM)
        assert restored.n_participants == model.n_participants
        assert restored.n_trials == model.n_trials
        assert restored.list_params == model.list_params
        assert restored.model_config.model_name == model.model_config.model_name
        assert restored.model is not None


class TestRLSSMSampling:
    """Slow sampling smoke tests."""

    @pytest.mark.slow
    def test_rlssm_sample_smoke(self, rldm_data, rlssm_config) -> None:
        """Minimal sampling run should return an InferenceData object."""
        model = RLSSM(data=rldm_data, model_config=rlssm_config)
        trace = model.sample(
            draws=4, tune=50, chains=1, cores=1, sampler="numpyro", target_accept=0.9
        )
        assert trace is not None


class TestRLSSMSimplifiedInterface:
    """Public model= kwarg API: registry lookups, register_rlssm_model, unsupported-feature properties."""

    def test_rlssm_is_subclass_of_internal(self) -> None:
        """RLSSM must be a subclass of _RLSSM."""
        assert issubclass(RLSSM, _RLSSM)

    @pytest.mark.parametrize(
        "model_name, expected_dp",
        [
            ("2AB_RescorlaWagner_DDM", "ddm"),
            ("2AB_RescorlaWagner_Angle", "angle"),
            ("2AB_RescorlaWagner_Weibull", "weibull"),
        ],
    )
    def test_rlssm_builtin_models_instantiate(
        self, rldm_data, model_name: str, expected_dp: str
    ) -> None:
        """All built-in 2AB_RescorlaWagner_* models should instantiate correctly."""
        model = RLSSM(data=rldm_data, model=model_name)
        assert isinstance(model, RLSSM)
        assert model.model_config.decision_process == expected_dp
        assert "rl_alpha" in model.params
        assert "scaler" in model.params
        assert "a" in model.params
        assert "v" not in model.params

    def test_rlssm_default_model_is_ddm(self, rldm_data) -> None:
        """Omitting model should default to '2AB_RescorlaWagner_DDM'."""
        model = RLSSM(data=rldm_data)
        assert isinstance(model, RLSSM)
        assert model.model_config.decision_process == "ddm"

    def test_rlssm_model_config_provided(self, rldm_data, rlssm_config) -> None:
        """Passing model_config= directly should bypass the registry."""
        model = RLSSM(data=rldm_data, model_config=rlssm_config)
        assert model.model_config.model_name == rlssm_config.model_name

    def test_rlssm_unregistered_model_raises(self, rldm_data) -> None:
        """Using an unknown model name should raise ValueError."""
        with pytest.raises(ValueError, match="not found in the RLSSM registry"):
            RLSSM(data=rldm_data, model="model_not_in_registry")

    def test_rlssm_missing_data_property_raises(self, rldm_data) -> None:
        """Accessing .missing_data on a built RLSSM instance must raise NotImplementedError."""
        model = RLSSM(data=rldm_data, model="2AB_RescorlaWagner_DDM")
        with pytest.raises(NotImplementedError, match="missing_data"):
            _ = model.missing_data

    def test_rlssm_deadline_property_raises(self, rldm_data) -> None:
        """Accessing .deadline on a built RLSSM instance must raise NotImplementedError."""
        model = RLSSM(data=rldm_data, model="2AB_RescorlaWagner_DDM")
        with pytest.raises(NotImplementedError, match="deadline"):
            _ = model.deadline

    def test_rlssm_loglik_missing_data_property_raises(self, rldm_data) -> None:
        """Accessing .loglik_missing_data on a built RLSSM instance must raise NotImplementedError."""
        model = RLSSM(data=rldm_data, model="2AB_RescorlaWagner_DDM")
        with pytest.raises(NotImplementedError, match="loglik_missing_data"):
            _ = model.loglik_missing_data

    def test_register_rlssm_model(self, rldm_data) -> None:
        """A user-registered model should be instantiable via the simplified interface."""
        # Re-use the existing annotated learning function and ssm logp from the
        # module-level helpers defined at the top of this test file.
        register_rlssm_model(
            name="rldm_custom_test",
            decision_process="angle",
            learning_process={"v": _compute_v_annotated},
            learning_process_params=["rl_alpha", "scaler"],
            learning_process_bounds={"rl_alpha": (0.0, 1.0), "scaler": (0.0, 10.0)},
            learning_process_params_default=[0.1, 1.0],
            extra_fields=["feedback"],
            choices=[0, 1],
        )
        model = RLSSM(data=rldm_data, model="rldm_custom_test")
        assert isinstance(model, RLSSM)
        assert "rl_alpha" in model.params
        assert "v" not in model.params

    def test_rlssm_init_args_uses_simplified_interface(self, rldm_data) -> None:
        """_init_args should reflect the simplified constructor, not model_config."""
        model = RLSSM(data=rldm_data, model="2AB_RescorlaWagner_DDM")
        assert "model" in model._init_args
        assert model._init_args["model"] == "2AB_RescorlaWagner_DDM"
        # model_config should not be baked in as a hard reference
        assert "model_config" in model._init_args
        assert model._init_args["model_config"] is None

    def test_rlssm_model_config_with_overrides_warns(
        self, rldm_data, rlssm_config, caplog
    ) -> None:
        """Warn when model_config is given alongside model/overrides.

        The extra arguments (model, learning_process, decision_process, choices)
        should be ignored and a warning emitted.
        """
        with caplog.at_level(logging.WARNING, logger="hssm"):
            RLSSM(data=rldm_data, model_config=rlssm_config, model="some_other_model")

        assert any("ignoring" in r.message for r in caplog.records)

    def test_rlssm_model_config_without_overrides_does_not_warn(
        self, rldm_data, rlssm_config, caplog
    ) -> None:
        """Passing only model_config should not emit the ignored-args warning."""
        with caplog.at_level(logging.WARNING, logger="hssm"):
            RLSSM(data=rldm_data, model_config=rlssm_config)

        assert not any("ignoring" in r.message for r in caplog.records)

    def test_rlssm_learning_process_override_keeps_matching_metadata(
        self, rldm_data
    ) -> None:
        """Public RLSSM constructor should accept LP overrides with the same sampled params."""

        @annotate_function(
            inputs=["rl_alpha", "scaler", "response", "feedback"],
            outputs=["v"],
        )
        def alt_compute_v(rl_alpha, scaler, response, feedback):
            return rl_alpha + scaler

        model = RLSSM(
            data=rldm_data,
            model="2AB_RescorlaWagner_DDM",
            learning_process={"v": alt_compute_v},
        )

        assert model.model_config.learning_process == {"v": alt_compute_v}
        assert "rl_alpha" in model.params
        assert "scaler" in model.params

    def test_rlssm_decision_process_override_updates_sampled_params(
        self, rldm_data
    ) -> None:
        """Public RLSSM constructor should respect decision_process overrides."""
        model = RLSSM(
            data=rldm_data,
            model="2AB_RescorlaWagner_DDM",
            decision_process="angle",
        )

        assert model.model_config.decision_process == "angle"
        assert "theta" in model.params

    def test_rlssm_builtin_model_re_resolves_registered_ssm(self, rldm_data) -> None:
        """Built-in RLSSM models should pick up later register_ssm() overrides."""

        @annotate_function(
            inputs=["v", "custom_a", "rt", "response"],
            outputs=["logp"],
        )
        def custom_ddm_logp(v, custom_a, rt, response):
            return custom_a

        registry.register_ssm(
            name="ddm",
            ssm_base_logp_func=custom_ddm_logp,
            list_params_ssm=["v", "custom_a"],
            bounds_ssm={"custom_a": (0.3, 3.0)},
            params_default_ssm=[0.0, 1.5],
            response=["rt", "response"],
        )

        model = RLSSM(data=rldm_data, model="2AB_RescorlaWagner_DDM")

        assert model.model_config.decision_process == "ddm"
        assert "rl_alpha" in model.params
        assert "scaler" in model.params
        assert "custom_a" in model.params
        assert "a" not in model.params

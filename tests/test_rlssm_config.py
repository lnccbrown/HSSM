import sys
import types

import jax.numpy as jnp
import pytest

import hssm
from hssm.config import (
    DEFAULT_SSM_CHOICES,
    DEFAULT_SSM_OBSERVED_DATA,
    Config,
    ModelConfig,
)
from hssm.rl import RLSSMConfig
from hssm.utils import annotate_function

# Define constants for repeated data structures
DEFAULT_RESPONSE = ("rt", "response")
DEFAULT_CHOICES = (0, 1)
DEFAULT_BOUNDS = {
    "alpha": (0.0, 1.0),
    "beta": (0.0, 1.0),
    "gamma": (0.0, 1.0),
    "v": (-3.0, 3.0),
    "a": (0.3, 2.5),
}


# Module-level annotated dummy used wherever from_rlssm_dict needs a valid
# ssm_logp_func but the test is not about ssm_logp_func itself.
@annotate_function(inputs=["v", "rt", "response"], outputs=["logp"], computed={})
def _module_dummy_ssm_logp(x):
    return x


# Helper function to create a config dictionary
def create_config_dict(
    model_name,
    list_params,
    params_default,
    bounds=DEFAULT_BOUNDS,
    response=DEFAULT_RESPONSE,
    choices=DEFAULT_CHOICES,
    extra_fields=[],
    learning_process={},
    decision_process="ddm",
    decision_process_loglik_kind="analytical",
    learning_process_kind="blackbox",
    ssm_logp_func=_module_dummy_ssm_logp,
):
    return dict(
        model_name=model_name,
        name=model_name,
        description=f"{model_name} model",
        list_params=list_params,
        params_default=params_default,
        bounds=bounds,
        response=response,
        choices=choices,
        extra_fields=extra_fields,
        learning_process=learning_process,
        decision_process=decision_process,
        decision_process_loglik_kind=decision_process_loglik_kind,
        learning_process_kind=learning_process_kind,
        ssm_logp_func=ssm_logp_func,
        data={},
    )


# region fixtures and helpers
@pytest.fixture
def valid_rlssmconfig_kwargs():
    @annotate_function(inputs=["v", "rt", "response"], outputs=["logp"], computed={})
    def _dummy_ssm_logp_func(x):
        return x

    return dict(
        model_name="test_model",
        list_params=["alpha", "beta"],
        params_default=[0.5, 0.3],
        bounds={"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
        decision_process="ddm",
        response=["rt", "response"],
        choices=[0, 1],
        extra_fields=["feedback"],
        decision_process_loglik_kind="analytical",
        learning_process_kind="blackbox",
        learning_process={},
        ssm_logp_func=_dummy_ssm_logp_func,
    )


hssm.set_floatX("float32")


def v_func(x):
    return x * 2


def a_func(x):
    return x + 1


# endregion


class TestRLSSMConfigCreation:
    rlwm_config = create_config_dict(
        model_name="rlwm",
        list_params=["alpha", "beta", "gamma", "v", "a"],
        params_default=[0.5, 0.3, 0.2, 1.0, 1.5],
        bounds=DEFAULT_BOUNDS,
        response=DEFAULT_RESPONSE,
        choices=DEFAULT_CHOICES,
        extra_fields=["feedback", "trial_id", "block"],
        learning_process={"v": "subject_wise_function"},
    )

    minimal_rlssm_config = create_config_dict(
        model_name="minimal_rlssm",
        list_params=["alpha", "beta"],
        params_default=[],
        bounds={},
        response=DEFAULT_RESPONSE,
        choices=DEFAULT_CHOICES,
        extra_fields=[],
        learning_process={},
    )

    testcase1 = (
        "rlwm",
        rlwm_config,
        "rlwm",
        [0.5, 0.3, 0.2, 1.0, 1.5],
        DEFAULT_BOUNDS,
        DEFAULT_RESPONSE,
        DEFAULT_CHOICES,
        {"v": "subject_wise_function"},
    )

    testcase2: tuple = (
        "minimal_rlssm",
        minimal_rlssm_config,
        "minimal_rlssm",
        [],
        {},
        DEFAULT_RESPONSE,
        DEFAULT_CHOICES,
        {},
    )
    testcase_params = (
        "model_name, config_dict, expected_model_name,"
        " expected_params_default, expected_bounds, expected_response, "
        "expected_choices, expected_learning_process"
    )

    @pytest.mark.parametrize(
        testcase_params,
        [
            # Test case for RLWM model
            testcase1,
            # Test case for minimal RLSSM model
            testcase2,
        ],
    )
    def test_from_rlssm_dict_cases(
        self,
        model_name,
        config_dict,
        expected_model_name,
        expected_params_default,
        expected_bounds,
        expected_response,
        expected_choices,
        expected_learning_process,
    ):
        config = RLSSMConfig.from_rlssm_dict(config_dict)
        assert config.model_name == expected_model_name
        assert config.params_default == expected_params_default
        assert config.bounds == expected_bounds
        assert config.response == expected_response
        assert config.choices == expected_choices
        assert config.learning_process == expected_learning_process


class _FakeSsMsModelConfig:
    def __init__(self, *, gradient="available"):
        self.gradient = gradient
        self.validated_data = None

    def validate(self):
        return None

    def compile(self, backend="auto"):
        return _FakeSsMsCompiledModel(self, backend=backend, gradient=self.gradient)

    def validate_data(self, data):
        self.validated_data = data

        class _Report:
            def raise_for_errors(self):
                return None

        return _Report()


class _FakeSsMsCompiledModel:
    def __init__(self, config, *, backend, gradient):
        self.config = config
        self.learning_backend = backend
        self.gradient = gradient
        self.model_name = "2AB_RW_Angle"
        self.decision_process = "angle"
        self.list_params = ["rl_alpha", "scaler", "a", "z", "t", "theta"]
        self.bounds = {
            "rl_alpha": (0.0, 1.0),
            "scaler": (0.001, 10.0),
            "a": (0.3, 3.0),
            "z": (0.1, 0.9),
            "t": (0.001, 2.0),
            "theta": (-0.1, 1.3),
        }
        self.params_default = [0.2, 2.0, 1.5, 0.5, 0.3, 0.2]
        self.response = ["rt", "response"]
        self.choices = (-1, 1)
        self.context_fields = ["feedback"]
        self.computed_params = ["v"]
        self.response_to_choice = {-1: 0, 1: 1}

    def participant_input_fields(self):
        return ["rl_alpha", "scaler", "response", "feedback"]

    def compile_participant_fn(self, output="array"):
        assert output == "dict"

        def compute(subject_trials):
            responses = subject_trials[:, 2]
            actions = jnp.where(responses == -1, 0.0, 1.0)
            return {"v": subject_trials[:, 1] * actions}

        return compute


def _install_fake_ssms_rl(monkeypatch, *, gradient="available"):
    config = _FakeSsMsModelConfig(gradient=gradient)
    fake_rl = types.SimpleNamespace(
        ModelConfig=_FakeSsMsModelConfig,
        resolve_model=lambda model: config,
    )
    monkeypatch.setitem(sys.modules, "ssms.rl", fake_rl)
    return config


def _fake_matrix_logp(_model):
    def logp(lan_matrix):
        return jnp.sum(lan_matrix, axis=1)

    return logp


class TestRLSSMConfigFromSsMsModel:
    def test_from_ssms_model_builds_hssm_config_from_compiled_metadata(
        self, monkeypatch
    ):
        _install_fake_ssms_rl(monkeypatch)
        monkeypatch.setattr(
            "hssm.rl.config.make_jax_matrix_logp_funcs_from_onnx",
            _fake_matrix_logp,
            raising=False,
        )

        config = RLSSMConfig.from_ssms_model("2AB_RW_Angle")

        assert config.model_name == "2AB_RW_Angle"
        assert config.decision_process == "angle"
        assert config.decision_process_loglik_kind == "approx_differentiable"
        assert config.learning_process_kind == "approx_differentiable"
        assert config.list_params == ["rl_alpha", "scaler", "a", "z", "t", "theta"]
        assert "rl.alpha" not in config.list_params
        assert "Z" not in config.list_params
        assert config.response == ["rt", "response"]
        assert config.choices == (-1, 1)
        assert config.extra_fields == ["feedback"]
        assert set(config.ssm_logp_func.computed) == {"v"}
        assert config.ssm_logp_func.inputs == [
            "v",
            "a",
            "z",
            "t",
            "theta",
            "rt",
            "response",
        ]
        assert config.ssm_logp_func.computed["v"].inputs == [
            "rl_alpha",
            "scaler",
            "response",
            "feedback",
        ]
        assert config._ssms_response_to_choice == {-1: 0, 1: 1}

    def test_compiled_compute_function_uses_ssms_response_to_choice(self, monkeypatch):
        _install_fake_ssms_rl(monkeypatch)
        monkeypatch.setattr(
            "hssm.rl.config.make_jax_matrix_logp_funcs_from_onnx",
            _fake_matrix_logp,
            raising=False,
        )
        config = RLSSMConfig.from_ssms_model("2AB_RW_Angle")

        compute_v = config.ssm_logp_func.computed["v"]
        subject_trials = jnp.asarray(
            [
                [0.2, 2.0, -1.0, 1.0],
                [0.2, 2.0, 1.0, 0.0],
            ]
        )

        assert jnp.allclose(compute_v(subject_trials), jnp.asarray([0.0, 2.0]))

    def test_from_ssms_model_rejects_unavailable_gradient(self, monkeypatch):
        _install_fake_ssms_rl(monkeypatch, gradient="unavailable")
        monkeypatch.setattr(
            "hssm.rl.config.make_jax_matrix_logp_funcs_from_onnx",
            _fake_matrix_logp,
            raising=False,
        )

        with pytest.raises(ValueError, match="gradient support"):
            RLSSMConfig.from_ssms_model("2AB_RW_Angle")

    def test_from_ssms_model_supports_multiple_computed_params(self, monkeypatch):
        """The bridge must wire every compiled computed decision parameter.

        Generic models can compute more than a single drift parameter (e.g.
        ``drift -> v`` and ``threshold -> a``) and may carry context fields
        beyond ``feedback`` (e.g. ``condition``). The bridge must not assume
        Rescorla-Wagner, a single computed ``v``, or a privileged outcome
        column.
        """

        class _MultiParamCompiled(_FakeSsMsCompiledModel):
            def __init__(self, config, *, backend, gradient):
                super().__init__(config, backend=backend, gradient=gradient)
                self.computed_params = ["v", "a"]
                self.context_fields = ["feedback", "condition"]

            def participant_input_fields(self):
                return ["rl_alpha", "scaler", "response", "feedback", "condition"]

            def compile_participant_fn(self, output="array"):
                assert output == "dict"

                def compute(subject_trials):
                    return {
                        "v": subject_trials[:, 1],
                        "a": subject_trials[:, 0],
                    }

                return compute

        class _MultiParamConfig(_FakeSsMsModelConfig):
            def compile(self, backend="auto"):
                return _MultiParamCompiled(
                    self, backend=backend, gradient=self.gradient
                )

        fake_rl = types.SimpleNamespace(
            ModelConfig=_MultiParamConfig,
            resolve_model=lambda model: _MultiParamConfig(),
        )
        monkeypatch.setitem(sys.modules, "ssms.rl", fake_rl)
        monkeypatch.setattr(
            "hssm.rl.config.make_jax_matrix_logp_funcs_from_onnx",
            _fake_matrix_logp,
            raising=False,
        )

        config = RLSSMConfig.from_ssms_model("multi_param_model")

        # Both computed decision parameters are exposed to the annotated path.
        assert set(config.ssm_logp_func.computed) == {"v", "a"}
        # Context fields flow through HSSM's extra_fields plumbing, untouched —
        # `condition` is not treated specially and `trial_id` is not injected.
        assert config.extra_fields == ["feedback", "condition"]
        for computed_fn in config.ssm_logp_func.computed.values():
            assert computed_fn.inputs == [
                "rl_alpha",
                "scaler",
                "response",
                "feedback",
                "condition",
            ]


class TestRLSSMConfigFromRealSsMs:
    """Integration tests against a real, rl-capable ``ssm-simulators``.

    These exercise the live ``ssms.rl`` handshake (not a stub), so they guard
    against silent contract drift like the Milestone-4 rename of
    ``extra_fields``/``response_to_action`` to ``context_fields``/
    ``response_to_choice``. They skip automatically when the installed
    ``ssm-simulators`` does not yet ship the RLSSM/JAX module (HSSM's pinned
    release lags the ssms RLSSM feature branch).
    """

    @staticmethod
    def _require_ssms_rl():
        ssms_rl = pytest.importorskip(
            "ssms.rl", reason="installed ssm-simulators has no ssms.rl module"
        )
        if not hasattr(ssms_rl, "resolve_model"):
            pytest.skip("ssms.rl does not expose resolve_model")
        return ssms_rl

    def test_from_ssms_model_real_2ab_rw_angle(self):
        self._require_ssms_rl()

        config = RLSSMConfig.from_ssms_model("2AB_RW_Angle")
        config.validate()

        assert config.model_name == "2AB_RW_Angle"
        assert config.decision_process == "angle"
        # Canonical ssms parameter / response vocabulary.
        assert "rl_alpha" in config.list_params
        assert config.response == ["rt", "response"]
        # context_fields surface through HSSM extra_fields, response mapping
        # is keyed on raw SSM response labels.
        assert config.extra_fields == ["feedback"]
        assert config._ssms_response_to_choice == {-1: 0, 1: 1}
        assert set(config.ssm_logp_func.computed) == {"v"}


class TestRLSSMConfigValidation:
    @pytest.mark.parametrize(
        "field, value, error_msg",
        [
            ("response", None, "Please provide `response` columns"),
            ("list_params", None, "Please provide `list_params"),
            ("choices", None, "Please provide `choices`"),
            ("decision_process", None, "Please specify a `decision_process`"),
            ("ssm_logp_func", None, "Please provide `ssm_logp_func`"),
        ],
    )
    def test_validate_missing_fields(
        self, field, value, error_msg, valid_rlssmconfig_kwargs
    ):
        # All required fields provided, then set one to None
        config = RLSSMConfig(**valid_rlssmconfig_kwargs)
        setattr(config, field, value)
        with pytest.raises(ValueError, match=error_msg):
            config.validate()

    @pytest.mark.parametrize(
        "missing_field",
        [
            "model_name",
            "params_default",
            "decision_process",
            "decision_process_loglik_kind",
            "learning_process_kind",
            "learning_process",
        ],
    )
    def test_constructor_missing_required_field(
        self, missing_field, valid_rlssmconfig_kwargs
    ):
        # Provide all required fields, then remove one
        kwargs = valid_rlssmconfig_kwargs
        kwargs.pop(missing_field)
        with pytest.raises(TypeError):
            RLSSMConfig(**kwargs)

    def test_validate_success(self, valid_rlssmconfig_kwargs):
        config = RLSSMConfig(**valid_rlssmconfig_kwargs)
        config.validate()

    def test_validate_params_default_mismatch(self, valid_rlssmconfig_kwargs):
        config = RLSSMConfig(
            **{
                **valid_rlssmconfig_kwargs,
                "params_default": [0.5],  # length 1, but list_params has 2 entries
            }
        )
        with pytest.raises(
            ValueError,
            match=r"params_default length \(1\) doesn't match list_params length \(2\)",
        ):
            config.validate()

    def test_validate_ssm_logp_func_not_callable(self, valid_rlssmconfig_kwargs):
        config = RLSSMConfig(**valid_rlssmconfig_kwargs)
        config.ssm_logp_func = "not_a_callable"
        with pytest.raises(ValueError, match="must be a callable"):
            config.validate()

    def test_validate_ssm_logp_func_missing_annotations(self, valid_rlssmconfig_kwargs):
        config = RLSSMConfig(**valid_rlssmconfig_kwargs)
        # Replace with a plain callable that lacks @annotate_function attributes
        config.ssm_logp_func = lambda x: x
        with pytest.raises(
            ValueError, match="must be decorated with `@annotate_function`"
        ):
            config.validate()

    def test_validate_ssm_logp_func_computed_not_callable(
        self, valid_rlssmconfig_kwargs
    ):
        """`computed` exists but contains non-callable values -> error."""
        config = RLSSMConfig(**valid_rlssmconfig_kwargs)
        # Inject a computed mapping with a non-callable value to trigger the
        # specific validation branch.
        config.ssm_logp_func.computed = {"x": "not_callable"}
        with pytest.raises(
            ValueError,
            match=r"`ssm_logp_func.computed` must be a dictionary with callable values\.",
        ):
            config.validate()

    def test_validate_missing_bounds_for_param(self, valid_rlssmconfig_kwargs):
        """validate() should raise early when a param has no bounds entry."""
        kwargs = {**valid_rlssmconfig_kwargs, "bounds": {}}  # strip all bounds
        config = RLSSMConfig(**kwargs)
        with pytest.raises(ValueError, match="Missing bounds for parameter"):
            config.validate()

    def test_from_defaults_raises(self):
        """RLSSMConfig.from_defaults() must raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="from_defaults"):
            RLSSMConfig.from_defaults("ddm", None)


class TestRLSSMConfigDefaults:
    @pytest.mark.parametrize(
        "list_params, params_default, bounds, param, expected_default, expected_bounds",
        [
            # params_default stores initialisation values, NOT priors.
            # get_defaults always returns None for the prior so that
            # prior_settings="safe" can assign priors from bounds.
            #
            # Case 1: queried param is present in bounds → bound returned.
            (
                ["alpha", "beta", "gamma"],
                [0.5, 0.3, 0.2],
                {"alpha": (0.0, 1.0), "beta": (0.0, 1.0), "gamma": (0.0, 1.0)},
                "beta",
                None,
                (0.0, 1.0),
            ),
            # Case 2: queried param is NOT in list_params and NOT in bounds
            # (e.g. a typo or an extra lookup) → both None.
            (
                ["alpha", "beta"],
                [0.5, 0.3],
                {"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
                "gamma",
                None,
                None,
            ),
            # Case 3: params_default may be empty; param in bounds → bound returned.
            (
                ["alpha", "beta"],
                [],
                {"alpha": (0.0, 1.0), "beta": (0.0, 1.0)},
                "alpha",
                None,
                (0.0, 1.0),
            ),
        ],
    )
    def test_get_defaults_cases(
        self,
        list_params,
        params_default,
        bounds,
        param,
        expected_default,
        expected_bounds,
    ):
        config = RLSSMConfig(
            model_name="test_model",
            list_params=list_params,
            params_default=params_default,
            bounds=bounds,
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            decision_process_loglik_kind="analytical",
            learning_process_kind="blackbox",
            learning_process={},
        )
        default_val, bounds_val = config.get_defaults(param)
        assert default_val == expected_default
        assert bounds_val == expected_bounds


class TestRLSSMConfigLearningProcess:
    def test_learning_process(self):
        config = RLSSMConfig(
            model_name="test_model",
            list_params=["alpha"],
            params_default=[0.0],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            learning_process={"v": v_func, "a": a_func},
            decision_process_loglik_kind="analytical",
            learning_process_kind="blackbox",
        )
        assert "v" in config.learning_process
        assert "a" in config.learning_process
        assert config.learning_process["v"] is v_func
        assert config.learning_process["a"] is a_func

    def test_immutable_defaults(self):
        config1 = RLSSMConfig(
            model_name="model1",
            list_params=["alpha"],
            params_default=[0.0],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            learning_process={"v": v_func},
            decision_process_loglik_kind="analytical",
            learning_process_kind="blackbox",
        )
        config2 = RLSSMConfig(
            model_name="model2",
            list_params=["beta"],
            params_default=[0.0],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            learning_process={"a": a_func},
            decision_process_loglik_kind="analytical",
            learning_process_kind="blackbox",
        )
        config1.learning_process["v"] = "function1"
        assert "v" not in config2.learning_process
        assert config1.learning_process != config2.learning_process


class TestRLSSMConfigEdgeCases:
    def test_from_rlssm_dict_missing_required(self):
        # Should raise ValueError if decision_process_loglik_kind is missing
        config_dict = {
            "model_name": "test_model",
            "name": "test_model",
            "list_params": ["alpha"],
            "params_default": [0.0],
            "decision_process": "ddm",
            "learning_process": {},
            "learning_process_kind": "blackbox",
            "response": ["rt", "response"],
            "choices": [0, 1],
            "description": "desc",
            "bounds": {},
            "data": {},
            "extra_fields": [],
            "ssm_logp_func": _module_dummy_ssm_logp,
        }
        with pytest.raises(
            ValueError, match="decision_process_loglik_kind must be provided"
        ):
            RLSSMConfig.from_rlssm_dict(config_dict)

    def test_from_rlssm_dict_missing_ssm_logp_func(self):
        # Should raise ValueError at construction time if ssm_logp_func is missing
        config_dict = {
            "model_name": "test_model",
            "name": "test_model",
            "list_params": ["alpha"],
            "params_default": [0.0],
            "decision_process": "ddm",
            "learning_process": {},
            "learning_process_kind": "blackbox",
            "decision_process_loglik_kind": "analytical",
            "response": ["rt", "response"],
            "choices": [0, 1],
            "description": "desc",
            "bounds": {},
            "data": {},
            "extra_fields": [],
        }
        with pytest.raises(ValueError, match="ssm_logp_func must be provided"):
            RLSSMConfig.from_rlssm_dict(config_dict)

    def test_missing_decision_process_loglik_kind(self):
        with pytest.raises(TypeError):
            RLSSMConfig(
                model_name="test_model",
                list_params=["alpha"],
                decision_process="ddm",
                response=["rt", "response"],
                choices=[0, 1],
            )
        config_dict = {
            "model_name": "test_model",
            "description": "desc",
            "list_params": ["alpha"],
            "params_default": [0.0],
            "bounds": {},
            "data": {},
            "decision_process": "ddm",
            "learning_process": {},
            "learning_process_kind": "blackbox",
            "response": ["rt", "response"],
            "choices": [0, 1],
            "extra_fields": [],
            "ssm_logp_func": _module_dummy_ssm_logp,
        }
        with pytest.raises(
            ValueError, match="decision_process_loglik_kind must be provided"
        ):
            RLSSMConfig.from_rlssm_dict(config_dict)

    def test_with_modelconfig_decision_process(self):
        decision_config = ModelConfig(
            response=["rt", "response"],
            list_params=["v", "a", "z", "t"],
            choices=[0, 1],
        )
        RLSSMConfig(
            model_name="test_model",
            list_params=["alpha"],
            params_default=[0.0],
            decision_process=decision_config,
            response=["rt", "response"],
            choices=[0, 1],
            decision_process_loglik_kind="analytical",
            learning_process_kind="blackbox",
            learning_process={},
        )


class TestRLSSMConfigDefaultWarnings:
    """Warnings are emitted when 'response' or 'choices' are missing from config_dict."""

    @pytest.fixture
    def _base_config_dict(self):
        return {
            "model_name": "test_model",
            "list_params": ["alpha"],
            "params_default": [0.0],
            "bounds": {"alpha": (0.0, 1.0)},
            "decision_process": "ddm",
            "learning_process": {},
            "learning_process_kind": "blackbox",
            "decision_process_loglik_kind": "analytical",
            "extra_fields": [],
            "ssm_logp_func": _module_dummy_ssm_logp,
        }

    def test_warns_when_response_missing(self, _base_config_dict, caplog):
        _base_config_dict["choices"] = (0, 1)
        # 'response' deliberately omitted
        with caplog.at_level("WARNING", logger="hssm"):
            config = RLSSMConfig.from_rlssm_dict(_base_config_dict)
        assert any("'response' not specified" in m for m in caplog.messages)
        assert config.response == list(DEFAULT_SSM_OBSERVED_DATA)

    def test_warns_when_choices_missing(self, _base_config_dict, caplog):
        _base_config_dict["response"] = ["rt", "response"]
        # 'choices' deliberately omitted
        with caplog.at_level("WARNING", logger="hssm"):
            config = RLSSMConfig.from_rlssm_dict(_base_config_dict)
        assert any("'choices' not specified" in m for m in caplog.messages)
        assert config.choices == DEFAULT_SSM_CHOICES

    def test_warns_when_both_missing(self, _base_config_dict, caplog):
        # Both 'response' and 'choices' omitted
        with caplog.at_level("WARNING", logger="hssm"):
            config = RLSSMConfig.from_rlssm_dict(_base_config_dict)
        assert any("'response' not specified" in m for m in caplog.messages)
        assert any("'choices' not specified" in m for m in caplog.messages)
        assert config.response == list(DEFAULT_SSM_OBSERVED_DATA)
        assert config.choices == DEFAULT_SSM_CHOICES

    def test_no_warning_when_both_provided(self, _base_config_dict, caplog):
        _base_config_dict["response"] = ["rt", "response"]
        _base_config_dict["choices"] = (0, 1)
        with caplog.at_level("WARNING", logger="hssm"):
            RLSSMConfig.from_rlssm_dict(_base_config_dict)
        assert not any("not specified" in m for m in caplog.messages)

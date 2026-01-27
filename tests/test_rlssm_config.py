import copy


@pytest.fixture
def valid_rlssmconfig_kwargs():
    return dict(
        model_name="test_model",
        list_params=["alpha", "beta"],
        params_default=[0.5, 0.3],
        decision_process="ddm",
        response=["rt", "response"],
        choices=[0, 1],
        extra_fields=["feedback"],
        decision_process_loglik_kind="analytical",
        learning_process_loglik_kind="blackbox",
        learning_process={},
    )


"""Tests for RLSSMConfig class."""

import pytest
import hssm
from hssm.config import Config, RLSSMConfig

hssm.set_floatX("float32")


def v_func(x):
    return x * 2


def a_func(x):
    return x + 1


class TestRLSSMConfigCreation:
    @pytest.mark.parametrize(
        "model_name, config_dict, expected_model_name, expected_params_default, expected_bounds, expected_response, expected_choices, expected_learning_process",
        [
            (
                "rlwm",
                {
                    "name": "rlwm",
                    "description": "Reinforcement Learning Working Memory model",
                    "list_params": ["alpha", "beta", "gamma", "v", "a"],
                    "extra_fields": ["feedback", "trial_id", "block"],
                    "decision_process": "ddm",
                    "params_default": [0.5, 0.3, 0.2, 1.0, 1.5],
                    "bounds": {
                        "alpha": (0.0, 1.0),
                        "beta": (0.0, 1.0),
                        "gamma": (0.0, 1.0),
                        "v": (-3.0, 3.0),
                        "a": (0.3, 2.5),
                    },
                    "response": ["rt", "response"],
                    "choices": [0, 1],
                    "learning_process": {"v": "subject_wise_function"},
                    "decision_process_loglik_kind": "analytical",
                    "learning_process_loglik_kind": "blackbox",
                },
                "rlwm",
                [0.5, 0.3, 0.2, 1.0, 1.5],
                {
                    "alpha": (0.0, 1.0),
                    "beta": (0.0, 1.0),
                    "gamma": (0.0, 1.0),
                    "v": (-3.0, 3.0),
                    "a": (0.3, 2.5),
                },
                ["rt", "response"],
                [0, 1],
                {"v": "subject_wise_function"},
            ),
            (
                "minimal_rlssm",
                {
                    "name": "minimal_rlssm",
                    "description": "Minimal RLSSM model",
                    "list_params": ["alpha", "beta"],
                    "decision_process": "ddm",
                    "decision_process_loglik_kind": "analytical",
                },
                "minimal_rlssm",
                [],
                {},
                ["rt", "response"],
                [0, 1],
                {},
            ),
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
        config = RLSSMConfig.from_rlssm_dict(model_name, config_dict)
        assert config.model_name == expected_model_name
        assert config.params_default == expected_params_default
        assert config.bounds == expected_bounds
        assert config.response == expected_response
        assert config.choices == expected_choices
        assert config.learning_process == expected_learning_process


class TestRLSSMConfigValidation:
    @pytest.mark.parametrize(
        "field, value, error_msg",
        [
            ("response", None, "Please provide `response` columns"),
            ("list_params", None, "Please provide `list_params"),
            ("choices", None, "Please provide `choices"),
            ("decision_process", None, "Please specify a `decision_process"),
        ],
    )
    def test_validate_missing_fields(self, field, value, error_msg):
        # All required fields provided, then set one to None
        config = RLSSMConfig(
            model_name="test_model",
            list_params=["alpha"],
            params_default=[0.0],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            decision_process_loglik_kind="analytical",
            learning_process_loglik_kind="blackbox",
            learning_process={},
        )
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
            "learning_process_loglik_kind",
            "learning_process",
        ],
    )
    def test_constructor_missing_required_field(self, missing_field):
        # Provide all required fields, then remove one
        kwargs = dict(
            model_name="test_model",
            list_params=["alpha"],
            params_default=[0.0],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            decision_process_loglik_kind="analytical",
            learning_process_loglik_kind="blackbox",
            learning_process={},
        )

        kwargs.pop(missing_field)

        with pytest.raises(TypeError):
            RLSSMConfig(**kwargs)

    def test_validate_success(self):
        config = RLSSMConfig(
            model_name="test_model",
            list_params=["alpha", "beta"],
            params_default=[0.5, 0.3],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            extra_fields=["feedback"],
            decision_process_loglik_kind="analytical",
        )
        config.validate()

    def test_validate_params_default_mismatch(self):
        config = RLSSMConfig(
            model_name="test_model",
            list_params=["alpha", "beta"],
            params_default=[0.5],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            decision_process_loglik_kind="analytical",
        )
        with pytest.raises(
            ValueError,
            match=r"params_default length \(1\) doesn't match list_params length \(2\)",
        ):
            config.validate()


class TestRLSSMConfigDefaults:
    @pytest.mark.parametrize(
        "list_params, params_default, bounds, param, expected_default, expected_bounds",
        [
            (
                ["alpha", "beta", "gamma"],
                [0.5, 0.3, 0.2],
                {"beta": (0.0, 1.0)},
                "beta",
                0.3,
                (0.0, 1.0),
            ),
            (["alpha", "beta"], [0.5, 0.3], {"alpha": (0.0, 1.0)}, "gamma", None, None),
            (["alpha", "beta"], [], {"alpha": (0.0, 1.0)}, "alpha", None, (0.0, 1.0)),
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
        )
        default_val, bounds_val = config.get_defaults(param)
        assert default_val == expected_default
        assert bounds_val == expected_bounds


class TestRLSSMConfigConversion:
    @pytest.mark.parametrize(
        "list_params, params_default, backend, expected_backend, expected_default_priors, raises",
        [
            (
                ["alpha", "beta", "v", "a"],
                [0.5, 0.3, 1.0, 1.5],
                "jax",
                "jax",
                {"alpha": 0.5, "beta": 0.3, "v": 1.0, "a": 1.5},
                None,
            ),
            (["alpha"], [0.5], None, "jax", {"alpha": 0.5}, None),
            (["alpha", "beta"], [], None, "jax", {}, None),
            (["alpha", "beta", "gamma"], [0.5, 0.3], None, None, None, ValueError),
        ],
    )
    def test_to_config_cases(
        self,
        list_params,
        params_default,
        backend,
        expected_backend,
        expected_default_priors,
        raises,
    ):
        rlssm_config = RLSSMConfig(
            model_name="test_model",
            list_params=list_params,
            params_default=params_default,
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            backend=backend,
            decision_process_loglik_kind="analytical",
        )
        if raises:
            with pytest.raises(raises):
                rlssm_config.to_config()
        else:
            config = rlssm_config.to_config()
            assert config.backend == expected_backend
            assert config.default_priors == expected_default_priors

    def test_to_config(self):
        rlssm_config = RLSSMConfig(
            model_name="rlwm",
            description="RLWM model",
            list_params=["alpha", "beta", "v", "a"],
            params_default=[0.5, 0.3, 1.0, 1.5],
            bounds={
                "alpha": (0.0, 1.0),
                "beta": (0.0, 1.0),
                "v": (-3.0, 3.0),
                "a": (0.3, 2.5),
            },
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            extra_fields=["feedback"],
            backend="jax",
            decision_process_loglik_kind="analytical",
        )
        config = rlssm_config.to_config()
        assert isinstance(config, Config)
        assert config.model_name == "rlwm"
        assert config.description == "RLWM model"
        assert config.list_params == ["alpha", "beta", "v", "a"]
        assert config.response == ["rt", "response"]
        assert config.choices == [0, 1]
        assert config.extra_fields == ["feedback"]
        assert config.backend == "jax"
        assert config.loglik_kind == "approx_differentiable"
        assert config.bounds == {
            "alpha": (0.0, 1.0),
            "beta": (0.0, 1.0),
            "v": (-3.0, 3.0),
            "a": (0.3, 2.5),
        }
        assert config.default_priors == {
            "alpha": 0.5,
            "beta": 0.3,
            "v": 1.0,
            "a": 1.5,
        }

    def test_to_config_defaults_backend(self):
        rlssm_config = RLSSMConfig(
            model_name="test_model",
            list_params=["alpha"],
            params_default=[0.5],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            decision_process_loglik_kind="analytical",
        )
        config = rlssm_config.to_config()
        assert config.backend == "jax"

    def test_to_config_no_defaults(self):
        rlssm_config = RLSSMConfig(
            model_name="test_model",
            list_params=["alpha", "beta"],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            decision_process_loglik_kind="analytical",
        )
        config = rlssm_config.to_config()
        assert config.default_priors == {}

    def test_to_config_mismatched_defaults_length(self):
        rlssm_config = RLSSMConfig(
            model_name="test_model",
            list_params=["alpha", "beta", "gamma"],
            params_default=[0.5, 0.3],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            decision_process_loglik_kind="analytical",
        )
        with pytest.raises(
            ValueError,
            match=r"params_default length \(2\) doesn't match list_params length \(3\)",
        ):
            rlssm_config.to_config()


class TestRLSSMConfigLearningProcess:
    def test_learning_process(self):
        config = RLSSMConfig(
            model_name="test_model",
            list_params=["alpha"],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            learning_process={"v": v_func, "a": a_func},
            decision_process_loglik_kind="analytical",
        )
        assert "v" in config.learning_process
        assert "a" in config.learning_process
        assert config.learning_process["v"] is v_func
        assert config.learning_process["a"] is a_func

    def test_immutable_defaults(self):
        config1 = RLSSMConfig(
            model_name="model1",
            list_params=["alpha"],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            learning_process={"v": v_func},
            decision_process_loglik_kind="analytical",
        )
        config2 = RLSSMConfig(
            model_name="model2",
            list_params=["beta"],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            learning_process={"a": a_func},
            decision_process_loglik_kind="analytical",
        )
        config1.learning_process["v"] = "function1"
        assert "v" not in config2.learning_process
        assert config1.learning_process != config2.learning_process


class TestRLSSMConfigEdgeCases:
    def test_with_modelconfig_decision_process(self):
        from hssm.config import ModelConfig

        decision_config = ModelConfig(
            response=["rt", "response"],
            list_params=["v", "a", "z", "t"],
            choices=[0, 1],
        )
        config = RLSSMConfig(
            model_name="test_model",
            list_params=["alpha"],
            decision_process=decision_config,
            response=["rt", "response"],
            choices=[0, 1],
            decision_process_loglik_kind="analytical",
        )

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
            "name": "test_model",
            "list_params": ["alpha"],
            "decision_process": "ddm",
        }
        with pytest.raises(
            ValueError, match="decision_process_loglik_kind must be provided"
        ):
            RLSSMConfig.from_rlssm_dict("test_model", config_dict)

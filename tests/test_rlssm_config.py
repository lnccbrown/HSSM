import pytest

import hssm
from hssm.config import Config, RLSSMConfig
from hssm.config import ModelConfig

# Define constants for repeated data structures
DEFAULT_RESPONSE = ["rt", "response"]
DEFAULT_CHOICES = [0, 1]
DEFAULT_BOUNDS = {
    "alpha": (0.0, 1.0),
    "beta": (0.0, 1.0),
    "gamma": (0.0, 1.0),
    "v": (-3.0, 3.0),
    "a": (0.3, 2.5),
}


# Helper function to create a config dictionary
def create_config_dict(
    model_name,
    list_params,
    params_default,
    bounds,
    response,
    choices,
    extra_fields,
    learning_process,
):
    return {
        "model_name": model_name,
        "name": model_name,
        "description": f"{model_name} model",
        "list_params": list_params,
        "params_default": params_default,
        "bounds": bounds,
        "response": response,
        "choices": choices,
        "extra_fields": extra_fields,
        "learning_process": learning_process,
        "decision_process": "ddm",
        "decision_process_loglik_kind": "analytical",
        "learning_process_loglik_kind": "blackbox",
        "data": {},
    }


# region fixtures and helpers
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

    testcase2 = (
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
            "learning_process_loglik_kind",
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

    def test_validate_params_default_mismatch(self):
        config = RLSSMConfig(
            model_name="test_model",
            list_params=["alpha", "beta"],
            params_default=[0.5],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            decision_process_loglik_kind="analytical",
            learning_process_loglik_kind="blackbox",
            learning_process={},
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
            learning_process_loglik_kind="blackbox",
            learning_process={},
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
            learning_process_loglik_kind="blackbox",
            learning_process={},
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
            learning_process_loglik_kind="blackbox",
            learning_process={},
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
            learning_process_loglik_kind="blackbox",
            learning_process={},
        )
        config = rlssm_config.to_config()
        assert config.backend == "jax"

    def test_to_config_no_defaults(self):
        rlssm_config = RLSSMConfig(
            model_name="test_model",
            list_params=["alpha", "beta"],
            params_default=[],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            decision_process_loglik_kind="analytical",
            learning_process_loglik_kind="blackbox",
            learning_process={},
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
            learning_process_loglik_kind="blackbox",
            learning_process={},
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
            params_default=[0.0],
            decision_process="ddm",
            response=["rt", "response"],
            choices=[0, 1],
            learning_process={"v": v_func, "a": a_func},
            decision_process_loglik_kind="analytical",
            learning_process_loglik_kind="blackbox",
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
            learning_process_loglik_kind="blackbox",
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
            learning_process_loglik_kind="blackbox",
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
            "learning_process_loglik_kind": "blackbox",
            "response": ["rt", "response"],
            "choices": [0, 1],
            "description": "desc",
            "bounds": {},
            "data": {},
            "extra_fields": [],
        }
        with pytest.raises(
            ValueError, match="decision_process_loglik_kind must be provided"
        ):
            RLSSMConfig.from_rlssm_dict("test_model", config_dict)

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
            "learning_process_loglik_kind": "blackbox",
            "response": ["rt", "response"],
            "choices": [0, 1],
            "extra_fields": [],
        }
        with pytest.raises(
            ValueError, match="decision_process_loglik_kind must be provided"
        ):
            RLSSMConfig.from_rlssm_dict("test_model", config_dict)

    def test_with_modelconfig_decision_process(self):
        decision_config = ModelConfig(
            response=["rt", "response"],
            list_params=["v", "a", "z", "t"],
            choices=[0, 1],
        )
        config = RLSSMConfig(
            model_name="test_model",
            list_params=["alpha"],
            params_default=[0.0],
            decision_process=decision_config,
            response=["rt", "response"],
            choices=[0, 1],
            decision_process_loglik_kind="analytical",
            learning_process_loglik_kind="blackbox",
            learning_process={},
        )

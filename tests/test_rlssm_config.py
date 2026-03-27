import pytest

import hssm
from hssm.config import Config, ModelConfig
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
    learning_process_loglik_kind="blackbox",
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
        learning_process_loglik_kind=learning_process_loglik_kind,
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
        learning_process_loglik_kind="blackbox",
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
        config = RLSSMConfig.from_rlssm_dict(config_dict)
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
            ("ssm_logp_func", None, "Please provide `ssm_logp_func"),
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
            learning_process_loglik_kind="blackbox",
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
            "learning_process_loglik_kind": "blackbox",
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
            "learning_process_loglik_kind": "blackbox",
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
            learning_process_loglik_kind="blackbox",
            learning_process={},
        )

"""Tests for RLSSMConfig class."""

import pytest

import hssm
from hssm.config import Config, RLSSMConfig

hssm.set_floatX("float32")


def test_rlssm_config_basic_creation():
    """Test basic RLSSMConfig creation."""
    config = RLSSMConfig(
        model_name="test_rlssm",
        description="Test RLSSM model",
        list_params=["alpha", "beta", "gamma"],
        params_default=[0.5, 0.3, 0.2],
        decision_process="ddm",
        response=["rt", "response"],
        choices=[0, 1],
        extra_fields=["feedback", "trial_id"],
    )

    assert config.model_name == "test_rlssm"
    assert config.description == "Test RLSSM model"
    assert config.list_params == ["alpha", "beta", "gamma"]
    assert config.params_default == [0.5, 0.3, 0.2]
    assert config.decision_process == "ddm"
    assert config.response == ["rt", "response"]
    assert config.choices == [0, 1]
    assert config.n_params == 3
    assert config.n_extra_fields == 2
    assert config.extra_fields == ["feedback", "trial_id"]


def test_rlssm_config_from_rlssm_dict():
    """Test creating RLSSMConfig from dictionary."""
    config_dict = {
        "name": "rlwm",
        "description": "Reinforcement Learning Working Memory model",
        "list_params": ["alpha", "beta", "gamma", "v", "a"],
        "extra_fields": ["feedback", "trial_id", "block"],
        "decision_model": "ddm",
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
    }

    config = RLSSMConfig.from_rlssm_dict("rlwm", config_dict)

    assert config.model_name == "rlwm"
    assert config.description == "Reinforcement Learning Working Memory model"
    assert config.n_params == 5
    assert config.n_extra_fields == 3
    assert config.list_params == ["alpha", "beta", "gamma", "v", "a"]
    assert config.extra_fields == ["feedback", "trial_id", "block"]
    assert config.decision_process == "ddm"
    assert config.params_default == [0.5, 0.3, 0.2, 1.0, 1.5]
    assert config.bounds == {
        "alpha": (0.0, 1.0),
        "beta": (0.0, 1.0),
        "gamma": (0.0, 1.0),
        "v": (-3.0, 3.0),
        "a": (0.3, 2.5),
    }
    assert config.response == ["rt", "response"]
    assert config.choices == [0, 1]
    assert config.learning_process == {"v": "subject_wise_function"}


def test_rlssm_config_from_rlssm_dict_with_defaults():
    """Test creating RLSSMConfig with default values."""
    config_dict = {
        "name": "minimal_rlssm",
        "description": "Minimal RLSSM model",
        "list_params": ["alpha", "beta"],
        "decision_model": "ddm",
    }

    config = RLSSMConfig.from_rlssm_dict("minimal_rlssm", config_dict)

    assert config.model_name == "minimal_rlssm"
    assert config.params_default == []
    assert config.bounds == {}
    assert config.response == ["rt", "response"]  # Default value
    assert config.choices == [0, 1]  # Default value
    assert config.learning_process == {}  # Default value


def test_rlssm_config_validate_success():
    """Test successful validation."""
    config = RLSSMConfig(
        model_name="test_model",
        list_params=["alpha", "beta"],
        params_default=[0.5, 0.3],
        decision_process="ddm",
        response=["rt", "response"],
        choices=[0, 1],
        extra_fields=["feedback"],
    )

    # Should not raise
    config.validate()


def test_rlssm_config_validate_missing_response():
    """Test validation fails when response is missing."""
    config = RLSSMConfig(
        model_name="test_model",
        list_params=["alpha"],
        decision_process="ddm",
        choices=[0, 1],
    )
    config.response = None

    with pytest.raises(ValueError, match="Please provide `response` columns"):
        config.validate()


def test_rlssm_config_validate_missing_list_params():
    """Test validation fails when list_params is missing."""
    config = RLSSMConfig(
        model_name="test_model",
        decision_process="ddm",
        response=["rt", "response"],
        choices=[0, 1],
    )
    config.list_params = None

    with pytest.raises(ValueError, match="Please provide `list_params`"):
        config.validate()


def test_rlssm_config_validate_missing_choices():
    """Test validation fails when choices is missing."""
    config = RLSSMConfig(
        model_name="test_model",
        list_params=["alpha"],
        decision_process="ddm",
        response=["rt", "response"],
    )
    config.choices = None

    with pytest.raises(ValueError, match="Please provide `choices`"):
        config.validate()


def test_rlssm_config_validate_missing_decision_process():
    """Test validation fails when decision_process is missing."""
    config = RLSSMConfig(
        model_name="test_model",
        list_params=["alpha"],
        response=["rt", "response"],
        choices=[0, 1],
    )
    config.decision_process = None

    with pytest.raises(ValueError, match="Please specify a `decision_process`"):
        config.validate()


def test_rlssm_config_validate_params_default_mismatch():
    """Test validation fails when params_default length doesn't match list_params."""
    config = RLSSMConfig(
        model_name="test_model",
        list_params=["alpha", "beta"],
        params_default=[0.5],  # Mismatch: only 1 default for 2 params
        decision_process="ddm",
        response=["rt", "response"],
        choices=[0, 1],
    )

    with pytest.raises(
        ValueError,
        match="params_default length \\(1\\) doesn't match list_params length \\(2\\)",
    ):
        config.validate()


def test_rlssm_config_get_defaults_with_values():
    """Test get_defaults returns correct values and bounds."""
    config = RLSSMConfig(
        model_name="test_model",
        list_params=["alpha", "beta", "gamma"],
        params_default=[0.5, 0.3, 0.2],
        bounds={
            "alpha": (0.0, 1.0),
            "beta": (0.0, 1.0),
            "gamma": (0.0, 1.0),
        },
        decision_process="ddm",
        response=["rt", "response"],
        choices=[0, 1],
    )

    # Test getting defaults for existing parameter
    default_val, bounds = config.get_defaults("beta")
    assert default_val == 0.3
    assert bounds == (0.0, 1.0)

    # Test first parameter
    default_val, bounds = config.get_defaults("alpha")
    assert default_val == 0.5
    assert bounds == (0.0, 1.0)


def test_rlssm_config_get_defaults_missing_param():
    """Test get_defaults for non-existent parameter."""
    config = RLSSMConfig(
        model_name="test_model",
        list_params=["alpha", "beta"],
        params_default=[0.5, 0.3],
        bounds={"alpha": (0.0, 1.0)},
        decision_process="ddm",
        response=["rt", "response"],
        choices=[0, 1],
    )

    # Test getting defaults for non-existent parameter
    default_val, bounds = config.get_defaults("gamma")
    assert default_val is None
    assert bounds is None


def test_rlssm_config_get_defaults_no_defaults():
    """Test get_defaults when params_default is empty."""
    config = RLSSMConfig(
        model_name="test_model",
        list_params=["alpha", "beta"],
        bounds={"alpha": (0.0, 1.0)},
        decision_process="ddm",
        response=["rt", "response"],
        choices=[0, 1],
    )

    # Should still return bounds even without defaults
    default_val, bounds = config.get_defaults("alpha")
    assert default_val is None
    assert bounds == (0.0, 1.0)


def test_rlssm_config_to_config():
    """Test converting RLSSMConfig to Config."""
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
    )

    config = rlssm_config.to_config()

    # Check it's a Config instance
    assert isinstance(config, Config)

    # Check all fields are correctly transferred
    assert config.model_name == "rlwm"
    assert config.description == "RLWM model"
    assert config.list_params == ["alpha", "beta", "v", "a"]
    assert config.response == ["rt", "response"]
    assert config.choices == [0, 1]
    assert config.extra_fields == ["feedback"]
    assert config.backend == "jax"
    assert config.loglik_kind == "approx_differentiable"

    # Check bounds are transferred
    assert config.bounds == {
        "alpha": (0.0, 1.0),
        "beta": (0.0, 1.0),
        "v": (-3.0, 3.0),
        "a": (0.3, 2.5),
    }

    # Check params_default list is converted to default_priors dict
    assert config.default_priors == {
        "alpha": 0.5,
        "beta": 0.3,
        "v": 1.0,
        "a": 1.5,
    }


def test_rlssm_config_to_config_defaults_backend():
    """Test to_config uses default backend when not specified."""
    rlssm_config = RLSSMConfig(
        model_name="test_model",
        list_params=["alpha"],
        params_default=[0.5],
        decision_process="ddm",
        response=["rt", "response"],
        choices=[0, 1],
    )

    config = rlssm_config.to_config()

    # Should default to "jax"
    assert config.backend == "jax"


def test_rlssm_config_to_config_no_defaults():
    """Test to_config when params_default is empty."""
    rlssm_config = RLSSMConfig(
        model_name="test_model",
        list_params=["alpha", "beta"],
        decision_process="ddm",
        response=["rt", "response"],
        choices=[0, 1],
    )

    config = rlssm_config.to_config()

    # default_priors should be empty
    assert config.default_priors == {}


def test_rlssm_config_to_config_mismatched_defaults_length():
    """Test that to_config raises error if params_default length doesn't match list_params."""
    rlssm_config = RLSSMConfig(
        model_name="test_model",
        list_params=["alpha", "beta", "gamma"],
        params_default=[0.5, 0.3],  # Only 2 values for 3 params
        decision_process="ddm",
        response=["rt", "response"],
        choices=[0, 1],
    )

    # Should raise ValueError to prevent silent truncation by zip
    with pytest.raises(
        ValueError,
        match=r"params_default length \(2\) doesn't match list_params length \(3\)",
    ):
        rlssm_config.to_config()


def test_rlssm_config_learning_process():
    """Test learning_process field can be set."""
    config = RLSSMConfig(
        model_name="test_model",
        list_params=["alpha"],
        decision_process="ddm",
        response=["rt", "response"],
        choices=[0, 1],
        learning_process={"v": lambda x: x * 2, "a": lambda x: x + 1},
    )

    assert "v" in config.learning_process
    assert "a" in config.learning_process
    assert callable(config.learning_process["v"])
    assert callable(config.learning_process["a"])


def test_rlssm_config_immutable_defaults():
    """Test that mutable defaults are not shared between instances."""
    config1 = RLSSMConfig(
        model_name="model1",
        list_params=["alpha"],
        decision_process="ddm",
        response=["rt", "response"],
        choices=[0, 1],
    )

    config2 = RLSSMConfig(
        model_name="model2",
        list_params=["beta"],
        decision_process="ddm",
        response=["rt", "response"],
        choices=[0, 1],
    )

    # Modify config1's learning_process
    config1.learning_process["v"] = "function1"

    # config2 should not be affected
    assert "v" not in config2.learning_process
    assert config1.learning_process != config2.learning_process


def test_rlssm_config_with_modelconfig_decision_process():
    """Test RLSSMConfig with ModelConfig as decision_process."""
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
    )

    assert isinstance(config.decision_process, ModelConfig)
    assert config.decision_process.list_params == ["v", "a", "z", "t"]

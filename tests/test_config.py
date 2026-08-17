import logging
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

import hssm
from hssm.config import Config, ModelConfig


hssm.set_floatX("float32")


def test_from_defaults():
    # Case 1: Has default prior
    config1 = Config.from_defaults("ddm", "analytical")

    assert config1.model_name == "ddm"
    assert config1.response == ["rt", "response"]
    assert config1.list_params == ["v", "a", "z", "t"]
    assert config1.loglik_kind == "analytical"
    assert config1.loglik is not None
    assert config1.response_kind == "categorical"
    assert config1.response_bounds == {}
    assert config1.choices == (-1, 1)
    assert "t" in config1.default_priors
    assert "v" in config1.bounds
    assert not config1.is_choice_only

    # Case 2: Model supported, but no default prior
    config2 = Config.from_defaults("angle", "analytical")

    assert config2.model_name == "angle"
    assert config2.response == ["rt", "response"]
    assert config2.list_params == ["v", "a", "z", "t", "theta"]
    assert config2.loglik_kind == "analytical"
    assert config2.loglik is None
    assert config2.default_priors == {}
    assert config2.bounds == {}
    assert not config2.is_choice_only

    # Case 3: Model supported, loglik_kind is None
    config3 = Config.from_defaults("ddm", None)

    assert config3 == config1

    # Case 4: No supported model, provided loglik_kind
    config4 = Config.from_defaults("custom", "analytical")
    assert config4.model_name == "custom"
    assert config4.response == ["rt", "response"]
    assert config4.list_params is None
    assert config4.loglik_kind == "analytical"
    assert config4.loglik is None
    assert config4.default_priors == {}
    assert config4.bounds == {}
    assert not config4.is_choice_only

    # Case 5: No supported model, provided loglik_kind
    config5 = Config.from_defaults("custom", "analytical")
    config5.response = ["response"]
    assert config5.is_choice_only

    # Case 6: No supported model, did not provide loglik_kind
    with pytest.raises(ValueError):
        Config.from_defaults("custom", None)


def test_update_config():
    config1 = Config.from_defaults("ddm", "analytical")
    assert config1.response == ["rt", "response"]

    v_prior, v_bounds = config1.get_defaults("v")

    assert v_prior is None
    assert v_bounds == (-np.inf, np.inf)

    user_config = ModelConfig(
        list_params=["a", "b", "c"],
        backend="jax",
        default_priors={
            "t": hssm.Prior("Uniform", lower=-5, upper=5),
            "v": hssm.Prior("Normal"),
        },
    )

    config1.update_config(user_config)

    assert config1.list_params == ["a", "b", "c"]
    assert config1.backend is None
    assert "t" in config1.default_priors
    assert "a" not in config1.default_priors

    v_prior, v_bounds = config1.get_defaults("v")

    assert v_prior.name == "Normal"
    assert v_bounds == (-np.inf, np.inf)


def test_non_categorical_response_metadata_round_trips():
    bounds = {"response": (-np.pi, np.pi)}
    user_config = ModelConfig(
        response=("rt", "response"),
        response_kind="circular",
        response_bounds=bounds,
        list_params=["v_x", "v_y", "a", "t"],
    )

    config = Config.from_defaults("custom", "blackbox")
    config.update_config(deepcopy(user_config))
    config.update_loglik(lambda *args: None)
    config.validate()

    assert config.response_kind == "circular"
    assert config.response_bounds == bounds
    assert config.choices is None
    assert asdict(config)["response_bounds"] == bounds
    assert user_config.response_bounds == bounds

    config.response_bounds["response"] = (-1.0, 1.0)
    assert user_config.response_bounds == bounds


@pytest.mark.parametrize("response_kind", ["continuous", "circular"])
def test_non_categorical_responses_reject_choices(response_kind):
    config = Config(
        model_name="custom",
        response_kind=response_kind,
        response_bounds={"response": (-np.pi, np.pi)},
        choices=(0, 1),
        list_params=["v"],
        loglik_kind="blackbox",
        loglik=lambda *args: None,
    )

    with pytest.raises(
        ValueError,
        match="`choices` must be None for continuous or circular responses",
    ):
        config.validate()


def test_categorical_responses_require_choices():
    config = Config(
        model_name="custom",
        choices=None,
        list_params=["v"],
        loglik_kind="blackbox",
        loglik=lambda *args: None,
    )

    with pytest.raises(ValueError, match="choices.*categorical responses"):
        config.validate()


def test_circular_responses_require_valid_bounds():
    config = Config(
        model_name="custom",
        response_kind="circular",
        choices=None,
        list_params=["v"],
        loglik_kind="blackbox",
        loglik=lambda *args: None,
    )

    with pytest.raises(ValueError, match="bounds are required.*response"):
        config.validate()

    config.response_bounds = {"response": (np.pi, -np.pi)}
    with pytest.raises(ValueError, match="lower < upper"):
        config.validate()

    config.response_bounds = {"response": (-np.inf, np.inf)}
    with pytest.raises(ValueError, match="must be finite"):
        config.validate()


@pytest.mark.parametrize(
    ("supported_samplers", "message"),
    [
        ((), "cannot be empty"),
        (("stan",), "Unrecognized supported sampler.*stan"),
        (("pymc", "pymc"), "cannot contain duplicates"),
    ],
)
def test_supported_sampler_metadata_is_validated(supported_samplers, message):
    """Sampler restrictions should be nonempty, recognized, and unique."""
    config = Config(
        model_name="custom",
        choices=(0, 1),
        list_params=["v"],
        loglik_kind="analytical",
        loglik=lambda *args: None,
        supported_samplers=supported_samplers,
    )

    with pytest.raises(ValueError, match=message):
        config.validate()


class TestConfigBuildModelConfigExtraLogic:
    def test_build_model_config_dict_with_choices_conflict(self, caplog):
        # model 'ddm' has defaults in hssm.defaults; use a minimal dict override
        model_config = {
            "response": ("rt", "response"),
            "list_params": ["v", "a"],
            "choices": (0, 1),
        }
        # provide a different choices argument — should log that model_config wins
        with caplog.at_level(logging.INFO):
            cfg = Config._build_model_config("ddm", None, model_config, choices=[1, 0])

        assert isinstance(cfg, Config)
        assert "choices list provided in both model_config" in caplog.text

    def test_build_model_config_modelconfig_adds_choices(self):
        # Create a ModelConfig without choices and pass choices argument
        mc = ModelConfig(response=("rt", "response"), list_params=["v"], choices=None)
        cfg = Config._build_model_config("ddm", None, mc, choices=(0, 1))
        # choices should be applied to resulting Config
        assert cfg.choices == (0, 1)

    def test_build_model_config_uses_ssms_model_config(self, monkeypatch):
        # High-level view of the test: ensures that when a model name is not in the built-in
        # SupportedModels and no choices argument is passed, _build_model_config will consult
        # the external ssms_model_config registry and use its defaults (here, the choices tuple).
        # The monkeypatch fixture isolates the change and will be undone after the test.

        # Simulate an external ssms_model_config entry for a model not in SupportedModels
        fake_model = "external_ssm"
        fake_choices = (2, 3)

        # Monkeypatch the ssms_model_config mapping in the module
        import hssm.config as cfgmod

        # Emulate an external package registering defaults for external_ssm.
        # Ensures `_build_model_config` will consult `ssms_model_config`
        # when the model name isn't in SupportedModels.
        monkeypatch.setitem(
            cfgmod.ssms_model_config, fake_model, {"choices": fake_choices}
        )

        # Build config with model not in SupportedModels and no choices arg.
        # Provide a minimal ModelConfig and a dummy `loglik` so
        # `Config.validate()` runs (loglik is required) while still
        # exercising the ssms-simulators choices fallback.
        mc = ModelConfig(response=("rt", "response"), list_params=["v"], choices=None)
        result = Config._build_model_config(
            fake_model,
            "analytical",
            mc,
            choices=None,
            loglik=(lambda *a, **k: None),  # required so Config.validate() passes
        )
        assert result.choices == fake_choices

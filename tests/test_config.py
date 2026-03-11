import logging

import pytest
import numpy as np

from hssm.config import Config, ModelConfig
import hssm


hssm.set_floatX("float32")


def test_from_defaults():
    # Case 1: Has default prior
    config1 = Config.from_defaults("ddm", "analytical")

    assert config1.model_name == "ddm"
    assert config1.response == ["rt", "response"]
    assert config1.list_params == ["v", "a", "z", "t"]
    assert config1.loglik_kind == "analytical"
    assert config1.loglik is not None
    assert "t" in config1.default_priors
    assert "v" in config1.bounds

    # Case 2: Model supported, but no default prior
    config2 = Config.from_defaults("angle", "analytical")

    assert config2.model_name == "angle"
    assert config2.response == ["rt", "response"]
    assert config2.list_params == ["v", "a", "z", "t", "theta"]
    assert config2.loglik_kind == "analytical"
    assert config2.loglik is None
    assert config2.default_priors == {}
    assert config2.bounds == {}

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

    # Case 5: No supported model, did not provide loglik_kind
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
        cfg = Config._build_model_config("ddm", None, mc, choices=[0, 1])
        # choices should be applied to resulting Config
        assert tuple(cfg.choices) == (0, 1)

    def test_build_model_config_uses_ssms_model_config(self, monkeypatch):
        # Simulate an external ssms_model_config entry for a model not in SupportedModels
        fake_model = "external_ssm"
        fake_choices = [2, 3]

        # Monkeypatch the ssms_model_config mapping in the module
        import hssm.config as cfgmod

        monkeypatch.setitem(
            cfgmod.ssms_model_config, fake_model, {"choices": fake_choices}
        )

        # Build config with model not in SupportedModels (string) and no choices arg
        # provide a loglik_kind so from_defaults does not raise for custom model
        # Monkeypatch Config.validate to skip strict checks for this synthetic case
        monkeypatch.setattr(Config, "validate", lambda self: None)
        result = Config._build_model_config(
            fake_model, "analytical", None, choices=None
        )
        assert tuple(result.choices) == tuple(fake_choices)

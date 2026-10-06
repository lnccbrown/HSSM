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


def test_from_defaults_reads_ndt_edge_shift():
    """A registry entry's declaration lands on Config as its own copy."""
    assert Config.from_defaults("ddm", "analytical").ndt_edge_shift is None

    config = Config.from_defaults("full_ddm", "blackbox")
    registry_entry = hssm.defaults.default_model_config["full_ddm"]["likelihoods"][
        "blackbox"
    ]["ndt_edge_shift"]

    assert config.ndt_edge_shift == {"param": "st", "scale": 0.5}
    # Deep-copied with the rest of the entry, like default_priors and bounds.
    assert config.ndt_edge_shift is not registry_entry


def test_update_config_ndt_edge_shift_precedence():
    """A user-supplied declaration overrides the registry's; None keeps it."""
    config = Config.from_defaults("full_ddm", "blackbox")

    config.update_config(ModelConfig())
    assert config.ndt_edge_shift == {"param": "st", "scale": 0.5}

    config.update_config(ModelConfig(ndt_edge_shift={"param": "st", "scale": 1.0}))
    assert config.ndt_edge_shift == {"param": "st", "scale": 1.0}


@pytest.mark.parametrize(
    ("list_params", "ndt_edge_shift", "match"),
    [
        (["v", "a", "z", "t"], {"param": "st", "scale": 1.0}, "names the parameter"),
        (["v", "a", "z", "st"], {"param": "st", "scale": 1.0}, "`t` is not in"),
        (["v", "a", "z", "t", "st"], {"param": "st", "scale": -1.0}, "scale"),
        (["v", "a", "z", "t", "st"], {"param": "st", "scale": float("nan")}, "scale"),
        (["v", "a", "z", "t", "st"], {"param": "st", "scale": float("inf")}, "scale"),
        (["v", "a", "z", "t", "st"], {"param": "st", "scale": True}, "scale"),
        (["v", "a", "z", "t", "st"], {"param": "st", "scale": "1.0"}, "scale"),
        (["v", "a", "z", "t", "st"], {"param": "st", "scale": None}, "scale"),
    ],
    ids=[
        "param_not_in_list_params",
        "no_t",
        "negative",
        "nan",
        "inf",
        "bool",
        "string",
        "none",
    ],
)
def test_validate_rejects_bad_ndt_edge_shift(list_params, ndt_edge_shift, match):
    """Each of the three validation rules raises and names the offending key."""
    with pytest.raises(ValueError, match=match):
        Config._build_model_config(
            "ddm",
            "analytical",
            ModelConfig(list_params=list_params, ndt_edge_shift=ndt_edge_shift),
            None,
        )


@pytest.mark.parametrize(
    "scale",
    [0, 0.5, 1, 3.0, np.float32(0.5), np.int64(2)],
    ids=["zero", "half", "int", "three", "np_float32", "np_int64"],
)
def test_validate_accepts_finite_non_negative_scale(scale):
    """Zero is the fixed-t edge; numpy scalars count as numbers."""
    config = Config._build_model_config(
        "ddm",
        "analytical",
        ModelConfig(
            list_params=["v", "a", "z", "t", "st"],
            ndt_edge_shift={"param": "st", "scale": scale},
        ),
        None,
    )
    assert config.ndt_edge_shift == {"param": "st", "scale": scale}


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

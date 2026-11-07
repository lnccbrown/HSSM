"""Unit tests for configuration defaults and model-config assembly."""

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
        (["v", "a", "z", "t", "st"], {"param": "st"}, "keys 'param' and 'scale'"),
        (["v", "a", "z", "t", "st"], {"scale": 1.0}, "keys 'param' and 'scale'"),
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
        "missing_scale",
        "missing_param",
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

    @pytest.mark.parametrize(
        ("n_choices", "expected"),
        [(2, (-1, 1)), (3, (0, 1, 2)), (np.int64(4), (0, 1, 2, 3))],
        ids=["two", "three", "np_int64"],
    )
    def test_build_model_config_int_choices(self, n_choices, expected):
        """An int `choices` is expanded to the default labels for that count.

        The `np.int64` case guards the `np.integer` branch: numpy integers are not
        `int` subclasses, and a count computed from data, such as
        `df["response"].max() + 1`, is a numpy integer.
        """
        mc = ModelConfig(response=("rt", "response"), list_params=["v"])
        cfg = Config._build_model_config(
            "custom", "analytical", mc, choices=n_choices, loglik=lambda *a: None
        )
        assert cfg.choices == expected

    @pytest.mark.parametrize("n_choices", [0, 1])
    def test_build_model_config_int_choices_below_two(self, n_choices):
        """Fewer than two choices is rejected."""
        mc = ModelConfig(response=("rt", "response"), list_params=["v"])
        with pytest.raises(ValueError, match="must be at least 2"):
            Config._build_model_config(
                "custom", "analytical", mc, choices=n_choices, loglik=lambda *a: None
            )

    @pytest.mark.parametrize(
        ("model", "choices"),
        [("custom", (-1, 1)), ("custom", 2), ("race_no_bias_2", None)],
        ids=["explicit_arg", "int_arg", "ssms_fallback"],
    )
    def test_build_model_config_custom_model_config_choices_win(self, model, choices):
        """Choices in `model_config` beat the argument and the ssms fallback."""
        mc = {"list_params": ["v"], "choices": (5, 6, 7)}
        cfg = Config._build_model_config(
            model, "analytical", mc, choices=choices, loglik=lambda *a: None
        )
        assert cfg.choices == (5, 6, 7)

    def test_build_model_config_uses_ssms_model_config(self, registered_ssms_model):
        """A name only ssms knows takes its choices when nothing else sets them.

        The model is registered after import, so it is in ssms' registry but
        not in the ``ssms.config.model_config`` snapshot (#1467).
        """
        mc = ModelConfig(response=("rt", "response"), list_params=["v"], choices=None)
        result = Config._build_model_config(
            registered_ssms_model["name"],
            "analytical",
            mc,
            choices=None,
            loglik=(lambda *a, **k: None),  # required so Config.validate() passes
        )
        assert result.choices == (-1, 1)


class TestSsmsRegisteredModel:
    """Models HSSM does not ship take their config from ssm-simulators (#1467)."""

    def test_from_defaults_registered_model(self, registered_ssms_model):
        """list_params, bounds and choices come from the registered ssms config."""
        config = Config.from_defaults(
            registered_ssms_model["name"], "approx_differentiable"
        )

        assert config.list_params == ["v", "a", "z", "t"]
        assert config.bounds == {
            "v": (-2.0, 2.0),
            "a": (0.5, 2.0),
            "z": (0.2, 0.8),
            "t": (0.1, 1.5),
        }
        assert config.choices == (-1, 1)

    def test_from_defaults_ssms_builtin(self):
        """An ssms built-in HSSM does not ship resolves the same way."""
        from ssms.config import get_model_registry

        ssms_config = get_model_registry().get("race_no_bias_2")
        config = Config.from_defaults("race_no_bias_2", "approx_differentiable")

        assert config.list_params == ssms_config["params"]
        assert config.bounds == ssms_config["param_bounds_dict"]
        assert config.choices == tuple(ssms_config["choices"])

    def test_from_defaults_unknown_model(self):
        """A name neither HSSM nor ssms knows still leaves everything to the user."""
        config = Config.from_defaults("not_a_model", "approx_differentiable")

        assert config.list_params is None
        assert config.bounds == {}
        assert config.choices == (0, 1)

    def test_user_model_config_wins(self, registered_ssms_model):
        """Keys set in `model_config` override the registered config's."""
        config = Config._build_model_config(
            registered_ssms_model["name"],
            "approx_differentiable",
            {
                "list_params": ["v", "a", "t"],
                "bounds": {"v": (-1.0, 1.0)},
                "choices": [0, 1],
                "backend": "jax",
            },
            choices=None,
            loglik=lambda *a: None,
        )

        assert config.list_params == ["v", "a", "t"]
        assert config.bounds["v"] == (-1.0, 1.0)
        assert config.bounds["a"] == (0.5, 2.0)
        assert config.choices == (0, 1)

    def test_choices_argument_wins(self, registered_ssms_model):
        """The `choices` argument beats the registered config's choices."""
        config = Config._build_model_config(
            registered_ssms_model["name"],
            "approx_differentiable",
            {"backend": "jax"},
            choices=(0, 1),
            loglik=lambda *a: None,
        )

        assert config.choices == (0, 1)

    @pytest.mark.parametrize(
        "ssms_config",
        [
            {"params": ["v", "a"], "param_bounds": [[-3, 0.3], [3, 2.5]]},
            {"params": ["v", "a"], "param_bounds": {"v": [-3, 3], "a": [0.3, 2.5]}},
            {
                "params": ["v", "a"],
                "param_bounds": [[-3, 0.3], [3, 2.5]],
                "param_bounds_dict": {"v": (-3, 3), "a": (0.3, 2.5)},
            },
        ],
        ids=["list", "dict", "param_bounds_dict"],
    )
    def test_bounds_from_ssms(self, ssms_config):
        """Both ssms bounds layouts give the same HSSM bounds dict."""
        from hssm.config import _bounds_from_ssms

        assert _bounds_from_ssms(ssms_config) == {"v": (-3.0, 3.0), "a": (0.3, 2.5)}

    def test_bounds_from_ssms_without_bounds(self):
        """A config without bounds gives no bounds."""
        from hssm.config import _bounds_from_ssms

        assert _bounds_from_ssms({"params": ["v", "a"]}) == {}


class TestApproxDifferentiableBackendDefault:
    """`backend` defaults to "jax" for an ONNX `loglik` (#1467)."""

    @pytest.mark.parametrize("as_path", [True, False], ids=["path", "str"])
    def test_onnx_loglik_defaults_to_jax(self, fixture_path, as_path):
        onnx_file = fixture_path / "ddm.onnx"
        config = Config._build_model_config(
            "custom",
            "approx_differentiable",
            {"list_params": ["v", "a", "z", "t"]},
            choices=None,
            loglik=onnx_file if as_path else str(onnx_file),
        )

        assert config.backend == "jax"

    def test_user_backend_wins(self, fixture_path):
        config = Config._build_model_config(
            "custom",
            "approx_differentiable",
            {"list_params": ["v", "a", "z", "t"], "backend": "pytensor"},
            choices=None,
            loglik=fixture_path / "ddm.onnx",
        )

        assert config.backend == "pytensor"

    def test_callable_loglik_still_needs_backend(self):
        with pytest.raises(ValueError, match="Please provide `backend`"):
            Config._build_model_config(
                "custom",
                "approx_differentiable",
                {"list_params": ["v", "a", "z", "t"]},
                choices=None,
                loglik=lambda *a: None,
            )

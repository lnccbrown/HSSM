"""Unit tests for the RL registry helpers.

These tests exercise the registry module directly without constructing full
RLSSM model instances, so regressions in lazy SSM resolution, config
composition, and registration validation are caught at the module boundary.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import pytest

from hssm.rl import registry
from hssm.rl.config import RLSSMConfig
from hssm.utils import annotate_function


@pytest.fixture(autouse=True)
def isolated_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate global registries so tests do not leak state."""
    monkeypatch.setattr(registry, "_SSM_REGISTRY", deepcopy(registry._SSM_REGISTRY))
    monkeypatch.setattr(registry, "_RLSSM_REGISTRY", deepcopy(registry._RLSSM_REGISTRY))
    monkeypatch.setattr(registry, "_SSM_LOGP_CACHE", dict(registry._SSM_LOGP_CACHE))


@pytest.fixture
def learning_process() -> dict[str, Any]:
    """Return an annotated RL learning rule for test models."""

    @annotate_function(
        inputs=["rl_alpha", "response", "feedback"],
        outputs=["v"],
    )
    def compute_v(rl_alpha, response, feedback):
        return rl_alpha

    return {"v": compute_v}


@pytest.fixture
def annotated_ssm_base_logp() -> Any:
    """Return an annotated SSM base log-likelihood function."""

    @annotate_function(
        inputs=["v", "a", "rt", "response"],
        outputs=["logp"],
    )
    def base_logp(v, a, rt, response):
        return a

    return base_logp


class TestBuildSsmSpecFromModelconfig:
    def test_unknown_model_raises(self) -> None:
        """Completely unknown names should raise ValueError from the modelconfig bridge."""
        with pytest.raises(ValueError, match="not a registered custom SSM"):
            registry._build_ssm_spec_from_modelconfig("totally_unknown_model")

    def test_no_approx_differentiable_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Models without an approx_differentiable likelihood must be rejected."""
        import hssm.modelconfig as mc_module

        def _fake_get(name):  # type: ignore[no-untyped-def]
            return {
                "list_params": ["v", "a"],
                "response": ["rt", "response"],
                "likelihoods": {
                    "analytical": {
                        "loglik": lambda: None,
                        "bounds": {},
                        "backend": None,
                    },
                },
            }

        monkeypatch.setattr(mc_module, "get_default_model_config", _fake_get)
        with pytest.raises(ValueError, match="no approx_differentiable likelihood"):
            registry._build_ssm_spec_from_modelconfig("ddm")

    def test_factory_calls_onnx_loader(
        self,
        monkeypatch: pytest.MonkeyPatch,
        annotated_ssm_base_logp: Any,
    ) -> None:
        """The lazy factory should call make_jax_matrix_logp_funcs_from_onnx with the
        correct filename."""
        import hssm.distribution_utils.onnx as onnx_module

        called_with: list[str] = []

        def _fake_onnx(model: str) -> Any:
            called_with.append(model)
            return annotated_ssm_base_logp

        monkeypatch.setattr(
            onnx_module, "make_jax_matrix_logp_funcs_from_onnx", _fake_onnx
        )
        monkeypatch.setattr(
            registry, "make_jax_matrix_logp_funcs_from_onnx", _fake_onnx
        )

        spec = registry._build_ssm_spec_from_modelconfig("angle")
        result = spec["ssm_base_logp_func_factory"]()

        assert called_with == ["angle.onnx"]
        assert callable(result)
        assert result.inputs == ["v", "a", "z", "t", "theta", "rt", "response"]
        assert result.outputs == ["logp"]


class TestGetSsmLogp:
    def test_builds_lazy_factory_once(self, annotated_ssm_base_logp: Any) -> None:
        """Lazy SSM factories should only build and cache one function instance."""
        call_count = 0

        def factory() -> Any:
            nonlocal call_count
            call_count += 1
            return annotated_ssm_base_logp

        registry._SSM_REGISTRY["lazy_unit_test_ssm"] = {
            "ssm_base_logp_func_factory": factory,
            "list_params_ssm": ["v", "a"],
            "bounds_ssm": {"a": (0.3, 3.0)},
            "params_default_ssm": [0.0, 1.5],
            "response": ["rt", "response"],
        }

        first = registry._get_ssm_logp("lazy_unit_test_ssm")
        second = registry._get_ssm_logp("lazy_unit_test_ssm")

        assert first is annotated_ssm_base_logp
        assert second is first
        assert call_count == 1

    def test_resolves_builtin_via_modelconfig(
        self,
        monkeypatch: pytest.MonkeyPatch,
        annotated_ssm_base_logp: Any,
    ) -> None:
        """_get_ssm_logp should build and cache the logp for a built-in SSM."""
        import hssm.distribution_utils.onnx as onnx_module

        monkeypatch.setattr(
            onnx_module,
            "make_jax_matrix_logp_funcs_from_onnx",
            lambda **_: annotated_ssm_base_logp,
        )
        monkeypatch.setattr(
            registry,
            "make_jax_matrix_logp_funcs_from_onnx",
            lambda **_: annotated_ssm_base_logp,
        )

        result = registry._get_ssm_logp("angle")

        assert callable(result)
        assert "angle" in registry._SSM_LOGP_CACHE
        # Second call returns cached value without re-building.
        assert registry._get_ssm_logp("angle") is result


class TestBuildSsmLogpFunc:
    def test_raises_if_already_computed(
        self,
        annotated_ssm_base_logp: Any,
        learning_process: dict[str, Any],
    ) -> None:
        """_build_ssm_logp_func must refuse functions that already have .computed set."""
        precomputed = annotate_function(
            inputs=annotated_ssm_base_logp.inputs,
            outputs=annotated_ssm_base_logp.outputs,
            computed=learning_process,
        )(annotated_ssm_base_logp)

        with pytest.raises(ValueError, match="already has a non-empty .computed"):
            registry._build_ssm_logp_func(precomputed, learning_process)


class TestDeriveRlParams:
    def test_excludes_response_and_extra_fields(
        self,
        learning_process: dict[str, Any],
    ) -> None:
        """Derived RL params should ignore response columns and extra fields."""
        derived = registry._derive_lp_params(
            learning_process=learning_process,
            response=["rt", "response"],
            extra_fields=["feedback"],
        )

        assert derived == ["rl_alpha"]

    def test_warns_for_unannotated_lp_func(self) -> None:
        """A learning-process function without .inputs should log a warning and be skipped."""

        def unannotated_func(x):  # type: ignore[no-untyped-def]
            return x

        lp = {"v": unannotated_func}
        result = registry._derive_lp_params(
            learning_process=lp,
            response=["rt", "response"],
            extra_fields=["feedback"],
        )
        # The unannotated function contributes no params.
        assert result == []


class TestRegisterSsm:
    def test_caches_prebuilt_function(self, annotated_ssm_base_logp: Any) -> None:
        """Registering a pre-built SSM should populate the cache immediately."""
        registry.register_ssm(
            name="cached_ssm",
            ssm_base_logp_func=annotated_ssm_base_logp,
            list_params_ssm=["v", "a"],
            bounds_ssm={"a": (0.3, 3.0)},
            params_default_ssm=[0.0, 1.5],
        )

        assert registry._SSM_LOGP_CACHE["cached_ssm"] is annotated_ssm_base_logp
        assert registry._SSM_REGISTRY["cached_ssm"]["response"] == ["rt", "response"]

    def test_rejects_precomputed_function(
        self,
        annotated_ssm_base_logp: Any,
        learning_process: dict[str, Any],
    ) -> None:
        """SSM registration should reject functions that already carry computed params."""
        precomputed_logp = annotate_function(
            inputs=annotated_ssm_base_logp.inputs,
            outputs=annotated_ssm_base_logp.outputs,
            computed=learning_process,
        )(annotated_ssm_base_logp)

        with pytest.raises(ValueError, match="should not have a non-empty .computed"):
            registry.register_ssm(
                name="invalid_ssm",
                ssm_base_logp_func=precomputed_logp,
                list_params_ssm=["v", "a"],
                bounds_ssm={"a": (0.3, 3.0)},
                params_default_ssm=[0.0, 1.5],
            )

    def test_rejects_non_callable(self) -> None:
        """register_ssm must raise when ssm_base_logp_func is not callable."""
        with pytest.raises(ValueError, match="must be callable"):
            registry.register_ssm(
                name="bad_ssm",
                ssm_base_logp_func="not_a_function",  # type: ignore[arg-type]
                list_params_ssm=["v"],
                bounds_ssm={},
                params_default_ssm=[0.0],
            )

    def test_rejects_unannotated_callable(self) -> None:
        """register_ssm must raise when the callable lacks .inputs or .outputs."""

        def plain(x):  # type: ignore[no-untyped-def]
            return x

        with pytest.raises(
            ValueError, match="must be decorated with @annotate_function"
        ):
            registry.register_ssm(
                name="unannotated_ssm",
                ssm_base_logp_func=plain,
                list_params_ssm=["v"],
                bounds_ssm={},
                params_default_ssm=[0.0],
            )

    def test_warns_on_overwrite(
        self,
        annotated_ssm_base_logp: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Re-registering an existing SSM name should emit a warning."""
        registry.register_ssm(
            name="dup_ssm",
            ssm_base_logp_func=annotated_ssm_base_logp,
            list_params_ssm=["v", "a"],
            bounds_ssm={"a": (0.3, 3.0)},
            params_default_ssm=[0.0, 1.5],
        )

        with caplog.at_level(logging.WARNING, logger="hssm"):
            registry.register_ssm(
                name="dup_ssm",
                ssm_base_logp_func=annotated_ssm_base_logp,
                list_params_ssm=["v", "a"],
                bounds_ssm={"a": (0.3, 3.0)},
                params_default_ssm=[0.0, 1.5],
            )

        assert any("dup_ssm" in r.message for r in caplog.records)


class TestRegisterRlssmModel:
    def test_copies_mutable_inputs(
        self,
        learning_process: dict[str, Any],
    ) -> None:
        """Caller mutations after registration must not alter the stored model."""
        learning_process_params = ["rl_alpha"]
        learning_process_bounds = {"rl_alpha": (0.0, 1.0)}
        rl_defaults = [0.2]
        extra_fields = ["feedback"]
        choices = [0, 1]

        registry.register_rlssm_model(
            name="copy_test_model",
            decision_process="angle",
            learning_process=learning_process,
            learning_process_params=learning_process_params,
            learning_process_bounds=learning_process_bounds,
            learning_process_params_default=rl_defaults,
            extra_fields=extra_fields,
            choices=choices,
        )

        learning_process_params.append("scaler")
        learning_process_bounds["scaler"] = (0.0, 10.0)
        rl_defaults.append(1.0)
        extra_fields.append("trial")
        choices.append(2)
        # Built-in SSMs (e.g. "angle") are derived from modelconfig on each call
        # and are never stored as shared mutable state in _SSM_REGISTRY, so there
        # is no registry entry to corrupt here.
        learning_process["other"] = next(iter(learning_process.values()))

        stored = registry._RLSSM_REGISTRY["copy_test_model"]
        metadata = stored.learning_process_metadata

        assert stored.decision_process["name"] == "angle"
        assert stored.decision_process["response"] == ["rt", "response"]
        assert metadata.sampled_params == ["rl_alpha"]
        assert metadata.bounds == {"rl_alpha": (0.0, 1.0)}
        assert metadata.defaults == [0.2]
        assert metadata.extra_fields == ["feedback"]
        assert metadata.kind == "blackbox"
        assert stored.choices == [0, 1]
        assert list(metadata.learning_process) == ["v"]

    def test_warns_on_overwrite(
        self,
        learning_process: dict[str, Any],
        annotated_ssm_base_logp: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Re-registering an existing RLSSM model name should emit a warning."""
        registry.register_ssm(
            name="overwrite_ssm",
            ssm_base_logp_func=annotated_ssm_base_logp,
            list_params_ssm=["v", "a"],
            bounds_ssm={"a": (0.3, 3.0)},
            params_default_ssm=[0.0, 1.5],
        )
        registry.register_rlssm_model(
            name="overwrite_rlssm",
            decision_process="overwrite_ssm",
            learning_process=learning_process,
            learning_process_params=["rl_alpha"],
            learning_process_bounds={"rl_alpha": (0.0, 1.0)},
            learning_process_params_default=[0.2],
        )

        with caplog.at_level(logging.WARNING, logger="hssm"):
            registry.register_rlssm_model(
                name="overwrite_rlssm",
                decision_process="overwrite_ssm",
                learning_process=learning_process,
                learning_process_params=["rl_alpha"],
                learning_process_bounds={"rl_alpha": (0.0, 1.0)},
                learning_process_params_default=[0.2],
            )

        assert any("overwrite_rlssm" in r.message for r in caplog.records)


class TestGetRlssmModelConfig:
    def test_builds_expected_config(
        self,
        annotated_ssm_base_logp: Any,
        learning_process: dict[str, Any],
    ) -> None:
        """Registry config composition should exclude computed SSM params."""
        registry.register_ssm(
            name="unit_test_ssm",
            ssm_base_logp_func=annotated_ssm_base_logp,
            list_params_ssm=["v", "a"],
            bounds_ssm={"a": (0.3, 3.0)},
            params_default_ssm=[0.0, 1.5],
            response=["rt", "response"],
        )
        registry.register_rlssm_model(
            name="unit_test_model",
            decision_process="unit_test_ssm",
            learning_process=learning_process,
            learning_process_params=["rl_alpha"],
            learning_process_bounds={"rl_alpha": (0.0, 1.0)},
            learning_process_params_default=[0.2],
            extra_fields=["feedback"],
            choices=[0, 1],
        )

        config = registry.get_rlssm_model_config("unit_test_model")

        assert isinstance(config, RLSSMConfig)
        assert config.list_params == ["rl_alpha", "a"]
        assert config.bounds == {"rl_alpha": (0.0, 1.0), "a": (0.3, 3.0)}
        assert config.params_default == [0.2, 1.5]
        assert config.response == ["rt", "response"]
        assert config.ssm_logp_func.computed == learning_process

        # Mutating the returned config must not corrupt the registry's stored list.
        config.response.append("mutated")
        assert registry._SSM_REGISTRY["unit_test_ssm"]["response"] == ["rt", "response"]

    def test_respects_explicit_empty_rl_fields(
        self,
        annotated_ssm_base_logp: Any,
        learning_process: dict[str, Any],
    ) -> None:
        """Explicit empty RL collections must not trigger fallback derivation."""
        registry.register_ssm(
            name="empty_rl_ssm",
            ssm_base_logp_func=annotated_ssm_base_logp,
            list_params_ssm=["v", "a"],
            bounds_ssm={"a": (0.3, 3.0)},
            params_default_ssm=[0.0, 1.5],
            response=["rt", "response"],
        )
        registry._RLSSM_REGISTRY["empty_rl_model"] = registry.RLSSMRegistryEntry(
            decision_process="empty_rl_ssm",
            learning_process_metadata=registry.LearningProcessMetadata(
                learning_process=learning_process,
                sampled_params=[],
                bounds={},
                defaults=[],
                extra_fields=["feedback"],
                kind="blackbox",
            ),
            choices=[0, 1],
            description="test model",
            decision_process_loglik_kind="approx_differentiable",
        )

        config = registry.get_rlssm_model_config("empty_rl_model")

        assert config.list_params == ["a"]
        assert config.bounds == {"a": (0.3, 3.0)}
        assert config.params_default == [1.5]

    def test_unknown_model_raises(self) -> None:
        """Unknown RLSSM model names should fail with a clear error."""
        with pytest.raises(ValueError, match="not found in the RLSSM registry"):
            registry.get_rlssm_model_config("does_not_exist")

    def test_learning_process_override_with_matching_metadata(
        self,
        annotated_ssm_base_logp: Any,
    ) -> None:
        """Overriding the LP with the same sampled params should preserve metadata."""

        @annotate_function(
            inputs=["rl_alpha", "scaler", "response", "feedback"],
            outputs=["v"],
        )
        def alt_compute_v(rl_alpha, scaler, response, feedback):
            return rl_alpha + scaler

        registry.register_ssm(
            name="override_lp_ssm",
            ssm_base_logp_func=annotated_ssm_base_logp,
            list_params_ssm=["v", "a"],
            bounds_ssm={"a": (0.3, 3.0)},
            params_default_ssm=[0.0, 1.5],
            response=["rt", "response"],
        )
        registry.register_rlssm_model(
            name="override_lp_model",
            decision_process="override_lp_ssm",
            learning_process={"v": registry._compute_v_annotated},
            learning_process_params=["rl_alpha", "scaler"],
            learning_process_bounds={"rl_alpha": (0.0, 1.0), "scaler": (0.0, 10.0)},
            learning_process_params_default=[0.2, 1.5],
            extra_fields=["feedback"],
            choices=[0, 1],
        )

        config = registry.get_rlssm_model_config(
            "override_lp_model", learning_process={"v": alt_compute_v}
        )

        assert config.list_params == ["rl_alpha", "scaler", "a"]
        assert config.bounds["rl_alpha"] == (0.0, 1.0)
        assert config.bounds["scaler"] == (0.0, 10.0)
        assert config.params_default[:2] == [0.2, 1.5]
        assert config.ssm_logp_func.computed == {"v": alt_compute_v}

    def test_learning_process_override_missing_metadata_raises(
        self,
        annotated_ssm_base_logp: Any,
    ) -> None:
        """Overriding the LP with new sampled params must fail early."""

        @annotate_function(
            inputs=["new_alpha", "response", "feedback"],
            outputs=["v"],
        )
        def alt_compute_v(new_alpha, response, feedback):
            return new_alpha

        registry.register_ssm(
            name="override_missing_meta_ssm",
            ssm_base_logp_func=annotated_ssm_base_logp,
            list_params_ssm=["v", "a"],
            bounds_ssm={"a": (0.3, 3.0)},
            params_default_ssm=[0.0, 1.5],
            response=["rt", "response"],
        )
        registry.register_rlssm_model(
            name="override_missing_meta_model",
            decision_process="override_missing_meta_ssm",
            learning_process={"v": registry._compute_v_annotated},
            learning_process_params=["rl_alpha", "scaler"],
            learning_process_bounds={"rl_alpha": (0.0, 1.0), "scaler": (0.0, 10.0)},
            learning_process_params_default=[0.2, 1.5],
            extra_fields=["feedback"],
            choices=[0, 1],
        )

        with pytest.raises(
            ValueError, match="override introduced sampled parameter metadata"
        ):
            registry.get_rlssm_model_config(
                "override_missing_meta_model",
                learning_process={"v": alt_compute_v},
            )

    def test_raises_for_missing_bounds(
        self,
        annotated_ssm_base_logp: Any,
        learning_process: dict[str, Any],
    ) -> None:
        """SSM params missing from bounds_ssm must raise immediately with a clear error.

        Previously the factory silently skipped such params, producing a broken
        RLSSMConfig that only failed later inside _RLSSM.__init__. Now it must
        raise at the factory boundary so the error message points at the root cause.
        """
        registry.register_ssm(
            name="no_bounds_ssm",
            ssm_base_logp_func=annotated_ssm_base_logp,
            list_params_ssm=["v", "a"],
            # "a" intentionally has no bounds entry
            bounds_ssm={},
            params_default_ssm=[0.0, 1.5],
            response=["rt", "response"],
        )
        registry.register_rlssm_model(
            name="no_bounds_model",
            decision_process="no_bounds_ssm",
            learning_process=learning_process,
            learning_process_params=["rl_alpha"],
            learning_process_bounds={"rl_alpha": (0.0, 1.0)},
            learning_process_params_default=[0.2],
            extra_fields=["feedback"],
            choices=[0, 1],
        )

        with pytest.raises(ValueError, match="no entry in bounds_ssm"):
            registry.get_rlssm_model_config("no_bounds_model")


class TestBuiltinModels:
    @pytest.mark.parametrize(
        "model_name, expected_dp",
        [
            ("2AB_RescorlaWagner_DDM", "ddm"),
            ("2AB_RescorlaWagner_Weibull", "weibull"),
        ],
    )
    def test_are_registered(self, model_name: str, expected_dp: str) -> None:
        """2AB_RescorlaWagner_DDM and _Weibull must be present in the RLSSM registry."""
        assert model_name in registry._RLSSM_REGISTRY
        entry = registry._RLSSM_REGISTRY[model_name]
        metadata = entry.learning_process_metadata
        assert entry.decision_process["name"] == expected_dp
        assert metadata.sampled_params == ["rl_alpha", "scaler"]
        assert metadata.extra_fields == ["feedback"]
        assert entry.choices == [0, 1]
        assert entry.decision_process_loglik_kind == "approx_differentiable"
        assert metadata.kind == "blackbox"

    @pytest.mark.parametrize(
        "model_name",
        ["2AB_RescorlaWagner_DDM", "2AB_RescorlaWagner_Weibull"],
    )
    def test_config_structure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        annotated_ssm_base_logp: Any,
        model_name: str,
    ) -> None:
        """get_rlssm_model_config should produce a well-formed RLSSMConfig for
        both the DDM and Weibull starter-pack models."""
        import hssm.distribution_utils.onnx as onnx_module

        monkeypatch.setattr(
            onnx_module,
            "make_jax_matrix_logp_funcs_from_onnx",
            lambda model: annotated_ssm_base_logp,
        )
        monkeypatch.setattr(
            registry,
            "make_jax_matrix_logp_funcs_from_onnx",
            lambda model: annotated_ssm_base_logp,
        )

        config = registry.get_rlssm_model_config(model_name)

        assert isinstance(config, RLSSMConfig)
        # RL params come first
        assert config.list_params[:2] == ["rl_alpha", "scaler"]
        assert "rl_alpha" in config.bounds
        assert "scaler" in config.bounds
        assert config.choices == (0, 1)
        assert config.extra_fields == ["feedback"]
        assert config.ssm_logp_func.computed == {"v": registry._compute_v_annotated}


class TestListModels:
    def test_returns_all_names(self) -> None:
        """list_models should return every key in _RLSSM_REGISTRY with its description."""
        result = registry.list_models()

        assert set(result.keys()) == set(registry._RLSSM_REGISTRY.keys())
        for name, desc in result.items():
            assert desc == registry._RLSSM_REGISTRY[name].description

    def test_public_rl_api_matches_registry(self) -> None:
        """The public hssm.rl and RLSSM accessors should delegate to the registry."""
        import hssm

        assert hssm.rl.list_models() == registry.list_models()
        assert hssm.RLSSM.list_models == registry.list_models()

"""Unit tests for the RL registry helpers.

These tests exercise the registry module directly without constructing full
RLSSM model instances, so regressions in lazy SSM resolution, config
composition, and registration validation are caught at the module boundary.
"""

from __future__ import annotations

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


def test_get_ssm_logp_builds_lazy_factory_once(annotated_ssm_base_logp: Any) -> None:
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


def test_derive_rl_params_excludes_response_and_extra_fields(
    learning_process: dict[str, Any],
) -> None:
    """Derived RL params should ignore response columns and extra fields."""
    derived = registry._derive_rl_params(
        learning_process=learning_process,
        response=["rt", "response"],
        extra_fields=["feedback"],
    )

    assert derived == ["rl_alpha"]


def test_get_rlssm_model_config_builds_expected_config(
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
        rl_params=["rl_alpha"],
        rl_bounds={"rl_alpha": (0.0, 1.0)},
        rl_params_default=[0.2],
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

    config.response.append("mutated")
    assert registry._SSM_REGISTRY["unit_test_ssm"]["response"] == ["rt", "response"]


def test_get_rlssm_model_config_respects_explicit_empty_rl_fields(
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
    registry._RLSSM_REGISTRY["empty_rl_model"] = {
        "decision_process": "empty_rl_ssm",
        "learning_process": learning_process,
        "rl_params": [],
        "rl_bounds": {},
        "rl_params_default": [],
        "extra_fields": ["feedback"],
        "choices": [0, 1],
        "description": "test model",
        "decision_process_loglik_kind": "approx_differentiable",
        "learning_process_kind": "blackbox",
    }

    config = registry.get_rlssm_model_config("empty_rl_model")

    assert config.list_params == ["a"]
    assert config.bounds == {"a": (0.3, 3.0)}
    assert config.params_default == [1.5]


def test_register_rlssm_model_copies_mutable_inputs(
    learning_process: dict[str, Any],
) -> None:
    """Caller mutations after registration must not alter the stored model."""
    rl_params = ["rl_alpha"]
    rl_bounds = {"rl_alpha": (0.0, 1.0)}
    rl_defaults = [0.2]
    extra_fields = ["feedback"]
    choices = [0, 1]

    registry.register_rlssm_model(
        name="copy_test_model",
        decision_process="angle",
        learning_process=learning_process,
        rl_params=rl_params,
        rl_bounds=rl_bounds,
        rl_params_default=rl_defaults,
        extra_fields=extra_fields,
        choices=choices,
    )

    rl_params.append("scaler")
    rl_bounds["scaler"] = (0.0, 10.0)
    rl_defaults.append(1.0)
    extra_fields.append("trial")
    choices.append(2)
    registry._SSM_REGISTRY["angle"]["response"].append("deadline")
    learning_process["other"] = next(iter(learning_process.values()))

    stored = registry._RLSSM_REGISTRY["copy_test_model"]

    assert stored["decision_process"]["name"] == "angle"
    assert stored["decision_process"]["response"] == ["rt", "response"]
    assert stored["rl_params"] == ["rl_alpha"]
    assert stored["rl_bounds"] == {"rl_alpha": (0.0, 1.0)}
    assert stored["rl_params_default"] == [0.2]
    assert stored["extra_fields"] == ["feedback"]
    assert stored["choices"] == [0, 1]
    assert list(stored["learning_process"]) == ["v"]


def test_register_ssm_caches_prebuilt_function(
    annotated_ssm_base_logp: Any,
) -> None:
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


def test_register_ssm_rejects_precomputed_function(
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


def test_get_rlssm_model_config_unknown_model_raises() -> None:
    """Unknown RLSSM model names should fail with a clear error."""
    with pytest.raises(ValueError, match="not found in the RLSSM registry"):
        registry.get_rlssm_model_config("does_not_exist")

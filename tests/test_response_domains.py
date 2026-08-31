"""Tests for canonical response-domain configuration."""

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import bambi as bmb
import pytest

import hssm
import hssm.config as config_module
from hssm.config import Config, ModelConfig
from hssm.defaults import default_model_config
from hssm.register import register_model


def _config(
    response: list[str],
    *,
    response_domains: Mapping[str, Mapping[str, object]] | None = None,
    choices: tuple[int, ...] | None = None,
) -> Config:
    return Config(
        model_name="custom_response_domains",
        loglik_kind="analytical",
        response=response,
        response_domains=response_domains,  # type: ignore[arg-type]
        choices=choices,
        list_params=["v"],
        loglik=lambda *args: 0.0,
    )


def test_legacy_choices_resolve_to_one_categorical_domain():
    """Legacy choices normalize to canonical metadata and remain projected."""
    config = _config(["rt", "response"], choices=(-1, 1))

    config.validate()

    assert config.response_domains == {
        "response": {"kind": "categorical", "values": (-1, 1)}
    }
    assert config.choices == (-1, 1)


def test_resolved_config_accepts_matching_derived_choices():
    """Resolved configs remain idempotent with an exact compatibility view."""
    config = _config(
        ["rt", "response"],
        response_domains={"response": {"kind": "categorical", "values": (0, 1)}},
        choices=(0, 1),
    )

    config.validate()
    assert config.choices == (0, 1)


def test_mutating_a_resolved_domain_snapshot_fails_closed():
    """Nested config mutation cannot silently stale a compatibility view."""
    config = _config(
        ["rt", "response"],
        response_domains={"response": {"kind": "categorical", "values": (0, 1)}},
    )
    config.validate()
    assert config.response_domains is not None

    config.response_domains["response"]["values"] = (2, 3)

    with pytest.raises(ValueError, match="either `response_domains` or legacy"):
        config.validate()


def test_domains_follow_physical_response_order_and_derive_no_global_choices():
    """Canonical domains follow physical response order without global choices."""
    config = _config(
        ["rt", "polar", "azimuth"],
        response_domains={
            "azimuth": {"kind": "circular", "bounds": (-3.14, 3.14)},
            "polar": {"kind": "continuous", "bounds": (0, 3.14)},
        },
    )

    config.validate()

    assert list(config.response_domains) == ["polar", "azimuth"]
    assert config.response_domains == {
        "polar": {"kind": "continuous", "bounds": (0.0, 3.14)},
        "azimuth": {"kind": "circular", "bounds": (-3.14, 3.14)},
    }
    assert config.choices is None


def test_single_categorical_domain_derives_legacy_choices():
    """One categorical domain retains the established choices projection."""
    config = _config(
        ["response"],
        response_domains={"response": {"kind": "categorical", "values": [0, 2, 4]}},
    )

    config.validate()

    assert config.response_domains == {
        "response": {"kind": "categorical", "values": (0, 2, 4)}
    }
    assert config.choices == (0, 2, 4)
    assert config.is_choice_only

    config.validate()
    assert config.choices == (0, 2, 4)


@pytest.mark.parametrize(
    ("response", "domains", "choices", "message"),
    [
        (["rt", "response"], None, None, "Provide `response_domains`"),
        (["rt"], {}, None, "At least one non-RT"),
        (["response", "other"], {}, None, "without `rt`"),
        (["response", "rt"], {}, None, "index zero"),
        (["rt", "rt", "response"], {}, None, "unique"),
        (
            ["rt", "response"],
            {"response": {"kind": "continuous"}},
            (0, 1),
            "either `response_domains` or legacy `choices`",
        ),
        (
            ["rt", "response"],
            {"other": {"kind": "continuous"}},
            None,
            "keys must match",
        ),
        (
            ["rt", "response"],
            {"response": {"kind": "ordinal"}},
            None,
            "invalid kind",
        ),
        (
            ["rt", "response"],
            {"response": {"kind": "continuous", "values": [0, 1]}},
            None,
            "unknown fields",
        ),
        (
            ["rt", "response"],
            {"response": {"kind": "categorical", "values": []}},
            None,
            "requires values",
        ),
        (
            ["rt", "response"],
            {"response": {"kind": "categorical", "values": [0, 0]}},
            None,
            "must be distinct",
        ),
        (
            ["rt", "response"],
            {"response": {"kind": "categorical", "values": [0, 0.5]}},
            None,
            "must be integers",
        ),
        (
            ["rt", "response"],
            {"response": {"kind": "circular"}},
            None,
            "requires two bounds",
        ),
        (
            ["rt", "response"],
            {"response": {"kind": "continuous", "bounds": None}},
            None,
            "requires two bounds",
        ),
        (
            ["rt", "response"],
            {"response": {"kind": "circular", "bounds": (0, float("inf"))}},
            None,
            "finite and strictly increasing",
        ),
        (
            ["rt", "response"],
            {"response": {"kind": "continuous", "bounds": (1, 1)}},
            None,
            "finite and strictly increasing",
        ),
    ],
)
def test_invalid_response_domain_contracts_fail(
    response: list[str],
    domains: dict[str, dict[str, Any]] | None,
    choices: tuple[int, ...] | None,
    message: str,
):
    """Malformed, incomplete, and ambiguous domain declarations fail closed."""
    config = _config(response, response_domains=domains, choices=choices)

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_model_config_rejects_canonical_and_legacy_inputs_together():
    """Canonical metadata cannot be combined with explicit legacy choices."""
    model_config = ModelConfig(
        response=("rt", "response"),
        response_domains={"response": {"kind": "continuous"}},
    )

    with pytest.raises(ValueError, match="either `response_domains` or legacy"):
        Config._build_model_config(
            "ddm", None, model_config, choices=(-1, 1), loglik=None
        )


def test_model_config_detaches_nested_domain_input():
    """ModelConfig normalization owns a detached canonical mapping."""
    domains: dict[str, Any] = {
        "response": {"kind": "circular", "bounds": [-3.14, 3.14]}
    }
    model_config = ModelConfig(response=("rt", "response"), response_domains=domains)

    config = Config._build_model_config("ddm", None, model_config, None)
    domains["response"]["bounds"][0] = 0.0

    assert config.response_domains == {
        "response": {"kind": "circular", "bounds": (-3.14, 3.14)}
    }
    assert config.choices is None


def test_registered_domains_are_detached_and_resolve_canonically():
    """Registration detaches caller input and reconstructs canonical config."""
    name = "custom_registered_response_domains"
    domains: dict[str, Any] = {"response": {"kind": "continuous", "bounds": [0.0, 1.0]}}
    likelihoods = {
        "analytical": {
            "loglik": lambda *args: 0.0,
            "backend": None,
            "default_priors": {},
            "bounds": {},
            "extra_fields": None,
        }
    }
    register_model(
        name=name,  # type: ignore[arg-type]
        response=["rt", "response"],
        list_params=["v"],
        choices=None,
        response_domains=domains,
        likelihoods=likelihoods,  # type: ignore[arg-type]
        description=None,
    )
    try:
        domains["response"]["bounds"][0] = -1.0
        config = Config.from_defaults(name, "analytical")
        config.validate()

        assert config.response_domains == {
            "response": {"kind": "continuous", "bounds": (0.0, 1.0)}
        }
        assert config.choices is None
    finally:
        default_model_config.pop(name, None)  # type: ignore[arg-type]


def test_registered_canonical_model_rejects_top_level_choices_override():
    """A registered canonical source cannot be combined with legacy choices."""
    name = "custom_registered_choice_conflict"
    register_model(
        name=name,  # type: ignore[arg-type]
        response=["rt", "response"],
        list_params=["v"],
        choices=None,
        response_domains={"response": {"kind": "categorical", "values": (0, 1)}},
        likelihoods={
            "analytical": {
                "loglik": lambda *args: 0.0,
                "backend": None,
                "default_priors": {},
                "bounds": {},
                "extra_fields": None,
            }
        },  # type: ignore[arg-type]
        description=None,
    )
    try:
        with pytest.raises(ValueError, match="either `response_domains` or legacy"):
            Config._build_model_config(
                name, "analytical", None, choices=(0, 1), loglik=None
            )
    finally:
        default_model_config.pop(name, None)  # type: ignore[arg-type]


def test_registered_canonical_model_ignores_ssms_legacy_fallback(monkeypatch):
    """Canonical registration wins over an ssms registry name collision."""
    name = "custom_registered_ssms_collision"
    register_model(
        name=name,  # type: ignore[arg-type]
        response=["rt", "response"],
        list_params=["v"],
        choices=None,
        response_domains={"response": {"kind": "categorical", "values": (0, 1)}},
        likelihoods={
            "analytical": {
                "loglik": lambda *args: 0.0,
                "backend": None,
                "default_priors": {},
                "bounds": {},
                "extra_fields": None,
            }
        },  # type: ignore[arg-type]
        description=None,
    )
    monkeypatch.setitem(config_module.ssms_model_config, name, {"choices": (8, 9)})
    try:
        config = Config._build_model_config(name, "analytical", None, None)

        assert config.choices == (0, 1)
    finally:
        default_model_config.pop(name, None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("model", "loglik_kind", "choices"),
    [
        ("ddm", "analytical", (-1, 1)),
        ("ddm_seq2_no_bias", "approx_differentiable", (0, 1, 2, 3)),
        ("lba4", "analytical", (0, 1, 2, 3)),
        ("softmax_inv_temperature_3", "analytical", (0, 1, 2)),
    ],
)
def test_legacy_builtins_resolve_without_changing_response_or_choices(
    model, loglik_kind, choices
):
    """Existing categorical factories acquire canonical internal metadata only."""
    config = Config.from_defaults(model, loglik_kind)

    config.validate()

    response_column = config.response[-1]
    assert config.response_domains == {
        response_column: {"kind": "categorical", "values": choices}
    }
    assert config.choices == choices


@pytest.mark.parametrize("as_dict", [False, True])
def test_constructor_snapshot_detaches_nested_response_domains(as_dict):
    """Save/load constructor arguments own their nested domain metadata."""
    domains: dict[str, Any] = {"response": {"kind": "continuous", "bounds": [0.0, 1.0]}}
    model_config: ModelConfig | dict[str, Any]
    if as_dict:
        model_config = {"response_domains": domains}
    else:
        model_config = ModelConfig(response_domains=domains)

    snapshot = hssm.HSSM._store_init_args(
        {"self": object(), "model_config": model_config}, {}
    )
    domains["response"]["bounds"][0] = -1.0
    stored = snapshot["model_config"]
    stored_domains = (
        stored["response_domains"]
        if isinstance(stored, dict)
        else stored.response_domains
    )

    assert stored_domains["response"]["bounds"] == [0.0, 1.0]

    stored_domains["response"]["bounds"][1] = 2.0
    assert domains["response"]["bounds"] == [-1.0, 1.0]


def test_model_config_positional_arguments_keep_their_legacy_meaning():
    """Appending response domains does not shift existing positional fields."""
    config = ModelConfig(("rt", "response"), ["v"], (-1, 1))

    assert config.response == ("rt", "response")
    assert config.list_params == ["v"]
    assert config.choices == (-1, 1)
    assert config.response_domains is None

    domain_field = next(
        field for field in fields(Config) if field.name == "response_domains"
    )
    assert domain_field.kw_only


def test_live_model_owns_one_detached_response_domain_mapping():
    """Live config and validation share one mapping detached from the caller."""
    domains: dict[str, Any] = {"response": {"kind": "categorical", "values": [-1, 1]}}
    model = hssm.HSSM(
        data=hssm.load_data("cavanagh_theta").head(8),
        model_config=ModelConfig(response_domains=domains),
        p_outlier=None,
        process_initvals=False,
    )

    domains["response"]["values"][0] = -2
    assert model.response_domains == {
        "response": {"kind": "categorical", "values": (-1, 1)}
    }
    assert model.response_domains is model.model_config.response_domains


def _lapse_shell(
    domains: dict[str, dict[str, Any]], *, choice_only: bool = False
) -> hssm.HSSM:
    model = object.__new__(hssm.HSSM)
    model.list_params = ["v"]
    model.response_domains = domains  # type: ignore[assignment]
    model.has_lapse = True
    model.is_choice_only = choice_only
    model.n_choices = 2 if choice_only else None
    return model


def test_lapse_remains_available_for_established_categorical_layouts():
    """RT+choice and choice-only models retain their established lapse behavior."""
    domains = {"response": {"kind": "categorical", "values": (0, 1)}}

    rt_model = _lapse_shell(domains)
    rt_model._check_lapse(None)
    assert isinstance(rt_model.lapse, bmb.Prior)

    choice_only_model = _lapse_shell(domains, choice_only=True)
    choice_only_model._check_lapse(None)
    assert choice_only_model.lapse == 0.5


@pytest.mark.parametrize(
    "domains",
    [
        {"response": {"kind": "continuous"}},
        {"response": {"kind": "circular", "bounds": (-3.14, 3.14)}},
        {
            "first": {"kind": "categorical", "values": (0, 1)},
            "second": {"kind": "categorical", "values": (0, 1)},
        },
        {
            "first": {"kind": "continuous"},
            "second": {"kind": "categorical", "values": (0, 1)},
        },
    ],
)
def test_lapse_rejects_noncategorical_or_multiresponse_layouts(domains):
    """Active outlier mixtures fail before unsupported domain layouts build."""
    model = _lapse_shell(domains)

    with pytest.raises(ValueError, match="only for one categorical response"):
        model._check_lapse(None)


def test_inactive_lapse_accepts_mixed_domains():
    """None or zero p_outlier maps to an inactive lapse for mixed responses."""
    model = _lapse_shell(
        {
            "first": {"kind": "continuous"},
            "second": {"kind": "categorical", "values": (0, 1)},
        }
    )
    model.has_lapse = False

    model._check_lapse(None)

    assert model.lapse is None
    assert model.list_params == ["v"]

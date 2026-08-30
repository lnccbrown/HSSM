"""Tests for canonical response-domain configuration."""

from collections.abc import Mapping
from typing import Any

import pytest

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

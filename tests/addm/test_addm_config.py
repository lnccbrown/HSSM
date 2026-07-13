"""Commit 3 — verify the aDDMConfig dataclass.

Pure-config tests (no jax/pytensor compute), but importing ``hssm.addm.config``
pulls in the ``hssm`` package, so run in the repo's uv venv::

    cd .../HSSM && source .venv/bin/activate
    JAX_PLATFORMS=cpu python src/hssm/addm/commit_tests/test_commit3_config.py
    # or: pytest src/hssm/addm/commit_tests/test_commit3_config.py -v

Patterned after tests/test_rlssm_config.py.
"""

import pytest

from hssm.addm.config import aDDMConfig
from hssm.config import BaseModelConfig

EXPECTED_LIST_PARAMS = ["eta", "kappa", "a", "b", "x0", "t"]
EXPECTED_EXTRA_FIELDS = ["r1", "r2", "flag", "sacc_array", "d", "sigma"]


def test_defaults():
    c = aDDMConfig()
    assert c.model_name == "addm"
    assert c.list_params == EXPECTED_LIST_PARAMS
    assert c.extra_fields == EXPECTED_EXTRA_FIELDS
    assert set(c.bounds) == set(EXPECTED_LIST_PARAMS)
    assert c.choices == (-1, 1)
    assert c.response == ["rt", "response"]
    assert c.attention_process == "standard_alternating"
    assert c.loglik_kind == "approx_differentiable"
    # params_default valid: same length as list_params, every value in-bounds.
    assert len(c.params_default) == len(c.list_params)
    for name, val in zip(c.list_params, c.params_default):
        lo, hi = c.bounds[name]
        assert lo <= val <= hi, f"default {name}={val} outside bounds {(lo, hi)}"
    # loglik / backend left for aDDM.__init__ (Commit 4) to inject.
    assert c.loglik is None
    assert c.backend is None


def test_instantiable_and_subclass():
    assert issubclass(aDDMConfig, BaseModelConfig)
    # Constructing at all proves both abstract methods (validate, get_defaults)
    # are overridden.
    assert isinstance(aDDMConfig(), aDDMConfig)


def test_validate_ok():
    aDDMConfig().validate()  # must not raise


def test_validate_rejects_unknown_attention_process():
    with pytest.raises(ValueError):
        aDDMConfig(attention_process="bogus").validate()


def test_validate_rejects_bounds_param_mismatch():
    with pytest.raises(ValueError):
        aDDMConfig(bounds={"eta": (0.0, 1.0)}).validate()


def test_validate_rejects_params_default_mismatch():
    with pytest.raises(ValueError):
        aDDMConfig(params_default=[0.3]).validate()


def test_get_defaults():
    c = aDDMConfig()
    assert c.get_defaults("eta") == (None, (0.0, 1.0))
    assert c.get_defaults("nope") == (None, None)


def test_from_addm_dict_roundtrip():
    src = aDDMConfig()
    payload = {
        "model_name": src.model_name,
        "list_params": list(src.list_params),
        "bounds": dict(src.bounds),
        "extra_fields": list(src.extra_fields),
        "params_default": list(src.params_default),
        "attention_process": src.attention_process,
        "choices": src.choices,
        "response": list(src.response),
        "unknown_key": "ignored",  # must be dropped, not crash
    }
    rebuilt = aDDMConfig.from_addm_dict(payload)
    rebuilt.validate()
    assert rebuilt.list_params == src.list_params
    assert rebuilt.bounds == src.bounds
    assert rebuilt.extra_fields == src.extra_fields
    assert rebuilt.params_default == src.params_default
    assert rebuilt.attention_process == src.attention_process
    assert rebuilt.model_name == src.model_name


if __name__ == "__main__":
    for fn in (
        test_defaults,
        test_instantiable_and_subclass,
        test_validate_ok,
        test_validate_rejects_unknown_attention_process,
        test_validate_rejects_bounds_param_mismatch,
        test_validate_rejects_params_default_mismatch,
        test_get_defaults,
        test_from_addm_dict_roundtrip,
    ):
        fn()
        print(f"PASSED: {fn.__name__}")
    print("\nAll Commit 3 config checks passed.")

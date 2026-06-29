"""Structure-only tests for per-parameter centered/non-centered parameterization.

A group-specific term is non-centered iff bambi emits a ``{label}_offset ~
Normal(0, 1)`` free RV (and a ``Deterministic(label, offset * sigma)``); a
centered term has a direct ``{label}`` group RV instead. We assert on the
presence/absence of ``*_offset`` free RVs, so no sampling is required.

These exercise bambi PR #983 (per-parameter ``noncentered``) end-to-end through
HSSM and require a bambi version that includes it.
"""

import pytest

import hssm
from hssm import Prior
from hssm.prior import _bambi_supports_per_parameter_noncentered

hssm.set_floatX("float32")

pytestmark = pytest.mark.skipif(
    not _bambi_supports_per_parameter_noncentered(),
    reason="per-parameter `noncentered` requires a bambi version with PR #983",
)

GS = "1|participant_id"


def _offset_params(model) -> set[str]:
    """HSSM params whose group term carries a non-centered ``*_offset`` RV."""
    return {
        name.split("_")[0]
        for name in (rv.name for rv in model.pymc_model.free_RVs)
        if "_offset" in name
    }


def _hierarchical_v(prior=None) -> list[dict]:
    """Include-spec for a hierarchical ``v ~ 1 + (1|participant_id)``."""
    spec: dict = {"name": "v", "formula": f"v ~ 1 + ({GS})"}
    if prior is not None:
        spec["prior"] = prior
    return [spec]


def _build(cavanagh_test, include, **kwargs):
    return hssm.HSSM(
        data=cavanagh_test, model="ddm", include=include, p_outlier=0.0, **kwargs
    )


def test_default_is_noncentered(cavanagh_test):
    """bambi's default (`noncentered=True`) non-centers the group term."""
    model = _build(cavanagh_test, _hierarchical_v())
    assert "v" in _offset_params(model)


def test_scalar_noncentered_both_directions(cavanagh_test):
    """The plain `bool` form works on every bambi and is not guarded."""
    assert (
        _offset_params(_build(cavanagh_test, _hierarchical_v(), noncentered=False))
        == set()
    )
    assert "v" in _offset_params(
        _build(cavanagh_test, _hierarchical_v(), noncentered=True)
    )


def test_model_level_dict_per_parameter(cavanagh_test):
    """`noncentered={"v": ...}` centers/non-centers just that parameter."""
    centered = _build(cavanagh_test, _hierarchical_v(), noncentered={"v": False})
    assert _offset_params(centered) == set()

    noncentered = _build(cavanagh_test, _hierarchical_v(), noncentered={"v": True})
    assert "v" in _offset_params(noncentered)


def test_unknown_dict_key_raises_at_construction(cavanagh_test):
    """A typo'd component name fails loudly with the valid names listed."""
    with pytest.raises(ValueError, match="[Uu]nknown component"):
        _build(cavanagh_test, _hierarchical_v(), noncentered={"nonexistent": True})


@pytest.mark.parametrize("flag, expect_offset", [(True, True), (False, False)])
def test_per_prior_dict_overrides_model_dict(cavanagh_test, flag, expect_offset):
    """A `noncentered` field in a prior dict beats the model-level dict."""
    prior = {
        GS: {
            "name": "Normal",
            "mu": 0.0,
            "sigma": {"name": "HalfNormal", "sigma": 1.0},
            "noncentered": flag,
        }
    }
    # model-level sets the OPPOSITE so a per-prior win is unambiguous.
    model = _build(cavanagh_test, _hierarchical_v(prior), noncentered={"v": not flag})
    assert ("v" in _offset_params(model)) is expect_offset


@pytest.mark.parametrize("flag, expect_offset", [(True, True), (False, False)])
def test_per_prior_object_overrides_model_dict(cavanagh_test, flag, expect_offset):
    """An `hssm.Prior(noncentered=...)` object beats the model-level dict."""
    prior = {
        GS: Prior(
            "Normal", mu=0.0, sigma=Prior("HalfNormal", sigma=1.0), noncentered=flag
        )
    }
    assert prior[GS].noncentered is flag  # captured as a named attr, not in .args
    model = _build(cavanagh_test, _hierarchical_v(prior), noncentered={"v": not flag})
    assert ("v" in _offset_params(model)) is expect_offset


def test_guard_raises_on_unsupported_bambi(cavanagh_test, monkeypatch):
    """With a bambi lacking PR #983, every per-parameter form raises clearly."""
    monkeypatch.setattr(
        "hssm.prior._bambi_supports_per_parameter_noncentered", lambda: False
    )
    monkeypatch.setattr(
        "hssm.base._bambi_supports_per_parameter_noncentered", lambda: False
    )
    # per-prior object path (guarded in hssm.Prior)
    with pytest.raises(ValueError, match="PR #983"):
        Prior("Normal", mu=0.0, sigma=1.0, noncentered=True)
    # model-level dict path (guarded in base.py)
    with pytest.raises(ValueError, match="PR #983"):
        _build(cavanagh_test, _hierarchical_v(), noncentered={"v": False})
    # per-prior dict path (guarded in base.py)
    prior = {
        GS: {
            "name": "Normal",
            "mu": 0.0,
            "sigma": {"name": "HalfNormal", "sigma": 1.0},
            "noncentered": True,
        }
    }
    with pytest.raises(ValueError, match="PR #983"):
        _build(cavanagh_test, _hierarchical_v(prior))

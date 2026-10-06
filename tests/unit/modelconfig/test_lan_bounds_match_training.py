"""The `approx_differentiable` bounds must describe the network's training box.

These bounds are not a modelling preference — they are the region the LAN was
fit on. Outside it the network does not fail; it extrapolates and returns a
finite, plausible, wrong density, which the sampler will happily explore. So a
bound wider than the training box is a correctness bug, and one narrower than
it silently withholds range the network was validated on.

The source of truth is ssms' `param_bounds` for the same model.

The RT support edge (`ndt_edge_shift`) is held to the same source for a
different reason. The network does not learn the edge: its labels are a KDE
fitted in log-RT space to simulated RTs, which smooths across the edge and is
floored only at rt <= 0, so below the simulator's edge the network returns
small, finite, meaningless values and above it an approximation of real
density. HSSM's floor is the correction that restores the model's zero density
below the edge, so it has to sit exactly where the simulator's support starts:
higher, and it clips density the network fitted; lower, and the KDE's leakage
stays in the likelihood for the sampler to exploit.
"""

import sys

import pytest
import ssms

from hssm.modelconfig import get_default_model_config
from hssm.defaults import SupportedModels
from typing import get_args

# Bounds HSSM declares that do NOT match the training box, with the reason.
# Pinned exactly: a new mismatch fails this test, and so does *fixing* one
# without removing it here — which is the point. A silently-growing waiver
# list is how this kind of drift becomes permanent.
KNOWN_MISMATCHES = {
    # HSSM allows z in (0, 1) while ddm.onnx was trained on [0.1, 0.9], so the
    # sampler can explore 20% of the interval where the network extrapolates.
    # Widening the training box or narrowing this bound is a user-facing
    # change to the most-used model in the ecosystem; tracked separately.
    ("ddm", "z"),
}


def _models_with_lan_bounds():
    for name in get_args(SupportedModels):
        try:
            cfg = get_default_model_config(name)
        except Exception:
            continue
        lik = (cfg.get("likelihoods") or {}).get("approx_differentiable")
        if lik and lik.get("bounds") and name in ssms.config.model_config:
            yield name, lik["bounds"]


def test_declared_bounds_match_the_training_box():
    found = set()
    for name, bounds in _models_with_lan_bounds():
        mc = ssms.config.model_config[name]
        lo, hi = mc["param_bounds"]
        train = dict(zip(mc["params"], zip(map(float, lo), map(float, hi))))
        for param, (declared_lo, declared_hi) in bounds.items():
            if param not in train:
                continue
            train_lo, train_hi = train[param]
            # The lower edge is conventionally rounded (0.001 -> 0.0); the
            # upper edge and any real difference are not.
            mismatch = (
                abs(declared_hi - train_hi) > 1e-6 or abs(declared_lo - train_lo) > 0.01
            )
            if mismatch:
                found.add((name, param))
    assert found == KNOWN_MISMATCHES, (
        f"bounds drifted from the training box.\n"
        f"  newly mismatched: {sorted(found - KNOWN_MISMATCHES)}\n"
        f"  fixed (remove from KNOWN_MISMATCHES): {sorted(KNOWN_MISMATCHES - found)}"
    )


def _normalised_edge(declaration):
    if declaration is None:
        return None
    return {
        "param": str(declaration["param"]),
        "scale": float(declaration["scale"]),
    }


def _support_edge_mismatches():
    """LAN models whose `ndt_edge_shift` differs from ssms', as (name, hssm, ssms).

    Only `approx_differentiable` likelihoods are compared: the floor marks
    where the network's output stops approximating the model's density, and
    that is the simulator's support edge (see the module docstring), so HSSM's
    declaration must equal ssms' for the same model (absent on both sides
    means the support starts at t).
    The simulator is `rv` when the config declares one (hssm.py resolves it
    the same way), else the model name. Blackbox and analytical likelihoods
    keep their own convention — hddm_wfpt reads st as the full width, so
    full_ddm declares scale 0.5 where ssms says 1.0 — and are deliberately
    left out.
    """
    found = []
    for name in get_args(SupportedModels):
        try:
            cfg = get_default_model_config(name)
        except Exception:
            continue
        lik = (cfg.get("likelihoods") or {}).get("approx_differentiable")
        if not lik:
            continue
        sim = lik.get("rv") if isinstance(lik.get("rv"), str) else name
        if sim not in ssms.config.model_config:
            continue
        declared = _normalised_edge(lik.get("ndt_edge_shift"))
        trained = _normalised_edge(ssms.config.model_config[sim].get("ndt_edge_shift"))
        if declared != trained:
            found.append((name, declared, trained))
    return sorted(found, key=lambda row: row[0])


def test_declared_support_edge_matches_the_simulator():
    """A LAN likelihood's floor sits at the simulator's support edge.

    Blackbox and analytical likelihoods own their own edge and are not checked
    here; see `_support_edge_mismatches`.
    """
    if not any("ndt_edge_shift" in mc for mc in ssms.config.model_config.values()):
        pytest.skip("installed ssms declares no ndt_edge_shift; nothing to compare")
    mismatched = _support_edge_mismatches()
    assert not mismatched, (
        "support edge drifted from the simulator (model, hssm, ssms):\n"
        + "\n".join(f"  {row}" for row in mismatched)
    )


def test_support_edge_guard_sees_drift_on_either_side(monkeypatch):
    # The guard above skips until ssms ships the key; pin here that it will
    # actually report drift from either direction once it runs.
    shift = {"param": "st", "scale": 1.0}

    # The ssms registry deep-copies on lookup, so replace the entry itself.
    patched = ssms.config.model_config["angle"]
    patched["ndt_edge_shift"] = shift
    with monkeypatch.context() as m:
        m.setitem(ssms.config.model_config, "angle", patched)
        assert ("angle", None, shift) in _support_edge_mismatches()

    # The helper reads the name bound in this module, so patch that binding.
    real = get_default_model_config

    def with_edge(name):
        cfg = real(name)  # a fresh dict per call
        if name == "angle":
            cfg["likelihoods"]["approx_differentiable"]["ndt_edge_shift"] = shift
        return cfg

    with monkeypatch.context() as m:
        m.setattr(sys.modules[__name__], "get_default_model_config", with_edge)
        assert ("angle", shift, None) in _support_edge_mismatches()

    # A declaration on both sides that differs only in scale must also be
    # reported — that is the full_ddm 0.5-vs-1.0 case, not a missing key.
    half = {"param": "st", "scale": 0.5}

    def with_half(name):
        cfg = real(name)
        if name == "angle":
            cfg["likelihoods"]["approx_differentiable"]["ndt_edge_shift"] = half
        return cfg

    with monkeypatch.context() as m:
        m.setitem(ssms.config.model_config, "angle", patched)
        m.setattr(sys.modules[__name__], "get_default_model_config", with_half)
        assert ("angle", half, shift) in _support_edge_mismatches()


def test_support_edge_guard_resolves_the_simulator_via_rv(monkeypatch):
    # A LAN that aliases its simulator (`"rv": "ddm_st"`, as ddm_uniform_st
    # does) was trained on *that* simulator's output, so the guard must compare
    # against the alias target's declaration, not the model name's.
    shift = {"param": "st", "scale": 1.0}

    patched = ssms.config.model_config["ddm_st"]
    patched["ndt_edge_shift"] = shift
    real = get_default_model_config

    def with_rv(name):
        cfg = real(name)
        if name == "angle":
            cfg["likelihoods"]["approx_differentiable"]["rv"] = "ddm_st"
        return cfg

    with monkeypatch.context() as m:
        m.setitem(ssms.config.model_config, "ddm_st", patched)
        m.setattr(sys.modules[__name__], "get_default_model_config", with_rv)
        assert ("angle", None, shift) in _support_edge_mismatches()


def test_ddm_sdv_sv_covers_the_full_trained_range():
    # Regression: this was (0.0, 1.0) — written before any ddm_sdv.onnx
    # existed — while training and the density gate both cover sv up to 2.5.
    bounds = get_default_model_config("ddm_sdv")["likelihoods"][
        "approx_differentiable"
    ]["bounds"]
    assert bounds["sv"] == (0.0, 2.5)

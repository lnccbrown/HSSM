"""The `approx_differentiable` bounds must describe the network's training box.

These bounds are not a modelling preference — they are the region the LAN was
fit on. Outside it the network does not fail; it extrapolates and returns a
finite, plausible, wrong density, which the sampler will happily explore. So a
bound wider than the training box is a correctness bug, and one narrower than
it silently withholds range the network was validated on.

The source of truth is ssms' `param_bounds` for the same model.
"""

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


def test_ddm_sdv_sv_covers_the_full_trained_range():
    # Regression: this was (0.0, 1.0) — written before any ddm_sdv.onnx
    # existed — while training and the density gate both cover sv up to 2.5.
    bounds = get_default_model_config("ddm_sdv")["likelihoods"][
        "approx_differentiable"
    ]["bounds"]
    assert bounds["sv"] == (0.0, 2.5)

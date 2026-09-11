"""Tests for restoring the JAX-NUTS "no jitter" behavior (issue #999).

HSSM disables the built-in initial-value jitter of PyMC's JAX samplers
("numpyro"/"blackjax") in favor of its own controlled `initval_jitter`. Because
PyMC 6 exposes no supported switch, HSSM does this by temporarily patching
`pymc.sampling.jax.sample_jax_nuts` via `_force_jax_nuts_no_jitter`.
"""

import pymc.sampling.jax as pymc_jax
import pytest

from hssm.utils import _force_jax_nuts_no_jitter


def test_cm_injects_jitter_false_and_restores(monkeypatch):
    """Inside the CM, `sample_jax_nuts` receives `jitter=False`; restored on exit."""
    captured = {}

    def fake_sample_jax_nuts(*args, **kwargs):
        captured["jitter"] = kwargs.get("jitter", "MISSING")
        return "idata-sentinel"

    monkeypatch.setattr(pymc_jax, "sample_jax_nuts", fake_sample_jax_nuts)
    original = pymc_jax.sample_jax_nuts

    with _force_jax_nuts_no_jitter(active=True):
        # The attribute is swapped for the wrapper while active.
        assert pymc_jax.sample_jax_nuts is not original
        # A caller-supplied jitter=True is overridden to False and delegated.
        assert pymc_jax.sample_jax_nuts(jitter=True) == "idata-sentinel"
        assert captured["jitter"] is False

    # Original (here, the monkeypatched fake) is restored on exit.
    assert pymc_jax.sample_jax_nuts is original


def test_cm_inactive_is_noop(monkeypatch):
    """`active=False` leaves `sample_jax_nuts` untouched."""
    sentinel = object()
    monkeypatch.setattr(pymc_jax, "sample_jax_nuts", sentinel)

    with _force_jax_nuts_no_jitter(active=False):
        assert pymc_jax.sample_jax_nuts is sentinel

    assert pymc_jax.sample_jax_nuts is sentinel


def test_cm_missing_attr_is_noop(monkeypatch):
    """A future pymc without `sample_jax_nuts` degrades to a no-op, no raise."""
    monkeypatch.delattr(pymc_jax, "sample_jax_nuts", raising=False)

    with _force_jax_nuts_no_jitter(active=True):
        assert not hasattr(pymc_jax, "sample_jax_nuts")

    assert not hasattr(pymc_jax, "sample_jax_nuts")


class _StopSampling(Exception):
    """Sentinel raised to abort once the jitter kwarg has been captured."""


def test_hssm_numpyro_forces_jitter_false(monkeypatch, basic_hssm_model):
    """`sampler="numpyro"` must reach `sample_jax_nuts` with `jitter=False`.

    Exercises the whole HSSM -> bambi -> pm.sample dispatch (including HSSM's own
    context manager) but aborts at the `pymc.sampling.jax` boundary, so no MCMC
    and no JAX sampler compilation actually run.
    """
    captured = {}

    def fake_sample_jax_nuts(*args, **kwargs):
        captured["jitter"] = kwargs.get("jitter", "MISSING")
        raise _StopSampling

    monkeypatch.setattr(pymc_jax, "sample_jax_nuts", fake_sample_jax_nuts)

    with pytest.raises(_StopSampling):
        basic_hssm_model.sample(sampler="numpyro", draws=10, tune=10, chains=1, cores=1)

    assert captured.get("jitter") is False

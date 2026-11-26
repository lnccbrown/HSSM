"""Tests for the JAX-NUTS "no jitter" default (issue #999).

HSSM applies its own controlled `initval_jitter`, so for PyMC's JAX samplers
("numpyro"/"blackjax") it disables the sampler's built-in uniform(-1, 1)
initial-value jitter by default. Since PyMC 6.3.0 (pymc#8380), `pm.sample` lifts
`nuts={"jitter": ...}` (and other sampler-driver options such as `chain_method`)
out of the NUTS-kernel kwargs and passes them as top-level arguments to
`pymc.sampling.jax.sample_jax_nuts`. HSSM therefore merges `jitter=False` *under*
the user's `nuts` dict -- a user-supplied value wins -- and no patching of pymc is
involved.

Each test replaces `sample_jax_nuts` with a fake that records its kwargs and raises
a sentinel, so the whole HSSM -> bambi -> pm.sample dispatch is exercised but no
MCMC and no JAX sampler compilation actually run.
"""

import pymc.sampling.jax as pymc_jax
import pytest


class _StopSampling(Exception):
    """Sentinel raised to abort once the kwargs have been captured."""


def _capture_sample_jax_nuts(monkeypatch) -> dict:
    """Swap `sample_jax_nuts` for a fake that records its kwargs, then aborts."""
    captured: dict = {}

    def fake_sample_jax_nuts(*args, **kwargs):
        captured.update(kwargs)
        raise _StopSampling

    monkeypatch.setattr(pymc_jax, "sample_jax_nuts", fake_sample_jax_nuts)
    return captured


def test_hssm_numpyro_forces_jitter_false(monkeypatch, basic_hssm_model):
    """`sampler="numpyro"` must reach `sample_jax_nuts` with `jitter=False`."""
    captured = _capture_sample_jax_nuts(monkeypatch)

    with pytest.raises(_StopSampling):
        basic_hssm_model.sample(sampler="numpyro", draws=10, tune=10, chains=1, cores=1)

    assert captured.get("jitter", "MISSING") is False


def test_hssm_numpyro_user_jitter_wins(monkeypatch, basic_hssm_model):
    """A user-supplied `nuts={"jitter": True}` overrides HSSM's default."""
    captured = _capture_sample_jax_nuts(monkeypatch)

    with pytest.raises(_StopSampling):
        basic_hssm_model.sample(
            sampler="numpyro",
            draws=10,
            tune=10,
            chains=1,
            cores=1,
            nuts={"jitter": True},
        )

    assert captured.get("jitter", "MISSING") is True


def test_hssm_numpyro_forwards_other_nuts_keys(monkeypatch, basic_hssm_model):
    """Other `nuts` keys reach `sample_jax_nuts` alongside HSSM's `jitter=False`."""
    captured = _capture_sample_jax_nuts(monkeypatch)

    with pytest.raises(_StopSampling):
        basic_hssm_model.sample(
            sampler="numpyro",
            draws=10,
            tune=10,
            chains=1,
            cores=1,
            nuts={"chain_method": "vectorized"},
        )

    assert captured.get("chain_method") == "vectorized"
    assert captured.get("jitter", "MISSING") is False

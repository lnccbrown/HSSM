"""Commit 1 — verify the vendored efficient-fpt JAX likelihood.

Asserts the vendoring in isolation: the retargeted imports resolve, the batched
kernel runs and matches the per-trial kernel, and the HSSM-authored
``compute_addm_loglikelihoods_from_mu`` wrapper agrees with the default path.

The vendored ``likelihoods/jax`` package is loaded directly from disk under a
synthetic module name, so this test runs with only ``jax`` + ``numpy`` installed
— it does **not** import the full ``hssm`` package (which needs bambi/pymc).

Run with either::

    python test_commit1_vendored_jax.py
    pytest test_commit1_vendored_jax.py -v
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Load the vendored jax/ subpackage in isolation (avoid full hssm import and
# avoid shadowing the real `jax` library by using a synthetic package name).
# ---------------------------------------------------------------------------
_JAX_PKG_DIR = Path(__file__).resolve().parent.parent / "likelihoods" / "jax"
_PKG_NAME = "_addm_vendored_jax_under_test"


def _load_vendored_pkg():
    spec = importlib.util.spec_from_file_location(
        _PKG_NAME,
        _JAX_PKG_DIR / "__init__.py",
        submodule_search_locations=[str(_JAX_PKG_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PKG_NAME] = module
    spec.loader.exec_module(module)
    return module


vjax = _load_vendored_pkg()

# x64 for the 1e-6 parity assertions.
vjax.set_jax_precision(True)

import jax.numpy as jnp  # noqa: E402  (after precision is set)

TOL = 1e-6


# ---------------------------------------------------------------------------
# Tiny hand-built fixture (5 trials).
# ---------------------------------------------------------------------------
def _make_fixture(seed=0):
    rng = np.random.default_rng(seed)
    n_trials = 5
    max_d = 6

    eta, kappa, sigma, a, b, x0 = 0.3, 1.0, 1.0, 1.5, 0.25, 0.0

    r1_data = rng.integers(1, 6, n_trials).astype(np.float64)
    r2_data = rng.integers(1, 6, n_trials).astype(np.float64)
    flag_data = rng.integers(0, 2, n_trials).astype(np.int32)
    d_data = rng.integers(2, max_d + 1, n_trials).astype(np.int32)

    rt_data = rng.uniform(0.8, 2.0, n_trials)
    choice_data = rng.choice(np.array([-1, 1]), size=n_trials).astype(np.float64)

    sacc_array_data = np.zeros((n_trials, max_d), dtype=np.float64)
    for i in range(n_trials):
        # Fixation onsets occur before the response; stage 0 anchored at 0.
        onsets = np.sort(rng.uniform(0.0, rt_data[i], d_data[i] - 1))
        sacc_array_data[i, 0] = 0.0
        sacc_array_data[i, 1 : d_data[i]] = onsets

    return dict(
        n_trials=n_trials,
        max_d=max_d,
        eta=eta,
        kappa=kappa,
        sigma=sigma,
        a=a,
        b=b,
        x0=x0,
        rt=jnp.asarray(rt_data),
        choice=jnp.asarray(choice_data),
        r1=jnp.asarray(r1_data),
        r2=jnp.asarray(r2_data),
        flag=jnp.asarray(flag_data),
        sacc=jnp.asarray(sacc_array_data),
        d=jnp.asarray(d_data),
    )


def _batched(fx):
    return vjax.compute_addm_loglikelihoods(
        fx["rt"], fx["choice"], fx["eta"], fx["kappa"], fx["sigma"],
        fx["a"], fx["b"], fx["x0"], fx["r1"], fx["r2"], fx["flag"],
        fx["sacc"], fx["d"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_imports():
    """Every symbol the jax/__init__ public surface promises resolves."""
    for name in (
        "compute_addm_loglikelihoods",
        "compute_addm_loglikelihoods_from_mu",
        "make_addm_nll_function",
        "compute_addm_logfptd",
        "_build_addm_mu_array_data",
        "set_jax_precision",
        "get_jax_dtype",
    ):
        assert hasattr(vjax, name), f"missing vendored symbol: {name}"


def test_batched_runs_on_tiny_fixture():
    """Batched kernel returns shape (n,), finite, dtype == get_jax_dtype()."""
    fx = _make_fixture()
    ll = _batched(fx)
    assert ll.shape == (fx["n_trials"],)
    assert bool(jnp.all(jnp.isfinite(ll))), f"non-finite log-likes: {ll}"
    assert ll.dtype == vjax.get_jax_dtype()


def test_batched_matches_per_trial():
    """Batched output equals the per-trial kernel within tolerance."""
    fx = _make_fixture()
    ll_batched = np.asarray(_batched(fx))

    ll_per_trial = np.array(
        [
            float(
                vjax.compute_addm_logfptd(
                    fx["rt"][i], fx["choice"][i], fx["eta"], fx["kappa"],
                    fx["sigma"], fx["a"], fx["b"], fx["x0"],
                    fx["r1"][i], fx["r2"][i], fx["flag"][i],
                    fx["sacc"][i], fx["d"][i],
                )
            )
            for i in range(fx["n_trials"])
        ]
    )
    assert np.allclose(ll_batched, ll_per_trial, atol=TOL, rtol=0.0), (
        f"batched vs per-trial mismatch:\n{ll_batched}\n{ll_per_trial}"
    )


def test_from_mu_matches_default():
    """from_mu with the default drift array equals the default kernel."""
    fx = _make_fixture()
    mu_array = vjax._build_addm_mu_array_data(
        fx["eta"], fx["kappa"], fx["r1"], fx["r2"], fx["flag"], fx["d"],
        fx["max_d"],
    )
    ll_from_mu = np.asarray(
        vjax.compute_addm_loglikelihoods_from_mu(
            fx["rt"], fx["choice"], mu_array, fx["sacc"], fx["d"],
            fx["sigma"], fx["a"], fx["b"], fx["x0"],
        )
    )
    ll_default = np.asarray(_batched(fx))
    assert np.allclose(ll_from_mu, ll_default, atol=TOL, rtol=0.0), (
        f"from_mu vs default mismatch:\n{ll_from_mu}\n{ll_default}"
    )


if __name__ == "__main__":
    for fn in (
        test_imports,
        test_batched_runs_on_tiny_fixture,
        test_batched_matches_per_trial,
        test_from_mu_matches_default,
    ):
        fn()
        print(f"PASSED: {fn.__name__}")
    print("\nAll Commit 1 vendored-JAX checks passed.")

"""Commit 2 — verify the aDDM likelihood builder + attention-process registry.

Asserts that ``make_addm_logp_func`` maps HSSM's column-ordering contract onto
the vendored kernel correctly, that the PyTensor ``Op`` from ``make_addm_logp_op``
matches the raw JAX ``logp``, that gradients flow, and that a custom attention
process routes through the ``from_mu`` path.

Environment
-----------
Unlike the Commit 1 test, this file imports ``hssm.addm.likelihoods.builder``,
which (per the faithful-imports design) pulls in ``hssm.distribution_utils``
(pytensor/pymc) and therefore the full ``hssm`` package. Run it in a full-stack
env with this repo on the path, e.g.::

    PYTHONPATH=src /users/azhan378/.conda/envs/hssm_cavanagh_oscar/bin/python \
        src/hssm/addm/commit_tests/test_commit2_builder.py

(or ``pytest`` under the same interpreter). Test 3 additionally needs pytensor;
it is skipped with a message if pytensor is unavailable.
"""

import numpy as np

from hssm.addm.attention_process import standard_alternating
from hssm.addm.likelihoods.builder import make_addm_logp_func, make_addm_logp_op
from hssm.addm.likelihoods.jax import (
    compute_addm_loglikelihoods,
    get_jax_dtype,
    set_jax_precision,
)

set_jax_precision(True)  # x64 for the 1e-6 parity assertions

import jax  # noqa: E402  (after precision is set)
import jax.numpy as jnp  # noqa: E402

TOL = 1e-6

LIST_PARAMS = ["eta", "kappa", "a", "b", "x0", "t"]
EXTRA_FIELDS = ["r1", "r2", "flag", "sacc_array", "d", "sigma"]


# ---------------------------------------------------------------------------
# Fixture (10 trials) — args in the model-build order the distribution uses:
#   data, *list_params, *extra_fields
#   = data, eta, kappa, a, b, x0, t, r1, r2, flag, sacc_array, d, sigma
# t defaults to 0.0 (the identity path), so the kernel-parity tests are unaffected.
# ---------------------------------------------------------------------------
def _make_fixture(seed=0):
    rng = np.random.default_rng(seed)
    n_trials = 10
    max_d = 6

    eta, kappa, a, b, x0, t, sigma = 0.3, 1.0, 1.5, 0.25, 0.0, 0.0, 1.0

    r1 = rng.integers(1, 6, n_trials).astype(np.float64)
    r2 = rng.integers(1, 6, n_trials).astype(np.float64)
    flag = rng.integers(0, 2, n_trials).astype(np.int32)
    d = rng.integers(2, max_d + 1, n_trials).astype(np.int32)

    rt = rng.uniform(0.8, 2.0, n_trials)
    choice = rng.choice(np.array([-1, 1]), size=n_trials).astype(np.float64)

    sacc = np.zeros((n_trials, max_d), dtype=np.float64)
    for i in range(n_trials):
        onsets = np.sort(rng.uniform(0.0, rt[i], d[i] - 1))
        sacc[i, 0] = 0.0
        sacc[i, 1 : d[i]] = onsets

    data = np.column_stack([rt, choice])

    # Positional args after `data`, matching list_params + extra_fields order.
    args = (
        jnp.asarray(eta),
        jnp.asarray(kappa),
        jnp.asarray(a),
        jnp.asarray(b),
        jnp.asarray(x0),
        jnp.asarray(t),
        jnp.asarray(r1),
        jnp.asarray(r2),
        jnp.asarray(flag),
        jnp.asarray(sacc),
        jnp.asarray(d),
        jnp.asarray(sigma),
    )
    return dict(
        n_trials=n_trials,
        max_d=max_d,
        data=jnp.asarray(data),
        args=args,
        # raw scalars/arrays for direct-kernel comparison
        eta=eta,
        kappa=kappa,
        a=a,
        b=b,
        x0=x0,
        t=t,
        sigma=sigma,
        rt=jnp.asarray(rt),
        choice=jnp.asarray(choice),
        r1=jnp.asarray(r1),
        r2=jnp.asarray(r2),
        flag=jnp.asarray(flag),
        sacc=jnp.asarray(sacc),
        d=jnp.asarray(d),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_logp_func_output_shape():
    """Default logp returns (n_trials,) with the active jax dtype."""
    fx = _make_fixture()
    logp = make_addm_logp_func()
    out = logp(fx["data"], *fx["args"])
    assert out.shape == (fx["n_trials"],)
    assert out.dtype == get_jax_dtype()


def test_logp_matches_kernel():
    """logp equals a direct compute_addm_loglikelihoods call (reordering OK)."""
    fx = _make_fixture()
    logp = make_addm_logp_func()
    out = np.asarray(logp(fx["data"], *fx["args"]))

    direct = np.asarray(
        compute_addm_loglikelihoods(
            fx["rt"],
            fx["choice"],
            fx["eta"],
            fx["kappa"],
            fx["sigma"],
            fx["a"],
            fx["b"],
            fx["x0"],
            fx["r1"],
            fx["r2"],
            fx["flag"],
            fx["sacc"],
            fx["d"],
        )
    )
    assert np.allclose(out, direct, atol=TOL, rtol=0.0), (
        f"logp vs direct-kernel mismatch:\n{out}\n{direct}"
    )


def test_logp_op_matches_func():
    """The PyTensor Op evaluates to the same values as the raw JAX logp."""
    try:
        import pytensor  # noqa: F401
    except ImportError:
        print("SKIP test_logp_op_matches_func: pytensor not available")
        return

    fx = _make_fixture()
    logp = make_addm_logp_func()
    ref = np.asarray(logp(fx["data"], *fx["args"]))

    op = make_addm_logp_op(
        attention_process="standard_alternating",
        list_params=LIST_PARAMS,
        extra_fields=EXTRA_FIELDS,
    )
    out = np.asarray(
        op(np.asarray(fx["data"]), *[np.asarray(a) for a in fx["args"]]).eval()
    )
    assert out.shape == ref.shape
    assert np.allclose(out, ref, atol=TOL, rtol=0.0), (
        f"Op vs func mismatch:\n{out}\n{ref}"
    )


def test_gradients_finite():
    """jax.grad of summed logp is finite w.r.t. eta, kappa, a, b, x0, t, sigma."""
    fx = _make_fixture()
    logp = make_addm_logp_func()

    def f(*all_args):
        return logp(*all_args).sum()

    # positions in (data, eta, kappa, a, b, x0, t, r1, r2, flag, sacc, d, sigma)
    #                 0    1    2     3  4  5   6  7   8    9    10   11  12
    grad_fn = jax.grad(f, argnums=(1, 2, 3, 4, 5, 6, 12))
    grads = grad_fn(fx["data"], *fx["args"])
    for name, g in zip(["eta", "kappa", "a", "b", "x0", "t", "sigma"], grads):
        assert bool(jnp.isfinite(g)), f"non-finite gradient for {name}: {g}"


def test_custom_attention_process():
    """A custom process routes through from_mu: c=1 matches default, c=2 differs."""
    fx = _make_fixture()

    def scaled(c):
        def proc(eta, kappa, r1, r2, flag, d, max_d):
            return c * standard_alternating(eta, kappa, r1, r2, flag, d, max_d)

        return proc

    default = np.asarray(make_addm_logp_func()(fx["data"], *fx["args"]))
    identity = np.asarray(make_addm_logp_func(scaled(1.0))(fx["data"], *fx["args"]))
    doubled = np.asarray(make_addm_logp_func(scaled(2.0))(fx["data"], *fx["args"]))

    # Identity custom process takes the from_mu branch but yields the same drift,
    # so it must agree with the default (kernel-internal) path.
    assert np.allclose(identity, default, atol=TOL, rtol=0.0), (
        "from_mu path with identity drift should match the default path"
    )
    # Doubling the drift must change the likelihood — confirms the custom branch
    # is actually exercised.
    assert not np.allclose(doubled, default, atol=TOL, rtol=0.0), (
        "doubled-drift custom process should differ from the default"
    )


if __name__ == "__main__":
    for fn in (
        test_logp_func_output_shape,
        test_logp_matches_kernel,
        test_logp_op_matches_func,
        test_gradients_finite,
        test_custom_attention_process,
    ):
        fn()
        print(f"PASSED: {fn.__name__}")
    print("\nAll Commit 2 builder checks passed.")

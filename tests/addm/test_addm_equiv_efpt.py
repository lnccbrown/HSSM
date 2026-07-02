"""CB oracle: HSSM's vendored JAX FPT kernel must match the efpt NumPy reference.

The ``tests/addm/oracle/`` package is an independent, verbatim vendoring of efpt's
pure-NumPy backend (at efpt's original ``DEFAULT_TRUNC_NUM=100``). This module is
the correctness anchor every later aDDM commit leans on: it pins the vendored JAX
single-stage first-passage-time functions to the NumPy reference to ~1e-6 under
x64, mirroring efpt's own cross-backend equivalence tests.
"""

import sys
from pathlib import Path

import jax

# x64 is required for the tight tolerance (the FPT series is precision-sensitive).
jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

# HSSM's vendored JAX single-stage kernel.
from hssm.addm.likelihoods.jax import single_stage as jax_ss  # noqa: E402

# The independent efpt NumPy oracle (self-contained package alongside this file).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle  # noqa: E402

RTOL = 1e-6
ATOL = 1e-9
TRUNC = 100  # match both sides explicitly (independent of the kernel's batch default)

# Physically-valid single-stage cases spanning drift sign, sigma, collapse, bias:
# (t, mu, sigma, a1, b1, a2, b2, x0, bdy)  -- upper bound a1+b1*t, lower a2+b2*t.
_SINGLE_CASES = [
    (0.5, 1.0, 2.0, 1.5, -0.3, -1.5, 0.3, 0.0, 1),
    (0.5, 1.0, 2.0, 1.5, -0.3, -1.5, 0.3, 0.0, -1),
    (0.3, -0.5, 1.0, 1.2, -0.2, -1.2, 0.2, 0.1, 1),
    (0.8, 0.0, 1.5, 2.0, -0.5, -2.0, 0.5, -0.2, -1),
    (1.2, 2.0, 0.8, 1.0, -0.1, -1.0, 0.1, 0.3, 1),
    (0.25, -1.5, 1.0, 1.5, 0.0, -1.5, 0.0, 0.0, -1),  # constant (non-collapsing) bounds
]
# (x, mu, sigma, a1, b1, a2, b2, T, x0)  -- survivor/q function cases.
_Q_CASES = [
    (0.2, 1.0, 2.0, 1.5, -0.3, -1.5, 0.3, 0.5, 0.0),
    (-0.4, -0.5, 1.0, 1.2, -0.2, -1.2, 0.2, 0.8, 0.1),
    (0.0, 0.0, 1.5, 2.0, -0.5, -2.0, 0.5, 1.0, -0.2),
]


@pytest.mark.parametrize("args", _SINGLE_CASES)
def test_fptd_single_matches_oracle(args):
    got = float(jax_ss.fptd_single(*args, trunc_num=TRUNC))
    ref = float(oracle.fptd_single(*args, trunc_num=TRUNC))
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("args", _SINGLE_CASES)
def test_log_fptd_single_matches_oracle(args):
    got = float(jax_ss.log_fptd_single(*args, trunc_num=TRUNC))
    ref = float(oracle.log_fptd_single(*args, trunc_num=TRUNC))
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("args", _Q_CASES)
def test_q_single_matches_oracle(args):
    got = float(jax_ss.q_single(*args, trunc_num=TRUNC))
    ref = float(oracle.q_single(*args, trunc_num=TRUNC))
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


def test_fptd_single_array_matches_oracle():
    """Vectorized over a time grid — the shape the batch kernel exercises."""
    t = np.linspace(0.15, 3.0, 40)
    common = dict(trunc_num=TRUNC)
    mu, sigma, a1, b1, a2, b2, x0, bdy = 0.8, 1.2, 1.5, -0.25, -1.5, 0.25, 0.1, 1
    got = np.asarray(jax_ss.fptd_single(t, mu, sigma, a1, b1, a2, b2, x0, bdy, **common))
    ref = np.asarray(oracle.fptd_single(t, mu, sigma, a1, b1, a2, b2, x0, bdy, **common))
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


def test_oracle_defaults_are_efpt_original():
    """Guard: the oracle must stay at efpt's original truncation, not HSSM's fork."""
    from oracle._defaults import DEFAULT_TRUNC_NUM

    assert DEFAULT_TRUNC_NUM == 100

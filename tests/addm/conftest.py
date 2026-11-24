"""Shared fixtures for the aDDM test suite."""

import jax
import pytensor
import pytest

from hssm.addm.likelihoods.jax import set_jax_precision


@pytest.fixture(autouse=True)
def _pin_float64_precision():
    """Run every aDDM test under float64 and restore the previous state after.

    The vendored FPT kernel's accuracy contract is float64. Module-level
    ``set_jax_precision(True)`` calls are order-fragile: pytest imports every
    test module at collection before any test body runs, so a later module's
    ``hssm.set_floatX("float32")`` (executed at import) flips the global JAX
    x64 flag back off and desyncs the kernel's cached dtype — the aDDM bodies
    then compute float32 against a float64 contract. Pinning per-test fixes
    that inbound leak; restoring afterwards fixes the outbound one (an aDDM
    test switching the suite to float64 would break later float32-dependent
    LAN/ONNX tests).
    """
    prev_floatx = pytensor.config.floatX
    prev_x64 = jax.config.jax_enable_x64
    pytensor.config.floatX = "float64"
    set_jax_precision(True)  # JAX x64 + the kernel's dtype cache + quadrature
    yield
    pytensor.config.floatX = prev_floatx
    set_jax_precision(prev_x64)

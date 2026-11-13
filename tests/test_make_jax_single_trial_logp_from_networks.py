from pathlib import Path

import jax.numpy as jnp
import numpy as np
import onnx

from hssm.distribution_utils.jax import make_jax_single_trial_logp_from_network_forward
from hssm.distribution_utils.onnx import make_jax_logp_funcs_from_onnx
from hssm.distribution_utils.onnx_utils.onnx2jax import make_jax_func


def test_lan_style_params_only_false_concatenates_params_and_data():
    """Test that LAN-style logp function correctly concatenates parameters and data.

    This test verifies that when params_only=False, the wrapper function:
    1. Properly handles parameters of different shapes (scalar, 1D, 2D arrays)
    2. Squeezes parameters into a flat vector
    3. Correctly concatenates the parameter vector with the data vector
    4. Returns output with expected shape and values

    The test uses an identity forward function to inspect the input construction.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Identity forward to inspect what wrapper builds
    def forward(x):
        return x

    logp = make_jax_single_trial_logp_from_network_forward(
        forward,
        params_only=False,
    )

    # Single-trial data vector (length D)
    data = jnp.array([0.1, 0.2])  # shape (2,)

    # Params; mix in singleton dims to exercise squeezing
    p1 = jnp.array([1.0])
    p2 = jnp.array([2.0])
    p3 = jnp.array([3.0])
    p4 = jnp.array([4.0])

    out = logp(data, p1, p2, p3, p4)

    expected_params = jnp.array([1.0, 2.0, 3.0, 4.0])  # squeezed flat
    expected = jnp.concatenate((expected_params, data))  # (6,)

    assert out.ndim == 1
    assert out.shape == expected.shape
    assert np.allclose(np.asarray(out), np.asarray(expected))


def test_opn_cpn_style_params_only_true_stacks_params_only():
    """Test that OPN/CPN-style logp function correctly stacks parameters only.

    This test verifies that when params_only=True, the wrapper function:
    1. Properly handles parameters of different types (JAX arrays and Python scalars)
    2. Correctly stacks parameters into a single vector
    3. Returns output with expected shape and values

    The test uses an identity forward function to inspect the input construction.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    def forward(x):
        return x

    logp = make_jax_single_trial_logp_from_network_forward(forward, params_only=True)

    # Pass scalar-like params (0-D arrays or Python floats) so jnp.array(...) is well-defined
    p1 = jnp.array([1.0])  # 0-d
    p2 = jnp.array([2.0])
    p3 = jnp.array([3.0])  # 0-d
    p4 = jnp.array([4.0])

    out = logp(p1, p2, p3, p4)

    expected = jnp.array([1.0, 2.0, 3.0, 4.0])  # (4,)
    assert out.ndim == 1
    assert out.shape == expected.shape
    assert np.allclose(np.asarray(out), np.asarray(expected))


# --------------------------------------------------------------------------
# CPN input layout: [theta..., choice] with the choice LAST (#1324)
# --------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
THETA = (0.5, 1.5, 0.5, 0.3)  # v, a, z, t


def test_cpn_single_trial_logp_receives_choice_last(cpn_onnxruntime):
    """The single-trial CPN logp is evaluated on ``[v, a, z, t, choice]``.

    Two inputs that differ only in the choice give different outputs, and each
    equals a direct onnxruntime evaluation of the same ``[theta..., choice]``
    row, which pins both the layout and the position of the choice.
    """
    forward = make_jax_func(onnx.load(FIXTURES / "ddm_cpn.onnx"))
    logp = make_jax_single_trial_logp_from_network_forward(forward, params_only=False)
    params = [jnp.array(x) for x in THETA]

    out = {c: float(logp(jnp.array([c]), *params)) for c in (-1.0, 1.0)}
    assert out[-1.0] != out[1.0]
    assert out[-1.0] <= 0.0 and out[1.0] <= 0.0  # log-probabilities

    expected = cpn_onnxruntime([[*THETA, -1.0], [*THETA, 1.0]])
    np.testing.assert_allclose([out[-1.0], out[1.0]], expected, rtol=1e-5)


def test_cpn_vmapped_logp_maps_over_the_response_column(cpn_onnxruntime):
    """``make_jax_logp_funcs_from_onnx(params_only=False)`` feeds each row's choice.

    The data argument is the ``(n, 1)`` response column; each row is
    concatenated after the parameters, so rows that differ only in the choice
    differ in the output and match onnxruntime on ``[theta..., choice]``.
    """
    logp, _, _ = make_jax_logp_funcs_from_onnx(
        onnx.load(FIXTURES / "ddm_cpn.onnx"),
        params_is_reg=[False] * 4,
        params_only=False,
    )
    choices = jnp.array([[1.0], [-1.0], [1.0]])
    out = np.asarray(logp(choices, *[jnp.array(x) for x in THETA]))

    assert out.shape == (3,)
    assert out[0] != out[1]
    assert out[0] == out[2]
    expected = cpn_onnxruntime([[*THETA, 1.0], [*THETA, -1.0], [*THETA, 1.0]])
    np.testing.assert_allclose(out, expected, rtol=1e-5)

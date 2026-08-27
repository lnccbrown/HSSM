"""Numerical handshake tests against the pinned JEAM implementation."""

import numpy as np
import pytest

pytest.importorskip("jeam")

import jax
import jax.numpy as jnp
import pytensor
import pytensor.tensor as pt
from jeam.likelihoods.jax_fixed import fixed_cdm_logpdf
from jeam.Models.Circular import CircularDiffusionModel
from pytensor.compile.mode import Mode

from hssm.distribution_utils.dist import (
    LOGP_LB,
    make_distribution,
    make_likelihood_callable,
)
from hssm.integrations.jeam import (
    logp_circular_diffusion,
    logp_circular_diffusion_jax,
)

PARAMETERS = ["v_x", "v_y", "a", "t"]
BOUNDS = {
    "v_x": (-3.0, 3.0),
    "v_y": (-3.0, 3.0),
    "a": (0.1, 3.0),
    "t": (0.0, 2.0),
}
# Allow small linker round-trip differences while keeping direct adapter comparisons
# at exact float64 precision.
COMPILED_RTOL = 5e-7
COMPILED_ATOL = 5e-7
JAX_RTOL = 2e-5
JAX_ATOL = 2e-6
GRADIENT_RTOL = 2e-4
GRADIENT_ATOL = 1e-5
PYTHON_MODE = Mode(linker="py", optimizer="fast_compile")


@pytest.fixture(scope="module", autouse=True)
def _pin_fixed_cdm_x64():
    """Run fixed-CDM JAX checks in their supported precision mode."""
    original_floatx = pytensor.config.floatX
    original_jax_x64 = jax.config.jax_enable_x64
    pytensor.config.floatX = "float64"
    jax.config.update("jax_enable_x64", True)
    yield
    pytensor.config.floatX = original_floatx
    jax.config.update("jax_enable_x64", original_jax_x64)


def _make_analytical_distribution():
    """Build the ordinary HSSM distribution around JEAM's batched JAX bridge."""
    likelihood = make_likelihood_callable(
        loglik=logp_circular_diffusion_jax,
        loglik_kind="analytical",
        backend="jax",
    )
    return make_distribution(
        rv="circular_diffusion",
        loglik=likelihood,
        list_params=PARAMETERS.copy(),
        bounds=BOUNDS,
    )


def test_jax_adapter_preserves_the_batched_data_contract():
    """The JAX bridge should map batched ``[rt, response]`` data without reordering."""
    data = jnp.asarray([[0.37, -0.65], [0.82, 1.20]])
    parameters = (0.45, -0.30, 1.20, 0.08)

    observed = logp_circular_diffusion_jax(data, *parameters)
    expected = fixed_cdm_logpdf(data[:, 0], data[:, 1], *parameters)

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
    assert observed.shape == (2,)


def test_jax_adapter_preserves_jeam_x64_guard():
    """The HSSM wrapper must not bypass or mutate JEAM's precision contract."""
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(
            RuntimeError,
            match=r"requires JAX x64 execution.*jax_enable_x64.*disabled",
        ):
            logp_circular_diffusion_jax(
                jnp.asarray([[0.37, -0.65]]),
                0.45,
                -0.30,
                1.20,
                0.08,
            )
        assert jax.config.jax_enable_x64 is False
    finally:
        jax.config.update("jax_enable_x64", True)


def test_jax_adapter_broadcasts_scalar_and_trialwise_parameters_in_row_order():
    """The batched bridge should retain scalar sharing and trial association."""
    data = jnp.asarray([[0.21, -1.7], [0.46, 0.2], [0.91, 2.2]])
    v_x = jnp.asarray([0.55, -0.20, 0.35])
    v_y = -0.25
    threshold = jnp.asarray([1.00, 1.25, 1.10])
    ndt = jnp.asarray([0.04, 0.11, 0.17])

    observed = logp_circular_diffusion_jax(data, v_x, v_y, threshold, ndt)
    expected = fixed_cdm_logpdf(
        data[:, 0],
        data[:, 1],
        v_x,
        v_y,
        threshold,
        ndt,
    )

    np.testing.assert_allclose(observed, expected, rtol=5e-7, atol=5e-7)


def test_jax_adapter_parameter_gradient_matches_the_direct_jeam_kernel():
    """The HSSM argument bridge must leave JEAM's four-parameter gradient intact."""
    data = jnp.asarray([[0.53, 0.7], [0.84, -1.1]])
    parameters = jnp.asarray([0.45, -0.30, 1.20, 0.08])

    observed = jax.grad(
        lambda values: logp_circular_diffusion_jax(data, *values).sum()
    )(parameters)
    expected = jax.grad(
        lambda values: fixed_cdm_logpdf(data[:, 0], data[:, 1], *values).sum()
    )(parameters)

    np.testing.assert_allclose(observed, expected, rtol=5e-7, atol=5e-7)
    assert np.all(np.isfinite(observed))


@pytest.fixture(scope="module")
def circular_distribution():
    """Build the real HSSM blackbox distribution around the JEAM adapter."""
    likelihood = make_likelihood_callable(
        loglik=logp_circular_diffusion,
        loglik_kind="blackbox",
        backend="pytensor",
    )
    return make_distribution(
        rv="circular_diffusion",
        loglik=likelihood,
        list_params=PARAMETERS.copy(),
        bounds=BOUNDS,
    )


def _direct_trialwise_jeam(
    data: np.ndarray,
    v_x: np.ndarray,
    v_y: np.ndarray,
    a: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Evaluate JEAM row by row when every parameter may be trial-wise."""
    model = CircularDiffusionModel(threshold_dynamic="fixed")
    return np.array(
        [
            model.joint_lpdf(
                rt=data[index : index + 1, 0],
                theta=data[index : index + 1, 1],
                drift_vec=np.array([[v_x[index], v_y[index]]]),
                ndt=np.array([t[index]]),
                threshold=float(a[index]),
                decay=0.0,
                threshold_function=None,
                dt_threshold_function=None,
                s_v=0.0,
                s_t=0.0,
                sigma=1.0,
            )[0]
            for index in range(data.shape[0])
        ],
        dtype=np.float64,
    )


def test_adapter_matches_direct_jeam_at_asymmetric_scalar_parameters():
    """The adapter should preserve JEAM values across RT and angle locations."""
    data = np.array([[0.18, -2.1], [0.37, 0.35], [0.82, 2.4]], dtype=np.float64)
    v_x = 0.45
    v_y = -0.30
    threshold = 1.20
    ndt = 0.08
    model = CircularDiffusionModel(threshold_dynamic="fixed")

    expected = model.joint_lpdf(
        rt=data[:, 0],
        theta=data[:, 1],
        drift_vec=np.column_stack(
            (np.full(data.shape[0], v_x), np.full(data.shape[0], v_y))
        ),
        ndt=np.full(data.shape[0], ndt),
        threshold=threshold,
        decay=0.0,
        threshold_function=None,
        dt_threshold_function=None,
        s_v=0.0,
        s_t=0.0,
        sigma=1.0,
    )
    observed = logp_circular_diffusion(data, v_x, v_y, threshold, ndt)

    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
    assert observed.shape == (3,)
    assert observed.dtype == np.float64


def test_adapter_matches_direct_jeam_with_trialwise_parameters():
    """All four HSSM parameters should support one value per observation."""
    data = np.array([[0.21, -1.7], [0.46, 0.2], [0.91, 2.2]], dtype=np.float64)
    v_x = np.array([0.55, -0.20, 0.35])
    v_y = np.array([-0.25, 0.45, -0.15])
    threshold = np.array([1.00, 1.25, 1.00])
    ndt = np.array([0.04, 0.11, 0.17])

    expected = _direct_trialwise_jeam(data, v_x=v_x, v_y=v_y, a=threshold, t=ndt)
    observed = logp_circular_diffusion(data, v_x=v_x, v_y=v_y, a=threshold, t=ndt)

    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_adapter_preserves_jeam_impossible_rt_support_value():
    """An RT at or below NDT should retain JEAM's pointwise support result."""
    data = np.array([[0.05, 0.2], [0.10, -0.4]], dtype=np.float64)
    v_x = np.array([0.45, -0.10])
    v_y = np.array([-0.30, 0.25])
    threshold = np.array([1.20, 1.20])
    ndt = np.array([0.10, 0.10])

    expected = _direct_trialwise_jeam(data, v_x=v_x, v_y=v_y, a=threshold, t=ndt)
    observed = logp_circular_diffusion(data, v_x=v_x, v_y=v_y, a=threshold, t=ndt)

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(observed, -np.inf)


def test_compiled_hssm_distribution_matches_adapter_and_direct_jeam(
    circular_distribution,
):
    """The compiled HSSM blackbox path should preserve pointwise JEAM values."""
    data = np.array([[0.18, -2.1], [0.37, 0.35], [0.82, 2.4]])
    v_x = np.full(3, 0.45)
    v_y = np.full(3, -0.30)
    threshold = np.full(3, 1.20)
    ndt = np.full(3, 0.08)

    direct = _direct_trialwise_jeam(data, v_x=v_x, v_y=v_y, a=threshold, t=ndt)
    adapted = logp_circular_diffusion(data, 0.45, -0.30, 1.20, 0.08)
    compiled = np.asarray(
        circular_distribution.logp(data, 0.45, -0.30, 1.20, 0.08).eval()
    )

    np.testing.assert_allclose(adapted, direct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(compiled, direct, rtol=COMPILED_RTOL, atol=COMPILED_ATOL)


def test_compiled_hssm_distribution_accepts_a_trialwise_drift(
    circular_distribution,
):
    """A regression-shaped drift should remain pointwise through compilation."""
    data = np.array([[0.21, -1.7], [0.46, 0.2], [0.91, 2.2]])
    v_x = np.array([0.55, -0.20, 0.35])
    v_y = np.full(3, -0.25)
    threshold = np.full(3, 1.15)
    ndt = np.full(3, 0.07)

    direct = _direct_trialwise_jeam(data, v_x=v_x, v_y=v_y, a=threshold, t=ndt)
    adapted = logp_circular_diffusion(data, v_x, -0.25, 1.15, 0.07)
    compiled = np.asarray(
        circular_distribution.logp(data, v_x, -0.25, 1.15, 0.07).eval()
    )

    np.testing.assert_allclose(adapted, direct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(compiled, direct, rtol=COMPILED_RTOL, atol=COMPILED_ATOL)


def test_hssm_distribution_applies_standard_ndt_support_mask(circular_distribution):
    """HSSM should replace JEAM's strict support value with its standard mask."""
    data = np.array([[0.18, -0.4], [0.37, 0.6], [0.82, 1.2]])
    ndt = np.array([0.08, 0.37, 0.90])

    adapted = logp_circular_diffusion(data, 0.45, -0.30, 1.20, ndt)
    compiled = np.asarray(
        circular_distribution.logp(data, 0.45, -0.30, 1.20, ndt).eval()
    )
    lower_bound = float(np.asarray(LOGP_LB))

    assert compiled[0] == pytest.approx(
        adapted[0], rel=COMPILED_RTOL, abs=COMPILED_ATOL
    )
    np.testing.assert_array_equal(compiled[1:], lower_bound)
    np.testing.assert_array_equal(adapted[1:], -np.inf)


def test_hssm_distribution_applies_parameter_bounds_pointwise(
    circular_distribution,
):
    """Only trials whose parameter values violate HSSM bounds should be masked."""
    data = np.array([[0.18, -0.4], [0.37, 0.6], [0.82, 1.2]])
    v_x = np.array([0.45, 3.01, -0.25])

    adapted = logp_circular_diffusion(data, v_x, -0.30, 1.20, 0.08)
    compiled = np.asarray(
        circular_distribution.logp(data, v_x, -0.30, 1.20, 0.08).eval()
    )
    lower_bound = float(np.asarray(LOGP_LB))

    np.testing.assert_allclose(
        compiled[[0, 2]],
        adapted[[0, 2]],
        rtol=COMPILED_RTOL,
        atol=COMPILED_ATOL,
    )
    assert compiled[1] == lower_bound


def test_differentiable_distribution_matches_jeam_through_both_linkers():
    """PyTensor and JAX lowering should preserve direct JAX and NumPy values."""
    distribution = _make_analytical_distribution()
    dtype = pytensor.config.floatX
    data = np.asarray(
        [[0.081, -2.1], [0.090, 0.35], [0.370, 2.4]],
        dtype=dtype,
    )
    parameters = np.asarray([0.45, -0.30, 1.20, 0.08], dtype=dtype)

    data_symbol = pt.matrix("data", dtype=dtype)
    parameter_symbols = [pt.scalar(name, dtype=dtype) for name in PARAMETERS]
    pointwise = distribution.logp(data_symbol, *parameter_symbols)
    inputs = [data_symbol, *parameter_symbols]
    values = [data, *parameters]

    compiled = np.asarray(
        pytensor.function(inputs, pointwise, mode=PYTHON_MODE)(*values)
    )
    lowered = np.asarray(pytensor.function(inputs, pointwise, mode="JAX")(*values))
    direct_jax = np.asarray(
        fixed_cdm_logpdf(
            data[:, 0],
            data[:, 1],
            *parameters,
        )
    )
    direct_numpy = logp_circular_diffusion(data, *parameters)

    np.testing.assert_allclose(compiled, direct_jax, rtol=JAX_RTOL, atol=JAX_ATOL)
    np.testing.assert_allclose(lowered, direct_jax, rtol=JAX_RTOL, atol=JAX_ATOL)
    np.testing.assert_allclose(direct_jax, direct_numpy, rtol=JAX_RTOL, atol=JAX_ATOL)
    assert compiled.shape == lowered.shape == direct_jax.shape == (3,)


def test_differentiable_distribution_symbolic_gradients_match_direct_jax():
    """HSSM's VJP Op and JAX lowering should retain all four scalar gradients."""
    distribution = _make_analytical_distribution()
    dtype = pytensor.config.floatX
    data = np.asarray(
        [[0.23, -1.4], [0.51, 0.25], [0.94, 2.1]],
        dtype=dtype,
    )
    parameters = np.asarray([0.45, -0.30, 1.20, 0.08], dtype=dtype)

    data_symbol = pt.matrix("data", dtype=dtype)
    parameter_symbols = [pt.scalar(name, dtype=dtype) for name in PARAMETERS]
    objective = distribution.logp(data_symbol, *parameter_symbols).sum()
    gradients = pt.stack(pytensor.grad(objective, wrt=parameter_symbols))
    inputs = [data_symbol, *parameter_symbols]
    values = [data, *parameters]

    compiled = np.asarray(
        pytensor.function(inputs, gradients, mode=PYTHON_MODE)(*values)
    )
    lowered = np.asarray(pytensor.function(inputs, gradients, mode="JAX")(*values))
    expected = np.asarray(
        jax.grad(
            lambda current: fixed_cdm_logpdf(
                data[:, 0],
                data[:, 1],
                *current,
            ).sum()
        )(jnp.asarray(parameters))
    )

    np.testing.assert_allclose(
        compiled,
        expected,
        rtol=GRADIENT_RTOL,
        atol=GRADIENT_ATOL,
    )
    np.testing.assert_allclose(
        lowered,
        expected,
        rtol=GRADIENT_RTOL,
        atol=GRADIENT_ATOL,
    )
    assert np.all(np.isfinite(compiled))
    assert np.all(np.isfinite(lowered))


def test_differentiable_distribution_preserves_trialwise_values_and_gradients():
    """Regression-generated ``v_x`` and ``t`` vectors should remain row-aligned."""
    distribution = _make_analytical_distribution()
    dtype = pytensor.config.floatX
    data = np.asarray(
        [[0.24, -1.7], [0.49, 0.2], [0.91, 2.2]],
        dtype=dtype,
    )
    v_x = np.asarray([0.55, -0.20, 0.35], dtype=dtype)
    v_y = np.asarray(-0.25, dtype=dtype)
    threshold = np.asarray(1.15, dtype=dtype)
    ndt = np.asarray([0.04, 0.11, 0.17], dtype=dtype)

    data_symbol = pt.matrix("data", dtype=dtype)
    v_x_symbol = pt.vector("v_x", dtype=dtype)
    v_y_symbol = pt.scalar("v_y", dtype=dtype)
    threshold_symbol = pt.scalar("a", dtype=dtype)
    ndt_symbol = pt.vector("t", dtype=dtype)
    pointwise = distribution.logp(
        data_symbol,
        v_x_symbol,
        v_y_symbol,
        threshold_symbol,
        ndt_symbol,
    )
    trialwise_gradients = pytensor.grad(
        pointwise.sum(),
        wrt=[v_x_symbol, ndt_symbol],
    )
    inputs = [
        data_symbol,
        v_x_symbol,
        v_y_symbol,
        threshold_symbol,
        ndt_symbol,
    ]
    values = [data, v_x, v_y, threshold, ndt]

    compiled = np.asarray(
        pytensor.function(inputs, pointwise, mode=PYTHON_MODE)(*values)
    )
    compiled_gradients = tuple(
        np.asarray(value)
        for value in pytensor.function(
            inputs,
            trialwise_gradients,
            mode=PYTHON_MODE,
        )(*values)
    )
    lowered_gradients = tuple(
        np.asarray(value)
        for value in pytensor.function(inputs, trialwise_gradients, mode="JAX")(*values)
    )
    direct_jax = np.asarray(
        fixed_cdm_logpdf(
            data[:, 0],
            data[:, 1],
            v_x,
            v_y,
            threshold,
            ndt,
        )
    )
    direct_numpy = logp_circular_diffusion(
        data,
        v_x,
        v_y,
        threshold,
        ndt,
    )
    expected_gradients = tuple(
        np.asarray(value)
        for value in jax.grad(
            lambda current_v_x, current_t: fixed_cdm_logpdf(
                data[:, 0],
                data[:, 1],
                current_v_x,
                v_y,
                threshold,
                current_t,
            ).sum(),
            argnums=(0, 1),
        )(jnp.asarray(v_x), jnp.asarray(ndt))
    )

    np.testing.assert_allclose(compiled, direct_jax, rtol=JAX_RTOL, atol=JAX_ATOL)
    np.testing.assert_allclose(direct_jax, direct_numpy, rtol=JAX_RTOL, atol=JAX_ATOL)
    for observed, expected in zip(
        compiled_gradients,
        expected_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            observed,
            expected,
            rtol=GRADIENT_RTOL,
            atol=GRADIENT_ATOL,
        )
    for observed, expected in zip(
        lowered_gradients,
        expected_gradients,
        strict=True,
    ):
        np.testing.assert_allclose(
            observed,
            expected,
            rtol=GRADIENT_RTOL,
            atol=GRADIENT_ATOL,
        )

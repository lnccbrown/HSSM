from pathlib import Path
import pytest

import jax
import jax.numpy as jnp
import numpy as np

import hssm
from hssm.distribution_utils.onnx import make_jax_matrix_logp_funcs_from_onnx
from hssm.likelihoods import rldm_optimized
from hssm.likelihoods.rldm_optimized import make_rldm_logp_func, make_rldm_logp_op

hssm.set_floatX("float32")

DECIMAL = 2

# The tests pass the local angle.onnx fixture, so a Hub download means `model` was
# ignored (the Hub copy is byte-identical and would otherwise pass silently).
pytestmark = pytest.mark.usefixtures("no_hf_hub_download")


@pytest.fixture
def fixture_path():
    return Path(__file__).parents[2] / "fixtures"


def test_make_rldm_logp_func(fixture_path):
    """Test the JAX log-likelihood function for the RLDM model."""
    data = np.load(fixture_path / "rldm_data.npy", allow_pickle=True).item()["data"]
    participant_id = data["participant_id"].values
    trial = data["trial"].values
    feedback = data["feedback"].values

    subj = np.unique(participant_id).astype(np.int32)
    total_trials = trial.size

    rl_alpha = np.ones(total_trials) * 0.60
    scaler = np.ones(total_trials) * 3.2
    a = np.ones(total_trials) * 1.2
    z = np.ones(total_trials) * 0.1
    t = np.ones(total_trials) * 0.1
    theta = np.ones(total_trials) * 0.1

    logp = make_rldm_logp_func(
        n_participants=len(subj),
        n_trials=total_trials // len(subj),
        model=fixture_path / "angle.onnx",
    )
    jitted_logp = jax.jit(logp)
    jax_ll = jitted_logp(
        data[["rt", "response"]].values,
        rl_alpha,
        scaler,
        a,
        z,
        t,
        theta,
        feedback,
    )

    assert jax_ll.shape[0] == total_trials
    np.testing.assert_allclose(jax_ll.sum(), -6618.5674, rtol=1e-2)


def test_make_rldm_logp_op(fixture_path):
    """Test the JAX log-likelihood function for the RLDM model."""
    data = np.load(fixture_path / "rldm_data.npy", allow_pickle=True).item()["data"]
    participant_id = data["participant_id"].values
    trial = data["trial"].values
    feedback = data["feedback"].values

    subj = np.unique(participant_id).astype(np.int32)
    total_trials = trial.size

    rl_alpha = np.ones(total_trials) * 0.60
    scaler = np.ones(total_trials) * 3.2
    a = np.ones(total_trials) * 1.2
    z = np.ones(total_trials) * 0.1
    t = np.ones(total_trials) * 0.1
    theta = np.ones(total_trials) * 0.1

    logp_op = make_rldm_logp_op(
        n_participants=len(subj),
        n_trials=total_trials // len(subj),
        n_params=6,
        model=fixture_path / "angle.onnx",
    )

    jax_ll = logp_op(
        data.loc[:, ["rt", "response"]].values,
        rl_alpha,
        scaler,
        a,
        z,
        t,
        theta,
        feedback,
    )

    np.testing.assert_almost_equal(jax_ll.sum().eval(), -6618.5674, decimal=DECIMAL)


@pytest.mark.parametrize("lower_code", [0.0, -1.0], ids=["0/1", "-1/1"])
@pytest.mark.parametrize("jitted", [False, True], ids=["eager", "jit"])
def test_response_coding_reaches_the_lan_as_minus_one_one(
    fixture_path, lower_code, jitted
):
    """Both response codings give the likelihood with -1/1 choices for the LAN.

    The reference computes the Rescorla-Wagner drifts in NumPy and evaluates the
    angle LAN with the -1/1 choices it declares (#1399).
    """
    data = np.load(fixture_path / "rldm_data.npy", allow_pickle=True).item()["data"]
    n_participants = data["participant_id"].nunique()
    n_trials = len(data) // n_participants
    rt = data["rt"].to_numpy()
    upper = data["response"].to_numpy() > 0
    feedback = data["feedback"].to_numpy()
    rl_alpha, scaler, a, z, t, theta = 0.6, 3.2, 1.2, 0.1, 0.1, 0.1

    v = np.empty(len(data))
    for start in range(0, len(data), n_trials):
        q = np.array([0.5, 0.5])
        for i in range(start, start + n_trials):
            v[i] = (q[1] - q[0]) * scaler
            q[int(upper[i])] += rl_alpha * (feedback[i] - q[int(upper[i])])
    angle_logp = make_jax_matrix_logp_funcs_from_onnx(model=fixture_path / "angle.onnx")
    expected = angle_logp(
        np.column_stack(
            [v, *(np.full(len(data), p) for p in (a, z, t, theta))]
            + [rt, np.where(upper, 1.0, -1.0)]
        )
    )

    logp = make_rldm_logp_func(
        n_participants, n_trials, model=fixture_path / "angle.onnx"
    )
    if jitted:
        logp = jax.jit(logp)
    params = [np.full(len(data), p) for p in (rl_alpha, scaler, a, z, t, theta)]
    responses = np.where(upper, 1.0, lower_code)
    result = logp(np.column_stack([rt, responses]), *params, feedback)

    # float32-safe tolerance; feeding the LAN a 0 choice is off by O(1).
    np.testing.assert_allclose(result, expected, atol=1e-4)


@pytest.fixture
def small_rldm_case(fixture_path):
    """A small mixed-choice dataset with a local angle LAN and valid parameters."""
    data = np.array([[0.9, 0.0], [1.1, 1.0], [0.8, 0.0], [1.2, 1.0]])
    params = tuple(np.full(len(data), p) for p in (0.6, 3.2, 1.2, 0.4, 0.1, 0.1))
    feedback = np.array([1.0, 0.0, 0.0, 1.0])
    logp = make_rldm_logp_func(1, len(data), model=fixture_path / "angle.onnx")
    return logp, data, params, feedback


@pytest.mark.parametrize("response", [np.nan, np.inf, -np.inf, -2.0, 2.0, 0.5])
@pytest.mark.parametrize("jitted", [False, True], ids=["eager", "jit"])
def test_invalid_response_does_not_become_a_boundary(small_rldm_case, response, jitted):
    """Missing and unsupported response codes must not yield a valid likelihood."""
    logp, data, params, feedback = small_rldm_case
    data[0, 1] = response
    if jitted:
        logp = jax.jit(logp)

    result = np.asarray(logp(data, *params, feedback)).ravel()

    assert np.isnan(result[0]), result


@pytest.mark.parametrize("shape", [(4, 3), (4, 1), (8,), (1, 4, 2)])
@pytest.mark.parametrize("jitted", [False, True], ids=["eager", "jit"])
def test_data_requires_exactly_rt_and_response(small_rldm_case, shape, jitted):
    """Reject extra/missing columns and wrong-rank data rather than dropping fields."""
    logp, _, params, feedback = small_rldm_case
    if jitted:
        logp = jax.jit(logp)

    with pytest.raises(ValueError, match=r"two columns.*\[rt, response\]"):
        logp(np.ones(shape), *params, feedback)


@pytest.mark.parametrize("jitted", [False, True], ids=["eager", "jit"])
def test_response_codings_have_identical_parameter_gradients(small_rldm_case, jitted):
    """The response recoding preserves gradients of all six fitted parameters."""
    logp, data, params, feedback = small_rldm_case

    def summed_logp(values, *parameters):
        return jnp.sum(logp(values, *parameters, feedback))

    gradient = jax.grad(summed_logp, argnums=tuple(range(1, 7)))
    if jitted:
        gradient = jax.jit(gradient)
    gradients_zero_one = gradient(data, *params)
    data[:, 1] = np.where(data[:, 1] == 0, -1.0, 1.0)
    gradients_minus_one_one = gradient(data, *params)

    for grad_zero_one, grad_minus_one_one in zip(
        gradients_zero_one, gradients_minus_one_one, strict=True
    ):
        assert np.isfinite(grad_zero_one).all()
        np.testing.assert_allclose(grad_zero_one, grad_minus_one_one, atol=1e-6)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [(make_rldm_logp_func, {}), (make_rldm_logp_op, {"n_params": 6})],
)
def test_default_model_is_hub_angle(monkeypatch, factory, kwargs):
    """Without ``model``, the factories load the angle model's network from the Hub."""
    requested = []
    monkeypatch.setattr(
        rldm_optimized,
        "make_jax_matrix_logp_funcs_from_onnx",
        lambda model: requested.append(model),
    )
    factory(n_participants=1, n_trials=1, **kwargs)
    assert requested == ["angle_extended.onnx"]

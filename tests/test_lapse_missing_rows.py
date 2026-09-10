"""Per-row marginalisation of the lapse mixture on missing-RT and omission rows.

Regression tests for lnccbrown/HSSM#1322. The observation model is the mixture

    f(rt, c | theta) = (1 - p) * s(rt, c | theta) + p * l(rt) * q(c),

with ``s`` the SSM likelihood, ``l`` the lapse RT density and
``q(c) = 1 / n_choices``. Rows with ``rt == -999.0`` are not observed at their
RT, so the lapse term must be marginalised the same way the SSM term is:

- omission rows (deadline ``d``): ``(1 - p) * S_s(d) + p * (1 - CDF_l(d))``,
- missing-RT rows (choice observed): ``(1 - p) * P_s(c) + p / n_choices``.

These tests are fast: no sampling, only graph construction and evaluation.
``scipy`` is used purely as an oracle for the Normal-lapse survival function.
"""

from pathlib import Path

import bambi as bmb
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from scipy import stats

from hssm.distribution_utils import (
    assemble_callables,
    make_distribution,
    make_likelihood_callable,
    make_missing_data_callable,
)

LIST_PARAMS = ["v", "a", "z", "t"]
THETA = dict(v=0.5, a=1.5, z=0.5, t=0.3)
FLOOR = 1e-29
LAPSE_UPPER = 20.0

# Omission rows on top (assemble_callables expects missing rows first),
# columns: rt, response, deadline.
DATA_DEADLINE = np.array(
    [
        [-999.0, 1.0, 1.5],
        [-999.0, -1.0, 0.8],
        [0.9, 1.0, 3.0],
        [1.2, -1.0, 3.0],
        [0.6, 1.0, 3.0],
    ]
)

# Missing-RT rows on top, columns: rt, response.
DATA_MISSING = np.array(
    [
        [-999.0, 1.0],
        [-999.0, -1.0],
        [0.9, 1.0],
        [1.2, -1.0],
    ]
)


@pytest.fixture(scope="module")
def fixture_path() -> Path:
    return Path(__file__).parent / "fixtures"


def _uniform_lapse() -> bmb.Prior:
    return bmb.Prior("Uniform", lower=0.0, upper=LAPSE_UPPER)


def _theta_tensors(v_vector_len: int | None = None) -> list:
    """Return [v, a, z, t] as pytensor tensors; ``v`` trialwise if a length is given."""
    v = THETA["v"] if v_vector_len is None else np.full(v_vector_len, THETA["v"])
    return [pt.as_tensor_variable(np.asarray(x, dtype=np.float64)) for x in (v,)] + [
        pt.as_tensor_variable(np.float64(THETA[k])) for k in ("a", "z", "t")
    ]


def _build_assembled(
    fixture_path: Path,
    backend: str,
    has_deadline: bool,
    n_obs: int,
):
    """Assemble the DDM LAN with the OPN (deadline) or CPN (no deadline) fixture."""
    params_is_reg = [True] + [False] * 3
    suffix = "opn" if has_deadline else "cpn"
    if backend == "jax":
        likelihood = make_likelihood_callable(
            fixture_path / "ddm.onnx",
            loglik_kind="approx_differentiable",
            backend="jax",
            params_is_reg=params_is_reg,
        )
        missing = make_missing_data_callable(
            fixture_path / f"ddm_{suffix}.onnx",
            backend="jax",
            params_is_reg=params_is_reg,
            params_only=not has_deadline,
        )
    else:
        likelihood = make_likelihood_callable(
            fixture_path / "ddm.onnx",
            loglik_kind="approx_differentiable",
            backend="pytensor",
        )
        missing = make_missing_data_callable(
            fixture_path / f"ddm_{suffix}.onnx", backend="pytensor"
        )
    assembled = assemble_callables(
        likelihood,
        missing,
        params_only=not has_deadline,
        has_deadline=has_deadline,
    )
    return assembled, likelihood, missing


def _make_dist(loglik, has_deadline: bool, lapse=None, n_choices: int | None = 2):
    return make_distribution(
        rv="ddm",
        loglik=loglik,
        list_params=list(LIST_PARAMS),
        lapse=_uniform_lapse() if lapse is None else lapse,
        n_choices=n_choices,
        has_deadline=has_deadline,
    )


def _old_formula(ssm_logp: np.ndarray, rt: np.ndarray, p: float) -> np.ndarray:
    """The pre-fix mixture: lapse density evaluated at the rt column of every row."""
    lapse_logp = stats.uniform.logpdf(rt, 0.0, LAPSE_UPPER)  # -inf at -999
    with np.errstate(divide="ignore"):
        return np.log((1.0 - p) * np.exp(ssm_logp) + p * np.exp(lapse_logp) + FLOOR)


# --------------------------------------------------------------------------
# 1. Omission rows (deadline, OPN)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.0, 0.05])
def test_omission_rows_use_lapse_survival_at_deadline(fixture_path, p):
    data = DATA_DEADLINE
    n_obs = data.shape[0]
    assembled, _, opn = _build_assembled(fixture_path, "jax", True, n_obs)
    dist = _make_dist(assembled, has_deadline=True)
    params = _theta_tensors(n_obs)

    logp = dist.logp(pt.as_tensor_variable(data), *params, np.float64(p)).eval()
    ssm_logp = assembled(pt.as_tensor_variable(data), *params).eval()

    # Omission rows: (1 - p) * S_s(d) + p * (1 - d / 20).
    v_missing = params[0][:2]
    opn_logp = opn(pt.as_tensor_variable(data[:2, -1:]), v_missing, *params[1:]).eval()
    deadline = data[:2, -1]
    expected_missing = np.log(
        (1.0 - p) * np.exp(opn_logp) + p * (1.0 - deadline / LAPSE_UPPER) + FLOOR
    )
    np.testing.assert_allclose(logp[:2], expected_missing, rtol=1e-10)
    # The OPN term entering the mixture is exactly the assembled SSM term.
    np.testing.assert_allclose(ssm_logp[:2], opn_logp, rtol=1e-10)

    # Observed rows are unchanged from the pre-fix formula.
    np.testing.assert_allclose(
        logp[2:], _old_formula(ssm_logp, data[:, 0], p)[2:], rtol=1e-10
    )


def test_omission_row_deadline_limits(fixture_path):
    """d = upper reproduces the p = 0 value; d -> 0 gives log((1-p) S_s + p)."""
    p = 0.05
    n_obs = DATA_DEADLINE.shape[0]
    assembled, _, _ = _build_assembled(fixture_path, "jax", True, n_obs)
    dist = _make_dist(assembled, has_deadline=True)
    params = _theta_tensors(n_obs)

    # d = 20: the lapse survival is zero, so the row equals its p = 0 value up
    # to the (1 - p) weight.
    data_upper = DATA_DEADLINE.copy()
    data_upper[0, -1] = LAPSE_UPPER
    logp_p = dist.logp(pt.as_tensor_variable(data_upper), *params, np.float64(p)).eval()
    logp_0 = dist.logp(
        pt.as_tensor_variable(data_upper), *params, np.float64(0.0)
    ).eval()
    np.testing.assert_allclose(logp_p[0], np.log(1.0 - p) + logp_0[0], rtol=1e-10)

    # d -> 0: the lapse survival is one.
    data_zero = DATA_DEADLINE.copy()
    data_zero[0, -1] = 1e-12
    logp_zero = dist.logp(
        pt.as_tensor_variable(data_zero), *params, np.float64(p)
    ).eval()
    ssm_zero = assembled(pt.as_tensor_variable(data_zero), *params).eval()
    np.testing.assert_allclose(
        logp_zero[0], np.log((1.0 - p) * np.exp(ssm_zero[0]) + p + FLOOR), rtol=1e-9
    )


# --------------------------------------------------------------------------
# 2. Missing-RT rows (no deadline, CPN)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.0, 0.05])
def test_missing_rt_rows_use_uniform_choice(fixture_path, p):
    data = DATA_MISSING
    n_obs = data.shape[0]
    assembled, _, cpn = _build_assembled(fixture_path, "pytensor", False, n_obs)
    dist = _make_dist(assembled, has_deadline=False, n_choices=2)
    params = _theta_tensors(n_obs)

    logp = dist.logp(pt.as_tensor_variable(data), *params, np.float64(p)).eval()
    cpn_logp = cpn(None, params[0][:2], *params[1:]).eval()

    expected_missing = np.log((1.0 - p) * np.exp(cpn_logp) + p / 2.0 + FLOOR)
    np.testing.assert_allclose(logp[:2], expected_missing, rtol=1e-10)


@pytest.mark.parametrize("p", [0.05, 0.5])
def test_missing_rt_rows_sum_to_one_over_choices(p):
    """With P_s(c) summing to one, sum_c [(1-p) P_s(c) + p q(c)] == 1 (pins q = 1/n)."""
    log_probs = {1.0: np.log(0.3), -1.0: np.log(0.7)}

    def stub_loglik(data, v, a, z, t):
        choice = data[:, 1]
        return pt.switch(pt.eq(choice, 1.0), log_probs[1.0], log_probs[-1.0])

    data = DATA_MISSING[:2]
    dist = _make_dist(stub_loglik, has_deadline=False, n_choices=2)
    logp = dist.logp(pt.as_tensor_variable(data), *_theta_tensors(), np.float64(p))
    total = np.exp(logp.eval()).sum()
    np.testing.assert_allclose(total, 1.0, atol=1e-12)


# --------------------------------------------------------------------------
# 3. p = 0 reproduces the pre-fix values on every row
# --------------------------------------------------------------------------


@pytest.mark.parametrize("has_deadline", [True, False])
def test_p_zero_matches_pre_fix_values(fixture_path, has_deadline):
    data = DATA_DEADLINE if has_deadline else DATA_MISSING
    n_obs = data.shape[0]
    backend = "jax" if has_deadline else "pytensor"
    assembled, _, _ = _build_assembled(fixture_path, backend, has_deadline, n_obs)
    dist = _make_dist(assembled, has_deadline=has_deadline)
    params = _theta_tensors(n_obs)

    logp = dist.logp(pt.as_tensor_variable(data), *params, np.float64(0.0)).eval()
    ssm_logp = assembled(pt.as_tensor_variable(data), *params).eval()
    np.testing.assert_allclose(
        logp, _old_formula(ssm_logp, data[:, 0], 0.0), rtol=1e-12
    )


# --------------------------------------------------------------------------
# 4. Non-uniform lapse: survival function of a Normal lapse
# --------------------------------------------------------------------------


def test_normal_lapse_uses_survival_function_on_omission_rows():
    p = 0.05
    m_s = 0.4  # stub SSM survival on the omission row

    def stub_loglik(data, v, a, z, t):
        return pt.full((data.shape[0],), np.log(m_s))

    data = np.array([[-999.0, 1.0, 1.2], [0.9, 1.0, 3.0]])
    lapse = bmb.Prior("Normal", mu=1.0, sigma=0.5)
    dist = _make_dist(stub_loglik, has_deadline=True, lapse=lapse)
    logp = dist.logp(pt.as_tensor_variable(data), *_theta_tensors(), np.float64(p))
    logp = logp.eval()

    # Solve the mixture for the lapse term on the omission row.
    lapse_term = (np.exp(logp[0]) - (1.0 - p) * m_s) / p
    np.testing.assert_allclose(lapse_term, stats.norm.sf(1.2, 1.0, 0.5), rtol=1e-7)

    # Observed row: lapse density at rt.
    lapse_obs = (np.exp(logp[1]) - (1.0 - p) * m_s) / p
    np.testing.assert_allclose(lapse_obs, stats.norm.pdf(0.9, 1.0, 0.5), rtol=1e-7)


# --------------------------------------------------------------------------
# 5. Gradients on an omission row
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.05, 0.5])
def test_gradients_on_omission_row(fixture_path, p):
    data = DATA_DEADLINE[:3]  # two omissions + one observed row
    n_obs = data.shape[0]
    assembled, _, opn = _build_assembled(fixture_path, "jax", True, n_obs)
    dist = _make_dist(assembled, has_deadline=True)

    p_outlier = pt.dscalar("p_outlier")
    v = pt.dvector("v")
    params = [v] + _theta_tensors()[1:]
    v_val = np.full(n_obs, THETA["v"])
    data_t = pt.as_tensor_variable(data)

    logp = dist.logp(data_t, *params, p_outlier)
    row = logp[0]
    d = data[0, -1]

    # Pieces of the closed form.
    opn_logp = opn(pt.as_tensor_variable(data[:2, -1:]), v[:2], *params[1:])
    m_s = np.exp(opn_logp[0].eval({v: v_val}))
    m_l = 1.0 - d / LAPSE_UPPER
    mix = (1.0 - p) * m_s + p * m_l + FLOOR

    # d logp / d p == (m_l - m_s) / ((1 - p) m_s + p m_l)
    grad_p = pytensor.grad(row, p_outlier).eval({v: v_val, p_outlier: p})
    np.testing.assert_allclose(grad_p, (m_l - m_s) / mix, rtol=1e-6)
    assert np.isfinite(grad_p)

    # d logp / d v == r * d opn / d v, r the SSM responsibility.
    r = (1.0 - p) * m_s / mix
    grad_v = pytensor.grad(row, v).eval({v: v_val, p_outlier: p})
    grad_opn_v = pytensor.grad(opn_logp[0], v).eval({v: v_val})
    assert np.all(np.isfinite(grad_v))
    np.testing.assert_allclose(grad_v, r * grad_opn_v, rtol=1e-6, atol=1e-12)
    # The responsibility is strictly below one whenever p > 0.
    assert abs(grad_v[0]) < abs(grad_opn_v[0])


def test_pre_fix_gradient_shape_is_gone(fixture_path):
    """Pre-fix, d logp / d p on a missing row was -1/(1-p) (no lapse mass)."""
    p = 0.05
    data = DATA_DEADLINE[:3]
    n_obs = data.shape[0]
    assembled, _, _ = _build_assembled(fixture_path, "jax", True, n_obs)
    dist = _make_dist(assembled, has_deadline=True)
    p_outlier = pt.dscalar("p_outlier")
    params = _theta_tensors(n_obs)
    logp = dist.logp(pt.as_tensor_variable(data), *params, p_outlier)
    grad_p = pytensor.grad(logp[0], p_outlier).eval({p_outlier: p})
    assert not np.isclose(grad_p, -1.0 / (1.0 - p))


# --------------------------------------------------------------------------
# 6. Backends agree
# --------------------------------------------------------------------------


def test_jax_and_pytensor_backends_agree_with_missing_rows(fixture_path):
    p = 0.05
    data = DATA_DEADLINE
    n_obs = data.shape[0]
    values, grads = {}, {}
    for backend in ("jax", "pytensor"):
        assembled, _, _ = _build_assembled(fixture_path, backend, True, n_obs)
        dist = _make_dist(assembled, has_deadline=True)
        v = pt.dvector("v")
        params = [v] + _theta_tensors()[1:]
        logp = dist.logp(pt.as_tensor_variable(data), *params, np.float64(p))
        v_val = np.full(n_obs, THETA["v"])
        values[backend] = logp.eval({v: v_val})
        grads[backend] = pytensor.grad(logp.sum(), v).eval({v: v_val})

    np.testing.assert_allclose(values["jax"], values["pytensor"], rtol=1e-4)
    np.testing.assert_allclose(grads["jax"], grads["pytensor"], rtol=1e-4)
    assert np.all(np.isfinite(values["jax"]))
    assert np.all(np.isfinite(grads["jax"]))


# --------------------------------------------------------------------------
# 7. Clear errors
# --------------------------------------------------------------------------


def _stub_loglik(data, v, a, z, t):
    return pt.zeros((data.shape[0],))


def test_lapse_without_logcdf_raises_on_deadline_data():
    lapse = bmb.Prior("SkewNormal", mu=0.0, sigma=1.0, alpha=1.0)
    with pytest.raises(ValueError, match="SkewNormal"):
        _make_dist(_stub_loglik, has_deadline=True, lapse=lapse)


def test_lapse_without_logcdf_is_fine_without_deadline():
    lapse = bmb.Prior("SkewNormal", mu=0.0, sigma=1.0, alpha=1.0)
    dist = _make_dist(_stub_loglik, has_deadline=False, lapse=lapse, n_choices=2)
    logp = dist.logp(
        pt.as_tensor_variable(DATA_MISSING), *_theta_tensors(), np.float64(0.05)
    ).eval()
    assert np.all(np.isfinite(logp))


def test_missing_rows_without_n_choices_raise():
    dist = _make_dist(_stub_loglik, has_deadline=False, n_choices=None)
    with pytest.raises(ValueError, match="n_choices"):
        dist.logp(
            pt.as_tensor_variable(DATA_MISSING), *_theta_tensors(), np.float64(0.05)
        )


def test_float_lapse_with_missing_rows_raises():
    dist = _make_dist(_stub_loglik, has_deadline=False, lapse=np.log(0.5), n_choices=2)
    with pytest.raises(ValueError, match="float `lapse`"):
        dist.logp(
            pt.as_tensor_variable(DATA_MISSING), *_theta_tensors(), np.float64(0.05)
        )


def test_no_missing_rows_keeps_working_without_n_choices():
    """Existing callers (no n_choices, no missing rows) are unaffected."""
    data = DATA_MISSING[2:]
    dist = _make_dist(_stub_loglik, has_deadline=False, n_choices=None)
    logp = dist.logp(
        pt.as_tensor_variable(data), *_theta_tensors(), np.float64(0.05)
    ).eval()
    expected = np.log(0.95 + 0.05 / LAPSE_UPPER + FLOOR)
    np.testing.assert_allclose(logp, expected, rtol=1e-12)

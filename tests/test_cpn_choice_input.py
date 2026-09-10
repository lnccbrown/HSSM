"""The choice-probability network (CPN) input contract (lnccbrown/HSSM#1324).

Since HSSM 0.6.0 a CPN takes the observed choice as its last input,
``[theta..., choice] -> log P(choice | theta)``, exactly like an OPN takes the
deadline, ``[theta..., deadline]``. HSSM passes each missing-RT row's
``response`` to the network, coded as in ``model_config.choices``. These tests
pin the contract end to end: the load-time width check, the data validation
of missing rows, and the mixture value on a missing row against a direct
onnxruntime evaluation of the fixture.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime
import pandas as pd
import pymc as pm
import pytensor
import pytest
from onnx import TensorProto, helper, numpy_helper

import hssm

FIXTURES = Path(__file__).parent / "fixtures"
LAN = FIXTURES / "ddm.onnx"
CPN = FIXTURES / "ddm_cpn.onnx"
THETA = dict(v=0.5, a=1.5, z=0.5, t=0.3)
FLOOR = 1e-29
WIDTH_MESSAGE = (
    r"CPN artifacts take the observed choice as their last input since HSSM "
    r"0\.6\.0 \(input width n_params \+ 1 = 5\); this network has input width 4\. "
    r"See https://github\.com/lnccbrown/HSSM/issues/1324\."
)


@pytest.fixture(autouse=True, scope="module")
def _float64():
    """Pin float64 (other modules switch to float32 at collection time)."""
    prev_floatx = pytensor.config.floatX
    prev_x64 = jax.config.jax_enable_x64
    hssm.set_floatX("float64", update_jax=True)
    yield
    pytensor.config.floatX = prev_floatx
    jax.config.update("jax_enable_x64", prev_x64)


def _simulated_missing_data(n_trials: int = 40, n_missing: int = 6) -> pd.DataFrame:
    """Simulated DDM data whose first ``n_missing`` rows have a missing RT.

    The missing rows keep their simulated response, which is what the CPN
    receives.
    """
    df = hssm.simulate_data(
        model="ddm", theta=list(THETA.values()), size=n_trials, random_state=1324
    )
    df = df[["rt", "response"]].reset_index(drop=True)
    df.loc[: n_missing - 1, "rt"] = -999.0
    assert set(df.loc[: n_missing - 1, "response"]) == {-1.0, 1.0}
    return df


def _cpn_onnxruntime(rows) -> np.ndarray:
    """Evaluate the CPN fixture directly on ``(n, 5)`` rows ``[v, a, z, t, c]``."""
    session = onnxruntime.InferenceSession(str(CPN))
    name = session.get_inputs()[0].name
    rows = np.asarray(rows, dtype=np.float32)
    return np.array([session.run(None, {name: row[None, :]})[0].item() for row in rows])


def _write_params_only_graph(path: Path, width: int = 4) -> Path:
    """A hand-built ``(1, width)`` Gemm graph: the pre-0.6.0 params-only CPN."""
    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array(
        rng.normal(size=(width, 1)).astype(np.float32), name="W"
    )
    bias = numpy_helper.from_array(np.zeros(1, dtype=np.float32), name="B")
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["X", "W", "B"], ["Y"])],
        "params_only_cpn",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, width])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])],
        initializer=[weight, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def _build(df: pd.DataFrame, cpn=CPN, **kwargs) -> hssm.HSSM:
    return hssm.HSSM(
        data=df,
        model="ddm",
        missing_data=True,
        loglik=LAN,
        loglik_missing_data=cpn,
        loglik_kind="approx_differentiable",
        process_initvals=False,
        **kwargs,
    )


# --------------------------------------------------------------------------
# 1. Load-time width check
# --------------------------------------------------------------------------


def test_params_only_cpn_is_rejected_with_a_pointer(tmp_path):
    cpn4 = _write_params_only_graph(tmp_path / "ddm_cpn_params_only.onnx", width=4)
    with pytest.raises(ValueError, match=WIDTH_MESSAGE):
        _build(_simulated_missing_data(), cpn=cpn4)


def test_wrong_width_opn_is_rejected(tmp_path):
    """The same width check guards the OPN (deadline as last input)."""
    opn4 = _write_params_only_graph(tmp_path / "ddm_opn_wrong.onnx", width=4)
    df = _simulated_missing_data()
    df["deadline"] = 5.0
    with pytest.raises(
        ValueError,
        match=r"OPN artifacts take the deadline as their last input .* input width 4",
    ):
        _build(df, cpn=opn4, deadline=True)


def test_width_check_accepts_the_fixture_and_a_callable():
    """The fixture (width 5) passes; a user callable is not inspected.

    A user-supplied single-trial JAX callable receives the row's response as
    its ``data`` argument, like the ONNX network does.
    """
    df = _simulated_missing_data()
    _build(df)

    def cpn_callable(data, v, a, z, t):
        return jnp.where(data[0] > 0, jnp.log(0.6), jnp.log(0.4))

    model = _build(df, cpn=cpn_callable)
    assert np.isfinite(
        model.pymc_model.compile_logp()(model.pymc_model.initial_point())
    )


# --------------------------------------------------------------------------
# 2. Missing rows must carry a valid response
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value, shown",
    [(0.0, r"\[0\.0\]"), (2.0, r"\[2\.0\]"), (np.nan, r"\['NaN'\]")],
)
def test_missing_row_with_invalid_response_raises(bad_value, shown):
    df = _simulated_missing_data()
    df.loc[0, "response"] = bad_value
    with pytest.raises(
        ValueError,
        match=r"Missing-RT rows \(rt == -999\.0\) must carry the observed "
        r"response.*invalid responses on missing-RT rows: "
        + shown
        + r"; valid choices are \[-1, 1\]\. See https://github\.com/lnccbrown/"
        r"HSSM/issues/1324\.",
    ):
        _build(df)


def test_invalid_response_on_observed_row_keeps_the_existing_message():
    df = _simulated_missing_data()
    df.loc[df.index[-1], "response"] = 0.0
    with pytest.raises(ValueError, match=r"Invalid responses found in your dataset"):
        _build(df)


# --------------------------------------------------------------------------
# 3. End to end: the mixture on a missing row under the new contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.05, 0.5])
def test_missing_row_logp_is_the_lapse_mixture_of_the_cpn(p):
    """log((1 - p) * exp(cpn(theta, c)) + p / 2) on every missing row.

    ``cpn`` is evaluated directly through onnxruntime on ``[theta..., c]`` with
    ``c`` the row's own response, so a network that ignored the choice, or
    received it in another position, would fail here.
    """
    model = _build(_simulated_missing_data(), p_outlier=p)
    assert model.n_choices == 2
    data = model.data[model.response].to_numpy(dtype=np.float64)
    is_missing = data[:, 0] == -999.0
    assert is_missing.sum() == 6

    logp = pm.logp(model.model_distribution.dist(**THETA, p_outlier=p), data).eval()
    assert np.all(np.isfinite(logp))

    theta = np.array(list(THETA.values()))
    rows = np.c_[np.tile(theta, (is_missing.sum(), 1)), data[is_missing, 1]]
    cpn_logp = _cpn_onnxruntime(rows)
    expected = np.log((1.0 - p) * np.exp(cpn_logp) + p / 2.0 + FLOOR)
    np.testing.assert_allclose(logp[is_missing], expected, rtol=1e-5)

    # The choice reaches the network: rows with the same theta but different
    # responses get different values, and flipping a response moves it.
    by_choice = {c: logp[is_missing & (data[:, 1] == c)] for c in (-1.0, 1.0)}
    assert len(by_choice[-1.0]) and len(by_choice[1.0])
    np.testing.assert_allclose(by_choice[-1.0], by_choice[-1.0][0])
    np.testing.assert_allclose(by_choice[1.0], by_choice[1.0][0])
    assert not np.isclose(by_choice[-1.0][0], by_choice[1.0][0])

    # The compiled model logp is finite with these rows in place.
    assert np.isfinite(
        model.pymc_model.compile_logp()(model.pymc_model.initial_point())
    )

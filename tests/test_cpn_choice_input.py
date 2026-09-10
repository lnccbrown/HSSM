"""The choice-probability network (CPN) input contract (lnccbrown/HSSM#1324).

Since HSSM 0.6.0 a CPN takes the observed choice as its last input,
``[theta..., choice] -> log P(choice | theta)``, exactly like an OPN takes the
deadline, ``[theta..., deadline]``. HSSM passes each missing-RT row's
``response`` to the network, coded as in ``model_config.choices``. These tests
pin the contract end to end: the load-time width check, the data validation
of missing rows, and the mixture value on a missing row against a direct
onnxruntime evaluation of the fixture.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pandas as pd
import pymc as pm
import pytensor
import pytest
from onnx import TensorProto, helper, numpy_helper

import hssm
from hssm.defaults import MissingDataNetwork
from hssm.hssm import _check_missing_data_network_input_width

FIXTURES = Path(__file__).parent / "fixtures"
LAN = FIXTURES / "ddm.onnx"
CPN = FIXTURES / "ddm_cpn.onnx"
OPN = FIXTURES / "ddm_opn.onnx"
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

    The missing rows' responses are set explicitly (alternating ``1, -1``) so
    that both choices are present by construction, independently of the
    simulator's draw order; they are what the CPN receives.
    """
    df = hssm.simulate_data(
        model="ddm", theta=list(THETA.values()), size=n_trials, random_state=1324
    )
    df = df[["rt", "response"]].reset_index(drop=True)
    df.loc[: n_missing - 1, "rt"] = -999.0
    df.loc[: n_missing - 1, "response"] = np.resize([1.0, -1.0], n_missing)
    return df


def _write_params_only_graph(path: Path, width: int = 4, batch: int | str = 1) -> Path:
    """A hand-built ``(batch, width)`` Gemm graph: the pre-0.6.0 params-only CPN.

    A string ``batch`` produces a symbolic first dimension.
    """
    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array(
        rng.normal(size=(width, 1)).astype(np.float32), name="W"
    )
    bias = numpy_helper.from_array(np.zeros(1, dtype=np.float32), name="B")
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["X", "W", "B"], ["Y"])],
        "params_only_cpn",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [batch, width])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [batch, 1])],
        initializer=[weight, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def _build(
    df: pd.DataFrame, cpn: Path | Callable[..., Any] = CPN, **kwargs: Any
) -> hssm.HSSM:
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
# 0. The fixture itself
# --------------------------------------------------------------------------


def test_cpn_fixture_matches_its_provenance(cpn_onnxruntime):
    """The binary matches the recorded provenance (sha256, shape, sample outputs)."""
    prov = json.loads((FIXTURES / "ddm_cpn.provenance.json").read_text())
    assert hashlib.sha256(CPN.read_bytes()).hexdigest() == prov["sha256"]

    graph_input = onnx.load(CPN).graph.input[0]
    dims = [d.dim_value for d in graph_input.type.tensor_type.shape.dim]
    assert dims == prov["input_shape"]
    assert prov["contract"] == "[v, a, z, t, choice] -> log P(choice | theta)"

    sensitivity = prov["checks"]["onnxruntime_choice_sensitivity"]
    theta = sensitivity["theta"]
    out = cpn_onnxruntime([[*theta, 1.0], [*theta, -1.0]])
    np.testing.assert_allclose(
        out, [sensitivity["choice=1"], sensitivity["choice=-1"]], atol=1e-6
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


def test_width_check_accepts_the_fixture():
    """The width-5 fixture passes the check for the 4-parameter ddm."""
    _build(_simulated_missing_data())


def test_width_check_helper_counts_extra_fields_and_skips_callables(tmp_path):
    """The helper compares the graph width against ``n_params + 1`` directly.

    ``n_params`` is what HSSM passes: model parameters plus extra fields, so
    the same width-5 fixture is accepted for four inputs and rejected for five
    (e.g. one extra field). Callables, ``NONE`` and symbolic-dimension graphs
    are not inspected.
    """
    check = _check_missing_data_network_input_width

    check(CPN, MissingDataNetwork.CPN, n_params=4)
    check(OPN, MissingDataNetwork.OPN, n_params=4)

    with pytest.raises(
        ValueError,
        match=r"n_params \+ 1 = 6\); this network has input width 5\.",
    ):
        check(CPN, MissingDataNetwork.CPN, n_params=5)
    with pytest.raises(
        ValueError,
        match=r"OPN artifacts .* n_params \+ 1 = 4\); this network has input width 5\.",
    ):
        check(OPN, MissingDataNetwork.OPN, n_params=3)

    # Callables are the user's responsibility, and NONE has no network.
    check(lambda data, *params: params[0], MissingDataNetwork.CPN, n_params=99)
    check(CPN, MissingDataNetwork.NONE, n_params=99)

    # Symbolic input dimensions are left to the loader's own check.
    symbolic = _write_params_only_graph(tmp_path / "symbolic.onnx", width=4, batch="N")
    dims = [
        d.dim_value
        for d in onnx.load(symbolic).graph.input[0].type.tensor_type.shape.dim
    ]
    assert dims == [0, 4]
    check(symbolic, MissingDataNetwork.CPN, n_params=4)


def test_user_callable_receives_the_missing_rows_response():
    """A user-supplied single-trial JAX callable gets the row's response as ``data``.

    The callable returns a different value per choice, so the per-row mixture
    on the missing rows pins that the response column (not the rt column, and
    not ``None``) reaches the callable.
    """
    p = 0.05
    prob = {1.0: 0.6, -1.0: 0.4}

    def cpn_callable(data, v, a, z, t):
        return jnp.where(data[0] > 0, jnp.log(prob[1.0]), jnp.log(prob[-1.0]))

    model = _build(_simulated_missing_data(), cpn=cpn_callable, p_outlier=p)
    data = model.data[model.response].to_numpy(dtype=np.float64)
    is_missing = data[:, 0] == -999.0

    logp = pm.logp(model.model_distribution.dist(**THETA, p_outlier=p), data).eval()
    expected = np.log(
        (1.0 - p) * np.where(data[is_missing, 1] > 0, prob[1.0], prob[-1.0])
        + p / 2.0
        + FLOOR
    )
    np.testing.assert_allclose(logp[is_missing], expected, rtol=1e-10)
    assert np.all(np.isfinite(logp))
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


def test_deadline_models_do_not_require_a_response_on_omission_rows():
    """The check is gated on ``not deadline``: the OPN never sees the response.

    An omission row carrying a non-choice code (``0.0``, which the CPN check
    rejects on a missing-RT row) builds under ``deadline=True``. ``NaN`` is not
    used here because bambi/formulae reject incomplete rows before HSSM's own
    validation runs.
    """
    df = _simulated_missing_data()
    df["deadline"] = 5.0
    df.loc[0, "response"] = 0.0  # omission row, response not a choice code
    model = _build(df, cpn=OPN, deadline=True)
    assert np.isfinite(
        model.pymc_model.compile_logp()(model.pymc_model.initial_point())
    )


# --------------------------------------------------------------------------
# 3. End to end: the mixture on a missing row under the new contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.05, 0.5])
def test_missing_row_logp_is_the_lapse_mixture_of_the_cpn(cpn_onnxruntime, p):
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
    cpn_logp = cpn_onnxruntime(rows)
    expected = np.log((1.0 - p) * np.exp(cpn_logp) + p / 2.0 + FLOOR)
    np.testing.assert_allclose(logp[is_missing], expected, rtol=1e-5)

    # The choice reaches the network: rows with the same theta but different
    # responses get different values, and flipping a response moves it.
    by_choice = {c: logp[is_missing & (data[:, 1] == c)] for c in (-1.0, 1.0)}
    assert len(by_choice[-1.0]) == 3 and len(by_choice[1.0]) == 3
    np.testing.assert_allclose(by_choice[-1.0], by_choice[-1.0][0])
    np.testing.assert_allclose(by_choice[1.0], by_choice[1.0][0])
    assert not np.isclose(by_choice[-1.0][0], by_choice[1.0][0])

    # The compiled model logp is finite with these rows in place.
    assert np.isfinite(
        model.pymc_model.compile_logp()(model.pymc_model.initial_point())
    )

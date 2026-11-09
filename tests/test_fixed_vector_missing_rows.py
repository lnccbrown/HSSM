"""Fixed-vector parameters follow their rows when missing rows are moved on top.

``HSSM`` moves the rows with ``rt == -999.0`` to the top of the data, because
``assemble_callables`` scores the first ``n_missing`` rows with the missing-data
network. A parameter fixed to a vector (``v=np.array(...)``, one value per row in
the user's row order) must be reordered with the rows; otherwise rows at or above
the last missing row get another row's value.

The log-likelihood is a sum over rows, so it cannot depend on the row order, nor
on where the missing rows sit, as long as every row keeps its own value. These
tests are fast: no sampling, only graph construction and evaluation.
"""

import jax
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest

import hssm
from hssm.likelihoods import logp_ddm

A, Z, T = 1.25, 0.5, 0.2
MISSING_ROWS = [2, 5]  # not at the top, so HSSM has to move them


@pytest.fixture(autouse=True, scope="module")
def _float64():
    """Pin float64 for this module; other modules set float32 at import time."""
    prev_floatx = pytensor.config.floatX
    prev_x64 = jax.config.jax_enable_x64
    hssm.set_floatX("float64", update_jax=True)
    yield
    pytensor.config.floatX = prev_floatx
    jax.config.update("jax_enable_x64", prev_x64)


def _data(deadline: bool) -> pd.DataFrame:
    """Eight trials whose missing rows (2 and 5) are not on top, as the user passes them."""
    df = pd.DataFrame(
        {
            "rt": [0.9, 1.2, -999.0, 0.6, 0.8, -999.0, 1.4, 0.7],
            "response": [1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0],
        }
    )
    if deadline:
        df["deadline"] = [2.0, 2.0, 1.1, 2.0, 2.0, 1.4, 2.0, 2.0]
    return df


def _row_logps(df, v, deadline, fixture_path) -> pd.Series:
    """Per-row log-likelihood of an HSSM model, labelled by the rows of ``df``."""
    model = hssm.HSSM(
        data=df,
        model="ddm",
        loglik_kind="analytical",
        missing_data=True,
        deadline=deadline,
        loglik_missing_data=fixture_path
        / ("ddm_opn.onnx" if deadline else "ddm_cpn.onnx"),
        p_outlier=None,
        v=v,
        a={"prior": {"name": "Uniform", "lower": 0.5, "upper": 2.0}},
        z=Z,
        t=T,
        process_initvals=False,
    )
    assert (model.data["rt"].iloc[: len(MISSING_ROWS)] == -999.0).all()
    np.testing.assert_array_equal(model.params["v"].prior, v)  # left as given
    data = model.data[model.response].to_numpy(dtype=np.float64)
    # `v=0.0` is Bambi's placeholder: logp substitutes the fixed vector for it.
    dist = model.model_distribution.dist(v=0.0, a=A, z=Z, t=T)
    return pd.Series(pm.logp(dist, data).eval(), index=model.data.index)


@pytest.mark.parametrize("deadline", [False, True], ids=["missing_data", "deadline"])
def test_fixed_vector_follows_its_row_when_missing_rows_move(fixture_path, deadline):
    """Each row is scored with its own fixed drift, wherever its row ends up."""
    df = _data(deadline)
    v = np.linspace(-2.0, 2.0, len(df))  # one known drift per row of df

    # The user's order: HSSM moves rows 2 and 5 on top.
    user_order = _row_logps(df, v, deadline, fixture_path)

    # Observed rows: the analytical DDM with each row's own drift. (np.float64:
    # PyTensor would make float32 constants of the Python floats.)
    observed = (df["rt"] != -999.0).to_numpy()
    expected = logp_ddm(
        df.loc[observed, ["rt", "response"]].to_numpy(),
        v[observed],
        *np.float64([A, Z, T]),
    ).eval()
    np.testing.assert_allclose(user_order[df.index[observed]], expected, rtol=1e-10)

    # All rows, the missing ones included: same rows and drifts with the missing
    # rows already on top, an order HSSM leaves as it is.
    top = MISSING_ROWS + [i for i in range(len(df)) if i not in MISSING_ROWS]
    reference = _row_logps(df.iloc[top], v[top], deadline, fixture_path)
    pd.testing.assert_series_equal(
        user_order.sort_index(), reference.sort_index(), rtol=1e-10
    )

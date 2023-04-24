import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pytest
import ssms

from hssm.wfpt.adjust_logp import adjust_logp
from hssm.wfpt.base import log_pdf_sv
from hssm.wfpt.config import default_model_config


@pytest.fixture
def data():
    v_true, a_true, z_true, t_true = [0.5, 1.5, 0.5, 0.5]
    obs_ddm = ssms.basic_simulators.simulator(
        [v_true, a_true, z_true, t_true], model="ddm", n_samples=1000
    )
    obs_ddm = np.column_stack([obs_ddm["rts"][:, 0], obs_ddm["choices"][:, 0]])
    dataset = pd.DataFrame(obs_ddm, columns=["rt", "response"])
    dataset["x"] = dataset["rt"] * 0.1
    dataset["y"] = dataset["rt"] * 0.5
    return dataset


def test_adjust_logp_with_log_pdf_sv(data):
    v = 1
    sv = 0
    a = 0.5
    z = 0.5
    t = 0.5
    err = 1e-7
    logp = log_pdf_sv(data, v, sv, a, z, t, err, k_terms=7)
    adjusted_logp = adjust_logp(
        logp,
        ["v", "sv", "a", "z", "t"],
        v,
        sv,
        a,
        z,
        t,
        err,
        default_boundaries=default_model_config["ddm"]["default_boundaries"],
    )
    assert pt.all(adjusted_logp == logp).eval()

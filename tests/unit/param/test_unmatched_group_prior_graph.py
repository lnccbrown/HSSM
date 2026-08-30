"""Graph regressions for generated unmatched group-specific safe priors."""

import bambi as bmb
import pandas as pd
import pytest

import hssm
from hssm.param.parameterization_check import find_disconnected_free_rvs


def _group_only_data() -> pd.DataFrame:
    """Return a tiny two-participant data set for structure-only model builds."""
    rows = []
    for participant in range(2):
        for trial in range(4):
            rows.append(
                {
                    "rt": 0.45 + 0.03 * trial + 0.01 * participant,
                    "response": (-1, 1, 1, -1)[trial],
                    "theta": (-1.5, -0.5, 0.5, 1.5)[trial],
                    "dbs": (-1.0, 1.0, -1.0, 1.0)[trial],
                    "participant_id": f"p{participant}",
                }
            )
    return pd.DataFrame(rows)


def _build_ddm(parameter: str, formula: str, noncentered=True) -> hssm.HSSM:
    """Build a small analytical DDM without sampling or init-value processing."""
    fixed = {"v": 1.0, "a": 1.5, "z": 0.5, "t": 0.2}
    fixed.pop(parameter)
    return hssm.HSSM(
        data=_group_only_data(),
        model="ddm",
        include=[{"name": parameter, "formula": formula}],
        prior_settings="safe",
        noncentered=noncentered,
        p_outlier=0.0,
        process_initvals=False,
        initval_jitter=0.0,
        **fixed,
    )


@pytest.mark.parametrize("noncentered", [True, False])
def test_generated_group_only_slope_is_connected_centered_term(caplog, noncentered):
    """Retain the generated free location without an offset or orphan node."""
    model = _build_ddm(
        "v", "v ~ 1 + (0 + theta | participant_id)", noncentered=noncentered
    )

    prior = model.params["v"].prior["theta|participant_id"]
    assert isinstance(prior, bmb.Prior)
    assert isinstance(prior.args["mu"], bmb.Prior)
    assert prior.noncentered is False

    free_names = {rv.name for rv in model.pymc_model.free_RVs}
    assert "v_theta|participant_id" in free_names
    assert "v_theta|participant_id_mu" in free_names
    assert "v_theta|participant_id_sigma" in free_names
    assert "v_theta|participant_id_offset" not in free_names
    assert find_disconnected_free_rvs(model.pymc_model) == []

    fallback_messages = [
        record.message
        for record in caplog.records
        if "generated location-bearing group-only term" in record.message
    ]
    assert bool(fallback_messages) is noncentered


def test_matched_and_unmatched_terms_use_different_parameterizations():
    """Non-center a matched deviation while centering its group-only neighbor."""
    model = _build_ddm(
        "v",
        "v ~ 1 + theta + (0 + theta + dbs | participant_id)",
        noncentered=True,
    )

    matched = model.params["v"].prior["theta|participant_id"]
    unmatched = model.params["v"].prior["dbs|participant_id"]
    assert matched.args["mu"] == 0.0
    assert matched.noncentered is None
    assert isinstance(unmatched.args["mu"], bmb.Prior)
    assert unmatched.noncentered is False

    free_names = {rv.name for rv in model.pymc_model.free_RVs}
    assert "v_theta|participant_id_offset" in free_names
    assert "v_theta|participant_id" not in free_names
    assert "v_dbs|participant_id" in free_names
    assert "v_dbs|participant_id_mu" in free_names
    assert "v_dbs|participant_id_sigma" in free_names
    assert "v_dbs|participant_id_offset" not in free_names
    assert find_disconnected_free_rvs(model.pymc_model) == []


def test_non_normal_group_only_intercept_builds_centered():
    """A generated HDDM Gamma group intercept builds despite model-level NC."""
    model = _build_ddm("a", "a ~ 0 + (1 | participant_id)", noncentered=True)

    prior = model.params["a"].prior["1|participant_id"]
    assert isinstance(prior, bmb.Prior)
    assert prior.name == "Gamma"
    assert prior.noncentered is False

    free_names = {rv.name for rv in model.pymc_model.free_RVs}
    assert "a_1|participant_id" in free_names
    assert "a_1|participant_id_mu" in free_names
    assert "a_1|participant_id_sigma" in free_names
    assert "a_1|participant_id_offset" not in free_names
    assert find_disconnected_free_rvs(model.pymc_model) == []

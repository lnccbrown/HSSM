"""Graph regressions for generated unmatched group-specific safe priors."""

import bambi as bmb
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pytest
from pytensor.graph.traversal import ancestors

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


def _build_group_only_intercept(
    parameter: str,
    *,
    link=None,
    link_settings=None,
    noncentered=True,
) -> hssm.HSSM:
    """Build the smallest DDM variant that contains ``parameter``."""
    if parameter in {"v", "a", "z", "t", "p_outlier"}:
        model_name = "ddm"
        loglik_kind = "analytical"
        fixed = {"v": 1.0, "a": 1.5, "z": 0.5, "t": 0.2}
    elif parameter == "sv":
        model_name = "ddm_sdv"
        loglik_kind = "analytical"
        fixed = {"v": 1.0, "a": 1.5, "z": 0.5, "t": 0.2, "sv": 0.3}
    else:
        model_name = "full_ddm"
        loglik_kind = "blackbox"
        fixed = {
            "v": 1.0,
            "a": 1.5,
            "z": 0.5,
            "t": 0.2,
            "sv": 0.3,
            "sz": 0.1,
            "st": 0.1,
        }

    formula = f"{parameter} ~ 0 + (1 | participant_id)"
    specification: dict[str, object] = {"formula": formula}
    if link is not None:
        specification["link"] = link

    fixed.pop(parameter, None)
    kwargs = {}
    include = []
    if parameter == "p_outlier":
        kwargs["p_outlier"] = specification
    else:
        include.append({"name": parameter, **specification})
        kwargs["p_outlier"] = 0.0

    return hssm.HSSM(
        data=_group_only_data(),
        model=model_name,
        loglik_kind=loglik_kind,
        include=include,
        prior_settings="safe",
        link_settings=link_settings,
        noncentered=noncentered,
        process_initvals=False,
        initval_jitter=0.0,
        **fixed,
        **kwargs,
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


@pytest.mark.parametrize(
    ("parameter", "outer_family", "hyperparameters"),
    [
        pytest.param("v", "Normal", {"mu", "sigma"}, id="v"),
        pytest.param("a", "Gamma", {"mu", "sigma"}, id="a"),
        pytest.param("z", "Beta", {"alpha", "beta"}, id="z"),
        pytest.param("t", "Gamma", {"mu", "sigma"}, id="t"),
        pytest.param("sv", "Gamma", {"mu", "sigma"}, id="sv"),
        pytest.param("sz", "Gamma", {"mu", "sigma"}, id="sz"),
        pytest.param("st", "Gamma", {"mu", "sigma"}, id="st"),
        pytest.param("p_outlier", "Beta", {"alpha", "beta"}, id="p-outlier"),
    ],
)
@pytest.mark.parametrize(
    "parameterization",
    [True, False, "component-nc", "component-centered", "missing-component-key"],
    ids=[
        "model-nc",
        "model-centered",
        "component-nc",
        "component-centered",
        "missing-component-key",
    ],
)
def test_all_hddm_group_only_intercepts_build_with_connected_hyperpriors(
    parameter, outer_family, hyperparameters, parameterization
):
    """Build every HDDM location hierarchy as a direct connected group RV."""
    if parameterization == "component-nc":
        noncentered = {parameter: True}
    elif parameterization == "component-centered":
        noncentered = {parameter: False}
    elif parameterization == "missing-component-key":
        other_component = "a" if parameter != "a" else "v"
        noncentered = {other_component: False}
    else:
        noncentered = parameterization

    model = _build_group_only_intercept(parameter, noncentered=noncentered)

    prior = model.params[parameter].prior["1|participant_id"]
    assert prior.name == outer_family
    assert prior.noncentered is False
    assert set(prior.args) == hyperparameters

    group_name = f"{parameter}_1|participant_id"
    free_names = {rv.name for rv in model.pymc_model.free_RVs}
    assert group_name in free_names
    assert f"{group_name}_offset" not in free_names
    for hyperparameter in hyperparameters:
        assert f"{group_name}_{hyperparameter}" in free_names

    parameter_ancestors = set(ancestors([model.pymc_model.named_vars[parameter]]))
    assert model.pymc_model.named_vars[group_name] in parameter_ancestors
    assert find_disconnected_free_rvs(model.pymc_model) == []


@pytest.mark.parametrize(
    ("parameter", "link"),
    [
        pytest.param("a", "log", id="log-string"),
        pytest.param("a", bmb.Link("log"), id="bambi-log"),
        pytest.param(
            "z",
            hssm.Link("gen_logit", bounds=(0.0, 1.0)),
            id="generalized-logit",
        ),
        pytest.param(
            "a",
            hssm.Link(
                "custom_log",
                link=np.log,
                linkinv=np.exp,
                linkinv_backend=pt.exp,
            ),
            id="custom-log",
        ),
    ],
)
def test_transformed_group_only_intercept_uses_predictor_scale_graph(
    caplog, parameter, link
):
    """Keep transformed coefficients unconstrained and connect through the link."""
    model = _build_group_only_intercept(parameter, link=link, noncentered=True)

    prior = model.params[parameter].prior["1|participant_id"]
    assert prior.name == "Normal"
    assert prior.noncentered is False
    assert prior.args["mu"].name == "Normal"
    assert prior.args["sigma"].name == "Weibull"

    group_name = f"{parameter}_1|participant_id"
    free_names = {rv.name for rv in model.pymc_model.free_RVs}
    assert group_name in free_names
    assert f"{group_name}_offset" not in free_names
    parameter_ancestors = set(ancestors([model.pymc_model.named_vars[parameter]]))
    assert model.pymc_model.named_vars[group_name] in parameter_ancestors
    assert find_disconnected_free_rvs(model.pymc_model) == []

    assert not any("HSSM #1269" in record.message for record in caplog.records)
    fallback_messages = [
        record.message
        for record in caplog.records
        if "generated location-bearing group-only term" in record.message
    ]
    assert len(fallback_messages) == 1
    assert "linear-predictor scale before the inverse link" in fallback_messages[0]


@pytest.mark.parametrize(
    "link",
    [
        pytest.param("identity", id="string"),
        pytest.param(bmb.Link("identity"), id="bambi-object"),
        pytest.param(hssm.Link("identity"), id="hssm-object"),
    ],
)
def test_explicit_identity_non_normal_group_intercept_reaches_bambi(link):
    """Carry the explicit-identity Gamma hierarchy into a centered PyMC graph."""
    model = _build_group_only_intercept("a", link=link, noncentered=True)

    prior = model.params["a"].prior["1|participant_id"]
    assert prior.name == "Gamma"
    assert prior.noncentered is False
    free_names = {rv.name for rv in model.pymc_model.free_RVs}
    assert "a_1|participant_id" in free_names
    assert "a_1|participant_id_mu" in free_names
    assert "a_1|participant_id_sigma" in free_names
    assert "a_1|participant_id_offset" not in free_names
    assert find_disconnected_free_rvs(model.pymc_model) == []


def test_preset_identity_group_intercept_uses_hddm_graph():
    """Exercise the full model-level log-logit preset route for unbounded drift."""
    model = _build_group_only_intercept(
        "v", link_settings="log_logit", noncentered=True
    )

    assert model.params["v"].link == "identity"
    prior = model.params["v"].prior["1|participant_id"]
    assert prior.name == "Normal"
    assert prior.noncentered is False
    assert prior.args["mu"].name == "Normal"
    assert prior.args["mu"].args == {"mu": 2.0, "sigma": 3.0}
    assert prior.args["sigma"].name == "HalfNormal"
    assert prior.args["sigma"].args == {"sigma": 2.0}
    assert find_disconnected_free_rvs(model.pymc_model) == []

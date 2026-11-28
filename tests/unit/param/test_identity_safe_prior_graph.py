"""PyMC graph regression for identity-link common-intercept safe priors."""

import bambi as bmb
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.basic import equal_computations

import hssm


def _build_a_regression(data, link=...):
    """Build a small analytical DDM without sampling or initial-point processing."""
    parameter = {"name": "a", "formula": "a ~ 1 + theta"}
    if link is not ...:
        parameter["link"] = link

    return hssm.HSSM(
        data=data.head(8),
        model="ddm",
        include=[parameter],
        prior_settings="safe",
        v=0.5,
        z=0.5,
        t=0.2,
        p_outlier=0.0,
        process_initvals=False,
        initval_jitter=0,
    )


def test_identity_link_spellings_build_equivalent_common_intercept_prior_graphs(
    cavanagh_test,
):
    """Explicit identity links retain the omitted-link truncated Gamma graph."""
    reference = _build_a_regression(cavanagh_test)
    reference_prior = reference.params["a"].prior["Intercept"]
    reference_rv = reference.pymc_model.named_vars["a_Intercept"]

    assert reference_prior.name == "Gamma"
    assert reference_prior.is_truncated
    assert reference_prior.bounds == (0.0, float("inf"))

    explicit_models = [
        _build_a_regression(cavanagh_test, "identity"),
        _build_a_regression(cavanagh_test, bmb.Link("identity")),
        _build_a_regression(cavanagh_test, hssm.Link("identity")),
    ]

    for model in explicit_models:
        candidate_prior = model.params["a"].prior["Intercept"]
        candidate_rv = model.pymc_model.named_vars["a_Intercept"]
        reference_value = pt.scalar("reference_value", dtype=reference_rv.dtype)
        candidate_value = pt.scalar("candidate_value", dtype=candidate_rv.dtype)

        assert candidate_prior == reference_prior
        assert type(candidate_rv.owner.op) is type(reference_rv.owner.op)
        assert equal_computations(
            [pm.logp(reference_rv, reference_value)],
            [pm.logp(candidate_rv, candidate_value)],
            in_xs=[reference_value],
            in_ys=[candidate_value],
        )

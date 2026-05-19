import logging

import bambi as bmb
import numpy as np
import pymc as pm
import pytest

import hssm
from hssm import Prior
from hssm.param.parameterization_check import (
    check_user_priors_against_parameterization,
    find_disconnected_free_rvs,
)

hssm.set_floatX("float32")


def test_truncation():
    hssm_prior = Prior("Uniform", lower=0.0, upper=1.0)
    bmb_prior = bmb.Prior("Uniform", lower=0.0, upper=1.0)
    assert hssm_prior.args == bmb_prior.args

    bounded_prior1 = Prior("Uniform", lower=0.0, upper=1.0, bounds=(0.2, 0.8))
    assert bounded_prior1.is_truncated
    assert bounded_prior1._args == bmb_prior.args
    assert callable(bounded_prior1.dist)
    assert not bounded_prior1.args

    prior2 = Prior("Uniform", lower=0.0, upper=1.0, bounds=(-np.inf, np.inf))
    assert not prior2.is_truncated
    assert prior2.dist is None

    with pytest.raises(AssertionError):
        bounded_prior_err = Prior(
            "Uniform", lower=0.0, upper=1.0, bounds=(0.2, 0.8), dist=lambda x: x
        )


def test_str():
    hssm_prior = Prior("Uniform", lower=0.3, upper=1.0)
    bmb_prior = bmb.Prior("Uniform", lower=0.3, upper=1.0)

    assert str(hssm_prior) == str(bmb_prior)

    bounded_prior = Prior("Uniform", lower=0.3, upper=1.0, bounds=(0.4, 0.8))
    assert str(bounded_prior) == str(bmb_prior)


def test_eq():
    hssm_prior = Prior("Uniform", lower=0.3, upper=1.0)
    bmb_prior = bmb.Prior("Uniform", lower=0.3, upper=1.0)

    bounded_prior = Prior("Uniform", lower=0.3, upper=1.0, bounds=(0.4, 0.8))
    bounded_prior1 = Prior("Uniform", lower=0.3, upper=1.0, bounds=(0.4, 0.8))
    bounded_prior2 = Prior("Uniform", lower=0.3, upper=1.0, bounds=(-np.inf, np.inf))

    dist = lambda x: x

    dist_hssm_prior = Prior("Uniform", dist=dist)
    dist_bmb_prior = bmb.Prior("Uniform", dist=dist)

    assert hssm_prior == bmb_prior
    assert bounded_prior != bmb_prior

    assert bounded_prior == bounded_prior1
    assert hssm_prior == bounded_prior2

    assert dist_hssm_prior == dist_bmb_prior


# ---------------------------------------------------------------------------
# Centered / non-centered parameterization checks
# ---------------------------------------------------------------------------


def _hierarchical_ddm_prior_with_mu_hyperprior():
    """Build an include-spec that exercises the disconnected-node footgun.

    The returned spec supplies a Normal prior on ``1|participant_id`` whose
    ``mu`` is itself a hyperprior.
    """
    return [
        {
            "name": "v",
            "formula": "v ~ 1 + (1|participant_id)",
            "prior": {
                "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.5},
                "1|participant_id": {
                    "name": "Normal",
                    "mu": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
                    "sigma": {"name": "HalfNormal", "sigma": 0.5},
                },
            },
        }
    ]


def test_noncentered_default_warns_on_mu_hyperprior(cavanagh_test, caplog):
    """Warn when a user supplies a `mu` hyperprior under noncentered=True.

    Under the default ``noncentered=True`` a Normal group-specific prior with
    a nested ``mu`` hyperprior must trigger both the targeted warning and the
    general disconnected-node warning naming the orphaned ``_mu`` RV.
    """
    with caplog.at_level(logging.WARNING, logger="hssm"):
        model = hssm.HSSM(
            data=cavanagh_test,
            model="ddm",
            include=_hierarchical_ddm_prior_with_mu_hyperprior(),
            p_outlier=0.0,
        )

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "non-centered" in messages or "noncentered" in messages.lower()
    assert "1|participant_id" in messages
    # The orphaned RV should appear in the disconnected-nodes warning.
    disconnected = find_disconnected_free_rvs(model.pymc_model)
    assert any("_mu" in name for name in disconnected), disconnected


def test_centered_no_parameterization_warning(cavanagh_test, caplog):
    """Centered parameterization keeps the same prior dict footgun-free.

    With ``noncentered=False`` the user's ``mu`` hyperprior is used by the
    centered Normal, so no parameterization warning is emitted and no
    disconnected free RV remains in the graph.
    """
    with caplog.at_level(logging.WARNING, logger="hssm"):
        model = hssm.HSSM(
            data=cavanagh_test,
            model="ddm",
            include=_hierarchical_ddm_prior_with_mu_hyperprior(),
            p_outlier=0.0,
            noncentered=False,
        )

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "disconnected" not in messages.lower()
    assert "non-centered" not in messages.lower()
    assert find_disconnected_free_rvs(model.pymc_model) == []


def test_check_user_priors_skips_default_hyperpriors(cavanagh_test, caplog):
    """Defaults that already use a `mu` hyperprior do not trigger the warning.

    The targeted check must only fire for keys the user supplied. HSSM's own
    ``group_specific`` defaults also use ``mu=Normal(...)`` and would
    otherwise flood the warning channel.
    """
    # No user prior at all -> defaults kick in, which include a `mu` hyperprior
    # for the group_specific term when there is no common counterpart.
    with caplog.at_level(logging.WARNING, logger="hssm"):
        hssm.HSSM(
            data=cavanagh_test,
            model="ddm",
            include=[{"name": "v", "formula": "v ~ 0 + (1|participant_id)"}],
            p_outlier=0.0,
        )

    targeted_messages = [
        r.getMessage() for r in caplog.records if "User prior" in r.getMessage()
    ]
    assert targeted_messages == []


def test_general_disconnected_detector_finds_orphan():
    """Detect an orphan free RV in a hand-built ``pm.Model``.

    Builds a minimal model containing one connected RV and one orphan RV and
    checks that the detector returns only the orphan.
    """
    rng = np.random.default_rng(0)
    obs = rng.normal(size=20).astype(np.float32)
    with pm.Model() as m:
        connected_mu = pm.Normal("connected_mu", mu=0.0, sigma=1.0)
        # Orphan: created in the graph but never used.
        pm.Normal("orphan_rv", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=connected_mu, sigma=1.0, observed=obs)

    disconnected = find_disconnected_free_rvs(m)
    assert "orphan_rv" in disconnected
    assert "connected_mu" not in disconnected


def test_check_user_priors_unit():
    """Unit-level check of ``check_user_priors_against_parameterization``.

    The function must flag a user prior with a ``mu`` hyperprior under
    ``noncentered=True`` and be silent under ``noncentered=False``.
    """

    # Build a minimal params dict by hand. We only need attributes the check
    # actually reads: `.prior` (dict[str, bmb.Prior]) and
    # `._user_specified_prior_keys` (set[str]).
    class _FakeParam:
        def __init__(self, prior, user_keys):
            self.prior = prior
            self._user_specified_prior_keys = user_keys

    fake_prior = {
        "Intercept": bmb.Prior("Normal", mu=0.0, sigma=1.0),
        "1|participant_id": bmb.Prior(
            "Normal",
            mu=bmb.Prior("Normal", mu=0.0, sigma=0.5),
            sigma=bmb.Prior("HalfNormal", sigma=0.5),
        ),
    }
    fake_params = {"v": _FakeParam(fake_prior, {"Intercept", "1|participant_id"})}

    flagged = check_user_priors_against_parameterization(fake_params, True)
    assert len(flagged) == 1
    assert flagged[0].parameter == "v"
    assert flagged[0].term == "1|participant_id"

    assert check_user_priors_against_parameterization(fake_params, False) == []

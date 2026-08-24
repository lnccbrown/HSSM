"""Tests for HSSM prior wrappers and regression-prior diagnostics."""

import logging

import bambi as bmb
import numpy as np
import pymc as pm
import pytest

import hssm
from hssm import Prior
from hssm.param.parameterization_check import (
    check_user_priors_against_parameterization,
    check_user_priors_for_location_overparameterization,
    find_disconnected_free_rvs,
)
from hssm.prior import (
    _is_identity_link,
    get_default_prior,
    get_hddm_default_prior,
)

hssm.set_floatX("float32")


IDENTITY_LINKS = [
    pytest.param(None, id="omitted"),
    pytest.param("identity", id="string"),
    pytest.param(bmb.Link("identity"), id="bambi-object"),
    pytest.param(hssm.Link("identity"), id="hssm-object"),
]

TRANSFORMED_LINKS = [
    pytest.param("log", id="log-string"),
    pytest.param("logit", id="logit-string"),
    pytest.param(bmb.Link("log"), id="bambi-log"),
    pytest.param(bmb.Link("logit"), id="bambi-logit"),
    pytest.param(hssm.Link("log"), id="hssm-log"),
    pytest.param(hssm.Link("logit"), id="hssm-logit"),
    pytest.param(hssm.Link("gen_logit", bounds=(0.0, 1.0)), id="hssm-gen-logit"),
]


def _assert_prior_spec(prior, name, args, bounds):
    """Assert a generated HSSM prior's distribution, arguments, and support."""
    assert isinstance(prior, Prior)
    assert prior.name == name
    actual_args = prior._args if prior.is_truncated else prior.args
    assert actual_args == args
    assert prior.bounds == bounds
    expected_truncated = bounds is not None and any(np.isfinite(bounds))
    assert prior.is_truncated is expected_truncated
    assert callable(prior.dist) is expected_truncated


class TestPriorUnit:
    """Unit tests for Prior wrappers and parameterization check helpers."""

    def test_truncation(self):
        """Bounded priors use truncation only for finite bounds."""
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
            Prior("Uniform", lower=0.0, upper=1.0, bounds=(0.2, 0.8), dist=lambda x: x)

    def test_str(self):
        """Bounded wrappers preserve Bambi's prior display."""
        hssm_prior = Prior("Uniform", lower=0.3, upper=1.0)
        bmb_prior = bmb.Prior("Uniform", lower=0.3, upper=1.0)

        assert str(hssm_prior) == str(bmb_prior)

        bounded_prior = Prior("Uniform", lower=0.3, upper=1.0, bounds=(0.4, 0.8))
        assert str(bounded_prior) == str(bmb_prior)

    def test_eq(self):
        """Prior equality accounts for bounds and custom distributions."""
        hssm_prior = Prior("Uniform", lower=0.3, upper=1.0)
        bmb_prior = bmb.Prior("Uniform", lower=0.3, upper=1.0)

        bounded_prior = Prior("Uniform", lower=0.3, upper=1.0, bounds=(0.4, 0.8))
        bounded_prior1 = Prior("Uniform", lower=0.3, upper=1.0, bounds=(0.4, 0.8))
        bounded_prior2 = Prior(
            "Uniform", lower=0.3, upper=1.0, bounds=(-np.inf, np.inf)
        )

        dist = lambda x: x

        dist_hssm_prior = Prior("Uniform", dist=dist)
        dist_bmb_prior = bmb.Prior("Uniform", dist=dist)

        assert hssm_prior == bmb_prior
        assert bounded_prior != bmb_prior

        assert bounded_prior == bounded_prior1
        assert hssm_prior == bounded_prior2

        assert dist_hssm_prior == dist_bmb_prior

    @pytest.mark.parametrize(
        "link",
        IDENTITY_LINKS,
    )
    def test_identity_link_classification(self, link):
        """Recognize every spelling of the effective identity link."""
        assert _is_identity_link(link)

    @pytest.mark.parametrize(
        "link",
        TRANSFORMED_LINKS,
    )
    def test_transformed_link_classification(self, link):
        """Classify non-identity strings and link objects as transformed."""
        assert not _is_identity_link(link)

    @pytest.mark.parametrize("link", IDENTITY_LINKS)
    @pytest.mark.parametrize(
        ("bounds", "expected_args"),
        [
            pytest.param((-2.0, 3.0), {"mu": 0.5, "sigma": 0.25}, id="finite"),
            pytest.param((0.0, np.inf), {"mu": 0.0, "sigma": 0.25}, id="lower-bounded"),
            pytest.param(
                (-np.inf, 4.0), {"mu": 0.0, "sigma": 0.25}, id="upper-bounded"
            ),
            pytest.param((-np.inf, np.inf), {"mu": 0.0, "sigma": 0.25}, id="unbounded"),
            pytest.param(None, {"mu": 0.0, "sigma": 0.25}, id="no-bounds"),
        ],
    )
    def test_generic_common_intercept_identity_equivalence(
        self, link, bounds, expected_args
    ):
        """Use response-scale bounds for every spelling of identity."""
        prior = get_default_prior("common_intercept", "x", bounds, link)

        _assert_prior_spec(prior, "Normal", expected_args, bounds)

    @pytest.mark.parametrize("link", TRANSFORMED_LINKS)
    def test_generic_common_intercept_transformed_link(self, link):
        """Keep transformed-link intercept priors on the coefficient scale."""
        prior = get_default_prior("common_intercept", "x", (-2.0, 3.0), link)

        _assert_prior_spec(prior, "Normal", {"mu": 0.0, "sigma": 0.25}, bounds=None)

    @pytest.mark.parametrize("link", IDENTITY_LINKS)
    @pytest.mark.parametrize(
        ("param", "bounds", "name", "expected_args"),
        [
            pytest.param(
                "v",
                (-np.inf, np.inf),
                "Normal",
                {"mu": 2.0, "sigma": 3.0},
                id="v",
            ),
            pytest.param(
                "a",
                (0.0, np.inf),
                "Gamma",
                {"mu": 1.5, "sigma": 0.75},
                id="a",
            ),
            pytest.param("z", (0.0, 1.0), "Beta", {"alpha": 10, "beta": 10}, id="z"),
            pytest.param(
                "t",
                (0.0, np.inf),
                "Gamma",
                {"mu": 0.2, "sigma": 0.2},
                id="t",
            ),
            pytest.param(
                "sv",
                (0.0, np.inf),
                "HalfNormal",
                {"sigma": 2.0},
                id="sv",
            ),
            pytest.param(
                "sz",
                (0.0, np.inf),
                "HalfNormal",
                {"sigma": 0.5},
                id="sz",
            ),
            pytest.param(
                "st",
                (0.0, np.inf),
                "HalfNormal",
                {"sigma": 0.3},
                id="st",
            ),
            pytest.param(
                "p_outlier",
                None,
                "Beta",
                {"alpha": 5, "beta": 100},
                id="p-outlier",
            ),
        ],
    )
    def test_hddm_common_intercept_identity_equivalence(
        self, link, param, bounds, name, expected_args
    ):
        """Retain every HDDM location prior under explicit identity links."""
        prior = get_hddm_default_prior("common_intercept", param, bounds, link)

        _assert_prior_spec(prior, name, expected_args, bounds)

    @pytest.mark.parametrize("link", TRANSFORMED_LINKS)
    def test_hddm_common_intercept_transformed_link(self, link):
        """Keep transformed HDDM intercepts on the coefficient scale."""
        prior = get_hddm_default_prior("common_intercept", "z", (0.0, 1.0), link)

        _assert_prior_spec(prior, "Normal", {"mu": 0.0, "sigma": 0.25}, bounds=None)

    @pytest.mark.parametrize("link", [*IDENTITY_LINKS, *TRANSFORMED_LINKS])
    def test_common_slopes_are_link_invariant(self, link):
        """Keep ordinary and HDDM common slopes independent of link spelling."""
        generic = get_default_prior("common", "x", (-2.0, 3.0), link)
        hddm = get_hddm_default_prior("common", "z", (0.0, 1.0), link)

        for prior in (generic, hddm):
            _assert_prior_spec(prior, "Normal", {"mu": 0.0, "sigma": 0.25}, bounds=None)

    def test_hddm_group_only_identity_policy_is_deferred(self):
        """Leave the link-presence policy for unmatched group terms to #1225."""
        omitted = get_hddm_default_prior("group_intercept", "z", None, None)
        explicit = get_hddm_default_prior("group_intercept", "z", None, "identity")

        assert omitted.name == "Beta"
        assert explicit.name == "Normal"


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


class TestPriorIntegration:
    """Integration tests that build HSSM models and inspect warnings/graphs."""

    def test_noncentered_default_warns_on_mu_hyperprior(self, cavanagh_test, caplog):
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

    def test_centered_warns_only_about_matched_location_ridge(
        self, cavanagh_test, caplog
    ):
        """Centered matching effects have a ridge, but no orphaned ``mu``."""
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
        assert "non-identifiable" in messages
        assert "Intercept" in messages
        assert find_disconnected_free_rvs(model.pymc_model) == []

    def test_check_user_priors_skips_default_hyperpriors(self, cavanagh_test, caplog):
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

    def test_prior_settings_none_preserves_prior_and_structural_warning(
        self, cavanagh_test, caplog
    ):
        """Formula metadata supports diagnostics without safe-prior generation."""
        group_prior = {
            "name": "Normal",
            "mu": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
            "sigma": {"name": "HalfNormal", "sigma": 0.5},
        }
        with caplog.at_level(logging.WARNING, logger="hssm"):
            model = hssm.HSSM(
                data=cavanagh_test,
                model="ddm",
                include=[
                    {
                        "name": "v",
                        "formula": ("v ~ 1 + theta + (0 + theta|participant_id)"),
                        "prior": {
                            "Intercept": {
                                "name": "Normal",
                                "mu": 0.0,
                                "sigma": 1.0,
                            },
                            "theta": {
                                "name": "Normal",
                                "mu": 0.0,
                                "sigma": 1.0,
                            },
                            "theta|participant_id": group_prior,
                        },
                    }
                ],
                prior_settings=None,
                p_outlier=0.0,
                noncentered=False,
            )

        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "non-identifiable" in messages
        assert "theta|participant_id" in messages
        assert "common 'theta'" in messages
        assert set(model.params["v"].prior) == {
            "Intercept",
            "theta",
            "theta|participant_id",
        }

    @pytest.mark.parametrize(
        ("prior_settings", "noncentered", "reason_fragment", "has_orphan"),
        [
            (None, False, "non-identifiable", False),
            ("safe", True, "disconnected", True),
        ],
    )
    def test_group_wildcard_drives_targeted_diagnostics(
        self,
        cavanagh_test,
        caplog,
        prior_settings,
        noncentered,
        reason_fragment,
        has_orphan,
    ):
        """Expand a user group wildcard over Formulae terms for warnings."""
        group_wildcard = {
            "name": "Normal",
            "mu": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
            "sigma": {"name": "HalfNormal", "sigma": 0.5},
        }
        with caplog.at_level(logging.WARNING, logger="hssm"):
            model = hssm.HSSM(
                data=cavanagh_test,
                model="ddm",
                include=[
                    {
                        "name": "v",
                        "formula": "v ~ 0 + theta + (0 + theta|participant_id)",
                        "prior": {"group_specific": group_wildcard},
                    }
                ],
                prior_settings=prior_settings,
                p_outlier=0.0,
                noncentered=noncentered,
                process_initvals=False,
            )

        targeted = [
            record.getMessage()
            for record in caplog.records
            if "User prior" in record.getMessage()
        ]
        assert len(targeted) == 1
        assert "theta|participant_id" in targeted[0]
        assert reason_fragment in targeted[0]
        assert "theta|participant_id" not in model.params["v"].prior
        assert isinstance(model.params["v"].prior["group_specific"], bmb.Prior)
        disconnected = find_disconnected_free_rvs(model.pymc_model)
        assert any(name.endswith("_mu") for name in disconnected) is has_orphan

    def test_group_only_slope_not_flagged_by_unrelated_intercept(
        self, cavanagh_test, caplog
    ):
        """An Intercept does not make an unmatched group slope redundant."""
        spec = _hierarchical_ddm_prior_with_mu_hyperprior()
        spec[0]["formula"] = "v ~ 1 + (0 + theta|participant_id)"
        group_prior = spec[0]["prior"].pop("1|participant_id")
        spec[0]["prior"]["theta|participant_id"] = group_prior

        with caplog.at_level(logging.WARNING, logger="hssm"):
            hssm.HSSM(
                data=cavanagh_test,
                model="ddm",
                include=spec,
                p_outlier=0.0,
                noncentered=False,
            )

        overparam_messages = [
            r.getMessage()
            for r in caplog.records
            if "non-identifiable" in r.getMessage()
        ]
        assert overparam_messages == []


class TestPriorParameterizationUnit:
    """Unit tests for parameterization check utilities."""

    def test_general_disconnected_detector_finds_orphan(self):
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

    @pytest.mark.parametrize(
        ("mu", "sigma", "reason_fragment"),
        [
            (
                bmb.Prior("Normal", mu=0.0, sigma=0.5),
                bmb.Prior("HalfNormal", sigma=0.5),
                "disconnected",
            ),
            (1.5, bmb.Prior("HalfNormal", sigma=0.5), "ignore"),
            (0.0, bmb.Prior("HalfNormal", sigma=0.5), None),
            (
                np.array([0.0, 1.5]),
                bmb.Prior("HalfNormal", sigma=0.5),
                "ignore",
            ),
            (
                np.array([0.0, 0.0]),
                bmb.Prior("HalfNormal", sigma=0.5),
                None,
            ),
            (bmb.Prior("Normal", mu=0.0, sigma=0.5), 0.5, "cannot be built"),
            (1.5, 0.5, None),
        ],
        ids=[
            "free-mu-hierarchical-sigma",
            "fixed-nonzero-mu-hierarchical-sigma",
            "fixed-zero-mu-hierarchical-sigma",
            "fixed-vector-mu-hierarchical-sigma",
            "fixed-zero-vector-hierarchical-sigma",
            "free-mu-fixed-sigma",
            "fixed-mu-fixed-sigma",
        ],
    )
    def test_noncentered_warning_matrix(self, mu, sigma, reason_fragment):
        """Warnings distinguish Bambi's orphan, ignored, and failure paths."""
        params = _fake_params(_group_prior(mu=mu, sigma=sigma))

        flagged = check_user_priors_against_parameterization(params, True)

        if reason_fragment is None:
            assert flagged == []
        else:
            assert len(flagged) == 1
            assert reason_fragment in flagged[0].reason.lower()
            assert flagged[0].parameter == "v"
            assert flagged[0].term == GROUP_TERM

    @pytest.mark.parametrize(
        ("prior_noncentered", "model_noncentered", "expect_noncentered_warning"),
        [
            (False, True, False),
            (True, False, True),
            (None, False, False),
            (None, True, True),
            (None, None, True),
            (None, {"v": False}, False),
            (None, {"v": True}, True),
            (None, {"a": False}, True),
        ],
    )
    def test_effective_parameterization_precedence(
        self,
        prior_noncentered,
        model_noncentered,
        expect_noncentered_warning,
    ):
        """Per-prior settings beat component dictionaries and model defaults."""
        prior = _group_prior(
            mu=bmb.Prior("Normal", mu=0.0, sigma=0.5),
            sigma=bmb.Prior("HalfNormal", sigma=0.5),
            noncentered=prior_noncentered,
        )
        params = _fake_params(prior)

        noncentered_mismatches = check_user_priors_against_parameterization(
            params, model_noncentered
        )
        ridge_mismatches = check_user_priors_for_location_overparameterization(
            params, model_noncentered
        )

        assert bool(noncentered_mismatches) is expect_noncentered_warning
        assert bool(ridge_mismatches) is (not expect_noncentered_warning)

    @pytest.mark.parametrize(
        ("matched", "mu", "noncentered", "expect_warning"),
        [
            (True, bmb.Prior("Normal", mu=0.0, sigma=0.5), False, True),
            (True, bmb.Prior("Normal", mu=0.0, sigma=0.5), True, False),
            (False, bmb.Prior("Normal", mu=0.0, sigma=0.5), False, False),
            (True, 1.5, False, False),
            (True, np.array([1.0, 2.0]), False, False),
        ],
    )
    def test_location_ridge_requires_free_matched_centered_mu(
        self, matched, mu, noncentered, expect_warning
    ):
        """Only a free matched group mean under centering creates a ridge."""
        params = _fake_params(
            _group_prior(mu=mu, sigma=bmb.Prior("HalfNormal", sigma=0.5)),
            matched=matched,
        )

        flagged = check_user_priors_for_location_overparameterization(
            params, noncentered
        )

        assert bool(flagged) is expect_warning
        if flagged:
            assert "common 'theta'" in flagged[0].suggestion
            assert "non-identifiable" in flagged[0].reason

    def test_centered_location_ridge_is_not_normal_specific(self):
        """A free location in another location family creates the same ridge."""
        prior = bmb.Prior(
            "StudentT",
            nu=4,
            mu=bmb.Prior("Normal", mu=0.0, sigma=0.5),
            sigma=bmb.Prior("HalfNormal", sigma=0.5),
        )

        flagged = check_user_priors_for_location_overparameterization(
            _fake_params(prior), False
        )

        assert len(flagged) == 1
        assert flagged[0].term == GROUP_TERM
        assert "non-identifiable" in flagged[0].reason

    def test_centered_non_location_mu_does_not_report_exact_ridge(self):
        """A Gamma mean is not an additive location despite being named mu."""
        prior = bmb.Prior(
            "Gamma",
            mu=bmb.Prior("HalfNormal", sigma=0.5),
            sigma=bmb.Prior("HalfNormal", sigma=0.5),
        )

        flagged = check_user_priors_for_location_overparameterization(
            _fake_params(prior), False
        )

        assert flagged == []

    def test_noncentered_suggestion_uses_structural_expression(self):
        """Suggestions use Formulae metadata for matched and unmatched groups."""
        prior = _group_prior(
            mu=bmb.Prior("Normal", mu=0.0, sigma=0.5),
            sigma=bmb.Prior("HalfNormal", sigma=0.5),
        )

        matched = check_user_priors_against_parameterization(
            _fake_params(prior, matched=True), True
        )
        unmatched = check_user_priors_against_parameterization(
            _fake_params(prior, matched=False), True
        )

        assert "common 'theta'" in matched[0].suggestion
        assert "add the common 'theta'" in unmatched[0].suggestion.lower()

    def test_checks_skip_generated_and_non_group_priors(self):
        """Only user-specified keys identified by Formulae are diagnosed."""
        prior = _group_prior(
            mu=bmb.Prior("Normal", mu=0.0, sigma=0.5),
            sigma=bmb.Prior("HalfNormal", sigma=0.5),
        )
        generated = _fake_params(prior, user_specified=False)
        fake_display_name = _fake_params(prior)
        fake_display_name["v"]._group_term_names = {}
        fake_display_name["v"]._group_terms_with_common = set()

        for params in (generated, fake_display_name):
            assert check_user_priors_against_parameterization(params, True) == []
            assert (
                check_user_priors_for_location_overparameterization(params, False) == []
            )

    @pytest.mark.parametrize(
        ("noncentered", "check"),
        [
            (True, check_user_priors_against_parameterization),
            (False, check_user_priors_for_location_overparameterization),
        ],
    )
    def test_exact_group_prior_precedes_wildcard_in_diagnostics(
        self, noncentered, check
    ):
        """Diagnose the wildcard only where no exact user group prior exists."""
        wildcard = _group_prior(
            mu=bmb.Prior("Normal", mu=0.0, sigma=0.5),
            sigma=bmb.Prior("HalfNormal", sigma=0.5),
        )
        exact = _group_prior(
            mu=0.0,
            sigma=bmb.Prior("HalfNormal", sigma=0.5),
        )
        params = _fake_wildcard_params(wildcard, exact)

        flagged = check(params, noncentered)

        assert [mismatch.term for mismatch in flagged] == ["1|participant_id"]


GROUP_TERM = "theta|participant_id"


class _FakeRegressionParam:
    """Minimal RegressionParam shape consumed by the diagnostic helpers."""

    def __init__(self, prior, *, matched: bool, user_specified: bool):
        self.prior = {GROUP_TERM: prior}
        self._user_specified_prior_keys = {GROUP_TERM} if user_specified else set()
        self._group_term_names = {GROUP_TERM: "theta"}
        self._group_terms_with_common = {GROUP_TERM} if matched else set()


def _group_prior(mu, sigma, noncentered=None):
    kwargs = {"mu": mu, "sigma": sigma}
    if noncentered is not None:
        kwargs["noncentered"] = noncentered
    return bmb.Prior("Normal", **kwargs)


def _fake_params(prior, *, matched=True, user_specified=True):
    return {
        "v": _FakeRegressionParam(prior, matched=matched, user_specified=user_specified)
    }


def _fake_wildcard_params(wildcard, exact):
    param = _FakeRegressionParam(exact, matched=True, user_specified=True)
    param.prior = {
        "group_specific": wildcard,
        GROUP_TERM: exact,
    }
    param._user_specified_prior_keys = {"group_specific", GROUP_TERM}
    param._group_term_names = {
        "1|participant_id": "Intercept",
        GROUP_TERM: "theta",
    }
    param._group_terms_with_common = set(param._group_term_names)
    return {"v": param}

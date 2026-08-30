"""Tests for HSSM prior wrappers and regression-prior diagnostics."""

import logging

import bambi as bmb
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

import hssm
from hssm import Prior
from hssm.param.parameterization_check import (
    check_user_group_prior_compatibility,
    check_user_priors_for_location_overparameterization,
    find_disconnected_free_rvs,
    raise_prior_compatibility_errors,
)
from hssm.prior import (
    HDDM_SETTINGS_GROUP,
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
    pytest.param(
        hssm.Link(
            "custom_log",
            link=np.log,
            linkinv=np.exp,
            linkinv_backend=pt.exp,
        ),
        id="hssm-custom-log",
    ),
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


def _assert_prior_tree(prior, specification):
    """Recursively compare a generated hierarchical prior with its settings."""
    if isinstance(specification, dict) and "dist" in specification:
        expected = specification.copy()
        assert isinstance(prior, bmb.Prior)
        assert prior.name == expected.pop("dist")
        assert set(prior.args) == set(expected)
        for key, value in expected.items():
            _assert_prior_tree(prior.args[key], value)
        return
    assert prior == specification


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

        with pytest.raises(ValueError):
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

    @pytest.mark.parametrize("link", IDENTITY_LINKS)
    @pytest.mark.parametrize("param", HDDM_SETTINGS_GROUP)
    def test_hddm_group_only_identity_equivalence(self, link, param):
        """Use the HDDM group hierarchy for every spelling of identity."""
        prior = get_hddm_default_prior("group_intercept", param, None, link)

        _assert_prior_tree(prior, HDDM_SETTINGS_GROUP[param])

    @pytest.mark.parametrize("link", TRANSFORMED_LINKS)
    @pytest.mark.parametrize("param", HDDM_SETTINGS_GROUP)
    def test_hddm_group_only_transformed_link(self, link, param):
        """Use an unconstrained predictor-scale hierarchy after a transformed link."""
        prior = get_hddm_default_prior("group_intercept", param, None, link)

        _assert_prior_tree(
            prior,
            {
                "dist": "Normal",
                "mu": {"dist": "Normal", "mu": 0.0, "sigma": 0.25},
                "sigma": {"dist": "Weibull", "alpha": 1.5, "beta": 0.3},
            },
        )


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


def _custom_group_normal(name, mu, sigma, dims=None):
    """Build a custom Normal to test centered handling of ``dist``."""
    return pm.Normal(name, mu=mu, sigma=sigma, dims=dims)


class TestPriorIntegration:
    """Integration tests that build HSSM models and inspect warnings/graphs."""

    def test_noncentered_default_rejects_mu_hyperprior(self, cavanagh_test):
        """Reject a `mu` hyperprior before bambi can orphan it."""
        with pytest.raises(ValueError) as error:
            hssm.HSSM(
                data=cavanagh_test,
                model="ddm",
                include=_hierarchical_ddm_prior_with_mu_hyperprior(),
                p_outlier=0.0,
            )

        message = str(error.value)
        assert "cannot be represented faithfully" in message
        assert "1|participant_id" in message
        assert "disconnected node" in message
        assert "noncentered=False" in message

    def test_prior_settings_none_cannot_bypass_compatibility(self, cavanagh_test):
        """Validate explicit group priors even when safe generation is off."""
        with pytest.raises(ValueError, match="1\\|participant_id"):
            hssm.HSSM(
                data=cavanagh_test,
                model="ddm",
                include=_hierarchical_ddm_prior_with_mu_hyperprior(),
                prior_settings=None,
                p_outlier=0.0,
                process_initvals=False,
            )

    def test_numeric_group_prior_gets_hssm_preflight_error(self, cavanagh_test):
        """Explain that numeric regression priors do not fix coefficients."""
        include = _hierarchical_ddm_prior_with_mu_hyperprior()
        include[0]["prior"]["1|participant_id"] = 2.0

        with pytest.raises(ValueError, match="numeric values do not fix"):
            hssm.HSSM(
                data=cavanagh_test,
                model="ddm",
                include=include,
                prior_settings=None,
                p_outlier=0.0,
                process_initvals=False,
            )

    def test_supported_explicit_noncentered_prior_builds_clean_graph(
        self, cavanagh_test
    ):
        """A faithful explicit NC prior builds an offset without an orphan."""
        include = _hierarchical_ddm_prior_with_mu_hyperprior()
        include[0]["prior"]["1|participant_id"]["mu"] = 0.0

        model = hssm.HSSM(
            data=cavanagh_test,
            model="ddm",
            include=include,
            prior_settings=None,
            p_outlier=0.0,
            process_initvals=False,
        )

        names = set(model.pymc_model.named_vars)
        assert "v_1|participant_id_offset" in names
        assert "v_1|participant_id_mu" not in names
        assert find_disconnected_free_rvs(model.pymc_model) == []

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

    def test_centered_group_wildcard_drives_ridge_warning(self, cavanagh_test, caplog):
        """Expand a centered user group wildcard over Formulae terms."""
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
                prior_settings=None,
                p_outlier=0.0,
                noncentered=False,
                process_initvals=False,
            )

        targeted = [
            record.getMessage()
            for record in caplog.records
            if "User prior" in record.getMessage()
        ]
        assert len(targeted) == 1
        assert "theta|participant_id" in targeted[0]
        assert "non-identifiable" in targeted[0]
        assert "theta|participant_id" not in model.params["v"].prior
        assert isinstance(model.params["v"].prior["group_specific"], bmb.Prior)
        assert find_disconnected_free_rvs(model.pymc_model) == []

    def test_noncentered_group_wildcard_fails_before_build(self, cavanagh_test):
        """Expand a non-centered wildcard into an aggregated fidelity error."""
        group_wildcard = {
            "name": "Normal",
            "mu": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
            "sigma": {"name": "HalfNormal", "sigma": 0.5},
        }

        with pytest.raises(ValueError) as error:
            hssm.HSSM(
                data=cavanagh_test,
                model="ddm",
                include=[
                    {
                        "name": "v",
                        "formula": "v ~ 0 + theta + (0 + theta|participant_id)",
                        "prior": {"group_specific": group_wildcard},
                    }
                ],
                prior_settings="safe",
                p_outlier=0.0,
                noncentered=True,
                process_initvals=False,
            )

        message = str(error.value)
        assert "theta|participant_id" in message
        assert "disconnected node" in message

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
            (bmb.Prior("Normal", mu=0.0, sigma=0.5), 0.5, "no hierarchical"),
            (1.5, 0.5, "no top-level hyperprior"),
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
    def test_noncentered_compatibility_matrix(self, mu, sigma, reason_fragment):
        """Compatibility checks distinguish faithful and lossy NC paths."""
        params = _fake_params(_group_prior(mu=mu, sigma=sigma))

        flagged = check_user_group_prior_compatibility(params, True)

        if reason_fragment is None:
            assert flagged == []
        else:
            assert len(flagged) == 1
            assert reason_fragment in flagged[0].reason.lower()
            assert flagged[0].parameter == "v"
            assert flagged[0].term == GROUP_TERM

    @pytest.mark.parametrize(
        "mu",
        [
            pytest.param(None, id="absent"),
            pytest.param(0.0, id="scalar-zero"),
            pytest.param(np.array([0.0, 0.0]), id="vector-zero"),
        ],
    )
    def test_noncentered_plain_normal_accepts_supported_mu(self, mu):
        """Accept only absent or nonempty all-zero locations under NC."""
        kwargs = {"sigma": bmb.Prior("HalfNormal", sigma=0.5)}
        if mu is not None:
            kwargs["mu"] = mu
        prior = bmb.Prior("Normal", **kwargs)

        assert check_user_group_prior_compatibility(_fake_params(prior), True) == []

    @pytest.mark.parametrize("noncentered", [False, True])
    @pytest.mark.parametrize(
        ("prior", "reason_fragment"),
        [
            pytest.param(2.0, "not a bambi Prior", id="numeric"),
            pytest.param(np.array([0.0, 1.0]), "not a bambi Prior", id="numeric-array"),
            pytest.param(
                bmb.Prior("Normal", mu=0.0, sigma=1.0),
                "no top-level hyperprior",
                id="fixed-only-prior",
            ),
            pytest.param(
                Prior(
                    "Normal",
                    bounds=(-1.0, 1.0),
                    mu=0.0,
                    sigma=bmb.Prior("HalfNormal", sigma=0.5),
                ),
                "is truncated",
                id="outer-truncated",
            ),
            pytest.param(
                bmb.Prior(
                    "Normal",
                    mu=0.0,
                    sigma=Prior(
                        "HalfNormal",
                        bounds=(0.0, 2.0),
                        sigma=bmb.Prior("HalfNormal", sigma=0.5),
                    ),
                ),
                "outer prior.sigma is an HSSM truncated Prior",
                id="nested-truncated",
            ),
        ],
    )
    def test_incompatible_under_both_parameterizations(
        self, prior, reason_fragment, noncentered
    ):
        """Catch group priors bambi rejects whether centered or not."""
        flagged = check_user_group_prior_compatibility(_fake_params(prior), noncentered)

        assert len(flagged) == 1
        assert reason_fragment in flagged[0].reason

    @pytest.mark.parametrize(
        ("prior", "reason_fragment"),
        [
            pytest.param(
                bmb.Prior(
                    "Gamma",
                    mu=1.0,
                    sigma=bmb.Prior("HalfNormal", sigma=0.5),
                ),
                "uses 'Gamma'",
                id="non-normal",
            ),
            pytest.param(
                bmb.Prior(
                    "Normal",
                    dist=_custom_group_normal,
                    mu=0.0,
                    sigma=bmb.Prior("HalfNormal", sigma=0.5),
                ),
                "uses a custom distribution",
                id="custom",
            ),
            pytest.param(
                bmb.Prior(
                    "Normal",
                    mu=0.0,
                    sigma=bmb.Prior("HalfNormal", sigma=0.5),
                    initval=0.1,
                ),
                "includes argument(s) ['initval']",
                id="extra-fixed-arg",
            ),
            pytest.param(
                bmb.Prior(
                    "Normal",
                    mu=0.0,
                    sigma=bmb.Prior("HalfNormal", sigma=0.5),
                    tau=bmb.Prior("HalfNormal", sigma=0.5),
                ),
                "includes argument(s) ['tau']",
                id="extra-stochastic-arg",
            ),
            pytest.param(
                bmb.Prior(
                    "Normal",
                    mu=0.0,
                    sigma=bmb.Prior(
                        "HalfNormal",
                        sigma=bmb.Prior("Exponential", lam=1.0),
                    ),
                ),
                "outer prior.sigma uses 'HalfNormal'",
                id="nested-non-normal-hierarchy",
            ),
        ],
    )
    def test_noncentered_rejects_unsupported_prior_trees(self, prior, reason_fragment):
        """Mirror every lossy or failing branch in bambi's NC shortcut."""
        flagged = check_user_group_prior_compatibility(_fake_params(prior), True)

        assert len(flagged) == 1
        assert reason_fragment in flagged[0].reason
        assert "plain built-in Normal" in flagged[0].suggestion
        assert "noncentered=False" in flagged[0].suggestion

    def test_nested_prior_override_can_make_tree_compatible(self):
        """Honor a centered override on a nested hierarchical hyperprior."""
        prior = bmb.Prior(
            "Normal",
            mu=0.0,
            sigma=bmb.Prior(
                "HalfNormal",
                sigma=bmb.Prior("Exponential", lam=1.0),
                noncentered=False,
            ),
        )

        assert check_user_group_prior_compatibility(_fake_params(prior), True) == []

    @pytest.mark.parametrize(
        "prior",
        [
            pytest.param(
                bmb.Prior(
                    "Gamma",
                    mu=1.0,
                    sigma=bmb.Prior("HalfNormal", sigma=0.5),
                ),
                id="non-normal",
            ),
            pytest.param(
                bmb.Prior(
                    "Normal",
                    dist=_custom_group_normal,
                    mu=0.0,
                    sigma=bmb.Prior("HalfNormal", sigma=0.5),
                ),
                id="custom-normal",
            ),
        ],
    )
    def test_centered_accepts_hierarchical_family_and_custom_dist(self, prior):
        """Do not impose NC's outer-family shortcut on centered models."""
        assert check_user_group_prior_compatibility(_fake_params(prior), False) == []

    def test_empty_mu_vector_is_not_an_all_zero_location(self):
        """Avoid NumPy's vacuous all-zero result for an empty location."""
        prior = _group_prior(
            mu=np.array([]),
            sigma=bmb.Prior("HalfNormal", sigma=0.5),
        )

        flagged = check_user_group_prior_compatibility(_fake_params(prior), True)

        assert len(flagged) == 1
        assert "not fixed entirely to zero" in flagged[0].reason

    @pytest.mark.parametrize(
        ("prior_noncentered", "model_noncentered", "expect_compatibility_error"),
        [
            (False, True, False),
            (True, False, True),
            (None, False, False),
            (None, True, True),
            (None, None, False),
            (None, {"v": False}, False),
            (None, {"v": True}, True),
            (None, {"a": False}, True),
        ],
    )
    def test_effective_parameterization_precedence(
        self,
        prior_noncentered,
        model_noncentered,
        expect_compatibility_error,
    ):
        """Per-prior settings beat component dictionaries and model defaults."""
        prior = _group_prior(
            mu=bmb.Prior("Normal", mu=0.0, sigma=0.5),
            sigma=bmb.Prior("HalfNormal", sigma=0.5),
            noncentered=prior_noncentered,
        )
        params = _fake_params(prior)

        noncentered_mismatches = check_user_group_prior_compatibility(
            params, model_noncentered
        )
        ridge_mismatches = check_user_priors_for_location_overparameterization(
            params, model_noncentered
        )

        assert bool(noncentered_mismatches) is expect_compatibility_error
        assert bool(ridge_mismatches) is (not expect_compatibility_error)

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

        matched = check_user_group_prior_compatibility(
            _fake_params(prior, matched=True), True
        )
        unmatched = check_user_group_prior_compatibility(
            _fake_params(prior, matched=False), True
        )

        assert "common formula term 'theta'" in matched[0].suggestion
        assert "add the exact common formula term 'theta'" in unmatched[0].suggestion

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
            assert check_user_group_prior_compatibility(params, True) == []
            assert (
                check_user_priors_for_location_overparameterization(params, False) == []
            )

    @pytest.mark.parametrize(
        ("noncentered", "check"),
        [
            (True, check_user_group_prior_compatibility),
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

    def test_exact_none_suppresses_incompatible_wildcard(self):
        """Mirror bambi's exact-``None`` precedence over a group wildcard."""
        wildcard = _group_prior(
            mu=bmb.Prior("Normal", mu=0.0, sigma=0.5),
            sigma=bmb.Prior("HalfNormal", sigma=0.5),
        )
        params = _fake_wildcard_params(wildcard, None)

        flagged = check_user_group_prior_compatibility(params, True)

        assert [mismatch.term for mismatch in flagged] == ["1|participant_id"]

    def test_compatibility_collection_does_not_mutate_prior(self):
        """Explicit prior objects remain authoritative and untouched."""
        mu = bmb.Prior("Normal", mu=0.0, sigma=0.5)
        sigma = bmb.Prior("HalfNormal", sigma=0.5)
        prior = _group_prior(mu=mu, sigma=sigma, noncentered=True)
        original_args = prior.args.copy()

        check_user_group_prior_compatibility(_fake_params(prior), False)

        assert prior.args == original_args
        assert prior.args["mu"] is mu
        assert prior.args["sigma"] is sigma
        assert prior.noncentered is True

    def test_compatibility_errors_are_aggregated_and_sorted(self):
        """Raise one deterministic error covering every incompatible term."""
        params = {
            "v": _FakeRegressionParam(2.0, matched=True, user_specified=True),
            "a": _FakeRegressionParam(
                bmb.Prior("Normal", mu=0.0, sigma=1.0),
                matched=True,
                user_specified=True,
            ),
        }
        mismatches = check_user_group_prior_compatibility(params, True)

        with pytest.raises(ValueError) as error:
            raise_prior_compatibility_errors(mismatches)

        message = str(error.value)
        assert message.count("\n-") == 2
        assert message.index("parameter 'a'") < message.index("parameter 'v'")
        assert "no top-level hyperprior" in message
        assert "not a bambi Prior" in message


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

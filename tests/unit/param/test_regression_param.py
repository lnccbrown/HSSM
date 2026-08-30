import bambi as bmb
import numpy as np
import pytensor.tensor as pt
import pytest

import hssm
from hssm import Prior
from hssm.link import Link
from hssm.modelconfig import get_default_model_config
from hssm.param import UserParam
from hssm.param.regression_param import RegressionParam, _make_priors_recursive
from hssm.prior import HDDM_SETTINGS_GROUP, HSSM_SETTINGS_DISTRIBUTIONS

v_reg = UserParam(
    name="v",
    formula="v ~ 1 + x + y",
    prior={
        "Intercept": {"name": "Uniform", "lower": 0.0, "upper": 0.5},
        "x": dict(name="Uniform", lower=0.0, upper=1.0),
        "y": bmb.Prior("Uniform", lower=0.0, upper=1.0),
        "z": 0.1,
    },
)

v_reg_1 = UserParam(
    name="v",
    formula="v ~ 1 + x + y",
    prior={
        "Intercept": {"name": "Uniform", "lower": 0.0, "upper": 0.5},
        "x": dict(name="Uniform", lower=0.0, upper=1.0),
        "y": bmb.Prior("Uniform", lower=0.0, upper=1.0),
        "z": 0.1,
    },
    bounds=(0.0, 1.0),
)


def test_from_user_param():
    v = RegressionParam.from_user_param(v_reg)

    assert v.name == "v"
    assert v.formula == "v ~ 1 + x + y"
    assert isinstance(v.prior, dict)
    assert v.prior["Intercept"]["name"] == "Uniform"
    assert v.prior["Intercept"]["lower"] == 0.0
    assert v.prior["Intercept"]["upper"] == 0.5
    x = v.prior["x"]
    assert isinstance(x, dict)
    assert x["name"] == "Uniform"
    assert x["lower"] == 0.0
    assert x["upper"] == 1.0
    assert isinstance(v.prior["y"], bmb.Prior)
    assert isinstance(v.prior["z"], float)

    assert v.bounds is None
    assert v.user_param is v_reg


def test_from_defaults(caplog):
    param = RegressionParam.from_defaults(
        name="v", formula="v ~ 1 + x + y", bounds=(0.0, 1.0)
    )

    assert param.name == "v"
    assert param.formula == "v ~ 1 + x + y"
    assert param.prior is None
    assert param.bounds == (0.0, 1.0)
    assert param.user_param is None
    assert param.link is None

    param = RegressionParam.from_defaults(
        name="v",
        formula="v ~ 1 + x + y",
        bounds=(0.0, 1.0),
        link_settings="log_logit",
    )
    assert param.name == "v"
    assert param.formula == "v ~ 1 + x + y"
    assert param.prior is None
    assert param.bounds == (0.0, 1.0)
    assert param.user_param is None
    assert isinstance(param.link, Link)
    assert param.link.name == "gen_logit"
    assert param.link.bounds == (0.0, 1.0)

    param = RegressionParam.from_defaults(
        name="v",
        formula="v ~ 1 + x + y",
        bounds=(0.0, np.inf),
        link_settings="log_logit",
    )
    assert isinstance(param.link, str)
    assert param.link == "log"

    param = RegressionParam.from_defaults(
        name="v",
        formula="v ~ 1 + x + y",
        bounds=(-np.inf, np.inf),
        link_settings="log_logit",
    )
    assert isinstance(param.link, str)
    assert param.link == "identity"

    param = RegressionParam.from_defaults(
        name="v",
        formula="v ~ 1 + x + y",
        bounds=(-np.inf, 2.0),
        link_settings="log_logit",
    )

    assert caplog.records[-1].levelname == "WARNING"
    assert caplog.records[-1].message == (
        "The bounds for parameter v (-inf, 2.000000) seem "
        + "strange. Nothing is done to the link function. "
        + "Please check if they are correct."
    )


def test_fill_defaults():
    v = RegressionParam.from_user_param(v_reg)
    v.fill_defaults(bounds=(0.0, 2.0))

    assert v.name == "v"
    assert v.formula == "v ~ 1 + x + y"
    assert isinstance(v.prior, dict)
    assert v.prior["Intercept"]["name"] == "Uniform"
    assert v.prior["Intercept"]["lower"] == 0.0
    assert v.prior["Intercept"]["upper"] == 0.5
    assert isinstance(v.prior["x"], dict)
    assert v.prior["x"]["name"] == "Uniform"
    assert v.prior["x"]["lower"] == 0.0
    assert v.prior["x"]["upper"] == 1.0
    assert isinstance(v.prior["y"], bmb.Prior)
    assert isinstance(v.prior["z"], float)

    assert v.bounds == (0.0, 2.0)
    assert v.user_param is v_reg

    v = RegressionParam.from_user_param(v_reg_1)
    v.fill_defaults(bounds=(0.0, 2.0))
    assert v.bounds == (0.0, 1.0)

    with pytest.raises(
        ValueError,
        match="v is a regression parameter. It should not have a default prior.",
    ):
        v = RegressionParam.from_user_param(v_reg)
        v.fill_defaults(prior=0, bounds=(2.0, 0.0))

    v = RegressionParam(name="v", formula=None, prior=None, bounds=(0.0, 1.0))
    assert v.formula is None
    v.fill_defaults(formula="v ~ 1 + x + y")
    assert v.formula == "v ~ 1 + x + y"

    v = RegressionParam(name="v", formula="v ~ 1 + x + y", bounds=(0.0, 1.0))
    v.fill_defaults(formula="v ~ 1 + x")
    assert v.formula == "v ~ 1 + x + y"

    with pytest.raises(
        ValueError,
        match="Formula not specified for parameter v.",
    ):
        v = RegressionParam(name="v", formula=None, bounds=(0.0, 1.0))
        v.fill_defaults(bounds=(0.0, 1.0))

    v = RegressionParam(name="v", formula="v ~ 1 + x + y", bounds=(0.0, 1.0))
    v.fill_defaults(bounds=(0.0, 2.0), link_settings="log_logit")

    assert isinstance(v.link, Link)
    assert v.link.name == "gen_logit"
    assert v.link.bounds == (0.0, 1.0)


def test_validate():
    with pytest.raises(
        ValueError,
        match="Formula not specified for parameter v.",
    ):
        v = RegressionParam(name="v", formula=None, bounds=(0.0, 1.0))
        v.validate()

    v = RegressionParam(name="v", formula="v ~1 + x + y", bounds=(0.0, 1.0))
    v.validate()
    assert v.formula == "v ~ 1 + x + y"
    assert v.link == "identity"

    v = RegressionParam(
        name="v", formula="1 + x + y", bounds=(0.0, 1.0), link="log_logit"
    )
    v.validate()
    assert v.formula == "v ~ 1 + x + y"
    assert isinstance(v.link, Link)
    assert v.link.name == "gen_logit"
    assert v.link.bounds == (0.0, 1.0)


def test_prepare_formula_terms_caches_structural_names(cavanagh_test):
    """Cache normalized Formulae names for varied common and group terms."""
    param = RegressionParam(
        name="v",
        formula=(
            "v ~ 1 + theta * dbs + C(stim) + np.exp(theta) + "
            "(1 + theta * dbs + C(stim) + np.exp(theta) | participant_id) + "
            "(0 + theta | conf)"
        ),
    )

    design = param._prepare_formula_terms(cavanagh_test, {"np": np})

    assert design.common is not None
    assert design.group is not None
    assert param._common_term_names == {
        "Intercept",
        "theta",
        "dbs",
        "theta:dbs",
        "C(stim)",
        "np.exp(theta)",
    }
    assert param._group_term_names == {
        "1|participant_id": "Intercept",
        "theta|participant_id": "theta",
        "dbs|participant_id": "dbs",
        "theta:dbs|participant_id": "theta:dbs",
        "C(stim)|participant_id": "C(stim)",
        "np.exp(theta)|participant_id": "np.exp(theta)",
        "theta|conf": "theta",
    }
    assert param._group_terms_with_common == set(param._group_term_names)


@pytest.mark.parametrize(
    ("formula", "expected_matches"),
    [
        ("v ~ 1 + (0 + theta | participant_id)", set()),
        ("v ~ 0 + theta + (0 + theta | participant_id)", {"theta|participant_id"}),
        ("v ~ 1 + theta + (0 + dbs | participant_id)", set()),
        ("v ~ 0 + theta:dbs + (0 + dbs:theta | participant_id)", set()),
    ],
)
def test_prepare_formula_terms_matches_exact_expressions(
    cavanagh_test, formula, expected_matches
):
    """Match group expressions to common terms by exact Formulae names."""
    param = RegressionParam(name="v", formula=formula)

    param._prepare_formula_terms(cavanagh_test, {})

    assert param._group_terms_with_common == expected_matches


@pytest.mark.parametrize(
    ("formula", "matched_group_terms"),
    [
        (
            "v ~ 1 + theta + (1 + theta | participant_id)",
            {"1|participant_id", "theta|participant_id"},
        ),
        (
            "v ~ 0 + theta * dbs + (0 + theta * dbs | participant_id)",
            {
                "theta|participant_id",
                "dbs|participant_id",
                "theta:dbs|participant_id",
            },
        ),
        (
            "v ~ 0 + C(stim) + (0 + C(stim) | participant_id)",
            {"C(stim)|participant_id"},
        ),
        (
            "v ~ 0 + np.exp(theta) + (0 + np.exp(theta) | participant_id)",
            {"np.exp(theta)|participant_id"},
        ),
        (
            "v ~ 1 + theta + (1 + theta | participant_id) + (0 + theta | conf)",
            {"1|participant_id", "theta|participant_id", "theta|conf"},
        ),
    ],
)
def test_safe_priors_zero_center_structurally_matched_groups(
    cavanagh_test, formula, matched_group_terms
):
    """Give every exact common/group match a fixed zero group location."""
    param = RegressionParam(name="v", formula=formula, bounds=(-3.0, 3.0))

    param.make_safe_priors(cavanagh_test, {"np": np}, is_ddm=False)

    assert param._group_terms_with_common == matched_group_terms
    for group_term in matched_group_terms:
        prior = param.prior[group_term]
        assert isinstance(prior, bmb.Prior)
        assert prior.args["mu"] == 0.0


def test_safe_priors_preserve_explicit_matched_group_prior(cavanagh_test):
    """Never replace a user prior even when its group term matches a common term."""
    user_prior = bmb.Prior(
        "Normal",
        mu=bmb.Prior("Normal", mu=1.0, sigma=0.5),
        sigma=bmb.Prior("HalfNormal", sigma=0.5),
    )
    param = RegressionParam(
        name="v",
        formula="v ~ 1 + theta + (0 + theta | participant_id)",
        prior={"theta|participant_id": user_prior},
        bounds=(-3.0, 3.0),
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    assert "theta|participant_id" in param._group_terms_with_common
    assert param.prior["theta|participant_id"] is user_prior


def test_safe_priors_preserve_common_wildcard(cavanagh_test):
    """A user common wildcard prevents shadowing exact safe defaults."""
    wildcard = bmb.Prior("Laplace", mu=3.0, b=4.0)
    param = RegressionParam(
        name="v",
        formula="v ~ 1 + theta + (1 + theta | participant_id)",
        prior={"common": wildcard},
        bounds=(-3.0, 3.0),
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    assert param.prior["common"] is wildcard
    assert "Intercept" not in param.prior
    assert "theta" not in param.prior
    _check_group_prior_with_common(param.prior["1|participant_id"])
    _check_group_prior_with_common(param.prior["theta|participant_id"])


def test_safe_priors_preserve_group_specific_wildcard(cavanagh_test):
    """A user group wildcard prevents shadowing exact safe defaults."""
    wildcard = bmb.Prior(
        "Normal",
        mu=0.0,
        sigma=bmb.Prior("HalfNormal", sigma=0.75),
    )
    param = RegressionParam(
        name="v",
        formula="v ~ 1 + theta + (1 + theta | participant_id)",
        prior={"group_specific": wildcard},
        bounds=(-3.0, 3.0),
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    assert param.prior["group_specific"] is wildcard
    assert "1|participant_id" not in param.prior
    assert "theta|participant_id" not in param.prior
    assert isinstance(param.prior["Intercept"], Prior)
    assert isinstance(param.prior["theta"], bmb.Prior)


def test_safe_priors_exact_terms_take_precedence_over_wildcards(cavanagh_test):
    """Retain exact user terms alongside category-wide Bambi wildcards."""
    common_wildcard = bmb.Prior("Laplace", mu=3.0, b=4.0)
    group_wildcard = bmb.Prior(
        "Normal",
        mu=0.0,
        sigma=bmb.Prior("HalfNormal", sigma=0.75),
    )
    exact_common = bmb.Prior("Normal", mu=8.0, sigma=2.0)
    exact_group = bmb.Prior(
        "Normal",
        mu=0.0,
        sigma=bmb.Prior("HalfNormal", sigma=1.25),
    )
    specified = {
        "common": common_wildcard,
        "group_specific": group_wildcard,
        "theta": exact_common,
        "theta|participant_id": exact_group,
    }
    param = RegressionParam(
        name="v",
        formula="v ~ 1 + theta + (1 + theta | participant_id)",
        prior=specified,
        bounds=(-3.0, 3.0),
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    assert param.prior == specified
    assert param.prior["theta"] is exact_common
    assert param.prior["theta|participant_id"] is exact_group


def test_safe_priors_preserve_unmatched_group_location(cavanagh_test):
    """Retain the free-mean family while centering a generated group-only slope."""
    param = RegressionParam(
        name="v",
        formula="v ~ 1 + (0 + theta | participant_id)",
        bounds=(-3.0, 3.0),
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    assert param._group_terms_with_common == set()
    _check_group_prior(param.prior["theta|participant_id"])


@pytest.mark.parametrize(
    ("noncentered", "expect_warning"),
    [
        (True, True),
        (False, False),
        (None, False),
        ({"v": True}, True),
        ({"v": False}, False),
        ({"a": False}, True),
    ],
)
def test_safe_priors_warn_when_centering_overrides_model_setting(
    cavanagh_test, caplog, noncentered, expect_warning
):
    """Explain a generated term-level centered fallback only when it overrides."""
    param = RegressionParam(
        name="v",
        formula="v ~ 1 + (0 + theta | participant_id)",
        bounds=(-3.0, 3.0),
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=False, noncentered=noncentered)

    messages = [
        record.message
        for record in caplog.records
        if "generated location-bearing group-only term" in record.message
    ]
    assert bool(messages) is expect_warning
    if expect_warning:
        assert len(messages) == 1
        assert "parameter 'v'" in messages[0]
        assert "'theta|participant_id' (expression 'theta')" in messages[0]
        assert "noncentered=False" in messages[0]
        assert "response/parameter scale" in messages[0]
        assert "common formula term(s) ['theta']" in messages[0]


def test_safe_priors_warn_once_for_all_generated_group_locations(cavanagh_test, caplog):
    """Aggregate every generated unmatched term for a component into one warning."""
    param = RegressionParam(
        name="v",
        formula="v ~ 1 + (0 + theta + dbs | participant_id)",
        link="log",
        bounds=(-3.0, 3.0),
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=False, noncentered=True)

    messages = [
        record.message
        for record in caplog.records
        if "generated location-bearing group-only term" in record.message
    ]
    assert len(messages) == 1
    assert "theta|participant_id" in messages[0]
    assert "dbs|participant_id" in messages[0]
    assert "linear-predictor scale before the inverse link" in messages[0]
    assert "common formula term(s) ['dbs', 'theta']" in messages[0]
    assert param.prior["theta|participant_id"].noncentered is False
    assert param.prior["dbs|participant_id"].noncentered is False


def test_safe_priors_do_not_rewrite_explicit_unmatched_group_prior(cavanagh_test):
    """An explicit group-only prior remains authoritative and unchanged."""
    explicit = bmb.Prior(
        "Normal",
        mu=bmb.Prior("Normal", mu=1.0, sigma=0.5),
        sigma=bmb.Prior("HalfNormal", sigma=0.75),
        noncentered=True,
    )
    param = RegressionParam(
        name="v",
        formula="v ~ 1 + (0 + theta | participant_id)",
        prior={"theta|participant_id": explicit},
        bounds=(-3.0, 3.0),
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=False, noncentered=True)

    assert param.prior["theta|participant_id"] is explicit
    assert explicit.noncentered is True


def test_safe_priors_do_not_rewrite_unmatched_group_wildcard(cavanagh_test, caplog):
    """A group wildcard owns an unmatched term without a generated fallback."""
    wildcard = bmb.Prior(
        "Normal",
        mu=0.0,
        sigma=bmb.Prior("HalfNormal", sigma=0.75),
        noncentered=True,
    )
    param = RegressionParam(
        name="v",
        formula="v ~ 1 + (0 + theta | participant_id)",
        prior={"group_specific": wildcard},
        bounds=(-3.0, 3.0),
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=False, noncentered=True)

    assert param.prior["group_specific"] is wildcard
    assert "theta|participant_id" not in param.prior
    assert not any(
        "generated location-bearing group-only term" in record.message
        for record in caplog.records
    )


def _explicit_group_prior(scale=0.75):
    """Return a valid explicit hierarchical group prior for ownership tests."""
    return bmb.Prior(
        "Normal",
        mu=0.0,
        sigma=bmb.Prior("HalfNormal", sigma=scale),
    )


def test_safe_priors_reject_repeated_unmatched_group_location(cavanagh_test):
    """Do not choose a population owner between two grouping factors."""
    param = RegressionParam(
        name="v",
        formula=("v ~ 1 + (0 + theta | participant_id) + (0 + theta | conf)"),
        bounds=(-3.0, 3.0),
    )

    with pytest.raises(ValueError) as error:
        param.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    message = str(error.value)
    assert "parameter 'v'" in message
    assert "expression 'theta'" in message
    assert "theta|participant_id" in message
    assert "theta|conf" in message
    assert "common formula term(s) ['theta']" in message
    assert "non-None hierarchical explicit prior" in message
    assert "does not support numeric fixed coefficients" in message


def test_safe_priors_render_intercept_remedy_as_formula_one(cavanagh_test):
    """Suggest formula term ``1`` for an ambiguous Formulae Intercept."""
    param = RegressionParam(
        name="v",
        formula="v ~ 0 + (1 | participant_id) + (1 | conf)",
        bounds=(-3.0, 3.0),
    )

    with pytest.raises(ValueError) as error:
        param.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    message = str(error.value)
    assert "expression 'Intercept'" in message
    assert "1|participant_id" in message
    assert "1|conf" in message
    assert "common formula term(s) ['1']" in message


def test_safe_priors_reject_partially_explicit_group_ownership(cavanagh_test):
    """One exact prior does not assign every competing group location."""
    explicit = _explicit_group_prior()
    param = RegressionParam(
        name="v",
        formula=("v ~ 1 + (0 + theta | participant_id) + (0 + theta | conf)"),
        prior={"theta|participant_id": explicit},
        bounds=(-3.0, 3.0),
    )

    with pytest.raises(ValueError, match="theta\\|conf"):
        param.make_safe_priors(cavanagh_test, {}, is_ddm=False)


def test_safe_priors_allow_fully_explicit_group_ownership(cavanagh_test):
    """Preserve exact priors when users own every repeated group location."""
    participant_prior = _explicit_group_prior(0.5)
    conf_prior = _explicit_group_prior(1.25)
    param = RegressionParam(
        name="v",
        formula=("v ~ 1 + (0 + theta | participant_id) + (0 + theta | conf)"),
        prior={
            "theta|participant_id": participant_prior,
            "theta|conf": conf_prior,
        },
        bounds=(-3.0, 3.0),
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    assert param.prior["theta|participant_id"] is participant_prior
    assert param.prior["theta|conf"] is conf_prior


def test_safe_priors_allow_explicit_group_wildcard_ownership(cavanagh_test):
    """Treat a non-None wildcard as explicit coverage with exact precedence."""
    wildcard = _explicit_group_prior(0.5)
    exact = _explicit_group_prior(1.25)
    param = RegressionParam(
        name="v",
        formula=("v ~ 1 + (0 + theta | participant_id) + (0 + theta | conf)"),
        prior={"group_specific": wildcard, "theta|conf": exact},
        bounds=(-3.0, 3.0),
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    assert param.prior["group_specific"] is wildcard
    assert param.prior["theta|conf"] is exact
    assert "theta|participant_id" not in param.prior


@pytest.mark.parametrize(
    "prior",
    [
        {"group_specific": None},
        {"theta|participant_id": None, "theta|conf": _explicit_group_prior()},
    ],
    ids=["none-wildcard", "none-exact"],
)
def test_safe_priors_none_does_not_assign_group_location(cavanagh_test, prior):
    """Delegation to Bambi is not an explicit population-location choice."""
    param = RegressionParam(
        name="v",
        formula=("v ~ 1 + (0 + theta | participant_id) + (0 + theta | conf)"),
        prior=prior,
        bounds=(-3.0, 3.0),
    )

    with pytest.raises(ValueError, match="non-None hierarchical explicit prior"):
        param.make_safe_priors(cavanagh_test, {}, is_ddm=False)


def test_safe_priors_allow_repeated_matched_group_deviations(cavanagh_test):
    """One common effect supports zero-mean deviations for many factors."""
    param = RegressionParam(
        name="v",
        formula=("v ~ 1 + theta + (0 + theta | participant_id) + (0 + theta | conf)"),
        bounds=(-3.0, 3.0),
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    for term_name in ("theta|participant_id", "theta|conf"):
        assert term_name in param._group_terms_with_common
        assert param.prior[term_name].args["mu"] == 0.0


@pytest.mark.parametrize(
    "formula",
    [
        "v ~ 1 + (0 + theta | participant_id) + (0 + dbs | conf)",
        ("v ~ 1 + (0 + theta:dbs | participant_id) + (0 + dbs:theta | conf)"),
    ],
    ids=["different-expressions", "interaction-order-is-exact"],
)
def test_safe_priors_keep_distinct_unmatched_expressions_separate(
    cavanagh_test, formula
):
    """Do not infer symbolic equivalence or unrelated location collisions."""
    param = RegressionParam(name="v", formula=formula, bounds=(-3.0, 3.0))

    param.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    assert len(param._group_term_names) == 2
    assert set(param.prior).issuperset(param._group_term_names)


def test_safe_priors_aggregate_expanded_ambiguous_expressions(cavanagh_test):
    """Report every repeated term produced by a Formulae interaction expansion."""
    param = RegressionParam(
        name="v",
        formula=(
            "v ~ 1 + (0 + theta * dbs | participant_id) + (0 + theta * dbs | conf)"
        ),
        bounds=(-3.0, 3.0),
    )

    with pytest.raises(ValueError) as error:
        param.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    message = str(error.value)
    for expression in ("'theta'", "'dbs'", "'theta:dbs'"):
        assert f"expression {expression}" in message
    for term in (
        "theta|participant_id",
        "theta|conf",
        "dbs|participant_id",
        "dbs|conf",
        "theta:dbs|participant_id",
        "theta:dbs|conf",
    ):
        assert term in message


angle_config = get_default_model_config("angle")
angle_params = angle_config["list_params"]
angle_bounds = angle_config["likelihoods"]["approx_differentiable"]["bounds"].values()
param_and_bounds_angle = list(
    zip(angle_params, angle_bounds, [False] * len(angle_params))
)

ddm_config = get_default_model_config("full_ddm")
ddm_params = ddm_config["list_params"]
ddm_bounds = ddm_config["likelihoods"]["blackbox"]["bounds"].values()
param_and_bounds_ddm = list(zip(ddm_params, ddm_bounds, [True] * len(ddm_params)))


HDDM_LOCATION_PRIOR_CASES = [
    pytest.param(
        "v",
        (-np.inf, np.inf),
        "Normal",
        {"mu": 2.0, "sigma": 3.0},
        False,
        id="v",
    ),
    pytest.param(
        "a",
        (0.0, np.inf),
        "Gamma",
        {"mu": 1.5, "sigma": 0.75},
        True,
        id="a",
    ),
    pytest.param(
        "z",
        (0.0, 1.0),
        "Beta",
        {"alpha": 10.0, "beta": 10.0},
        True,
        id="z",
    ),
    pytest.param(
        "t",
        (0.0, np.inf),
        "Gamma",
        {"mu": 0.2, "sigma": 0.2},
        True,
        id="t",
    ),
    pytest.param(
        "sv",
        (0.0, np.inf),
        "HalfNormal",
        {"sigma": 2.0},
        True,
        id="sv",
    ),
    pytest.param(
        "sz",
        (0.0, np.inf),
        "HalfNormal",
        {"sigma": 0.5},
        True,
        id="sz",
    ),
    pytest.param(
        "st",
        (0.0, np.inf),
        "HalfNormal",
        {"sigma": 0.3},
        True,
        id="st",
    ),
    pytest.param(
        "p_outlier",
        None,
        "Beta",
        {"alpha": 5.0, "beta": 100.0},
        False,
        id="p-outlier",
    ),
]

IDENTITY_LINK_CASES = [
    pytest.param(None, id="omitted"),
    pytest.param("identity", id="string"),
    pytest.param(bmb.Link("identity"), id="bambi-object"),
    pytest.param(hssm.Link("identity"), id="hssm-object"),
]


def _assert_scalar_prior_contract(
    prior,
    *,
    name,
    args,
    bounds,
    is_truncated,
):
    """Assert the complete scalar prior contract, including hidden bound args."""
    assert isinstance(prior, Prior)
    assert prior.name == name
    assert prior.bounds == bounds
    assert prior.is_truncated is is_truncated
    assert (prior.dist is not None) is is_truncated

    effective_args = prior._args if prior.is_truncated else prior.args
    assert set(effective_args) == set(args)
    for key, expected in args.items():
        np.testing.assert_allclose(effective_args[key], expected)


def _assert_hierarchical_prior_contract(prior, specification):
    """Recursively compare a group prior against an HSSM settings tree."""
    if isinstance(specification, dict) and "dist" in specification:
        expected = specification.copy()
        assert isinstance(prior, bmb.Prior)
        assert prior.name == expected.pop("dist")
        assert set(prior.args) == set(expected)
        for key, value in expected.items():
            _assert_hierarchical_prior_contract(prior.args[key], value)
        return
    assert prior == specification


@pytest.mark.parametrize("link", IDENTITY_LINK_CASES)
@pytest.mark.parametrize(
    ("param_name", "bounds", "prior_name", "prior_args", "is_truncated"),
    HDDM_LOCATION_PRIOR_CASES,
)
def test_hddm_safe_common_intercept_uses_response_scale_prior_for_identity_links(
    cavanagh_test,
    param_name,
    bounds,
    prior_name,
    prior_args,
    is_truncated,
    link,
):
    """Route every identity spelling through RegressionParam before validation."""
    param = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 1 + theta",
        bounds=bounds,
        link=link,
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=True)
    param.process_prior()

    _assert_scalar_prior_contract(
        param.prior["Intercept"],
        name=prior_name,
        args=prior_args,
        bounds=bounds,
        is_truncated=is_truncated,
    )
    _assert_scalar_prior_contract(
        param.prior["theta"],
        name="Normal",
        args={"mu": 0.0, "sigma": 0.25},
        bounds=None,
        is_truncated=False,
    )
    assert getattr(param.link, "name", param.link) == "identity"


@pytest.mark.parametrize("link", IDENTITY_LINK_CASES)
@pytest.mark.parametrize(
    ("param_name", "bounds", "_prior_name", "_prior_args", "_is_truncated"),
    HDDM_LOCATION_PRIOR_CASES,
)
def test_hddm_safe_group_only_intercept_uses_identity_scale_hierarchy(
    cavanagh_test,
    param_name,
    bounds,
    _prior_name,
    _prior_args,
    _is_truncated,
    link,
):
    """Route every identity spelling through group-only safe-prior generation."""
    param = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 0 + (1 | participant_id)",
        bounds=bounds,
        link=link,
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=True, noncentered=False)
    param.process_prior()

    prior = param.prior["1|participant_id"]
    assert prior.noncentered is False
    _assert_hierarchical_prior_contract(prior, HDDM_SETTINGS_GROUP[param_name])
    assert getattr(param.link, "name", param.link) == "identity"


def test_hddm_safe_group_only_intercept_uses_preset_identity(cavanagh_test):
    """Treat identity selected by the log-logit preset like omitted identity."""
    param = RegressionParam.from_defaults(
        name="v",
        formula="v ~ 0 + (1 | participant_id)",
        bounds=(-np.inf, np.inf),
        link_settings="log_logit",
    )
    assert param.link == "identity"

    param.make_safe_priors(cavanagh_test, {}, is_ddm=True, noncentered=False)

    prior = param.prior["1|participant_id"]
    assert prior.noncentered is False
    _assert_hierarchical_prior_contract(prior, HDDM_SETTINGS_GROUP["v"])


@pytest.mark.parametrize(
    ("link", "expect_bounds_warning"),
    [
        pytest.param(None, True, id="omitted-identity"),
        pytest.param("identity", True, id="string-identity"),
        pytest.param(bmb.Link("identity"), True, id="bambi-identity"),
        pytest.param(hssm.Link("identity"), True, id="hssm-identity"),
        pytest.param("log", False, id="log"),
        pytest.param(
            hssm.Link("gen_logit", bounds=(0.0, 1.0)),
            False,
            id="gen-logit",
        ),
        pytest.param(
            hssm.Link(
                "custom_log",
                link=np.log,
                linkinv=np.exp,
                linkinv_backend=pt.exp,
            ),
            False,
            id="custom",
        ),
    ],
)
def test_group_only_bounds_warning_is_identity_specific(
    cavanagh_test, caplog, link, expect_bounds_warning
):
    """Do not warn about response bounds on a transformed predictor scale."""
    param = RegressionParam(
        name="v",
        formula="v ~ 0 + (1 | participant_id)",
        bounds=(0.0, 1.0),
        link=link,
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=False, noncentered=False)

    messages = [
        record.message for record in caplog.records if "HSSM #1269" in record.message
    ]
    assert bool(messages) is expect_bounds_warning
    if expect_bounds_warning:
        assert len(messages) == 1
        assert "Likelihood-level parameter bounds still apply" in messages[0]


@pytest.mark.parametrize(
    "bounds", [None, (-np.inf, np.inf)], ids=["no-bounds", "unbounded"]
)
def test_group_only_bounds_warning_requires_finite_bounds(
    cavanagh_test, caplog, bounds
):
    """Do not claim omitted bounds when no finite response bound exists."""
    param = RegressionParam(
        name="v",
        formula="v ~ 0 + (1 | participant_id)",
        bounds=bounds,
        link="identity",
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=True, noncentered=False)

    assert not any("HSSM #1269" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    "link",
    [*IDENTITY_LINK_CASES, pytest.param("log", id="log-control")],
)
def test_safe_priors_preserve_explicit_common_priors_across_links(cavanagh_test, link):
    """Never replace exact user intercept or slope priors based on link semantics."""
    intercept_prior = bmb.Prior("StudentT", nu=4.0, mu=1.0, sigma=0.5)
    slope_prior = bmb.Prior("Laplace", mu=-0.2, b=0.3)
    param = RegressionParam(
        name="a",
        formula="a ~ 1 + theta",
        prior={"Intercept": intercept_prior, "theta": slope_prior},
        bounds=(0.0, np.inf),
        link=link,
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=True)
    param.process_prior()

    assert set(param.prior) == {"Intercept", "theta"}
    assert param.prior["Intercept"] is intercept_prior
    assert param.prior["theta"] is slope_prior


@pytest.mark.parametrize(
    ("param_name", "bounds", "is_ddm"),
    param_and_bounds_angle,
)
def test_make_safe_priors(cavanagh_test, caplog, param_name, bounds, is_ddm):
    # Necessary for verifying the values of certain parameters of the priors
    hssm.set_floatX("float64")
    # The basic regression case, no group-specific terms
    param = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 1 + theta",
        bounds=bounds,
    )

    param.make_safe_priors(data=cavanagh_test, eval_env={}, is_ddm=is_ddm)

    assert param.prior is not None
    assert (intercept_prior := param.prior["Intercept"]) is not None
    assert (slope_prior := param.prior["theta"]) is not None

    assert isinstance(intercept_prior, Prior)
    assert intercept_prior.is_truncated
    assert intercept_prior.bounds == bounds
    assert intercept_prior.dist is not None
    lower, upper = intercept_prior.bounds
    _mu = intercept_prior._args["mu"]
    if isinstance(_mu, np.ndarray):
        assert _mu.item() == (lower + upper) / 2
    else:
        assert _mu == (lower + upper) / 2
    assert intercept_prior._args["sigma"] == 0.25

    assert isinstance(slope_prior, bmb.Prior)
    assert slope_prior.dist is None
    assert slope_prior.args["mu"] == 0.0
    assert slope_prior.args["sigma"] == 0.25

    unif_prior = {"name": "Uniform", "lower": 0.0, "upper": 1.0}
    set_prior = {
        "Intercept": unif_prior,
        "theta": unif_prior,
    }

    # Test that nothing is overwritten if the prior is already set
    param_with_prior = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 1 + theta",
        prior=set_prior,
        bounds=bounds,
    )

    param_with_prior.make_safe_priors(data=cavanagh_test, eval_env={}, is_ddm=False)
    assert param_with_prior.prior == set_prior

    # The regression case, with group-specific terms
    param_group = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 1 + theta + (1 + theta | participant_id)",
        bounds=bounds,
    )

    param_group.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    assert all(
        param in param_group.prior
        for param in [
            "Intercept",
            "theta",
            "1|participant_id",
            "theta|participant_id",
        ]
    )

    assert param_group.prior["Intercept"].is_truncated

    group_intercept_prior = param_group.prior["1|participant_id"]
    group_slope_prior = param_group.prior["theta|participant_id"]

    _check_group_prior_with_common(group_intercept_prior)
    _check_group_prior_with_common(group_slope_prior)

    param_no_common_intercept = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 0 + (1 + theta | participant_id)",
        bounds=bounds,
    )

    param_no_common_intercept.make_safe_priors(cavanagh_test, {}, is_ddm=False)

    assert any("limitation" in record.msg for record in caplog.records)
    assert "Intercept" not in param_no_common_intercept.prior
    group_intercept_prior = param_no_common_intercept.prior["1|participant_id"]
    group_slope_prior = param_no_common_intercept.prior["theta|participant_id"]

    _check_group_prior(group_intercept_prior)
    _check_group_prior(group_slope_prior)

    # Change back after testing
    hssm.set_floatX("float32")


def _check_group_prior(group_prior):
    assert isinstance(group_prior, bmb.Prior)
    assert group_prior.dist is None
    assert group_prior.name == "Normal"
    assert group_prior.noncentered is False

    mu = group_prior.args["mu"]
    sigma = group_prior.args["sigma"]

    assert isinstance(group_prior, bmb.Prior)
    assert mu.name == "Normal"
    assert mu.args["mu"] == 0.0
    assert mu.args["sigma"] == 0.25

    assert isinstance(group_prior, bmb.Prior)
    assert sigma.name == "Weibull"
    assert sigma.args["alpha"] == 1.5
    assert sigma.args["beta"] == 0.3


def _check_group_prior_with_common(group_prior):
    assert isinstance(group_prior, bmb.Prior)
    assert group_prior.dist is None
    assert group_prior.name == "Normal"
    assert group_prior.noncentered is None

    mu = group_prior.args["mu"]
    sigma = group_prior.args["sigma"]

    assert mu == 0.0

    assert isinstance(group_prior, bmb.Prior)
    assert sigma.name == "Weibull"
    assert sigma.args["alpha"] == 1.5
    assert sigma.args["beta"] == 0.3


v_mu = {"name": "Normal", "mu": 2.0, "sigma": 3.0}
v_sigma = {"name": "HalfNormal", "sigma": 2.0}
v_prior = {"name": "Normal", "mu": v_mu, "sigma": v_sigma}

a_mu = {"name": "Gamma", "mu": 1.5, "sigma": 0.75}
a_sigma = {"name": "HalfNormal", "sigma": 0.1}
a_prior = {"name": "Gamma", "mu": a_mu, "sigma": a_sigma}

# AF-TODO: Test below tests for equality between priors name
# and mu name .... z is a special case for this
# These tests probably need to be rewritten following a different
# approach that relies on default dictionaries from prior.py?

# Skipping z for now because I couldn't come up with an immediate
# solution
# z_mu = {"name": "Gamma", "mu": 10.0, "sigma": 10.0}
# z_sigma = {"name": "Gamma", "mu": 10.0, "sigma": 10.0}
# z_prior = {"name": "Beta", "alpha": z_mu, "beta": z_sigma}

t_mu = {"name": "Gamma", "mu": 0.2, "sigma": 0.2}
t_sigma = {"name": "HalfNormal", "sigma": 0.2}
t_prior = {"name": "Gamma", "mu": t_mu, "sigma": t_sigma}


@pytest.mark.parametrize(
    ("param_name", "mu", "prior"),
    [
        ("v", v_mu, v_prior),
        ("a", a_mu, a_prior),
        # ("z", z_mu, z_prior),
        ("t", t_mu, t_prior),
    ],
)
def test_make_safe_priors_ddm(cavanagh_test, caplog, param_name, mu, prior):
    # Necessary for verifying the values of certain parameters of the priors
    hssm.set_floatX("float64")

    bounds = (-10, 10)

    # The basic regression case, no group-specific terms
    param = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 1 + theta",
        bounds=bounds,  # invalid, just for testing
    )

    param.make_safe_priors(cavanagh_test, {}, is_ddm=True)

    intercept_prior = param.prior["Intercept"]
    slope_prior = param.prior["theta"]

    assert isinstance(intercept_prior, Prior)
    assert intercept_prior.bounds == bounds
    assert intercept_prior.dist is not None
    mu1 = mu.copy()
    print(f"{intercept_prior}=")
    print(f"{mu1}=")

    assert intercept_prior.name == mu1.pop("name")
    for key, val in mu1.items():
        val1 = intercept_prior._args[key]
        np.testing.assert_almost_equal(val1, val)

    assert isinstance(slope_prior, bmb.Prior)
    assert slope_prior.dist is None
    assert slope_prior.args["mu"] == 0.0
    assert slope_prior.args["sigma"] == 0.25

    # If prior is set, do not override
    unif_prior = {"name": "Uniform", "lower": 0.0, "upper": 1.0}
    set_prior = {
        "Intercept": unif_prior,
        "theta": unif_prior,
    }

    param_with_prior = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 1 + theta",
        bounds=bounds,
        prior=set_prior,
    )

    param_with_prior.make_safe_priors(cavanagh_test, {}, is_ddm=True)
    assert param_with_prior.prior == set_prior

    # The regression case, with group-specific terms
    param_group = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 1 + theta + (1 + theta | participant_id)",
        bounds=bounds,
    )

    param_group.make_safe_priors(cavanagh_test, {}, is_ddm=True)

    assert all(
        param in param_group.prior
        for param in [
            "Intercept",
            "theta",
            "1|participant_id",
            "theta|participant_id",
        ]
    )

    assert param_group.prior["Intercept"].is_truncated

    group_intercept_prior = param_group.prior["1|participant_id"]
    group_slope_prior = param_group.prior["theta|participant_id"]

    def _check_group_prior_intercept_ddm(group_prior, prior):
        assert isinstance(group_prior, bmb.Prior)
        assert group_prior.dist is None
        assert group_prior.noncentered is False
        prior1 = prior.copy()
        assert group_prior.name == prior1.pop("name")
        for key, val in prior1.items():
            hyperprior = group_prior.args[key]
            val1 = val.copy()
            assert hyperprior.name == val1.pop("name")
            for key2, val2 in val1.items():
                assert hyperprior.args[key2] == val2

    _check_group_prior_with_common(group_intercept_prior)
    _check_group_prior_with_common(group_slope_prior)

    param_no_common_intercept = RegressionParam(
        name=param_name,
        formula=f"{param_name} ~ 0 + (1 + theta | participant_id)",
        bounds=bounds,
    )

    param_no_common_intercept.make_safe_priors(cavanagh_test, {}, is_ddm=True)
    assert any("limitation" in record.msg for record in caplog.records)

    assert "Intercept" not in param_no_common_intercept.prior
    group_intercept_prior = param_no_common_intercept.prior["1|participant_id"]
    group_slope_prior = param_no_common_intercept.prior["theta|participant_id"]

    _check_group_prior_intercept_ddm(group_intercept_prior, prior)
    _check_group_prior(group_slope_prior)

    # Change back after testing
    hssm.set_floatX("float32")


def test_make_safe_priors_skips_hsgp_terms(cavanagh_test):
    # Bambi rejects any HSGP-term prior that is not None or a dict of
    # covariance-function priors, so the safe-prior machinery must not blanket
    # hsgp() terms with its scalar defaults (gh-624).
    hsgp_term = "hsgp(theta, m=8, c=2)"
    param = RegressionParam(
        name="v",
        formula=f"v ~ 1 + {hsgp_term}",
        bounds=(-3.0, 3.0),
    )

    param.make_safe_priors(data=cavanagh_test, eval_env={}, is_ddm=False)

    # Linear terms still receive safe defaults; the HSGP term receives nothing.
    assert isinstance(param.prior["Intercept"], bmb.Prior)
    assert hsgp_term not in param.prior
    # The term is still tracked for parameterization bookkeeping.
    assert hsgp_term in param.terms


def test__make_priors_recursive():
    test_dict = {
        "name": "Uniform",
        "lower": 0.1,
        "upper": {"name": "Normal", "mu": 0.5, "sigma": 0.1},
    }

    result_prior = _make_priors_recursive(test_dict)
    assert isinstance(result_prior, bmb.Prior)
    assert isinstance(result_prior.args["upper"], bmb.Prior)
    assert result_prior.args["upper"].name == "Normal"


def test_process_prior():
    prior1 = {
        "name": "Normal",
        "mu": {"name": "Normal", "mu": 0.0, "sigma": 1},
        "sigma": {"name": "HalfNormal", "sigma": 1},
    }
    prior2 = 0.4
    prior3 = bmb.Prior("Normal", mu=0.0, sigma=1.0)

    v = RegressionParam(
        name="v",
        formula="v ~ 1 + x + y",
        prior={
            "Intercept": prior1,
            "x": prior2,
            "y": prior3,
        },
    )

    v.process_prior()

    assert isinstance(v.prior["y"], bmb.Prior)

    assert isinstance(v.prior["Intercept"], bmb.Prior)
    assert v.prior["Intercept"].name == "Normal"
    assert isinstance(v.prior["Intercept"].args["mu"], bmb.Prior)
    assert v.prior["Intercept"].args["mu"].name == "Normal"
    assert v.prior["Intercept"].args["mu"].args["mu"] == 0.0
    assert v.prior["Intercept"].args["mu"].args["sigma"] == 1.0
    assert v.prior["Intercept"].args["sigma"].name == "HalfNormal"
    assert v.prior["Intercept"].args["sigma"].args["sigma"] == 1.0

    assert isinstance(v.prior["x"], float)
    assert v.prior["x"] == prior2

    assert isinstance(v.prior["y"], bmb.Prior)
    assert v.prior["y"] is prior3


def test_process_prior_hsgp_dict_passes_through():
    # Priors for hsgp() terms must reach bambi as dicts of covariance-function
    # priors, not be collapsed into a single bmb.Prior (gh-624). Inner
    # HSSM-style dict specs still convert; a top-level "name": "HSGP" marker
    # is dropped as convention residue.
    hsgp_term = "hsgp(theta, m=8, c=2)"
    v = RegressionParam(
        name="v",
        formula=f"v ~ 0 + {hsgp_term}",
        prior={
            hsgp_term: {
                "name": "HSGP",
                "sigma": bmb.Prior("Exponential", lam=3.0),
                "ell": {"name": "InverseGamma", "mu": 2.0, "sigma": 0.2},
            },
        },
    )

    v.process_prior()

    hsgp_prior = v.prior[hsgp_term]
    assert isinstance(hsgp_prior, dict)
    assert "name" not in hsgp_prior
    assert isinstance(hsgp_prior["sigma"], bmb.Prior)
    assert hsgp_prior["sigma"].name == "Exponential"
    assert isinstance(hsgp_prior["ell"], bmb.Prior)
    assert hsgp_prior["ell"].name == "InverseGamma"


def test_repr():
    prior1 = {
        "name": "Normal",
        "mu": {"name": "Normal", "mu": 0.0, "sigma": 1},
        "sigma": {"name": "HalfNormal", "sigma": 1},
    }
    prior2 = 0.4
    prior3 = bmb.Prior("Normal", mu=0.0, sigma=1.0)

    v = RegressionParam(
        name="v",
        formula="v ~ 1 + x + y",
        prior={
            "Intercept": prior1,
            "x": prior2,
            "y": prior3,
        },
    )

    v.process_prior()

    assert (
        repr(v) == "v:\n"
        "    Formula: v ~ 1 + x + y\n"
        "    Priors:\n"
        "        Intercept ~ Normal(mu: Normal(mu: 0.0, sigma: 1.0), "
        "sigma: HalfNormal(sigma: 1.0))\n"
        "        x: 0.4\n"
        "        y ~ Normal(mu: 0.0, sigma: 1.0)\n"
        "    Link: identity"
    )

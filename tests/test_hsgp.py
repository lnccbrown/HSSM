"""Model construction with hsgp() regression terms (gh-624).

Every prior shape a user can plausibly write for an HSGP term must construct:
bambi requires such priors to be None or a dict of covariance-function priors,
so HSSM must neither collapse the dict into a single bmb.Prior nor blanket the
term with a scalar safe default.
"""

import bambi as bmb
import numpy as np
import pytest

import hssm

HSGP_TERM = "hsgp(x, m=8, c=2)"


@pytest.fixture(scope="module")
def hsgp_data():
    rng = np.random.default_rng(42)
    data = hssm.simulate_data(
        model="ddm", theta=dict(v=0.5, a=1.5, z=0.5, t=0.3), size=300
    )
    data["x"] = rng.uniform(0.0, 6.0, len(data))
    data["grp"] = rng.choice(["g1", "g2"], len(data))
    data["participant_id"] = rng.integers(0, 5, len(data))
    return data


def make_model(hsgp_data, formula, prior=None, prior_settings="safe"):
    spec = {"name": "v", "formula": formula}
    if prior is not None:
        spec["prior"] = prior
    return hssm.HSSM(
        data=hsgp_data,
        model="ddm",
        include=[spec],
        loglik_kind="analytical",
        prior_settings=prior_settings,
    )


def cov_priors():
    return {
        "sigma": bmb.Prior("Exponential", lam=3.0),
        "ell": bmb.Prior("InverseGamma", mu=2.0, sigma=0.2),
    }


def get_hsgp_bambi_term(model, name):
    for component in model.model.components.values():
        term = getattr(component, "terms", {}).get(name)
        if term is not None:
            return term
    raise AssertionError(f"term {name!r} not found in the bambi model")


@pytest.mark.parametrize(
    "prior",
    [
        # HSSM-style dict with the "name" marker — the gh-624 reproduction.
        lambda: {HSGP_TERM: {"name": "HSGP", **cov_priors()}},
        # The dict shape bambi documents.
        lambda: {HSGP_TERM: cov_priors()},
        # No prior at all: bambi's automatic HSGP priors.
        lambda: None,
        # JSON-able inner specs convert to bmb.Prior.
        lambda: {
            HSGP_TERM: {
                "sigma": {"name": "Exponential", "lam": 3.0},
                "ell": bmb.Prior("InverseGamma", mu=2.0, sigma=0.2),
            }
        },
    ],
    ids=["hssm_style_dict", "bambi_native_dict", "no_prior", "inner_dict_spec"],
)
def test_hsgp_model_constructs(hsgp_data, prior):
    model = make_model(hsgp_data, f"v ~ 0 + {HSGP_TERM}", prior=prior())
    # The bambi-side term is an HSGPTerm whose prior must be None (automatic
    # priors) or the covariance-function dict — never a bmb.Prior.
    term = get_hsgp_bambi_term(model, HSGP_TERM)
    assert term.prior is None or isinstance(term.prior, dict)
    assert not isinstance(term.prior, bmb.Prior)


@pytest.mark.parametrize("prior_settings", ["safe", None])
def test_hsgp_constructs_without_safe_priors(hsgp_data, prior_settings):
    # The prior_settings=None path skips make_safe_priors entirely, so the
    # dict pass-through must carry the fix on its own.
    make_model(
        hsgp_data,
        f"v ~ 0 + {HSGP_TERM}",
        prior={HSGP_TERM: cov_priors()},
        prior_settings=prior_settings,
    )
    make_model(hsgp_data, f"v ~ 0 + {HSGP_TERM}", prior_settings=prior_settings)


def test_hsgp_by_group_variant(hsgp_data):
    term = "hsgp(x, by=grp, m=8, c=2)"
    make_model(hsgp_data, f"v ~ 0 + {term}", prior={term: cov_priors()})


def test_two_hsgp_terms(hsgp_data):
    other = "hsgp(x, m=6, c=1.5)"
    make_model(
        hsgp_data,
        f"v ~ 0 + {HSGP_TERM} + {other}",
        prior={HSGP_TERM: cov_priors(), other: cov_priors()},
    )


def test_hsgp_mixed_formula_keeps_safe_defaults(hsgp_data):
    # Linear terms alongside an HSGP term still receive safe default priors.
    model = make_model(
        hsgp_data,
        f"v ~ 1 + x + {HSGP_TERM}",
        prior={HSGP_TERM: cov_priors()},
    )
    v_prior = model.params["v"].prior
    assert isinstance(v_prior["Intercept"], bmb.Prior)
    assert isinstance(v_prior["x"], bmb.Prior)
    assert isinstance(v_prior[HSGP_TERM], dict)


def test_hsgp_dict_prior_is_inert_to_noncentering(hsgp_data, recwarn):
    # The parameterization checks iterate prior values expecting bmb.Prior;
    # a dict-valued HSGP prior must fall through their isinstance guards.
    # The group term must still be non-centered (its *_offset RV exists), and
    # no parameterization warning may fire about the HSGP entry.
    model = make_model(
        hsgp_data,
        f"v ~ 1 + (1|participant_id) + {HSGP_TERM}",
        prior={HSGP_TERM: cov_priors()},
    )
    offset_rvs = [
        rv.name for rv in model.pymc_model.free_RVs if rv.name.endswith("_offset")
    ]
    assert any(name.startswith("v_") for name in offset_rvs)
    assert not any("hsgp" in str(w.message).lower() for w in recwarn.list)

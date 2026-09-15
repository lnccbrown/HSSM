"""Out-of-sample prediction for unseen and unknown groups (#1316).

bambi 0.20 retired ``sample_new_groups``; the strategy is now read off the
grouping value in the new data instead:

* a missing value (``None``, ``np.nan``, ``pd.NA``) is an observation of
  *unknown identity* — it belongs to one of the fitted groups, so bambi draws a
  donor group per observation and per posterior draw;
* a non-missing value that was not observed during fitting is a *new group* —
  its coefficients are drawn from the population-level prior with the fitted
  hyperparameters, once per group, and shared by all of its observations.

These tests pin both paths as they surface through HSSM.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import hssm
from hssm.utils import _compute_likelihood_params

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

N_GROUPS = 3
FITTED_GROUPS = list(range(N_GROUPS))
NEW_GROUP = 99


@pytest.fixture(scope="module")
def hierarchical_ddm():
    """Fit a DDM with a participant-level intercept on `v` (a few draws)."""
    rng = np.random.default_rng(0)
    groups = np.repeat(FITTED_GROUPS, 30)
    v = 0.5 + rng.normal(0, 0.3, size=N_GROUPS)[groups]
    theta = np.column_stack(
        [v, np.full_like(v, 1.5), np.full_like(v, 0.5), np.full_like(v, 0.3)]
    )
    data = hssm.simulate_data(model="ddm", theta=theta, size=1)
    data["participant_id"] = groups

    model = hssm.HSSM(
        data=data,
        model="ddm",
        p_outlier=0,
        include=[
            {
                "name": "v",
                "formula": "v ~ 1 + (1|participant_id)",
                "link": "identity",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                    "1|participant_id": {
                        "name": "Normal",
                        "mu": 0.0,
                        "sigma": {"name": "HalfNormal", "sigma": 0.5},
                    },
                },
            }
        ],
    )
    model.sample(draws=8, tune=8, chains=1, cores=1, progressbar=False)
    return model


def _new_data(participant_id) -> pd.DataFrame:
    n = len(participant_id)
    return pd.DataFrame(
        {
            "rt": np.linspace(0.6, 1.2, n),
            "response": np.where(np.arange(n) % 2 == 0, 1.0, -1.0),
            "participant_id": participant_id,
        }
    )


def _fitted_group_values(model: hssm.HSSM) -> np.ndarray:
    """Return `v` for every fitted group, shaped (chain, draw, group)."""
    posterior = model.traces["posterior"]
    intercept = posterior["v_Intercept"].values[..., np.newaxis]
    return intercept + posterior["v_1|participant_id"].values


def _trialwise_v(model: hssm.HSSM, data: pd.DataFrame) -> np.ndarray:
    """Evaluate `v` on `data` without touching the model's stored traces."""
    dt = model.traces.copy(deep=True)
    return _compute_likelihood_params(model.model, dt, data=data)["v"].values


def test_fitted_group_uses_its_own_coefficients(hierarchical_ddm):
    """A seen grouping value still resolves to that group's fitted coefficient."""
    v = _trialwise_v(hierarchical_ddm, _new_data(participant_id=[0, 2]))
    fitted = _fitted_group_values(hierarchical_ddm)

    np.testing.assert_allclose(v[..., 0], fitted[..., 0])
    np.testing.assert_allclose(v[..., 1], fitted[..., 2])


@pytest.mark.parametrize(
    "participant_id",
    [
        pd.Series([np.nan] * 6 + [1], dtype="object"),
        pd.Series([None] * 6 + [1], dtype="object"),
        pd.Series([pd.NA] * 6 + [1], dtype="Int64"),
        # A plain float column, as produced by inserting NaN into an int column.
        np.array([np.nan] * 6 + [1.0]),
    ],
    ids=["nan-object", "None-object", "NA-Int64", "nan-float"],
)
def test_unknown_identity_samples_a_fitted_group(hierarchical_ddm, participant_id):
    """A missing grouping value borrows the coefficients of a fitted group.

    Every (draw, observation) value must coincide with one of the fitted groups
    for that draw, and the donor is re-drawn per observation, so two unknown
    observations do not have to agree.
    """
    v = _trialwise_v(hierarchical_ddm, _new_data(participant_id=participant_id))
    fitted = _fitted_group_values(hierarchical_ddm)

    unknown = v[..., :6]
    matches_a_group = np.isclose(
        unknown[..., np.newaxis], fitted[..., np.newaxis, :]
    ).any(axis=-1)
    assert matches_a_group.all()
    # The donor is drawn independently per observation; with six unknown
    # observations, eight draws and three groups they cannot all coincide.
    assert not np.allclose(unknown, unknown[..., :1])
    # A known observation in the same frame is unaffected.
    np.testing.assert_allclose(v[..., 6], fitted[..., 1])


def test_new_group_draws_shared_population_coefficients(hierarchical_ddm):
    """An unseen, non-missing grouping value is a new group.

    Its coefficient is generated from the population-level model once per group,
    so its observations agree with each other but with none of the fitted groups.
    """
    v = _trialwise_v(hierarchical_ddm, _new_data(participant_id=[NEW_GROUP] * 3))
    fitted = _fitted_group_values(hierarchical_ddm)

    np.testing.assert_allclose(v, np.broadcast_to(v[..., :1], v.shape))
    matches_a_group = np.isclose(v[..., :1], fitted).any(axis=-1)
    assert not matches_a_group.any()


def test_two_new_groups_get_distinct_coefficients(hierarchical_ddm):
    """Distinct unseen values are distinct new groups with their own draws."""
    v = _trialwise_v(
        hierarchical_ddm, _new_data(participant_id=[NEW_GROUP, NEW_GROUP, 100, 100])
    )

    np.testing.assert_allclose(v[..., 0], v[..., 1])
    np.testing.assert_allclose(v[..., 2], v[..., 3])
    assert not np.allclose(v[..., 0], v[..., 2])


def test_log_likelihood_out_of_sample_covers_both_strategies(hierarchical_ddm):
    """`log_likelihood(data=...)` accepts fitted, unknown and new groups together."""
    new_data = _new_data(
        participant_id=pd.Series([0, np.nan, NEW_GROUP, NEW_GROUP], dtype="object")
    )
    dt = hierarchical_ddm.log_likelihood(data=new_data, inplace=False)

    ll = dt["log_likelihood"]["rt,response"]
    assert ll.sizes["__obs__"] == len(new_data)
    assert np.isfinite(ll.values).all()
    # The stored traces keep their in-sample log likelihood.
    stored = hierarchical_ddm.traces["log_likelihood"]["rt,response"]
    assert stored.sizes["__obs__"] == len(hierarchical_ddm.data)


def test_sample_new_groups_is_not_forwarded(hierarchical_ddm):
    """HSSM no longer passes the retired `sample_new_groups` flag to bambi."""
    with pytest.warns(FutureWarning, match="sample_new_groups"):
        hierarchical_ddm.model.predict(
            hierarchical_ddm.traces.copy(deep=True),
            data=_new_data(participant_id=[0]),
            sample_new_groups=False,
        )
    # ... whereas HSSM's own out-of-sample path raises no FutureWarning at all.
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        _trialwise_v(hierarchical_ddm, _new_data(participant_id=[0]))

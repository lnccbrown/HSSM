"""Fast (non-sampling) tests for the RSSSM class and its components.

Covers: imports / re-exports, the three construction paths and config
variants, validation and v1 rejections, the spec resolvers and ordering
heuristic, the unbalanced-panel padding (and its exact-marginal property), and
the structural bit-for-bit equivalence of RSSSM's emission+forward to the
hand-written tutorial forward algorithm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest

import hssm
from hssm import RSSSM
from hssm.hmm import (
    AutoOrdering,
    DirichletInitialDistribution,
    NoOrdering,
    OrderByParam,
    RSSSMConfig,
    StickyDirichlet,
)
from hssm.hmm.likelihoods.forward import forward_log_marginal
from hssm.hmm.ordering import resolve_anchor
from hssm.hmm.specs import (
    DirichletConcentration,
    FixedInitialDistribution,
    resolve_initial_distribution,
    resolve_transition_prior,
)
from hssm.hmm.utils import pad_and_align_to_T_max

from .conftest import (
    TUTORIAL_P,
    build_tutorial_forward_marginal,
    make_panel,
    simulate_hmm_ddm_data,
)


# ---------------------------------------------------------------------------
# Imports / re-exports
# ---------------------------------------------------------------------------


def test_rsssm_reexported():
    assert hssm.RSSSM is RSSSM
    from hssm.hmm import RSSSM as RSSSM2

    assert RSSSM2 is RSSSM


# ---------------------------------------------------------------------------
# Construction & graph structure
# ---------------------------------------------------------------------------


def _logp_finite(model):
    ip = model.pymc_model.initial_point()
    return np.isfinite(model.pymc_model.compile_logp()(ip))


def test_build_single_participant(small_single_participant):
    m = RSSSM(
        data=small_single_participant,
        model="ddm",
        K=2,
        switching_params=["v"],
        v={"name": "Normal", "mu": 0.0, "sigma": 3.0},
    )
    rv_names = {rv.name for rv in m.pymc_model.free_RVs}
    assert rv_names == {"P", "v", "a", "z", "t"}
    assert [p.name for p in m.pymc_model.potentials] == ["hmm_loglik"]
    assert m.n_participants == 1
    assert m.n_trials == 60
    assert _logp_finite(m)


def test_build_k3_multi_switching():
    df = make_panel(3, 80)
    m = RSSSM(
        data=df,
        model="ddm",
        K=3,
        switching_params=["v", "a"],
        participant_col="participant_id",
    )
    # v is the (K,) anchor; a is also (K,).
    v_rv = m.pymc_model["v"]
    assert v_rv.type.shape == (3,)
    assert _logp_finite(m)


def test_build_no_pooling():
    df = make_panel(4, 50)
    m = RSSSM(
        data=df,
        model="ddm",
        K=2,
        switching_params=["v"],
        pooling="none",
        participant_col="participant_id",
    )
    # Per-participant switching param has shape (N, K).
    assert m.pymc_model["v"].type.shape == (4, 2)
    assert _logp_finite(m)


def test_estimable_pi0():
    df = make_panel(2, 60)
    m = RSSSM(
        data=df,
        model="ddm",
        K=2,
        switching_params=["v"],
        initial_distribution=DirichletInitialDistribution(alpha=[1, 1]),
        participant_col="participant_id",
    )
    assert "pi0" in {rv.name for rv in m.pymc_model.free_RVs}
    assert _logp_finite(m)


def test_fixed_per_regime_param_has_no_rv(small_single_participant):
    m = RSSSM(
        data=small_single_participant,
        model="ddm",
        K=2,
        switching_params=["v"],
        a=[0.8, 0.9],
    )
    assert "a" not in {rv.name for rv in m.pymc_model.free_RVs}
    assert _logp_finite(m)


def test_advanced_config_path(small_single_participant):
    cfg = RSSSMConfig(
        model_name="rsssm_ddm",
        model="ddm",
        K=2,
        switching_params=["v"],
        list_params=["v", "a", "z", "t"],
        bounds={
            "v": (-np.inf, np.inf),
            "a": (0.0, np.inf),
            "z": (0.0, 1.0),
            "t": (0.0, np.inf),
        },
        loglik_kind="analytical",
        transition_prior=DirichletConcentration(alpha=np.array([[20, 2], [2, 20]])),
        initial_distribution=FixedInitialDistribution(pi0=[0.6, 0.4]),
    )
    m = RSSSM(data=small_single_participant, model_config=cfg)
    assert _logp_finite(m)


def test_config_and_granular_args_conflict(small_single_participant):
    cfg = RSSSMConfig(
        model_name="rsssm_ddm",
        model="ddm",
        K=2,
        switching_params=["v"],
        list_params=["v", "a", "z", "t"],
        bounds={"v": (-np.inf, np.inf)},
        loglik_kind="analytical",
    )
    with pytest.raises(ValueError, match="not both"):
        RSSSM(data=small_single_participant, model="ddm", K=2, model_config=cfg)


# ---------------------------------------------------------------------------
# Validation & v1 rejections
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "rt": np.abs(rng.normal(size=20)) + 0.3,
            "response": rng.choice([-1.0, 1.0], 20),
            "participant_id": 0,
        }
    )


def test_validation_k_too_small(tiny_df):
    with pytest.raises(ValueError, match="K must be >= 2"):
        RSSSM(data=tiny_df, model="ddm", K=1, switching_params=["v"])


def test_validation_unknown_switching_param(tiny_df):
    with pytest.raises(ValueError, match="not parameters of model"):
        RSSSM(data=tiny_df, model="ddm", K=2, switching_params=["nope"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_outlier": 0.05},
        {"lapse": {"name": "Uniform", "lower": 0, "upper": 20}},
        {"missing_data": True},
        {"deadline": True},
    ],
)
def test_v1_rejections(tiny_df, kwargs):
    with pytest.raises(NotImplementedError):
        RSSSM(data=tiny_df, model="ddm", K=2, switching_params=["v"], **kwargs)


# ---------------------------------------------------------------------------
# Spec resolvers & ordering heuristic
# ---------------------------------------------------------------------------


def test_resolve_transition_prior_sticky_shorthand():
    spec = resolve_transition_prior({"sticky_diag": 30.0, "sticky_offdiag": 1.0})
    assert isinstance(spec, StickyDirichlet)
    alpha = spec.concentration(2)
    assert np.allclose(alpha, [[30, 1], [1, 30]])


def test_resolve_transition_prior_dirichlet_dict():
    spec = resolve_transition_prior({"name": "Dirichlet", "alpha": [[5, 1], [1, 5]]})
    assert isinstance(spec, DirichletConcentration)
    assert np.allclose(spec.concentration(2), [[5, 1], [1, 5]])


def test_resolve_initial_distribution_variants():
    assert isinstance(
        resolve_initial_distribution("uniform").__class__(),
        type(resolve_initial_distribution("uniform")),
    )
    fixed = resolve_initial_distribution([0.7, 0.3])
    assert isinstance(fixed, FixedInitialDistribution)
    est = resolve_initial_distribution({"name": "Dirichlet", "alpha": [1, 1]})
    assert isinstance(est, DirichletInitialDistribution)


def test_anchor_prefers_v():
    a = resolve_anchor(AutoOrdering(), ["a", "v", "z"])
    assert a.name == "v" and a.direction == "asc"


def test_anchor_single_param():
    a = resolve_anchor(AutoOrdering(), ["a"])
    assert a.name == "a"


def test_anchor_order_by_param_desc():
    a = resolve_anchor(OrderByParam(name="a", direction="desc"), ["v", "a"])
    assert a.name == "a" and a.direction == "desc"


def test_anchor_none_for_no_ordering():
    assert resolve_anchor(NoOrdering(), ["v"]) is None


def test_anchor_none_when_no_switching():
    assert resolve_anchor(AutoOrdering(), []) is None


# ---------------------------------------------------------------------------
# Unbalanced-panel padding
# ---------------------------------------------------------------------------


def test_padding_shapes_and_mask():
    df = pd.concat(
        [
            make_panel(1, 50, seed=1).assign(participant_id=0),
            make_panel(1, 30, seed=2).assign(participant_id=1),
        ],
        ignore_index=True,
    )
    data_padded, mask, n, t_max = pad_and_align_to_T_max(
        df, "participant_id", ["rt", "response"]
    )
    assert n == 2 and t_max == 50
    assert data_padded.shape == (2, 50, 2)
    assert mask[0].sum() == 50 and mask[1].sum() == 30
    # Padded rows duplicate the last real trial.
    assert np.allclose(data_padded[1, 30:], data_padded[1, 29])


def test_padding_requires_contiguous_participants():
    df = pd.DataFrame(
        {
            "rt": [0.5, 0.6, 0.7, 0.8],
            "response": [1, -1, 1, -1],
            "participant_id": [0, 1, 0, 1],
        }
    )
    with pytest.raises(ValueError, match="contiguous"):
        pad_and_align_to_T_max(df, "participant_id", ["rt", "response"])


def test_masked_marginal_equals_real_marginal():
    """Padding to T_max with an emission mask leaves the marginal unchanged."""
    rng = np.random.default_rng(3)
    K, T, Tm = 2, 30, 50
    em = rng.normal(size=(1, T, K)).astype("float32")
    log_P = np.log(np.array([[0.9, 0.1], [0.2, 0.8]], dtype="float32"))
    log_pi0 = np.log(np.ones(K, dtype="float32") / K)

    real = forward_log_marginal(
        pt.as_tensor_variable(em),
        pt.as_tensor_variable(log_P),
        pt.as_tensor_variable(log_pi0),
        pt.ones((1, T)),
    ).eval()

    em_pad = np.concatenate(
        [em, np.tile(em[:, -1:, :], (1, Tm - T, 1))], axis=1
    ).astype("float32")
    mask = np.zeros((1, Tm), dtype="float32")
    mask[:, :T] = 1.0
    padded = forward_log_marginal(
        pt.as_tensor_variable(em_pad),
        pt.as_tensor_variable(log_P),
        pt.as_tensor_variable(log_pi0),
        pt.as_tensor_variable(mask),
    ).eval()

    assert abs(float(real) - float(padded)) < 1e-4


def test_missing_participant_col_synthesised(small_single_participant):
    m = RSSSM(data=small_single_participant, model="ddm", K=2, switching_params=["v"])
    assert m.n_participants == 1


@pytest.mark.parametrize(
    "switching_params",
    [
        ["v"],  # default: v is the ordered anchor
        ["v", "t"],  # t is a non-anchor switching param
        ["t"],  # t is the ordered anchor (the bypass case)
    ],
)
def test_finite_gradient_at_init(small_single_participant, switching_params):
    """The start must give finite gradients across switching configurations.

    The non-decision time `t` must be seeded below the minimum RT so the start
    does not land in the SSM's invalid region (`rt < t`), where the gradient is
    NaN — which otherwise makes the PyMC NUTS sampler diverge on every draw.
    This must hold whether `t` is a non-anchor switching param (seeded via
    `_param_initval`) or the ordered *anchor* (seeded via `_ascending_initval`),
    since the anchor path bypasses `_param_initval`.
    """
    m = RSSSM(
        data=small_single_participant,
        model="ddm",
        K=2,
        switching_params=switching_params,
    )
    ip = m.pymc_model.initial_point()
    grad = m.pymc_model.compile_dlogp()(ip)
    assert np.all(np.isfinite(grad))


def test_v_anchor_initval_unchanged(small_single_participant):
    """The `v` anchor's seeded grid is unchanged by the safe-seed centering.

    `v` is unbounded with safe seed 0, so centering reproduces the historical
    `linspace(-2, 2, K)` exactly — guarding against a regression in the
    well-tested default anchor while the fix targets `t`.
    """
    from hssm.hmm.rsssm import _ascending_initval

    asc = _ascending_initval(2, bounds=None, center=0.0)
    np.testing.assert_allclose(asc, np.array([-2.0, 2.0]))


# ---------------------------------------------------------------------------
# Forward algorithm correctness (the definitive checks)
# ---------------------------------------------------------------------------


def test_forward_marginal_matches_brute_force_enumeration():
    """The forward marginal equals an exact sum over all K**T regime paths."""
    import itertools

    rng = np.random.default_rng(0)
    K, T = 3, 7  # 3**7 = 2187 enumerable paths
    log_em = rng.normal(size=(1, T, K))
    praw = rng.uniform(0.1, 1.0, size=(K, K))
    P = praw / praw.sum(1, keepdims=True)
    pi0raw = rng.uniform(0.1, 1.0, size=K)
    pi0 = pi0raw / pi0raw.sum()
    log_P, log_pi0 = np.log(P), np.log(pi0)

    total = -np.inf
    for path in itertools.product(range(K), repeat=T):
        lp = log_pi0[path[0]] + log_em[0, 0, path[0]]
        for t in range(1, T):
            lp += log_P[path[t - 1], path[t]] + log_em[0, t, path[t]]
        total = np.logaddexp(total, lp)

    fwd = float(
        forward_log_marginal(
            pt.as_tensor_variable(log_em),
            pt.as_tensor_variable(log_P),
            pt.as_tensor_variable(log_pi0),
            pt.ones((1, T)),
        ).eval()
    )
    assert abs(total - fwd) < 1e-9


def test_joint_marginal_is_sum_over_participants():
    """sum_n L_n: the joint marginal equals the per-participant marginals summed."""
    rng = np.random.default_rng(1)
    K, T, N = 2, 6, 4
    log_em = rng.normal(size=(N, T, K))
    log_P = np.log(np.array([[0.8, 0.2], [0.3, 0.7]]))
    log_pi0 = np.log(np.ones(K) / K)

    joint = float(
        forward_log_marginal(
            pt.as_tensor_variable(log_em),
            pt.as_tensor_variable(log_P),
            pt.as_tensor_variable(log_pi0),
            pt.ones((N, T)),
        ).eval()
    )
    per = sum(
        float(
            forward_log_marginal(
                pt.as_tensor_variable(log_em[n : n + 1]),
                pt.as_tensor_variable(log_P),
                pt.as_tensor_variable(log_pi0),
                pt.ones((1, T)),
            ).eval()
        )
        for n in range(N)
    )
    assert abs(joint - per) < 1e-9


def _np_logsumexp(x, axis):
    """Independent numpy log-sum-exp (reference, no pytensor)."""
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)), axis)


@pytest.mark.parametrize("T", [400, 500, 1000])
def test_forward_gradient_finite_for_long_sequences(T):
    """The scaled forward keeps gradients finite (and correct) for long panels.

    The textbook un-normalised forward recursion drifts ~linearly with the
    sequence length, so its reverse-mode gradient becomes NaN past ~400 trials
    (in *both* the C/PyTensor and JAX backends) even though the marginal value
    stays finite — which silently breaks NUTS on realistic long panels.  The
    scaled (normalised) recursion must stay finite.

    The emission has ``theta`` added to every trial's log-density, so the
    marginal is ``Z(theta) = Z(0) + T * theta`` exactly; the gradient w.r.t.
    ``theta`` must therefore equal ``T`` — catching a finite-but-wrong gradient,
    not merely a NaN.  Emissions are drawn in the realistic SSM range ``[-18,
    -1]`` so the un-normalised recursion would genuinely overflow.
    """
    import pytensor

    rng = np.random.default_rng(0)
    K = 3
    base = rng.uniform(-18.0, -1.0, size=(1, T, K))
    praw = rng.uniform(0.1, 1.0, size=(K, K))
    P = praw / praw.sum(1, keepdims=True)
    pi0 = np.ones(K) / K

    theta = pt.scalar("theta")
    log_em = pt.as_tensor_variable(base) + theta  # gradient flows through every step
    marginal = forward_log_marginal(
        log_em,
        pt.as_tensor_variable(np.log(P)),
        pt.as_tensor_variable(np.log(pi0)),
        pt.ones((1, T)),
    )
    grad_fn = pytensor.function([theta], pt.grad(marginal, theta))
    g = float(grad_fn(0.0))
    assert np.isfinite(g)
    assert abs(g - T) < 1e-6  # exact: d/dtheta (Z0 + T*theta) = T


def test_forward_marginal_value_matches_numpy_reference_long():
    """Scaled forward *value* matches an independent numpy log-forward at T=500.

    The brute-force enumeration test pins correctness at T=7; this pins the
    value for a long sequence (where the rewrite's normalisation could in
    principle drift) against a stable numpy log-space forward.
    """
    rng = np.random.default_rng(3)
    K, T = 3, 500
    log_em = rng.uniform(-18.0, -1.0, size=(1, T, K))
    praw = rng.uniform(0.1, 1.0, size=(K, K))
    P = praw / praw.sum(1, keepdims=True)
    pi0 = np.ones(K) / K
    log_P, log_pi0 = np.log(P), np.log(pi0)

    a = log_pi0 + log_em[0, 0]
    for t in range(1, T):
        a = _np_logsumexp(a[:, None] + log_P, axis=0) + log_em[0, t]
    ref = float(_np_logsumexp(a, axis=0))

    got = float(
        forward_log_marginal(
            pt.as_tensor_variable(log_em),
            pt.as_tensor_variable(log_P),
            pt.as_tensor_variable(log_pi0),
            pt.ones((1, T)),
        ).eval()
    )
    assert abs(ref - got) < 1e-6


# ---------------------------------------------------------------------------
# Structural bit-for-bit equivalence to the tutorial
# ---------------------------------------------------------------------------


def test_forward_marginal_matches_tutorial():
    """RSSSM's emission+forward equals the hand-written tutorial marginal."""
    data, _ = simulate_hmm_ddm_data(
        80,
        {
            0: {"v": 1.5, "a": 0.8, "z": 0.5, "t": 0.3},
            1: {"v": 0.2, "a": 0.8, "z": 0.5, "t": 0.3},
        },
        TUTORIAL_P,
        np.array([0.8, 0.2]),
        seed=7,
    )
    v = np.array([0.2, 1.5])
    a, z, t = 0.8, 0.5, 0.3
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    K = 2

    tutorial = build_tutorial_forward_marginal(data, v, a, z, t, P, K)

    from hssm.hmm.likelihoods.builder import make_hmm_logp_op
    from hssm.hmm.likelihoods.emissions import resolve_emission_dist

    dist_class = resolve_emission_dist("ddm", "analytical", "pytensor")
    with pm.Model():
        builder = make_hmm_logp_op(
            dist_class=dist_class,
            data_padded=data[None, :, :],
            mask=np.ones((1, data.shape[0])),
            K=K,
            n_participants=1,
            n_trials=data.shape[0],
            regime_params={"v"},
            pooling="full",
        )
        pot = builder(
            {
                "v": pt.as_tensor_variable(v.astype("float32")),
                "a": pt.as_tensor_variable(np.float32(a)),
                "z": pt.as_tensor_variable(np.float32(z)),
                "t": pt.as_tensor_variable(np.float32(t)),
            },
            pt.log(pt.as_tensor_variable(P.astype("float32"))),
            pt.log(pt.ones(K) / K),
        )
    rsssm_val = float(pot.eval())

    assert abs(tutorial - rsssm_val) < 1e-3

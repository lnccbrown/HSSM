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
    UniformInitialDistribution,
    resolve_initial_distribution,
    resolve_transition_prior,
)
from hssm.hmm.utils import pad_and_align_to_T_max
from hssm.modelconfig import get_default_model_config

from .conftest import (
    TUTORIAL_P,
    build_tutorial_forward_marginal,
    eval_at_point,
    make_panel,
    simulate_hmm_data,
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


def test_no_pooling_emission_aligns_participants_to_rows():
    """Each participant's parameters must land on that participant's own rows.

    The panel is flattened participant-major (row = p * T + t), so each
    participant's value is expanded with `repeat` (not `tile`).  Feeding
    *distinct* per-participant values is what discriminates the two: with
    identical values across participants (or with only shape/finiteness
    assertions) the wrong expansion is invisible.
    """
    from hssm.hmm import ffbs

    n, t_each = 3, 30
    panel = make_panel(n, t_each)
    m = RSSSM(
        data=panel,
        model="ddm",
        K=2,
        switching_params=["v"],
        pooling="none",
        participant_col="participant_id",
    )
    values = {
        "v": np.array([[-1.5, 0.5], [0.2, 1.2], [-0.4, 2.0]]),  # (N, K)
        "a": np.array([0.8, 1.2, 1.5]),
        "z": np.array([0.4, 0.5, 0.6]),
        "t": np.array([0.05, 0.10, 0.15]),
    }
    fn, order = ffbs._compile_emission_fn(m)
    emission = np.asarray(fn(*[values[name] for name in order]))  # (N, T, K)

    # Reference: the same participant fitted on its own, fully pooled.
    for p in range(n):
        sub = panel.loc[panel["participant_id"] == p, ["rt", "response"]]
        m_p = RSSSM(
            data=sub.reset_index(drop=True), model="ddm", K=2, switching_params=["v"]
        )
        fn_p, order_p = ffbs._compile_emission_fn(m_p)
        vals_p = {name: values[name][p] for name in order_p}
        expected = np.asarray(fn_p(*[vals_p[name] for name in order_p]))  # (1, T, K)
        np.testing.assert_allclose(emission[p], expected[0], rtol=1e-4, atol=1e-4)


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


def _advanced_config() -> RSSSMConfig:
    """A complete, hand-built config for the advanced `model_config=` path."""
    return RSSSMConfig(
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


def test_advanced_config_path(small_single_participant):
    m = RSSSM(data=small_single_participant, model_config=_advanced_config())
    assert _logp_finite(m)


def test_model_as_config_object_builds(small_single_participant):
    """`model=<BaseModelConfig>` resolves the emission from the wrapped SSM.

    The RSSSM config's own `model_name` is the prefixed `"rsssm_ddm"`, which is
    not a supported SSM, so the emission has to be resolved from the underlying
    model identifier.
    """
    from hssm.config import Config

    m = RSSSM(
        data=small_single_participant,
        model=Config.from_defaults("ddm", "analytical"),
        K=2,
        switching_params=["v"],
    )
    assert {rv.name for rv in m.pymc_model.free_RVs} == {"P", "v", "a", "z", "t"}
    assert _logp_finite(m)


def test_model_as_config_object_uses_registry_choices(small_single_participant):
    """A wrapped config that never set `choices` still accepts `{-1, 1}` data.

    `BaseModelConfig.choices` defaults to the generic `(0, 1)`, so trusting the
    config's own field rejected standard DDM responses. The coding must come
    from the registry entry the emission is rebuilt from — which is also what
    `hssm.HSSM` does for a registered model string.
    """
    from hssm.config import Config

    cfg = Config(
        model_name="ddm",
        loglik_kind="analytical",
        list_params=["v", "a", "z", "t"],
        bounds={"v": (-np.inf, np.inf), "a": (0.0, np.inf), "z": (0.0, 1.0)},
    )
    assert cfg.choices == (0, 1)  # the class default the bug trusted

    m = RSSSM(
        data=small_single_participant,
        model=cfg,
        K=2,
        switching_params=["v"],
    )
    assert m.choices == [-1, 1]
    assert _logp_finite(m)


def test_model_as_config_rejects_custom_loglik(small_single_participant):
    """A custom `loglik` on a `model=` config is rejected, not silently dropped.

    `resolve_emission_dist` re-enters the registry, so the custom likelihood was
    discarded and the model computed the stock one — bit-identical logp to plain
    `model="ddm"`, i.e. silently wrong science.
    """
    from hssm.config import Config

    def my_loglik(data, *args):  # pragma: no cover - never called
        raise AssertionError("must not be reached")

    cfg = Config(
        model_name="ddm",
        loglik_kind="analytical",
        loglik=my_loglik,
        list_params=["v", "a", "z", "t"],
    )
    with pytest.raises(NotImplementedError, match="custom `loglik`"):
        RSSSM(
            data=small_single_participant,
            model=cfg,
            K=2,
            switching_params=["v"],
        )


@pytest.mark.parametrize("model_name", ["ddm", "angle"])
def test_model_as_config_accepts_stock_lan_loglik(small_single_participant, model_name):
    """A stock LAN config is not mistaken for a custom `loglik`.

    The registered LAN `loglik` is an `.onnx` *path string*, and
    `get_default_model_config` re-imports the config module on every call, so an
    identity check rejected every unmodified `Config.from_defaults(...,
    "approx_differentiable")` — 8 of the 20 registered (model, kind) entries.
    """
    from hssm.config import Config

    cfg = Config.from_defaults(model_name, "approx_differentiable")
    assert isinstance(cfg.loglik, str)  # the case identity comparison missed
    assert RSSSM._is_registry_loglik(cfg, "approx_differentiable")


def test_model_as_config_rejects_loglik_from_another_kind(small_single_participant):
    """Borrowing another kind's loglik is a custom loglik for the resolved kind.

    `loglik=logp_ddm` under `loglik_kind="approx_differentiable"` would be
    silently rebuilt as the stock LAN emission, so it must be rejected.
    """
    from hssm.config import Config

    analytical = Config.from_defaults("ddm", "analytical").loglik
    cfg = Config(
        model_name="ddm",
        loglik_kind="approx_differentiable",
        loglik=analytical,
        list_params=["v", "a", "z", "t"],
    )
    assert not RSSSM._is_registry_loglik(cfg, "approx_differentiable")


def test_config_path_fills_list_params_and_bounds(small_single_participant):
    """The `model_config=` path derives `list_params`/`bounds` from the SSM.

    Design §6.2's own example leaves both unset; without the derivation it died
    with "list_params must be populated from the SSM model before validation".
    """
    cfg = RSSSMConfig(
        model_name="rsssm_ddm",
        model="ddm",
        K=3,
        switching_params=["v", "a"],
        loglik_kind="analytical",
        transition_prior=DirichletConcentration(
            alpha=np.array([[30, 2, 2], [2, 30, 2], [2, 2, 30]])
        ),
        initial_distribution=FixedInitialDistribution(pi0=[0.5, 0.3, 0.2]),
        ordering=OrderByParam(name="v", direction="desc"),
    )
    assert cfg.list_params is None

    m = RSSSM(data=small_single_participant, model_config=cfg)
    assert m.list_params == ["v", "a", "z", "t"]
    assert m.bounds["z"] == (0.0, 1.0)
    assert _logp_finite(m)


def test_config_path_keeps_user_bounds(small_single_participant):
    """Deriving the SSM metadata must not clobber bounds the user did set."""
    cfg = RSSSMConfig(
        model_name="rsssm_ddm",
        model="ddm",
        K=2,
        switching_params=["v"],
        loglik_kind="analytical",
        bounds={"z": (0.4, 0.6)},
    )
    m = RSSSM(data=small_single_participant, model_config=cfg)
    assert m.bounds["z"] == (0.4, 0.6)
    assert m.bounds["a"] == (0.0, np.inf)  # still filled from the registry


@pytest.mark.parametrize(
    "extra",
    [
        {"pooling": "none"},
        {"ordering": "none"},
        {"p_outlier": 0.5},
        {"transition_prior": {"sticky_diag": 5.0}},
        {"initial_distribution": [0.9, 0.1]},
        {"loglik_kind": "approx_differentiable"},
        {"a": [0.1, 5.0]},
    ],
)
def test_config_path_rejects_all_granular_kwargs(small_single_participant, extra):
    """Every granular arg is carried by the config, so passing one is an error.

    Silently dropping it (the previous behaviour) built a model that disagreed
    with the call — e.g. `pooling="none"` still fully pooled.
    """
    with pytest.raises(ValueError, match="not both"):
        RSSSM(data=small_single_participant, model_config=_advanced_config(), **extra)


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


def _logp_at(model, point):
    """Total logp of `model` at an explicit transformed point."""
    return float(model.pymc_model.compile_logp()(point))


def _assert_same_model(granular, config):
    """The two construction paths must yield the *same* model, not just a model.

    Compared away from the shared initial point, so agreement cannot come from
    both merely starting in the same place.
    """
    ip = granular.pymc_model.initial_point()
    assert set(ip) == set(config.pymc_model.initial_point())
    point = {k: np.asarray(v) + 0.05 for k, v in ip.items()}
    lp = _logp_at(granular, point)
    assert np.isfinite(lp)
    assert abs(lp - _logp_at(config, point)) < 1e-9


def test_config_path_supports_inferred_per_regime_p_outlier(small_single_participant):
    """`p_outlier` is an RSSSM addition, so the config path must append it too.

    No registered SSM carries `p_outlier` in `list_params`, so deriving them
    from the registry alone left `switching_params=["p_outlier"]` failing
    validation with "not parameters of model 'ddm'" — while the granular path
    appends it in `_build_config`.
    """
    cfg = RSSSMConfig(
        model_name="rsssm_ddm",
        model="ddm",
        K=2,
        switching_params=["v", "p_outlier"],
        loglik_kind="analytical",
    )
    m = RSSSM(data=small_single_participant, model_config=cfg)
    assert m.list_params == ["v", "a", "z", "t", "p_outlier"]
    assert m._has_p_outlier
    _assert_same_model(
        RSSSM(
            data=small_single_participant,
            model="ddm",
            K=2,
            switching_params=["v", "p_outlier"],
        ),
        m,
    )


def test_config_path_supports_fixed_per_regime_p_outlier(small_single_participant):
    """The same for a length-K fixed lapse supplied through `param_specs`."""
    cfg = RSSSMConfig(
        model_name="rsssm_ddm",
        model="ddm",
        K=2,
        switching_params=["v"],
        loglik_kind="analytical",
        param_specs={"p_outlier": [0.02, 0.10]},
    )
    m = RSSSM(data=small_single_participant, model_config=cfg)
    assert m.list_params == ["v", "a", "z", "t", "p_outlier"]
    assert m._has_p_outlier
    _assert_same_model(
        RSSSM(
            data=small_single_participant,
            model="ddm",
            K=2,
            switching_params=["v"],
            p_outlier=[0.02, 0.10],
        ),
        m,
    )


def test_config_path_rejects_global_iid_p_outlier(small_single_participant):
    """The decision-10.1.9 rejection must not be bypassable via `model_config=`.

    The granular path enforces it in `_resolve_p_outlier_spec`; a hand-built
    config went straight to `param_specs`, so a scalar `p_outlier` produced the
    shared-across-regimes lapse the design explicitly rules out.
    """
    cfg = RSSSMConfig(
        model_name="rsssm_ddm",
        model="ddm",
        K=2,
        switching_params=["v"],
        loglik_kind="analytical",
        param_specs={"p_outlier": 0.05},
    )
    with pytest.raises(NotImplementedError, match="global iid"):
        RSSSM(data=small_single_participant, model_config=cfg)


def test_config_path_defaults_lan_backend_with_explicit_list_params(
    small_single_participant,
):
    """A config that sets `list_params` by hand still gets the LAN jax default.

    The metadata-derivation early-return skipped the backend default too, so
    `loglik_kind="approx_differentiable"` silently resolved to the (much slower)
    pytensor backend while the granular path picked jax.
    """
    lan = get_default_model_config("ddm")["likelihoods"]["approx_differentiable"]
    cfg = RSSSMConfig(
        model_name="rsssm_ddm",
        model="ddm",
        K=2,
        switching_params=["v"],
        loglik_kind="approx_differentiable",
        list_params=["v", "a", "z", "t"],
        bounds=dict(lan["bounds"]),
    )
    m = RSSSM(data=small_single_participant, model_config=cfg)
    assert m.model_config.backend == "jax"
    assert m._broadcast_params
    _assert_same_model(
        RSSSM(
            data=small_single_participant,
            model="ddm",
            K=2,
            switching_params=["v"],
            loglik_kind="approx_differentiable",
        ),
        m,
    )


def test_extra_fields_rejected(small_single_participant):
    """`extra_fields` would be validated and then ignored, so it is rejected.

    They feed bambi's trial-wise regression machinery, which the direct-build
    path does not have — the emission is fed `(rt, response)` only.
    """
    cfg = RSSSMConfig(
        model_name="rsssm_ddm",
        model="ddm",
        K=2,
        switching_params=["v"],
        loglik_kind="analytical",
        extra_fields=["cue"],
    )
    with pytest.raises(NotImplementedError, match="extra_fields"):
        RSSSM(data=small_single_participant.assign(cue=1.0), model_config=cfg)


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
    ("param", "value"),
    [
        ("a", -1.0),  # a > 0
        ("z", 2.0),  # z in [0, 1]
        ("t", -0.5),  # t > 0
    ],
)
def test_fixed_scalar_out_of_bounds_rejected(tiny_df, param, value):
    """An out-of-support *fixed* scalar must raise, as it does in `hssm.HSSM`.

    The direct-build path never goes through `Param.validate`, so these built a
    model with a *finite but wrong* logp instead of erroring.
    """
    with pytest.raises(ValueError, match="not in bounds"):
        RSSSM(
            data=tiny_df,
            model="ddm",
            K=2,
            switching_params=["v"],
            **{param: value},
        )


def test_fixed_vector_out_of_bounds_rejected(tiny_df):
    """One out-of-support entry in a length-K fixed vector is enough to raise."""
    with pytest.raises(ValueError, match="not in bounds"):
        RSSSM(
            data=tiny_df,
            model="ddm",
            K=2,
            switching_params=["v"],
            a=[0.8, -1.2],
        )


@pytest.mark.parametrize("p_outlier", [[0.3, 1.5], [-0.1, 0.2]])
def test_fixed_per_regime_p_outlier_out_of_bounds_rejected(tiny_df, p_outlier):
    """Per-regime `p_outlier` outside [0, 1] raises instead of yielding NaN logp."""
    with pytest.raises(ValueError, match="not in bounds"):
        RSSSM(
            data=tiny_df,
            model="ddm",
            K=2,
            switching_params=["v"],
            p_outlier=p_outlier,
        )


def test_fixed_value_on_bound_is_accepted(tiny_df):
    """The bounds check is inclusive: a value exactly on the bound is valid."""
    m = RSSSM(data=tiny_df, model="ddm", K=2, switching_params=["v"], z=[0.0, 1.0])
    assert "z" not in {rv.name for rv in m.pymc_model.free_RVs}


@pytest.mark.parametrize(
    "spec",
    [
        {"a": [1.0, float("nan")]},  # per-regime: the silent case
        {"z": float("nan")},
        {"v": float("inf")},  # bounds are (-inf, inf): no bound to fail
    ],
)
def test_non_finite_fixed_value_rejected(tiny_df, spec):
    """NaN / inf compare False against both bounds, so they slipped the check.

    The per-regime NaN is the worst of them: it clamps that regime's emission at
    every trial, so the `logsumexp` over regimes silently degenerates to a K=1
    model with a perfectly plausible finite logp (`a=[1.0, nan]` scored -292).
    """
    switching = ["a"] if "v" in spec else ["v"]
    with pytest.raises(ValueError, match="must be finite"):
        RSSSM(data=tiny_df, model="ddm", K=2, switching_params=switching, **spec)


@pytest.mark.parametrize("model_name", ["angle", "levy", "ornstein", "weibull"])
def test_explicit_loglik_kind_is_never_substituted(tiny_df, model_name):
    """Asking for a likelihood the model lacks must raise, not swap in another.

    The kind resolved here fixes the bounds, the default priors *and* the
    emission, so falling back is right when the user said nothing (`angle` has
    no analytical likelihood, so plain `RSSSM(model="angle")` must still work)
    but wrong for an explicit request: it silently returns the neural
    approximation where `hssm.HSSM` raises. 7 of the 16 registered models have
    no analytical likelihood.
    """
    with pytest.raises(ValueError, match="has no 'analytical' likelihood"):
        RSSSM(
            data=tiny_df,
            model=model_name,
            K=2,
            switching_params=["v"],
            loglik_kind="analytical",
        )


def test_omitted_loglik_kind_still_falls_back(tiny_df):
    """The fallback itself is intact: only the *explicit* request is protected."""
    m = RSSSM(data=tiny_df, model="angle", K=2, switching_params=["v"])
    assert m.loglik_kind == "approx_differentiable"
    assert RSSSM(
        data=tiny_df, model="ddm", K=2, switching_params=["v"]
    ).loglik_kind == ("analytical")


def test_unknown_loglik_kind_rejected(tiny_df):
    """An unrecognised kind names the available ones rather than falling back."""
    with pytest.raises(ValueError, match="has no 'nope' likelihood"):
        RSSSM(
            data=tiny_df, model="ddm", K=2, switching_params=["v"], loglik_kind="nope"
        )


def test_blackbox_loglik_kind_rejected(tiny_df):
    """`blackbox` builds and then dies at gradient time, so reject it eagerly."""
    with pytest.raises(NotImplementedError, match="no gradient"):
        RSSSM(
            data=tiny_df,
            model="ddm",
            K=2,
            switching_params=["v"],
            loglik_kind="blackbox",
        )


def test_blackbox_only_model_rejected(tiny_df):
    """`full_ddm` has only a blackbox likelihood, so it cannot be an emission.

    It used to build fine and produce a finite logp; the failure surfaced only
    at `sample()` as `NotImplementedError: pullback not implemented for
    BlackBoxOp`.
    """
    with pytest.raises(NotImplementedError, match="no differentiable likelihood"):
        RSSSM(data=tiny_df, model="full_ddm", K=2, switching_params=["v"])


def test_model_as_config_loglik_kind_validated_against_registry(tiny_df):
    """The emission is rebuilt from the registry, so the kind is checked there."""
    from hssm.config import Config

    cfg = Config(
        model_name="angle",
        loglik_kind="analytical",
        list_params=["v", "a", "z", "t", "theta"],
    )
    with pytest.raises(ValueError, match="has no 'analytical' likelihood"):
        RSSSM(data=tiny_df, model=cfg, K=2, switching_params=["v"])


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


def test_choice_only_model_rejected(tiny_df):
    """A choice-only SSM (single response column) is not supported in v1."""
    with pytest.raises(NotImplementedError, match="choice-only"):
        RSSSM(
            data=tiny_df,
            model="softmax_inv_temperature_2",  # a supported choice-only SSM
            K=2,
            switching_params=["v"],
        )


def test_invalid_response_coding_rejected(tiny_df):
    """A response coding the SSM does not use must raise, not silently rescore.

    The DDM's choices are `{-1, 1}`; `{1, 2}` used to be accepted and every
    trial scored as an upper-boundary response — a wrong likelihood with no
    warning, where `hssm.HSSM` raises on the same frame.
    """
    df = tiny_df.copy()
    df["response"] = df["response"].replace({-1.0: 2.0})
    with pytest.raises(ValueError, match="Invalid responses"):
        RSSSM(data=df, model="ddm", K=2, switching_params=["v"])


def test_negative_rt_rejected(tiny_df):
    """Negative RTs used to yield a finite, wrong logp."""
    df = tiny_df.copy()
    df.loc[3, "rt"] = -0.5
    with pytest.raises(ValueError, match="negative response times"):
        RSSSM(data=df, model="ddm", K=2, switching_params=["v"])


def test_nan_rt_rejected(tiny_df):
    """NaN RTs used to yield a NaN logp instead of an error."""
    df = tiny_df.copy()
    df.loc[3, "rt"] = np.nan
    with pytest.raises(ValueError, match="NaN response times"):
        RSSSM(data=df, model="ddm", K=2, switching_params=["v"])


def test_numpy_scalar_spec_fixes_the_parameter(small_single_participant):
    """Numpy scalars fix a parameter rather than silently becoming an RV.

    `np.float64` subclasses `float` and was already handled; `np.float32` /
    `np.int64` are not `float`/`int` subclasses and used to fall through to the
    inferred branch.
    """
    m = RSSSM(
        data=small_single_participant,
        model="ddm",
        K=2,
        switching_params=["v"],
        z=np.float32(0.5),
        a=np.int64(1),
    )
    rv_names = {rv.name for rv in m.pymc_model.free_RVs}
    assert "z" not in rv_names and "a" not in rv_names
    assert _logp_finite(m)


@pytest.mark.parametrize("spec", ["0.8", {"0.8"}])
def test_uninterpretable_param_spec_rejected(small_single_participant, spec):
    """A spec that is neither numeric, array-like, nor a prior raises."""
    with pytest.raises(TypeError, match="Unsupported input for 'z'"):
        RSSSM(
            data=small_single_participant,
            model="ddm",
            K=2,
            switching_params=["v"],
            z=spec,
        )


def test_fixed_vector_wrong_length_rejected_eagerly(tiny_df):
    """A fixed-per-regime ndarray of the wrong length is caught at validate()."""
    with pytest.raises(ValueError, match="length 3, expected K=2"):
        RSSSM(
            data=tiny_df,
            model="ddm",
            K=2,
            switching_params=["v"],
            a=np.array([0.8, 0.9, 1.0]),
        )


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
        resolve_initial_distribution("uniform"), UniformInitialDistribution
    )
    assert isinstance(resolve_initial_distribution(None), UniformInitialDistribution)
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


def test_explicit_anchor_validated_even_without_switching_params():
    """`ordering="v"` must not be silently ignored when nothing switches.

    The early return for an empty `switching_params` used to fire *before* the
    `OrderByParam` checks, so an explicit (and unsatisfiable) anchor built a
    model with no ordering at all and no error — while the same typo with one
    switching parameter raised.
    """
    with pytest.raises(ValueError, match="is not in switching_params"):
        resolve_anchor(OrderByParam(name="v"), [])


def test_explicit_anchor_rejected_at_construction(tiny_df):
    """The same, through the constructor."""
    with pytest.raises(ValueError, match="is not in switching_params"):
        RSSSM(
            data=tiny_df,
            model="ddm",
            K=2,
            switching_params=[],
            ordering="v",
            a=[0.8, 1.2],
        )


def test_auto_and_no_ordering_stay_silent_without_switching_params():
    """Only an *explicit* anchor raises; the implicit specs still return None."""
    assert resolve_anchor(AutoOrdering(), []) is None
    assert resolve_anchor(NoOrdering(), []) is None


def test_fixed_initial_distribution_rejects_negative_entries():
    """`pi0 = [1.5, -0.5]` sums to 1 but makes `log(pi0)` — and the logp — NaN."""
    with pytest.raises(ValueError, match="non-negative"):
        FixedInitialDistribution(pi0=[1.5, -0.5]).pi0_value(2)


def test_fixed_initial_distribution_rejected_at_construction(tiny_df):
    """The same, through the constructor."""
    with pytest.raises(ValueError, match="non-negative"):
        RSSSM(
            data=tiny_df,
            model="ddm",
            K=2,
            switching_params=["v"],
            initial_distribution=[1.5, -0.5],
        )


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


def test_single_trial_only_panel_rejected():
    """A panel where every participant has one trial (T_max == 1) is rejected."""
    df = pd.DataFrame(
        {
            "rt": [0.5, 0.6, 0.7],
            "response": [1, -1, 1],
            "participant_id": [0, 1, 2],
        }
    )
    with pytest.raises(ValueError, match="at least 2 trials"):
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


def test_build_under_float32(small_single_participant):
    """Initial values must respect `floatX`.

    `hssm.set_floatX("float32")` makes every RV float32, and PyMC refuses to
    store a float64 start in one ("cannot store a value of dtype float64").
    Both seeded-vector paths are exercised: the anchor grid (`v`), the
    per-parameter safe seed (`t`), and the no-pooling `(N, K)` broadcast.
    """
    import pytensor

    prev_floatx = pytensor.config.floatX
    hssm.set_floatX("float32", update_jax=True)
    try:
        m = RSSSM(
            data=small_single_participant,
            model="ddm",
            K=2,
            switching_params=["v", "t"],
        )
        assert _logp_finite(m)
        m_np = RSSSM(
            data=make_panel(2, 30),
            model="ddm",
            K=2,
            switching_params=["v"],
            pooling="none",
            participant_col="participant_id",
        )
        assert _logp_finite(m_np)
    finally:
        hssm.set_floatX(prev_floatx, update_jax=True)


def _hmm_data_dtype(model) -> str:
    """Return the dtype of the emission's `hmm_data` constant in the model graph."""
    from pytensor.graph.traversal import ancestors

    (potential,) = model.pymc_model.potentials
    constants = [
        node
        for node in ancestors([potential])
        if getattr(node, "name", None) == "hmm_data"
    ]
    assert constants, "the emission's `hmm_data` constant is not in the graph"
    return constants[0].dtype


@pytest.mark.parametrize("float_x", ["float32", "float64"])
def test_panel_data_is_cast_to_floatx(small_single_participant, float_x):
    """The panel enters the emission in `floatX`, not a hard-coded float32.

    `small_single_participant` is a float32 frame, so a hard `.astype("float32")`
    also passes under `float32` — the property only bites under the float64
    default, where the old cast truncated genuine float64 RTs.
    """
    import pytensor

    prev_floatx = pytensor.config.floatX
    hssm.set_floatX(float_x, update_jax=True)
    try:
        m = RSSSM(
            data=small_single_participant.astype("float64"),
            model="ddm",
            K=2,
            switching_params=["v"],
        )
        assert _hmm_data_dtype(m) == float_x
    finally:
        hssm.set_floatX(prev_floatx, update_jax=True)


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


# ---------------------------------------------------------------------------
# Per-regime p_outlier lapse mixture (§1.2)
# ---------------------------------------------------------------------------


def test_p_outlier_switching_builds_lapse_mixture(small_single_participant):
    """`p_outlier` in switching_params adds a per-regime RV + lapse mixture."""
    import pytensor.tensor as pt_

    from hssm.hmm import ffbs

    m = RSSSM(
        data=small_single_participant,
        model="ddm",
        K=2,
        switching_params=["v", "p_outlier"],
        p_outlier={"name": "Beta", "alpha": 1, "beta": 15},
    )
    assert m.list_params[-1] == "p_outlier"
    assert "p_outlier" in m._regime_params
    rvs = {rv.name: rv for rv in m.pymc_model.free_RVs}
    assert "p_outlier" in rvs and tuple(rvs["p_outlier"].type.shape) == (2,)

    ip = m.pymc_model.initial_point()
    assert np.all(np.isfinite(m.pymc_model.compile_dlogp()(ip)))

    # The mixture genuinely alters the emission: p_outlier=0 vs 0.3 differ.
    fn, order = ffbs._compile_emission_fn(m)
    base = {"v": np.array([0.2, 1.5]), "a": 0.8, "z": 0.5, "t": 0.3}

    def total(po):
        vals = {**base, "p_outlier": np.array([po, po])}
        return float(np.sum(fn(*[vals[n] for n in order])))

    assert abs(total(0.0) - total(0.3)) > 1.0


def test_p_outlier_fixed_per_regime_is_constant(small_single_participant):
    """A length-K `p_outlier` list fixes the lapse per regime (no RV)."""
    m = RSSSM(
        data=small_single_participant,
        model="ddm",
        K=2,
        switching_params=["v"],
        p_outlier=[0.02, 0.1],
    )
    assert "p_outlier" in m._regime_params  # carries a regime axis
    assert "p_outlier" not in [rv.name for rv in m.pymc_model.free_RVs]  # but fixed


def test_p_outlier_global_iid_rejected(small_single_participant):
    """A scalar / non-per-regime `p_outlier` is rejected (decision 10.1.9)."""
    for bad in (0.05, {"name": "Beta", "alpha": 1, "beta": 15}):
        with pytest.raises(NotImplementedError, match="global iid"):
            RSSSM(
                data=small_single_participant,
                model="ddm",
                K=2,
                switching_params=["v"],
                p_outlier=bad,
            )


def test_top_level_lapse_still_rejected(small_single_participant):
    """The top-level `lapse` kwarg remains rejected in v1."""
    with pytest.raises(NotImplementedError):
        RSSSM(
            data=small_single_participant,
            model="ddm",
            K=2,
            switching_params=["v"],
            lapse={"name": "Uniform", "lower": 0.0, "upper": 10.0},
        )


def test_validate_rejects_mismatched_transition_prior(tiny_df):
    """A transition prior whose shape disagrees with K is caught at construction."""
    with pytest.raises(ValueError, match="alpha matrix has shape"):
        RSSSM(
            data=tiny_df,
            model="ddm",
            K=2,
            switching_params=["v"],
            transition_prior={
                "name": "Dirichlet",
                "alpha": [[20, 1, 1], [1, 20, 1], [1, 1, 20]],
            },
        )


def test_validate_requires_model():
    """A config with no `model` fails validation: the emission is unresolvable."""
    cfg = RSSSMConfig(
        model_name="x",
        model=None,
        K=2,
        switching_params=["v"],
        list_params=["v", "a", "z", "t"],
        bounds={"v": (-np.inf, np.inf)},
        loglik_kind="analytical",
    )
    with pytest.raises(ValueError, match="`model` must be provided"):
        cfg.validate()


def test_config_has_no_emission_logp_func():
    """`emission_logp_func` is gone: it was never read anywhere in `src/`."""
    assert not hasattr(
        RSSSMConfig(model_name="x", model="ddm", loglik_kind="analytical"),
        "emission_logp_func",
    )


def test_validate_warns_on_degenerate_no_switching(tiny_df, caplog):
    """No per-regime variation -> warn that regimes are unidentifiable."""
    import logging

    with caplog.at_level(logging.WARNING, logger="hssm"):
        RSSSM(data=tiny_df, model="ddm", K=2, switching_params=[])
    assert any(
        "interchangeable" in rec.message and "unidentifiable" in rec.message
        for rec in caplog.records
    )


def test_validate_no_warn_with_fixed_per_regime(tiny_df, caplog):
    """A fixed-per-regime vector distinguishes regimes -> no degeneracy warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="hssm"):
        RSSSM(
            data=tiny_df,
            model="ddm",
            K=2,
            switching_params=[],
            v=[-1.0, 1.0],
        )
    assert not any("interchangeable" in rec.message for rec in caplog.records)


def test_p_outlier_alone_is_not_an_anchor(small_single_participant):
    """`p_outlier` as the sole switching param builds (unordered), not NaN.

    Auto-anchoring `p_outlier` would apply the `ordered` transform to the
    bounded Beta lapse parameter, which gives a non-finite logp at the start.
    It is excluded from auto-anchoring; the model builds without an ordering
    constraint and has a finite gradient.
    """
    m = RSSSM(
        data=small_single_participant, model="ddm", K=2, switching_params=["p_outlier"]
    )
    assert "p_outlier" in {rv.name for rv in m.pymc_model.free_RVs}
    ip = m.pymc_model.initial_point()
    assert np.isfinite(m.pymc_model.compile_logp()(ip))
    assert np.all(np.isfinite(m.pymc_model.compile_dlogp()(ip)))


def test_order_by_p_outlier_rejected(small_single_participant):
    """Explicitly ordering on `p_outlier` raises (unstable ordered-Beta)."""
    with pytest.raises(NotImplementedError, match="Ordering on `p_outlier`"):
        RSSSM(
            data=small_single_participant,
            model="ddm",
            K=2,
            switching_params=["v", "p_outlier"],
            ordering={"name": "p_outlier", "direction": "asc"},
        )


def test_no_pooling_fixed_scalar_and_list_build():
    """Fixed scalar/per-regime values build under pooling='none' (broadcast to N)."""
    panel = make_panel(3, 40)
    m1 = RSSSM(
        data=panel,
        model="ddm",
        K=2,
        switching_params=["v"],
        pooling="none",
        participant_col="participant_id",
        a=1.5,
    )
    assert _logp_finite(m1)
    assert "a" not in {rv.name for rv in m1.pymc_model.free_RVs}  # fixed, no RV
    m2 = RSSSM(
        data=panel,
        model="ddm",
        K=2,
        switching_params=["v"],
        pooling="none",
        participant_col="participant_id",
        a=[0.8, 0.9],
    )
    assert _logp_finite(m2)
    ip = m2.pymc_model.initial_point()
    assert np.all(np.isfinite(m2.pymc_model.compile_dlogp()(ip)))


@pytest.mark.parametrize("fixed", [0.5, [0.2, 1.5]])
def test_switching_param_with_fixed_value_raises(small_single_participant, fixed):
    """A param both in switching_params and given a fixed value is a conflict."""
    with pytest.raises(ValueError, match="in switching_params"):
        RSSSM(
            data=small_single_participant,
            model="ddm",
            K=2,
            switching_params=["v"],
            v=fixed,
        )


# ---------------------------------------------------------------------------
# Edge cases (plan §7.3) and the remaining method surface
# ---------------------------------------------------------------------------


def test_unoccupied_and_imbalanced_regimes_do_not_crash():
    """§7.3: a never-occupied / highly-imbalanced regime must not produce NaN/Inf.

    Data is generated almost entirely from regime 0 (regime 1 is effectively
    never visited), but the K=2 model still declares both regimes. The
    unoccupied regime's parameters are prior-driven and constrained only by the
    `ordered` transform; the joint logp and its gradient must stay finite.
    """
    P = np.array([[1.0 - 1e-9, 1e-9], [0.5, 0.5]])
    data, regimes = simulate_hmm_ddm_data(
        150,
        {
            0: {"v": -1.0, "a": 1.0, "z": 0.5, "t": 0.3},
            1: {"v": 1.5, "a": 1.0, "z": 0.5, "t": 0.3},
        },
        P,
        np.array([1.0, 0.0]),
        seed=3,
    )
    assert set(np.unique(regimes)) == {0}  # regime 1 never occupied
    m = RSSSM(
        data=pd.DataFrame(data, columns=["rt", "response"]),
        model="ddm",
        K=2,
        switching_params=["v"],
    )
    ip = m.pymc_model.initial_point()
    assert np.isfinite(m.pymc_model.compile_logp()(ip))
    assert np.all(np.isfinite(m.pymc_model.compile_dlogp()(ip)))


def test_descending_anchor_builds_with_finite_gradient(small_single_participant):
    """`OrderByParam(direction="desc")` builds a valid graph (negated anchor)."""
    m = RSSSM(
        data=small_single_participant,
        model="ddm",
        K=2,
        switching_params=["v"],
        ordering={"name": "v", "direction": "desc"},
        v={"name": "Normal", "mu": 0.0, "sigma": 2.0},
    )
    assert "v" in {d.name for d in m.pymc_model.deterministics}  # exposed as -u
    ip = m.pymc_model.initial_point()
    assert np.all(np.isfinite(m.pymc_model.compile_dlogp()(ip)))


def test_descending_anchor_starts_in_support(small_single_participant):
    """The descending anchor must start on the *reversed* grid, not the negated one.

    The ordered RV is `u = -anchor`, so its start has to be `-asc[::-1]`; plain
    `-asc` starts a one-sided anchor (here `a > 0`) outside its support, where
    every trial's emission is clamped to `LOGP_LB` and the likelihood
    contributes exactly zero gradient — a silent flat plateau for NUTS.
    """
    from hssm.distribution_utils.dist import LOGP_LB

    m = RSSSM(
        data=small_single_participant,
        model="ddm",
        K=2,
        switching_params=["a"],
        ordering={"name": "a", "direction": "desc"},
        a={"name": "Normal", "mu": 1.5, "sigma": 1.0},
    )
    ip = m.pymc_model.initial_point()
    a_init = eval_at_point(m, m.pymc_model["a"], ip)
    assert np.all(a_init > 0.0)  # in support
    assert a_init[0] > a_init[1]  # descending

    # The likelihood is not sitting on the clamped floor (T * LOGP_LB).
    potential = float(eval_at_point(m, m.pymc_model.potentials[0], ip))
    assert potential > m.n_trials * float(LOGP_LB)


def test_vi_and_log_likelihood_raise(small_single_participant):
    """`vi` and `log_likelihood` are unavailable on the scalar-marginal graph."""
    m = RSSSM(data=small_single_participant, model="ddm", K=2, switching_params=["v"])
    with pytest.raises(NotImplementedError):
        m.vi()
    with pytest.raises(NotImplementedError, match="compute_log_likelihood"):
        m.log_likelihood()


def test_predictive_family_raises_cleanly(small_single_participant):
    """Out-of-scope predictive methods raise NotImplementedError, not AttributeError.

    The inherited implementations reach through `self.model` (the bambi model
    RSSSM never builds), which would otherwise leak a bare AttributeError. They
    are overridden to point at the design §6.3 rationale.
    """
    m = RSSSM(data=small_single_participant, model="ddm", K=2, switching_params=["v"])
    with pytest.raises(NotImplementedError, match="§6.3"):
        m.sample_posterior_predictive()
    with pytest.raises(NotImplementedError, match="§6.3"):
        m.predict()
    with pytest.raises(NotImplementedError, match="§6.3"):
        m.sample_prior_predictive()
    with pytest.raises(NotImplementedError, match="§6.3"):
        m.plot_predictive()
    with pytest.raises(NotImplementedError, match="§6.3"):
        m.sample_do({"v": [0.5, 1.5]})
    with pytest.raises(NotImplementedError, match="bambi"):
        m.set_alias({"v": "drift"})
    with pytest.raises(NotImplementedError, match="infer_regimes"):
        m.add_likelihood_parameters_to_datatree()


def test_sample_rejects_bambi_sampler_kwarg(small_single_participant):
    """`sample(sampler=...)` raises a pointed TypeError, not a PyMC internal one.

    `HSSM.sample` selects the backend with `sampler=`; RSSSM calls `pm.sample`
    directly, where the same name collided with `_sample_external_nuts`'s own
    `sampler` argument ("got multiple values for keyword argument 'sampler'").
    """
    m = RSSSM(data=small_single_participant, model="ddm", K=2, switching_params=["v"])
    with pytest.raises(TypeError, match="nuts_sampler"):
        m.sample(sampler="numpyro")


def test_compile_logp_works_on_rsssm(small_single_participant):
    """`compile_logp()` (untransformed) must work despite RSSSM's explicit initvals.

    `remove_value_transforms` goes through `fgraph_from_model`, which refuses
    models with non-default initial values; RSSSM sets several. The initvals are
    cleared for the conversion and must be restored afterwards.
    """
    m = RSSSM(data=small_single_participant, model="ddm", K=2, switching_params=["v"])
    before = dict(m.pymc_model.rvs_to_initial_values)
    assert any(value is not None for value in before.values())

    logp_fn = m.compile_logp()
    # The value variables are untransformed here, so the point is the natural
    # parameterisation (a simplex `P`, an ascending `v`, `a > 0`, `z` in [0, 1]).
    untransformed_point = {
        "P": np.array([[0.9, 0.1], [0.1, 0.9]]),
        "v": np.array([-1.0, 1.0]),
        "a": np.array(1.0),
        "z": np.array(0.5),
        "t": np.array(0.1),
    }
    assert np.isfinite(logp_fn(untransformed_point))
    assert m.pymc_model.rvs_to_initial_values == before


def test_repr_summarises_the_model(small_single_participant):
    """`repr(model)` must summarise the model, not raise.

    `HSSMBase.__repr__` walks `self.params`, which the direct-build path never
    creates — a bare `model` cell in a notebook used to be a traceback.
    """
    m = RSSSM(
        data=small_single_participant,
        model="ddm",
        K=2,
        switching_params=["v"],
        pooling="none",
    )
    text = repr(m)
    assert "Regime-Switching Sequential Sampling Model" in text
    assert "Regimes (K): 2" in text
    assert "Switching parameters: v" in text
    assert "Pooling: none" in text
    assert str(m) == text


def test_graph_returns_graphviz(small_single_participant):
    """`graph()` renders the directly-built model via graphviz."""
    pytest.importorskip("graphviz")
    m = RSSSM(data=small_single_participant, model="ddm", K=2, switching_params=["v"])
    assert m.graph() is not None

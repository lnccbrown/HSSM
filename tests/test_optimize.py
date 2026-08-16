"""Tests for point estimation --- ``find_MAP``, ``find_MLE`` and ``PointEstimate``.

See issue #1102.
"""

import pickle
import warnings

import arviz as az
import cloudpickle
import jax
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest
from scipy.optimize import OptimizeResult
from ssms.basic_simulators.simulator import simulator

import hssm
from hssm.optimize import (
    REPR_MAX_ROWS,
    PointEstimate,
    audit_result,
    describe_failures,
    group_specific_terms,
    jittered_starts,
    points_equal,
    score_point,
)

TRUE_DDM = {"v": 0.5, "a": 1.5, "z": 0.5, "t": 0.5}
DATA_SEED = 10


@pytest.fixture(autouse=True)
def _pin_float64_precision():
    """Pin every test in this module to float64 and restore the caller's state.

    Point estimation is gradient based and therefore precision sensitive: the
    recovery tolerances below only hold at double precision, and the float32
    test deliberately flips the setting. 29 test modules call
    ``hssm.set_floatX("float32")`` at *import* time and pytest imports every
    module during collection, so without this fixture the precision a test runs
    under depends on collection order.

    The raw configs are saved and restored directly rather than going through
    ``hssm.set_floatX`` / ``set_jax_precision``, which would pull in the aDDM
    kernel and its dtype cache.
    """
    previous_floatx = pytensor.config.floatX
    previous_x64 = jax.config.jax_enable_x64
    pytensor.config.floatX = "float64"
    jax.config.update("jax_enable_x64", True)
    yield
    pytensor.config.floatX = previous_floatx
    jax.config.update("jax_enable_x64", previous_x64)


@pytest.fixture(scope="module")
def ddm_data():
    """Return 500 DDM trials simulated from ``TRUE_DDM``.

    Seeded on purpose. The recovery tests assert the estimate lands within
    ``0.2`` of the truth, but at 500 trials the drift rate `v` is the least
    well-identified parameter: across seeds its estimate sits around `0.585`
    with an SD near `0.055`, so an unseeded draw puts the assertion only about
    two SDs from its bound and the suite fails a small fraction of runs for no
    reason. This seed keeps the whole module deterministic; its worst
    deviation from truth is `0.058` across both MAP and MLE.
    """
    simulated = simulator(
        list(TRUE_DDM.values()), model="ddm", n_samples=500, random_state=DATA_SEED
    )
    return pd.DataFrame(
        np.column_stack([simulated["rts"][:, 0], simulated["choices"][:, 0]]),
        columns=["rt", "response"],
    )


@pytest.fixture
def ddm_model(ddm_data):
    """Return a plain analytical DDM model over ``ddm_data``."""
    return hssm.HSSM(data=ddm_data, model="ddm", loglik_kind="analytical")


@pytest.fixture(scope="module")
def hierarchical_data():
    """Return a three-participant slice of ``cavanagh_theta``.

    Small enough for the fast suite, and it still reproduces the non-finite
    gradient at PyMC's default start that motivated this work.
    """
    data = hssm.load_data("cavanagh_theta")
    keep = sorted(data.participant_id.unique())[:3]
    return data[data.participant_id.isin(keep)].reset_index(drop=True)


@pytest.fixture
def hierarchical_model(hierarchical_data):
    """Return a DDM with a group-specific intercept on ``v``."""
    return hssm.HSSM(
        data=hierarchical_data,
        model="ddm",
        loglik_kind="analytical",
        include=[{"name": "v", "formula": "v ~ 1 + (1|participant_id)"}],
    )


def failed_optimize_result():
    """Return an ``OptimizeResult`` that looks like a failed L-BFGS-B run."""
    return OptimizeResult(
        x=np.zeros(4), fun=np.nan, success=False, message="ABNORMAL:", nit=0
    )


# region ===== find_MAP =====
def test_find_map_recovers_parameters(ddm_model):
    """MAP on a well-specified analytical DDM should recover the truth."""
    estimate = ddm_model.find_MAP(progressbar=False)

    assert estimate is not None
    assert estimate.success
    assert estimate.kind == "MAP"
    for name, true_value in TRUE_DDM.items():
        assert estimate.params[name] == pytest.approx(true_value, abs=0.2), name


def test_find_map_defaults_start_to_hssm_initvals(ddm_model, monkeypatch):
    """`find_MAP` must hand PyMC HSSM's processed initial values, not PyMC's."""
    seen = {}
    original = pm.find_MAP

    def spy(*args, **kwargs):
        seen["start"] = kwargs["start"]
        return original(*args, **kwargs)

    monkeypatch.setattr(pm, "find_MAP", spy)
    ddm_model.find_MAP(progressbar=False)

    assert seen["start"] == ddm_model.initvals
    # The whole point of the fix: HSSM's start differs from PyMC's own.
    assert seen["start"]["t"] != ddm_model.initial_point()["t"]


def test_find_map_converges_where_pymc_default_start_fails(hierarchical_model):
    """Regression test for the silently-returned start point (issue #1102).

    At PyMC's default start (``t=2.0``) the gradient of this model's log-density
    has non-finite entries. The *logp* is finite there, so ``check_start_vals``
    passes and L-BFGS-B aborts at ``nit=0`` --- which the old code stored as the
    MAP without complaint.
    """
    start = hierarchical_model.initial_point(transformed=True)
    gradient = hierarchical_model.pymc_model.compile_dlogp(jacobian=False)(start)
    assert not np.all(np.isfinite(gradient)), (
        "precondition: PyMC's default start must have a non-finite gradient"
    )

    estimate = hierarchical_model.find_MAP(progressbar=False)

    assert estimate is not None
    assert estimate.success
    assert estimate.opt_result.nit > 0
    # Assert the point actually moved, not merely that the optimizer iterated.
    assert estimate.params["a"] != pytest.approx(
        float(hierarchical_model.initvals["a"])
    )
    assert estimate.params["t"] != pytest.approx(
        float(hierarchical_model.initvals["t"])
    )


def test_find_map_reports_failure_instead_of_returning_the_start(hierarchical_model):
    """A failed run warns, returns None, and is never cached."""
    bad_start = hierarchical_model.initial_point()

    with pytest.warns(UserWarning, match="find_MAP failed to converge"):
        estimate = hierarchical_model.find_MAP(
            start=bad_start, method="L-BFGS-B", progressbar=False
        )

    assert estimate is None
    assert hierarchical_model._map_dict is None
    with pytest.raises(ValueError, match="compute map first"):
        _ = hierarchical_model.map


def test_find_map_failure_names_the_non_finite_gradient(hierarchical_model):
    """The warning should point at the likely cause, not just say 'failed'."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hierarchical_model.find_MAP(
            start=hierarchical_model.initial_point(),
            method="L-BFGS-B",
            progressbar=False,
        )

    messages = [str(warning.message) for warning in caught]
    assert any("never moved off its starting point" in message for message in messages)


def test_find_map_strict_raises(hierarchical_model):
    """`strict=True` turns the failure warning into an error."""
    with pytest.raises(RuntimeError, match="find_MAP failed to converge"):
        hierarchical_model.find_MAP(
            start=hierarchical_model.initial_point(),
            method="L-BFGS-B",
            progressbar=False,
            strict=True,
        )


def test_find_map_failure_clears_an_earlier_estimate(ddm_model):
    """A failed run must not leave the *previous* run's estimate reachable.

    The "a failed estimate is never cached" contract is what `model.map` and
    `sample(initvals="map")`'s `_map_dict is None` guard both rest on, so it
    has to hold on the second call as much as the first.
    """
    ddm_model.find_MAP(progressbar=False)
    assert ddm_model.map is not None

    with pytest.warns(UserWarning, match="find_MAP failed to converge"):
        assert ddm_model.find_MAP(maxeval=3, progressbar=False) is None

    # The attribute, not just the property: it is what `sample` branches on.
    assert ddm_model._map_dict is None
    with pytest.raises(ValueError, match="Please compute map first"):
        _ = ddm_model.map


def test_find_map_unprocessed_initvals_still_guarded(hierarchical_data):
    """With `process_initvals=False` the fix no-ops, so the guard must catch it.

    Without initval processing (and without jitter) ``model.initvals`` falls
    back to PyMC's own start --- the very point that breaks the optimizer. The
    default start no longer helps here, so the failure guard is the only thing
    standing between the user and a bogus estimate.
    """
    model = hssm.HSSM(
        data=hierarchical_data,
        model="ddm",
        loglik_kind="analytical",
        include=[{"name": "v", "formula": "v ~ 1 + (1|participant_id)"}],
        process_initvals=False,
        initval_jitter=0.0,
    )
    assert model.initvals["t"] == pytest.approx(float(model.initial_point()["t"]))

    with pytest.warns(UserWarning, match="find_MAP failed to converge"):
        estimate = model.find_MAP(method="L-BFGS-B", progressbar=False)

    assert estimate is None


def test_find_map_multi_start(ddm_model):
    """`n_starts` tries jittered starts and reports how many converged."""
    estimate = ddm_model.find_MAP(n_starts=3, seed=0, progressbar=False)

    assert estimate is not None
    assert estimate.n_starts == 3
    assert estimate.n_converged >= 1
    assert estimate.logp >= ddm_model.find_MAP(progressbar=False).logp - 1e-6


def test_find_map_multi_start_leaves_initvals_untouched(ddm_model):
    """Multi-start jitter must not mutate the model's stored initial values."""
    before = {name: np.copy(value) for name, value in ddm_model.initvals.items()}
    ddm_model.find_MAP(n_starts=3, seed=0, progressbar=False)

    for name, value in before.items():
        np.testing.assert_array_equal(ddm_model.initvals[name], value)


def test_find_map_return_raw(ddm_model):
    """`return_raw=True` returns the `(point, OptimizeResult)` tuple pymc does."""
    point, raw = ddm_model.find_MAP(return_raw=True, progressbar=False)

    assert isinstance(point, PointEstimate)
    assert isinstance(raw, OptimizeResult)
    assert raw.success


def test_find_map_include_transformed_false(ddm_model):
    """`include_transformed=False` drops the transformed value-var entries."""
    estimate = ddm_model.find_MAP(include_transformed=False, progressbar=False)

    assert "t" in estimate
    assert "t_log__" not in estimate
    assert "z_interval__" not in estimate


def _spy_on_extra_fields(model, monkeypatch, calls):
    """Make `_check_extra_fields` report work to do and record both calls.

    The check must return something *truthy*, or the update it guards never
    runs and the test cannot tell a working refresh from a missing one.
    """

    def check(self, *args):
        calls.append("check")
        return True

    def update(self, *args):
        calls.append("update")

    monkeypatch.setattr(type(model), "_check_extra_fields", check)
    monkeypatch.setattr(type(model), "_update_extra_fields", update)


def test_find_map_refreshes_extra_fields(ddm_model, monkeypatch):
    """RL/aDDM extra fields must be refreshed, as `sample()` does."""
    calls = []
    _spy_on_extra_fields(ddm_model, monkeypatch, calls)
    ddm_model.find_MAP(progressbar=False)

    assert calls[:2] == ["check", "update"]


def test_find_map_reports_the_method_that_actually_ran(ddm_data):
    """A blackbox likelihood forces Powell --- the estimate must say so.

    `pm.find_MAP` swaps the method out internally and never reports it, so
    recording the requested method would name an optimizer that never ran.
    """
    model = hssm.HSSM(data=ddm_data, model="ddm", loglik_kind="blackbox")
    estimate = model.find_MAP(progressbar=False)

    assert estimate is not None
    assert estimate.method == "Powell"


def test_find_map_audits_against_the_expanded_start(ddm_model, monkeypatch):
    """A partial `start=` must be expanded before the did-not-move audit.

    The hint is decided by comparing the returned point against the start, so a
    one-key `start=` would let a single unchanged parameter stand in for "the
    optimizer never moved" and send the user after the wrong cause. Asserted on
    the argument rather than on the resulting message: with a partial start the
    optimizer usually moves anyway, so a message-level assertion would pass
    whether or not the expansion happened.
    """
    seen = {}
    original = hssm.optimize.audit_result

    def spy(opt_result, point, start):
        seen["start"] = start
        return original(opt_result, point, start)

    monkeypatch.setattr(hssm.optimize, "audit_result", spy)
    ddm_model.find_MAP(start={"t": np.array(0.1)}, progressbar=False)

    # Every value variable, not just the one the caller named.
    expected = {var.name for var in ddm_model.pymc_model.value_vars}
    assert set(seen["start"]) == expected
    assert len(expected) > 1


def test_find_map_partial_start_is_still_honoured(ddm_model):
    """Expanding the start for the audit must not discard the caller's values."""
    estimate = ddm_model.find_MAP(start={"t": np.array(0.1)}, progressbar=False)

    assert estimate is not None
    assert estimate.success


def test_find_map_propagates_optimizer_usage_errors(ddm_model):
    """A bad `method=` is the caller's mistake, not a rejected starting point.

    SciPy raises `ValueError` from inside `pm.find_MAP` for an unknown solver.
    Catching it alongside the `SamplingError` that really does mean "PyMC
    rejected this start" would relabel a typo as a convergence failure and send
    the user off to debug their initial values instead.
    """
    with pytest.raises(ValueError, match="Unknown solver"):
        ddm_model.find_MAP(method="LBFGSB", progressbar=False)


def test_find_map_multi_start_jitter_cannot_leave_the_support(ddm_model):
    """Jitter near an upper bound must not burn starts.

    `z` lives in an interval, and a *relative* jitter of a start at 0.95 reaches
    1.045 --- outside it, so the forward transform is NaN and PyMC rejects the
    start. Applying the jitter in transformed space instead leaves no bound to
    cross. Seeded: with this seed the constrained-space jitter put one of the
    four extra starts out of range.
    """
    near_bound = {
        "v": np.array(0.5),
        "a": np.array(1.5),
        "z": np.array(0.95),
        "t": np.array(0.3),
    }
    estimate = ddm_model.find_MAP(
        start=near_bound, n_starts=5, seed=0, progressbar=False
    )

    assert estimate is not None
    assert estimate.n_converged == 5


def test_find_map_reports_every_failed_start(ddm_model):
    """With `n_starts>1` the warning must cover all starts, not just the last.

    `maxiter=0` fails every start identically, which is the case the reporting
    is expected to collapse rather than repeat three times.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimate = ddm_model.find_MAP(
            n_starts=3, seed=0, progressbar=False, options={"maxiter": 0}
        )

    assert estimate is None
    messages = [str(warning.message) for warning in caught]
    assert any("all 3 starts failed the same way" in message for message in messages)


# endregion


# region ===== float32 guard =====
def test_float32_warns_and_falls_back_to_powell(ddm_data):
    """float32 must warn and, absent an explicit method, switch to Powell.

    At single precision L-BFGS-B stops early *and reports success*, so the
    warning cannot be conditioned on the success flag.
    """
    hssm.set_floatX("float32", update_jax=True)
    model = hssm.HSSM(data=ddm_data, model="ddm", loglik_kind="analytical")

    with pytest.warns(UserWarning, match="float32"):
        estimate = model.find_MAP(progressbar=False)

    assert estimate is not None
    assert estimate.method == "Powell"
    # The gradient-free path still lands near the truth, unlike L-BFGS-B here.
    for name, true_value in TRUE_DDM.items():
        assert estimate.params[name] == pytest.approx(true_value, abs=0.25), name


def test_float32_honours_an_explicit_method(ddm_data):
    """An explicit `method=` is respected, but the warning still fires."""
    hssm.set_floatX("float32", update_jax=True)
    model = hssm.HSSM(data=ddm_data, model="ddm", loglik_kind="analytical")

    with pytest.warns(UserWarning, match="float32"):
        estimate = model.find_MAP(method="L-BFGS-B", progressbar=False)

    assert estimate.method == "L-BFGS-B"


def test_no_float32_warning_at_float64(ddm_model):
    """The precision warning must not fire at double precision."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ddm_model.find_MAP(progressbar=False)

    assert not [w for w in caught if "float32" in str(w.message)]


# endregion


# region ===== find_MLE =====
def test_find_mle_recovers_parameters(ddm_model):
    """MLE on a well-specified analytical DDM should recover the truth."""
    estimate = ddm_model.find_MLE(progressbar=False)

    assert estimate is not None
    assert estimate.kind == "MLE"
    for name, true_value in TRUE_DDM.items():
        assert estimate.params[name] == pytest.approx(true_value, abs=0.2), name


def test_mle_beats_map_on_the_same_objective(ddm_model):
    """The MLE must not score below the MAP on the observed log-likelihood.

    Scoring both points on `observedlogp` is the whole comparison: the MAP
    maximizes the joint density, so comparing its objective to the MLE's would
    say nothing at all.
    """
    map_estimate = ddm_model.find_MAP(progressbar=False)
    mle_estimate = ddm_model.find_MLE(progressbar=False)

    map_score = score_point(ddm_model.pymc_model, map_estimate, observed_only=True)
    mle_score = score_point(ddm_model.pymc_model, mle_estimate, observed_only=True)

    # A strict `>` flakes when the priors are near-flat or the optimum sits on a
    # bound; the meaningful claim is that MLE does not do worse.
    assert mle_score >= map_score - 1e-3


def test_find_mle_raises_on_hierarchical_models(hierarchical_model):
    """Group-level scale is unidentified without the priors --- refuse to fit."""
    assert group_specific_terms(hierarchical_model) == ["1|participant_id"]

    with pytest.raises(ValueError, match="not well posed for hierarchical models"):
        hierarchical_model.find_MLE()


def test_find_mle_hierarchical_escape_hatch(hierarchical_model):
    """`allow_unidentified=True` proceeds anyway, for penalized-likelihood work."""
    estimate = hierarchical_model.find_MLE(allow_unidentified=True, progressbar=False)

    assert estimate is not None
    assert estimate.kind == "MLE"
    # The whole point of the escape hatch is that the group-level terms are
    # still estimated -- unidentifiably, but present.
    assert "v_1|participant_id_sigma" in estimate.params
    assert "v_1|participant_id_offset" in estimate.params


def test_find_mle_escape_hatch_on_a_centered_hierarchical_model(hierarchical_data):
    """The escape hatch must survive a parameter the likelihood never reads.

    Under the centered parameterization `sigma` does not enter `observedlogp` at
    all. Compiling the objective against every value variable then hits
    PyTensor's unused-input check, and differentiating it hits the disconnected
    -input check -- both of which have to be told that this is expected here.
    """
    model = hssm.HSSM(
        data=hierarchical_data,
        model="ddm",
        loglik_kind="analytical",
        include=[{"name": "v", "formula": "v ~ 1 + (1|participant_id)"}],
        noncentered=False,
    )
    assert "v_1|participant_id_sigma" in [rv.name for rv in model.pymc_model.free_RVs]

    estimate = model.find_MLE(allow_unidentified=True, progressbar=False)

    assert estimate is not None
    # `sigma` is absent from the objective, so its gradient is exactly zero and
    # the optimizer leaves it where it started. That is the flat ridge the
    # default refusal exists to warn about, not a crash.
    assert estimate.params["v_1|participant_id_sigma"] == pytest.approx(
        float(model.initvals["v_1|participant_id_sigma"])
    )


def test_find_mle_reports_the_method_that_actually_ran(ddm_data):
    """A blackbox likelihood forces Powell --- the estimate must say so."""
    model = hssm.HSSM(data=ddm_data, model="ddm", loglik_kind="blackbox")
    estimate = model.find_MLE(progressbar=False)

    assert estimate is not None
    assert estimate.method == "Powell"


def test_find_mle_maxeval_failure_does_not_blame_the_starting_point(ddm_model):
    """Running out of budget is a different failure from a stalled gradient.

    The point reported after exhausting `maxeval` is the last iterate, not the
    start, so the did-not-move hint cannot fire and mislead the user into
    changing `start=` when the fix is a larger `maxeval`.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimate = ddm_model.find_MLE(maxeval=3, progressbar=False)

    assert estimate is None
    messages = [str(warning.message) for warning in caught]
    assert any("maxeval" in message for message in messages)
    assert not any("never moved off" in message for message in messages)


def test_find_mle_accepts_progressbar(ddm_model):
    """`progressbar=` works on both methods; `find_MAP`'s docs invite the idiom."""
    estimate = ddm_model.find_MLE(progressbar=False)

    assert estimate is not None


def test_find_mle_raises_on_potentials(ddm_model):
    """`observedlogp` silently drops potentials, so refuse the model outright."""
    with ddm_model.pymc_model:
        pm.Potential("extra_term", ddm_model.pymc_model.free_RVs[0] * 0.0)

    with pytest.raises(ValueError, match="pm.Potential"):
        ddm_model.find_MLE(progressbar=False)


def test_mle_property_guard(ddm_model):
    """`.mle` raises until `find_MLE` has produced an estimate."""
    with pytest.raises(ValueError, match="compute mle first"):
        _ = ddm_model.mle

    ddm_model.find_MLE(progressbar=False)
    assert ddm_model.mle is ddm_model._mle_dict


def test_find_mle_multi_start(ddm_model):
    """`n_starts` tries jittered starts and keeps the best likelihood."""
    estimate = ddm_model.find_MLE(n_starts=3, seed=0, progressbar=False)

    assert estimate is not None
    assert estimate.n_starts == 3
    assert estimate.n_converged >= 1
    assert estimate.logp >= ddm_model.find_MLE(progressbar=False).logp - 1e-6


def test_find_mle_multi_start_compiles_the_objective_once(ddm_model, monkeypatch):
    """The compile is hoisted out of the start loop --- it is the dominant cost.

    Asserted on the call count rather than on wall-clock, which is far too noisy
    to distinguish one PyTensor compile from three.
    """
    calls = []
    original = hssm.optimize.compile_objective

    def spy(*args, **kwargs):
        calls.append(args[:2])
        return original(*args, **kwargs)

    monkeypatch.setattr(hssm.optimize, "compile_objective", spy)
    ddm_model.find_MLE(n_starts=3, seed=0, progressbar=False)

    assert len(calls) == 1


def test_find_mle_does_not_probe_the_gradient_separately(ddm_model, monkeypatch):
    """`compile_objective` already answers the gradient question.

    Probing it beforehand would build the gradient graph of the observed
    log-likelihood twice per call --- the expensive half of both operations.
    """
    calls = []
    monkeypatch.setattr(
        hssm.optimize,
        "gradient_available",
        lambda *args, **kwargs: calls.append(args) or True,
    )
    ddm_model.find_MLE(progressbar=False)

    assert calls == []


def test_find_mle_scores_with_the_compiled_objective(ddm_model, monkeypatch):
    """The scorer must reuse the compiled objective, not compile a second copy.

    `find_MLE` scores candidates on exactly the objective it maximizes, so the
    two compiles would be of the same graph.
    """
    seen = {}
    original = hssm.optimize.make_scorer

    def spy(pymc_model, observed_only=False, function=None):
        seen["function"] = function
        return original(pymc_model, observed_only=observed_only, function=function)

    monkeypatch.setattr(hssm.optimize, "make_scorer", spy)
    ddm_model.find_MLE(progressbar=False)

    assert seen["function"] is not None


def test_find_mle_multi_start_expands_points_with_one_compile(ddm_model, monkeypatch):
    """The point expander is hoisted out of the start loop like the objective.

    Every candidate's raveled result has to be mapped back to a full point, but
    the graph doing it is the same for all of them --- and all but the winning
    candidate's expansion is discarded.
    """
    calls = []
    original = hssm.optimize.make_point_expander

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(hssm.optimize, "make_point_expander", spy)
    ddm_model.find_MLE(n_starts=3, seed=0, progressbar=False)

    assert len(calls) == 1


def test_find_mle_return_raw(ddm_model):
    """`return_raw=True` returns the `(estimate, OptimizeResult)` tuple."""
    point, raw = ddm_model.find_MLE(return_raw=True, progressbar=False)

    assert isinstance(point, PointEstimate)
    assert isinstance(raw, OptimizeResult)


def test_find_mle_strict_raises(ddm_model):
    """`strict=True` turns the MLE failure warning into an error."""
    with pytest.raises(RuntimeError, match="find_MLE failed to converge"):
        ddm_model.find_MLE(maxeval=3, progressbar=False, strict=True)


def test_find_mle_failure_clears_an_earlier_estimate(ddm_model):
    """`find_MLE` owes `model.mle` the same contract `find_MAP` owes `model.map`."""
    ddm_model.find_MLE(progressbar=False)
    assert ddm_model.mle is not None

    with pytest.warns(UserWarning, match="find_MLE failed to converge"):
        assert ddm_model.find_MLE(maxeval=3, progressbar=False) is None

    assert ddm_model._mle_dict is None
    with pytest.raises(ValueError, match="Please compute mle first"):
        _ = ddm_model.mle


def test_find_mle_to_datatree_round_trips(ddm_model):
    """An MLE feeds the ArviZ toolchain exactly as a MAP does."""
    posterior = ddm_model.find_MLE(progressbar=False).to_datatree().posterior

    assert posterior.sizes["chain"] == 1
    assert posterior.sizes["draw"] == 1
    assert set(TRUE_DDM) <= set(posterior.data_vars)


def test_find_mle_float32_warns_and_falls_back(ddm_data):
    """The float32 safety net covers `find_MLE`, not just `find_MAP`."""
    hssm.set_floatX("float32", update_jax=True)
    model = hssm.HSSM(data=ddm_data, model="ddm", loglik_kind="analytical")

    with pytest.warns(UserWarning, match="float32"):
        estimate = model.find_MLE(progressbar=False)

    assert estimate is not None
    assert estimate.method == "Powell"


@pytest.mark.parametrize("method_name", ["find_MAP", "find_MLE"])
def test_n_starts_below_one_is_rejected(ddm_model, method_name):
    """`n_starts=0` would run one start while reporting `n_starts=0`."""
    with pytest.raises(ValueError, match="n_starts must be at least 1"):
        getattr(ddm_model, method_name)(n_starts=0, progressbar=False)


def test_find_mle_refreshes_extra_fields(ddm_model, monkeypatch):
    """The extra-fields refresh applies to MLE too, not just MAP."""
    calls = []
    _spy_on_extra_fields(ddm_model, monkeypatch, calls)
    ddm_model.find_MLE(progressbar=False)

    assert calls[:2] == ["check", "update"]


# endregion


# region ===== standard errors =====
def test_standard_errors_are_reported_on_the_constrained_scale(ddm_model):
    """`se=True` adds finite, plausibly-sized standard errors."""
    estimate = ddm_model.find_MAP(se=True, progressbar=False)

    assert estimate.se is not None
    frame = estimate.to_dataframe()
    assert "se" in frame.columns
    assert np.all(np.isfinite(frame["se"].to_numpy()))
    # With 500 trials the errors should be small but non-zero.
    assert np.all(frame["se"].to_numpy() > 0)
    assert np.all(frame["se"].to_numpy() < 0.5)


def test_standard_errors_for_mle_use_the_observed_information(ddm_model):
    """MLE standard errors come from `observedlogp`, not the joint density."""
    estimate = ddm_model.find_MLE(se=True)

    assert estimate.se is not None
    assert np.all(np.isfinite(list(estimate.se.values())))


def test_standard_errors_are_nan_when_the_hessian_is_not_negative_definite(
    ddm_model, monkeypatch
):
    """A Hessian that is not negative definite must yield NaN, not a number.

    Driven through the Cholesky gate rather than by hunting for a model with a
    degenerate optimum: the branch under test is "the curvature check failed",
    and how it failed makes no difference to what is reported.
    """

    def not_negative_definite(matrix):
        raise np.linalg.LinAlgError("Matrix is not positive definite")

    monkeypatch.setattr(np.linalg, "cholesky", not_negative_definite)

    with pytest.warns(UserWarning, match="singular or not negative definite"):
        estimate = ddm_model.find_MAP(se=True, progressbar=False)

    assert estimate.se is not None
    assert np.all(np.isnan(np.asarray(list(estimate.se.values()), dtype=float)))


def test_standard_errors_unavailable_without_a_gradient(ddm_data):
    """Blackbox likelihoods have no gradient: warn and return the estimate."""
    model = hssm.HSSM(data=ddm_data, model="ddm", loglik_kind="blackbox")

    with pytest.warns(UserWarning, match="Standard errors are not available"):
        estimate = model.find_MAP(se=True, progressbar=False)

    assert estimate is not None
    assert estimate.se is None


# endregion


# region ===== PointEstimate =====
def test_point_estimate_is_a_dict(ddm_model):
    """Existing code paths type-check the estimate with `isinstance(..., dict)`."""
    estimate = ddm_model.find_MAP(progressbar=False)

    assert isinstance(estimate, dict)
    assert estimate["a"] == estimate.params["a"]
    assert set(estimate.params) == {"v", "a", "z", "t"}
    # The raw mapping keeps the transformed names and deterministics.
    assert "t_log__" in estimate


def test_point_estimate_to_dataframe(ddm_model):
    """`to_dataframe` gives one row per scalar entry, no `se` column by default."""
    frame = ddm_model.find_MAP(progressbar=False).to_dataframe()

    assert list(frame.columns) == ["estimate"]
    assert set(frame.index) == {"v", "a", "z", "t"}


def test_point_estimate_to_datatree_feeds_arviz(ddm_model):
    """The estimate must round-trip into the ArviZ/HSSM toolchain."""
    estimate = ddm_model.find_MAP(progressbar=False)
    datatree = estimate.to_datatree()

    assert datatree.posterior.sizes["chain"] == 1
    assert datatree.posterior.sizes["draw"] == 1

    summary = az.summary(datatree)
    # arviz 1.3 column names --- the legacy `hdi_3%`/`hdi_97%` no longer exist.
    assert {"mean", "sd", "eti89_lb", "eti89_ub", "ess_bulk", "r_hat"} <= set(
        summary.columns
    )
    # A single draw has no sampling distribution: sd collapses and the
    # convergence diagnostics are undefined. Documented, not a bug.
    assert np.allclose(summary["sd"].to_numpy(), 0.0)
    assert np.all(np.isnan(summary["r_hat"].to_numpy().astype(float)))

    ddm_model.restore_traces(datatree)
    assert ddm_model.traces is datatree


def test_point_estimate_to_datatree_keeps_vector_dims(hierarchical_model):
    """Vector-valued (group-level) parameters keep their dims and coords."""
    estimate = hierarchical_model.find_MAP(progressbar=False)
    posterior = estimate.to_datatree().posterior

    offsets = posterior["v_1|participant_id_offset"]
    assert offsets.sizes["chain"] == 1
    assert offsets.sizes["draw"] == 1
    assert offsets.ndim == 3


def test_point_estimate_survives_pickle(ddm_model):
    """Extra attributes on a dict subclass are easy to lose in serialization."""
    estimate = ddm_model.find_MAP(se=True, progressbar=False)
    restored = pickle.loads(pickle.dumps(estimate))

    assert isinstance(restored, PointEstimate)
    assert restored.kind == estimate.kind
    assert restored.success == estimate.success
    assert restored.method == estimate.method
    assert restored.logp == pytest.approx(estimate.logp)
    assert restored.se.keys() == estimate.se.keys()
    assert restored.params.keys() == estimate.params.keys()
    assert dict(restored).keys() == dict(estimate).keys()


def test_point_estimate_copy_keeps_the_metadata(ddm_model):
    """`dict.copy` on a dict subclass returns a plain dict --- override it."""
    estimate = ddm_model.find_MAP(se=True, progressbar=False)
    duplicate = estimate.copy()

    assert isinstance(duplicate, PointEstimate)
    assert duplicate.kind == estimate.kind
    assert duplicate.method == estimate.method
    assert duplicate.params == estimate.params
    assert duplicate.se.keys() == estimate.se.keys()
    # A copy, not an alias: mutating one must not reach the other.
    duplicate.params["a"] = 999.0
    assert estimate.params["a"] != 999.0


def test_point_estimate_repr_truncates_long_estimates(hierarchical_model):
    """A hierarchy has one row per offset; `repr` must not dump them all."""
    estimate = hierarchical_model.find_MAP(progressbar=False)
    estimate.params = {f"p{index}": float(index) for index in range(REPR_MAX_ROWS + 5)}

    text = repr(estimate)

    assert "and 5 more row(s)" in text
    assert f"p{REPR_MAX_ROWS - 1}" in text
    assert f"p{REPR_MAX_ROWS}" not in text
    # The full frame is still reachable, which is what the hint promises.
    assert len(estimate.to_dataframe()) == REPR_MAX_ROWS + 5


def test_empty_estimate_does_not_read_as_not_computed(ddm_model):
    """`.map`'s guard is `is None`; an empty-but-computed estimate is computed."""
    ddm_model._map_dict = PointEstimate({})

    assert ddm_model.map == {}


def test_point_estimate_is_exported_at_the_top_level(ddm_model):
    """`isinstance` checks need the class without reaching into a submodule."""
    assert hssm.PointEstimate is PointEstimate


# endregion


# region ===== sample() integration =====
def test_sample_initvals_map_raises_when_map_fails(ddm_model, monkeypatch):
    """A failed MAP must not silently degrade to PyMC's default start.

    `find_MAP` caches nothing on failure, so falling through would hand bambi
    `initvals=None` --- sampling from a point the user never asked for.
    """
    monkeypatch.setattr(type(ddm_model), "find_MAP", lambda self, **kwargs: None)

    with pytest.raises(RuntimeError, match="did not converge"):
        ddm_model.sample(initvals="map", draws=1, tune=1, chains=1, cores=1)


def test_sample_initvals_map_passes_constrained_params_only(ddm_model, monkeypatch):
    """The handoff should carry the constrained free parameters and nothing else."""
    seen = {}

    def fake_fit(*args, **kwargs):
        seen["initvals"] = kwargs["initvals"]
        raise RuntimeError("stop here")

    ddm_model.find_MAP(progressbar=False)
    monkeypatch.setattr(ddm_model.model, "fit", fake_fit)

    with pytest.raises(RuntimeError, match="stop here"):
        ddm_model.sample(initvals="map", draws=1, tune=1, chains=1, cores=1)

    assert set(seen["initvals"]) == {"v", "a", "z", "t"}


def test_laplace_sampler_warns(ddm_model, monkeypatch, caplog):
    """Bambi's laplace path cannot receive HSSM's initial values --- warn."""

    def fake_fit(*args, **kwargs):
        raise RuntimeError("stop here")

    monkeypatch.setattr(ddm_model.model, "fit", fake_fit)

    with caplog.at_level("WARNING", logger="hssm"):
        with pytest.raises(RuntimeError, match="stop here"):
            ddm_model.sample(sampler="laplace")

    assert "laplace" in caplog.text
    assert "find_MAP" in caplog.text


# endregion


# region ===== helper units =====
def test_audit_result_accepts_a_converged_run_that_did_not_move():
    """A start already at the optimum returns `nit=0` --- that is not a failure."""
    point = {"a": np.array(1.5)}
    result = OptimizeResult(x=np.zeros(1), fun=-1.0, success=True, nit=0)

    assert audit_result(result, point, point) == []


def test_audit_result_flags_a_none_result():
    """`opt_result is None` is the only signal for interrupt / maxeval."""
    reasons = audit_result(None, {"a": np.array(1.5)}, {"a": np.array(2.0)})

    assert any("maxeval" in reason for reason in reasons)


def test_points_equal_requires_the_reference_to_be_covered():
    """A partial reference cannot establish that the optimizer did not move."""
    point = {"a": np.array(1.5), "t": np.array(0.3)}

    assert points_equal(point, {"a": np.array(1.5), "t": np.array(0.3)})
    # `t` moved, so the points differ even though `a` matches.
    assert not points_equal(point, {"a": np.array(1.5), "t": np.array(0.9)})
    # A key the point does not carry means the comparison is not decidable.
    assert not points_equal(point, {"a": np.array(1.5), "z": np.array(0.5)})
    assert not points_equal(point, {})


def test_describe_failures_collapses_identical_reasons():
    """Five starts failing the same way should read as one sentence, not five."""
    text = describe_failures([["gradient is non-finite"]] * 5)

    assert text == "all 5 starts failed the same way: gradient is non-finite"


def test_describe_failures_keeps_distinct_reasons_apart():
    """When the starts fail differently, that difference *is* the diagnosis."""
    text = describe_failures([["rejected by PyMC"], ["ran out of maxeval"]])

    assert "start 1: rejected by PyMC" in text
    assert "start 2: ran out of maxeval" in text


def test_describe_failures_handles_no_failures():
    """`report_failure` is also reached when every start was skipped."""
    assert describe_failures([]) == "no start converged"


def test_audit_result_adds_the_gradient_hint_only_on_failure():
    """The did-not-move hint sharpens a failure; it never creates one."""
    point = {"a": np.array(1.5)}

    reasons = audit_result(failed_optimize_result(), point, point)
    assert any("never moved off its starting point" in reason for reason in reasons)

    moved = audit_result(failed_optimize_result(), point, {"a": np.array(2.0)})
    assert not any("never moved off" in reason for reason in moved)


def test_jittered_starts_yields_the_original_first():
    """The unjittered start is always tried, so `n_starts=1` is a no-op."""
    initvals = {"t": np.array(0.025), "a": np.array(1.5)}
    starts = list(jittered_starts(initvals, 3, np.random.default_rng(0)))

    assert len(starts) == 3
    assert starts[0] == initvals
    # Relative jitter cannot push a positive parameter through zero.
    assert all(candidate["t"] > 0 for candidate in starts)
    assert starts[1] != starts[2]


def test_jittered_starts_does_not_truncate_integer_valued_entries():
    """An integer-dtype entry must still be jittered by the requested fraction.

    Casting the perturbed value back to the input dtype would round `2 * 1.02`
    to `2` --- no jitter at all, while the estimate still claims `n_starts`
    starts --- or down to `1`, a 50% jump rather than the requested 10%.
    """
    starts = list(
        jittered_starts({"a": np.array(2)}, 6, np.random.default_rng(1), jitter=0.1)
    )
    jittered = [float(candidate["a"]) for candidate in starts[1:]]

    assert all(value != 2.0 for value in jittered)
    assert all(1.8 <= value <= 2.2 for value in jittered)


def test_points_equal_sees_an_untouched_zero_as_unmoved():
    """A zero entry must survive the transform round trip as "did not move".

    Zero is the usual start for a hierarchical model's group-level offsets, and
    the round trip through a transform and its inverse can return `1e-17`. A
    purely relative tolerance is a tolerance of exactly zero there, which would
    read that as movement and suppress the non-finite-gradient hint.
    """
    assert points_equal({"offset": np.array(1e-17)}, {"offset": np.array(0.0)})
    assert not points_equal({"offset": np.array(0.5)}, {"offset": np.array(0.0)})


def test_scoring_a_constrained_only_estimate_says_what_is_wrong(ddm_model):
    """A point missing its transformed entries is unscoreable, not zero-density.

    `include_transformed=False` drops exactly the `*_log__`/`*_interval__`
    entries the scorer indexes, so the failure has to name that cause. Silently
    returning `-inf` would report the estimate as infinitely improbable.
    """
    estimate = ddm_model.find_MAP(include_transformed=False, progressbar=False)

    with pytest.raises(KeyError, match="include_transformed=False"):
        score_point(ddm_model.pymc_model, estimate)


def test_failure_message_does_not_advertise_return_raw():
    """A failed run returns None, so `return_raw=True` is not an escape hatch."""
    with pytest.warns(UserWarning, match="find_MAP failed to converge") as caught:
        hssm.optimize.report_failure("MAP", [["it failed"]], strict=False)

    message = str(caught[0].message)
    assert "return_raw" not in message
    assert "n_starts" in message


# endregion


# region ===== slow =====
@pytest.mark.slow
def test_find_map_full_hierarchical_model():
    """The full `cavanagh_theta` hierarchical fit --- the original repro.

    The fast suite exercises a three-participant slice; this covers the whole
    19-parameter model, which takes far too many L-BFGS-B iterations for a fast
    test.
    """
    model = hssm.HSSM(
        data=hssm.load_data("cavanagh_theta"),
        model="ddm",
        loglik_kind="analytical",
        include=[{"name": "v", "formula": "v ~ 1 + (1|participant_id)"}],
    )
    estimate = model.find_MAP(progressbar=False)

    assert estimate is not None
    assert estimate.success
    assert estimate.params["a"] != pytest.approx(float(model.initvals["a"]))
    # The group-level scale should not collapse: hierarchical MAP is well
    # behaved once the start is right.
    assert estimate.params["v_1|participant_id_sigma"] > 0.1


@pytest.mark.slow
def test_point_estimate_survives_cloudpickle(ddm_model):
    """`save_model` uses cloudpickle, which must preserve the extra attributes.

    Extra attributes on a `dict` subclass are exactly the kind of state that
    quietly disappears through a serializer that treats the object as a plain
    mapping.
    """
    estimate = ddm_model.find_MAP(se=True, progressbar=False)
    restored = cloudpickle.loads(cloudpickle.dumps(estimate))

    assert isinstance(restored, PointEstimate)
    assert restored.kind == "MAP"
    assert restored.success
    assert restored.logp == pytest.approx(estimate.logp)
    assert restored.se.keys() == estimate.se.keys()
    assert restored.dims == estimate.dims


@pytest.mark.slow
def test_save_load_does_not_carry_a_point_estimate(ddm_model, tmp_path):
    """Model persistence rebuilds from constructor args and drops inference state.

    `HSSMBase.__getstate__` stores only the constructor arguments, so a saved
    model carries no traces, no VI approximation and no point estimate --- those
    are persisted (or not) separately. This pins that contract down for point
    estimates so the behaviour is a decision rather than a surprise.
    """
    ddm_model.find_MAP(progressbar=False)
    ddm_model.save_model(
        model_name="point_estimate_roundtrip",
        base_path=tmp_path,
        allow_absolute_base_path=True,
    )

    loaded = hssm.HSSM.load_model(path=tmp_path / "point_estimate_roundtrip")

    assert loaded._map_dict is None
    with pytest.raises(ValueError, match="compute map first"):
        _ = loaded.map


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["ddm", "angle"])
def test_find_map_approx_differentiable(model_name):
    """Point estimation works through the ONNX/JAX likelihoods too."""
    theta = {
        "ddm": [0.5, 1.5, 0.5, 0.3],
        "angle": [0.5, 1.5, 0.5, 0.3, 0.2],
    }[model_name]
    simulated = simulator(theta, model=model_name, n_samples=500)
    data = pd.DataFrame(
        np.column_stack([simulated["rts"][:, 0], simulated["choices"][:, 0]]),
        columns=["rt", "response"],
    )
    model = hssm.HSSM(data=data, model=model_name, loglik_kind="approx_differentiable")

    estimate = model.find_MAP(se=True, progressbar=False)

    assert estimate is not None
    assert estimate.success
    assert estimate.params["a"] == pytest.approx(1.5, abs=0.3)
    assert estimate.params["v"] == pytest.approx(0.5, abs=0.3)
    # Second derivatives do not exist symbolically for these Ops; standard
    # errors come from finite differences of the analytic gradient instead.
    assert estimate.se is not None


@pytest.mark.slow
def test_find_map_and_mle_on_a_regression_model(cavanagh_test):
    """A fixed-effects regression is fine for MLE --- only group terms are not."""
    model = hssm.HSSM(
        data=cavanagh_test,
        model="ddm",
        loglik_kind="analytical",
        include=[{"name": "v", "formula": "v ~ 1 + theta"}],
    )

    map_estimate = model.find_MAP(progressbar=False)
    mle_estimate = model.find_MLE(progressbar=False)

    assert map_estimate is not None
    assert mle_estimate is not None
    assert "v_theta" in map_estimate.params
    assert (
        score_point(model.pymc_model, mle_estimate, observed_only=True)
        >= score_point(model.pymc_model, map_estimate, observed_only=True) - 1e-3
    )


# endregion

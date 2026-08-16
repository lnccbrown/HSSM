"""Point estimation (MAP / MLE) for HSSM models.

This module holds the machinery shared by ``hssm.HSSM.find_MAP`` and
``hssm.HSSM.find_MLE``:

* ``PointEstimate`` --- the ``dict`` subclass both methods return. It keeps
  the plain-mapping behaviour older code relies on (``sample(initvals="map")``
  checks ``isinstance(initvals, dict)``) while carrying optimizer metadata and
  ArviZ-friendly exporters.
* the start-value, failure-auditing, ``float32`` and standard-error helpers.
* ``optimize_objective``, a small ``scipy.optimize.minimize`` driver used by
  ``find_MLE``. ``find_MAP`` keeps delegating to ``pymc.find_MAP``, which
  already implements the same loop for the *joint* log-density; only the
  objective differs for MLE, and PyMC offers no hook to swap it.

Of these, only ``PointEstimate`` (re-exported as ``hssm.PointEstimate``) and
``score_point`` are meant for users; see ``__all__``. The rest is implementation
shared with ``hssm.base``.

Notes
-----
Point estimation is gradient-based and therefore precision sensitive. Under
``hssm.set_floatX("float32")`` the gradient noise sits above L-BFGS-B's default
tolerances, so the optimizer stops early *and reports success*. The helpers
below warn unconditionally in that configuration and fall back to the
gradient-free ``"Powell"`` method unless the caller picked one explicitly.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import arviz as az
import numpy as np
import pandas as pd
import pytensor
import pytensor.gradient as tg
import pytensor.tensor as pt
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.initial_point import make_initial_point_fn
from pymc.model.transform.conditioning import remove_value_transforms
from pymc.progress_bar import CustomProgress, default_progress_theme
from pymc.pytensorf import rewrite_pregrad
from pymc.util import get_default_varnames
from rich.console import Console
from rich.progress import Progress, TextColumn
from scipy import optimize as sp_optimize

if TYPE_CHECKING:
    from pymc import Model as PyMCModel
    from xarray import DataTree

_logger = logging.getLogger("hssm")

#: The supported surface of this module. Everything else defined here is
#: machinery shared with ``hssm.base`` --- importable, but not API: it may
#: change shape whenever the two ``find_*`` methods need it to.
__all__ = [
    "PointEstimate",
    "score_point",
]

#: Exceptions PyTensor raises when a graph has no usable gradient. Mirrors the
#: tuple ``pymc.find_MAP`` catches, so blackbox likelihoods take the same
#: gradient-free path here as they do there.
NO_GRADIENT_ERRORS = (AttributeError, NotImplementedError, tg.NullTypeGradError)

#: PyMC's own default. Kept as a literal because HSSM uses ``method=None`` as
#: its own "user did not choose" sentinel and must never forward that ``None``
#: to ``scipy.optimize.minimize`` (where it silently auto-selects a *gradient*
#: method while ``jac=True`` is still set).
DEFAULT_METHOD = "L-BFGS-B"

#: Must match ``pymc.find_MAP``'s case-sensitive ``method != "Powell"`` guard.
GRADIENT_FREE_METHOD = "Powell"

_FLOAT32_WARNING = (
    "Point estimation is running with `floatX == 'float32'`. Gradient noise at "
    "single precision exceeds the optimizer's default tolerances, so the "
    "optimizer can stop far from the optimum *and still report success*. The "
    "returned estimate should be treated as approximate. Call "
    "`hssm.set_floatX('float64')` before building the model for reliable "
    "results."
)

#: Rows of ``PointEstimate.to_dataframe`` shown in ``repr``. A hierarchical
#: model has one row per group-level offset, and an unbounded ``to_string()``
#: would dump hundreds of them into the REPL.
REPR_MAX_ROWS = 20

#: Above this many scalar parameters, ``standard_errors`` logs what it is
#: about to do: the finite-difference Hessian costs ``2 * size`` full gradient
#: evaluations plus a ``size x size`` inverse, which can take minutes.
_SE_SIZE_HINT = 100


# region ===== PointEstimate =====
def _rebuild_point_estimate(mapping: dict, state: dict) -> "PointEstimate":
    """Reconstruct a ``PointEstimate`` during unpickling."""
    obj = PointEstimate.__new__(PointEstimate)
    dict.__init__(obj, mapping)
    obj.__dict__.update(state)
    return obj


class PointEstimate(dict):
    """A point estimate of the model parameters, with optimizer metadata.

    This is a ``dict`` subclass, so it can be passed anywhere a plain point
    dictionary was accepted before (notably ``model.sample(initvals=...)``). The
    mapping itself is what the optimizer returned --- constrained values,
    transformed value-variable names (``t_log__``, ``z_interval__``) and
    deterministics --- while ``params`` holds only the constrained free
    parameters.

    Attributes
    ----------
    kind
        ``"MAP"`` or ``"MLE"``.
    params
        The constrained free parameters only, with transformed duplicates and
        deterministics dropped.
    success
        Whether the optimizer converged.
    message
        The optimizer's termination message.
    logp
        The value of the maximized objective at the returned point --- the joint
        log-density for ``"MAP"``, the observed log-likelihood for ``"MLE"``.
    method
        The ``scipy.optimize.minimize`` method that actually produced the
        estimate. This is not necessarily the one that was requested: a
        likelihood without a gradient (``blackbox``) forces the gradient-free
        fallback, and so does ``float32`` precision.
    n_starts
        How many starting points were tried.
    n_converged
        How many of those starts converged.
    se
        Square roots of the diagonal of the inverse negative Hessian at the
        estimate, keyed like ``params``, or ``None`` if they were not
        requested (or could not be computed). For ``"MLE"`` these are standard
        errors in the usual sense --- the observed-information approximation to
        the sampling distribution. For ``"MAP"`` they are posterior standard
        deviations under a Laplace (normal) approximation around the mode, which
        is a different object that happens to be computed the same way.
    opt_result
        The raw ``scipy.optimize.OptimizeResult``, or ``None`` when the
        optimizer was interrupted or exhausted ``maxeval``.

    Notes
    -----
    ``dict.copy`` on a ``dict`` subclass returns a plain ``dict``, which would
    silently drop every attribute above. ``copy`` is overridden to preserve
    them; ``pickle`` and ``cloudpickle`` are handled by ``__reduce__``.
    """

    _META_FIELDS = (
        "kind",
        "params",
        "success",
        "message",
        "logp",
        "method",
        "n_starts",
        "n_converged",
        "se",
        "opt_result",
        "dims",
        "coords",
    )

    def __init__(
        self,
        point: dict[str, Any],
        *,
        kind: Literal["MAP", "MLE"] = "MAP",
        params: dict[str, Any] | None = None,
        success: bool = True,
        message: str = "",
        logp: float = np.nan,
        method: str | None = None,
        n_starts: int = 1,
        n_converged: int = 1,
        se: dict[str, Any] | None = None,
        opt_result: Any = None,
        dims: dict[str, list[str]] | None = None,
        coords: dict[str, list[Any]] | None = None,
    ):
        super().__init__(point)
        self.kind = kind
        self.params = dict(point) if params is None else dict(params)
        self.success = success
        self.message = message
        self.logp = logp
        self.method = method
        self.n_starts = n_starts
        self.n_converged = n_converged
        self.se = None if se is None else dict(se)
        self.opt_result = opt_result
        self.dims = dict(dims) if dims else {}
        self.coords = dict(coords) if coords else {}

    def __reduce__(self):
        """Preserve the metadata attributes through ``pickle``/``cloudpickle``."""
        state = {field: getattr(self, field) for field in self._META_FIELDS}
        return (_rebuild_point_estimate, (dict(self), state))

    def copy(self) -> "PointEstimate":
        """Return a shallow copy that keeps the optimizer metadata.

        ``dict.copy`` would return a plain ``dict``, silently dropping
        ``params``, ``se`` and everything else that makes this more than
        a mapping.

        Returns
        -------
        PointEstimate
            A new estimate with the same mapping and metadata.
        """
        # Every entry of ``_META_FIELDS`` is also an ``__init__`` keyword, so the
        # constructor is the single place that knows how to rebuild the object.
        return PointEstimate(
            dict(self), **{field: getattr(self, field) for field in self._META_FIELDS}
        )

    __copy__ = copy

    def __repr__(self) -> str:  # noqa: D105
        status = "converged" if self.success else "DID NOT CONVERGE"
        header = f"{self.kind} estimate ({self.method}, {status}, logp={self.logp:.4g})"
        frame = self.to_dataframe()
        if len(frame) > REPR_MAX_ROWS:
            hidden = len(frame) - REPR_MAX_ROWS
            body = (
                f"{frame.head(REPR_MAX_ROWS).to_string()}\n"
                f"... and {hidden} more row(s); use `.to_dataframe()` for all."
            )
        else:
            body = frame.to_string()
        return f"{header}\n{body}"

    def to_dataframe(self) -> pd.DataFrame:
        """Return the estimate as a tidy DataFrame, one row per scalar entry.

        Vector-valued parameters are expanded to ``name[i]`` rows. An ``se``
        column is included only when standard errors were computed.

        Returns
        -------
        pd.DataFrame
            Indexed by parameter name, with an ``estimate`` column and,
            optionally, an ``se`` column.
        """
        names: list[str] = []
        estimates: list[float] = []
        errors: list[float] = []

        for name, value in self.params.items():
            array = np.atleast_1d(np.asarray(value, dtype=float))
            se_array = (
                np.atleast_1d(np.asarray(self.se[name], dtype=float)).ravel()
                if self.se is not None and name in self.se
                else None
            )
            flat = array.ravel()
            scalar = np.ndim(value) == 0 or flat.size == 1
            for i, entry in enumerate(flat):
                names.append(name if scalar else f"{name}[{i}]")
                estimates.append(float(entry))
                errors.append(
                    float(se_array[i])
                    if se_array is not None and i < se_array.size
                    else np.nan
                )

        frame = pd.DataFrame(
            {"estimate": estimates, "se": errors},
            index=pd.Index(names, name="parameter"),
        )
        if frame["se"].isna().all():
            frame = frame.drop(columns=["se"])
        return frame

    def to_datatree(self) -> "DataTree":
        """Return the estimate as a 1-chain, 1-draw posterior ``DataTree``.

        This makes the point estimate usable with the rest of the ArviZ/HSSM
        toolchain: ``az.summary``, ``model.restore_traces``,
        ``model.sample_posterior_predictive`` and the plotting functions all
        accept the result.

        Returns
        -------
        DataTree
            A ``DataTree`` with a single ``posterior`` group of shape
            ``(chain=1, draw=1, ...)``.

        Notes
        -----
        With a single draw the posterior summaries degenerate: ``az.summary``
        reports a standard deviation of ``0`` and ``NaN`` for ``ess_bulk``,
        ``ess_tail`` and ``r_hat``. That is expected --- there is no sampling
        distribution to summarize --- not a sign that something went wrong.
        """
        data = {
            name: np.asarray(value)[np.newaxis, np.newaxis, ...]
            for name, value in self.params.items()
        }
        dims = {name: dim for name, dim in self.dims.items() if name in data}
        used = {dim for entry in dims.values() for dim in entry}
        coords = {key: value for key, value in self.coords.items() if key in used}
        dataset = az.dict_to_dataset(
            data, coords=cast("Any", coords), dims=cast("Any", dims)
        )
        return az.convert_to_datatree(dataset)


# endregion


# region ===== shared helpers =====
def resolve_method(method: str | None, kind: str) -> str:
    """Resolve the optimizer method, applying the ``float32`` safety net.

    Parameters
    ----------
    method
        The user's choice, or ``None`` if they did not make one. HSSM keeps
        ``None`` as its own sentinel because PyMC's default is the *string*
        ``"L-BFGS-B"``, which makes "user chose L-BFGS-B" indistinguishable from
        "user chose nothing".
    kind
        ``"MAP"`` or ``"MLE"``, used in the warning text.

    Returns
    -------
    str
        The method to use.
    """
    if pytensor.config.floatX != "float32":
        return method or DEFAULT_METHOD

    fallback = method is None
    warnings.warn(
        f"find_{kind} was called with float32 precision. {_FLOAT32_WARNING}"
        + (
            f" Falling back to the gradient-free '{GRADIENT_FREE_METHOD}' "
            "method, which is more robust at this precision; pass an explicit "
            "`method=` to override."
            if fallback
            else ""
        ),
        UserWarning,
        # resolve_method <- _point_estimate_setup <- find_MAP/find_MLE <- user
        stacklevel=4,
    )
    # Narrow on `method is None` rather than `fallback`: mypy does not carry the
    # narrowing through the intermediate boolean.
    return GRADIENT_FREE_METHOD if method is None else method


def gradient_available(pymc_model: "PyMCModel", objective: Any = None) -> bool:
    """Check whether a symbolic gradient of ``objective`` can be built.

    Only the *graph* is built, not a compiled function, and it raises exactly
    where ``pymc.find_MAP`` does --- which is what makes it a faithful predictor
    of whether the optimizer will be able to use gradients, rather than a guess
    from ``loglik_kind``.

    It is not free: building the log-density and its gradient graph costs on the
    order of a tenth of a second for an ONNX/JAX model, a few percent of the
    optimization that follows. That is the price of recording the method that
    really ran instead of the one that was asked for. Callers that already know
    the answer skip it by passing the gradient-free method explicitly.

    Parameters
    ----------
    pymc_model
        The model whose continuous value variables the gradient is taken over.
    objective
        The objective to differentiate. Defaults to the joint log-density, the
        objective ``find_MAP`` maximizes.

    Returns
    -------
    bool
        ``True`` unless the likelihood exposes no gradient (``blackbox``).
    """
    objective = pymc_model.logp(jacobian=False) if objective is None else objective
    try:
        flat_gradient(objective, pymc_model.continuous_value_vars)
    except NO_GRADIENT_ERRORS:
        return False
    return True


def method_for_gradient(method: str, has_gradient: bool) -> str:
    """Return the method the optimizer will *actually* run.

    Split from ``resolve_gradient_method`` so that a caller which has *already*
    established gradient availability --- ``find_MLE`` learns it from
    ``compile_objective``, which has to build the gradient anyway --- can settle
    the method without building the gradient graph a second time.

    Parameters
    ----------
    method
        The method resolved so far (never ``None``).
    has_gradient
        Whether a gradient of the objective could be built.

    Returns
    -------
    str
        ``method``, or ``GRADIENT_FREE_METHOD`` if no gradient is available.
    """
    if method == GRADIENT_FREE_METHOD or has_gradient:
        return method

    _logger.warning(
        "No gradient is available for this likelihood, so '%s' cannot be used. "
        "Falling back to the gradient-free '%s' method.",
        method,
        GRADIENT_FREE_METHOD,
    )
    return GRADIENT_FREE_METHOD


def resolve_gradient_method(
    pymc_model: "PyMCModel", method: str, objective: Any = None
) -> str:
    """Return the method the optimizer will *actually* run, probing the gradient.

    Both ``pymc.find_MAP`` and ``optimize_objective`` silently swap in
    the gradient-free method when the likelihood has no gradient, but neither
    reports that it did --- so without this the estimate's recorded ``method``
    would name an optimizer that never ran.

    Parameters
    ----------
    pymc_model
        The model being optimized.
    method
        The method resolved so far (never ``None``).
    objective
        The objective that will be maximized. Defaults to the joint
        log-density.

    Returns
    -------
    str
        ``method``, or ``GRADIENT_FREE_METHOD`` if no gradient is available.
    """
    # Short-circuited before the probe: the answer cannot change the method, and
    # `gradient_available` is the expensive part of this call.
    if method == GRADIENT_FREE_METHOD:
        return method
    return method_for_gradient(method, gradient_available(pymc_model, objective))


def points_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Check whether ``right`` is fully covered by ``left`` and holds equal values.

    ``right`` is the *reference* (the start): every one of its entries must be
    present in ``left`` and match. Callers use this to decide whether the
    optimizer moved at all, so the reference has to be the *complete* start
    point --- which is why ``find_MAP`` expands a user's partial
    ``start=`` through ``make_start_point`` before auditing. Comparing only
    the keys the two happen to share would otherwise call a run "unmoved" on the
    evidence of one unchanged parameter, while the ones the user did not name
    are exactly those that may have moved.

    Compared up to a tight relative tolerance rather than bit-for-bit: the
    optimizer's point comes back through a transform and its inverse, so an
    untouched parameter can differ from the start in the last ulp. This only
    feeds a diagnostic message, so a near-match is the useful reading.

    The absolute tolerance carries that reading through zero. A relative
    tolerance around ``0.0`` is a tolerance of exactly zero, so the round trip
    turning an untouched ``0.0`` into ``1e-17`` would read as movement --- and
    zero is the *usual* start for the group-level offsets of a hierarchical
    model, which is precisely where the hint below is worth having. It is set
    far below any step an optimizer that really moved would take.
    """
    shared = set(right)
    if not shared or not shared <= set(left):
        return False
    return all(
        np.allclose(
            np.asarray(left[key], dtype=float),
            np.asarray(right[key], dtype=float),
            rtol=1e-10,
            atol=1e-12,
        )
        for key in shared
    )


def audit_result(
    opt_result: Any,
    point: dict[str, Any],
    start: dict[str, Any],
) -> list[str]:
    """Return the reasons an optimizer run should be considered a failure.

    An empty list means the run is trustworthy.

    Parameters
    ----------
    opt_result
        A ``scipy.optimize.OptimizeResult``, or ``None`` --- which is the *only*
        signal that the optimizer was interrupted or that ``maxeval`` was
        exhausted.
    point
        The point the optimizer returned.
    start
        The *complete* point the optimizer started from. A partial point would
        make the did-not-move hint below fire on incomplete evidence.

    Notes
    -----
    A point identical to the start is not evidence of failure on its own: a
    start that already sits at the optimum legitimately returns ``success=True``
    with ``nit=0``. It only sharpens the diagnosis of a run that has *already*
    failed on another criterion, where it points at a non-finite gradient at the
    start.

    That includes a run that ran out of ``maxeval``, which reports its *last
    iterate* rather than its start --- so a point equal to the start there is
    not the budget being too small but the very first evaluation having a
    non-finite gradient, which is exactly what the hint says. In practice a
    budget-limited run has moved and the hint stays silent.
    """
    reasons: list[str] = []

    if opt_result is None:
        reasons.append(
            "the optimizer was interrupted or exhausted its `maxeval` budget "
            "(raise `maxeval=` if the run was still making progress)"
        )
    else:
        if not bool(getattr(opt_result, "success", True)):
            message = str(getattr(opt_result, "message", "")).strip()
            reasons.append(
                "the optimizer reported failure"
                + (f" ({message})" if message else "")
                + f" after {getattr(opt_result, 'nit', '?')} iteration(s)"
            )
        objective = getattr(opt_result, "fun", None)
        if objective is not None and not np.all(
            np.isfinite(np.asarray(objective, dtype=float))
        ):
            reasons.append("the objective at the returned point is not finite")

    if reasons and points_equal(point, start):
        reasons.append(
            "the optimizer never moved off its starting point, which usually "
            "means the gradient of the log-density is non-finite there. Pass a "
            "different `start=` (HSSM's own `model.initvals` is the default) or "
            "try `method='Powell'`"
        )

    return reasons


def describe_failures(failures: list[list[str]]) -> str:
    """Render one message covering every start that failed.

    Reporting only the last start's reasons --- which is what a single
    overwritten variable gives you --- hides the case where the starts failed
    for *different* reasons, and that difference is the diagnosis. Identical
    reasons are collapsed so the common case stays short.
    """
    if not failures:
        return "no start converged"

    unique: list[list[str]] = []
    for reasons in failures:
        if reasons not in unique:
            unique.append(reasons)

    if len(unique) == 1:
        detail = "; ".join(unique[0])
        if len(failures) > 1:
            return f"all {len(failures)} starts failed the same way: {detail}"
        return detail

    return " | ".join(
        f"start {index + 1}: " + "; ".join(reasons)
        for index, reasons in enumerate(failures)
    )


def report_failure(kind: str, failures: list[list[str]], strict: bool) -> None:
    """Warn --- or raise under ``strict`` --- about failed optimizer runs.

    Parameters
    ----------
    kind
        ``"MAP"`` or ``"MLE"``.
    failures
        One list of reasons per start that failed, in the order they were tried.
    strict
        Raise a ``RuntimeError`` instead of warning.
    """
    # Deliberately does not point at `return_raw=True`: a failed run returns
    # `None` before that branch is reached, so it is not an inspection channel.
    message = (
        f"find_{kind} failed to converge: "
        + describe_failures(failures)
        + f". No estimate was stored, and `find_{kind}` returned None. "
        "Try `n_starts>1`, a different `method=`, or a different `start=`; "
        "`strict=True` raises instead of warning."
    )
    if strict:
        raise RuntimeError(message)
    warnings.warn(message, UserWarning, stacklevel=3)


def validate_n_starts(n_starts: int) -> None:
    """Reject a ``n_starts`` the multi-start loop cannot honour.

    ``jittered_starts`` always yields the unjittered start first, so anything
    below 1 would silently run exactly one start while the estimate reported
    ``n_starts=0``. Refusing it keeps the recorded metadata truthful.
    """
    if n_starts < 1:
        raise ValueError(f"n_starts must be at least 1, got {n_starts}.")


def jittered_starts(
    initvals: dict[str, Any],
    n_starts: int,
    rng: np.random.Generator,
    jitter: float = 0.1,
):
    """Yield ``n_starts`` starting points: the given one, then jittered copies.

    The jitter is *relative* (``value * (1 + U(-jitter, jitter))``), with an
    additive fallback for entries that are exactly zero, where a relative
    perturbation would be no perturbation at all.

    Both callers pass a point in *transformed* (unconstrained) space, and that
    is what keeps the jitter inside the model's support. A relative jitter is
    safe at a lower bound of zero but not at an upper one --- on the constrained
    scale it would take ``z = 0.95`` to ``1.045``, outside the interval
    ``z`` lives in, whose forward transform is then ``NaN`` and whose start PyMC
    rejects. In transformed space there are no bounds left to cross: the
    transforms map the whole real line onto the supported range, so every
    jittered candidate comes back in support.

    Unlike ``HSSMBase._jitter_initvals`` this never mutates the model's stored
    initial values and draws from an explicit ``Generator``, so multi-start runs
    are reproducible and leave the model untouched.
    """
    yield dict(initvals)
    for _ in range(max(n_starts - 1, 0)):
        candidate = {}
        for name, value in initvals.items():
            array = np.asarray(value)
            # The perturbation is multiplicative, so it has to land in floating
            # point. Casting back to an integer dtype would truncate
            # ``2 * (1 + 0.02)`` to ``2`` --- no jitter at all, while the
            # estimate still reports ``n_starts`` starts --- or down to ``1``, a
            # 50% jump rather than the requested 10%.
            dtype = (
                array.dtype if np.issubdtype(array.dtype, np.floating) else np.float64
            )
            noise = rng.uniform(-jitter, jitter, array.shape)
            perturbed = np.where(array == 0, noise, array * (1.0 + noise))
            candidate[name] = perturbed.astype(dtype).reshape(array.shape)
        yield candidate


def value_var_names(pymc_model: "PyMCModel") -> list[str]:
    """Return the names of the model's value variables."""
    return [var.name for var in pymc_model.value_vars]


def free_rv_names(pymc_model: "PyMCModel") -> list[str]:
    """Return the names of the model's free random variables (constrained)."""
    return [rv.name for rv in pymc_model.free_RVs]


def constrained_params(
    pymc_model: "PyMCModel", point: dict[str, Any]
) -> dict[str, Any]:
    """Extract the constrained free parameters from a raw optimizer point.

    The raw point mixes constrained values, transformed value-variable names and
    trial-wise deterministics; only the first group is a parameter estimate.
    """
    return {name: point[name] for name in free_rv_names(pymc_model) if name in point}


def drop_transformed(pymc_model: "PyMCModel", point: dict[str, Any]) -> dict[str, Any]:
    """Drop the transformed value-variable entries from a raw optimizer point.

    Mirrors ``pm.find_MAP(include_transformed=False)``, which HSSM cannot pass
    through directly because scoring and the failure audit need the transformed
    names.
    """
    # A value variable whose name differs from its RV's is the transformed one
    # (``t`` -> ``t_log__``); the rest are already on the constrained scale.
    transformed = set(value_var_names(pymc_model)) - set(free_rv_names(pymc_model))
    return {name: value for name, value in point.items() if name not in transformed}


def dims_and_coords(
    pymc_model: "PyMCModel", names: list[str]
) -> tuple[dict[str, list[str]], dict[str, list[Any]]]:
    """Collect the ArviZ ``dims``/``coords`` metadata for ``names``."""
    dims = {
        name: list(pymc_model.named_vars_to_dims[name])
        for name in names
        if name in pymc_model.named_vars_to_dims
        and pymc_model.named_vars_to_dims[name] is not None
    }
    coords = {
        key: list(value)
        for key, value in pymc_model.coords.items()
        if value is not None
    }
    return dims, coords


def make_start_point(
    pymc_model: "PyMCModel",
    overrides: dict[str, Any] | None,
    seed: int | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Build a transformed-space start point from constrained overrides.

    ``HSSM.initvals`` is keyed by constrained RV names; the optimizer works on
    transformed value variables. PyMC's override machinery applies the forward
    transform for us, so no manual conversion is needed --- but only
    ``pm.find_MAP`` calls it internally, which is why ``find_MLE`` has to do it
    here.
    """
    point_fn = make_initial_point_fn(
        model=pymc_model,
        jitter_rvs=set(),
        return_transformed=True,
        overrides=cast("Any", overrides),
    )
    start = point_fn(seed)
    if validate:
        pymc_model.check_start_vals(start)
    return start


def flat_gradient(objective: Any, wrt: list) -> Any:
    """Return the gradient of ``objective`` w.r.t. ``wrt`` as one flat vector.

    Concatenating in ``wrt`` order matches the layout
    ``DictToArrayBijection.map`` produces for the same variables, so the vector
    lines up with the optimizer's parameter vector entry for entry.

    ``disconnected_inputs="ignore"`` is deliberate. PyTensor's default is to
    raise when a variable does not reach the objective, but that is a legitimate
    state here rather than a user error: the observed log-likelihood of a
    centrally-parameterized hierarchical model genuinely does not contain the
    group-level ``sigma``. Zero --- what ``return_disconnected="zero"`` supplies
    --- *is* the gradient in that case, and the flat-ridge problem it signals is
    reported up front by ``find_MLE``'s hierarchical check.

    ``rewrite_pregrad`` is applied first, as ``pymc.Model.dlogp`` does before its
    own ``grad`` call. Those are PyTensor's ``canonicalize`` and ``stabilize``
    passes: on the exp/log chains that fill a log-density graph they are what
    keeps the differentiated form from overflowing where the objective itself is
    finite. Skipping them would leave every gradient built here --- the MLE
    optimizer's, the availability probe's and the standard-error Hessian's ---
    on shakier numerics than the gradient ``pymc.find_MAP`` uses for the very
    same model.
    """
    gradients = cast(
        "list[Any]",
        pytensor.grad(rewrite_pregrad(objective), wrt, disconnected_inputs="ignore"),
    )
    return pt.concatenate([pt.atleast_1d(gradient).ravel() for gradient in gradients])


def compile_point_fn(pymc_model: "PyMCModel", outputs: Any, inputs: list) -> Any:
    """Compile ``outputs`` as a function of ``inputs``, tolerating unused ones.

    Every caller here compiles against the model's *full* set of value
    variables, so that one raveled parameter vector can be mapped back onto the
    inputs without bookkeeping. PyTensor's default is to reject an input the
    graph never reads, which turns a legitimate objective --- the observed
    log-likelihood, which excludes prior-only parameters --- into an opaque
    ``UnusedInputError``. Ignoring unused inputs is what makes the mapping
    uniform.
    """
    return pymc_model.compile_fn(outputs, inputs=inputs, on_unused_input="ignore")


def make_point_expander(
    pymc_model: "PyMCModel",
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Compile the map from a transformed-space point to a full point.

    The result carries the constrained values, the transformed value-variable
    names and the deterministics --- the same shape
    ``pm.find_MAP(include_transformed=True)`` returns.

    Compiled once and reused. The graph does not depend on the point, so a
    multi-start run that rebuilt it inside the loop would pay the compile on
    every candidate to expand points that all but one of them discards.

    Returns
    -------
    Callable
        A function taking a point over the model's value variables and
        returning the expanded point.
    """
    unobserved = get_default_varnames(pymc_model.unobserved_value_vars, True)
    function = compile_point_fn(pymc_model, unobserved, pymc_model.value_vars)
    names = [var.name for var in unobserved]

    def expand(point: dict[str, Any]) -> dict[str, Any]:
        return dict(zip(names, function(point)))

    return expand


def compile_objective(
    pymc_model: "PyMCModel",
    objective: Any,
) -> tuple[Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any] | None]:
    """Compile ``objective`` and, when available, its gradient.

    Both are returned as functions of a *point dictionary* over the model's
    value variables. The gradient is returned as a single flat vector whose
    entry order matches ``DictToArrayBijection.map`` over the continuous value
    variables.

    Deliberately independent of any start point: the compiled graphs are the
    same for every candidate in a multi-start run, so ``find_MLE`` compiles once
    and reuses the result. Binding a start (which ``optimize_objective`` does
    with ``DictToArrayBijection.mapf``) is the cheap step and stays per-run ---
    it also supplies the values of any value variable that is *not* raveled into
    the parameter vector, which must come from that run's own start.
    """
    inputs = pymc_model.value_vars
    logp_fn = compile_point_fn(pymc_model, objective, inputs)

    try:
        gradient = flat_gradient(objective, pymc_model.continuous_value_vars)
        dlogp_fn = compile_point_fn(pymc_model, gradient, inputs)
    except NO_GRADIENT_ERRORS:
        dlogp_fn = None

    return logp_fn, dlogp_fn


def optimize_objective(
    pymc_model: "PyMCModel",
    objective: Any,
    start: dict[str, Any],
    method: str = DEFAULT_METHOD,
    maxeval: int = 5000,
    progressbar: bool = True,
    label: str = "MLE",
    compiled: tuple[Callable[[dict[str, Any]], Any], Any] | None = None,
    expand: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    **minimize_kwargs,
) -> tuple[dict[str, Any], Any, str]:
    """Maximize ``objective`` over the model's continuous value variables.

    A small stand-in for ``pymc.find_MAP`` for objectives PyMC cannot build
    itself --- specifically the observed log-likelihood that ``find_MLE``
    maximizes.

    Parameters
    ----------
    pymc_model
        The model whose value variables are optimized.
    objective
        A scalar PyTensor variable to *maximize*.
    start
        A transformed-space start point covering all value variables.
    method
        A ``scipy.optimize.minimize`` method. Never ``None``: SciPy would then
        auto-select a gradient-based method while ``jac`` is still set. Swapped
        for ``GRADIENT_FREE_METHOD`` if the objective has no gradient ---
        check the third return value rather than assuming.
    maxeval
        Maximum number of objective evaluations. Exhausting it yields
        ``opt_result is None``, mirroring PyMC's behaviour (including its
        off-by-one: the budget is checked *before* the counter is bumped, so
        ``maxeval + 1`` evaluations are allowed).
    progressbar
        Whether to display a progress bar, as ``pymc.find_MAP`` does.
    label
        Title for that progress bar.
    compiled
        The ``(logp_fn, dlogp_fn)`` pair from ``compile_objective``, to reuse
        across the starts of a multi-start run. Compiled here when omitted.
    expand
        The point expander from ``make_point_expander``, likewise reused across
        starts. Compiled here when omitted.
    minimize_kwargs
        Forwarded to ``scipy.optimize.minimize``.

    Returns
    -------
    tuple[dict, OptimizeResult | None, str]
        The point --- constrained values, transformed names and deterministics,
        exactly like ``pm.find_MAP(include_transformed=True)``; the raw
        optimizer result (``None`` if interrupted or out of evaluations); and
        the method that actually ran.

    Notes
    -----
    When the run is interrupted or runs out of evaluations, the point returned
    is the last iterate at which the objective (and gradient, when used) was
    finite --- not the start. Falling back to the start instead would make every
    exhausted run look like an optimizer that never moved, which is a different
    failure with a different remedy.
    """
    continuous = pymc_model.continuous_value_vars
    if not continuous:
        raise ValueError("Model has no unobserved continuous variables.")

    names = [var.name for var in continuous]
    x0 = DictToArrayBijection.map({name: start[name] for name in names})
    raw_logp_fn, raw_dlogp_fn = (
        compile_objective(pymc_model, objective) if compiled is None else compiled
    )
    # Bound to *this* run's start, so the value variables that stay outside the
    # raveled parameter vector take their values from it.
    logp_fn = DictToArrayBijection.mapf(raw_logp_fn, start)
    dlogp_fn = (
        None if raw_dlogp_fn is None else DictToArrayBijection.mapf(raw_dlogp_fn, start)
    )

    use_gradient = dlogp_fn is not None and method != GRADIENT_FREE_METHOD
    if dlogp_fn is None and method != GRADIENT_FREE_METHOD:
        _logger.warning(
            "Gradient not available for this likelihood; falling back to the "
            "gradient-free '%s' method.",
            GRADIENT_FREE_METHOD,
        )
        method = GRADIENT_FREE_METHOD

    evaluations = 0
    previous_x = x0.data
    progress = CustomProgress(
        *Progress.get_default_columns(),
        TextColumn("{task.fields[loss]}"),
        console=Console(theme=default_progress_theme),
        disable=not progressbar,
    )
    task = progress.add_task(label, total=maxeval, loss="")

    def cost(x: np.ndarray):
        nonlocal evaluations, previous_x
        raveled = RaveledVars(x, x0.point_map_info)
        value = -np.float64(logp_fn(raveled))

        gradient = None
        if use_gradient:
            gradient = -np.asarray(dlogp_fn(raveled), dtype=np.float64)  # type: ignore[misc]
            # Only a point where the gradient is usable is worth reporting back:
            # it is the last place the optimizer could actually have moved from.
            # Copied, not referenced: SciPy hands over an array it owns, and
            # whether it reallocates or writes through it on the next iteration
            # is an implementation detail that varies by method and version.
            if np.all(np.isfinite(gradient)):
                previous_x = np.array(x, copy=True)
        elif np.isfinite(value):
            previous_x = np.array(x, copy=True)

        if evaluations > maxeval:
            raise StopIteration(
                f"Maximum number of objective evaluations ({maxeval}) reached."
            )
        evaluations += 1
        # Refreshing the rendered loss on every evaluation costs more than the
        # objective does on cheap likelihoods, so throttle it the way PyMC's
        # own optimizer progress bar throttles.
        if evaluations % 10 == 0:
            progress.update(task, completed=evaluations, loss=f"logp = {-value:,.5g}")
        else:
            progress.update(task, completed=evaluations)

        return (value, gradient) if use_gradient else value

    with progress:
        try:
            opt_result = sp_optimize.minimize(
                cost, x0.data, method=method, jac=use_gradient, **minimize_kwargs
            )
            best_x = opt_result["x"]
        except (KeyboardInterrupt, StopIteration) as error:
            best_x, opt_result = previous_x, None
            _logger.info(str(error))
        finally:
            progress.update(task, completed=evaluations, refresh=True)

    raveled = RaveledVars(np.asarray(best_x), x0.point_map_info)
    expand = make_point_expander(pymc_model) if expand is None else expand
    point = expand(DictToArrayBijection.rmap(raveled, start))

    return point, opt_result, method


def make_scorer(
    pymc_model: "PyMCModel",
    observed_only: bool = False,
    function: Callable[[dict[str, Any]], Any] | None = None,
) -> Callable[[dict[str, Any]], float]:
    """Compile a reusable log-density scorer for ``pymc_model``.

    Compiling once and reusing the result matters for multi-start runs, which
    would otherwise pay the compile cost per candidate.

    Parameters
    ----------
    pymc_model
        The model to score against.
    observed_only
        If ``True``, score the observed log-likelihood only --- the objective
        ``find_MLE`` maximizes. MAP and MLE estimates are only comparable when
        both are scored on the *same* objective.
    function
        An already-compiled evaluator of the same objective over the model's
        value variables, as ``compile_objective`` returns. ``find_MLE`` scores
        exactly the objective it optimizes, so passing that one in saves
        compiling the observed log-likelihood a second time per call. Compiled
        here when omitted.

    Returns
    -------
    Callable
        A function taking a point (every value variable, transformed names
        included) and returning its log-density, or ``-inf`` if it could not be
        evaluated. It raises ``KeyError`` if the point is missing any value
        variable --- unscoreable input, rather than a point of zero density.
    """
    if function is None:
        objective = (
            pymc_model.observedlogp
            if observed_only
            else pymc_model.logp(jacobian=False)
        )
        function = compile_point_fn(pymc_model, objective, pymc_model.value_vars)
    scoring_fn = function
    names = value_var_names(pymc_model)

    def score(point: dict[str, Any]) -> float:
        missing = [name for name in names if name not in point]
        if missing:
            # Not a `-inf`: that is the answer for a point of zero density, and
            # reusing it for a malformed point would report "infinitely bad"
            # where the truth is "unscoreable". The likeliest way to get here
            # is scoring a `find_MAP(include_transformed=False)` estimate,
            # which drops exactly the transformed entries needed below, so the
            # message names that case rather than only the missing keys.
            raise KeyError(
                f"Cannot score this point: {', '.join(missing)} missing from "
                "it. Scoring requires every value variable, transformed names "
                "included; an estimate produced with "
                "`include_transformed=False` does not carry them. Re-run "
                "without that flag, or score the raw optimizer point."
            )
        try:
            return float(scoring_fn({name: point[name] for name in names}))
        except (ValueError, FloatingPointError):  # pragma: no cover - defensive
            return -np.inf

    return score


def score_point(
    pymc_model: "PyMCModel",
    point: dict[str, Any],
    observed_only: bool = False,
) -> float:
    """Evaluate the model's log-density at ``point``.

    A one-shot convenience wrapper around ``make_scorer``; prefer the latter
    when scoring more than one point against the same model.

    Parameters
    ----------
    pymc_model
        The model to score against.
    point
        A point containing every value variable (transformed names included).
    observed_only
        If ``True``, score the observed log-likelihood only --- the objective
        ``find_MLE`` maximizes. MAP and MLE estimates are only comparable when
        both are scored on the *same* objective.

    Returns
    -------
    float
        The log-density, or ``-inf`` if it could not be evaluated.

    Raises
    ------
    KeyError
        If ``point`` is missing any of the model's value variables, as an
        estimate produced with ``include_transformed=False`` is.
    """
    return make_scorer(pymc_model, observed_only=observed_only)(point)


def standard_errors(
    pymc_model: "PyMCModel",
    point: dict[str, Any],
    observed_only: bool = False,
) -> dict[str, Any] | None:
    """Compute standard errors from the inverse Hessian at ``point``.

    The Hessian is taken on the *untransformed* model, so the result is on the
    scale users report. This matters twice over: ``pm.find_hessian`` works in
    transformed value-variable space with ``jacobian=True``, which is both the
    wrong scale *and* the curvature of a different function from the one
    ``find_MAP`` maximizes (``jacobian=False``).

    The Hessian itself is obtained by central differences of the analytic
    gradient. Symbolic second derivatives do not exist for ONNX/JAX-wrapped
    (``approx_differentiable``) likelihoods, and this way the analytical and the
    LAN path share one implementation.

    Parameters
    ----------
    pymc_model
        The model to differentiate.
    point
        The estimate, containing every constrained parameter.
    observed_only
        Take the curvature of the observed log-likelihood rather than the joint
        log-density --- the observed information, which is what an MLE wants.

    Returns
    -------
    dict | None
        Errors keyed by constrained parameter name, or ``None`` if no gradient
        is available (blackbox likelihoods). Individual entries are ``NaN`` when
        the Hessian is singular or not negative definite --- which is expected
        at an optimum sitting on a bound.

        For ``observed_only=True`` these are standard errors in the usual sense.
        For the joint log-density they are posterior standard deviations under a
        Laplace approximation around the mode; the arithmetic is identical but
        the object is not.

    Warnings
    --------
    The cost grows with the number of scalar parameters ``n``: the central
    differences need ``2 * n`` full-data gradient evaluations, followed by an
    ``n x n`` inverse. On a hierarchical model with hundreds of group-level
    offsets, ``se=True`` can take considerably longer than the optimization it
    follows. A line is logged at ``INFO`` before starting when ``n`` is large.
    """
    untransformed = remove_value_transforms(pymc_model)
    objective = (
        untransformed.observedlogp
        if observed_only
        else untransformed.logp(jacobian=False)
    )
    continuous = untransformed.continuous_value_vars
    names = [var.name for var in continuous]

    missing = [name for name in names if name not in point]
    if missing:  # pragma: no cover - defensive
        _logger.warning(
            "Cannot compute standard errors: %s missing from the estimate.",
            ", ".join(missing),
        )
        return None

    base = {
        name: np.asarray(point[name], dtype=untransformed.named_vars[name].dtype)
        for name in names
    }
    x0 = DictToArrayBijection.map(base)

    try:
        gradient = flat_gradient(objective, continuous)
        gradient_fn = DictToArrayBijection.mapf(
            compile_point_fn(untransformed, gradient, untransformed.value_vars),
            base,
        )
    except NO_GRADIENT_ERRORS:
        warnings.warn(
            "Standard errors are not available for this likelihood: it exposes "
            "no gradient, so the Hessian cannot be approximated. Returning the "
            "estimate without standard errors.",
            UserWarning,
            # standard_errors <- _make_point_estimate <- find_MAP/find_MLE <- user
            stacklevel=4,
        )
        return None

    def gradient_at(x: np.ndarray) -> np.ndarray:
        return np.asarray(gradient_fn(RaveledVars(x, x0.point_map_info)), dtype=float)

    size = x0.data.size
    if size > _SE_SIZE_HINT:
        _logger.info(
            "Computing standard errors for %d parameters: %d full gradient "
            "evaluations plus a %dx%d matrix inverse. This may take a while.",
            size,
            2 * size,
            size,
            size,
        )
    # Scaled to the precision the *gradient* is computed at, not to float64's.
    # The central-difference step that balances truncation against rounding goes
    # as the cube root of the relative noise in the differentiated quantity, and
    # under `set_floatX("float32")` that noise is ~1e-7 rather than ~2e-16 --- so
    # a float64-sized step is roughly three orders of magnitude too small there
    # and divides the gradient's rounding noise by far too little. On a smooth
    # analytical likelihood the resulting Hessian is still usable (the errors
    # move by a fraction of a percent); the margin matters on the noisier
    # ONNX/JAX gradients, which is also where float32 is actually used.
    step = np.cbrt(np.finfo(pytensor.config.floatX).eps) * np.maximum(
        1.0, np.abs(x0.data)
    )
    hessian = np.empty((size, size))
    for index in range(size):
        forward = x0.data.astype(float).copy()
        backward = x0.data.astype(float).copy()
        forward[index] += step[index]
        backward[index] -= step[index]
        hessian[:, index] = (gradient_at(forward) - gradient_at(backward)) / (
            2 * step[index]
        )
    hessian = 0.5 * (hessian + hessian.T)

    errors = np.full(size, np.nan)
    if np.all(np.isfinite(hessian)):
        try:
            # `inv` alone is not a test of the documented precondition: it raises
            # only on *exact* singularity, so a merely ill-conditioned Hessian
            # comes back as large finite numbers that would pass the
            # `variances > 0` check below and be reported as standard errors.
            # Cholesky is the honest gate --- it fails on anything that is not
            # positive definite --- and the condition number catches the
            # near-singular remainder, where the inverse is numerical noise.
            np.linalg.cholesky(-hessian)
            if np.linalg.cond(-hessian) >= 1.0 / np.finfo(float).eps:
                raise np.linalg.LinAlgError("Hessian is numerically singular.")
            covariance = np.linalg.inv(-hessian)
            variances = np.diag(covariance)
            with np.errstate(invalid="ignore"):
                errors = np.where(variances > 0, np.sqrt(variances), np.nan)
        except np.linalg.LinAlgError:
            pass

    if not np.all(np.isfinite(errors)):
        warnings.warn(
            "The Hessian at the estimate is singular or not negative definite, "
            "so some standard errors are NaN. This is expected when the optimum "
            "sits on a parameter bound or a parameter is weakly identified.",
            UserWarning,
            stacklevel=4,
        )

    return DictToArrayBijection.rmap(RaveledVars(errors, x0.point_map_info))


def group_specific_terms(hssm_model) -> list[str]:
    """Return the names of any group-specific (hierarchical) terms in the model.

    Detected structurally rather than by name-matching ``*_sigma``. Note that
    ``bambi.Model.components`` is a ``dict``, so it must be iterated over
    ``.values()`` --- bare iteration yields the component *names* and the check
    would silently never fire.
    """
    terms: list[str] = []
    for component in hssm_model.model.components.values():
        component_terms = getattr(component, "group_specific_terms", None)
        if component_terms:
            terms.extend(list(component_terms))
    return terms


# endregion

"""Validation hooks for centered vs. non-centered group-specific priors.

Two checks are exposed:

* ``check_user_priors_against_parameterization`` looks at the user's prior dict
  for each :class:`RegressionParam` and flags the specific footgun where a
  ``Normal`` group-specific prior carries a nested hyperprior on ``mu`` while
  the effective parameterization is non-centered. Under non-centered, bambi
  reparameterizes the term as ``offset * sigma`` (see
  ``bambi/backend/terms.py``), so the ``mu`` hyperprior is created in the
  PyMC graph but never wired into the likelihood -- it becomes a disconnected
  free RV.

* ``find_disconnected_free_rvs`` walks the PyMC graph after ``model.build()``
  and reports any free RV that is not an ancestor of an observed RV. This is
  the generic safety net that also catches problems we have not anticipated.

Both checks only produce reports; emission of warnings is left to the caller
so that messages can be aggregated and addressed consistently with the rest
of the HSSM logger output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import bambi as bmb
import numpy as np

if TYPE_CHECKING:
    import pymc as pm

    from .params import Params

_logger = logging.getLogger("hssm")


@dataclass
class PriorMismatch:
    """A single user-prior / parameterization mismatch detected before build."""

    parameter: str
    term: str
    reason: str
    suggestion: str


def _resolve_noncentered(
    noncentered: bool | dict[str, bool] | None,
    component_name: str,
    prior_noncentered: bool | None,
) -> bool:
    """Compute the effective ``noncentered`` flag for a single term.

    Mirrors bambi's resolution order in ``Model._set_priors``:

    1. A per-:class:`bmb.Prior` ``noncentered`` override takes precedence.
    2. Otherwise the model-level value is used, which may be a ``dict`` keyed
       by distributional component name.
    3. Missing dict keys fall back to ``True`` (bambi's default).
    """
    if prior_noncentered is not None:
        return prior_noncentered
    if isinstance(noncentered, dict):
        return noncentered.get(component_name, True)
    if noncentered is None:
        return True
    return noncentered


def _has_hyperprior_mu(prior: Any) -> bool:
    """Check whether ``prior`` is a Normal with a hyperprior on ``mu``.

    Returns True if ``prior`` is a Normal :class:`bmb.Prior` whose ``mu`` is
    itself a :class:`bmb.Prior`. A scalar non-zero ``mu`` does not produce a
    disconnected node in the graph (only its intent is dropped), so we
    restrict the targeted check to the hyperprior case, which is the source
    of the orphan RV.
    """
    if not isinstance(prior, bmb.Prior):
        return False
    if prior.name != "Normal":
        return False
    return isinstance(prior.args.get("mu"), bmb.Prior)


def _has_nontrivial_mu(prior: Any) -> bool:
    """Check whether a Normal prior has a `mu` that is not a scalar zero.

    Returns True if ``prior`` is a Normal :class:`bmb.Prior` whose ``mu``
    argument is either a :class:`bmb.Prior` (a hyperprior) or a non-zero
    scalar. These are the cases in which `mu` contributes an extra location
    parameter to the linear predictor; a scalar zero is benign.
    """
    if not isinstance(prior, bmb.Prior) or prior.name != "Normal":
        return False
    mu = prior.args.get("mu")
    if isinstance(mu, bmb.Prior):
        return True
    if mu is None:
        return False
    try:
        return bool(np.asarray(mu).reshape(-1)[0] != 0.0)
    except (ValueError, TypeError, IndexError):
        return False


def check_user_priors_against_parameterization(
    params: Params,
    noncentered: bool | dict[str, bool] | None,
) -> list[PriorMismatch]:
    """Detect user priors that will be silently dropped by non-centered bambi.

    Iterates over each :class:`RegressionParam` and inspects user-supplied
    group-specific Normal priors with nested ``mu`` hyperpriors. When the
    effective ``noncentered`` is ``True`` for that component, the ``mu``
    hyperprior is created as an orphan RV in the graph and ignored in the
    deterministic ``offset * sigma`` reparameterization.

    Parameters
    ----------
    params
        The HSSM ``Params`` container, after ``process_prior`` /
        ``make_safe_priors`` have run, so values are :class:`bmb.Prior`.
    noncentered
        The model-level ``noncentered`` setting, as it will be passed through
        to ``bmb.Model``. May be ``bool``, ``dict``, or ``None``.

    Returns
    -------
    list[PriorMismatch]
        One entry per (parameter, group-specific term) flagged. Empty if
        nothing was flagged.
    """
    mismatches: list[PriorMismatch] = []
    for param_name, param in params.items():
        prior_dict = getattr(param, "prior", None)
        if not isinstance(prior_dict, dict):
            continue
        user_keys: set[str] = getattr(param, "_user_specified_prior_keys", set())
        for term_name, prior in prior_dict.items():
            if "|" not in term_name:
                continue
            if term_name not in user_keys:
                continue
            if not _has_hyperprior_mu(prior):
                continue
            effective_nc = _resolve_noncentered(
                noncentered,
                component_name=param_name,
                prior_noncentered=getattr(prior, "noncentered", None),
            )
            if not effective_nc:
                continue
            mismatches.append(
                PriorMismatch(
                    parameter=param_name,
                    term=term_name,
                    reason=(
                        f"User prior for '{term_name}' on parameter "
                        f"'{param_name}' supplies a hyperprior on `mu`, but "
                        "the effective parameterization is non-centered. "
                        "bambi will reparameterize this term as "
                        "`offset * sigma` and drop the `mu` hyperprior, "
                        "leaving it as a disconnected node in the PyMC "
                        "graph."
                    ),
                    suggestion=(
                        "Either pass `noncentered=False` to `HSSM(...)` so "
                        "that `mu` is used in the centered Normal, or move "
                        "the location prior to the common `Intercept` (e.g. "
                        f"use a formula like '{param_name} ~ 1 + "
                        f"({term_name.split('|')[0]}|"
                        f"{term_name.split('|')[1]})' and attach the `mu` "
                        "prior to 'Intercept'). To silence this warning "
                        "without changing the model, set the `mu` argument "
                        "to a scalar (e.g. `mu=0`)."
                    ),
                )
            )
    return mismatches


def check_user_priors_for_location_overparameterization(
    params: Params,
) -> list[PriorMismatch]:
    """Detect group-specific terms whose location collides with a common Intercept.

    When a regression formula contains a common `Intercept` and the user
    supplies a group-specific Normal prior whose `mu` is non-trivial (a
    hyperprior or a non-zero scalar), the linear predictor sees only
    `Intercept + mu_u`. The likelihood is invariant under shifts of mass
    between the two parameters, so they are non-identifiable individually
    and the posterior has a ridge along the anti-diagonal of the two.

    This is a statistical concern (separate from the disconnected-node
    problem). It applies under both centered and non-centered
    parameterizations: under centered the ridge is real and degrades
    sampling; under non-centered the user's `mu` is silently ignored anyway,
    so the warning doubles as a heads-up that the spec is not doing what
    they probably think.
    """
    mismatches: list[PriorMismatch] = []
    for param_name, param in params.items():
        prior_dict = getattr(param, "prior", None)
        if not isinstance(prior_dict, dict):
            continue
        user_keys: set[str] = getattr(param, "_user_specified_prior_keys", set())
        terms: set[str] = set(getattr(param, "terms", []))
        # A common `Intercept` is present iff the design matrix contained it
        # (in which case `make_safe_priors` appended it to `terms`) or the
        # user explicitly supplied a prior key for it.
        if "Intercept" not in (terms | user_keys):
            continue
        for term_name, prior in prior_dict.items():
            if "|" not in term_name:
                continue
            if term_name not in user_keys:
                continue
            if not _has_nontrivial_mu(prior):
                continue
            mismatches.append(
                PriorMismatch(
                    parameter=param_name,
                    term=term_name,
                    reason=(
                        f"User prior for '{term_name}' on parameter "
                        f"'{param_name}' has a non-trivial `mu`, and the "
                        "formula also includes a common `Intercept` for "
                        f"'{param_name}'. The data only constrains the sum "
                        "`Intercept + mu`; the two are non-identifiable "
                        "individually and the posterior will have a ridge "
                        "along the anti-diagonal."
                    ),
                    suggestion=(
                        "Set `mu=0` on the group term so the common "
                        "`Intercept` owns the location, or drop the common "
                        "intercept from the formula (e.g. "
                        f"`{param_name} ~ 0 + ({term_name})`)."
                    ),
                )
            )
    return mismatches


def find_disconnected_free_rvs(pymc_model: pm.Model) -> list[str]:
    """Return names of free RVs that do not feed any observed RV.

    Uses :func:`pytensor.graph.basic.ancestors` to traverse the graph
    upward from each observed RV; any free RV outside this ancestor set is
    unreachable from the likelihood and will sit in the graph contributing
    only to its own prior.
    """
    try:
        from pytensor.graph.traversal import ancestors
    except ImportError:  # pragma: no cover - older pytensor
        from pytensor.graph.basic import ancestors  # type: ignore[no-redef]

    observed = list(getattr(pymc_model, "observed_RVs", []))
    free = list(getattr(pymc_model, "free_RVs", []))
    if not observed or not free:
        return []
    connected: set[int] = set()
    for obs in observed:
        for var in ancestors([obs]):
            connected.add(id(var))
    return [rv.name for rv in free if id(rv) not in connected]


def emit_parameterization_warnings(mismatches: list[PriorMismatch]) -> None:
    """Log one warning per :class:`PriorMismatch` via the ``hssm`` logger."""
    for m in mismatches:
        _logger.warning("%s %s", m.reason, m.suggestion)


def emit_disconnected_node_warnings(disconnected: list[str]) -> None:
    """Log a single warning enumerating any disconnected free RV names."""
    if not disconnected:
        return
    _logger.warning(
        "The PyMC graph contains free random variables that do not "
        "influence the likelihood: %s. This typically happens when a "
        "hyperprior is supplied for a parameter that the chosen "
        "parameterization does not use (e.g. `mu` under "
        "`noncentered=True`). These nodes will be sampled but will not "
        "affect inference; consider switching the parameterization or "
        "adjusting the prior specification.",
        ", ".join(repr(name) for name in disconnected),
    )

"""Validation hooks for centered vs. non-centered group-specific priors.

Two checks are exposed:

* ``check_user_priors_against_parameterization`` looks at the user's prior dict
  for each :class:`RegressionParam` and flags group-specific ``Normal`` priors
  whose ``mu`` bambi cannot honor under the effective non-centered
  parameterization.

* ``check_user_priors_for_location_overparameterization`` flags a free group
  mean only when its Formulae expression has an exact common-effect counterpart
  and the effective parameterization is centered.

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
    from collections.abc import Iterator

    import pymc as pm

    from .params import Params

_logger = logging.getLogger("hssm")

# Distributions whose ``mu`` parameter translates the entire distribution.
# Only these families have the exact fixed-effect / group-mean shift invariance
# described by ``check_user_priors_for_location_overparameterization``. Other
# distributions may call a mean or shape parameter ``mu`` without being a
# location family (for example, Gamma).
_ADDITIVE_LOCATION_PRIORS = frozenset(
    {
        "AsymmetricLaplace",
        "ExGaussian",
        "Gumbel",
        "Laplace",
        "Logistic",
        "Moyal",
        "Normal",
        "SkewNormal",
        "SkewStudentT",
        "StudentT",
    }
)


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


def _is_zero(value: Any) -> bool:
    """Whether a fixed scalar or array is entirely zero."""
    try:
        return bool(np.all(np.asarray(value) == 0.0))
    except (TypeError, ValueError):
        return False


def _iter_user_group_priors(
    param: Any,
) -> Iterator[tuple[str, str, bool, bmb.Prior]]:
    """Yield structurally identified, user-supplied group priors.

    Formulae owns term parsing. ``RegressionParam`` caches the full group key
    to expression-name mapping while preparing its design matrices, so warning
    code never has to infer structure from display names such as ``x|id``.
    """
    prior_dict = getattr(param, "prior", None)
    if not isinstance(prior_dict, dict):
        return

    user_keys: set[str] = getattr(param, "_user_specified_prior_keys", set())
    group_term_names: dict[str, str] = getattr(param, "_group_term_names", {})
    groups_with_common: set[str] = getattr(param, "_group_terms_with_common", set())
    for term_name, expression_name in group_term_names.items():
        if term_name in user_keys:
            prior = prior_dict.get(term_name)
        elif "group_specific" in user_keys:
            prior = prior_dict.get("group_specific")
        else:
            continue
        if not isinstance(prior, bmb.Prior):
            continue
        yield term_name, expression_name, term_name in groups_with_common, prior


def _parameterization_suggestion(
    param_name: str,
    expression_name: str,
    has_common: bool,
    free_mu: bool,
) -> str:
    """Build a term-aware correction for a non-centered mismatch."""
    if has_common:
        suggestion = (
            f"Keep the common '{expression_name}' effect and set `mu=0` on "
            "the matching group term"
        )
        if free_mu:
            return (
                f"{suggestion}. If a free group mean is intended instead, "
                f"remove the common '{expression_name}' effect and pass "
                f"`noncentered=False` for '{param_name}'."
            )
        return (
            f"{suggestion}, or pass `noncentered=False` if the fixed group "
            "location is intentional."
        )
    return (
        f"Either add the common '{expression_name}' effect and set `mu=0` "
        "on the group term, or pass `noncentered=False` so the group "
        "location is used."
    )


def check_user_priors_against_parameterization(
    params: Params,
    noncentered: bool | dict[str, bool] | None,
) -> list[PriorMismatch]:
    """Detect user priors that conflict with non-centered bambi.

    Iterates over each :class:`RegressionParam` and inspects structurally
    identified, user-supplied group-specific Normal priors. When the effective
    ``noncentered`` is ``True`` for that component, the outcome depends on
    ``mu`` and ``sigma``:

    * With hierarchical ``sigma`` and free ``mu``, bambi creates ``mu`` as an
      orphan and reparameterizes the term as ``offset * sigma``.
    * With hierarchical ``sigma`` and fixed non-zero ``mu``, bambi silently
      ignores the fixed location (including vector-valued locations).
    * With fixed ``sigma`` and free ``mu``, bambi raises
      :class:`NotImplementedError` during model construction.

    Each is flagged with a message tailored to the actual outcome.

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
        for term_name, expression_name, has_common, prior in _iter_user_group_priors(
            param
        ):
            if prior.name != "Normal":
                continue
            effective_nc = _resolve_noncentered(
                noncentered,
                component_name=param_name,
                prior_noncentered=getattr(prior, "noncentered", None),
            )
            if not effective_nc:
                continue

            mu = prior.args.get("mu")
            sigma = prior.args.get("sigma")
            free_mu = isinstance(mu, bmb.Prior)
            hierarchical_sigma = isinstance(sigma, bmb.Prior)

            if free_mu and hierarchical_sigma:
                reason = (
                    f"User prior for '{term_name}' on parameter "
                    f"'{param_name}' supplies a hyperprior on `mu`, but the "
                    "effective parameterization is non-centered. bambi will "
                    "reparameterize this term as `offset * sigma` and drop "
                    "the `mu` hyperprior, leaving it as a disconnected node "
                    "in the PyMC graph."
                )
            elif free_mu:
                reason = (
                    f"User prior for '{term_name}' on parameter "
                    f"'{param_name}' supplies a hyperprior on `mu` with a "
                    "fixed `sigma`, but the effective "
                    "parameterization is non-centered. bambi's non-centered "
                    "reparameterization only supports a Normal whose `sigma` "
                    "is itself a hyperprior, so this term cannot be built "
                    "under non-centered: bambi raises NotImplementedError at "
                    "model build time."
                )
            elif hierarchical_sigma and mu is not None and not _is_zero(mu):
                reason = (
                    f"User prior for '{term_name}' on parameter "
                    f"'{param_name}' supplies a fixed non-zero `mu`, but the "
                    "effective parameterization is non-centered. bambi will "
                    "reparameterize this term as `offset * sigma` and ignore "
                    "the specified `mu`, so the requested group location is "
                    "not represented in the PyMC graph."
                )
            else:
                continue

            mismatches.append(
                PriorMismatch(
                    parameter=param_name,
                    term=term_name,
                    reason=reason,
                    suggestion=_parameterization_suggestion(
                        param_name,
                        expression_name,
                        has_common,
                        free_mu,
                    ),
                )
            )
    return mismatches


def check_user_priors_for_location_overparameterization(
    params: Params,
    noncentered: bool | dict[str, bool] | None,
) -> list[PriorMismatch]:
    """Detect centered group means that collide with matching common effects.

    When a Formulae group expression also occurs as a common term and the user
    supplies a free ``mu`` for a group-specific translation-family prior under
    centering, the linear predictor sees only ``beta + mu``. The likelihood is
    invariant under shifts of mass between those parameters, so the posterior
    has a ridge.

    This check is intentionally silent for fixed ``mu`` values, unmatched
    group-only expressions, and effective non-centering. Non-centered problems
    are reported separately by
    :func:`check_user_priors_against_parameterization`.
    """
    mismatches: list[PriorMismatch] = []
    for param_name, param in params.items():
        for term_name, expression_name, has_common, prior in _iter_user_group_priors(
            param
        ):
            if not has_common or prior.name not in _ADDITIVE_LOCATION_PRIORS:
                continue
            if not isinstance(prior.args.get("mu"), bmb.Prior):
                continue
            effective_nc = _resolve_noncentered(
                noncentered,
                component_name=param_name,
                prior_noncentered=getattr(prior, "noncentered", None),
            )
            if effective_nc:
                continue
            mismatches.append(
                PriorMismatch(
                    parameter=param_name,
                    term=term_name,
                    reason=(
                        f"User prior for '{term_name}' on parameter "
                        f"'{param_name}' has a free `mu`, and its Formulae "
                        f"expression '{expression_name}' also occurs as a "
                        "common effect under the effective centered "
                        "parameterization. The data only constrains their sum; "
                        "the common and group locations are non-identifiable "
                        "individually and the posterior will have a ridge "
                        "along the anti-diagonal."
                    ),
                    suggestion=(
                        f"Keep the common '{expression_name}' effect and set "
                        "`mu=0` on the matching group term, or remove that "
                        "common effect if the group-level mean should own the "
                        "location."
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

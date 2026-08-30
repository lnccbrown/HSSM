"""Validation hooks for centered vs. non-centered group-specific priors.

Three validation layers are exposed:

* ``check_user_group_prior_compatibility`` looks at the user's prior dict for
  each :class:`RegressionParam` and rejects group-specific priors that bambi
  cannot build or cannot honor under the effective parameterization.

* ``check_user_priors_for_location_overparameterization`` flags a free group
  mean only when its Formulae expression has an exact common-effect counterpart
  and the effective parameterization is centered.

* ``find_disconnected_free_rvs`` walks the PyMC graph after ``model.build()``
  and reports any free RV that is not an ancestor of an observed RV. This is
  the generic safety net that also catches problems we have not anticipated.

The two prior checks only produce reports. The caller aggregates compatibility
errors into one exception and emits statistical-identifiability warnings via
the HSSM logger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import bambi as bmb
import numpy as np

from .parameterization import NoncenteredSetting, _resolve_noncentered

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


def _is_zero(value: Any) -> bool:
    """Whether a fixed scalar or array is entirely zero."""
    try:
        array = np.asarray(value)
        return bool(array.size > 0 and np.all(array == 0.0))
    except (TypeError, ValueError):
        return False


def _iter_user_group_prior_specs(
    param: Any,
) -> Iterator[tuple[str, str, bool, Any]]:
    """Yield structurally identified user group-prior specifications.

    Formulae owns term parsing. ``RegressionParam`` caches the full group key
    to expression-name mapping while preparing its design matrices, so
    validation never has to infer structure from display names such as
    ``x|id``. Exact keys take precedence over the ``group_specific`` wildcard,
    matching bambi's prior resolution.
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
        yield term_name, expression_name, term_name in groups_with_common, prior


def _iter_user_group_priors(
    param: Any,
) -> Iterator[tuple[str, str, bool, bmb.Prior]]:
    """Yield user-supplied group specifications that are bambi priors."""
    for term_name, expression_name, has_common, prior in _iter_user_group_prior_specs(
        param
    ):
        if isinstance(prior, bmb.Prior):
            yield term_name, expression_name, has_common, prior


def _noncentered_compatibility_suggestion(
    param_name: str,
    expression_name: str,
    has_common: bool,
) -> str:
    """Build a complete term-aware correction for an NC incompatibility."""
    formula_expression = "1" if expression_name == "Intercept" else expression_name
    faithful_prior = (
        "a plain built-in Normal with hierarchical `sigma`, absent or "
        "fixed-all-zero `mu`, and no additional arguments"
    )
    centered_remedy = (
        "set `noncentered=False` on this prior and on any nested hierarchical "
        "hyperpriors (or remove their overrides and make the effective "
        f"component setting for '{param_name}' centered)"
    )
    if has_common:
        return (
            f"Keep the common formula term '{formula_expression}' and use "
            f"{faithful_prior} for its zero-mean group deviation. To retain the "
            f"explicit prior instead, {centered_remedy}; remove the common "
            "effect as well if a free group mean should own the population "
            "location."
        )
    return (
        f"Either add the exact common formula term '{formula_expression}' and "
        f"use {faithful_prior} for the resulting zero-mean group deviation, or "
        f"{centered_remedy} to retain the group location and explicit prior "
        "family."
    )


def _prior_tree_noncentered_issues(
    prior: bmb.Prior,
    noncentered: NoncenteredSetting,
    param_name: str,
    path: str = "outer prior",
) -> list[str]:
    """Mirror bambi's recursive non-centering contract for a prior tree."""
    if getattr(prior, "is_truncated", False):
        return [
            f"{path} is an HSSM truncated Prior, whose hidden arguments and "
            "custom distribution cannot be built recursively as a bambi "
            "group-prior node"
        ]

    issues: list[str] = []
    args = prior.args
    hyperprior_args = {
        name for name, value in args.items() if isinstance(value, bmb.Prior)
    }
    for name in sorted(hyperprior_args):
        issues.extend(
            _prior_tree_noncentered_issues(
                args[name],
                noncentered,
                param_name,
                path=f"{path}.{name}",
            )
        )

    effective_nc = _resolve_noncentered(
        noncentered,
        component_name=param_name,
        prior_noncentered=getattr(prior, "noncentered", None),
    )
    if not effective_nc or not hyperprior_args:
        return issues

    if prior.name != "Normal" or prior.dist is not None:
        family = "a custom distribution" if prior.dist is not None else repr(prior.name)
        issues.append(
            f"{path} uses {family}, while bambi can non-center only a built-in "
            "untruncated Normal node with hierarchical `sigma`"
        )
        return issues

    extra_args = sorted(set(args) - {"mu", "sigma"})
    sigma = args.get("sigma")
    mu_is_present = "mu" in args
    mu = args.get("mu")
    if extra_args:
        issues.append(
            f"{path} includes argument(s) {extra_args!r}, which bambi discards "
            "when it constructs `offset * sigma`"
        )
    if not isinstance(sigma, bmb.Prior):
        issues.append(
            f"{path} has stochastic argument(s) {sorted(hyperprior_args)!r} "
            "but no hierarchical `sigma`, so bambi cannot non-center it"
        )
    if isinstance(mu, bmb.Prior):
        issues.append(
            f"{path} supplies a `mu` hyperprior that bambi creates and then "
            "omits from `offset * sigma`, leaving a disconnected node"
        )
    elif mu_is_present and not _is_zero(mu):
        issues.append(
            f"{path} supplies a `mu` that is not fixed entirely to zero and "
            "would be silently ignored in `offset * sigma`"
        )
    return issues


def check_user_group_prior_compatibility(
    params: Params,
    noncentered: NoncenteredSetting,
) -> list[PriorMismatch]:
    """Detect explicit group priors bambi cannot represent faithfully.

    A group-specific bambi prior needs at least one top-level hyperprior under
    either parameterization. Under effective non-centering, bambi's current
    shortcut faithfully represents only an untruncated built-in ``Normal``
    with hierarchical ``sigma``, absent or fixed-all-zero ``mu``, and no other
    distribution arguments. Anything else either fails during bambi model
    construction or is silently discarded by ``offset * sigma``.

    Explicit specifications are never rewritten. This checker reports every
    incompatible term so the caller can raise one aggregated pre-build error.

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
        One entry per incompatible (parameter, group-specific term). Empty if
        all explicit group priors are representable.
    """
    mismatches: list[PriorMismatch] = []
    for param_name, param in params.items():
        for (
            term_name,
            expression_name,
            has_common,
            prior,
        ) in _iter_user_group_prior_specs(param):
            # ``None`` deliberately delegates this term to bambi's defaults.
            if prior is None:
                continue

            if not isinstance(prior, bmb.Prior):
                mismatches.append(
                    PriorMismatch(
                        parameter=param_name,
                        term=term_name,
                        reason=(
                            f"User specification for group term '{term_name}' on "
                            f"parameter '{param_name}' is not a bambi Prior. bambi "
                            "requires regression group-term priors to be "
                            "`bmb.Prior` objects; numeric values do not fix a "
                            "group coefficient."
                        ),
                        suggestion=(
                            "Supply a hierarchical `bmb.Prior` for this group "
                            "term, or remove the explicit key to use a default."
                        ),
                    )
                )
                continue

            # HSSM's truncated Prior stores its original arguments in ``_args``
            # and exposes an empty ``args`` mapping to bambi. Consequently bambi
            # cannot see the required top-level hyperprior for a group term.
            if getattr(prior, "is_truncated", False):
                mismatches.append(
                    PriorMismatch(
                        parameter=param_name,
                        term=term_name,
                        reason=(
                            f"User prior for group term '{term_name}' on parameter "
                            f"'{param_name}' is truncated. HSSM's truncated prior "
                            "wrapper hides its distribution arguments from bambi, "
                            "so bambi cannot construct it as a hierarchical "
                            "group-specific prior."
                        ),
                        suggestion=(
                            "Use an untruncated hierarchical group prior and enforce "
                            "parameter support with an appropriate link function."
                        ),
                    )
                )
                continue

            args = prior.args
            hyperprior_args = {
                name for name, value in args.items() if isinstance(value, bmb.Prior)
            }
            if not hyperprior_args:
                reason = (
                    f"User prior for group term '{term_name}' on parameter "
                    f"'{param_name}' has no top-level hyperprior. bambi requires "
                    "at least one distribution argument of every group-specific "
                    "prior to be another `bmb.Prior`."
                )
                suggestion = (
                    "Make at least one group-prior argument hierarchical (usually "
                    "`sigma=bmb.Prior(...)`), or remove the explicit key to use a "
                    "default."
                )
            else:
                issues = _prior_tree_noncentered_issues(
                    prior,
                    noncentered,
                    param_name,
                )
                if not issues:
                    continue
                reason = (
                    f"User prior for group term '{term_name}' on parameter "
                    f"'{param_name}' is incompatible with the effective "
                    f"parameterization: {'; '.join(issues)}. Continuing would "
                    "either fail in bambi or change the requested prior."
                )
                suggestion = _noncentered_compatibility_suggestion(
                    param_name, expression_name, has_common
                )

            mismatches.append(
                PriorMismatch(
                    parameter=param_name,
                    term=term_name,
                    reason=reason,
                    suggestion=suggestion,
                )
            )
    return mismatches


def check_user_priors_for_location_overparameterization(
    params: Params,
    noncentered: NoncenteredSetting,
) -> list[PriorMismatch]:
    """Detect centered group means that collide with matching common effects.

    When a Formulae group expression also occurs as a common term and the user
    supplies a free ``mu`` for a group-specific translation-family prior under
    centering, the linear predictor sees only ``beta + mu``. The likelihood is
    invariant under shifts of mass between those parameters, so the likelihood
    has a ridge and their decomposition is identified only by the priors.

    The same location ridge occurs when two or more centered group terms share
    an exact Formulae expression but no common term owns its population effect.
    Shifting one group mean up and another down leaves the predictor unchanged.

    This check is intentionally silent for fixed ``mu`` values, a single
    unmatched free location, and effective non-centering. Non-centered problems
    are reported separately by :func:`check_user_group_prior_compatibility`.
    """
    mismatches: list[PriorMismatch] = []
    for param_name, param in params.items():
        unmatched_free_locations: dict[str, list[str]] = {}
        for term_name, expression_name, has_common, prior in _iter_user_group_priors(
            param
        ):
            if prior.dist is not None or prior.name not in _ADDITIVE_LOCATION_PRIORS:
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
            if not has_common:
                unmatched_free_locations.setdefault(expression_name, []).append(
                    term_name
                )
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

        for expression_name, term_names in sorted(unmatched_free_locations.items()):
            if len(term_names) < 2:
                continue
            sorted_term_names = sorted(term_names)
            formula_expression = (
                "1" if expression_name == "Intercept" else expression_name
            )
            mismatches.append(
                PriorMismatch(
                    parameter=param_name,
                    term=", ".join(sorted_term_names),
                    reason=(
                        f"User priors for group terms {sorted_term_names!r} on "
                        "parameter "
                        f"'{param_name}' each have a free `mu` under the effective "
                        f"centered parameterization, and their exact Formulae "
                        f"expression '{expression_name}' has no common effect. "
                        "The likelihood is invariant when one group location is "
                        "shifted up and another is shifted down; their decomposition "
                        "is identified only by the priors, and the likelihood has a "
                        "location ridge. Proper priors may still yield a proper "
                        "posterior."
                    ),
                    suggestion=(
                        f"Add the exact common formula term '{formula_expression}' "
                        "and set `mu=0` on every matching group deviation, or choose "
                        "exactly one group term to own the free population location "
                        "and fix the other group locations intentionally."
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


def raise_prior_compatibility_errors(mismatches: list[PriorMismatch]) -> None:
    """Raise one pre-build error containing all incompatible explicit priors."""
    if not mismatches:
        return
    details = "\n".join(
        f"- {m.reason} {m.suggestion}"
        for m in sorted(mismatches, key=lambda item: (item.parameter, item.term))
    )
    raise ValueError(
        "Explicit group-specific prior specification(s) cannot be represented "
        f"faithfully by bambi:\n{details}"
    )


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

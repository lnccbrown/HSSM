"""Label-switching: pick and order an anchor switching parameter.

The regime labels are arbitrary, so the posterior has ``K!`` equivalent modes.
``AutoOrdering`` (the default) breaks this by declaring **one** switching
parameter (the *anchor*) with PyMC's ``ordered`` transform, making permuted
modes unreachable.  A soft Potential barrier was tried first and fails at
``K >= 3`` (design doc §5.3); the transform is used instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .specs import AutoOrdering, NoOrdering, OrderByParam, OrderingSpec

_logger = logging.getLogger("hssm")


@dataclass
class AnchorInfo:
    """The resolved anchor: which parameter and in which direction."""

    name: str
    direction: str  # "asc" or "desc"


def resolve_anchor(
    ordering: OrderingSpec,
    switching_params: list[str],
) -> AnchorInfo | None:
    """Resolve the ordering anchor from the ordering spec.

    Returns ``None`` when no ordering should be applied (``NoOrdering`` or no
    switching parameters to anchor on).

    The ``AutoOrdering`` heuristic (in order of preference):

    1. If ``v`` is among ``switching_params``, anchor on ``v``.
    2. Else if there is exactly one (non-``p_outlier``) switching parameter,
       anchor on it.
    3. Else use the first (non-``p_outlier``) switching parameter and warn.

    ``p_outlier`` is never an auto-anchor: the ``ordered`` transform on the
    bounded ``Beta`` lapse parameter is numerically unstable (it produces a
    non-finite logp at the start), and ordering regimes by lapse rate is rarely
    the intent.
    """
    if isinstance(ordering, NoOrdering):
        _logger.warning(
            "NoOrdering is set: the regime posterior may be multi-modal due to "
            "label-switching."
        )
        return None

    if not switching_params:
        # Nothing inferred per regime -> no label-switching to break.
        return None

    if isinstance(ordering, OrderByParam):
        if ordering.name not in switching_params:
            raise ValueError(
                f"ordering anchor {ordering.name!r} is not in switching_params "
                f"{switching_params}."
            )
        if ordering.direction not in ("asc", "desc"):
            raise ValueError(
                f"ordering direction must be 'asc' or 'desc', got "
                f"{ordering.direction!r}."
            )
        if ordering.name == "p_outlier":
            raise NotImplementedError(
                "Ordering on `p_outlier` is not supported: the `ordered` "
                "transform on the bounded Beta lapse parameter is numerically "
                "unstable. Anchor on a drift/threshold parameter, or use "
                "NoOrdering."
            )
        return AnchorInfo(name=ordering.name, direction=ordering.direction)

    if isinstance(ordering, AutoOrdering):
        # p_outlier is never an auto-anchor (unstable ordered-Beta; see above).
        candidates = [p for p in switching_params if p != "p_outlier"]
        if not candidates:
            _logger.warning(
                "AutoOrdering: the only switching parameter is `p_outlier`, "
                "which cannot anchor label-switching ordering; proceeding "
                "without an ordering constraint (the regime posterior may be "
                "multi-modal). Add a drift/threshold switching parameter or use "
                "OrderByParam."
            )
            return None
        if "v" in candidates:
            name = "v"
        elif len(candidates) == 1:
            name = candidates[0]
        else:
            name = candidates[0]
            _logger.warning(
                "AutoOrdering: multiple switching parameters and no 'v'; "
                "anchoring on %r. Use OrderByParam to choose another anchor or "
                "NoOrdering to disable.",
                name,
            )
        return AnchorInfo(name=name, direction="asc")

    raise TypeError(f"Unsupported ordering spec of type {type(ordering)!r}.")

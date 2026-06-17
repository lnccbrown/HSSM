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
    2. Else if there is exactly one switching parameter, anchor on it.
    3. Else use the first switching parameter and warn.
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
        return AnchorInfo(name=ordering.name, direction=ordering.direction)

    if isinstance(ordering, AutoOrdering):
        if "v" in switching_params:
            name = "v"
        elif len(switching_params) == 1:
            name = switching_params[0]
        else:
            name = switching_params[0]
            _logger.warning(
                "AutoOrdering: multiple switching parameters and no 'v'; "
                "anchoring on %r. Use OrderByParam to choose another anchor or "
                "NoOrdering to disable.",
                name,
            )
        return AnchorInfo(name=name, direction="asc")

    raise TypeError(f"Unsupported ordering spec of type {type(ordering)!r}.")

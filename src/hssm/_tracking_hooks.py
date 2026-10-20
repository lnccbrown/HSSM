"""MLflow-free state shared by HSSM's core and :mod:`hssm.tracking`.

The fitting code and the ONNX loader run for every user, tracked or not, so
what they need from tracking lives here rather than in :mod:`hssm.tracking`,
which requires MLflow: whether a :func:`hssm.tracking.track` block is open, and
which HuggingFace networks a model was built with. Nothing here imports MLflow.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # pragma: no cover
    import os

    from hssm.tracking import Tracker

#: Role of the network that implements a model's likelihood.
LIKELIHOOD_ROLE = "likelihood"

#: Role of the network that scores missing or deadline-censored trials.
MISSING_DATA_ROLE = "missing_data"

# The networks fetched from HuggingFace while a model is built, keyed by the
# role each plays, recorded by the ONNX loader so the model can claim them. A
# model with missing data loads two networks, and one record per role keeps
# the second from overwriting the first. A ContextVar for the same reason as
# `_ACTIVE` below: two models constructed concurrently would otherwise
# interleave on one dict and could end up claiming each other's network.
# Always replaced wholesale, never mutated in place.
_NETWORKS: ContextVar[dict[str, dict[str, str]] | None] = ContextVar(
    "hssm_networks", default=None
)

# Network files to fetch at a fixed HuggingFace revision, by filename; see
# `pinned_revisions`. Empty outside `load_run`, so building a model normally
# fetches the default branch.
_PINNED_REVISIONS: ContextVar[dict[str, str] | None] = ContextVar(
    "hssm_pinned_revisions", default=None
)

# The role of whatever network is loaded next; see `network_role`.
_NETWORK_ROLE: ContextVar[str] = ContextVar(
    "hssm_network_role", default=LIKELIHOOD_ROLE
)

# Which tracker the enclosing `track()` block installed. A ContextVar rather
# than a plain global: a new thread starts with a fresh context, so an untracked
# `sample()` running alongside a tracked one cannot see — and so cannot mutate —
# the tracked run's timing and model state.
_ACTIVE: ContextVar[Tracker | None] = ContextVar("hssm_active_tracker", default=None)


# --------------------------------------------------------------------------- #
# Network provenance (called from distribution_utils.onnx_utils.model)
# --------------------------------------------------------------------------- #


def hf_revision_from_path(local_path: str | os.PathLike) -> str | None:
    """Commit sha of a file in the huggingface_hub cache, or None.

    ``hf_hub_download`` returns ``.../snapshots/<commit sha>/<file>``, with
    the file's folders in the repository below the sha; reading the sha off
    the path costs no network round-trip. The search runs from the outermost
    folder inward, so a repository folder that is itself named ``snapshots``
    is not mistaken for the cache's.
    """
    return next(
        (
            folder.name
            for folder in reversed(Path(local_path).parents)
            if folder.parent.name == "snapshots"
        ),
        None,
    )


def record_network(filename: str, local_path: str | os.PathLike) -> None:
    """Remember which network file was just fetched from HuggingFace.

    Recorded under the current :func:`network_role`.
    """
    record: dict[str, str] = {"network_file": str(filename)}
    if revision := hf_revision_from_path(local_path):
        record["hf_revision"] = revision
    _NETWORKS.set({**(_NETWORKS.get() or {}), _NETWORK_ROLE.get(): record})


@contextlib.contextmanager
def network_role(role: str) -> Iterator[None]:
    """Record networks loaded inside the block under ``role``."""
    token = _NETWORK_ROLE.set(role)
    try:
        yield
    finally:
        _NETWORK_ROLE.reset(token)


@contextlib.contextmanager
def pinned_revisions(revisions: dict[str, str]) -> Iterator[None]:
    """Fetch the named network files at these HuggingFace revisions in the block.

    ``revisions`` maps a network filename to a commit sha. :func:`load_run`
    uses it so a rebuilt model gets the network its run was fitted with, even
    after the file has been updated on the default branch.
    """
    token = _PINNED_REVISIONS.set(dict(revisions))
    try:
        yield
    finally:
        _PINNED_REVISIONS.reset(token)


def pinned_revision(filename: str) -> str | None:
    """Return the revision to fetch ``filename`` at, or None for the default."""
    return (_PINNED_REVISIONS.get() or {}).get(filename)


def recorded_networks() -> dict[str, dict[str, str]]:
    """Return the recorded networks, keyed by role.

    Scoped to the execution context, so a download in another thread is not
    mistaken for this one's.
    """
    return {role: dict(record) for role, record in (_NETWORKS.get() or {}).items()}


def last_network() -> dict[str, str]:
    """Return the recorded likelihood network (``network_file``, ``hf_revision``)."""
    return recorded_networks().get(LIKELIHOOD_ROLE, {})


def reset_network_record() -> None:
    """Forget the last recorded network.

    Called before a model builds its likelihood so that what is recorded
    afterwards belongs to *that* model. Without it a model which downloads
    nothing — an analytical likelihood, say — would inherit whichever network
    the previous model in the session happened to load.
    """
    _NETWORKS.set(None)


def active() -> Tracker | None:
    """Return the tracker of the enclosing :func:`track` block, or None.

    Scoped to the execution context: work started in another thread sees None
    unless it opened a block of its own.
    """
    return _ACTIVE.get()

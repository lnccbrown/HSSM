"""Panel-preprocessing utilities for regime-switching SSMs.

Unlike RLSSM (which requires a *balanced* panel), :class:`RSSSM` supports
unbalanced panels by end-padding every participant to ``T_max`` and carrying an
emission mask that zeroes the padded steps inside the forward recursion (design
doc §3.5).  The padded marginal is **exact**: a padded step contributes 0 to
the emission term while the transition still advances, and because each row of
``P`` sums to 1 the running marginal is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def pad_and_align_to_T_max(
    data: pd.DataFrame,
    participant_col: str,
    data_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Group rows by participant, end-pad to ``T_max``, and build the mask.

    Rows must be grouped by participant and ordered by trial within each
    participant (the standard long-format / RLSSM row-order convention).  Each
    participant ``n`` with ``T_n < T_max`` is padded to ``T_max`` by
    duplicating its last real ``data_cols`` row (kept in-support so the masked
    emission never evaluates an out-of-support placeholder).

    Parameters
    ----------
    data
        Long-format trial data.
    participant_col
        Column identifying participants.
    data_cols
        The emission columns to extract (e.g. ``["rt", "response"]``).

    Returns
    -------
    data_padded
        ``(N, T_max, len(data_cols))`` float array.
    mask
        ``(N, T_max)`` float array, 1.0 for real trials and 0.0 for padded.
    n_participants
        Number of participants ``N``.
    n_trials
        ``T_max`` (the padded trial count).

    Raises
    ------
    ValueError
        If ``participant_col`` is missing, contains NaNs, or participant rows
        are not contiguous.
    """
    if participant_col not in data.columns:
        raise ValueError(
            f"Column '{participant_col}' not found in data. Provide the correct "
            "participant column via `participant_col`."
        )
    missing_cols = [c for c in data_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Data is missing required column(s): {missing_cols}.")

    n_null = data[participant_col].isna().sum()
    if n_null > 0:
        raise ValueError(
            f"Column '{participant_col}' contains {n_null} NaN value(s). All rows "
            "must have a valid participant identifier."
        )

    # Participant rows must be contiguous; the forward recursion reshapes by
    # row position, so interleaved rows would mix subjects.
    pid = data[participant_col]
    n_runs = int((pid != pid.shift()).sum())
    n_unique = int(pid.nunique())
    if n_runs != n_unique:
        raise ValueError(
            "Data rows must be contiguous per participant. Sort by participant "
            f"before passing to RSSSM (e.g. data.sort_values('{participant_col}'))."
        )

    # Preserve first-appearance order of participants.
    order = pid.drop_duplicates().tolist()
    values = data[data_cols].to_numpy(dtype=float)

    groups = []
    start = 0
    counts = pid.groupby(pid, sort=False).size()
    for p in order:
        t_n = int(counts[p])
        groups.append(values[start : start + t_n])
        start += t_n

    n_participants = len(groups)
    t_max = max(g.shape[0] for g in groups)
    n_cols = len(data_cols)

    # A regime-switching model needs at least one transition to identify the
    # Markov structure: the forward recursion scans over trials 1..T_max-1, so a
    # panel in which *every* participant has a single trial (T_max == 1) is
    # degenerate (the scan would iterate an empty sequence).  Single-trial
    # participants are fine as long as some participant has >= 2 trials.
    if t_max < 2:
        raise ValueError(
            "RSSSM needs at least 2 trials for at least one participant "
            f"(got T_max={t_max}); a single-trial-only panel cannot identify a "
            "Markov chain."
        )

    data_padded = np.empty((n_participants, t_max, n_cols), dtype=float)
    mask = np.zeros((n_participants, t_max), dtype=float)
    for i, g in enumerate(groups):
        t_n = g.shape[0]
        data_padded[i, :t_n] = g
        mask[i, :t_n] = 1.0
        if t_n < t_max:
            # Duplicate the last real trial into the padded tail (in-support).
            data_padded[i, t_n:] = g[-1]

    return data_padded, mask, n_participants, t_max

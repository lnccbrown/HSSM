"""Utility functions for reinforcement learning + SSM models."""

import pandas as pd


def validate_balanced_panel(
    data: pd.DataFrame,
    participant_col: str = "participant_id",
) -> tuple[int, int]:
    """Validate that data forms a balanced panel and return its shape.

    A balanced panel requires every participant to have exactly the same number
    of trials (rows in *data*).

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to validate.
    participant_col : str, optional
        Name of the column that identifies participants.
        Defaults to ``"participant_id"``.

    Returns
    -------
    tuple[int, int]
        ``(n_participants, n_trials)`` where *n_trials* is the number of rows
        per participant.

    Raises
    ------
    ValueError
        If *participant_col* is not present in *data*, if the panel is
        unbalanced (participants have different trial counts), or if the rows
        of a participant are not contiguous.

    Notes
    -----
    Only the grouping of rows by participant is checked. The order of trials
    within each participant cannot be checked here (no trial column is
    inspected): the RL likelihood takes it from row order, so the rows of each
    participant must already be in trial order.
    """
    if participant_col not in data.columns:
        raise ValueError(
            f"Column '{participant_col}' not found in data. "
            "Please provide the correct participant column name via "
            "`participant_col`."
        )

    n_null = data[participant_col].isna().sum()
    if n_null > 0:
        raise ValueError(
            f"Column '{participant_col}' contains {n_null} NaN value(s). "
            "All rows must have a valid participant identifier."
        )

    counts = data.groupby(participant_col).size()
    if counts.nunique() != 1:
        raise ValueError(
            "Data must form balanced panels: all participants must have the "
            f"same number of trials. Observed trial counts: {dict(counts)}"
        )

    # Check that each participant's rows are contiguous (not interleaved).
    # make_rl_logp_op reshapes data as (n_participants, n_trials, ...) by row
    # order, so interleaved rows would silently mix subjects and corrupt the
    # RL learning dynamics.
    # The suggested fix must keep each participant's trials in order: pandas'
    # default sort (kind="quicksort") is not stable, so sorting on the
    # participant column alone scrambles the trial order within participants.
    n_runs = int((data[participant_col] != data[participant_col].shift()).sum())
    if n_runs != len(counts):
        raise ValueError(
            "Data rows must be contiguous per participant. "
            "The RL likelihood reshapes data by row position; interleaved "
            "participant rows will corrupt per-participant trial sequences. "
            "Please sort the data by participant and, within participant, by "
            "trial order before passing it to RLSSM, e.g. "
            f"data.sort_values(['{participant_col}', 'trial_id'], kind='stable') "
            "with your trial-order column in place of 'trial_id', or, if the "
            "rows are already in trial order within each participant, "
            f"data.sort_values('{participant_col}', kind='stable'). "
            "The default sort is not stable and scrambles the order of trials "
            "within participants."
        )

    return int(len(counts)), int(counts.iloc[0])
